"""Native pure-sulfate mean particle properties and both M7 water options."""
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import _ad
from scipy.io import netcdf_file

from .constants import (PI, AVOGADRO, SO4_MOLAR_MASS, H2SO4_MOLAR_MASS,
    SULFATE_DENSITY, SULFATE_KAPPA, CMIN_AERNL, CMIN_AERML, CMINVOL, CMINRAD,
    CRITICAL_MOLECULES, RAM2CMR, CMR2RAM, WVB, GMB,
    COMPONENT_SPECIES, COMPONENT_MODE, SPECIES_DENSITY, SPECIES_KAPPA,
    FAC_UG, FAC_MOLEC, CDCONV, CEPSSEC)

_METER_PER_CM = float(np.float32(1e-2))  # Native default-real 1.E-2 promoted to dp.
_COMPONENTS_BY_MODE = tuple(tuple(jn for jn in range(18) if COMPONENT_MODE[jn] == mode) for mode in range(7))


class KappaTable(NamedTuple):
    growth: object
    radius: object
    temperature: object
    humidity: object
    kappa: object


class PreparedKappaTable(NamedTuple):
    """Pre-transposed, stop-gradient kappa lookup table for 1-D cache-coherent gathers.

    ``growth_thrk_flat`` flattens ``growth`` after transposing ``(ir, it, ih, ik)``
    to ``(it, ih, ir, ik)`` so all four modes of a cell share one ``(it, ih)``
    21.5 KB block in cache and avoid 4-D gather/scatter overhead under AD.
    """
    growth_thrk_flat: object
    n_r: int
    n_t: int
    n_rh: int
    n_k: int
    r_log_lo: object
    r_log_hi: object
    t_lo: object
    t_hi: object
    k_lo: object
    k_hi: object
    rh_0: object
    rh_1: object
    rh_2: object


def prepare_kappa_table(table, *, dtype=None):
    """Prepare a :class:`KappaTable` for fast 1-D gather without AD scatter bloat.

    Idempotent: a :class:`PreparedKappaTable` is returned as is, or re-cast
    when ``dtype`` differs. Prepare once outside ``jit`` when the table would
    otherwise be a traced argument, so the transpose is not repeated per call.
    """
    if isinstance(table, PreparedKappaTable):
        if dtype is None or table.growth_thrk_flat.dtype == jnp.dtype(dtype):
            return table
        return table._replace(**{name: _ad.frozen(getattr(table, name), dtype)
                                 for name in _PREPARED_KAPPA_ARRAYS})
    growth, radius, temperature, humidity, kappa = (_ad.frozen(x, dtype) for x in table)
    n_r, n_t, n_rh, n_k = growth.shape
    log_r = jnp.log(radius)
    return PreparedKappaTable(
        growth_thrk_flat=jnp.transpose(growth, (1, 2, 0, 3)).reshape(-1),
        n_r=n_r, n_t=n_t, n_rh=n_rh, n_k=n_k,
        r_log_lo=log_r[0], r_log_hi=log_r[-1],
        t_lo=temperature[0], t_hi=temperature[-1],
        k_lo=kappa[0], k_hi=kappa[-1],
        rh_0=humidity[0], rh_1=humidity[1], rh_2=humidity[2],
    )


_PREPARED_KAPPA_ARRAYS = tuple(f for f in PreparedKappaTable._fields if not f.startswith('n_'))


def default_kappa_table_path():
    """Path of the authentic kappa table shipped as package data."""
    return Path(__file__).resolve().parent / 'data' / 'lut_kappa.nc'


def load_kappa_table(path):
    """Read the authentic version-2 NetCDF table outside JIT; retain native axes."""
    with netcdf_file(Path(path), mmap=False) as data:
        if data._attributes.get('Version') != b'2.0':
            raise ValueError('This port requires the native version-2 kappa lookup table')
        names = ('GF', 'DryRadius', 'Temperature', 'RelativeHumidity', 'kappa')
        values = [np.array(data.variables[name].data, dtype=np.float64, copy=True) for name in names]
    if values[0].shape != tuple(len(x) for x in values[1:]):
        raise ValueError('Unexpected kappa table axis ordering')
    if not all(np.isfinite(x).all() for x in values) or any(np.any(np.diff(x) <= 0) for x in values[1:]):
        raise ValueError('Nonfinite lookup or unordered coordinate axis')
    return KappaTable(*(jnp.asarray(x) for x in values))


class Properties(NamedTuple):
    dry_radius: object  # cm; four soluble count-median radii
    wet_radius: object  # cm; seven modes, insoluble entries zero
    density: object     # g/cm3
    water: object       # kg/m3 air
    mean_sulfate: object  # molecules/particle; includes virtual first-mode cluster


def average_sulfate(sulfate, number):
    sulfate = jnp.asarray(sulfate)
    number = jnp.asarray(number, dtype=sulfate.dtype)[..., :4]
    active = (number > CMIN_AERNL) & (sulfate > CMIN_AERML)
    fallback = jnp.array((CRITICAL_MOLECULES, 0., 0., 0.), dtype=sulfate.dtype)
    return jnp.maximum(jnp.where(active, sulfate/jnp.where(active, number, 1.), fallback), 0.)


def _pad(values):
    return jnp.concatenate((values, jnp.zeros(values.shape[:-1]+(3,), dtype=values.dtype)), axis=-1)


def _nearest_index_bounds(value, n, lo, hi):
    """NINT-style (round-half-away-from-zero, here always nonnegative) nearest
    lookup index into an ``n``-point evenly spaced axis from ``lo`` to ``hi``
    (0-based; the native ``1+NINT(...)`` is 1-based into a Fortran array of the
    same length). Shared by the sulfate-only and multi-species kappa lookups.
    """
    position = (n - 1) * (jnp.clip(value, lo, hi) - lo) / (hi - lo)
    return jnp.floor(position + .5).astype(jnp.int32)


def _relative_humidity_index_bounds(rh, n_rh, rh_0, rh_1, rh_2):
    """Kappa-table RH index (redmine #260): the native compressed-geometric
    spacing ``ix_rh = CEILING(LOG(...)/log_rh_step_compress)`` clamped to
    [1, N_rh], from the axis' first three points.
    """
    first_step = rh_1 - rh_0
    compression = (rh_2 - rh_1) / first_step
    index = jnp.ceil(jnp.log(1 - (1 - compression) * (rh - rh_0) / first_step) / jnp.log(compression))
    return jnp.clip(index, 1, n_rh).astype(jnp.int32) - 1


def _kappa_th_offset(temperature, relative_humidity, prepared):
    """Cell-invariant ``(it * n_rh + ih) * (n_r * n_k)`` offset into ``growth_thrk_flat``."""
    it = _nearest_index_bounds(temperature, prepared.n_t, prepared.t_lo, prepared.t_hi)
    ih = _relative_humidity_index_bounds(
        relative_humidity, prepared.n_rh, prepared.rh_0, prepared.rh_1, prepared.rh_2
    )
    return (it * prepared.n_rh + ih) * (prepared.n_r * prepared.n_k)


def _pure_sulfate_dry_radius(sulfate, number):
    """Four soluble dry count-median radii [cm] for the pure-sulfate ``water_scheme=1`` path."""
    mean = average_sulfate(sulfate, number)
    ram = jnp.asarray(RAM2CMR[:4], dtype=mean.dtype)
    dry_mass = SO4_MOLAR_MASS/AVOGADRO*mean
    volume = dry_mass/SULFATE_DENSITY
    return ram*(volume/(4*PI/3))**(1/3)


def properties(sulfate, number, temperature, humidity, table=None, *, water_scheme=1, pressure=101325., previous=None):
    """Pure sulfate reduction of original averageproperties + kappa/equil chain.

    Registry SO4 molecular weight is used for kappa, native H2SO4 weight for
    Jacobson. This inherited distinction is deliberate. Empty first modes have
    a virtual critical cluster radius without adding physical sulfate.
    """
    mean = average_sulfate(sulfate, number)
    dtype = mean.dtype
    t = jnp.asarray(temperature, dtype=dtype)[..., None]
    rh = jnp.asarray(humidity, dtype=dtype)[..., None]
    ram = jnp.asarray(RAM2CMR[:4], dtype=dtype)
    if water_scheme == 1:
        if table is None:
            raise ValueError('water_scheme=1 requires the authentic kappa table')
        prepared = prepare_kappa_table(table, dtype=dtype)
        dry_mass = SO4_MOLAR_MASS/AVOGADRO*mean
        volume = dry_mass/SULFATE_DENSITY
        dry = ram*(volume/(4*PI/3))**(1/3)
        active = dry > CMINRAD
        safe = jnp.where(active, dry, CMINRAD)
        kap = SO4_MOLAR_MASS/AVOGADRO*SULFATE_KAPPA*(mean/SULFATE_DENSITY)/(volume+CEPSSEC)
        th_offset = _kappa_th_offset(t, rh, prepared)
        ik = _nearest_index_bounds(kap, prepared.n_k, prepared.k_lo, prepared.k_hi)
        ir = _nearest_index_bounds(jnp.log(_METER_PER_CM*safe), prepared.n_r, prepared.r_log_lo, prepared.r_log_hi)
        growth = prepared.growth_thrk_flat[th_offset + ir * prepared.n_k + ik]
        wet = safe*growth
        cmr = jnp.asarray(CMR2RAM[:4], dtype=dtype)
        dry_volume = 4*PI/3*(cmr*safe)**3
        wet_volume = 4*PI/3*(cmr*wet)**3
        water_mass = wet_volume-dry_volume
        density = (dry_mass+water_mass)/wet_volume
        water = 1e3*jnp.asarray(number, dtype=dtype)[..., :4]*water_mass
        return Properties(dry, _pad(jnp.where(active, wet, dry)),
                          _pad(jnp.where(active, density, 1.)),
                          _pad(jnp.where(active, water, 0.)), mean)
    if water_scheme != 0:
        raise ValueError('water_scheme must be 0 (Jacobson) or 1 (native kappa lookup)')
    active = mean > 0
    safe = jnp.where(active, mean, 1.)
    t = jnp.maximum(t, 240.)
    rh = jnp.clip(rh, .05, .90)
    lm = jnp.log(safe); lm2 = lm*lm; rh2 = rh*rh; t2 = t*t; lm3 = lm/3; lm32 = lm3*lm3
    w = WVB
    fraction = (w[0]+w[1]*lm+w[2]*rh*lm+w[3]*t*lm+w[4]*rh/t+w[5]*lm2*rh+
        w[6]*lm2*t+w[7]*lm*rh2+w[8]*lm*t2+w[9]*lm2*rh2+w[10]*lm2*t2+
        w[11]*rh2/t2+w[12]*lm2+w[13]*lm2*lm+w[14]*lm2*lm2+w[15]*rh2*rh/(t2*t)+
        w[16]*jnp.log(rh*t/jnp.asarray(pressure, dtype=dtype)[..., None]))
    dry_mass = safe*H2SO4_MOLAR_MASS/AVOGADRO
    total_mass = dry_mass/(fraction/100.)
    g = GMB
    density = jnp.clip(g[0]+g[1]*fraction+g[2]*lm3+g[3]*rh+g[4]*t+g[5]*lm32+
        g[6]*lm3/rh+g[7]*lm3/t+g[8]*t2, 1., SULFATE_DENSITY)
    dry = (dry_mass/SULFATE_DENSITY*.75/PI)**(1/3)*ram
    wet = (total_mass/density*.75/PI)**(1/3)*ram
    water = (total_mass-dry_mass)*jnp.asarray(number, dtype=dtype)[..., :4]*1e3
    old = previous if previous is not None else Properties(jnp.zeros_like(mean), _pad(jnp.zeros_like(mean)),
        _pad(jnp.zeros_like(mean)), _pad(jnp.zeros_like(mean)), mean)
    return Properties(jnp.where(active, dry, old.dry_radius),
        _pad(jnp.where(active, wet, old.wet_radius[..., :4])),
        _pad(jnp.where(active, density, old.density[..., :4])),
        _pad(jnp.where(active, water, old.water[..., :4])), mean)


def require_sulfate_only_for_jacobson(mass, water_scheme):
    """Guard against the unported Jacobson multi-species path.

    `water_scheme=0` native hydration (m7_equiz/m7_equimix/m7_equil,
    mo_ham_m7.F90) is out of scope for this port: it is only ever exercised
    through the pure-sulfate `properties()` above, which keeps its own
    sulfate-only Jacobson branch. If a caller is about to combine
    `water_scheme=0` with the full 18-component state, that would silently
    ignore every non-sulfate component rather than raising, so call this
    first (host-side, like `model.validate_state`; not meant to run under
    `jit`) and it raises loudly instead.
    """
    if water_scheme == 0 and bool(np.any(np.asarray(mass)[..., 4:] != 0)):
        raise NotImplementedError(
            'water_scheme=0 (m7_equiz/m7_equimix/m7_equil, mo_ham_m7.F90) is not '
            'ported for non-sulfate species; use water_scheme=1 (kappa_hydration) '
            'or restrict to the pure-sulfate properties() path')


class AverageProperties(NamedTuple):
    pttn: object               # (...,18); molecules/particle (SO4, idx 0-3), g/particle (others)
    insoluble_radius: object   # (...,7) cm; meaningful only for insoluble modes (KI,AI,CI = idx 4,5,6)
    insoluble_density: object  # (...,7) g/cm3; meaningful only for insoluble modes


def _average_pttn_soa(mass18, number7, *, soluble_only=False):
    """Compute per-component mean particle mass ``pttn`` as an 18-tuple of cell arrays."""
    pttn = []
    for jn in range(18):
        mode = COMPONENT_MODE[jn]
        if soluble_only and mode >= 4:
            pttn.append(None)
            continue
        unitfac = FAC_MOLEC if COMPONENT_SPECIES[jn] == 0 else FAC_UG
        n_m = number7[mode]
        m_j = mass18[jn]
        active = (n_m > CMIN_AERNL) & (m_j > CMIN_AERML)
        safe_number = jnp.where(active, n_m, 1.)
        value = _ad.div(unitfac * m_j, safe_number)
        fallback = CRITICAL_MOLECULES if jn == 0 else 0.
        pttn.append(jnp.maximum(jnp.where(active, value, fallback), 0.))
    return tuple(pttn)


def _insoluble_geometry_soa(pttn18, number7, prev_r_insol3, prev_d_insol3):
    """Compute insoluble-mode ``(KI, AI, CI)`` count-median radii [cm] and densities [g/cm3]."""
    radii = []
    densities = []
    for idx, mode in enumerate(range(4, 7)):
        members = _COMPONENTS_BY_MODE[mode]
        component_density = [CDCONV * SPECIES_DENSITY[COMPONENT_SPECIES[jn]] for jn in members]
        mass_sum = pttn18[members[0]]
        volume_sum = pttn18[members[0]] / component_density[0]
        for jn, den in zip(members[1:], component_density[1:]):
            mass_sum = mass_sum + pttn18[jn]
            volume_sum = volume_sum + pttn18[jn] / den
        active = (number7[mode] > CMIN_AERNL) & (volume_sum > CMINVOL)
        safe_volume = jnp.where(active, volume_sum, 1.)
        new_density = _ad.div(mass_sum, safe_volume)
        new_radius = RAM2CMR[mode] * _ad.cbrt_volume(safe_volume / (4 * PI / 3))
        radii.append(jnp.where(active, new_radius, prev_r_insol3[idx]))
        densities.append(jnp.where(active, new_density, prev_d_insol3[idx]))
    return tuple(radii), tuple(densities)


def _soluble_mode_sums(pttn18, mode):
    """``m7_kappa``'s per-mode ``(dry mass, dry volume, kappa-weighted volume)``
    of one soluble mode's mean particle, summed in component order.

    The optimization barrier on each component's ``unitfac*pttn`` is
    load-bearing. Without it XLA rearranges and fuses this chain differently:
    outputs change at the rounding level (amplified to ~1e-8 relative where
    the float64 native forms cancel) and the float32 step ran ~30% slower on
    an A100. Re-benchmark before removing it.
    """
    dry_mass = dry_vol = kappa_vol = None
    for jn in _COMPONENTS_BY_MODE[mode]:
        species = COMPONENT_SPECIES[jn]
        unitfac = SO4_MOLAR_MASS / AVOGADRO if species == 0 else 1.
        mass_j = jax.lax.optimization_barrier(unitfac * pttn18[jn])
        volume_j = mass_j / (CDCONV * SPECIES_DENSITY[species])
        kvol_j = SPECIES_KAPPA[species] * volume_j
        if dry_mass is None:
            dry_mass, dry_vol, kappa_vol = mass_j, volume_j, kvol_j
        else:
            dry_mass, dry_vol, kappa_vol = dry_mass + mass_j, dry_vol + volume_j, kappa_vol + kvol_j
    return dry_mass, dry_vol, kappa_vol


def _dry_radius(mode, dry_vol):
    return RAM2CMR[mode] * _ad.cbrt_volume(dry_vol / (4 * PI / 3))


def _soluble_dry_radius_soa(mass18, number7):
    """The four soluble dry count-median radii [cm] that ``m7_dconc`` reads.

    The same ``average_properties`` -> ``kappa_hydration`` dry-radius chain,
    without the growth-factor lookup and wet/insoluble geometry it does not need.
    """
    pttn = _average_pttn_soa(mass18, number7, soluble_only=True)
    return tuple(_dry_radius(mode, _soluble_mode_sums(pttn, mode)[1]) for mode in range(4))


def average_properties(mass, number, *, previous=None):
    """Port of m7_averageproperties (mo_ham_m7.F90:782-917), r7492 semantics.

    `mass` is the full 18-component state in HAM aerocomp order (native
    units: molecules cm-3 for the four sulfate components, ug m-3 for
    everything else; see constants.COMPONENT_SPECIES/COMPONENT_MODE for the
    component -> species/mode table). `number` is the 7-mode number
    concentration (cm-3).

    Returns the mean per-particle component mass `pttn` for every component
    (the empty SO4-NS mode gets the virtual critical-cluster mass `critn`,
    every other empty/negligible component gets 0 - mo_ham_m7.F90:869-874),
    and the insoluble modes' count-median radius [cm] / density [g cm-3],
    computed from the mean particle VOLUME. That volume-based radius is the
    r7492 correction the oracle applies at build time (see
    tools/m7_oracle.py's SOURCE_CORRECTIONS and
    test_insoluble_radius_follows_mean_particle_volume): this checkout's
    reference/fortran/mo_ham_m7.F90 (an uncorrected SALSA-box copy) instead
    takes the insoluble radius from the cube root of the particle DENSITY,
    which is dimensionally wrong; this port follows the corrected ECHAM6.3-
    HAM2.3 r7492 formula, not the file as it reads in this checkout.

    `previous` (an `AverageProperties`, or `None` for an all-zero start) is
    the geometry to keep wherever the native INOUT arguments are left
    untouched: soluble-mode entries (0-3) always, and insoluble entries when
    that mode's number or volume is negligible.
    """
    mass = jnp.asarray(mass)
    dtype = mass.dtype
    number = jnp.asarray(number, dtype=dtype)
    cell_shape = number.shape[:-1]
    zero_cell = jnp.zeros(cell_shape, dtype=dtype)
    if previous is None:
        prev_r_sol4 = (zero_cell, zero_cell, zero_cell, zero_cell)
        prev_d_sol4 = (zero_cell, zero_cell, zero_cell, zero_cell)
        prev_r_insol3 = (zero_cell, zero_cell, zero_cell)
        prev_d_insol3 = (zero_cell, zero_cell, zero_cell)
    else:
        prev_r = jnp.asarray(previous.insoluble_radius, dtype=dtype)
        prev_d = jnp.asarray(previous.insoluble_density, dtype=dtype)
        prev_r_sol4 = tuple(prev_r[..., m] for m in range(4))
        prev_d_sol4 = tuple(prev_d[..., m] for m in range(4))
        prev_r_insol3 = tuple(prev_r[..., m] for m in range(4, 7))
        prev_d_insol3 = tuple(prev_d[..., m] for m in range(4, 7))

    mass18 = tuple(mass[..., jn] for jn in range(18))
    number7 = tuple(number[..., m] for m in range(7))
    pttn18 = _average_pttn_soa(mass18, number7)
    r_insol3, d_insol3 = _insoluble_geometry_soa(pttn18, number7, prev_r_insol3, prev_d_insol3)
    pttn = jnp.stack(pttn18, axis=-1)
    radius = jnp.stack((*prev_r_sol4, *r_insol3), axis=-1)
    density = jnp.stack((*prev_d_sol4, *d_insol3), axis=-1)
    return AverageProperties(pttn, radius, density)


class KappaHydration(NamedTuple):
    dry_radius: object  # (...,4) cm; soluble modes NS,KS,AS,CS
    wet_radius: object  # (...,7) cm; soluble modes hydrated, insoluble = previous dry radius
    density: object      # (...,7) g/cm3; soluble modes hydrated, insoluble = previous density
    water: object         # (...,4) kg/m3 air; soluble modes only


def _kappa_hydration_soa(pttn18, number7, r_insol3, d_insol3, prepared, th_offset, *, compute_water=True):
    """Tuple-of-arrays implementation of ``m7_kappa`` for the four soluble modes.

    Returns ``(dry4, wet7, density7, water4)``; the insoluble entries of
    ``wet7``/``density7`` pass ``r_insol3``/``d_insol3`` through, and ``water4``
    is empty when ``compute_water`` is False.
    """
    dry_list, wet_list, density_list, water_list = [], [], [], []
    four_pi_over_3 = 4 * PI / 3
    for mode in range(4):
        dry_mass, dry_vol, kappa_vol = _soluble_mode_sums(pttn18, mode)
        dry_m = _dry_radius(mode, dry_vol)
        kappa_m = _ad.div(kappa_vol, dry_vol + CEPSSEC)
        active_m = dry_m > CMINRAD
        safe_m = jnp.where(active_m, dry_m, CMINRAD)
        ik_m = _nearest_index_bounds(kappa_m, prepared.n_k, prepared.k_lo, prepared.k_hi)
        ir_m = _nearest_index_bounds(
            jnp.log(_METER_PER_CM * safe_m), prepared.n_r, prepared.r_log_lo, prepared.r_log_hi
        )
        growth_m = prepared.growth_thrk_flat[th_offset + ir_m * prepared.n_k + ik_m]
        wet_m = safe_m * growth_m

        cmr_m = CMR2RAM[mode]
        dry_pvol = four_pi_over_3 * (cmr_m * safe_m) ** 3
        wet_pvol = four_pi_over_3 * (cmr_m * wet_m) ** 3
        water_mass = wet_pvol - dry_pvol
        dens_m = _ad.div(dry_mass + water_mass, wet_pvol)

        dry_list.append(dry_m)
        wet_list.append(jnp.where(active_m, wet_m, dry_m))
        density_list.append(jnp.where(active_m, dens_m, 1.))
        if compute_water:
            water_list.append(jnp.where(active_m, 1e3 * number7[mode] * water_mass, 0.))

    return tuple(dry_list), (*wet_list, *r_insol3), (*density_list, *d_insol3), tuple(water_list)


def kappa_hydration(pttn, number, temperature, relative_humidity, table, *, previous=None):
    """Port of m7_kappa (mo_ham_m7.F90:920-1185): hygroscopic growth for ALL
    species of the four soluble modes (water_scheme=1 / native kappa lookup).

    The dry volume and the volume-weighted kappa of a soluble mode sum over
    every component whose mode is soluble, regardless of species (mo_ham_m7.
    F90:1065-1091: the loop keys on `aerocomp(jn)%iclass`, not species) - so
    e.g. a nucleation/Aitken mode is pure SO4, but the accumulation/coarse
    modes' dry volume and kappa also include whatever BC/OC/SS/DU mass is
    internally mixed into that same soluble mode. `pttn` is
    `average_properties`'s 18-component mean particle mass.

    `previous` (an `AverageProperties`, matching `average_properties`'s
    return value and typically the one that produced `pttn`) supplies the
    insoluble-mode radius/density to pass straight through: m7_kappa only
    ever updates the soluble classes (mo_ham_m7.F90:1172-1174 MERGE on
    `lsoluble`), leaving the three insoluble entries exactly as given.
    `previous=None` passes through zeros, matching average_properties.

    Water scheme 0 (Jacobson) is a different, unported routine - see
    `require_sulfate_only_for_jacobson`; this function always implements the
    native kappa lookup (water_scheme=1).
    """
    pttn = jnp.asarray(pttn)
    dtype = pttn.dtype
    number = jnp.asarray(number, dtype=dtype)
    t = jnp.asarray(temperature, dtype=dtype)
    rh = jnp.asarray(relative_humidity, dtype=dtype)
    cell_shape = number.shape[:-1]
    if previous is None:
        zero_cell = jnp.zeros(cell_shape, dtype=dtype)
        r_insol3 = (zero_cell, zero_cell, zero_cell)
        d_insol3 = (zero_cell, zero_cell, zero_cell)
    else:
        prev_r = jnp.asarray(previous.insoluble_radius, dtype=dtype)
        prev_d = jnp.asarray(previous.insoluble_density, dtype=dtype)
        r_insol3 = tuple(prev_r[..., m] for m in range(4, 7))
        d_insol3 = tuple(prev_d[..., m] for m in range(4, 7))

    prepared = prepare_kappa_table(table, dtype=dtype)
    th_offset = _kappa_th_offset(t, rh, prepared)
    pttn18 = tuple(pttn[..., jn] for jn in range(18))
    number7 = tuple(number[..., m] for m in range(7))
    dry4, wet7, density7, water4 = _kappa_hydration_soa(
        pttn18, number7, r_insol3, d_insol3, prepared, th_offset, compute_water=True
    )
    return KappaHydration(
        jnp.stack(dry4, axis=-1),
        jnp.stack(wet7, axis=-1),
        jnp.stack(density7, axis=-1),
        jnp.stack(water4, axis=-1),
    )
