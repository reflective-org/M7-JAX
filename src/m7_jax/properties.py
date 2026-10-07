"""Native pure-sulfate mean particle properties and both M7 water options."""
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from . import _ad
from scipy.io import netcdf_file

from .constants import (PI, AVOGADRO, SO4_MOLAR_MASS, H2SO4_MOLAR_MASS,
    SULFATE_DENSITY, SULFATE_KAPPA, CMIN_AERNL, CMIN_AERML, CMINVOL, CMINRAD,
    CRITICAL_MOLECULES, RAM2CMR, CMR2RAM, WVB, GMB,
    COMPONENT_SPECIES, COMPONENT_MODE, SPECIES_DENSITY, SPECIES_KAPPA,
    FAC_UG, FAC_MOLEC, CDCONV, CEPSSEC)


class KappaTable(NamedTuple):
    growth: object
    radius: object
    temperature: object
    humidity: object
    kappa: object


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
    sulfate, number = jnp.asarray(sulfate), jnp.asarray(number)[..., :4]
    active = (number > CMIN_AERNL) & (sulfate > CMIN_AERML)
    fallback = jnp.array((CRITICAL_MOLECULES, 0., 0., 0.))
    return jnp.maximum(jnp.where(active, sulfate/jnp.where(active, number, 1.), fallback), 0.)


def _pad(values):
    return jnp.concatenate((values, jnp.zeros(values.shape[:-1]+(3,), dtype=values.dtype)), axis=-1)


def _nearest_index(value, axis):
    """NINT-style (round-half-away-from-zero, here always nonnegative) nearest
    lookup index into a monotonically increasing table axis (0-based; the
    native `1+NINT(...)` is 1-based into a Fortran array of the same length).
    Shared by the sulfate-only and multi-species kappa lookups.
    """
    position = (len(axis)-1)*(jnp.clip(value, axis[0], axis[-1])-axis[0])/(axis[-1]-axis[0])
    return jnp.floor(position+.5).astype(jnp.int32)


def _relative_humidity_index(rh, table):
    """Version-2 kappa table RH index (mo_ham_kappa.F90 compressed-step
    lookup, redmine #260): same compressed-geometric spacing as the native
    `ix_rh = CEILING(LOG(...)/log_rh_step_compress)` clamped to [1, N_rh].
    """
    first_step = table.humidity[1]-table.humidity[0]
    compression = (table.humidity[2]-table.humidity[1])/first_step
    index = jnp.ceil(jnp.log(1-(1-compression)*(rh-table.humidity[0])/first_step)/jnp.log(compression))
    return jnp.clip(index, 1, len(table.humidity)).astype(jnp.int32)-1


def properties(sulfate, number, temperature, humidity, table=None, *, water_scheme=1, pressure=101325., previous=None):
    """Pure sulfate reduction of original averageproperties + kappa/equil chain.

    Registry SO4 molecular weight is used for kappa, native H2SO4 weight for
    Jacobson. This inherited distinction is deliberate. Empty first modes have
    a virtual critical cluster radius without adding physical sulfate.
    """
    mean = average_sulfate(sulfate, number)
    t = jnp.asarray(temperature)[..., None]
    rh = jnp.asarray(humidity)[..., None]
    ram = jnp.asarray(RAM2CMR[:4])
    if water_scheme == 1:
        if table is None:
            raise ValueError('water_scheme=1 requires the authentic kappa table')
        dry_mass = SO4_MOLAR_MASS/AVOGADRO*mean
        volume = dry_mass/SULFATE_DENSITY
        dry = ram*(volume/(4*PI/3))**(1/3)
        active = dry > CMINRAD
        safe = jnp.where(active, dry, CMINRAD)
        kap = SO4_MOLAR_MASS/AVOGADRO*SULFATE_KAPPA*(mean/SULFATE_DENSITY)/(volume+CEPSSEC)
        it = _nearest_index(t, table.temperature)
        ik = _nearest_index(kap, table.kappa)
        # Preserve the native default-real 1.E-2 promoted to dp.
        ir = _nearest_index(jnp.log(float(np.float32(1e-2))*safe), jnp.log(table.radius))
        ih = _relative_humidity_index(rh, table)
        growth = table.growth[ir, it, ih, ik]
        wet = safe*growth
        dry_volume = 4*PI/3*(jnp.asarray(CMR2RAM[:4])*safe)**3
        wet_volume = 4*PI/3*(jnp.asarray(CMR2RAM[:4])*wet)**3
        water_mass = wet_volume-dry_volume
        density = (dry_mass+water_mass)/wet_volume
        water = 1e3*jnp.asarray(number)[..., :4]*water_mass
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
        w[16]*jnp.log(rh*t/jnp.asarray(pressure)[..., None]))
    dry_mass = safe*H2SO4_MOLAR_MASS/AVOGADRO
    total_mass = dry_mass/(fraction/100.)
    g = GMB
    density = jnp.clip(g[0]+g[1]*fraction+g[2]*lm3+g[3]*rh+g[4]*t+g[5]*lm32+
        g[6]*lm3/rh+g[7]*lm3/t+g[8]*t2, 1., SULFATE_DENSITY)
    dry = (dry_mass/SULFATE_DENSITY*.75/PI)**(1/3)*ram
    wet = (total_mass/density*.75/PI)**(1/3)*ram
    water = (total_mass-dry_mass)*jnp.asarray(number)[..., :4]*1e3
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
    mass, number = jnp.asarray(mass), jnp.asarray(number)
    shape7 = number.shape[:-1]+(7,)
    if previous is None:
        prev_radius = jnp.zeros(shape7, dtype=mass.dtype)
        prev_density = jnp.zeros(shape7, dtype=mass.dtype)
    else:
        prev_radius = jnp.asarray(previous.insoluble_radius)
        prev_density = jnp.asarray(previous.insoluble_density)

    pttn_components = []
    for jn in range(18):
        mode = COMPONENT_MODE[jn]
        unitfac = FAC_MOLEC if COMPONENT_SPECIES[jn] == 0 else FAC_UG
        active = (number[..., mode] > CMIN_AERNL) & (mass[..., jn] > CMIN_AERML)
        safe_number = jnp.where(active, number[..., mode], 1.)
        value = _ad.div(unitfac*mass[..., jn], safe_number)
        fallback = CRITICAL_MOLECULES if jn == 0 else 0.
        pttn_components.append(jnp.maximum(jnp.where(active, value, fallback), 0.))
    pttn = jnp.stack(pttn_components, axis=-1)

    radius, density = prev_radius, prev_density
    for mode in range(4, 7):
        members = [jn for jn in range(18) if COMPONENT_MODE[jn] == mode]
        component_density = [CDCONV*SPECIES_DENSITY[COMPONENT_SPECIES[jn]] for jn in members]
        mass_sum = sum(pttn[..., jn] for jn in members)
        volume_sum = sum(pttn[..., jn]/den for jn, den in zip(members, component_density))
        active = (number[..., mode] > CMIN_AERNL) & (volume_sum > CMINVOL)
        safe_volume = jnp.where(active, volume_sum, 1.)
        new_density = _ad.div(mass_sum, safe_volume)
        new_radius = RAM2CMR[mode]*_ad.cbrt_volume(safe_volume/(4*PI/3))
        radius = radius.at[..., mode].set(jnp.where(active, new_radius, prev_radius[..., mode]))
        density = density.at[..., mode].set(jnp.where(active, new_density, prev_density[..., mode]))

    return AverageProperties(pttn, radius, density)


class KappaHydration(NamedTuple):
    dry_radius: object  # (...,4) cm; soluble modes NS,KS,AS,CS
    wet_radius: object  # (...,7) cm; soluble modes hydrated, insoluble = previous dry radius
    density: object      # (...,7) g/cm3; soluble modes hydrated, insoluble = previous density
    water: object         # (...,4) kg/m3 air; soluble modes only


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
    pttn, number = jnp.asarray(pttn), jnp.asarray(number)
    t = jnp.asarray(temperature)[..., None]
    rh = jnp.asarray(relative_humidity)[..., None]
    shape7 = number.shape[:-1]+(7,)
    if previous is None:
        prev_radius = jnp.zeros(shape7, dtype=pttn.dtype)
        prev_density = jnp.zeros(shape7, dtype=pttn.dtype)
    else:
        prev_radius = jnp.asarray(previous.insoluble_radius)
        prev_density = jnp.asarray(previous.insoluble_density)

    dry_mass = [0.]*4
    dry_volume = [0.]*4
    kappa_volume = [0.]*4
    for jn in range(18):
        mode = COMPONENT_MODE[jn]
        if mode >= 4:
            continue  # insoluble component: never contributes to a soluble mode's hydration
        species = COMPONENT_SPECIES[jn]
        unitfac = SO4_MOLAR_MASS/AVOGADRO if species == 0 else 1.
        component_density = CDCONV*SPECIES_DENSITY[species]
        mass_j = unitfac*pttn[..., jn]
        volume_j = mass_j/component_density
        dry_mass[mode] = dry_mass[mode]+mass_j
        dry_volume[mode] = dry_volume[mode]+volume_j
        kappa_volume[mode] = kappa_volume[mode]+SPECIES_KAPPA[species]*volume_j
    dry_mass = jnp.stack(dry_mass, axis=-1)
    dry_volume = jnp.stack(dry_volume, axis=-1)
    kappa_volume = jnp.stack(kappa_volume, axis=-1)

    ram, cmr = jnp.asarray(RAM2CMR[:4]), jnp.asarray(CMR2RAM[:4])
    dry = ram*_ad.cbrt_volume(dry_volume/(4*PI/3))
    kappa = _ad.div(kappa_volume, dry_volume+CEPSSEC)

    active = dry > CMINRAD
    safe = jnp.where(active, dry, CMINRAD)
    it = _nearest_index(t, table.temperature)
    ik = _nearest_index(kappa, table.kappa)
    # Preserve the native default-real 1.E-2 promoted to dp.
    ir = _nearest_index(jnp.log(float(np.float32(1e-2))*safe), jnp.log(table.radius))
    ih = _relative_humidity_index(rh, table)
    growth = table.growth[ir, it, ih, ik]
    wet = safe*growth

    dry_particle_volume = 4*PI/3*(cmr*safe)**3
    wet_particle_volume = 4*PI/3*(cmr*wet)**3
    water_mass = wet_particle_volume-dry_particle_volume
    density4 = _ad.div(dry_mass+water_mass, wet_particle_volume)
    water4 = 1e3*number[..., :4]*water_mass

    wet4 = jnp.where(active, wet, dry)
    density4 = jnp.where(active, density4, 1.)
    water4 = jnp.where(active, water4, 0.)

    wet_radius = jnp.concatenate((wet4, prev_radius[..., 4:]), axis=-1)
    density = jnp.concatenate((density4, prev_density[..., 4:]), axis=-1)
    return KappaHydration(dry, wet_radius, density, water4)
