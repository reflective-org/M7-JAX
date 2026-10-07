"""M7 binary sulfuric-acid/water nucleation in native cm-based units.

Vehkamaeki polynomial terms, input clamps and rate/cluster limiters are copied
from mo_ham_m7_nucl.F90. Scheme 2 (:func:`kazil_lovejoy`, Kazil and Lovejoy
2007) needs an external PARNUC lookup table (:func:`load_kazil_lovejoy_table`),
not shipped with this package (MPI-M/HAMMOZ data, not source: see
``docs/validation.md``) -- callers without it stay on schemes 0/1. Standalone
M7 compiles organic activation/kinetic calls out; no SALSA growth/survival
formula is substituted.
"""
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.io import netcdf_file

from . import precision
from .constants import EPS_DP


class NucleationRate(NamedTuple):
    rate: object
    cluster_sulfate: object


class NucleationResult(NamedTuple):
    gas: object
    number: object
    sulfate: object


def vehkamaeki(temperature_k, relative_humidity, gas_cm3):
    """Return rate (particles/cm3/s) and H2SO4 molecules per critical cluster.

    Temperature clamps to 190.15..300.15 K, RH to 1e-4..1, and gas to 1e4..1e11
    molecules/cm3. These are native parameterization clamps, including its
    evaluation at 1e4 when supplied vapor is zero. Rates below 1e-7 or total
    clusters below four molecules become zero; rates are capped at 1e10.
    """
    zt = jnp.clip(jnp.asarray(temperature_k), 190.15, 300.15)
    zrh = jnp.clip(jnp.asarray(relative_humidity), 1e-4, 1.)
    zrhoa = jnp.clip(jnp.asarray(gas_cm3), 1e4, 1e11)
    zt2 = zt * zt
    zt3 = zt2 * zt
    zlogrh = jnp.log(zrh)
    zlogrh2 = zlogrh * zlogrh
    zlogrh3 = zlogrh2 * zlogrh
    zlogrhoa = jnp.log(zrhoa)
    zlogrhoa2 = zlogrhoa * zlogrhoa
    zlogrhoa3 = zlogrhoa2 * zlogrhoa
    x = (0.7409967177282139 - 0.002663785665140117*zt
        + 0.002010478847383187*zlogrh
        - 0.0001832894131464668*zt*zlogrh
        + 0.001574072538464286*zlogrh2
        - 0.00001790589121766952*zt*zlogrh2
        + 0.0001844027436573778*zlogrh3
        -  1.503452308794887e-6*zt*zlogrh3
        - 0.003499978417957668*zlogrhoa
        + 0.0000504021689382576*zt*zlogrhoa)
    zix = 1. / x
    zjnuc = (0.1430901615568665 + 2.219563673425199*zt -
        0.02739106114964264*zt2 +
        0.00007228107239317088*zt3 + 5.91822263375044*zix +
        0.1174886643003278*zlogrh + 0.4625315047693772*zt*zlogrh -
        0.01180591129059253*zt2*zlogrh +
        0.0000404196487152575*zt3*zlogrh +
        (15.79628615047088*zlogrh)*zix -
        0.215553951893509*zlogrh2 -
        0.0810269192332194*zt*zlogrh2 +
        0.001435808434184642*zt2*zlogrh2 -
        4.775796947178588e-6*zt3*zlogrh2 -
        (2.912974063702185*zlogrh2)*zix -
        3.588557942822751*zlogrh3 +
        0.04950795302831703*zt*zlogrh3 -
        0.0002138195118737068*zt2*zlogrh3 +
        3.108005107949533e-7*zt3*zlogrh3 -
        (0.02933332747098296*zlogrh3)*zix +
        1.145983818561277*zlogrhoa -
        0.6007956227856778*zt*zlogrhoa +
        0.00864244733283759*zt2*zlogrhoa -
        0.00002289467254710888*zt3*zlogrhoa -
        (8.44984513869014*zlogrhoa)*zix +
        2.158548369286559*zlogrh*zlogrhoa +
        0.0808121412840917*zt*zlogrh*zlogrhoa -
        0.0004073815255395214*zt2*zlogrh*zlogrhoa -
        4.019572560156515e-7*zt3*zlogrh*zlogrhoa +
        (0.7213255852557236*zlogrh*zlogrhoa)*zix +
        1.62409850488771*zlogrh2*zlogrhoa -
        0.01601062035325362*zt*zlogrh2*zlogrhoa +
        0.00003771238979714162*zt2*zlogrh2*zlogrhoa +
        3.217942606371182e-8*zt3*zlogrh2*zlogrhoa -
        (0.01132550810022116*zlogrh2*zlogrhoa)*zix +
        9.71681713056504*zlogrhoa2 -
        0.1150478558347306*zt*zlogrhoa2 +
        0.0001570982486038294*zt2*zlogrhoa2 +
        4.009144680125015e-7*zt3*zlogrhoa2 +
        (0.7118597859976135*zlogrhoa2)*zix -
        1.056105824379897*zlogrh*zlogrhoa2 +
        0.00903377584628419*zt*zlogrh*zlogrhoa2 -
        0.00001984167387090606*zt2*zlogrh*zlogrhoa2 +
        2.460478196482179e-8*zt3*zlogrh*zlogrhoa2 -
        (0.05790872906645181*zlogrh*zlogrhoa2)*zix -
        0.1487119673397459*zlogrhoa3 +
        0.002835082097822667*zt*zlogrhoa3 -
        9.24618825471694e-6*zt2*zlogrhoa3 +
        5.004267665960894e-9*zt3*zlogrhoa3 -
        (0.01270805101481648*zlogrhoa3)*zix)
    zjnuc = jnp.exp(zjnuc)
    zntot = (-0.002954125078716302 - 0.0976834264241286*zt +
        0.001024847927067835*zt2 - 2.186459697726116e-6*zt3 -
        0.1017165718716887*zix - 0.002050640345231486*zlogrh -
        0.007585041382707174*zt*zlogrh +
        0.0001926539658089536*zt2*zlogrh -
        6.70429719683894e-7*zt3*zlogrh -
        (0.2557744774673163*zlogrh)*zix +
        0.003223076552477191*zlogrh2 +
        0.000852636632240633*zt*zlogrh2 -
        0.00001547571354871789*zt2*zlogrh2 +
        5.666608424980593e-8*zt3*zlogrh2 +
        (0.03384437400744206*zlogrh2)*zix +
        0.04743226764572505*zlogrh3 -
        0.0006251042204583412*zt*zlogrh3 +
        2.650663328519478e-6*zt2*zlogrh3 -
        3.674710848763778e-9*zt3*zlogrh3 -
        (0.0002672510825259393*zlogrh3)*zix -
        0.01252108546759328*zlogrhoa +
        0.005806550506277202*zt*zlogrhoa -
        0.0001016735312443444*zt2*zlogrhoa +
        2.881946187214505e-7*zt3*zlogrhoa +
        (0.0942243379396279*zlogrhoa)*zix -
        0.0385459592773097*zlogrh*zlogrhoa -
        0.0006723156277391984*zt*zlogrh*zlogrhoa +
        2.602884877659698e-6*zt2*zlogrh*zlogrhoa +
        1.194163699688297e-8*zt3*zlogrh*zlogrhoa -
        (0.00851515345806281*zlogrh*zlogrhoa)*zix -
        0.01837488495738111*zlogrh2*zlogrhoa +
        0.0001720723574407498*zt*zlogrh2*zlogrhoa -
        3.717657974086814e-7*zt2*zlogrh2*zlogrhoa -
        5.148746022615196e-10*zt3*zlogrh2*zlogrhoa +
        (0.0002686602132926594*zlogrh2*zlogrhoa)*zix -
        0.06199739728812199*zlogrhoa2 +
        0.000906958053583576*zt*zlogrhoa2 -
        9.11727926129757e-7*zt2*zlogrhoa2 -
        5.367963396508457e-9*zt3*zlogrhoa2 -
        (0.007742343393937707*zlogrhoa2)*zix +
        0.0121827103101659*zlogrh*zlogrhoa2 -
        0.0001066499571188091*zt*zlogrh*zlogrhoa2 +
        2.534598655067518e-7*zt2*zlogrh*zlogrhoa2 -
        3.635186504599571e-10*zt3*zlogrh*zlogrhoa2 +
        (0.0006100650851863252*zlogrh*zlogrhoa2)*zix +
        0.0003201836700403512*zlogrhoa3 -
        0.0000174761713262546*zt*zlogrhoa3 +
        6.065037668052182e-8*zt2*zlogrhoa3 -
        1.421771723004557e-11*zt3*zlogrhoa3 +
        (0.0001357509859501723*zlogrhoa3)*zix)
    zntot = jnp.exp(zntot)
    rate = jnp.where((zjnuc < 1e-7) | (zntot < 4.), 0., zjnuc)
    rate = jnp.minimum(rate, 1e10)
    return NucleationRate(rate, zntot * x)


class KazilLovejoyTable(NamedTuple):
    """PARNUC lookup table (``mo_ham_m7_nucl.F90::ham_nucl_initialize``).

    Axes are the native ascending coordinate arrays: ``temperature`` [K],
    ``relative_humidity`` [%], ``h2so4`` [cm-3], ``ion_pair_rate`` [cm-3 s-1],
    ``condensation_sink`` [s-1]. ``log_pfr`` is ``ln`` of the tabulated
    particle formation rate [cm-3 s-1], shaped
    ``(len(temperature), len(relative_humidity), len(h2so4),
    len(ion_pair_rate), len(condensation_sink))`` -- the file's own variable
    order (``pfr(temperature, RH, [H2SO4], ionization, condensation_sink)``,
    confirmed against the staged ``parnuc.15H2SO4.A0.total.nc``), not the
    Fortran's column-major storage order (``pfr_table(cs_n,ipr_n,h2so4_n,
    rh_n,t_n)``, chosen there only because NetCDF's row-major file layout
    maps to a Fortran array with dimensions reversed) -- :func:`kazil_lovejoy`
    indexes this table with axes in the order named here. The table itself
    already holds ``ln(rate)``, not the rate (``ppfr = exp(...)`` is the very
    last step of ``nucl_kazil_lovejoy``, applied once per query by
    :func:`kazil_lovejoy`, not baked into this table).
    """
    temperature: object
    relative_humidity: object
    h2so4: object
    ion_pair_rate: object
    condensation_sink: object
    log_pfr: object


def load_kazil_lovejoy_table(path):
    """Read the PARNUC table outside JIT (SciPy NetCDF reader, like `load_kappa_table`).

    ``ham_nucl_initialize`` reads the dimension/variable 'H2SO4', falling
    back to the obsolete '[H2SO4]' name if 'H2SO4' is absent -- both are
    accepted here, in the same order. Every axis is validated finite and
    strictly increasing (the bisection search in :func:`kazil_lovejoy`
    assumes it); the table itself is validated finite.
    """
    with netcdf_file(Path(path), mmap=False) as data:
        names = data.variables
        h2so4_name = 'H2SO4' if 'H2SO4' in names else '[H2SO4]'
        axis_names = ('temperature', 'RH', h2so4_name, 'ionization', 'condensation_sink')
        axes = [np.array(names[name].data, dtype=np.float64, copy=True) for name in axis_names]
        pfr = np.array(names['pfr'].data, dtype=np.float64, copy=True)
    if pfr.shape != tuple(len(a) for a in axes):
        raise ValueError('Unexpected Kazil/Lovejoy table axis ordering')
    if any(not np.isfinite(a).all() for a in axes) or any(np.any(np.diff(a) <= 0) for a in axes):
        raise ValueError('Nonfinite lookup or unordered coordinate axis')
    if not np.isfinite(pfr).all():
        raise ValueError('Nonfinite particle formation rate in table')
    return KazilLovejoyTable(*(jnp.asarray(x) for x in (*axes, pfr)))


def _bracket(table_axis, value):
    """0-based bracketing indices ``(i0, i0+1)`` such that
    ``table_axis[i0] <= value <= table_axis[i0+1]`` (``value`` already
    clamped into the axis range by the caller).

    Vectorized equivalent of ``nucl_kazil_lovejoy``'s bisection search
    (lines ~485-558): that loop converges to the unique adjacent pair
    bracketing ``value`` for a strictly increasing table, which is exactly
    ``searchsorted(..., side='right') - 1`` (the count of entries ``<=
    value``, minus one) -- verified against the Fortran at every table
    entry and every strictly-between point, including the ``value ==
    table_axis[-1]`` edge (clipped to ``n-2`` so ``i0+1`` stays in range,
    matching the Fortran converging to ``(n-1, n)`` there instead of
    dividing by zero at ``(n, n)``).
    """
    n = table_axis.shape[0]
    i0 = jnp.clip(jnp.searchsorted(table_axis, value, side='right') - 1, 0, n - 2)
    return i0, i0 + 1


def kazil_lovejoy(temperature_k, relative_humidity_pct, h2so4_cm3,
                  condensation_sink, ion_pair_rate, table):
    """Port of ``nucl_kazil_lovejoy`` (mo_ham_m7_nucl.F90:320-668).

    Inputs/units match the Fortran exactly: ``relative_humidity_pct`` is RH
    in **percent** (0-100), not the 0-1 fraction :func:`vehkamaeki` and the
    rest of this module use (``nucl_kazil_lovejoy`` is called with ``zrh =
    100*prhp1`` in ``m7_nuck`` -- :func:`nucleate_all` does that conversion
    before calling this, so its own public ``relative_humidity`` argument
    stays a 0-1 fraction for every scheme); ``h2so4_cm3`` is the CLEAR-SKY
    (cloud-free-portion) concentration, ``condensation_sink`` [s-1],
    ``ion_pair_rate`` [cm-3 s-1].

    Returns :class:`NucleationRate`: the formation rate [cm-3 s-1] (zero
    where temperature is above the table's range, RH or [H2SO4] are below
    it -- the native ``lset_zero`` branches, lines 422-473) and the fixed
    cluster size of 15 H2SO4 molecules (``pns``, line 400) wherever the
    rate is nonzero, matching the Fortran's unconditional
    ``pns(1:kproma,:) = 15.0_dp`` before the per-cell clamp/interpolation
    loop (so a `lset_zero` cell's returned cluster size is immaterial --
    its rate is zero either way, and the caller's mass-conservation ratio
    ``cluster*rate`` is zero too).
    """
    # Work in the gas field's dtype, like the other nucleation routines: the
    # step_all chain passes whatever working precision it runs (float64, or
    # the float32 forward path).
    zh2so4 = jnp.asarray(h2so4_cm3)
    dtype = zh2so4.dtype
    zt = jnp.asarray(temperature_k, dtype=dtype)
    zrh = jnp.asarray(relative_humidity_pct, dtype=dtype)
    zcs = jnp.asarray(condensation_sink, dtype=dtype)
    zipr = jnp.asarray(ion_pair_rate, dtype=dtype)
    # The table is loaded in float64; bring it to the working dtype so a
    # float32 run never mixes in float64 arrays.
    table = KazilLovejoyTable(*(jnp.asarray(x, dtype=dtype) for x in table))

    t_lo, t_hi = table.temperature[0], table.temperature[-1]
    rh_lo, rh_hi = table.relative_humidity[0], table.relative_humidity[-1]
    h_lo, h_hi = table.h2so4[0], table.h2so4[-1]
    i_lo, i_hi = table.ion_pair_rate[0], table.ion_pair_rate[-1]
    c_lo, c_hi = table.condensation_sink[0], table.condensation_sink[-1]

    # Native lset_zero: ztemp too high, OR zrh/zh2so4 too low (lines 427-453).
    # ipr/cs out-of-range only clamps (never zeroes); temp-too-low/rh-too-high/
    # h2so4-too-high likewise only clamp.
    set_zero = (zt > t_hi) | (zrh < rh_lo) | (zh2so4 < h_lo)

    zt = jnp.clip(zt, t_lo, t_hi)
    zrh = jnp.clip(zrh, rh_lo, rh_hi)
    zh2so4 = jnp.clip(zh2so4, h_lo, h_hi)
    zipr = jnp.clip(zipr, i_lo, i_hi)
    zcs = jnp.clip(zcs, c_lo, c_hi)

    it0, it1 = _bracket(table.temperature, zt)
    ir0, ir1 = _bracket(table.relative_humidity, zrh)
    ih0, ih1 = _bracket(table.h2so4, zh2so4)
    ii0, ii1 = _bracket(table.ion_pair_rate, zipr)
    ic0, ic1 = _bracket(table.condensation_sink, zcs)

    zv = (zt - table.temperature[it0]) / (table.temperature[it1] - table.temperature[it0])
    zw = (zrh - table.relative_humidity[ir0]) / (table.relative_humidity[ir1] - table.relative_humidity[ir0])
    zx = (zh2so4 - table.h2so4[ih0]) / (table.h2so4[ih1] - table.h2so4[ih0])
    zy = (zipr - table.ion_pair_rate[ii0]) / (table.ion_pair_rate[ii1] - table.ion_pair_rate[ii0])
    zz = (zcs - table.condensation_sink[ic0]) / (table.condensation_sink[ic1] - table.condensation_sink[ic0])

    zv_, zw_, zx_, zy_, zz_ = zv - 1., zw - 1., zx - 1., zy - 1., zz - 1.

    zvw, zvw_, zv_w, zv_w_ = zv*zw, zv*zw_, zv_*zw, zv_*zw_
    zxy, zxy_, zx_y, zx_y_ = zx*zy, zx*zy_, zx_*zy, zx_*zy_

    zxyz, zxyz_ = zxy*zz, zxy*zz_
    zxy_z, zxy_z_ = zxy_*zz, zxy_*zz_
    zx_yz, zx_yz_ = zx_y*zz, zx_y*zz_
    zx_y_z, zx_y_z_ = zx_y_*zz, zx_y_*zz_

    # Fortran's pfr_table(i_cs,i_ipr,i_h2so4,i_rh,i_temp) (column-major, cs
    # fastest) addresses the SAME corner as this table's log_pfr[it,ir,ih,ii,ic]
    # (see KazilLovejoyTable's docstring) -- every corner lookup below just
    # reorders/relabels the same five 0-based brackets, index-for-index.
    p = table.log_pfr
    ppfr = (
        - (zv_w_*zx_y_z_) * p[it0, ir0, ih0, ii0, ic0]
        + (zv_w_*zx_y_z)  * p[it0, ir0, ih0, ii0, ic1]
        + (zv_w_*zx_yz_)  * p[it0, ir0, ih0, ii1, ic0]
        - (zv_w_*zx_yz)   * p[it0, ir0, ih0, ii1, ic1]
        + (zv_w_*zxy_z_)  * p[it0, ir0, ih1, ii0, ic0]
        - (zv_w_*zxy_z)   * p[it0, ir0, ih1, ii0, ic1]
        - (zv_w_*zxyz_)   * p[it0, ir0, ih1, ii1, ic0]
        + (zv_w_*zxyz)    * p[it0, ir0, ih1, ii1, ic1]
        + (zv_w*zx_y_z_)  * p[it0, ir1, ih0, ii0, ic0]
        - (zv_w*zx_y_z)   * p[it0, ir1, ih0, ii0, ic1]
        - (zv_w*zx_yz_)   * p[it0, ir1, ih0, ii1, ic0]
        + (zv_w*zx_yz)    * p[it0, ir1, ih0, ii1, ic1]
        - (zv_w*zxy_z_)   * p[it0, ir1, ih1, ii0, ic0]
        + (zv_w*zxy_z)    * p[it0, ir1, ih1, ii0, ic1]
        + (zv_w*zxyz_)    * p[it0, ir1, ih1, ii1, ic0]
        - (zv_w*zxyz)     * p[it0, ir1, ih1, ii1, ic1]
        + (zvw_*zx_y_z_)  * p[it1, ir0, ih0, ii0, ic0]
        - (zvw_*zx_y_z)   * p[it1, ir0, ih0, ii0, ic1]
        - (zvw_*zx_yz_)   * p[it1, ir0, ih0, ii1, ic0]
        + (zvw_*zx_yz)    * p[it1, ir0, ih0, ii1, ic1]
        - (zvw_*zxy_z_)   * p[it1, ir0, ih1, ii0, ic0]
        + (zvw_*zxy_z)    * p[it1, ir0, ih1, ii0, ic1]
        + (zvw_*zxyz_)    * p[it1, ir0, ih1, ii1, ic0]
        - (zvw_*zxyz)     * p[it1, ir0, ih1, ii1, ic1]
        - (zvw*zx_y_z_)   * p[it1, ir1, ih0, ii0, ic0]
        + (zvw*zx_y_z)    * p[it1, ir1, ih0, ii0, ic1]
        + (zvw*zx_yz_)    * p[it1, ir1, ih0, ii1, ic0]
        - (zvw*zx_yz)     * p[it1, ir1, ih0, ii1, ic1]
        + (zvw*zxy_z_)    * p[it1, ir1, ih1, ii0, ic0]
        - (zvw*zxy_z)     * p[it1, ir1, ih1, ii0, ic1]
        - (zvw*zxyz_)     * p[it1, ir1, ih1, ii1, ic0]
        + (zvw*zxyz)      * p[it1, ir1, ih1, ii1, ic1]
    )
    rate = jnp.where(set_zero, 0., jnp.exp(ppfr))
    cluster = jnp.full_like(rate, 15.)
    return NucleationRate(rate, cluster)


def organic_activation(h2so4_clear, forest, in_pbl):
    """Native ``nucl_activation`` (mo_ham_m7_nucl.F90:672-736).

    Activation-type organic nucleation (Kulmala et al., ACP 2006), active
    only over forests and only in the boundary layer. ``in_pbl`` is a
    per-cell boolean the host precomputes as the Fortran's own level test,
    ``jk >= int(ppbl)`` with a 1-based, top-first ``jk``; this routine has no
    notion of levels itself. Returns the formation rate [cm-3 s-1] and the
    number of H2SO4 molecules in each new particle (always 1 here).
    """
    # Dtype-generic (#1017 W1 task 3): anchor on h2so4_clear, the carrier the
    # step_all chain passes in whatever working dtype it is running.
    h2so4_clear = jnp.asarray(h2so4_clear)
    forest = jnp.asarray(forest, dtype=h2so4_clear.dtype)
    in_pbl = jnp.asarray(in_pbl)
    rate = jnp.where(in_pbl, 1.0e-6 * h2so4_clear * forest, 0.0)
    return NucleationRate(rate, jnp.where(in_pbl, 1.0, 0.0))


def organic_kinetic(h2so4_clear, forest, in_pbl):
    """Native ``nucl_kinetic`` (mo_ham_m7_nucl.F90:740-808).

    Kinetic-type organic nucleation (Laakso et al., ACP 2004), same forest/
    boundary-layer gating as :func:`organic_activation`. Each new particle
    carries 2 H2SO4 molecules here.
    """
    h2so4_clear = jnp.asarray(h2so4_clear)
    forest = jnp.asarray(forest, dtype=h2so4_clear.dtype)
    in_pbl = jnp.asarray(in_pbl)
    rate = jnp.where(in_pbl, 3.86e-13 * h2so4_clear * h2so4_clear * forest, 0.0)
    return NucleationRate(rate, jnp.where(in_pbl, 2.0, 0.0))


def _deplete(gas, clear_gas, cloud, consumed):
    """H2SO4 taken by nucleation and the gas left, ``zdelta`` and
    ``zgso4 - zdelta`` of m7_nuck (mo_ham_m7.F90:2257-2465).

    ``consumed = dt*loss/(1 + dt*sink)`` is the cloud-free depletion. Native::

        clear_final = MIN(MAX(clear_gas - consumed, 0), clear_gas)
        delta       = (clear_gas - clear_final) * (1 - cloud)
        gas_left    = gas - delta

    Both differences subtract nearly equal totals: ``delta`` whenever the
    depletion is small next to the reservoir (in float32 a depletion below
    one rounding of a 1e7 cm-3 reservoir becomes exactly zero, so no
    particles form at all), and ``gas_left`` whenever nucleation exhausts
    the cloud-free gas (``gas = clear_gas*(1 - cloud)``, so the result is
    rounding noise of either sign: -4e-8 molecules cm-3 in float64). In real
    arithmetic ``clear_gas - clear_final = clip(consumed, 0, clear_gas)``,
    and ``gas_left = clear_final*(1 - cloud)`` plus the gas outside the
    cloud-free part (zero for 0 < cloud < 1, ``gas*cloud`` for cloud <= 0,
    ``gas`` for cloud >= 1, where ``clear_gas = 0``). Narrower-than-float64
    dtypes (:func:`precision.cancellation_free`) use those non-negative
    forms; float64 keeps the native differences.
    """
    clear_final = jnp.minimum(jnp.maximum(clear_gas - consumed, 0.), clear_gas)
    if not precision.cancellation_free(jnp.result_type(gas, clear_gas, consumed)):
        delta = (clear_gas - clear_final) * (1. - cloud)
        return gas - delta, delta
    delta = jnp.minimum(jnp.maximum(consumed, 0.), clear_gas) * (1. - cloud)
    outside = jnp.where(cloud >= 1., gas, jnp.where(cloud <= 0., gas * cloud, 0.))
    return outside + clear_final * (1. - cloud), delta


def nucleate_all(gas_cm3, temperature_k, relative_humidity, sink, cloud_fraction,
                  forest, in_pbl, timestep, *, scheme=1, organic_scheme=0,
                  initial_number=0., initial_sulfate=0.,
                  ion_pair_rate=0., kazil_lovejoy_table=None):
    """Port standalone ``m7_nuck`` (mo_ham_m7.F90:2257-2465), combining
    sulfuric-acid nucleation (``scheme``, as in :func:`nucleate`) with
    organic nucleation (``organic_scheme``: 0 off, 1 activation, 2 kinetic)
    exactly as the native routine does -- summed rates, combined H2SO4 loss,
    the same cloud-free scaling and the ``zeps`` activity gate -- so that the
    number/mass increments come out identical whichever mechanism (or both)
    is active. ``scheme`` and ``organic_scheme`` are static JIT arguments.

    ``scheme=2`` (Kazil and Lovejoy 2007) additionally needs
    ``ion_pair_rate`` [cm-3 s-1] (``pipr`` -- the host's GCR ion-pair
    production; jcm's :mod:`jam.chemistry.gcr_ionisation` is the source of
    that field, Part B of jax-gcm#1017's Kazil/GCR task) and
    ``kazil_lovejoy_table`` (:func:`load_kazil_lovejoy_table`'s result).
    ``relative_humidity`` stays this function's native 0-1 fraction for
    every scheme; the native ``zrh = 100*prhp1`` percent conversion ``m7_nuck``
    applies only ahead of its ``nucl_kazil_lovejoy`` call happens here, not
    at this function's public boundary.
    """
    if scheme == 2 and kazil_lovejoy_table is None:
        raise ValueError(
            'M7 Kazil/Lovejoy nucleation (scheme=2) requires kazil_lovejoy_table '
            '(load_kazil_lovejoy_table(path), path from HAM_INPUT_DIR)'
        )
    if scheme not in (0, 1, 2):
        raise ValueError('Supported M7 sulfur nucleation schemes are 0, 1 and 2')
    if organic_scheme not in (0, 1, 2):
        raise ValueError('Supported M7 organic nucleation schemes are 0, 1 and 2')
    gas, cloud, sink = jnp.asarray(gas_cm3), jnp.asarray(cloud_fraction), jnp.asarray(sink)
    clear_gas = jnp.where(cloud >= 1., 0.,
        jnp.where(cloud <= 0., gas, gas / jnp.where(cloud < 1., 1. - cloud, 1.)))
    if scheme == 1:
        snrate, sns = vehkamaeki(temperature_k, relative_humidity, clear_gas)
    elif scheme == 2:
        snrate, sns = kazil_lovejoy(
            temperature_k, jnp.asarray(relative_humidity)*100., clear_gas,
            sink, ion_pair_rate, kazil_lovejoy_table,
        )
    else:
        snrate = sns = jnp.zeros_like(clear_gas)
    if organic_scheme == 1:
        onrate, ons = organic_activation(clear_gas, forest, in_pbl)
    elif organic_scheme == 2:
        onrate, ons = organic_kinetic(clear_gas, forest, in_pbl)
    else:
        onrate = ons = jnp.zeros_like(clear_gas)
    total_rate = snrate + onrate
    active = total_rate > EPS_DP
    loss = sns * snrate + ons * onrate
    remaining, delta = _deplete(gas, clear_gas, cloud, timestep * loss / (1. + timestep * sink))
    number = delta * total_rate / jnp.where(active, loss, 1.)
    return NucleationResult(jnp.where(active, remaining, gas),
        jnp.where(active, number, initial_number),
        jnp.where(active, delta, initial_sulfate))


def nucleate(gas_cm3, temperature_k, relative_humidity, sink, cloud_fraction,
             timestep, *, scheme=1, initial_number=0., initial_sulfate=0.):
    """Port standalone m7_nuck, returning gas and new-particle/sulfate deltas.

    Inputs gas/sulfate are molecules/cm3, number particles/cm3, sink s-1 and
    timestep seconds. The scheme is a static JIT argument (0 disabled, 1 binary).
    Inactive rates preserve the caller's initial delta values, matching native
    INOUT behavior. Standard composition initializes both deltas to zero.
    """
    if scheme == 2:
        raise NotImplementedError('M7 Kazil/Lovejoy nucleation requires its external lookup table')
    if scheme not in (0, 1):
        raise ValueError('Supported M7 sulfur nucleation schemes are 0 and 1')
    gas, cloud, sink = jnp.asarray(gas_cm3), jnp.asarray(cloud_fraction), jnp.asarray(sink)
    if scheme == 0:
        shape = jnp.broadcast_shapes(gas.shape, cloud.shape, sink.shape)
        return NucleationResult(jnp.broadcast_to(gas, shape),
                                jnp.broadcast_to(jnp.asarray(initial_number), shape),
                                jnp.broadcast_to(jnp.asarray(initial_sulfate), shape))
    clear_gas = jnp.where(cloud >= 1., 0.,
        jnp.where(cloud <= 0., gas, gas / jnp.where(cloud < 1., 1. - cloud, 1.)))
    rate, cluster = vehkamaeki(temperature_k, relative_humidity, clear_gas)
    active = rate > EPS_DP
    loss = cluster * rate
    remaining, delta = _deplete(gas, clear_gas, cloud, timestep * loss / (1. + timestep * sink))
    number = delta * rate / jnp.where(active, loss, 1.)
    return NucleationResult(jnp.where(active, remaining, gas),
        jnp.where(active, number, initial_number),
        jnp.where(active, delta, initial_sulfate))
