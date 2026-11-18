"""Native M7 cumulative-normal approximation and pure-sulfate mode transfer."""
import functools
import operator

import jax.numpy as jnp

from . import _ad
from .constants import (PI, AVOGADRO, H2SO4_MOLAR_MASS, SULFATE_DENSITY,
    MODE_RADIUS_BOUNDS, CMR2RAM, SIGMA_LOG, CMIN_AERNL, CMINRAD,
    COMPONENT_SPECIES, COMPONENT_MODE, SPECIES_DENSITY, FAC_UG, CDCONV,
    ZFACSO4M, ZFACSO4V, EPS_DP)

A = (2.2352520354606837, 161.02823106855587, 1067.6894854603709, 18154.98125334356, 0.06568233791820745)
B = (47.202581904688245, 976.0985517377767, 10260.932208618979, 45507.78933502673)
C = (0.39894151208813466, 8.883149794388377, 93.50665613217785, 597.2702763948002, 2494.5375852903726, 6848.190450536283, 11602.65143764735, 9842.714838383978, 1.0765576773720192e-08)
D = (22.266688044328117, 235.387901782625, 1519.3775994075547, 6485.558298266761, 18615.571640885097, 34900.95272114598, 38912.00328609327, 19685.429676859992)
P = (0.215898534057957, 0.12740116116024736, 0.022235277870649807, 0.0014216191932278934, 2.9112874951168793e-05, 0.023073441764940174)
Q = (1.284260096144911, 0.4682382124808651, 0.06598813786892856, 0.0037823963320275824, 7.297515550839662e-05)


def _cumulative_normal_core(argument):
    """Shared rational branches and single tail exponential for :func:`cumulative_normal`."""
    x = jnp.clip(jnp.asarray(argument), -40., 40.)
    y = jnp.abs(x)
    eps = EPS_DP
    small = y <= .66291
    middle = y <= 5.656854248
    xs = jnp.where(small, x, 0.)
    square = jnp.where(jnp.abs(xs) > eps*.5, xs*xs, 0.)
    numerator, denominator = A[4]*square, square
    for i in range(3):
        numerator = (numerator+A[i])*square
        denominator = (denominator+B[i])*square
    central = xs*(numerator+A[3])/(denominator+B[3])
    ym = jnp.where(~small & middle, y, 1.)
    numerator, denominator = C[8]*ym, ym
    for i in range(7):
        numerator = (numerator+C[i])*ym
        denominator = (denominator+D[i])*ym
    mid_fraction = (numerator+C[7])/(denominator+D[7])
    xl = jnp.where(~middle, x, 8.)
    square = 1/(xl*xl)
    numerator, denominator = P[5]*square, square
    for i in range(4):
        numerator = (numerator+P[i])*square
        denominator = (denominator+Q[i])*square
    far_fraction = square*(numerator+P[4])/(denominator+Q[4])
    far_fraction = (.39894228040143267794-far_fraction)/jnp.abs(xl)
    t_arg = jnp.where(middle, ym, xl)
    truncated = jnp.trunc(t_arg*16)/16
    delta = (t_arg-truncated)*(t_arg+truncated)
    tail = jnp.exp(-truncated*truncated*.5)*jnp.exp(-delta*.5)*jnp.where(middle, mid_fraction, far_fraction)
    return x, small, central, tail


def _cumulative_normal_cdf(argument):
    """Return only the CDF output of :func:`cumulative_normal`."""
    x, small, central, tail = _cumulative_normal_core(argument)
    cdf = jnp.where(small, .5+central, jnp.where(x > 0, 1-tail, tail))
    return jnp.where(cdf < EPS_DP, 0., cdf)


def cumulative_normal(argument):
    """Original three rational branches and epsilon tail cutoff, not scipy CDF."""
    x, small, central, tail = _cumulative_normal_core(argument)
    eps = EPS_DP
    cdf = jnp.where(small, .5+central, jnp.where(x > 0, 1-tail, tail))
    complement = jnp.where(small, .5-central, jnp.where(x > 0, tail, 1-tail))
    return jnp.where(cdf < eps, 0., cdf), jnp.where(complement < eps, 0., complement)


def repartition(sulfate, number, dry_radius):
    """Native simultaneous upward transfer between four soluble lognormal modes.

    Sulfate: molecules/cm3 (...,4); number: cm-3 (...,7); dry count median
    radii: cm (...,4). Empty insoluble modes are retained, never populated.
    """
    sulfate, number, dry_radius = map(jnp.asarray, (sulfate, number, dry_radius))
    dtype = sulfate.dtype
    eps = EPS_DP
    fractions_n = jnp.ones_like(sulfate)
    fractions_m = jnp.ones_like(sulfate)
    for mode in range(3):
        lower, upper = MODE_RADIUS_BOUNDS[mode:mode+2]
        lower_d = jnp.asarray(lower, dtype=dtype)
        upper_d = jnp.asarray(upper, dtype=dtype)
        average_radius = jnp.exp((jnp.log(lower_d)+jnp.log(upper_d))/2)*CMR2RAM[mode]
        mode_mass = sulfate[..., mode]*H2SO4_MOLAR_MASS/AVOGADRO
        mode_volume = sulfate[..., mode]*H2SO4_MOLAR_MASS/AVOGADRO/SULFATE_DENSITY
        n, radius = number[..., mode], dry_radius[..., mode]
        active = (n > CMIN_AERNL) & (radius > CMINRAD) & (mode_volume > 0)
        density = mode_mass/jnp.where(active, mode_volume, 1.)
        particle_mass = mode_mass/jnp.where(active, n, 1.)
        target_mass = (4*PI/3)*average_radius**3*density
        upper_mass = (4*PI/3)*upper**3*density
        active = active & (particle_mass > target_mass)
        z = jnp.log(upper/jnp.where(active, radius, upper))/SIGMA_LOG[mode]
        cdf = _cumulative_normal_cdf(z)
        stay_n = jnp.where(radius > upper*10, 0., jnp.where(cdf > eps, cdf, 1.))
        transferred_n = (1-stay_n)*n
        transfer = active & (mode_mass > 0) & (n > 0) & (transferred_n > 0)
        remaining_mass = jnp.maximum(stay_n*n*target_mass, 0.)
        transferred_mass = mode_mass-remaining_mass
        stay_m = remaining_mass/jnp.where(transfer, mode_mass, 1.)
        adjust = transferred_mass/jnp.where(transfer, transferred_n, 1.) < upper_mass
        stay_n = jnp.where(adjust, 1-(transferred_mass/jnp.where(transfer, upper_mass, 1.))/jnp.where(transfer, n, 1.), stay_n)
        fractions_n = fractions_n.at[..., mode].set(jnp.where(transfer, stay_n, 1.))
        fractions_m = fractions_m.at[..., mode].set(jnp.where(transfer, stay_m, 1.))
    new_number = number[..., :4]*fractions_n
    new_number = new_number.at[..., 1:].add(number[..., :3]*(1-fractions_n[..., :3]))
    new_mass = sulfate*fractions_m
    new_mass = new_mass.at[..., 1:].add(sulfate[..., :3]*(1-fractions_m[..., :3]))
    return new_mass, number.at[..., :4].set(new_number)


# --- Multi-species redistribution (m7_dconc, mo_ham_m7.F90:2788-3337) ------
# Static component bookkeeping, derived once from constants.COMPONENT_MODE/
# COMPONENT_SPECIES: which 18-component indices feed each of the three
# source modes (NS, KS, AS; CS=mode 3 never emits - `DO jclass=1,nsol-1`),
# and, for each of those components, the unique same-species component one
# mode up that receives its transferred mass. Every soluble-mode species in
# this table recurs in the next mode up (mo_ham_m7.F90's per-class SELECT
# CASE lists bear this out: e.g. ibcks->ibcas->ibccs, issas->isscs), so the
# lookup is total over its domain; a missing target would be a layout bug,
# not a native branch, so it fails fast at import time rather than silently
# dropping mass.
_MODE_MEMBERS = {mode: tuple(jn for jn in range(18) if COMPONENT_MODE[jn] == mode) for mode in range(4)}


def _upward_target(jn):
    mode, species = COMPONENT_MODE[jn], COMPONENT_SPECIES[jn]
    targets = [k for k in range(18) if COMPONENT_MODE[k] == mode+1 and COMPONENT_SPECIES[k] == species]
    if len(targets) != 1:
        raise RuntimeError(f'component {jn} (mode {mode}, species {species}) has no unique upward target')
    return targets[0]


_TRANSFER_TARGET = {jn: _upward_target(jn) for mode in range(3) for jn in _MODE_MEMBERS[mode]}
_SOURCE_OF = {target: jn for jn, target in _TRANSFER_TARGET.items()}


# m7_dconc's per-component mass and volume factors (sulfate in molecules, others in ug).
_MASS_FACTOR = tuple(ZFACSO4M if COMPONENT_SPECIES[jn] == 0 else FAC_UG for jn in range(18))
_VOLUME_FACTOR = tuple(ZFACSO4V if COMPONENT_SPECIES[jn] == 0
                       else FAC_UG / (CDCONV * SPECIES_DENSITY[COMPONENT_SPECIES[jn]]) for jn in range(18))


def _repartition_all_soa(mass18, number7, dry_radius4):
    """Tuple-of-arrays implementation of ``m7_dconc``."""
    dtype = mass18[0].dtype
    eps = EPS_DP
    fn = []
    fm = []
    fn_leave = []
    fm_leave = []
    for mode in range(3):
        members = _MODE_MEMBERS[mode]
        total_mass = functools.reduce(operator.add, [_MASS_FACTOR[jn] * mass18[jn] for jn in members])
        total_volume = functools.reduce(operator.add, [_VOLUME_FACTOR[jn] * mass18[jn] for jn in members])
        lower, upper = MODE_RADIUS_BOUNDS[mode:mode+2]
        lower_d = jnp.asarray(lower, dtype=dtype)
        upper_d = jnp.asarray(upper, dtype=dtype)
        average_radius = jnp.exp((jnp.log(lower_d) + jnp.log(upper_d)) / 2) * CMR2RAM[mode]
        n, radius = number7[mode], dry_radius4[mode]
        active = (n > CMIN_AERNL) & (radius > CMINRAD) & (total_volume > 0)
        density = _ad.div(total_mass, jnp.where(active, total_volume, 1.))
        particle_mass = _ad.div(total_mass, jnp.where(active, n, 1.))
        target_mass = (4 * PI / 3) * average_radius**3 * density
        upper_mass = (4 * PI / 3) * upper**3 * density
        active = active & (particle_mass > target_mass)
        z = jnp.log(upper / jnp.where(active, radius, upper)) / SIGMA_LOG[mode]
        cdf = _cumulative_normal_cdf(z)
        stay_n = jnp.where(radius > upper * 10, 0., jnp.where(cdf > eps, cdf, 1.))
        transferred_n = (1 - stay_n) * n
        transfer = active & (total_mass > 0) & (n > 0) & (transferred_n > 0)
        remaining_mass = jnp.maximum(stay_n * n * target_mass, 0.)
        transferred_mass = total_mass - remaining_mass
        stay_m = _ad.div(remaining_mass, jnp.where(transfer, total_mass, 1.))
        adjust = _ad.div(transferred_mass, jnp.where(transfer, transferred_n, 1.)) < upper_mass
        stay_n = jnp.where(
            adjust,
            1 - _ad.div(_ad.div(transferred_mass, jnp.where(transfer, upper_mass, 1.)),
                        jnp.where(transfer, n, 1.)),
            stay_n,
        )
        fn_m = jnp.where(transfer, stay_n, 1.)
        fm_m = jnp.where(transfer, stay_m, 1.)
        fn.append(fn_m)
        fm.append(fm_m)
        fn_leave.append(1 - fn_m)
        fm_leave.append(1 - fm_m)

    new_number = (
        number7[0] * fn[0],
        number7[1] * fn[1] + number7[0] * fn_leave[0],
        number7[2] * fn[2] + number7[1] * fn_leave[1],
        number7[3] + number7[2] * fn_leave[2],
        number7[4],
        number7[5],
        number7[6],
    )

    # Every component's new mass is its own retained fraction plus whatever
    # (at most one) same-species component one mode down transferred into it;
    # both terms read the ORIGINAL mass, matching the native routine computing
    # all zsumm entries from the untouched paerml before writing any back
    # (mo_ham_m7.F90:3072-3077 and below). CS never transfers upward.
    columns = []
    for jn in range(18):
        mode = COMPONENT_MODE[jn]
        value = mass18[jn] * fm[mode] if mode < 3 else mass18[jn]
        source = _SOURCE_OF.get(jn)
        if source is not None:
            value = value + mass18[source] * fm_leave[COMPONENT_MODE[source]]
        columns.append(value)
    return tuple(columns), new_number


def repartition_all(mass, number, dry_radius):
    """Port of m7_dconc (mo_ham_m7.F90:2788-3337): simultaneous upward
    transfer of every species component of a soluble mode, by the same mass
    fraction that mode's own (all-species) geometry says has grown past its
    upper radius bound. Insoluble modes, and their components, are never
    redistributed (the native routine only ever loops `jclass=1,nsol-1`).

    mass: (...,18) HAM aerocomp order, native units (molecules cm-3 for the
    four sulfate components, ug m-3 for the rest). number: (...,7) cm-3.
    dry_radius: (...,4) cm, the four soluble modes' dry count-median radius
    (`properties.kappa_hydration`'s `dry_radius`).

    Per-component density/volume factors mirror mo_ham_m7.F90:2951-3051's
    zfacso4m/zfacso4v (sulphate, using the H2SO4 weight wh2so4 - not the
    species-registry mw_so4 that `kappa_hydration` uses, matching the native
    code's own inherited inconsistency) and zfac2cgs/dbc/doc/dnacl/ddust for
    the rest (equal to CDCONV*SPECIES_DENSITY given the registry densities
    mo_ham_m7ctl.F90 hardcodes for BC/OC/SS/DU, so the registry is reused
    here rather than duplicating those four constants).
    """
    mass = jnp.asarray(mass)
    dtype = mass.dtype
    number = jnp.asarray(number, dtype=dtype)
    dry_radius = jnp.asarray(dry_radius, dtype=dtype)
    mass18 = tuple(mass[..., jn] for jn in range(18))
    number7 = tuple(number[..., m] for m in range(7))
    dry4 = tuple(dry_radius[..., m] for m in range(4))
    out_mass18, out_number7 = _repartition_all_soa(mass18, number7, dry4)
    return jnp.stack(out_mass18, axis=-1), jnp.stack(out_number7, axis=-1)
