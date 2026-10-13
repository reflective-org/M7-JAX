# Copyright 2026 Reflective
# Portions derived from HAMMOZ M7, Copyright (c) 2021 hammoz;
# see NOTICE and LICENSES/BSD-3-Clause-HAMMOZ.txt.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
"""Native M7 cumulative-normal approximation and pure-sulfate mode transfer."""
import jax.numpy as jnp
from .constants import (PI, AVOGADRO, H2SO4_MOLAR_MASS, SULFATE_DENSITY,
    MODE_RADIUS_BOUNDS, CMR2RAM, SIGMA_LOG, CMIN_AERNL, CMINRAD)

A = (2.2352520354606837, 161.02823106855587, 1067.6894854603709, 18154.98125334356, 0.06568233791820745)
B = (47.202581904688245, 976.0985517377767, 10260.932208618979, 45507.78933502673)
C = (0.39894151208813466, 8.883149794388377, 93.50665613217785, 597.2702763948002, 2494.5375852903726, 6848.190450536283, 11602.65143764735, 9842.714838383978, 1.0765576773720192e-08)
D = (22.266688044328117, 235.387901782625, 1519.3775994075547, 6485.558298266761, 18615.571640885097, 34900.95272114598, 38912.00328609327, 19685.429676859992)
P = (0.215898534057957, 0.12740116116024736, 0.022235277870649807, 0.0014216191932278934, 2.9112874951168793e-05, 0.023073441764940174)
Q = (1.284260096144911, 0.4682382124808651, 0.06598813786892856, 0.0037823963320275824, 7.297515550839662e-05)


def cumulative_normal(argument):
    """Original three rational branches and epsilon tail cutoff, not scipy CDF."""
    x = jnp.clip(jnp.asarray(argument), -40., 40.)
    y = jnp.abs(x)
    eps = jnp.finfo(jnp.float64).eps
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
    fraction = (numerator+C[7])/(denominator+D[7])
    truncated = jnp.trunc(ym*16)/16
    delta = (ym-truncated)*(ym+truncated)
    mid_tail = jnp.exp(-truncated*truncated*.5)*jnp.exp(-delta*.5)*fraction
    xl = jnp.where(~middle, x, 8.)
    square = 1/(xl*xl)
    numerator, denominator = P[5]*square, square
    for i in range(4):
        numerator = (numerator+P[i])*square
        denominator = (denominator+Q[i])*square
    fraction = square*(numerator+P[4])/(denominator+Q[4])
    fraction = (.39894228040143267794-fraction)/jnp.abs(xl)
    truncated = jnp.trunc(xl*16)/16
    delta = (xl-truncated)*(xl+truncated)
    far_tail = jnp.exp(-truncated*truncated*.5)*jnp.exp(-delta*.5)*fraction
    tail = jnp.where(middle, mid_tail, far_tail)
    cdf = jnp.where(small, .5+central, jnp.where(x > 0, 1-tail, tail))
    complement = jnp.where(small, .5-central, jnp.where(x > 0, tail, 1-tail))
    return jnp.where(cdf < eps, 0., cdf), jnp.where(complement < eps, 0., complement)


def repartition(sulfate, number, dry_radius):
    """Native simultaneous upward transfer between four soluble lognormal modes.

    Sulfate: molecules/cm3 (...,4); number: cm-3 (...,7); dry count median
    radii: cm (...,4). Empty insoluble modes are retained, never populated.
    """
    sulfate, number, dry_radius = map(jnp.asarray, (sulfate, number, dry_radius))
    eps = jnp.finfo(jnp.float64).eps
    fractions_n = jnp.ones_like(sulfate)
    fractions_m = jnp.ones_like(sulfate)
    for mode in range(3):
        lower, upper = MODE_RADIUS_BOUNDS[mode:mode+2]
        average_radius = jnp.exp((jnp.log(lower)+jnp.log(upper))/2)*CMR2RAM[mode]
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
        cdf, _ = cumulative_normal(z)
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
