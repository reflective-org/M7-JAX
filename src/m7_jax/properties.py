"""Native pure-sulfate mean particle properties and both M7 water options."""
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import netcdf_file

from .constants import (PI, AVOGADRO, SO4_MOLAR_MASS, H2SO4_MOLAR_MASS,
    SULFATE_DENSITY, SULFATE_KAPPA, CMIN_AERNL, CMIN_AERML, CMINRAD,
    CRITICAL_MOLECULES, RAM2CMR, CMR2RAM, WVB, GMB,
    KOHLER_SIGMA0, KOHLER_DSIGMA_DT, KOHLER_T0, KOHLER_WATER_MOLAR_MASS,
    KOHLER_GAS_CONSTANT, KOHLER_WATER_DENSITY, KOHLER_ITERATIONS)


class KappaTable(NamedTuple):
    growth: object
    radius: object
    temperature: object
    humidity: object
    kappa: object


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


def kappa_kohler_growth(dry_radius_m, temperature, humidity, kappa, iterations=KOHLER_ITERATIONS):
    """Wet/dry radius ratio g solving the kappa-Koehler equation at (T, RH).

    RH = a_w(g) exp(A/(r_d g)), a_w = (g^3 - 1)/(g^3 - 1 + kappa),
    A = 2 sigma_w(T) M_w/(R T rho_w), sigma_w(T) = 0.0761 - 1.55e-4 (T - 273.15) N/m.
    This is the form the version-2 lookup table holds: evaluated at the table's
    RH coordinate i+1, it reproduces the entries stored at RH node i within 0.5 %
    on every checked entry (median 0.07 %). Solved by bisection on
    1 < g <= (1 + kappa RH/(1 - RH))^(1/3), the Kelvin-free root, which bounds the
    stable-branch root from above for RH < 1. RH <= 0 gives g = 1 (no water).
    Units: radius m, temperature K, humidity fraction; returns a dimensionless ratio.
    """
    r = jnp.asarray(dry_radius_m)
    t = jnp.asarray(temperature)
    rh = jnp.asarray(humidity)
    k = jnp.maximum(jnp.asarray(kappa), 0.)
    wet = rh > 0
    rh_safe = jnp.where(wet, rh, .5)
    sigma = KOHLER_SIGMA0-KOHLER_DSIGMA_DT*(t-KOHLER_T0)
    kelvin = 2*sigma*KOHLER_WATER_MOLAR_MASS/(KOHLER_GAS_CONSTANT*t*KOHLER_WATER_DENSITY)

    def mismatch(g):
        g3 = g*g*g-1.
        return jnp.log(g3/(g3+k+1e-300))+kelvin/(r*g)-jnp.log(rh_safe)

    upper = (1.+k*rh_safe/(1.-rh_safe))**(1/3)
    shape = jnp.broadcast_shapes(r.shape, t.shape, rh.shape, k.shape)
    lower, upper = jnp.ones(shape), jnp.broadcast_to(upper, shape)

    def bisect(_, bounds):
        lo, hi = bounds
        mid = .5*(lo+hi)
        below = mismatch(mid) < 0
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, iterations, bisect, (lower, upper))
    return jnp.where(wet, .5*(lo+hi), 1.)


def properties(sulfate, number, temperature, humidity, table=None, *, water_scheme=1, pressure=101325., previous=None,
               unclamped_water=False):
    """Pure sulfate reduction of original averageproperties + kappa/equil chain.

    Registry SO4 molecular weight is used for kappa, native H2SO4 weight for
    Jacobson. This inherited distinction is deliberate. Empty first modes have
    a virtual critical cluster radius without adding physical sulfate.

    unclamped_water (opt-in fix, default False = native): with water_scheme=1,
    where RH is below the table's first RH coordinate (0.3) or T below its first
    temperature (233 K), the native lookup clamps the index and so returns the
    first table entry (the growth of RH 0.33, see kappa_kohler_growth) at 233 K. With the flag
    on, the growth factor there is instead solved from the table's own
    kappa-Koehler form at the actual T and RH (``kappa_kohler_growth``); the
    native lookup is kept unchanged inside the table range. RH above the table
    top is capped at the top node in the solve, as the native lookup does.
    It has no effect on water_scheme=0.
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
        kap = SO4_MOLAR_MASS/AVOGADRO*SULFATE_KAPPA*(mean/SULFATE_DENSITY)/(volume+1e-25)
        def nearest(x, axis):
            position = (len(axis)-1)*(jnp.clip(x, axis[0], axis[-1])-axis[0])/(axis[-1]-axis[0])
            return jnp.floor(position+.5).astype(jnp.int32)
        it = nearest(t, table.temperature)
        ik = nearest(kap, table.kappa)
        # Preserve the native default-real 1.E-2 promoted to dp.
        ir = nearest(jnp.log(float(np.float32(1e-2))*safe), jnp.log(table.radius))
        first_step = table.humidity[1]-table.humidity[0]
        compression = (table.humidity[2]-table.humidity[1])/first_step
        ih = jnp.ceil(jnp.log(1-(1-compression)*(rh-table.humidity[0])/first_step)/jnp.log(compression))
        ih = jnp.clip(ih, 1, len(table.humidity)).astype(jnp.int32)-1
        growth = table.growth[ir, it, ih, ik]
        if unclamped_water:
            outside = (rh < table.humidity[0]) | (t < table.temperature[0])
            solved = kappa_kohler_growth(1e-2*safe, t, jnp.minimum(rh, table.humidity[-1]), kap)
            growth = jnp.where(outside, solved, growth)
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
