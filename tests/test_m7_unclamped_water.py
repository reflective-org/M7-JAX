"""Opt-in unclamped kappa water: properties(..., unclamped_water=True).

Default (False) is the native lookup, whose RH index is clamped at the table's
first coordinate (0.3) and temperature at 233 K. With the flag on, states
outside the table range get the kappa-Koehler growth solved at the actual RH
and T; states inside the range keep the native lookup bit for bit.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import netcdf_file
from scipy.optimize import brentq

from m7_jax.constants import AVOGADRO, SO4_MOLAR_MASS, SULFATE_DENSITY, SULFATE_KAPPA
from m7_jax.model import initialize, step, sulfur_inventory, STATIC_OPTIONS
from m7_jax.properties import kappa_kohler_growth, load_kappa_table, properties

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT/'reference/input/lut_kappa.nc'


@pytest.fixture(scope='module')
def table():
    return load_kappa_table(TABLE_PATH)


@pytest.fixture(scope='module')
def state():
    # Empty-but-virtual nucleation mode, a 25 nm Aitken seed, accumulation and coarse.
    return initialize([0., 100., 10., 1e-3], [3., 25., 200., 2000.], 0.)


def reference_growth(radius_m, temperature, humidity, kappa):
    sigma = .0761-1.55e-4*(temperature-273.15)
    kelvin = 2*sigma*.018015/(8.314*temperature*1000.)
    def mismatch(g):
        return np.log((g**3-1)/(g**3-1+kappa))+kelvin/(radius_m*g)-np.log(humidity)
    return brentq(mismatch, 1+1e-15, 50., xtol=1e-15, rtol=1e-15)


@pytest.mark.parametrize('radius_m,temperature,humidity', [
    (5e-10, 200., .01), (1e-9, 220., .05), (1.25e-8, 220., .05), (1e-7, 250., .2),
    (1e-6, 220., .95), (5e-6, 313., .98)])
def test_growth_solves_kappa_kohler(radius_m, temperature, humidity):
    for function in (kappa_kohler_growth, jax.jit(kappa_kohler_growth)):
        g = float(function(radius_m, temperature, humidity, SULFATE_KAPPA))
        assert g == pytest.approx(reference_growth(radius_m, temperature, humidity, SULFATE_KAPPA), rel=2e-15)


def test_growth_dry_limits():
    assert float(kappa_kohler_growth(1e-8, 220., 0., .6)) == 1.
    assert float(kappa_kohler_growth(1e-8, 220., .05, 0.)) == 1.


def test_growth_form_is_the_table_form():
    """The entry stored at RH node i is the kappa-Koehler growth at RH coordinate i+1."""
    with netcdf_file(TABLE_PATH, mmap=False) as data:
        growth = np.array(data.variables['GF'].data, dtype=np.float64)
        radius, temperature, humidity, kappa = (np.array(data.variables[k].data, dtype=np.float64) for k in
            ('DryRadius', 'Temperature', 'RelativeHumidity', 'kappa'))
    nodes = np.append(humidity, humidity[-1]+.96*(humidity[-1]-humidity[-2]))
    i, j, h, k = np.meshgrid(np.arange(0, 98, 7), np.arange(0, 17, 4), np.arange(0, 61, 6),
                             np.array([0, 14, 28, 54]), indexing='ij')
    solved = kappa_kohler_growth(radius[i], temperature[j], nodes[h+1], kappa[k])
    error = np.asarray(solved)/growth[i, j, h, k]-1
    assert np.abs(error).max() < 6e-3
    assert np.median(np.abs(error)) < 1e-3


@pytest.mark.parametrize('temperature,humidity', [(250., .5), (233., .3), (313., .9878263)])
def test_flag_inside_table_is_native(table, state, temperature, humidity):
    native = properties(state.sulfate, state.number, temperature, humidity, table)
    fixed = properties(state.sulfate, state.number, temperature, humidity, table, unclamped_water=True)
    for a, b in zip(native, fixed):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize('temperature,humidity', [(220., .05), (250., .05), (220., .5)])
def test_flag_outside_table_uses_actual_conditions(table, state, temperature, humidity):
    call = lambda s, n: properties(s, n, temperature, humidity, table, unclamped_water=True)
    for function in (call, jax.jit(call)):
        p = function(state.sulfate, state.number)
        dry = np.asarray(p.dry_radius)
        wet = np.asarray(p.wet_radius)[:4]
        # The native volume-weighted kappa, kappa V/(V + 1e-25 cm3), as the lookup uses it.
        volume = SO4_MOLAR_MASS/AVOGADRO*np.asarray(p.mean_sulfate)/SULFATE_DENSITY
        kappa = SULFATE_KAPPA*volume/(volume+1e-25)
        expected = [reference_growth(1e-2*r, temperature, humidity, k) for r, k in zip(dry, kappa)]
        np.testing.assert_allclose(wet/dry, expected, rtol=1e-12)
    native = properties(state.sulfate, state.number, temperature, humidity, table)
    if humidity < .3:
        # Occupied modes only; the empty nucleation mode carries no water either way.
        assert np.all(np.asarray(p.water)[1:4] < np.asarray(native.water)[1:4])
    # Water mass follows the wet volume (water at 1 g/cm3, as in the native branch).
    np.testing.assert_array_equal(np.asarray(p.dry_radius), np.asarray(native.dry_radius))


def test_aitken_seed_at_paper_conditions(table, state):
    """25 nm count-median seed at 220 K, RH 0.05: native gives RH ~0.33 water, fix gives RH 0.05."""
    native = properties(state.sulfate, state.number, 220., .05, table)
    fixed = properties(state.sulfate, state.number, 220., .05, table, unclamped_water=True)
    ratio = lambda p: float(p.wet_radius[1]/p.dry_radius[1])
    assert ratio(native) == pytest.approx(1.0782, abs=1e-4)
    assert ratio(fixed) == pytest.approx(1.00908, abs=1e-5)


def test_step_flag_off_is_native_and_on_conserves(table, state):
    assert 'unclamped_water' in STATIC_OPTIONS
    compiled = jax.jit(step, static_argnames=STATIC_OPTIONS)
    gassy = state._replace(gas=jnp.asarray(1e8))
    args = (gassy, 220., 5000., .05, 5.)
    options = dict(table=table, coagulation=True, condensation=True, nucleation_scheme=0, water_scheme=1)
    default = compiled(*args, **options)
    off = compiled(*args, **options, unclamped_water=False)
    for a, b in zip(jax.tree_util.tree_leaves(default), jax.tree_util.tree_leaves(off)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    on = compiled(*args, **options, unclamped_water=True)
    assert bool(on.valid)
    np.testing.assert_allclose(sulfur_inventory(on.state), sulfur_inventory(gassy), rtol=1e-14)
    # Drier particles are smaller, so they take up less vapor in the step.
    assert float(on.state.gas) > float(default.state.gas)
