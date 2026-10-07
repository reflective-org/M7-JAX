"""Multi-species average particle properties and kappa hydration parity.

Covers constants.COMPONENT_SPECIES/COMPONENT_MODE's full 18-component table:
each species alone in every mode it can occupy, internal mixtures (KS with
BC+OC; AS and CS with SO4+BC+OC+SS+DU), empty modes, and values straddling
cmin_aernl/cmin_aerml/cminvol. T/RH is pushed to and beyond the kappa table's
edges (233-313 K, 0.3-0.9878263 RH) for the hydration parity check.
"""
from pathlib import Path

import jax
import numpy as np
import pytest

from m7_jax.constants import (CMIN_AERNL, CMIN_AERML, CMINVOL,
    COMPONENT_MODE, COMPONENT_SPECIES)
from m7_jax.properties import (average_properties, kappa_hydration,
    load_kappa_table, require_sulfate_only_for_jacobson, AverageProperties)
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    return build_oracle(tmp_path_factory.mktemp('m7-properties-multispecies'))[0]


@pytest.fixture(scope='module')
def table():
    path = ROOT/'reference/input/lut_kappa.nc'
    return load_kappa_table(path if path.exists() else ROOT/'lut_kappa.nc')


def _component_alone_cases():
    """One row per of the 18 components, that component alone nonzero with
    its mode's number populated; plus rows exercising cmin_aernl/cmin_aerml/
    cminvol on both sides, internal mixtures, an empty-everything row, and a
    row with number present but zero mass (and vice versa) per component.
    """
    rows_mass, rows_number = [], []

    def add(mass_by_component, number_by_mode):
        mass = np.zeros(18)
        for idx, value in mass_by_component.items():
            mass[idx] = value
        number = np.zeros(7)
        for idx, value in number_by_mode.items():
            number[idx] = value
        rows_mass.append(mass)
        rows_number.append(number)

    # Each component alone, active (above both thresholds).
    for jn in range(18):
        mode = COMPONENT_MODE[jn]
        add({jn: 1e8 if COMPONENT_SPECIES[jn] == 0 else 5.0}, {mode: 1e3})

    # Each component alone, but inactive: below cmin_aernl (mode-1 scenario)
    # or below cmin_aerml (component value), one row each, for a soluble and
    # an insoluble component.
    add({1: 1e8}, {1: CMIN_AERNL*0.5})          # SO4 KS, number below threshold
    add({1: CMIN_AERML*0.5}, {1: 1e3})          # SO4 KS, mass below threshold
    add({7: 5.0}, {4: CMIN_AERNL*0.5})          # BC KI, number below threshold
    add({7: CMIN_AERML*0.5}, {4: 1e3})          # BC KI, mass below threshold
    # Straddle exactly at the threshold value (native uses a strict ">").
    add({2: CMIN_AERML}, {2: 1e3})              # SO4 AS mass == cmin_aerml exactly
    add({2: 1e8}, {2: CMIN_AERNL})              # SO4 AS number == cmin_aernl exactly

    # KS with BC+OC, mixed.
    add({1: 1e8, 4: 1.0, 8: 2.0}, {1: 1e3})
    # AS fully internally mixed: SO4+BC+OC+SS+DU.
    add({2: 1e9, 5: 1.0, 9: 2.0, 12: 3.0, 14: 4.0}, {2: 1e3})
    # CS fully internally mixed: SO4+BC+OC+SS+DU.
    add({3: 1e8, 6: 0.5, 10: 0.6, 13: 0.7, 15: 0.8}, {3: 1e2})
    # Insoluble mode (KI) total volume straddling cminvol: BC+OC at density
    # 2 g/cm3, mass [ug m-3] -> volume [cm3] via fac_ug(1e-12)/density, so
    # 2 components at `above` each sum to 2*above*1e-12/2 = 2e-23 > cminvol
    # and at `below` sum to well under it.
    above = CMINVOL*1e12  # ug m-3 per component giving combined volume 2x cminvol
    add({7: above, 11: above}, {4: 1e2})
    add({7: above*1e-10, 11: above*1e-10}, {4: 1e2})  # combined volume far below cminvol
    # All modes with insoluble components populated together (AI, CI).
    add({16: 5.0}, {5: 10.0})
    add({17: 20.0}, {6: 1.0})
    # Completely empty state.
    add({}, {})

    return np.stack(rows_mass), np.stack(rows_number)


def _relerr(value, expected):
    value, expected = np.asarray(value), np.asarray(expected)
    denominator = np.maximum(np.abs(expected), 1e-300)
    return np.max(np.abs(value-expected)/denominator)


@pytest.fixture(scope='module')
def designed_cases():
    mass, number = _component_alone_cases()
    previous_radius = np.full((mass.shape[0], 7), 1e-6)
    previous_density = np.full((mass.shape[0], 7), 1.841)
    return dict(mass=mass, number=number, radius=previous_radius.copy(), density=previous_density.copy())


def test_average_properties_against_native(oracle, designed_cases):
    native, _ = run_oracle(oracle, designed_cases, operation='properties')
    previous = AverageProperties(None, designed_cases['radius'], designed_cases['density'])
    errors = {}
    for function in (average_properties, jax.jit(average_properties)):
        result = function(designed_cases['mass'], designed_cases['number'], previous=previous)
        errors['pttn'] = max(errors.get('pttn', 0.), _relerr(result.pttn, native['mean_particle_mass']))
        errors['insoluble_radius'] = max(errors.get('insoluble_radius', 0.),
            _relerr(result.insoluble_radius, native['radius']))
        errors['insoluble_density'] = max(errors.get('insoluble_density', 0.),
            _relerr(result.insoluble_density, native['density']))
        np.testing.assert_allclose(result.pttn, native['mean_particle_mass'], rtol=1e-12, atol=1e-30)
        np.testing.assert_allclose(result.insoluble_radius, native['radius'], rtol=1e-12, atol=1e-30)
        np.testing.assert_allclose(result.insoluble_density, native['density'], rtol=1e-12, atol=1e-30)
        # Soluble-mode geometry entries are untouched, exactly like the native INOUT args.
        np.testing.assert_array_equal(result.insoluble_radius[..., :4], designed_cases['radius'][..., :4])
        np.testing.assert_array_equal(result.insoluble_density[..., :4], designed_cases['density'][..., :4])
    print('average_properties max relative error per field:', errors)


def test_average_properties_default_previous_is_zero():
    mass = np.zeros((2, 18))
    number = np.zeros((2, 7))
    result = average_properties(mass, number)
    np.testing.assert_array_equal(result.insoluble_radius, 0.)
    np.testing.assert_array_equal(result.insoluble_density, 0.)


@pytest.mark.parametrize('temperature,relative_humidity', [
    (280., .5), (233., .3), (313., .9878263), (200., .3), (350., .9878263),
    (233., 0.), (313., 1.),
])
def test_kappa_hydration_against_native(oracle, table, designed_cases, temperature, relative_humidity):
    cases = dict(designed_cases, temperature=temperature, relative_humidity=relative_humidity,
                 dry_radius=np.zeros((designed_cases['mass'].shape[0], 4)), water=1)
    native, _ = run_oracle(oracle, cases, operation='hydrate')
    previous = AverageProperties(None, cases['radius'], cases['density'])
    avg = average_properties(cases['mass'], cases['number'], previous=previous)
    errors = {}
    for function in (kappa_hydration, jax.jit(kappa_hydration)):
        result = function(avg.pttn, cases['number'], temperature, relative_humidity, table, previous=avg)
        errors['dry_radius'] = max(errors.get('dry_radius', 0.), _relerr(result.dry_radius, native['dry_radius']))
        errors['wet_radius'] = max(errors.get('wet_radius', 0.), _relerr(result.wet_radius, native['radius']))
        errors['density'] = max(errors.get('density', 0.), _relerr(result.density, native['density']))
        errors['water'] = max(errors.get('water', 0.), _relerr(result.water, native['water'][:, :4]))
        np.testing.assert_allclose(result.dry_radius, native['dry_radius'], rtol=1e-12, atol=1e-30)
        np.testing.assert_allclose(result.wet_radius, native['radius'], rtol=1e-12, atol=1e-30)
        np.testing.assert_allclose(result.density, native['density'], rtol=1e-12, atol=1e-30)
        np.testing.assert_allclose(result.water, native['water'][:, :4], rtol=1e-12, atol=1e-30)
    print(f'kappa_hydration T={temperature} RH={relative_humidity} max relative error per field:', errors)


def test_require_sulfate_only_for_jacobson_rejects_nonzero_non_sulfate():
    mass = np.zeros((2, 18))
    mass[:, :4] = 1.
    require_sulfate_only_for_jacobson(mass, 0)   # pure sulfate: water_scheme=0 stays supported
    require_sulfate_only_for_jacobson(mass, 1)   # kappa path: never restricted by species
    mixed = mass.copy()
    mixed[0, 7] = 1.
    require_sulfate_only_for_jacobson(mixed, 1)  # kappa path with non-sulfate mass: fine
    with pytest.raises(NotImplementedError, match='m7_equiz'):
        require_sulfate_only_for_jacobson(mixed, 0)
