"""Round-trip checks for the host (mo_ham_subm.f90::ham_subm_interface) unit
conversions. mo_ham_subm.f90 is not in this checkout, so there is no native
executable to compare against here (see interface.py's module docstring);
these are round-trip and literal-value checks of the #1017 lead's specified
formulas instead.
"""
import jax
import numpy as np
import pytest

from m7_jax.constants import SO4_MOLAR_MASS, HAM_SUBM_AVOGADRO
from m7_jax.interface import (so4_mixing_ratio_to_native, native_to_so4_mixing_ratio,
    mass_mixing_ratio_to_native, native_to_mass_mixing_ratio,
    number_mixing_ratio_to_native, native_to_number_mixing_ratio,
    aerosol_mass_to_native, native_to_aerosol_mass)

jax.config.update('jax_enable_x64', True)

AIR_DENSITY = np.array([0.1, 1.0, 1.225, 10., 100.])  # kg m-3, including edge-ish values


def test_ham_subm_avogadro_literal_matches_spec():
    # The lead's spec (TASK.md) is the literal 6.022e20, not the full-precision
    # Avogadro constant the M7 core itself uses (6.02214179e23).
    assert HAM_SUBM_AVOGADRO == 6.022e20


@pytest.mark.parametrize('molar_mass', [SO4_MOLAR_MASS, 98.0734])
def test_so4_mixing_ratio_round_trip(molar_mass):
    q = np.array([0., 1e-12, 1e-9, 1e-6, 5e-5])
    native = so4_mixing_ratio_to_native(q, AIR_DENSITY, molar_mass)
    np.testing.assert_allclose(native, q*AIR_DENSITY*HAM_SUBM_AVOGADRO/molar_mass, rtol=1e-14)
    back = native_to_so4_mixing_ratio(native, AIR_DENSITY, molar_mass)
    np.testing.assert_allclose(back, q, rtol=1e-12, atol=0)


def test_mass_mixing_ratio_round_trip():
    q = np.array([0., 1e-15, 1e-10, 1e-6, 1e-3])
    native = mass_mixing_ratio_to_native(q, AIR_DENSITY)
    np.testing.assert_allclose(native, q*AIR_DENSITY*1e9, rtol=1e-14)
    back = native_to_mass_mixing_ratio(native, AIR_DENSITY)
    np.testing.assert_allclose(back, q, rtol=1e-12, atol=0)


def test_number_mixing_ratio_round_trip():
    n = np.array([0., 1e3, 1e6, 1e9, 1e12])
    native = number_mixing_ratio_to_native(n, AIR_DENSITY)
    np.testing.assert_allclose(native, n*AIR_DENSITY*1e-6, rtol=1e-14)
    back = native_to_number_mixing_ratio(native, AIR_DENSITY)
    np.testing.assert_allclose(back, n, rtol=1e-12, atol=0)


def test_aerosol_mass_full_18_component_round_trip():
    rng = np.random.default_rng(0)
    q = rng.uniform(0., 1e-6, size=(5, 18))
    air_density = AIR_DENSITY[:, None]  # one air density per cell, broadcast over components
    native = aerosol_mass_to_native(q, air_density)
    # SO4 components (0:4) use the molecules cm-3 formula; the rest use ug m-3.
    np.testing.assert_allclose(native[..., :4], q[..., :4]*air_density*HAM_SUBM_AVOGADRO/SO4_MOLAR_MASS, rtol=1e-13)
    np.testing.assert_allclose(native[..., 4:], q[..., 4:]*air_density*1e9, rtol=1e-13)
    back = native_to_aerosol_mass(native, air_density)
    np.testing.assert_allclose(back, q, rtol=1e-11, atol=0)


def test_aerosol_mass_round_trip_custom_molar_mass():
    rng = np.random.default_rng(1)
    q = rng.uniform(0., 1e-6, size=(3, 18))
    native = aerosol_mass_to_native(q, 1.2, molar_mass=98.0734)
    back = native_to_aerosol_mass(native, 1.2, molar_mass=98.0734)
    np.testing.assert_allclose(back, q, rtol=1e-11, atol=0)


def test_interface_functions_are_jit_compatible():
    q4 = np.array([1e-9, 2e-9, 3e-9, 4e-9])
    q14 = np.array([1e-9]*14)
    so4_native = jax.jit(so4_mixing_ratio_to_native)(q4, 1.2)
    other_native = jax.jit(mass_mixing_ratio_to_native)(q14, 1.2)
    full_native = jax.jit(aerosol_mass_to_native)(np.concatenate([q4, q14]), 1.2)
    number_native = jax.jit(number_mixing_ratio_to_native)(np.array([1e3, 1e6]), 1.2)
    assert np.all(np.isfinite(number_native))
    np.testing.assert_allclose(full_native[:4], so4_native, rtol=1e-14)
    np.testing.assert_allclose(full_native[4:], other_native, rtol=1e-14)
