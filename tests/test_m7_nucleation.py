# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Native M7 binary nucleation has independent coefficients and integration."""
from functools import partial

import jax
import numpy as np
import pytest

from m7_jax.nucleation import nucleate, vehkamaeki
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    return build_oracle(tmp_path_factory.mktemp('m7-nucleation'))[0]


def rate_cases():
    gas, temperature, humidity = np.meshgrid(
        [0., 1e4, 1e6, 1e8, 1e10, 1e12],
        [180., 190.15, 230., 270., 300.15, 320.],
        [0., 1e-4, .1, .8, 1., 1.2], indexing='ij')
    return {'gas': gas.ravel(), 'temperature': temperature.ravel(),
            'relative_humidity': humidity.ravel()}


@pytest.mark.parametrize('compiled', [False, True])
def test_binary_rate_and_critical_cluster_match_actual_native(oracle, compiled):
    cases = rate_cases()
    expected, _ = run_oracle(oracle, cases, operation='vehkamaeki')
    kernel = jax.jit(vehkamaeki) if compiled else vehkamaeki
    actual = kernel(cases['temperature'], cases['relative_humidity'], cases['gas'])
    # Exp amplifies cancellation in the fitted log-polynomial. This is the
    # established fitted-polynomial tolerance; native structural zeros are exact.
    for key in actual._fields:
        np.testing.assert_allclose(getattr(actual, key), expected[key], rtol=1e-11, atol=0)
    assert np.any(np.asarray(actual.rate) == 0.)
    assert np.all(np.asarray(actual.rate) <= 1e10)
    assert np.all(np.isfinite(np.stack(actual)))


def process_cases():
    return {'gas': np.array([0., 1e4, 1e8, 1e8, 1e8, 1e8, 1e10, 1e8]),
            'temperature': np.array([280., 300., 250., 250., 250., 250., 270., 280.]),
            'relative_humidity': np.full(8, .5),
            'total_sink': np.array([0., .1, 0., .1, .01, .1, 1., .2]),
            'cloud_fraction': np.array([0., 0., 0., .5, 1., 1.2, -.1, .2]),
            'timestep': np.array([1., 1., 1., 1., 10., 10., .1, 60.]),
            'scheme': np.ones(8, dtype=int)}


@pytest.mark.parametrize('compiled', [False, True])
def test_nucleation_integration_matches_actual_native_and_closes_sulfur(oracle, compiled):
    cases = process_cases()
    expected, _ = run_oracle(oracle, cases, operation='nucleate')
    kernel = partial(nucleate, scheme=1)
    if compiled:
        kernel = jax.jit(kernel)
    actual = kernel(*(cases[key] for key in ('gas', 'temperature', 'relative_humidity',
                                            'total_sink', 'cloud_fraction', 'timestep')))
    for value, reference in ((actual.gas, expected['gas']),
                             (actual.number, expected['nucleated_number']),
                             (actual.sulfate, expected['sulfate_delta'][..., 0])):
        np.testing.assert_allclose(value, reference, rtol=1e-11, atol=0)
    np.testing.assert_array_equal(expected['sulfate_delta'][..., 1:], 0.)
    np.testing.assert_allclose(actual.gas + actual.sulfate, cases['gas'], rtol=3e-15, atol=0)
    assert np.all(np.asarray(actual.number) >= 0.)
    assert np.all(np.asarray(actual.gas) >= 0.)


def test_rate_clamps_and_multiaxis_inputs():
    raw = vehkamaeki(np.array([[180.], [320.]]), np.array([0., 1.2]), 1e12)
    bounded = vehkamaeki(np.array([[190.15], [300.15]]), np.array([1e-4, 1.]), 1e11)
    for left, right in zip(raw, bounded):
        np.testing.assert_array_equal(left, right)
    state = jax.jit(nucleate)(np.full((2, 3), 1e8), 250., .5, .1, 0., 1.)
    assert state.gas.shape == state.number.shape == state.sulfate.shape == (2, 3)


def test_disabled_and_inactive_rates_preserve_native_inout_deltas():
    off = jax.jit(partial(nucleate, scheme=0))(1e8, 250., .5, .1, 0., 1.,
                                           initial_number=7., initial_sulfate=11.)
    np.testing.assert_array_equal(off, (1e8, 7., 11.))
    inactive = nucleate(0., 300., .5, .1, 0., 1., initial_number=7., initial_sulfate=11.)
    np.testing.assert_array_equal(inactive, (0., 7., 11.))


def test_kazil_external_table_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match='external lookup table'):
        nucleate(1e8, 280., .5, .1, 0., 1., scheme=2)
    with pytest.raises(ValueError, match='schemes are 0 and 1'):
        nucleate(1e8, 280., .5, .1, 0., 1., scheme=3)
