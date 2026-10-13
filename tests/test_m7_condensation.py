# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""M7 condensation independently compared with its own original Fortran."""
import jax
import numpy as np
import pytest

from m7_jax.condensation import condensation_sink, produce_and_condense
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    return build_oracle(tmp_path_factory.mktemp('m7-condensation'))[0]


def sink_cases():
    number = np.broadcast_to([10., 100., 80., 2., 30., 4., .5], (6, 7)).copy()
    radius = np.broadcast_to([1e-7, 2e-6, 1e-5, 8e-5, 3e-6, 2e-5, 9e-5], (6, 7)).copy()
    number[0] = 0.
    radius[1, ::2] = 0.
    radius[2, 1] = -1.
    number[3, 4:] = 0.
    return {'number': number, 'radius': radius,
            'temperature': np.array([280., 190., 310., 270., 300., 230.]),
            'pressure': np.array([101325., 20000., 90000., 105000., 85000., 50000.])}


@pytest.mark.parametrize('compiled', [False, True])
def test_modal_and_total_sinks_match_actual_native(oracle, compiled):
    cases = sink_cases()
    expected, _ = run_oracle(oracle, cases, operation='condensation_sink')
    kernel = jax.jit(condensation_sink) if compiled else condensation_sink
    actual = kernel(cases['number'], cases['radius'], cases['temperature'], cases['pressure'])
    np.testing.assert_allclose(actual.modal, expected['modal_sink'], rtol=3e-13, atol=0)
    np.testing.assert_allclose(actual.total, expected['total_sink'], rtol=3e-13, atol=0)
    np.testing.assert_array_equal(np.asarray(actual.modal)[cases['radius'] <= 0.], 0.)
    np.testing.assert_array_equal(actual.modal[0], 0.)


def condensation_cases():
    modal = np.broadcast_to(np.arange(1., 8.) * .001, (12, 7)).copy()
    # Exercise native threshold and every cloudy deposition priority/fallback.
    modal[0] = 0.
    modal[1] = 0.; modal[1, 0] = 1e-10
    for row, index in enumerate([3, 2, 1, 0, 6, 5, 4], start=2):
        modal[row] = 0.; modal[row, index] = .1
    return {'sulfate': np.broadcast_to([1e6, 2e6, 3e6, 4e6], (12, 4)).copy(),
            'gas': np.full(12, 1e8), 'production': np.arange(12.) * 1e5,
            'total_sink': modal.sum(-1), 'modal_sink': modal,
            'cloud_fraction': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 0., .4, -.2]),
            'timestep': np.linspace(.5, 10., 12)}


@pytest.mark.parametrize('compiled', [False, True])
def test_production_and_condensation_match_native_and_close_budget(oracle, compiled):
    cases = condensation_cases()
    expected, _ = run_oracle(oracle, cases, operation='produce_and_condense')
    kernel = jax.jit(produce_and_condense) if compiled else produce_and_condense
    actual = kernel(*(cases[key] for key in ('gas', 'production', 'sulfate', 'total_sink',
                                            'modal_sink', 'cloud_fraction', 'timestep')))
    for key in actual._fields:
        np.testing.assert_allclose(getattr(actual, key), expected[key], rtol=3e-13, atol=0)
    total_before = cases['gas'] + cases['sulfate'].sum(-1) + cases['production'] * cases['timestep']
    total_after = actual.gas + actual.sulfate.sum(-1) + actual.insoluble_sulfate.sum(-1)
    np.testing.assert_allclose(total_after, total_before, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(actual.sulfate[:2], cases['sulfate'][:2])
    np.testing.assert_array_equal(actual.insoluble_sulfate[:2], 0.)
    np.testing.assert_array_equal(actual.gas[2:9], 0.)


def test_scalar_multiaxis_broadcast_and_inputs_are_unchanged():
    number, radius = np.ones((2, 3, 7)), np.full((2, 3, 7), 1e-5)
    before = number.copy(), radius.copy()
    sink = jax.jit(condensation_sink)(number, radius, 280., 101325.)
    sulfate = np.zeros((2, 3, 4))
    result = jax.jit(produce_and_condense)(1e8, 1e5, sulfate, sink.total, sink.modal, 0., 1.)
    assert result.sulfate.shape == (2, 3, 4)
    assert result.gas.shape == (2, 3)
    np.testing.assert_array_equal(number, before[0])
    np.testing.assert_array_equal(radius, before[1])
    np.testing.assert_array_equal(sulfate, 0.)


@pytest.mark.parametrize('call', [
    lambda: condensation_sink(np.ones(4), np.ones(4), 280., 1e5),
    lambda: condensation_sink(np.ones((2, 7)), np.ones(7), 280., 1e5),
    lambda: produce_and_condense(1., 0., np.ones(7), .1, np.ones(7), 0., 1.),
])
def test_invalid_component_shapes_rejected(call):
    with pytest.raises(ValueError):
        call()
