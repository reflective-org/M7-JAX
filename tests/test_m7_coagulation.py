"""Pure-sulfate M7 coagulation parity against the actual Fortran module."""

from functools import partial

import jax
import numpy as np
import pytest

from m7_jax.coagulation import coaset, dnum
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    executable, _ = build_oracle(tmp_path_factory.mktemp("m7-coagulation"), checked=True)
    return executable


def cases():
    number = np.array([
        [2e4, 300, 30, 0.2, 0, 0, 0],
        [2e4, 0, 0, 0, 0, 0, 0],
        [0, 2e4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1e4, 2e3, 300, 3, 0, 0, 0],
        [1e4, 0, 0, 0, 0, 0, 0],
        [1e-10, 20, 3, 0, 0, 0, 0],
        [10, 2, 0.3, 0.01, 0, 0, 0],
    ], dtype=float)
    radius = np.tile([2e-7, 2e-6, 1e-5, 1e-4, 0, 0, 0], (8, 1))
    density = np.tile([1.841, 1.7, 1.4, 1.2, 0, 0, 0], (8, 1))
    sulfate = number[:, :4] * (4 * np.pi / 3) * radius[:, :4]**3 * density[:, :4] / (98.08 / 6.022e23)
    formed = np.array([0, 0, 0, 0, 30000, 30000, 0, 1.0])
    delta = np.zeros((8, 4))
    delta[:, 0] = formed * 30
    return dict(number=number, radius=radius, density=density,
                temperature=np.linspace(230, 310, 8), pressure=np.linspace(3e4, 101325, 8),
                timestep=np.array([600, 600, 600, 600, 600, 600, 600, 1200.]),
                sulfate=sulfate, nucleated_number=formed, sulfate_delta=delta)


def invoke(data, *, coagulation=True, compiled=False):
    function = partial(dnum, coagulation=coagulation)
    if compiled:
        function = jax.jit(function)
    return function(*(data[name] for name in (
        "sulfate", "number", "temperature", "pressure", "radius", "density",
        "nucleated_number", "sulfate_delta", "timestep")))


def test_coaset_native_all_pairs_and_thresholds(native):
    data = cases()
    # Collision kernel supports all seven modes, independently of the restricted
    # pure-sulfate time-integrator contract.
    data["number"][:, 4:] = [50, 3, 0.1]
    data["radius"][:, 4:] = [2e-6, 1e-5, 1e-4]
    data["density"][:, 4:] = [1.8, 2.6, 2.6]
    data["radius"][1, 0] = 1e-8
    data["density"][2, 1] = 1e-10
    expected, _ = run_oracle(native, data, operation="coaset")
    args = [data[name] for name in ("number", "temperature", "pressure", "radius", "density")]
    eager = np.asarray(coaset(*args))
    compiled = np.asarray(jax.jit(coaset)(*args))
    np.testing.assert_allclose(eager, expected["coefficients"], rtol=3e-13, atol=0)
    np.testing.assert_allclose(compiled, expected["coefficients"], rtol=3e-13, atol=0)
    assert np.isfinite(eager).all() and (eager >= 0).all()
    np.testing.assert_array_equal(eager[1, :7], 0)
    np.testing.assert_array_equal(eager[2, 7:13], 0)
    scalar = coaset(*(a[0] for a in args))
    np.testing.assert_allclose(scalar, expected["coefficients"][0], rtol=3e-13, atol=0)


@pytest.mark.parametrize("coagulation", [True, False])
def test_dnum_native_ordered_transfers_and_nucleation(native, coagulation):
    data = cases()
    data["coagulation"] = np.full(8, coagulation)
    expected, _ = run_oracle(native, data, operation="dnum")
    for compiled in (False, True):
        outputs = invoke(data, coagulation=coagulation, compiled=compiled)
        for result, field in zip(outputs, ("sulfate", "number", "sulfate_delta")):
            np.testing.assert_allclose(result, expected[field], rtol=3e-12, atol=1e-20)
            assert np.isfinite(result).all()
        mass, number, _ = map(np.asarray, outputs)
        np.testing.assert_allclose(mass.sum(axis=-1), data["sulfate"].sum(axis=-1) + data["sulfate_delta"][:, 0],
                                   rtol=3e-14, atol=1e-20)
        assert (mass >= 0).all() and (number >= 0).all()
        assert np.all(number.sum(axis=-1) <= (data["number"].sum(axis=-1) + data["nucleated_number"]) * (1 + 1e-14))
        np.testing.assert_array_equal(number[:, 4:], 0)
    # Multiple leading cell dimensions are independent; mode axis stays last.
    grid = {key: value.reshape((2, 4) + value.shape[1:]) for key, value in data.items()}
    output_grid = invoke(grid, coagulation=coagulation, compiled=True)
    for result, field in zip(output_grid, ("sulfate", "number", "sulfate_delta")):
        np.testing.assert_allclose(np.asarray(result).reshape(expected[field].shape), expected[field], rtol=3e-12, atol=1e-20)


def test_invalid_mode_shape_rejected_before_jax_index_clipping():
    with pytest.raises(ValueError, match="seven"):
        coaset(np.ones(4), 280., 1e5, np.ones(4), np.ones(4))
    data = {name: value[0] for name, value in cases().items()}
    data["sulfate"] = np.ones(7)
    data["sulfate_delta"] = np.zeros(7)
    with pytest.raises(ValueError, match="dimension 4"):
        invoke(data, coagulation=False)
