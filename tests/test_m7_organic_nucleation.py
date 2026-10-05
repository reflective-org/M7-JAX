"""Organic nucleation (``organic_activation``/``organic_kinetic``/``nucleate_all``)

parity against the actual native ``m7_nuck``, built with ``-DHAMMOZ`` so its
organic branch (compiled out of the standalone source otherwise) is live.
"""
import functools
from itertools import product

import jax
import numpy as np
import pytest

from m7_jax.nucleation import nucleate_all, organic_activation, organic_kinetic
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def hammoz(tmp_path_factory):
    return build_oracle(tmp_path_factory.mktemp("m7-organic-nucleation"), hammoz=True)[0]


def combined_cases():
    """Full cross-product: nsnucl x nonucl x forest x cloud x in_pbl, plus a

    populated/empty gas-phase H2SO4 pair.
    """
    nsnucl = (0, 1)
    nonucl = (1, 2)
    forest = (0.0, 0.3, 1.0)
    cloud = (0.0, 0.5, 1.0)
    in_pbl = (True, False)
    gas = (1e8, 0.0)
    rows = list(product(nsnucl, nonucl, forest, cloud, in_pbl, gas))
    n = len(rows)
    nsnucl_arr, nonucl_arr, forest_arr, cloud_arr, pbl_arr, gas_arr = (np.array(c) for c in zip(*rows))
    return dict(
        gas=gas_arr, temperature=np.full(n, 270.), relative_humidity=np.full(n, .6),
        total_sink=np.full(n, .05), cloud_fraction=cloud_arr, timestep=np.full(n, 3.),
        scheme=nsnucl_arr.astype(int), organic_scheme=nonucl_arr.astype(int),
        forest_fraction=forest_arr, pbl_level=np.where(pbl_arr, 1, 2)), pbl_arr


@pytest.mark.parametrize("compiled", [False, True])
def test_nucleate_all_native_full_cross_product(hammoz, compiled):
    cases, in_pbl = combined_cases()
    expected, _ = run_oracle(hammoz, cases, operation="nucleate")

    results = {}
    for scheme in (0, 1):
        for organic_scheme in (1, 2):
            mask = (cases["scheme"] == scheme) & (cases["organic_scheme"] == organic_scheme)
            fn = functools.partial(nucleate_all, scheme=scheme, organic_scheme=organic_scheme)
            if compiled:
                fn = jax.jit(fn)
            idx = np.where(mask)[0]
            out = fn(cases["gas"][idx], cases["temperature"][idx], cases["relative_humidity"][idx],
                     cases["total_sink"][idx], cases["cloud_fraction"][idx], cases["forest_fraction"][idx],
                     in_pbl[idx], cases["timestep"][idx])
            results[(scheme, organic_scheme)] = (idx, out)

    gas = np.empty_like(cases["gas"])
    number = np.empty_like(cases["gas"])
    sulfate = np.empty_like(cases["gas"])
    for (scheme, organic_scheme), (idx, out) in results.items():
        gas[idx] = np.asarray(out.gas)
        number[idx] = np.asarray(out.number)
        sulfate[idx] = np.asarray(out.sulfate)

    np.testing.assert_allclose(gas, expected["gas"], rtol=1e-11, atol=0)
    np.testing.assert_allclose(number, expected["nucleated_number"], rtol=1e-11, atol=0)
    np.testing.assert_allclose(sulfate, expected["sulfate_delta"][..., 0], rtol=1e-11, atol=0)
    np.testing.assert_array_equal(expected["sulfate_delta"][..., 1:], 0.)
    # Sulfur closes exactly: whatever leaves the gas phase becomes the new
    # particles' sulfate, nothing else.
    np.testing.assert_allclose(gas + sulfate, cases["gas"], rtol=3e-15, atol=0)
    assert np.all(number >= 0.) and np.all(gas >= 0.)


def test_organic_activation_and_kinetic_rate_forms():
    h2so4 = np.array([0., 1e6, 1e8, 1e10])
    forest = np.array([0., .3, .3, 1.])
    in_pbl = np.array([True, True, False, True])

    activation = organic_activation(h2so4, forest, in_pbl)
    np.testing.assert_allclose(np.asarray(activation.rate), np.where(in_pbl, 1e-6 * h2so4 * forest, 0.))
    np.testing.assert_array_equal(np.asarray(activation.cluster_sulfate), np.where(in_pbl, 1., 0.))

    kinetic = organic_kinetic(h2so4, forest, in_pbl)
    np.testing.assert_allclose(np.asarray(kinetic.rate), np.where(in_pbl, 3.86e-13 * h2so4**2 * forest, 0.))
    np.testing.assert_array_equal(np.asarray(kinetic.cluster_sulfate), np.where(in_pbl, 2., 0.))

    for fn in (organic_activation, organic_kinetic):
        jitted = jax.jit(fn)
        result = jitted(h2so4, forest, in_pbl)
        eager = fn(h2so4, forest, in_pbl)
        for a, b in zip(result, eager):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_nucleate_all_rejects_kazil_and_unsupported_organic_scheme():
    with pytest.raises(NotImplementedError, match="external lookup table"):
        nucleate_all(1e8, 280., .5, .1, 0., .3, True, 1., scheme=2)
    with pytest.raises(ValueError, match="organic nucleation schemes"):
        nucleate_all(1e8, 280., .5, .1, 0., .3, True, 1., organic_scheme=3)
