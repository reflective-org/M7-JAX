# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Check the original-source reference harness independently of the JAX port."""
import hashlib
import numpy as np
import pytest
from tools.m7_oracle import SOURCES, SOURCE, OPERATIONS, build_oracle, reference_cases, run_oracle


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    return [build_oracle(tmp_path_factory.mktemp(f'm7-reference-{checked}'), checked=checked)
            for checked in (True, False)]


def test_provenance_and_visibility_overlay(references):
    for executable, provenance in references:
        assert provenance['source_sha256'] == {
            name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in SOURCES}
        assert provenance['registry']['components'] == 18
        source = (executable.parent / 'mo_ham_m7.F90').read_text()
        original = (executable.parent / 'mo_ham_m7.original.F90').read_text()
        overlay = source[len(original.split('CONTAINS', 1)[0]):source.index('CONTAINS')]
        assert overlay.strip().startswith('PUBLIC ::')
        assert source.replace(overlay, '', 1) == original
        assert ('-fcheck=all' in provenance['flags']) is provenance['checked']


@pytest.mark.parametrize('operation', OPERATIONS)
def test_checked_optimized_finite_and_repeatable(references, operation):
    cases = reference_cases()
    checked, timing = run_oracle(references[0][0], cases, operation=operation)
    optimized, _ = run_oracle(references[1][0], cases, operation=operation)
    repeated, repeated_timing = run_oracle(references[0][0], cases, operation=operation, repetitions=3)
    for key, value in checked.items():
        assert np.isfinite(value).all(), (operation, key)
        np.testing.assert_array_equal(value, optimized[key])
        np.testing.assert_array_equal(value, repeated[key])
    assert timing['native_cpu_seconds'] >= 0
    assert repeated_timing['repetitions'] == 3


@pytest.mark.parametrize('water', [0, 1])
def test_full_original_call_closes_sulfate_and_preserves_empty_modes(references, water):
    cases = reference_cases()
    cases['water'] = np.full(3, water)
    out, _ = run_oracle(references[0][0], cases, operation='full')
    before = cases['sulfate'].sum(-1) + cases['gas'] + cases['production'] * cases['timestep']
    after = out['sulfate'].sum(-1) + out['gas']
    np.testing.assert_allclose(after, before, rtol=5e-15, atol=0)
    np.testing.assert_array_equal(out['number'][:, 4:], 0)
    np.testing.assert_array_equal(out['mass'][:, 4:], 0)
    assert np.all(out['water'] >= 0)
    assert out['number'][2].sum() > cases['number'][2].sum()


def test_kazil_and_uninitialized_native_combination_are_rejected(references):
    cases = reference_cases()
    cases['scheme'] = np.full(3, 2)
    with pytest.raises(ValueError, match='Kazil'):
        run_oracle(references[0][0], cases)
    cases['scheme'] = np.ones(3)
    cases['condensation'] = np.zeros(3)
    with pytest.raises(ValueError, match='uninitialized'):
        run_oracle(references[0][0], cases)
