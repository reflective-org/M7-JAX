# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Per-call native observations must preserve the original full-call behavior."""
import numpy as np
import pytest

from tools.m7_oracle import reference_cases, run_oracle
from tools.m7_trajectory import build_trajectory, run_trajectory

CORE_FIELDS = ('number', 'sulfate', 'gas', 'radius', 'density', 'dry_radius', 'water', 'mass')


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    return build_trajectory(tmp_path_factory.mktemp('m7-trajectory'))


@pytest.mark.parametrize('steps', [1, 10])
@pytest.mark.parametrize('water', [0, 1])
def test_checkpoints_match_unobserved_native_final_and_budgets(native, steps, water):
    executable, provenance = native
    cases = reference_cases()
    cases['water'] = water
    trajectory, timing = run_trajectory(executable, cases, steps=steps)
    final, _ = run_oracle(executable.parent / 'm7_oracle', cases, steps=steps)
    for field in CORE_FIELDS:
        assert trajectory[field].shape == (steps,) + final[field].shape
        assert np.isfinite(trajectory[field]).all()
        np.testing.assert_array_equal(trajectory[field][-1], final[field])
    before = cases['sulfate'].sum(-1) + cases['gas']
    expected_budget = before + np.arange(1, steps + 1)[:, None] * cases['production'] * cases['timestep']
    actual_budget = trajectory['sulfate'].sum(-1) + trajectory['gas']
    np.testing.assert_allclose(actual_budget, expected_budget, rtol=1e-14, atol=0)
    assert timing['steps'] == steps
    assert provenance['checked']
    assert 'No process source changes' in provenance['observation']


@pytest.mark.parametrize('water', [0, 1])
def test_every_checkpoint_matches_manual_split_replay(native, water):
    executable, _ = native
    cases = reference_cases()
    cases['water'] = water
    trajectory, _ = run_trajectory(executable, cases, steps=3)
    for step in range(3):
        single, _ = run_oracle(executable.parent / 'm7_oracle', cases)
        for field in CORE_FIELDS:
            np.testing.assert_array_equal(trajectory[field][step], single[field])
        for field in ('number', 'sulfate', 'gas', 'radius', 'density', 'dry_radius'):
            cases[field] = single[field]


@pytest.mark.parametrize('steps', [0, -1, True, 1.5])
def test_nonpositive_or_noninteger_steps_rejected(steps):
    with pytest.raises(ValueError, match='positive integer'):
        run_trajectory('/unused', reference_cases(), steps=steps)
