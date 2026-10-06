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


@pytest.mark.parametrize('operation', [op for op in OPERATIONS if op != 'kazil_lovejoy'])
def test_checked_optimized_finite_and_repeatable(references, operation):
    # 'kazil_lovejoy' is excluded here: it needs an oracle built WITH a real
    # table (build_oracle(..., kazil_table_path=...)), unlike every other
    # operation, which the default `references` fixture's table-less oracle
    # still stubs with 'error stop'. Its own data-gated coverage lives in
    # test_m7_nucleation.py.
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


def test_insoluble_radius_follows_mean_particle_volume(references):
    """The r7492 correction: insoluble radii come from mean volume, not density."""
    cases = reference_cases()
    mass = np.zeros((3, 18))
    mass[:, :4] = cases['sulfate']
    mass[:, 7] = 1.0    # BC in KI [ug m-3]
    mass[:, 11] = 2.0   # OC in KI
    mass[:, 16] = 5.0   # DU in AI
    mass[:, 17] = 20.0  # DU in CI
    cases['mass'] = mass
    cases['number'] = cases['number'].copy()
    cases['number'][:, 4:] = [100., 1., .1]
    out, _ = run_oracle(references[0][0], cases, operation='properties')
    # fac_ug (mo_ham_m7.F90) converts ug m-3 / cm-3 to g per particle (1e-12).
    density = {4: (1. + 2.) / (1. / 2. + 2. / 2.), 5: 2.65, 6: 2.65}
    grams = {4: 3e-12 / 100., 5: 5e-12 / 1., 6: 20e-12 / .1}
    sigma = {4: 1.59, 5: 1.59, 6: 2.}
    for mode in (4, 5, 6):
        volume = grams[mode] / density[mode]
        expected = np.exp(-1.5 * np.log(sigma[mode])**2) * (volume / (4 * np.pi / 3))**(1 / 3)
        np.testing.assert_allclose(out['density'][:, mode], density[mode], rtol=1e-14)
        np.testing.assert_allclose(out['radius'][:, mode], expected, rtol=1e-14)
