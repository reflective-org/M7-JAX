"""Native M7 binary nucleation has independent coefficients and integration."""
import os
from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.io import netcdf_file

from m7_jax.nucleation import (
    kazil_lovejoy,
    load_kazil_lovejoy_table,
    nucleate,
    nucleate_all,
    vehkamaeki,
)
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update('jax_enable_x64', True)

# jax-gcm#1017 Kazil/GCR task, Part A: the real PARNUC table is MPI-M/HAMMOZ
# data, not shipped with this package. HAM_INPUT_DIR (also the GCR task's
# data location) names the directory; ECHAM itself opens the table as
# 'parnuc.15H2SO4.nc', but the staged file is 'parnuc.15H2SO4.A0.total.nc'
# (see docs/validation.md) -- both names are accepted, in that order.
_KAZIL_TABLE_NAMES = ('parnuc.15H2SO4.nc', 'parnuc.15H2SO4.A0.total.nc')


def _kazil_table_path():
    base = os.environ.get('HAM_INPUT_DIR')
    if not base:
        return None
    for name in _KAZIL_TABLE_NAMES:
        candidate = Path(base) / name
        if candidate.exists():
            return candidate
    return None


_kazil_skip = pytest.mark.skipif(
    _kazil_table_path() is None,
    reason=(
        "Kazil/Lovejoy PARNUC table not found: set HAM_INPUT_DIR to a "
        f"directory containing one of {_KAZIL_TABLE_NAMES} "
        "(docs/validation.md has the exact command)"
    ),
)


def _write_synthetic_format_test_table(path):
    """A tiny, clearly-fake table exercising only the reader's parsing/
    indexing (shapes, axis names, the 'H2SO4' lookup, finite/monotonic
    validation) -- NEVER used for a physics number. See the module naming
    rule in kazil_gcr_TASK.md: this is the one permitted stand-in.
    """
    t_n, rh_n, h_n, i_n, c_n = 3, 3, 3, 2, 2
    with netcdf_file(path, 'w') as nc:
        nc.createDimension('temperature', t_n)
        nc.createDimension('RH', rh_n)
        nc.createDimension('H2SO4', h_n)
        nc.createDimension('ionization', i_n)
        nc.createDimension('condensation_sink', c_n)
        for name, n, values in (
            ('temperature', t_n, [200., 250., 300.]),
            ('RH', rh_n, [1., 50., 100.]),
            ('H2SO4', h_n, [1e5, 1e7, 1e9]),
            ('ionization', i_n, [1., 20.]),
            ('condensation_sink', c_n, [1e-4, 1e-1]),
        ):
            var = nc.createVariable(name, 'd', (name,))
            var[:] = values
        pfr = nc.createVariable('pfr', 'f', ('temperature', 'RH', 'H2SO4', 'ionization', 'condensation_sink'))
        pfr[:] = np.linspace(-50., 10., t_n*rh_n*h_n*i_n*c_n).reshape(t_n, rh_n, h_n, i_n, c_n)


def test_load_kazil_lovejoy_table_synthetic_format(tmp_path):
    path = tmp_path / 'parnuc_synthetic_format_test.nc'
    _write_synthetic_format_test_table(path)
    table = load_kazil_lovejoy_table(path)
    assert table.temperature.shape == (3,)
    assert table.relative_humidity.shape == (3,)
    assert table.h2so4.shape == (3,)
    assert table.ion_pair_rate.shape == (2,)
    assert table.condensation_sink.shape == (2,)
    assert table.log_pfr.shape == (3, 3, 3, 2, 2)
    np.testing.assert_array_equal(np.asarray(table.temperature), [200., 250., 300.])


def test_load_kazil_lovejoy_table_accepts_bracketed_h2so4_name(tmp_path):
    """The obsolete '[H2SO4]' dimension/variable name (lookup tables built
    before NetCDF forbade '[' as a leading character may still use it;
    ham_nucl_initialize tries 'H2SO4' first, '[H2SO4]' second).
    """
    t_n, rh_n, h_n, i_n, c_n = 2, 2, 2, 2, 2
    path = tmp_path / 'parnuc_synthetic_format_test_bracketed.nc'
    with netcdf_file(path, 'w') as nc:
        nc.createDimension('temperature', t_n)
        nc.createDimension('RH', rh_n)
        nc.createDimension('[H2SO4]', h_n)
        nc.createDimension('ionization', i_n)
        nc.createDimension('condensation_sink', c_n)
        for name, values in (
            ('temperature', [200., 300.]), ('RH', [1., 100.]),
            ('[H2SO4]', [1e5, 1e9]), ('ionization', [1., 20.]),
            ('condensation_sink', [1e-4, 1e-1]),
        ):
            var = nc.createVariable(name, 'd', (name,))
            var[:] = values
        pfr = nc.createVariable('pfr', 'f', ('temperature', 'RH', '[H2SO4]', 'ionization', 'condensation_sink'))
        pfr[:] = np.linspace(-10., 5., t_n*rh_n*h_n*i_n*c_n).reshape(t_n, rh_n, h_n, i_n, c_n)
    table = load_kazil_lovejoy_table(path)
    np.testing.assert_array_equal(np.asarray(table.h2so4), [1e5, 1e9])


def test_load_kazil_lovejoy_table_rejects_nonmonotonic_axis(tmp_path):
    path = tmp_path / 'parnuc_synthetic_format_test_bad.nc'
    with netcdf_file(path, 'w') as nc:
        nc.createDimension('temperature', 2)
        nc.createDimension('RH', 2)
        nc.createDimension('H2SO4', 2)
        nc.createDimension('ionization', 2)
        nc.createDimension('condensation_sink', 2)
        for name, values in (
            ('temperature', [300., 200.]), ('RH', [1., 100.]),
            ('H2SO4', [1e5, 1e9]), ('ionization', [1., 20.]),
            ('condensation_sink', [1e-4, 1e-1]),
        ):
            var = nc.createVariable(name, 'd', (name,))
            var[:] = values
        pfr = nc.createVariable('pfr', 'f', ('temperature', 'RH', 'H2SO4', 'ionization', 'condensation_sink'))
        pfr[:] = np.zeros((2, 2, 2, 2, 2))
    with pytest.raises(ValueError, match='monotonic|unordered'):
        load_kazil_lovejoy_table(path)


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
    # The independent, simpler nucleate() API (distinct from nucleate_all,
    # which DOES support scheme=2 -- see the oracle-gated tests below)
    # stays scheme 0/1 only.
    with pytest.raises(NotImplementedError, match='external lookup table'):
        nucleate(1e8, 280., .5, .1, 0., 1., scheme=2)
    with pytest.raises(ValueError, match='schemes are 0 and 1'):
        nucleate(1e8, 280., .5, .1, 0., 1., scheme=3)


# --- Kazil/Lovejoy (scheme=2): needs the real PARNUC table, HAM_INPUT_DIR ---

@pytest.fixture(scope='module')
def kazil_table():
    return load_kazil_lovejoy_table(_kazil_table_path())


@pytest.fixture(scope='module')
def kazil_oracle(tmp_path_factory):
    return build_oracle(tmp_path_factory.mktemp('m7-kazil'),
                         kazil_table_path=_kazil_table_path())[0]


def kazil_matrix_cases():
    """Designed edge points plus a random sweep of the task's matrix: T
    190-300 K, RH 5-100%, [H2SO4] 1e4-1e10 cm-3, ion-pair 0-50 cm-3 s-1,
    sink 1e-5-1e-1 s-1 (``relative_humidity`` here is the 0-1 fraction, like
    every other case dict in this file; callers multiply by 100 for the
    direct kazil_lovejoy kernel, which -- like the Fortran -- takes percent).
    """
    rng = np.random.default_rng(1234)
    n = 300
    designed = dict(
        temperature=np.array([190., 300., 250., 190., 300., 245.]),
        relative_humidity=np.array([.05, 1.0, .5, 1.0, .05, .5]),
        gas=np.array([1e4, 1e10, 1e7, 1e10, 1e4, 1e8]),
        ion_pair_rate=np.array([0., 50., 10., 0., 50., 25.]),
        total_sink=np.array([1e-5, 1e-1, 1e-3, 1e-1, 1e-5, 1e-2]),
    )
    random_sweep = dict(
        temperature=rng.uniform(190., 300., n),
        relative_humidity=rng.uniform(.05, 1.0, n),
        gas=10.**rng.uniform(4., 10., n),
        ion_pair_rate=rng.uniform(0., 50., n),
        total_sink=10.**rng.uniform(-5., -1., n),
    )
    return {k: np.concatenate([designed[k], random_sweep[k]]) for k in designed}


@_kazil_skip
@pytest.mark.parametrize('compiled', [False, True])
def test_kazil_lovejoy_matches_actual_native(kazil_oracle, kazil_table, compiled):
    cases = kazil_matrix_cases()
    expected, _ = run_oracle(kazil_oracle, cases, operation='kazil_lovejoy', kazil=True)
    kernel = jax.jit(kazil_lovejoy) if compiled else kazil_lovejoy
    rate, cluster = kernel(cases['temperature'], cases['relative_humidity']*100., cases['gas'],
                            cases['total_sink'], cases['ion_pair_rate'], kazil_table)
    np.testing.assert_allclose(np.asarray(rate), expected['rate'], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(np.asarray(cluster), expected['cluster_sulfate'])
    assert np.any(np.asarray(rate) == 0.)  # the lset_zero branches are exercised
    assert np.any(np.asarray(rate) > 0.)


def nucleate_all_kazil_cases():
    cases = kazil_matrix_cases()
    n = len(cases['gas'])
    rng = np.random.default_rng(99)
    cases = dict(cases)
    cases['cloud_fraction'] = rng.uniform(0., 1., n)
    cases['timestep'] = rng.uniform(1., 600., n)
    return cases


@_kazil_skip
@pytest.mark.parametrize('compiled', [False, True])
def test_nucleate_all_kazil_matches_actual_native_and_closes_sulfur(kazil_oracle, kazil_table, compiled):
    cases = nucleate_all_kazil_cases()
    oracle_cases = dict(cases, scheme=np.full(len(cases['gas']), 2))
    expected, _ = run_oracle(kazil_oracle, oracle_cases, operation='nucleate', kazil=True)
    kernel = partial(nucleate_all, scheme=2, kazil_lovejoy_table=kazil_table)
    if compiled:
        kernel = jax.jit(kernel, static_argnames=('scheme', 'organic_scheme'))
    result = kernel(cases['gas'], cases['temperature'], cases['relative_humidity'],
                     cases['total_sink'], cases['cloud_fraction'], 0., False, cases['timestep'],
                     ion_pair_rate=cases['ion_pair_rate'])
    # atol floors a handful of designed-matrix cells where almost all clear
    # gas is consumed (expected == 0.0 exactly, by eager cancellation): XLA's
    # jit fuses the clear_final clamp/division chain differently than eager,
    # so the compiled path lands a ~1e-7 absolute residual there instead of
    # an exact 0 -- standard jit-vs-eager float non-associativity (the eager
    # path, and every cell that is not near-total consumption, holds
    # rtol=1e-12 with no floor; this never masks a genuine relative error at
    # the gas/number/sulfate scales here, 1e4-1e10).
    np.testing.assert_allclose(np.asarray(result.gas), expected['gas'], rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.number), expected['nucleated_number'], rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.sulfate), expected['sulfate_delta'][..., 0], rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.gas) + np.asarray(result.sulfate), cases['gas'], rtol=3e-15, atol=0)


@_kazil_skip
def test_kazil_lovejoy_follows_the_input_dtype():
    """float32 inputs run in float32 (no silent float64 request), close to float64."""
    import warnings
    from m7_jax.nucleation import kazil_lovejoy
    table = load_kazil_lovejoy_table(_kazil_table_path())
    t = np.array([200., 230., 260., 290.]); rh = np.array([5., 30., 60., 95.])
    h2so4 = np.array([1e6, 1e7, 1e8, 1e9]); cs = np.array([1e-4, 1e-3, 1e-2, 5e-2])
    ipr = np.array([1., 5., 20., 50.])
    with jax.enable_x64(True):
        ref = kazil_lovejoy(t, rh, h2so4, cs, ipr, table)
    with jax.enable_x64(False), warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        f32 = kazil_lovejoy(*(x.astype(np.float32) for x in (t, rh, h2so4, cs, ipr)), table)
    assert f32.rate.dtype == np.float32
    np.testing.assert_allclose(np.asarray(f32.rate), np.asarray(ref.rate), rtol=1e-4)
