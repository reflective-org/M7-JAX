# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Kazil-Lovejoy (nucleation scheme 2) against the actual native routine,
using a synthetic PARNUC-layout table so the default run needs no
HAM_INPUT_DIR.

The table below is made up: smooth, physically shaped ``ln(rate)`` values on
the real file's variable layout. It checks code logic only -- the netCDF
reader, the oracle's ``pfr.bin`` serialisation into Fortran column-major
order, the 32-corner interpolation, the axis clamps and the ``lset_zero``
branches -- and is never a source of physics numbers. The real-table parity
tests stay in tests/test_m7_nucleation.py behind HAM_INPUT_DIR.
"""
from functools import partial

import jax
import numpy as np
import pytest
from scipy.io import netcdf_file

from m7_jax.nucleation import kazil_lovejoy, load_kazil_lovejoy_table, nucleate_all
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update('jax_enable_x64', True)

# Unequal axis lengths and non-uniform spacing, so any transposed or
# mis-strided serialisation addresses a different corner.
AXES = {
    'temperature': np.array([190., 205., 220., 235., 250., 265., 282., 300.]),
    'RH': np.array([5., 15., 30., 50., 75., 100.]),
    'H2SO4': 10.**np.array([4., 5., 6., 7., 8., 9., 10.]),
    'ionization': np.array([1., 3., 10., 25., 60.]),
    'condensation_sink': 10.**np.array([-5., -4., -3., -2., -1.]),
}


def _synthetic_log_rate():
    """ln J [cm-3 s-1]: rises with [H2SO4], RH and ion-pair rate, falls with
    temperature and condensation sink -- plausible signs, smooth in every
    axis, and every corner distinct."""
    t, rh, h, ipr, cs = np.meshgrid(*AXES.values(), indexing='ij')
    return (-4. + 2.2*np.log(h/1e7) + 1.3*np.log(rh/50.) - 0.09*(t - 240.)
            + 0.4*np.log(ipr) - 0.35*np.log(cs/1e-3)
            + 0.01*np.log(h/1e7)*(t - 240.)/10.)


def _write_table(path):
    with netcdf_file(path, 'w') as nc:
        for name, values in AXES.items():
            nc.createDimension(name, len(values))
            nc.createVariable(name, 'd', (name,))[:] = values
        nc.createVariable('pfr', 'd', tuple(AXES))[:] = _synthetic_log_rate()


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    directory = tmp_path_factory.mktemp('m7-kazil-synthetic')
    path = directory / 'parnuc_synthetic_layout.nc'
    _write_table(path)
    executable = build_oracle(directory / 'oracle', kazil_table_path=path)[0]
    return executable, load_kazil_lovejoy_table(path)


def kazil_cases():
    """Interior random points, exact table nodes (incl. both ends of every
    axis), and points beyond each edge of each axis. ``relative_humidity``
    is the 0-1 fraction; the direct kernel gets percent."""
    lo = {k: v[0] for k, v in AXES.items()}
    hi = {k: v[-1] for k, v in AXES.items()}
    mid = {'temperature': 243., 'RH': 42., 'H2SO4': 3e7, 'ionization': 7., 'condensation_sink': 2e-3}
    rows = []
    # Beyond, on and just inside each edge, other axes interior. Beyond-edge
    # zeroes for T > max, RH < min and [H2SO4] < min (native lset_zero); all
    # other excursions clamp.
    for name in AXES:
        for value in (lo[name]*0.5 if name != 'temperature' else lo[name] - 20.,
                      lo[name],
                      hi[name],
                      hi[name]*2. if name != 'temperature' else hi[name] + 20.):
            rows.append(dict(mid, **{name: value}))
    # Every axis outside at once, both directions.
    rows.append({'temperature': 150., 'RH': 120., 'H2SO4': 1e12, 'ionization': 100., 'condensation_sink': 1.})
    rows.append({'temperature': 150., 'RH': 120., 'H2SO4': 1e12, 'ionization': 0., 'condensation_sink': 1e-7})
    rows.append({'temperature': 350., 'RH': 1., 'H2SO4': 1e2, 'ionization': 0., 'condensation_sink': 1e-7})
    # Exact nodes, including interior ones.
    rng = np.random.default_rng(2026)
    for _ in range(40):
        rows.append({k: v[rng.integers(len(v))] for k, v in AXES.items()})
    cases = {key: np.array([row[name] for row in rows])
             for key, name in (('temperature', 'temperature'), ('relative_humidity', 'RH'),
                               ('gas', 'H2SO4'), ('ion_pair_rate', 'ionization'),
                               ('total_sink', 'condensation_sink'))}
    # Random sweep spanning and overshooting every axis.
    n = 300
    sweep = dict(temperature=rng.uniform(170., 320., n),
                 relative_humidity=rng.uniform(1., 110., n),
                 gas=10.**rng.uniform(3., 11., n),
                 ion_pair_rate=rng.uniform(0., 80., n),
                 total_sink=10.**rng.uniform(-6., 0., n))
    cases = {k: np.concatenate([cases[k], sweep[k]]) for k in cases}
    cases['relative_humidity'] = cases['relative_humidity']/100.
    return cases


@pytest.mark.parametrize('compiled', [False, True])
def test_synthetic_kazil_lovejoy_matches_native(synthetic, compiled):
    executable, table = synthetic
    cases = kazil_cases()
    expected, _ = run_oracle(executable, cases, operation='kazil_lovejoy', kazil=True)
    kernel = jax.jit(kazil_lovejoy) if compiled else kazil_lovejoy
    rate, cluster = kernel(cases['temperature'], cases['relative_humidity']*100., cases['gas'],
                           cases['total_sink'], cases['ion_pair_rate'], table)
    rate = np.asarray(rate)
    # Same zero pattern, then the values. Measured max relative error
    # 3.5e-15 eager, 2.0e-16 jit (ln J reaches ~19 here, so exp turns a few
    # ulp of the interpolated logarithm into a few 1e-15 of the rate).
    np.testing.assert_array_equal(rate == 0., expected['rate'] == 0.)
    np.testing.assert_allclose(rate, expected['rate'], rtol=2e-14, atol=0)
    nonzero = expected['rate'] != 0.
    np.testing.assert_array_equal(np.asarray(cluster)[nonzero], expected['cluster_sulfate'][nonzero])

    # Every branch is exercised: each lset_zero condition alone, and a
    # clamped (nonzero) cell beyond each of the other edges.
    t, rh, gas = cases['temperature'], cases['relative_humidity']*100., cases['gas']
    ipr, cs = cases['ion_pair_rate'], cases['total_sink']
    interior = lambda v, axis: (v >= AXES[axis][0]) & (v <= AXES[axis][-1])
    inside = (interior(t, 'temperature') & interior(rh, 'RH') & interior(gas, 'H2SO4')
              & interior(ipr, 'ionization') & interior(cs, 'condensation_sink'))
    assert np.all(rate[inside] > 0.)
    assert np.all(rate[t > AXES['temperature'][-1]] == 0.)
    assert np.all(rate[rh < AXES['RH'][0]] == 0.)
    assert np.all(rate[gas < AXES['H2SO4'][0]] == 0.)
    keep = (t <= AXES['temperature'][-1]) & (rh >= AXES['RH'][0]) & (gas >= AXES['H2SO4'][0])
    for clamped in (t < AXES['temperature'][0], rh > AXES['RH'][-1], gas > AXES['H2SO4'][-1],
                    ipr < AXES['ionization'][0], ipr > AXES['ionization'][-1],
                    cs < AXES['condensation_sink'][0], cs > AXES['condensation_sink'][-1]):
        assert np.any(clamped & keep & (rate > 0.))


def test_synthetic_kazil_lovejoy_reproduces_table_nodes(synthetic):
    """At exact nodes the interpolation returns the stored value."""
    _, table = synthetic
    rng = np.random.default_rng(7)
    index = [rng.integers(len(v), size=50) for v in AXES.values()]
    point = [v[i] for v, i in zip(AXES.values(), index)]
    rate, _ = kazil_lovejoy(point[0], point[1], point[2], point[4], point[3], table)
    np.testing.assert_allclose(np.log(np.asarray(rate)), _synthetic_log_rate()[tuple(index)],
                               rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('compiled', [False, True])
def test_synthetic_nucleate_all_scheme2_matches_native(synthetic, compiled):
    executable, table = synthetic
    cases = kazil_cases()
    n = len(cases['gas'])
    rng = np.random.default_rng(99)
    cases['cloud_fraction'] = rng.uniform(0., 1., n)
    cases['timestep'] = rng.uniform(1., 600., n)
    expected, _ = run_oracle(executable, dict(cases, scheme=np.full(n, 2)),
                             operation='nucleate', kazil=True)
    kernel = partial(nucleate_all, scheme=2, kazil_lovejoy_table=table)
    if compiled:
        kernel = jax.jit(kernel)
    result = kernel(cases['gas'], cases['temperature'], cases['relative_humidity'],
                    cases['total_sink'], cases['cloud_fraction'], 0., False, cases['timestep'],
                    ion_pair_rate=cases['ion_pair_rate'])
    # Measured: relative error up to 3.9e-14 eager and 5.8e-16 jit wherever
    # the native value is not a near-total-consumption residual. Where the
    # clear-sky gas is (almost) all consumed the native gas is 0 or tiny and
    # the remainder is cancellation noise of the input: measured absolute
    # error up to 1.6e-15 of the input gas eager (about 1.8e-6 cm-3 out of
    # 4e10 under jit). So the floor is scaled by the input gas, cell by cell.
    floor = 4e-15*cases['gas']
    for value, reference in ((result.gas, expected['gas']),
                             (result.number, expected['nucleated_number']),
                             (result.sulfate, expected['sulfate_delta'][..., 0])):
        error = np.abs(np.asarray(value) - reference)
        assert np.all(error <= 1e-13*np.abs(reference) + floor), np.max(error/(np.abs(reference) + floor))
    np.testing.assert_array_equal(expected['sulfate_delta'][..., 1:], 0.)
    np.testing.assert_allclose(np.asarray(result.gas) + np.asarray(result.sulfate), cases['gas'],
                               rtol=3e-15, atol=0)
    assert np.any(np.asarray(result.number) == 0.) and np.any(np.asarray(result.number) > 0.)
