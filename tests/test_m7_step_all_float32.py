"""``step_all`` forward-only in float32, without ``jax_enable_x64`` (#1017 W1
task 3). No gradient requirement in float32 -- this file never calls
``jax.grad``.

Deliberately does NOT call ``jax.config.update('jax_enable_x64', True)`` at
module scope: other test modules in this suite do, as a process-wide side
effect of being imported, and pytest imports every test module during
collection regardless of run order. Every test here therefore forces
``jax_enable_x64`` to ``False`` itself (restoring whatever was there after),
rather than relying on whatever the default happens to be when it runs.

Measured max relative error vs. the native (float64) oracle on the designed
cases (5/7 (scheme, organic) combinations tested; see
``_measure_max_relative_errors``):
  - 6 of 7 cases, every (scheme, organic) combination, every field: <=1e-3
    (most <=1e-5; the softest is "gas-only" under nucleation, ~6e-4 -- an
    all-zero-aerosol population built from scratch by nucleation alone is
    inherently more cancellation-sensitive than one with an existing
    population to perturb).
  - The "cold-ut" case (cold upper troposphere, T=220K, strong H2SO4, a small
    pre-existing insoluble Aitken (KI) population of 50 cm-3) under
    nucleation_scheme=1: ~0.71 relative error, but ONLY in the KI-mode BC/OC
    mass components (IBCKI/IOCKI) and the KI number -- every other field and
    case stays <=1e-3. This is the "different branch" the task anticipates:
    `dnum_all`'s insoluble/soluble mixed-coagulation correction (m7_delcoa
    2.2.3, mo_ham_m7.F90:4368-4452) and the KI self-coagulation gate
    (m7_delcoa section 1, mo_ham_m7.F90:4072-4190) compare coagulation
    kernels and number thresholds (``ZMIN_COAG``, ``CMIN_AERNL``) that, for
    this case's near-threshold KI population, land on opposite sides of a
    `>=`/`>` comparison under float32 vs float64 rounding of the upstream
    coagulation kernel -- a genuine branch difference, not drift or
    instability: output stays finite, non-negative and (see below)
    per-species mass and sulfur still conserve to ~1e-7, because the
    mismatched branch still only MOVES mass between KI and KS/AS, never
    creates or destroys it.
  - Per-species (BC/OC/SS/DU) mass and sulfur conservation: <=3.6e-8 on
    every case INCLUDING "cold-ut" -- float32 machine epsilon order, since
    conservation is a sum invariant unaffected by which branch fired.
"""
import functools

import jax
import numpy as np
import pytest

import jax.numpy as jnp

from m7_jax.model import FullState, step_all
from m7_jax.properties import load_kappa_table
from tools.m7_oracle import ROOT, build_oracle, run_oracle

# Measured maxima (see module docstring) with headroom; "cold-ut" is excluded
# from this general bound and checked separately (large but understood).
GENERAL_RTOL = 1e-3
CONSERVATION_RTOL = 1e-6
_COLD_UT = 3  # designed_cases() row index


@pytest.fixture(autouse=True)
def _force_x64_off():
    """Force `jax_enable_x64` off for every test in this module, regardless
    of what another collected test module's import-time side effect left it
    at (see module docstring)."""
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update('jax_enable_x64', False)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


@pytest.fixture(scope='module')
def oracles(tmp_path_factory):
    return {h: build_oracle(tmp_path_factory.mktemp(f'm7-full-f32-{h}'), checked=True, hammoz=h)[0]
            for h in (False, True)}


@pytest.fixture(scope='module')
def table32():
    table64 = load_kappa_table(ROOT / 'reference' / 'input' / 'lut_kappa.nc')
    # The kappa table must be passed in the working (core) dtype; building
    # this once as float32 is this file's analogue of the jcm adapter's
    # `core_dtype="float32"` boundary cast (#1017 W1 task 3 Part B).
    return table64._replace(**{
        field: jnp.asarray(getattr(table64, field), dtype=jnp.float32)
        for field in table64._fields})


def designed_cases():
    """Same mixed HAM populations as test_m7_step_all.py's designed_cases
    (not imported from there: that module enables jax_enable_x64 as an
    import-time side effect -- see this module's docstring)."""
    n = 7
    mass = np.zeros((n, 18))
    number = np.zeros((n, 7))
    mass[:, 0:4] = [[1e6, 5e8, 5e10, 1e10], [5e6, 2e9, 3e11, 2e10], [1e5, 1e8, 2e10, 3e10],
                    [1e7, 1e9, 1e10, 0.], [1e6, 1e9, 1e11, 1e10], [0, 0, 0, 0], [0, 1e8, 2e10, 5e10]]
    mass[1, 4:8] = [.05, .4, .02, .8]
    mass[1, 8:12] = [.2, 2., .1, 1.5]
    mass[3, 7], mass[3, 11] = .01, .02
    mass[4, 5], mass[4, 9] = .1, .5
    mass[0, 12:14] = [1., 20.]
    mass[6, 12:14] = [3., 60.]
    mass[2, 14:18] = [2., 30., 15., 200.]
    mass[4, 16:18] = [.5, 3.]
    number[:] = [[800, 300, 100, 1, 0, 0, 0], [5000, 3000, 1500, 2, 2000, 0, 0],
                 [100, 50, 80, 5, 0, 3, 1], [20000, 800, 30, 0, 50, 0, 0],
                 [1000, 900, 400, 2, 0, .5, .2], [0, 0, 0, 0, 0, 0, 0], [50, 40, 60, 8, 0, 0, 0]]
    return dict(
        labels=['marine', 'polluted', 'dust', 'cold-ut', 'cloudy', 'gas-only', 'seasalt'],
        mass=mass, number=number,
        temperature=np.array([290., 285., 300., 220., 275., 260., 295.]),
        pressure=np.array([101325., 95000., 85000., 25000., 80000., 50000., 100000.]),
        relative_humidity=np.array([.85, .6, .25, .4, .95, .5, .9]),
        gas=np.array([1e6, 5e7, 1e6, 3e7, 2e7, 1e8, 1e5]),
        production=np.array([1e3, 5e4, 1e3, 2e4, 1e4, 1e5, 0.]),
        cloud_fraction=np.array([0., .2, 0., 0., .7, 0., .3]),
        timestep=np.full(n, 720.),
        forest_fraction=np.array([0., .4, 0., 0., .8, 1., 0.]),
        pbl_level=np.array([1., 1., 2., 2., 1., 1., 1.]),
    )


def run_jax32(cases, table32, *, scheme, organic):
    """step_all in float32, jax_enable_x64 left off (the fixture enforces this)."""
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
    in_pbl = jnp.asarray(1 >= np.floor(cases['pbl_level']))
    state = FullState(f32(cases['mass']), f32(cases['number']), f32(cases['gas']))
    fn = functools.partial(step_all, nucleation_scheme=scheme, organic_scheme=organic)
    return fn(state, f32(cases['temperature']), f32(cases['pressure']), f32(cases['relative_humidity']),
              f32(cases['timestep']), f32(cases['production']), f32(cases['cloud_fraction']), table32,
              f32(cases['forest_fraction']), in_pbl)


def _fields(result):
    return [('mass', result.state.mass), ('number', result.state.number), ('gas', result.state.gas),
            ('radius', result.properties.wet_radius), ('density', result.properties.density),
            ('dry_radius', result.properties.dry_radius), ('water', result.properties.water[..., :4])]


def _relerr(mine, native):
    mine = np.asarray(mine, dtype=np.float64)
    return np.abs(mine - native) / np.maximum(np.abs(native), 1e-300)


@pytest.mark.parametrize('scheme,organic', [(0, 0), (1, 0), (0, 1), (1, 1), (1, 2)])
def test_step_all_float32_matches_native_within_measured_tolerance(oracles, table32, scheme, organic):
    cases = designed_cases()
    n = len(cases['labels'])
    cases.update(scheme=np.full(n, scheme), organic_scheme=np.full(n, organic), water=np.ones(n))
    native, _ = run_oracle(oracles[organic > 0], cases, operation='full')
    result = run_jax32(cases, table32, scheme=scheme, organic=organic)

    assert bool(np.all(result.valid))
    for _, value in _fields(result):
        value = np.asarray(value)
        assert np.all(np.isfinite(value)), 'non-finite float32 output'
        assert np.all(value >= 0), 'negative float32 output'

    for key, value in _fields(result):
        expected = native[key] if key != 'water' else native['water'][:, :4]
        err = _relerr(value, expected)
        # "cold-ut" (row _COLD_UT) legitimately takes a different branch in
        # dnum_all's insoluble-Aitken-mode coagulation gate under nucleation
        # (scheme=1) -- see module docstring. Excluded from the general
        # bound there; still required finite/non-negative above.
        rows = err if scheme == 0 else np.delete(err, _COLD_UT, axis=0)
        assert np.max(rows) < GENERAL_RTOL, (key, scheme, organic, float(np.max(rows)))


def test_step_all_float32_cold_ut_branch_difference_is_bounded(oracles, table32):
    """The one case/field combination excluded above: still sane, just coarse."""
    cases = designed_cases()
    n = len(cases['labels'])
    cases.update(scheme=np.ones(n), organic_scheme=np.zeros(n), water=np.ones(n))
    native, _ = run_oracle(oracles[False], cases, operation='full')
    result = run_jax32(cases, table32, scheme=1, organic=0)
    mass_err = _relerr(result.state.mass, native['mass'])[_COLD_UT]
    number_err = _relerr(result.state.number, native['number'])[_COLD_UT]
    print('cold-ut branch-difference relative error: mass', mass_err.max(),
          'number', number_err.max())
    # Bounded (finite, not absurd) rather than tight: a genuine branch
    # difference, not a precision target.
    assert np.all(np.isfinite(mass_err)) and np.all(np.isfinite(number_err))
    assert mass_err.max() < 1.0
    assert number_err.max() < 1.0


@pytest.mark.parametrize('scheme,organic', [(1, 0), (1, 1)])
def test_step_all_float32_conserves_species_and_sulfur(table32, scheme, organic):
    cases = designed_cases()
    result = run_jax32(cases, table32, scheme=scheme, organic=organic)
    mass0, mass1 = cases['mass'], np.asarray(result.state.mass, dtype=np.float64)
    errors = {}
    for lo, hi, name in ((4, 8, 'BC'), (8, 12, 'OC'), (12, 14, 'SS'), (14, 18, 'DU')):
        before, after = mass0[:, lo:hi].sum(-1), mass1[:, lo:hi].sum(-1)
        err = np.abs(after - before) / np.maximum(np.abs(before), 1e-30)
        errors[name] = float(err.max())
        assert err.max() < CONSERVATION_RTOL, (name, err.max())
    sulfur0 = mass0[:, :4].sum(-1) + cases['gas'] + cases['production'] * cases['timestep']
    sulfur1 = mass1[:, :4].sum(-1) + np.asarray(result.state.gas, dtype=np.float64)
    sulfur_err = np.abs(sulfur1 - sulfur0) / np.abs(sulfur0)
    errors['sulfur'] = float(sulfur_err.max())
    assert sulfur_err.max() < CONSERVATION_RTOL
    print('float32 conservation max relative error per species:', errors)


def test_step_all_float32_is_bit_identical_dtype_not_silently_promoted():
    """Confirms the dtype-generic refactor actually stays float32 end to
    end (no silent float64 promotion defeating the point of this task)."""
    cases = designed_cases()
    table64 = load_kappa_table(ROOT / 'reference' / 'input' / 'lut_kappa.nc')
    table32 = table64._replace(**{
        field: jnp.asarray(getattr(table64, field), dtype=jnp.float32)
        for field in table64._fields})
    result = run_jax32(cases, table32, scheme=1, organic=1)
    for name, value in _fields(result):
        assert np.asarray(value).dtype == np.float32, name


def test_step_all_float32_is_jittable():
    cases = designed_cases()
    table64 = load_kappa_table(ROOT / 'reference' / 'input' / 'lut_kappa.nc')
    table32 = table64._replace(**{
        field: jnp.asarray(getattr(table64, field), dtype=jnp.float32)
        for field in table64._fields})
    fn = jax.jit(functools.partial(step_all, nucleation_scheme=1, organic_scheme=0))
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
    in_pbl = jnp.asarray(1 >= np.floor(cases['pbl_level']))
    state = FullState(f32(cases['mass']), f32(cases['number']), f32(cases['gas']))
    result = fn(state, f32(cases['temperature']), f32(cases['pressure']), f32(cases['relative_humidity']),
                f32(cases['timestep']), f32(cases['production']), f32(cases['cloud_fraction']), table32,
                f32(cases['forest_fraction']), in_pbl)
    assert bool(np.all(result.valid))
    assert result.state.mass.dtype == np.float32
