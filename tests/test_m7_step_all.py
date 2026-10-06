"""The composed 18-component ``step_all`` against the complete native ``m7`` call."""
import functools
import os
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from m7_jax.model import FullState, step_all  # noqa: E402
from m7_jax.nucleation import load_kazil_lovejoy_table  # noqa: E402
from m7_jax.properties import load_kappa_table  # noqa: E402
from tools.m7_oracle import ROOT, build_oracle, run_oracle  # noqa: E402

RTOL = 1e-12

# jax-gcm#1017 Kazil/GCR task, Part A: see test_m7_nucleation.py's own copy
# of this helper/skip for why HAM_INPUT_DIR and both table names are needed.
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


@pytest.fixture(scope='module')
def oracles(tmp_path_factory):
    return {h: build_oracle(tmp_path_factory.mktemp(f'm7-full-{h}'), checked=True, hammoz=h)[0]
            for h in (False, True)}


@pytest.fixture(scope='module')
def table():
    return load_kappa_table(ROOT / 'reference' / 'input' / 'lut_kappa.nc')


def designed_cases():
    """Mixed HAM populations across the native switches.

    Rows: clean marine, polluted continental with fresh insoluble carbon,
    dust outbreak, cold upper troposphere with strong H2SO4, cloudy cell,
    an all-empty cell with only gas, and a coarse sea-salt-dominated cell.
    """
    n = 7
    mass = np.zeros((n, 18))
    number = np.zeros((n, 7))
    # SO4 molecules cm-3 per soluble mode (NS KS AS CS)
    mass[:, 0:4] = [[1e6, 5e8, 5e10, 1e10], [5e6, 2e9, 3e11, 2e10], [1e5, 1e8, 2e10, 3e10],
                    [1e7, 1e9, 1e10, 0.], [1e6, 1e9, 1e11, 1e10], [0, 0, 0, 0], [0, 1e8, 2e10, 5e10]]
    # BC KS AS CS KI, OC KS AS CS KI [ug m-3]
    mass[1, 4:8] = [.05, .4, .02, .8]
    mass[1, 8:12] = [.2, 2., .1, 1.5]
    mass[3, 7], mass[3, 11] = .01, .02
    mass[4, 5], mass[4, 9] = .1, .5
    # SS AS CS, DU AS CS AI CI [ug m-3]
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


def run_jax(cases, table, *, scheme, organic, compiled):
    fn = functools.partial(step_all, nucleation_scheme=scheme, organic_scheme=organic)
    if compiled:
        fn = jax.jit(fn)
    in_pbl = 1 >= np.floor(cases['pbl_level'])  # single-level native column: jk = 1
    state = FullState(cases['mass'], cases['number'], cases['gas'])
    return fn(state, cases['temperature'], cases['pressure'], cases['relative_humidity'],
              cases['timestep'], cases['production'], cases['cloud_fraction'], table,
              cases['forest_fraction'], in_pbl)


def check(native, result):
    pairs = [('mass', result.state.mass), ('number', result.state.number), ('gas', result.state.gas),
             ('radius', result.properties.wet_radius), ('density', result.properties.density),
             ('dry_radius', result.properties.dry_radius), ('water', result.properties.water[..., :4])]
    for key, value in pairs:
        expected = native[key] if key != 'water' else native['water'][:, :4]
        np.testing.assert_allclose(np.asarray(value), expected, rtol=RTOL, atol=0, err_msg=key)
    assert bool(np.all(result.valid))


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('scheme,organic', [(0, 0), (1, 0), (0, 1), (1, 1), (1, 2)])
def test_step_all_matches_native_m7(oracles, table, scheme, organic, compiled):
    cases = designed_cases()
    n = len(cases['labels'])
    cases.update(scheme=np.full(n, scheme), organic_scheme=np.full(n, organic), water=np.ones(n))
    native, _ = run_oracle(oracles[organic > 0], cases, operation='full')
    check(native, run_jax(cases, table, scheme=scheme, organic=organic, compiled=compiled))


def test_step_all_coagulation_only(oracles, table):
    cases = designed_cases()
    n = len(cases['labels'])
    cases.update(scheme=np.zeros(n), condensation=np.zeros(n), water=np.ones(n))
    native, _ = run_oracle(oracles[False], cases, operation='full')
    in_pbl = np.ones(n, bool)
    result = step_all(FullState(cases['mass'], cases['number'], cases['gas']), cases['temperature'],
                      cases['pressure'], cases['relative_humidity'], cases['timestep'],
                      cases['production'], cases['cloud_fraction'], table, cases['forest_fraction'],
                      in_pbl, condensation=False, nucleation_scheme=0)
    check(native, result)


def test_step_all_conserves_species_and_sulfur(table):
    cases = designed_cases()
    result = run_jax(cases, table, scheme=1, organic=1, compiled=False)
    mass0, mass1 = cases['mass'], np.asarray(result.state.mass)
    for lo, hi in ((4, 8), (8, 12), (12, 14), (14, 18)):  # BC, OC, SS, DU
        np.testing.assert_allclose(mass1[:, lo:hi].sum(-1), mass0[:, lo:hi].sum(-1), rtol=1e-13)
    sulfur0 = mass0[:, :4].sum(-1) + cases['gas'] + cases['production'] * cases['timestep']
    sulfur1 = mass1[:, :4].sum(-1) + np.asarray(result.state.gas)
    np.testing.assert_allclose(sulfur1, sulfur0, rtol=1e-13)


# --- nucleation_scheme=2 (Kazil/Lovejoy): needs the real PARNUC table ---

@pytest.fixture(scope='module')
def kazil_oracle(tmp_path_factory):
    # organic_scheme stays 0 here: build_oracle does not support combining
    # hammoz=True (organic nucleation) with kazil_table_path (both edit the
    # driver init anchor) -- see its docstring. m7_nuck's scheme/organic
    # branches are independent (mo_ham_m7.F90:2401-2422), so this covers
    # nucleation_scheme=2 fully; organic+Kazil together is untested (neither
    # the Fortran call site nor this task asks for that combination).
    return build_oracle(tmp_path_factory.mktemp('m7-full-kazil'), checked=True,
                         kazil_table_path=_kazil_table_path())[0]


@pytest.fixture(scope='module')
def kazil_table():
    return load_kazil_lovejoy_table(_kazil_table_path())


@_kazil_skip
@pytest.mark.parametrize('compiled', [False, True])
def test_step_all_kazil_matches_native_m7(kazil_oracle, table, kazil_table, compiled):
    cases = designed_cases()
    n = len(cases['labels'])
    rng = np.random.default_rng(2024)
    cases['ion_pair_rate'] = rng.uniform(0., 50., n)
    cases.update(scheme=np.full(n, 2), organic_scheme=np.zeros(n), water=np.ones(n))
    native, _ = run_oracle(kazil_oracle, cases, operation='full', kazil=True)
    fn = functools.partial(step_all, nucleation_scheme=2, organic_scheme=0,
                            ion_pair_rate=cases['ion_pair_rate'], kazil_lovejoy_table=kazil_table)
    if compiled:
        fn = jax.jit(fn, static_argnames=('nucleation_scheme', 'organic_scheme'))
    in_pbl = 1 >= np.floor(cases['pbl_level'])
    state = FullState(cases['mass'], cases['number'], cases['gas'])
    result = fn(state, cases['temperature'], cases['pressure'], cases['relative_humidity'],
                cases['timestep'], cases['production'], cases['cloud_fraction'], table,
                cases['forest_fraction'], in_pbl)
    check(native, result)
