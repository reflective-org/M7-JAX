"""Multi-species redistribution (m7_dconc) parity against the actual Fortran.

Each case row below is an independent cell, covering: a mode kept entirely
below its upper radius bound (no transfer), one straddling it (partial
transfer), one whose dry radius is more than 10x the upper bound (the
native Q&D cutoff, mo_ham_m7.F90:3193-3197: zfconn forced to 0), an empty
mode (number=0), and KS/AS/CS internal species mixtures (BC+OC in KS;
SO4+BC+OC+SS+DU in AS and CS).
"""
from pathlib import Path

import jax
import numpy as np
import pytest

from m7_jax.constants import COMPONENT_MODE, COMPONENT_SPECIES, MODE_RADIUS_BOUNDS
from m7_jax.redistribution import repartition_all
from tools.m7_oracle import build_oracle, run_oracle

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    return build_oracle(tmp_path_factory.mktemp('m7-redistribution-multispecies'))[0]


def _designed_cases():
    lower = np.array(MODE_RADIUS_BOUNDS)      # NS,KS,AS,CS lower/upper mode boundaries
    upper = np.r_[lower[1:], lower[-1]*10]
    midpoint = np.exp((np.log(lower)+np.log(upper))/2)
    number = (1e4, 1e3, 1e2, 10.)

    # m7_dconc's transfer gate ("IF zttnj > zavnj", mo_ham_m7.F90:3176) compares
    # each mode's mean particle MASS (total mode mass / number) against the
    # mass of a particle at that mode's fixed geometric-mean radius; it does
    # not depend on the `dry_radius` argument at all (that only sets the CDF
    # used for the transferred FRACTION once the gate is already open). The
    # three emitting modes' (NS, KS, AS) per-particle mass thresholds, in SO4
    # molecules, scale as 1000x per mode (radius bounds are a decade apart):
    # ~493, ~4.93e5, ~4.93e8 (matches ZFACSO4M*4pi/3*(geometric-mean
    # radius*CMR2RAM)**3*SULFATE_DENSITY). "below"/"above" mass picks a mean
    # particle mass well under/over that threshold for every emitting mode.
    below_mass = (number[0]*100., number[1]*1e4, number[2]*1e6, 1e8)         # SO4 NS,KS,AS,CS
    above_mass = (number[0]*1e4, number[1]*1e7, number[2]*1e10, 1e8)        # SO4 NS,KS,AS,CS

    rows_mass, rows_number, rows_radius = [], [], []

    def add(mass_by_component, number_by_mode, dry_radius):
        mass = np.zeros(18)
        for idx, value in mass_by_component.items():
            mass[idx] = value
        full_number = np.zeros(7)
        for idx, value in number_by_mode.items():
            full_number[idx] = value
        rows_mass.append(mass)
        rows_number.append(full_number)
        rows_radius.append(np.asarray(dry_radius, dtype=float))

    all_modes_number = dict(enumerate(number))

    # All four modes' mean particle mass below the transfer-gate threshold:
    # no transfer anywhere, regardless of the given (here, mid-range) radius.
    add(dict(enumerate(below_mass)), all_modes_number, midpoint)

    # All four modes above threshold and straddling their upper radius bound:
    # partial transfer, every species of KS/AS moving together (BC+OC in KS;
    # BC+OC+SS+DU in AS and CS).
    mixed_mass = dict(enumerate(above_mass))
    mixed_mass.update({4: 0.1, 8: 0.2})                    # BC+OC in KS
    mixed_mass.update({5: 1.0, 9: 2.0, 12: 3.0, 14: 4.0})  # BC+OC+SS+DU in AS
    mixed_mass.update({6: 0.5, 10: 0.6, 13: 0.7, 15: 0.8})  # BC+OC+SS+DU in CS
    add(mixed_mass, all_modes_number, midpoint)

    # Above threshold, but NS/KS/AS dry radius more than 10x the upper bound:
    # the Q&D zero-stay cutoff (mo_ham_m7.F90:3193-3197) transfers everything.
    add(mixed_mass, all_modes_number, upper*11)

    # Empty state: nothing active (number below cmin_aernl), nothing to transfer.
    add({}, {}, lower*.5)

    # Only KS active and above threshold; NS/AS/CS have zero number.
    add(mixed_mass, {1: number[1]}, midpoint)

    # Internally mixed AS above threshold and straddling, sea salt/dust dominant.
    add({2: above_mass[2], 5: .1, 9: .1, 12: 50., 14: 80.}, {2: number[2]}, midpoint)

    return dict(mass=np.stack(rows_mass), number=np.stack(rows_number), dry_radius=np.stack(rows_radius))


@pytest.fixture(scope='module')
def designed_cases():
    return _designed_cases()


def _relerr(value, expected):
    value, expected = np.asarray(value), np.asarray(expected)
    denominator = np.maximum(np.abs(expected), 1e-300)
    return np.max(np.abs(value-expected)/denominator)


def test_repartition_all_against_native(oracle, designed_cases):
    native, _ = run_oracle(oracle, designed_cases, operation='repartition')
    errors = {}
    for function in (repartition_all, jax.jit(repartition_all)):
        mass, number = function(designed_cases['mass'], designed_cases['number'], designed_cases['dry_radius'])
        errors['mass'] = max(errors.get('mass', 0.), _relerr(mass, native['mass']))
        errors['number'] = max(errors.get('number', 0.), _relerr(number, native['number']))
        np.testing.assert_allclose(mass, native['mass'], rtol=1e-12, atol=1e-30)
        np.testing.assert_allclose(number, native['number'], rtol=1e-12, atol=1e-30)
    print('repartition_all max relative error per field:', errors)


def test_repartition_all_conserves_per_species_mass_and_total_number(designed_cases):
    mass, number = repartition_all(designed_cases['mass'], designed_cases['number'], designed_cases['dry_radius'])
    for species in range(5):  # SO4, BC, OC, SS, DU
        members = [jn for jn in range(18) if COMPONENT_SPECIES[jn] == species]
        before = designed_cases['mass'][..., members].sum(-1)
        after = mass[..., members].sum(-1)
        np.testing.assert_allclose(after, before, rtol=1e-14, atol=0)
    np.testing.assert_allclose(number.sum(-1), designed_cases['number'].sum(-1), rtol=1e-14, atol=0)
    # Insoluble components (and modes) are never touched.
    insoluble = [jn for jn in range(18) if COMPONENT_MODE[jn] >= 4]
    np.testing.assert_array_equal(mass[..., insoluble], designed_cases['mass'][..., insoluble])
    np.testing.assert_array_equal(number[..., 4:], designed_cases['number'][..., 4:])


def test_repartition_all_no_transfer_leaves_state_unchanged(designed_cases):
    row = 0  # every mode's mean particle mass below the transfer-gate threshold
    mass_row = designed_cases['mass'][row:row+1]
    number_row = designed_cases['number'][row:row+1]
    dry_radius_row = designed_cases['dry_radius'][row:row+1]
    mass, number = repartition_all(mass_row, number_row, dry_radius_row)
    np.testing.assert_allclose(mass, mass_row, rtol=1e-12)
    np.testing.assert_allclose(number, number_row, rtol=1e-12)


def test_repartition_all_straddling_mode_does_transfer(designed_cases):
    row = 1  # above threshold and straddling every mode's upper radius bound
    mass_row = designed_cases['mass'][row:row+1]
    number_row = designed_cases['number'][row:row+1]
    dry_radius_row = designed_cases['dry_radius'][row:row+1]
    mass, number = repartition_all(mass_row, number_row, dry_radius_row)
    # Every soluble species present in a transferring mode actually moved.
    for component in (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15):
        assert not np.isclose(mass[0, component], mass_row[0, component]), component
    assert not np.allclose(number[0, :4], number_row[0, :4])
