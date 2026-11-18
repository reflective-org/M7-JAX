# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""PARNUC lookup tables for the Kazil and Lovejoy (``scheme=2``) parity tests.

Two sources feed the same tests (``KAZIL_TABLE_SOURCES``):

``'synthetic'``
    A small table written at test time with the layout of the staged
    ``parnuc.15H2SO4.A0.total.nc``: the same variable and dimension names,
    ``pfr`` over the axis variables' own dimensions in file order, float64
    axes and a float32 ``pfr`` holding ln(formation rate). Its values are an
    arbitrary smooth function, not physics. The table exists only so the JAX
    port and the compiled native ``nucl_kazil_lovejoy`` can be compared on
    identical table data when the real file is absent. Never use it for a
    physical number.
``'parnuc'``
    The real MPI-M/HAMMOZ table, found through ``HAM_INPUT_DIR`` and never
    shipped. Tests on this source skip without it.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

# ECHAM opens the table as 'parnuc.15H2SO4.nc'; the staged distribution file
# is 'parnuc.15H2SO4.A0.total.nc' (see docs/validation.md). Both are
# accepted, in that order.
KAZIL_TABLE_NAMES = ('parnuc.15H2SO4.nc', 'parnuc.15H2SO4.A0.total.nc')

# Variable -> (dimensions, NetCDF type code) of the staged real table. The
# synthetic table reproduces it, and a data-gated test checks it against the
# real file.
PARNUC_LAYOUT = {
    'temperature': (('temperature',), 'd'),
    'RH': (('RH',), 'd'),
    '[H2SO4]': (('[H2SO4]',), 'd'),
    'ionization': (('ionization',), 'd'),
    'condensation_sink': (('condensation_sink',), 'd'),
    'pfr': (('temperature', 'RH', '[H2SO4]', 'ionization', 'condensation_sink'), 'f'),
}

# Synthetic axes in file order: (name, units, ascending nodes). The axes are
# unevenly spaced and of five different lengths, so a transposed or
# misindexed table cannot match the native routine. Each range sits just
# inside the Kazil case matrix of tests/test_m7_nucleation.py (T 190-300 K,
# RH 5-100 %, [H2SO4] 1e4-1e10 cm-3, ion pairs 0-50 cm-3 s-1, sink 1e-5-1e-1
# s-1): most cases interpolate, and the cases at the matrix bounds fall
# outside both ends of every axis, so every native clamp and set-to-zero
# branch runs.
SYNTHETIC_AXES = (
    ('temperature', 'K', (193., 205., 222., 244., 269., 297.)),
    ('RH', '%', (6.5, 16., 34., 61., 98.)),
    ('[H2SO4]', 'cm-3', (2e4, 2e5, 1.8e6, 1.5e7, 1.2e8, 1e9, 7e9)),
    ('ionization', 'cm-3 s-1', (.6, 3., 11., 47.)),
    ('condensation_sink', 's-1', (1.3e-5, 5e-5, 1.4e-4, 4e-4, 1.1e-3, 3.5e-3, 1.5e-2, 8e-2)),
)


def real_kazil_table_path():
    """The real PARNUC file under ``HAM_INPUT_DIR``, or None."""
    base = os.environ.get('HAM_INPUT_DIR')
    if not base:
        return None
    for name in KAZIL_TABLE_NAMES:
        candidate = Path(base) / name
        if candidate.exists():
            return candidate
    return None


def synthetic_log_pfr(temperature, relative_humidity, h2so4, ion_pair_rate, sink):
    """Synthetic ln(formation rate [cm-3 s-1]) on broadcast coordinates.

    Units follow the table axes: K, %, cm-3, cm-3 s-1 and s-1. The function
    is smooth but not separable in the log coordinates (pairwise products and
    a sine of a mixed argument), so all 32 hypercube corners carry distinct
    weights. On the synthetic nodes it spans about -14 to 21; the real table
    reaches 21 at its upper end.
    """
    t = (np.asarray(temperature) - 245.) / 50.
    r = np.log(np.asarray(relative_humidity) / 30.)
    h = np.log(np.asarray(h2so4) / 1e7)
    i = np.log(np.asarray(ion_pair_rate) / 5.)
    c = np.log(np.asarray(sink) / 1e-3)
    return (1.5 + 1.6*h - 2.5*t + .9*r + .3*i - .5*c
            + .12*h*r - .35*h*t + .1*i*c - .2*t*r - .03*h*h
            + .6*np.sin(.7*h - 1.3*t + .5*r + .2*i*c))


def write_synthetic_kazil_table(path):
    """Write the synthetic PARNUC table to ``path`` (NetCDF classic)."""
    with netcdf_file(path, 'w') as nc:
        nc.info = 'Synthetic PARNUC layout for M7-JAX native parity tests; not physical data.'
        for name, units, nodes in SYNTHETIC_AXES:
            nc.createDimension(name, len(nodes))
            variable = nc.createVariable(name, PARNUC_LAYOUT[name][1], (name,))
            variable[:] = nodes
            variable.units = units
        pfr = nc.createVariable('pfr', PARNUC_LAYOUT['pfr'][1], PARNUC_LAYOUT['pfr'][0])
        pfr[:] = synthetic_log_pfr(*np.meshgrid(*(nodes for _, _, nodes in SYNTHETIC_AXES),
                                                 indexing='ij'))
        pfr.units = 'cm-3 s-1'
        pfr.description = 'Nat. log. of a synthetic particle formation rate'
    return path


requires_parnuc = pytest.mark.skipif(
    real_kazil_table_path() is None,
    reason=(
        "Kazil/Lovejoy PARNUC table not found: set HAM_INPUT_DIR to a "
        f"directory containing one of {KAZIL_TABLE_NAMES} "
        "(docs/validation.md has the exact command)"
    ),
)

# Parameters for a module-scoped table fixture: the synthetic table always
# runs; the real one runs only when HAM_INPUT_DIR names it.
KAZIL_TABLE_SOURCES = ('synthetic', pytest.param('parnuc', marks=requires_parnuc))


def kazil_table_file(source, tmp_path_factory):
    """The table file for one ``KAZIL_TABLE_SOURCES`` entry."""
    if source == 'synthetic':
        return write_synthetic_kazil_table(
            tmp_path_factory.mktemp('parnuc-synthetic') / 'parnuc.synthetic.nc')
    return real_kazil_table_path()
