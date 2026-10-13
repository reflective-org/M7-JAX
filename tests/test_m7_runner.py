# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Configuration and persisted coupled-trajectory evidence."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from tools.run_m7_sulfate import Configuration,run


@pytest.mark.parametrize('changes',[{'steps':0},{'steps':1.5},{'timestep_s':0},{'temperature_k':float('nan')},
    {'pressure_pa':-1},{'relative_humidity':1.1},{'cloud_fraction':-1},{'initial_h2so4_cm3':-1},
    {'h2so4_production_cm3_s':-1},{'coagulation':1},{'water_scheme':2},{'nucleation_scheme':2},
    {'condensation':False}])
def test_configuration_rejects_unphysical_or_unsupported_inputs(changes):
    with pytest.raises(ValueError):
        replace(Configuration(),**changes)


def test_saved_native_trajectory_units_hashes_and_budget(tmp_path):
    report=run(Configuration(steps=3),tmp_path/'run')
    assert report['accepted']
    assert report['checked_optimized_exact']
    assert json.loads((tmp_path/'run/report.json').read_text())['accepted']
    for name,expected in report['files_sha256'].items():
        assert hashlib.sha256((tmp_path/'run'/name).read_bytes()).hexdigest()==expected
    with np.load(tmp_path/'run/jax.npz') as data:
        np.testing.assert_array_equal(data['time_s'],[1.,2.,3.])
        assert data['number'].shape==(3,1,7)
        assert data['sulfate'].shape==(3,1,4)
        table=np.loadtxt(tmp_path/'run/jax-timeseries.csv',delimiter=',',skiprows=1)
        np.testing.assert_array_equal(table[:,1:8],data['number'][:,0])
        np.testing.assert_array_equal(table[:,8],data['gas'][:,0])
    with pytest.raises(FileExistsError):
        run(Configuration(steps=3),tmp_path/'run')
