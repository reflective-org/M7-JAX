# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Run a configured sulfate M7 trajectory and compare every step with Fortran."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

from m7_jax.model import State, initialize, step, sulfur_inventory
from m7_jax.properties import load_kappa_table
from tools.m7_trajectory import build_trajectory, run_trajectory

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Configuration:
    steps: int = 600
    timestep_s: float = 1.
    temperature_k: float = 280.
    pressure_pa: float = 101325.
    relative_humidity: float = .5
    cloud_fraction: float = 0.
    initial_number_cm3: tuple = (100., 900., 0., 0.)
    initial_dry_diameter_nm: tuple = (3., 25., 200., 2000.)
    initial_h2so4_cm3: float = 1e8
    h2so4_production_cm3_s: float = 1e5
    coagulation: bool = True
    condensation: bool = True
    nucleation_scheme: int = 1
    water_scheme: int = 1

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError('steps must be a positive integer')
        for name in ('timestep_s','temperature_k','pressure_pa'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('relative_humidity','cloud_fraction'):
            if not np.isfinite(getattr(self,name)) or not 0 <= getattr(self,name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        for name in ('initial_h2so4_cm3','h2so4_production_cm3_s'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.nucleation_scheme not in (0,1) or self.water_scheme not in (0,1):
            raise ValueError('Supported nucleation/water switches are 0 or 1')
        if type(self.coagulation) is not bool or type(self.condensation) is not bool:
            raise ValueError('Process switches must be boolean')
        if not self.condensation and (self.nucleation_scheme or self.h2so4_production_cm3_s):
            raise ValueError('Condensation-disabled native core requires nucleation and production off')


def load_configuration(path):
    return Configuration(**tomllib.loads(Path(path).read_text()))


def jax_trajectory(config):
    jax.config.update('jax_enable_x64', True)
    initial = initialize(config.initial_number_cm3, config.initial_dry_diameter_nm,
                         config.initial_h2so4_cm3, water_scheme=config.water_scheme)
    table_path = ROOT/'reference/input/lut_kappa.nc'
    if not table_path.exists():
        table_path = ROOT/'lut_kappa.nc'
    table = load_kappa_table(table_path) if config.water_scheme == 1 else None
    def body(state, _):
        result = step(state,config.temperature_k,config.pressure_pa,config.relative_humidity,
            config.timestep_s,config.h2so4_production_cm3_s,config.cloud_fraction,table,
            coagulation=config.coagulation,condensation=config.condensation,
            nucleation_scheme=config.nucleation_scheme,water_scheme=config.water_scheme)
        return result.state, result
    result = jax.jit(lambda state: jax.lax.scan(body,state,None,length=config.steps))(initial)[1]
    result = jax.device_get(result)
    output = dict(zip(('sulfate','number','gas'),result.state))
    output.update(zip(('dry_radius','radius','density','water'),result.properties[:4]))
    if not np.all(result.valid):
        raise ValueError('Invalid input/state encountered during JAX trajectory')
    return initial, {key:np.asarray(value)[:,None,...] for key,value in output.items()}


def run(config, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    initial,actual=jax_trajectory(config)
    cases=dict(number=np.asarray(initial.number)[None,:],sulfate=np.asarray(initial.sulfate)[None,:],
        gas=np.asarray(initial.gas)[None],temperature=config.temperature_k,pressure=config.pressure_pa,
        relative_humidity=config.relative_humidity,cloud_fraction=config.cloud_fraction,timestep=config.timestep_s,
        production=config.h2so4_production_cm3_s,coagulation=config.coagulation,condensation=config.condensation,
        scheme=config.nucleation_scheme,water=config.water_scheme)
    checked,provenance=build_trajectory(output/'checked-fortran')
    optimized,optimized_provenance=build_trajectory(output/'optimized-fortran',checked=False)
    native,_=run_trajectory(checked,cases,steps=config.steps)
    fast,_=run_trajectory(optimized,cases,steps=config.steps)
    exact=all(np.array_equal(native[key],fast[key]) for key in native)
    comparisons={}
    for key,a in actual.items():
        f=native[key];difference=np.abs(a-f)
        relative=np.divide(difference,np.abs(f),out=np.zeros_like(f),where=f!=0)
        comparisons[key]={'passed':bool(np.allclose(a,f,rtol=3e-10,atol=0)),
            'maximum_absolute_error':float(difference.max()),
            'maximum_relative_error_nonzero':float(relative.max()),
            'nonzero_at_zero_reference':int(np.count_nonzero((f==0)&(a!=0)))}
    budgets={}
    source=config.h2so4_production_cm3_s*config.timestep_s*np.arange(1,config.steps+1)[:,None]
    expected=float(sulfur_inventory(initial))+source
    for label,values in [('jax',actual),('fortran',native)]:
        inventory=values['gas']+values['sulfate'].sum(-1)
        residual=np.abs(inventory-expected)/np.maximum(np.abs(expected),1.)
        budgets[label]={'maximum_relative_sulfur_residual':float(residual.max()),
                        'finite_nonnegative':all(bool(np.isfinite(v).all() and np.all(v>=0)) for v in values.values()),
                        'final_number_cm3':float(values['number'][-1].sum()),
                        'final_h2so4_cm3':float(values['gas'][-1,0]),
                        'insoluble_number_zero':bool(np.all(values['number'][...,4:]==0))}
    accepted=exact and all(row['passed'] for row in comparisons.values()) and all(
        row['maximum_relative_sulfur_residual']<=1e-10 and row['finite_nonnegative'] and row['insoluble_number_zero']
        for row in budgets.values())
    times=np.arange(1,config.steps+1)*config.timestep_s
    np.savez_compressed(output/'jax.npz',time_s=times,**actual)
    np.savez_compressed(output/'fortran.npz',time_s=times,**{k:native[k] for k in actual})
    for label,values in [('jax',actual),('fortran',native)]:
        np.savetxt(output/f'{label}-timeseries.csv',np.column_stack((times,values['number'][:,0],values['gas'][:,0],values['sulfate'][:,0])),
            delimiter=',',comments='',header='time_s,'+','.join(f'number_mode{i+1}_cm3' for i in range(7))+',h2so4_cm3,'+','.join(f'sulfate_mode{i+1}_cm3' for i in range(4)))
    report={'configuration':asdict(config),'accepted':bool(accepted),'comparison_rtol':3e-10,'comparison_atol':0,
        'budget_rtol':1e-10,'checked_optimized_exact':exact,'comparison':comparisons,'budgets':budgets,
        'checked_provenance':provenance,'optimized_provenance':optimized_provenance,
        'platform':platform.platform(),'jax':jax.__version__,'numpy':np.__version__,'dtype':'float64',
        'initial_state':{key:np.asarray(value).tolist() for key,value in zip(('sulfate','number','gas'),initial)},
        'scope':'Pure sulfate, four soluble lognormal modes plus three empty insoluble modes; binary nucleation only; original core, not HAM driver',
        'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [*sorted((ROOT/'src/m7_jax').glob('*.py')),Path(__file__),ROOT/'tools/m7_oracle.py',ROOT/'tools/m7_trajectory.py']}}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    report=run(load_configuration(options.config),options.output)
    print(json.dumps({k:report[k] for k in ('accepted','comparison','budgets')},indent=2))
    raise SystemExit(0 if report['accepted'] else 1)
