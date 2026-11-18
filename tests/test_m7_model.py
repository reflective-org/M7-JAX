# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Native hydration, redistribution and composed sulfate trajectories."""
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from m7_jax.model import State, initialize, step, sulfur_inventory, validate_state, STATIC_OPTIONS
from m7_jax.properties import properties, load_kappa_table, Properties
from m7_jax.redistribution import repartition
from tools.m7_oracle import build_oracle, run_oracle, reference_cases

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    return build_oracle(tmp_path_factory.mktemp('m7-core'))[0]


@pytest.fixture(scope='module')
def table():
    path = ROOT/'reference/input/lut_kappa.nc'
    return load_kappa_table(path if path.exists() else ROOT/'lut_kappa.nc')


@pytest.mark.parametrize('water', [0, 1])
def test_properties_against_native(oracle, table, water):
    data = reference_cases()
    # Zero previous geometry, as in the complete native call.
    data.update(radius=np.zeros((3,7)),density=np.zeros((3,7)),dry_radius=np.zeros((3,4)),water=water)
    native, _ = run_oracle(oracle, data, operation='hydrate')
    args = (data['sulfate'],data['number'],data['temperature'],data['relative_humidity'],table)
    call = partial(properties, water_scheme=water, pressure=data['pressure'])
    for function in (call, jax.jit(call)):
        result = function(*args)
        for value, key in zip(result, ('dry_radius','radius','density','water','mean_particle_mass')):
            expected = native[key][:,:4] if key=='mean_particle_mass' else native[key]
            np.testing.assert_allclose(value, expected, rtol=3e-13, atol=1e-30)


def test_repartition_original_upward_transfers_and_conservation(oracle):
    # Modal centers below, across and far above the upper boundaries.
    centers = [(1.,10.,100.,1000.),(4.,40.,400.,1000.),(100.,1000.,10000.,1000.)]
    states=[initialize([1e4,1e3,1e2,1],np.asarray(x)*2,water_scheme=0) for x in centers]
    data=dict(number=np.stack([s.number for s in states]),sulfate=np.stack([s.sulfate for s in states]),
              dry_radius=np.asarray(centers)*1e-7)
    native, _ = run_oracle(oracle,data,operation='repartition')
    for function in (repartition,jax.jit(repartition)):
        mass,number=function(data['sulfate'],data['number'],data['dry_radius'])
        np.testing.assert_allclose(mass,native['sulfate'],rtol=4e-13,atol=0)
        np.testing.assert_allclose(number,native['number'],rtol=4e-13,atol=0)
        np.testing.assert_allclose(mass.sum(-1),data['sulfate'].sum(-1),rtol=3e-15)
        np.testing.assert_allclose(number.sum(-1),data['number'].sum(-1),rtol=3e-15)


@pytest.mark.parametrize('water',[0,1])
@pytest.mark.parametrize('coag,cond,nucl',[(False,False,0),(True,False,0),(False,True,0),(True,True,0),(False,True,1),(True,True,1)])
def test_full_native_process_switches(oracle, table, water, coag, cond, nucl):
    data=reference_cases()
    data.update(coagulation=coag,condensation=cond,scheme=nucl,water=water)
    native,_=run_oracle(oracle,data)
    state=State(*(jnp.asarray(data[key]) for key in ('sulfate','number','gas')))
    kwargs=dict(coagulation=coag,condensation=cond,nucleation_scheme=nucl,water_scheme=water)
    call=jax.jit(step,static_argnames=STATIC_OPTIONS)
    result=call(state,data['temperature'],data['pressure'],data['relative_humidity'],data['timestep'],
        data['production'],0.,table,**kwargs)
    assert np.all(result.valid)
    for value,key in zip((*result.state,*result.properties[:4]),('sulfate','number','gas','dry_radius','radius','density','water')):
        np.testing.assert_allclose(value,native[key],rtol=3e-11,atol=1e-25,err_msg=key)
    expected=sulfur_inventory(state)+(data['production']*data['timestep'] if cond else 0)
    np.testing.assert_allclose(sulfur_inventory(result.state),expected,rtol=5e-14)


@pytest.mark.parametrize('water',[0,1])
def test_repeated_coupled_native_steps(oracle, table, water):
    data=reference_cases();data.update(water=water,scheme=1,timestep=1.)
    state=State(*(jnp.asarray(data[key]) for key in ('sulfate','number','gas')))
    def advance(_, state):
        return step(state,data['temperature'],data['pressure'],data['relative_humidity'],1.,
            data['production'],0.,table,water_scheme=water).state
    final=jax.jit(lambda s: jax.lax.fori_loop(0,100,advance,s))(state)
    native,_=run_oracle(oracle,data,steps=100)
    for value,key in zip(final,('sulfate','number','gas')):
        np.testing.assert_allclose(value,native[key],rtol=3e-10,atol=1e-20,err_msg=key)
    np.testing.assert_allclose(sulfur_inventory(final),sulfur_inventory(state)+100*data['production'],rtol=1e-12)


def test_public_state_and_undefined_switches_rejected():
    state=initialize()
    with pytest.raises(ValueError,match='insoluble'):
        validate_state(state._replace(number=state.number.at[4].set(1.)))
    with pytest.raises(ValueError,match='nonnegative'):
        validate_state(state._replace(gas=jnp.array(-1.)))
    with pytest.raises(ValueError,match='sink undefined'):
        step(state,280.,101325.,.5,1.,condensation=False)
    with pytest.raises(ValueError,match='Only native'):
        step(state,280.,101325.,.5,1.,nucleation_scheme=2)


@pytest.mark.parametrize('key,value', [('temperature',np.nan),('humidity',np.nan),('humidity',1.1),
    ('timestep',0.),('pressure',-1.),('cloud_fraction',2.),('production',-1.)])
def test_invalid_environment_cannot_pass_validity(table,key,value):
    inputs=dict(temperature=280.,pressure=101325.,humidity=.5,timestep=1.,cloud_fraction=0.,production=0.)
    inputs[key]=value
    result=step(initialize(),table=table,coagulation=False,condensation=False,nucleation_scheme=0,**inputs)
    assert not bool(result.valid)


def test_cumulative_normal_native_branches_and_cutoff(oracle):
    from m7_jax.redistribution import cumulative_normal
    x=np.array([-40.,-9.,-8.,-5.656854249,-5.656854248,-.662910001,-.66291,-1e-17,0.,
                 1e-17,.66291,.662910001,5.656854248,5.656854249,8.,9.,40.])
    native,_=run_oracle(oracle,dict(gas=x),operation='cumulative_normal')
    for fn in (cumulative_normal,jax.jit(cumulative_normal)):
        cdf,tail=fn(x)
        np.testing.assert_allclose(cdf,native['cdf'],rtol=3e-15,atol=0)
        np.testing.assert_allclose(tail,native['complement'],rtol=3e-15,atol=0)


def test_constants_match_initialized_fortran(oracle):
    import json
    from m7_jax import constants as c
    native=json.loads((oracle.parent/'provenance.json').read_text())['compiled_constants']
    pairs={'avo':c.AVOGADRO,'bk':c.BK,'rerg':c.RERG,'mw_so4':c.SO4_MOLAR_MASS,
           'wh2so4':c.H2SO4_MOLAR_MASS,'dh2so4':c.SULFATE_DENSITY,'critn':c.CRITICAL_MOLECULES,
           'sigma':c.SIGMA,'sigmaln':c.SIGMA_LOG,'cmr2ram':c.CMR2RAM,'ram2cmr':c.RAM2CMR,
           'caccso4':c.ACCOMMODATION,'crdiv':c.MODE_RADIUS_BOUNDS}
    for key,value in pairs.items():
        np.testing.assert_allclose(value,native[key],rtol=3e-16,atol=0,err_msg=key)
