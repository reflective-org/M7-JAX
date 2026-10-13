# Copyright 2026 Reflective
# Portions derived from HAMMOZ M7, Copyright (c) 2021 hammoz;
# see NOTICE and LICENSES/BSD-3-Clause-HAMMOZ.txt.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
"""Composition of the native M7 call for sulfuric acid in four soluble modes."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .constants import PI, AVOGADRO, SO4_MOLAR_MASS, H2SO4_MOLAR_MASS, SULFATE_DENSITY, CMR2RAM
from .properties import properties, Properties
from .condensation import condensation_sink, produce_and_condense
from .nucleation import nucleate
from .coagulation import dnum
from .redistribution import repartition


class State(NamedTuple):
    sulfate: object  # molecules/cm3, trailing four soluble modes
    number: object   # particles/cm3, trailing seven modes; last three zero
    gas: object      # H2SO4 molecules/cm3, no trailing mode axis


class StepResult(NamedTuple):
    state: State
    properties: Properties
    valid: object


STATIC_OPTIONS = ('coagulation', 'condensation', 'nucleation_scheme', 'water_scheme')


def validate_state(state):
    """Host-side entry check; call once before a compiled trajectory."""
    mass, number, gas = map(np.asarray, state)
    if mass.shape[-1:] != (4,) or number.shape != mass.shape[:-1]+(7,) or gas.shape != mass.shape[:-1]:
        raise ValueError('Require sulfate (...,4), number (...,7), gas (...) with matching cells')
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in (mass, number, gas)):
        raise ValueError('State must be finite and nonnegative')
    if np.any(number[..., 4:] != 0):
        raise ValueError('Pure-sulfate M7 supports empty insoluble modes only')


def initialize(number_cm3=(100., 900., 0., 0.), dry_diameter_nm=(3., 50., 200., 2000.),
               gas_cm3=1e8, *, water_scheme=1):
    """Construct modal lognormals from count-median dry diameters, not bin edges."""
    if not jax.config.jax_enable_x64:
        raise ValueError('Enable jax_enable_x64 before constructing M7 state')
    n = np.asarray(number_cm3, dtype=np.float64)
    diameter = np.asarray(dry_diameter_nm, dtype=np.float64)
    if n.shape != (4,) or diameter.shape != (4,) or not np.isfinite(diameter).all() or np.any(diameter <= 0):
        raise ValueError('Supply four soluble-mode numbers and four positive finite diameters')
    if water_scheme not in (0, 1):
        raise ValueError('water_scheme must be 0 or 1')
    weight = SO4_MOLAR_MASS if water_scheme == 1 else H2SO4_MOLAR_MASS
    radius = diameter*.5e-7
    volume = (4*PI/3)*(radius*np.asarray(CMR2RAM[:4]))**3
    state = State(jnp.asarray(n*volume*SULFATE_DENSITY*AVOGADRO/weight),
                  jnp.asarray(np.r_[n, 0., 0., 0.]), jnp.asarray(gas_cm3, dtype=jnp.float64))
    validate_state(state)
    return state


def sulfur_inventory(state):
    return jnp.sum(state.sulfate, axis=-1)+state.gas


def step(state, temperature, pressure, humidity, timestep, production=0., cloud_fraction=0., table=None,
         *, coagulation=True, condensation=True, nucleation_scheme=1, water_scheme=1):
    """Native order: properties → condensation → nucleation → coagulation →
    properties → redistribution → properties. All concentrations use cm-3.

    Condensation-disabled runs do not apply production, matching native M7.
    Nucleation with condensation disabled is rejected: native core leaves its
    sink uninitialized. The independent nucleate() API accepts an explicit sink.
    """
    if nucleation_scheme not in (0, 1):
        raise ValueError('Only native off (0) and binary sulfate nucleation (1) are supported')
    if nucleation_scheme and not condensation:
        raise ValueError('Native M7 leaves the sink undefined when nucleation is on and condensation off; use nucleate with an explicit sink')
    if not jax.config.jax_enable_x64:
        raise ValueError('Enable jax_enable_x64 before calling M7')
    input_valid = jnp.ones_like(state.gas, dtype=bool)
    for value in state[:2]:
        input_valid = input_valid & jnp.all(jnp.isfinite(value) & (value >= 0), axis=-1)
    input_valid = input_valid & jnp.isfinite(state.gas) & (state.gas >= 0)
    input_valid = input_valid & jnp.all(state.number[..., 4:] == 0, axis=-1)
    for value in (temperature, pressure, timestep):
        value = jnp.asarray(value)
        input_valid = input_valid & jnp.isfinite(value) & (value > 0)
    for value in (humidity, cloud_fraction):
        value = jnp.asarray(value)
        input_valid = input_valid & jnp.isfinite(value) & (value >= 0) & (value <= 1)
    input_valid = input_valid & jnp.isfinite(production) & (jnp.asarray(production) >= 0)
    sulfate, number, gas = state
    prop = properties(sulfate, number, temperature, humidity, table,
                      water_scheme=water_scheme, pressure=pressure)
    sink = jnp.zeros_like(gas)
    if condensation:
        sinks = condensation_sink(number, prop.wet_radius, temperature, pressure)
        sink = sinks.total
        condensed = produce_and_condense(gas, production, sulfate, sink, sinks.modal, cloud_fraction, timestep)
        sulfate, gas = condensed.sulfate, condensed.gas
    delta = jnp.zeros_like(sulfate)
    new_number = jnp.zeros_like(gas)
    if nucleation_scheme:
        result = nucleate(gas, temperature, humidity, sink, cloud_fraction, timestep, scheme=nucleation_scheme)
        gas, new_number = result.gas, result.number
        delta = delta.at[..., 0].set(result.sulfate)
    if nucleation_scheme or coagulation:
        sulfate, number, _ = dnum(sulfate, number, temperature, pressure, prop.wet_radius, prop.density,
                                  new_number, delta, timestep, coagulation=coagulation)
    prop = properties(sulfate, number, temperature, humidity, table,
                      water_scheme=water_scheme, pressure=pressure, previous=prop)
    if condensation or coagulation:
        sulfate, number = repartition(sulfate, number, prop.dry_radius)
    prop = properties(sulfate, number, temperature, humidity, table,
                      water_scheme=water_scheme, pressure=pressure, previous=prop)
    valid = input_valid & (jnp.all(jnp.isfinite(sulfate) & (sulfate >= 0), axis=-1)
             & jnp.all(jnp.isfinite(number) & (number >= 0), axis=-1)
             & jnp.all(number[..., 4:] == 0, axis=-1)
             & jnp.isfinite(gas) & (gas >= 0))
    for value in prop:
        valid = valid & jnp.all(jnp.isfinite(value), axis=-1)
    return StepResult(State(sulfate, number, gas), prop, valid)
