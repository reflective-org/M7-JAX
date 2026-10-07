"""Composition of the native M7 call for sulfuric acid in four soluble modes."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .constants import PI, AVOGADRO, SO4_MOLAR_MASS, H2SO4_MOLAR_MASS, SULFATE_DENSITY, CMR2RAM
from .properties import properties, Properties, average_properties, kappa_hydration, AverageProperties
from .condensation import condensation_sink, produce_and_condense
from .nucleation import nucleate, nucleate_all
from .coagulation import dnum, dnum_all
from .redistribution import repartition, repartition_all


class State(NamedTuple):
    sulfate: object  # molecules/cm3, trailing four soluble modes
    number: object   # particles/cm3, trailing seven modes; last three zero
    gas: object      # H2SO4 molecules/cm3, no trailing mode axis


class StepResult(NamedTuple):
    state: State
    properties: Properties
    valid: object


STATIC_OPTIONS = ('coagulation', 'condensation', 'nucleation_scheme', 'water_scheme', 'cancellation_free')


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
         *, coagulation=True, condensation=True, nucleation_scheme=1, water_scheme=1,
         cancellation_free=None):
    """Native order: properties → condensation → nucleation → coagulation →
    properties → redistribution → properties. All concentrations use cm-3.

    Condensation-disabled runs do not apply production, matching native M7.
    Nucleation with condensation disabled is rejected: native core leaves its
    sink uninitialized. The independent nucleate() API accepts an explicit sink.
    ``cancellation_free`` (static ``None``/``True``/``False``) selects the
    expression forms as in :func:`step_all`; this float64 API's ``None`` is
    the native forms.
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
        condensed = produce_and_condense(gas, production, sulfate, sink, sinks.modal, cloud_fraction, timestep,
                                         cancellation_free=cancellation_free)
        sulfate, gas = condensed.sulfate, condensed.gas
    delta = jnp.zeros_like(sulfate)
    new_number = jnp.zeros_like(gas)
    if nucleation_scheme:
        result = nucleate(gas, temperature, humidity, sink, cloud_fraction, timestep, scheme=nucleation_scheme,
                          cancellation_free=cancellation_free)
        gas, new_number = result.gas, result.number
        delta = delta.at[..., 0].set(result.sulfate)
    if nucleation_scheme or coagulation:
        sulfate, number, _ = dnum(sulfate, number, temperature, pressure, prop.wet_radius, prop.density,
                                  new_number, delta, timestep, coagulation=coagulation,
                                  cancellation_free=cancellation_free)
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


class FullState(NamedTuple):
    mass: object    # (...,18) HAM aerocomp order; SO4 molecules/cm3, others ug/m3
    number: object  # (...,7) particles/cm3, NS KS AS CS KI AI CI
    gas: object     # H2SO4 molecules/cm3 at the start of the step (zgso4)


class FullProperties(NamedTuple):
    dry_radius: object  # (...,4) cm, soluble count-median dry radii
    wet_radius: object  # (...,7) cm; insoluble modes carry their dry radius
    density: object     # (...,7) g/cm3
    water: object       # (...,4) kg/m3 air


class FullStepResult(NamedTuple):
    state: FullState
    properties: FullProperties
    valid: object


FULL_STATIC_OPTIONS = ('coagulation', 'condensation', 'nucleation_scheme', 'organic_scheme',
                       'cancellation_free')


def _full_properties(mass, number, temperature, humidity, table, previous):
    avg = average_properties(mass, number, previous=previous)
    hyd = kappa_hydration(avg.pttn, number, temperature, humidity, table, previous=avg)
    return avg, hyd


def _full_input_valid(mass, number, gas, temperature, pressure, humidity, timestep, production,
                      cloud_fraction, forest_fraction, in_pbl, ion_pair_rate):
    """Per-cell domain of every ``step_all`` input, so that an invalid input
    flags its cell even where the calculation hides it (clamped lookups and
    MIN/MAX limiters can map a NaN or out-of-range input to a finite
    output, so checking the outputs alone is not enough). The same checks as :func:`step` for the shared inputs;
    in addition the ion-pair rate is finite and non-negative, the forest
    fraction lies in [0, 1] and ``in_pbl`` is a truth value (0 or 1, the
    host's ``jk >= int(ppbl)``). Like ``step``'s check of ``production``,
    each input is checked whether or not the selected switches read it.
    """
    valid = jnp.ones(gas.shape, dtype=bool)
    for value in (mass, number):
        valid = valid & jnp.all(jnp.isfinite(value) & (value >= 0), axis=-1)
    valid = valid & jnp.isfinite(gas) & (gas >= 0)
    for value in (temperature, pressure, timestep):
        value = jnp.asarray(value)
        valid = valid & jnp.isfinite(value) & (value > 0)
    for value in (humidity, cloud_fraction, forest_fraction):
        value = jnp.asarray(value)
        valid = valid & jnp.isfinite(value) & (value >= 0) & (value <= 1)
    for value in (production, ion_pair_rate):
        value = jnp.asarray(value)
        valid = valid & jnp.isfinite(value) & (value >= 0)
    in_pbl = jnp.asarray(in_pbl)
    return valid & ((in_pbl == 0) | (in_pbl == 1))


def step_all(state, temperature, pressure, humidity, timestep, production=0., cloud_fraction=0.,
             table=None, forest_fraction=0., in_pbl=False, *, coagulation=True, condensation=True,
             nucleation_scheme=1, organic_scheme=0, ion_pair_rate=0., kazil_lovejoy_table=None,
             cancellation_free=None):
    """The complete native ``m7`` call on the 18-component HAM M7 state.

    Native order (mo_ham_m7.F90 ``m7``): mean properties and kappa hydration
    -> H2SO4 condensation sink and analytical production/condensation ->
    nucleation (H2SO4/H2O and organic) -> coagulation, insoluble ageing
    (``m7_dnum``/``m7_delcoa``/``m7_concoag``) -> properties -> upward modal
    redistribution (``m7_dconc``) -> properties. Only ``nwater = 1`` (the kappa
    lookup) is supported for the mixed state. ``humidity`` is the clear-sky
    relative humidity the host computes; ``production`` the H2SO4 production
    rate [molecules cm-3 s-1]; ``in_pbl`` the host's per-cell test
    ``jk >= int(ppbl)`` used by organic nucleation. ``nucleation_scheme=2``
    (Kazil and Lovejoy 2007) additionally needs ``ion_pair_rate`` [cm-3 s-1]
    and ``kazil_lovejoy_table`` (see :func:`nucleation.nucleate_all`).

    ``cancellation_free`` (static): ``None`` (default) uses the native M7
    expressions in float64 and their algebraically identical
    cancellation-free forms in narrower dtypes; ``True`` uses the
    cancellation-free forms in any dtype (in float64 they differ from the
    native forms only by the native forms' own cancellation error); ``False``
    uses the native forms in any dtype. See :mod:`m7_jax.precision` and
    ``docs/model.md``.

    ``valid`` is False for a cell whose inputs leave the native domain
    (non-finite or negative state; non-positive temperature, pressure or
    timestep; humidity, cloud or forest fraction outside [0, 1]; negative
    production or ion-pair rate; ``in_pbl`` not 0/1) or whose outputs are
    non-finite or negative.

    Dtype-generic (#1017 W1 task 3, forward only -- no gradient requirement
    in float32): computes in whichever float dtype ``state`` (and ``table``)
    carry. Call with ``jax_enable_x64`` left off and float32 inputs/table for
    a float32-forward run (``kazil_lovejoy_table``/``ion_pair_rate`` must
    then be float32 too -- the caller's job, e.g. the jcm adapter's
    ``core_dtype``); float64 behaviour (``jax_enable_x64`` on) is unchanged
    and bit-identical to before this dtype generalisation.
    """
    if nucleation_scheme not in (0, 1, 2):
        raise ValueError('Supported M7 sulfur nucleation schemes are 0, 1 and 2')
    if organic_scheme not in (0, 1, 2):
        raise ValueError('Supported M7 organic nucleation schemes are 0, 1 and 2')
    if (nucleation_scheme or organic_scheme) and not condensation:
        raise ValueError('Native M7 leaves the sink undefined when nucleation is on and condensation off')
    if table is None:
        raise ValueError('The mixed-state core requires the kappa table (nwater = 1)')
    # Dtype-generic (#1017 W1 task 3): unlike `step`/`initialize` (the
    # pure-sulfate legacy path, unchanged), `step_all` and everything it
    # calls compute in whichever float dtype `state`/`table` carry, so it
    # runs float32-forward with `jax_enable_x64` left off (the table must
    # then also be float32 -- the caller's job, e.g. the jcm adapter's
    # `core_dtype`) and is bit-identical to before under x64 float64.
    mass, number, gas = (jnp.asarray(x) for x in state)
    temperature, pressure, humidity = (jnp.asarray(x) for x in (temperature, pressure, humidity))
    input_valid = _full_input_valid(mass, number, gas, temperature, pressure, humidity, timestep, production,
                                    cloud_fraction, forest_fraction, in_pbl, ion_pair_rate)
    zero7 = jnp.zeros(number.shape, dtype=number.dtype)
    # m7 zeroes pm6rp/prhop on entry, so the INOUT "previous" geometry is zero.
    start = AverageProperties(jnp.zeros(mass.shape, dtype=mass.dtype), zero7, zero7)
    avg, hyd = _full_properties(mass, number, temperature, humidity, table, start)
    sink = jnp.zeros_like(gas)
    insoluble_sulfate = jnp.zeros(gas.shape + (3,), dtype=gas.dtype)
    if condensation:
        sinks = condensation_sink(number, hyd.wet_radius, temperature, pressure)
        sink = sinks.total
        condensed = produce_and_condense(gas, production, mass[..., :4], sink, sinks.modal,
                                         cloud_fraction, timestep, cancellation_free=cancellation_free)
        gas = condensed.gas
        mass = mass.at[..., :4].set(condensed.sulfate)
        insoluble_sulfate = condensed.insoluble_sulfate
    new_number = jnp.zeros_like(gas)
    delta = jnp.zeros_like(mass)
    if nucleation_scheme or organic_scheme:
        nuc = nucleate_all(gas, temperature, humidity, sink, cloud_fraction, forest_fraction,
                           in_pbl, timestep, scheme=nucleation_scheme, organic_scheme=organic_scheme,
                           ion_pair_rate=ion_pair_rate, kazil_lovejoy_table=kazil_lovejoy_table,
                           cancellation_free=cancellation_free)
        gas, new_number = nuc.gas, nuc.number
        delta = delta.at[..., 0].set(nuc.sulfate)
    if nucleation_scheme or organic_scheme or coagulation:
        mass, number, _ = dnum_all(mass, number, temperature, pressure, hyd.wet_radius, hyd.density,
                                   insoluble_sulfate, new_number, delta, avg.pttn, timestep,
                                   coagulation=coagulation, cancellation_free=cancellation_free)
    avg, hyd = _full_properties(mass, number, temperature, humidity, table, avg)
    if condensation or coagulation:
        mass, number = repartition_all(mass, number, hyd.dry_radius)
    avg, hyd = _full_properties(mass, number, temperature, humidity, table, avg)
    valid = input_valid & (jnp.all(jnp.isfinite(mass) & (mass >= 0), axis=-1)
             & jnp.all(jnp.isfinite(number) & (number >= 0), axis=-1)
             & jnp.isfinite(gas) & (gas >= 0))
    for value in hyd:
        valid = valid & jnp.all(jnp.isfinite(value), axis=-1)
    return FullStepResult(FullState(mass, number, gas),
                          FullProperties(hyd.dry_radius, hyd.wet_radius, hyd.density, hyd.water), valid)
