"""Composition of the native M7 call for sulfuric acid in four soluble modes."""
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .constants import PI, AVOGADRO, SO4_MOLAR_MASS, H2SO4_MOLAR_MASS, SULFATE_DENSITY, CMR2RAM
from .properties import (
    properties, Properties, prepare_kappa_table, _kappa_th_offset, _average_pttn_soa,
    _insoluble_geometry_soa, _soluble_dry_radius_soa, _kappa_hydration_soa, _pure_sulfate_dry_radius,
)
from .condensation import (
    condensation_sink, produce_and_condense, _condensation_sink_soa, _produce_and_condense_soa,
)
from .nucleation import nucleate, nucleate_all, prepare_kazil_lovejoy_table
from .coagulation import dnum, _dnum_all_soa
from .redistribution import repartition, _repartition_all_soa


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
    if water_scheme == 1:
        if condensation or coagulation:
            dry_mid = _pure_sulfate_dry_radius(sulfate, number)
            sulfate, number = repartition(sulfate, number, dry_mid)
        prop = properties(sulfate, number, temperature, humidity, table,
                          water_scheme=water_scheme, pressure=pressure, previous=prop)
    else:
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


class SoAState(NamedTuple):
    """Cell-minor tuple-of-arrays M7 state (18 mass fields, 7 number fields, 1 gas field)."""
    mass: tuple
    number: tuple
    gas: object


class SoAProperties(NamedTuple):
    """Cell-minor tuple-of-arrays modal properties (4 dry radii, 7 wet radii, 7 densities, 4 waters)."""
    dry_radius: tuple
    wet_radius: tuple
    density: tuple
    water: tuple


class SoAStepResult(NamedTuple):
    state: SoAState
    properties: SoAProperties
    valid: object


FULL_STATIC_OPTIONS = ('coagulation', 'condensation', 'nucleation_scheme', 'organic_scheme',
                       'cancellation_free', 'chunk_size', 'remat_chunks')
SOA_STATIC_OPTIONS = FULL_STATIC_OPTIONS + ('compute_valid', 'compute_water')


def _all_finite_nonnegative(valid, arrays):
    for arr in arrays:
        valid = valid & jnp.isfinite(arr) & (arr >= 0)
    return valid


def _full_input_valid(cell_shape, mass18, number7, gas, temperature, pressure, humidity, timestep,
                      production, cloud_fraction, forest_fraction, in_pbl, ion_pair_rate):
    """Per-cell domain of every ``step_all`` input, so that an invalid input
    flags its cell even where the calculation hides it (clamped lookups and
    MIN/MAX limiters can map a NaN or out-of-range input to a finite
    output, so checking the outputs alone is not enough). The shared inputs
    get :func:`step`'s checks, except that ``production`` need only be
    finite: it is the host's H2SO4 tendency (see below). In addition the
    ion-pair rate is finite and non-negative, the forest fraction lies in
    [0, 1] and ``in_pbl`` is a truth value (0 or 1, the host's
    ``jk >= int(ppbl)``). Each input is checked whether or not the selected
    switches read it.
    """
    valid = _all_finite_nonnegative(jnp.ones(cell_shape, dtype=bool), (*mass18, *number7, gas))
    for value in (temperature, pressure, timestep):
        value = jnp.asarray(value)
        valid = valid & jnp.isfinite(value) & (value > 0)
    for value in (humidity, cloud_fraction, forest_fraction):
        value = jnp.asarray(value)
        valid = valid & jnp.isfinite(value) & (value >= 0) & (value <= 1)
    # The H2SO4 production rate is the host's gas-phase H2SO4 tendency
    # (ECHAM's zdgso4 = pxtte, mo_ham_subm.f90:371), negative wherever earlier
    # processes of the step removed gas; m7_prod_cond integrates it either
    # way, so it only has to be finite.
    valid = valid & jnp.isfinite(jnp.asarray(production))
    ion_pair_rate = jnp.asarray(ion_pair_rate)
    valid = valid & jnp.isfinite(ion_pair_rate) & (ion_pair_rate >= 0)
    in_pbl = jnp.asarray(in_pbl)
    return valid & ((in_pbl == 0) | (in_pbl == 1))


def _step_all_soa_core(mass18, number7, gas, temperature, pressure, humidity, timestep,
                       production, cloud_fraction, forest_fraction, in_pbl, ion_pair_rate,
                       prepared_table, prepared_kl_table, *, coagulation, condensation,
                       nucleation_scheme, organic_scheme, cancellation_free,
                       compute_valid, compute_water):
    dtype = mass18[0].dtype
    cell_shape = jnp.broadcast_shapes(mass18[0].shape, number7[0].shape, gas.shape)
    if compute_valid:
        input_valid = _full_input_valid(
            cell_shape, mass18, number7, gas, temperature, pressure, humidity,
            timestep, production, cloud_fraction, forest_fraction, in_pbl, ion_pair_rate,
        )
    t_arr = jnp.asarray(temperature, dtype=dtype)
    p_arr = jnp.asarray(pressure, dtype=dtype)
    rh_arr = jnp.asarray(humidity, dtype=dtype)
    th_offset = _kappa_th_offset(t_arr, rh_arr, prepared_table)

    zero_cell = jnp.zeros(cell_shape, dtype=dtype)
    zero_insol3 = (zero_cell,) * 3

    pttn1 = _average_pttn_soa(mass18, number7)
    r_insol1, d_insol1 = _insoluble_geometry_soa(pttn1, number7, zero_insol3, zero_insol3)
    _, wet1, density1, _ = _kappa_hydration_soa(
        pttn1, number7, r_insol1, d_insol1, prepared_table, th_offset, compute_water=False
    )

    if condensation:
        sink, modal_sink7 = _condensation_sink_soa(number7, wet1, t_arr, p_arr)
        gas, so4_4, insoluble_sulfate3 = _produce_and_condense_soa(
            gas, production, mass18[:4], sink, modal_sink7, cloud_fraction, timestep,
            cancellation_free=cancellation_free,
        )
        mass18 = (*so4_4, *mass18[4:])
    else:
        sink = zero_cell
        insoluble_sulfate3 = zero_insol3

    if nucleation_scheme or organic_scheme:
        nuc = nucleate_all(
            gas, t_arr, rh_arr, sink, cloud_fraction, forest_fraction, in_pbl, timestep,
            scheme=nucleation_scheme, organic_scheme=organic_scheme,
            ion_pair_rate=ion_pair_rate, kazil_lovejoy_table=prepared_kl_table,
            cancellation_free=cancellation_free,
        )
        gas, new_number = nuc.gas, nuc.number
        delta18 = (nuc.sulfate,) + (zero_cell,) * 17
    else:
        new_number = zero_cell
        delta18 = (zero_cell,) * 18

    if nucleation_scheme or organic_scheme or coagulation:
        mass18, number7, _ = _dnum_all_soa(
            mass18, number7, t_arr, p_arr, wet1, density1, insoluble_sulfate3,
            new_number, delta18, pttn1, timestep,
            coagulation=coagulation, cancellation_free=cancellation_free,
        )

    if condensation or coagulation:
        dry_mid4 = _soluble_dry_radius_soa(mass18, number7)
        mass18, number7 = _repartition_all_soa(mass18, number7, dry_mid4)

    pttn_final = _average_pttn_soa(mass18, number7)
    r_insol_final, d_insol_final = _insoluble_geometry_soa(pttn_final, number7, r_insol1, d_insol1)
    dry_final4, wet_final7, density_final7, water_final4 = _kappa_hydration_soa(
        pttn_final, number7, r_insol_final, d_insol_final, prepared_table, th_offset,
        compute_water=compute_water,
    )

    valid = None
    if compute_valid:
        valid = _all_finite_nonnegative(input_valid, (gas, *mass18, *number7))
        for arr in (*dry_final4, *wet_final7, *density_final7, *water_final4):
            valid = valid & jnp.isfinite(arr)

    return SoAStepResult(
        SoAState(mass18, number7, gas),
        SoAProperties(dry_final4, wet_final7, density_final7, water_final4),
        valid,
    )


def step_all_soa(state, temperature, pressure, humidity, timestep, production=0., cloud_fraction=0.,
                 table=None, forest_fraction=0., in_pbl=False, *, coagulation=True, condensation=True,
                 nucleation_scheme=1, organic_scheme=0, ion_pair_rate=0., kazil_lovejoy_table=None,
                 cancellation_free=None, compute_valid=True, compute_water=True,
                 chunk_size=None, remat_chunks=True):
    """Cell-minor tuple-of-arrays entry point for the 18-component HAM M7 step.

    Identical physics and numerical rounding to :func:`step_all`, without
    trailing-mode ``slice`` / ``dynamic-update-slice`` traffic or TPU minor-axis
    padding waste. Optional ``chunk_size`` tiles cells through ``jax.lax.scan``
    (with ``jax.checkpoint`` per chunk when ``remat_chunks=True``) to bound peak
    HBM and keep working sets in accelerator L2/VMEM.
    """
    if nucleation_scheme not in (0, 1, 2):
        raise ValueError('Supported M7 sulfur nucleation schemes are 0, 1 and 2')
    if organic_scheme not in (0, 1, 2):
        raise ValueError('Supported M7 organic nucleation schemes are 0, 1 and 2')
    if (nucleation_scheme or organic_scheme) and not condensation:
        raise ValueError('Native M7 leaves the sink undefined when nucleation is on and condensation off')
    if table is None:
        raise ValueError('The mixed-state core requires the kappa table (nwater = 1)')
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer when provided')

    raw_mass, raw_number, raw_gas = state
    if len(raw_mass) != 18 or len(raw_number) != 7:
        raise ValueError('SoAState requires 18 mass fields and 7 number fields')

    dtype = jnp.asarray(raw_mass[0]).dtype
    mass18 = tuple(jnp.asarray(x, dtype=dtype) for x in raw_mass)
    number7 = tuple(jnp.asarray(x, dtype=dtype) for x in raw_number)
    gas = jnp.asarray(raw_gas, dtype=dtype)

    prepared_table = prepare_kappa_table(table, dtype=dtype)
    if nucleation_scheme == 2:
        if kazil_lovejoy_table is None:
            raise ValueError(
                'M7 Kazil/Lovejoy nucleation (scheme=2) requires kazil_lovejoy_table '
                '(load_kazil_lovejoy_table(path), path from HAM_INPUT_DIR)'
            )
        prepared_kl = prepare_kazil_lovejoy_table(kazil_lovejoy_table, dtype=dtype)
    else:
        prepared_kl = None

    run_batch = functools.partial(
        _step_all_soa_core,
        prepared_table=prepared_table, prepared_kl_table=prepared_kl,
        coagulation=coagulation, condensation=condensation,
        nucleation_scheme=nucleation_scheme, organic_scheme=organic_scheme,
        cancellation_free=cancellation_free,
        compute_valid=compute_valid, compute_water=compute_water,
    )
    aux = (temperature, pressure, humidity, timestep, production, cloud_fraction,
           forest_fraction, in_pbl, ion_pair_rate)

    cell_shape = jnp.broadcast_shapes(mass18[0].shape, number7[0].shape, gas.shape)
    n_cells = math.prod(cell_shape)
    if chunk_size is None or n_cells <= chunk_size:
        if chunk_size is not None and remat_chunks and n_cells > 0:
            run_batch = jax.checkpoint(run_batch)
        return run_batch(mass18, number7, gas, *aux)

    n_chunks = -(-n_cells // chunk_size)
    pad = n_chunks * chunk_size - n_cells

    def _to_chunks(x):
        # Flatten the cells and pad the tail with copies of the last real cell
        # (gradient-detached, so padding never feeds back into a real cotangent).
        flat = jnp.broadcast_to(x, cell_shape).reshape(n_cells)
        if pad:
            flat = jnp.concatenate((flat, jnp.repeat(jax.lax.stop_gradient(flat[-1:]), pad)))
        return flat.reshape(n_chunks, chunk_size)

    # Per-cell auxiliary fields are scanned over; scalars are closed over (None in xs).
    aux_xs = tuple(None if jnp.ndim(x) == 0 else _to_chunks(jnp.asarray(x)) for x in aux)

    def _eval_chunk(xs):
        c_mass, c_num, c_gas, c_aux = xs
        resolved = (whole if part is None else part for part, whole in zip(c_aux, aux))
        return run_batch(c_mass, c_num, c_gas, *resolved)

    step_chunk = jax.checkpoint(_eval_chunk) if remat_chunks else _eval_chunk
    _, out_chunks = jax.lax.scan(
        lambda carry, xs: (carry, step_chunk(xs)), None,
        (tuple(map(_to_chunks, mass18)), tuple(map(_to_chunks, number7)), _to_chunks(gas), aux_xs),
    )
    # tree_map skips None leaves (``valid`` when compute_valid=False).
    return jax.tree_util.tree_map(lambda x: x.reshape(-1)[:n_cells].reshape(cell_shape), out_chunks)


def step_all(state, temperature, pressure, humidity, timestep, production=0., cloud_fraction=0.,
             table=None, forest_fraction=0., in_pbl=False, *, coagulation=True, condensation=True,
             nucleation_scheme=1, organic_scheme=0, ion_pair_rate=0., kazil_lovejoy_table=None,
             cancellation_free=None, chunk_size=None, remat_chunks=True):
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
    timestep; humidity, cloud or forest fraction outside [0, 1]; non-finite
    production; negative ion-pair rate; ``in_pbl`` not 0/1) or whose outputs are
    non-finite or negative.

    Dtype-generic (#1017 W1 task 3): computes in whichever float dtype
    ``state`` carries, and supports optional ``chunk_size`` cell tiling.
    """
    mass, number, gas = (jnp.asarray(x) for x in state)
    if not mass.ndim or mass.shape[-1] != 18:
        raise ValueError('mass needs trailing dimension 18')
    if not number.ndim or number.shape[-1] != 7:
        raise ValueError('number needs trailing dimension 7')
    soa_in = SoAState(
        tuple(mass[..., jn] for jn in range(18)),
        tuple(number[..., m] for m in range(7)),
        gas,
    )
    soa_out = step_all_soa(
        soa_in, temperature, pressure, humidity, timestep,
        production=production, cloud_fraction=cloud_fraction,
        table=table, forest_fraction=forest_fraction, in_pbl=in_pbl,
        coagulation=coagulation, condensation=condensation,
        nucleation_scheme=nucleation_scheme, organic_scheme=organic_scheme,
        ion_pair_rate=ion_pair_rate, kazil_lovejoy_table=kazil_lovejoy_table,
        cancellation_free=cancellation_free,
        compute_valid=True, compute_water=True,
        chunk_size=chunk_size, remat_chunks=remat_chunks,
    )
    return FullStepResult(
        FullState(
            jnp.stack(soa_out.state.mass, axis=-1),
            jnp.stack(soa_out.state.number, axis=-1),
            soa_out.state.gas,
        ),
        FullProperties(
            jnp.stack(soa_out.properties.dry_radius, axis=-1),
            jnp.stack(soa_out.properties.wet_radius, axis=-1),
            jnp.stack(soa_out.properties.density, axis=-1),
            jnp.stack(soa_out.properties.water, axis=-1),
        ),
        soa_out.valid,
    )
