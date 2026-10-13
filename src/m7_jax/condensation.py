# Copyright 2026 Reflective
# Portions derived from HAMMOZ M7, Copyright (c) 2021 hammoz;
# see NOTICE and LICENSES/BSD-3-Clause-HAMMOZ.txt.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
"""Native M7 sulfuric-acid condensation, in cm-based aerosol units.

These formulas are specific to M7: gas/sulfate molecules/cm3, number particles/
cm3, radius cm, sink s-1 and production molecules/cm3/s. Insoluble condensate is
returned separately for the native coating stage; it is zero in sulfate-only
soluble experiments. Temperature and pressure are K and Pa.
"""
from typing import NamedTuple

import jax.numpy as jnp

from m7_jax.constants import ACCOMMODATION, H2SO4_MOLAR_MASS, PI, RERG


class CondensationSink(NamedTuple):
    total: object
    modal: object


class CondensationResult(NamedTuple):
    gas: object
    sulfate: object
    insoluble_sulfate: object


def condensation_sink(number_cm3, radius_cm, temperature_k, pressure_pa):
    """Port m7_h2so4_cs for seven modes, preserving its ordered reduction."""
    number, radius = jnp.asarray(number_cm3), jnp.asarray(radius_cm)
    if number.shape[-1:] != (7,) or radius.shape != number.shape:
        raise ValueError('number and radius must have matching (...,7) shape')
    temperature = jnp.broadcast_to(jnp.asarray(temperature_k), number.shape[:-1])[..., None]
    pressure = jnp.broadcast_to(jnp.asarray(pressure_pa), number.shape[:-1])[..., None]
    diffusion = .073 * (1000. / (pressure / 100.)) * (temperature / 298.15)**1.5
    velocity = jnp.sqrt(8. * RERG * temperature / PI / H2SO4_MOLAR_MASS)
    mean_path = 8. * diffusion / PI / velocity
    safe_radius = jnp.where(radius > 0., radius, 1.)
    transition_distance = (((safe_radius + mean_path)**3
                            - (safe_radius**2 + mean_path**2)**1.5)
                           / (3. * safe_radius * mean_path) - safe_radius)
    modal = (4. * PI * diffusion * safe_radius) / (
        (4. * diffusion) / (velocity * safe_radius * jnp.asarray(ACCOMMODATION))
        + safe_radius / (safe_radius + transition_distance))
    modal = jnp.where(radius > 0., modal * number, 0.)
    total = jnp.zeros(number.shape[:-1], dtype=modal.dtype)
    for index in range(7):
        total = total + modal[..., index]
    return CondensationSink(total, modal)


def produce_and_condense(gas_cm3, production_cm3_s, sulfate, sink, modal_sink,
                         cloud_fraction, timestep):
    """Port m7_prod_cond; sulfate has four soluble modes on its last axis.

    For sink <=1e-10/s only gas production occurs, even in a full cloud. Above
    that threshold, cloud-free condensate follows modal sinks and cloudy
    condensate goes to the largest occupied soluble mode (then insoluble 7,6,5).
    The native expression and clipping are preserved, including cancellation
    when the source/sink balance is large.
    """
    sulfate, modal_sink = jnp.asarray(sulfate), jnp.asarray(modal_sink)
    if sulfate.shape[-1:] != (4,) or modal_sink.shape != sulfate.shape[:-1] + (7,):
        raise ValueError('sulfate (...,4) and modal_sink (...,7) must share leading dimensions')
    gas, production, sink = jnp.asarray(gas_cm3), jnp.asarray(production_cm3_s), jnp.asarray(sink)
    active = sink > 1e-10
    safe_sink = jnp.where(active, sink, 1.)
    cloud = jnp.clip(jnp.asarray(cloud_fraction), 0., 1.)
    clear_final = jnp.maximum((gas - production / safe_sink)
        * jnp.exp(-safe_sink * timestep) + production / safe_sink, 0.)
    clear_condensed = (1. - cloud) * (gas - clear_final + production * timestep)
    cloudy_condensed = cloud * (gas + production * timestep)
    deposits = modal_sink / safe_sink[..., None] * clear_condensed[..., None]
    # Native precedence CS, AS, KS, NS, CI, AI, KI, including cloudy fallback.
    remaining = jnp.ones(sulfate.shape[:-1], dtype=bool)
    for index in (3, 2, 1, 0, 6, 5, 4):
        receives = remaining & (modal_sink[..., index] > 0.)
        deposits = deposits.at[..., index].add(jnp.where(receives, cloudy_condensed, 0.))
        remaining = remaining & ~receives
    deposits = jnp.where(active[..., None], deposits, 0.)
    final_gas = jnp.where(active, (1. - cloud) * clear_final, gas + timestep * production)
    return CondensationResult(final_gas, sulfate + deposits[..., :4], deposits[..., 4:])
