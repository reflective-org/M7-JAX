# Copyright 2026 Reflective
# Portions derived from HAMMOZ M7, Copyright (c) 2021 hammoz;
# see NOTICE and LICENSES/BSD-3-Clause-HAMMOZ.txt.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
"""M7 coagulation in native CGS units, restricted to pure sulfate.

The 16 collision kernels retain all seven M7 modes. The time integration is
the soluble sulfate branch of ``m7_dnum``/``m7_delcoa``: insoluble modes must
have zero number, and no other species or insoluble condensate is supported.
Callers enable JAX float64. Leading dimensions describe independent cells.
"""

import jax.numpy as jnp

from .constants import BK, CMIN_AERNL, CMINRAD, CMINRHO, COAG_MODE_PAIRS, PI, ZMIN_COAG


def coaset(number, temperature, pressure, radius, density):
    """Return the 16 native M7 collision coefficients [cm³/s].

    ``number``, ``radius`` and ``density`` have trailing dimension seven,
    in cm⁻³, cm and g/cm³ respectively. Temperature [K] and pressure [Pa]
    broadcast over the leading cell dimensions. Inactive modes yield zero.
    """
    arrays = [jnp.asarray(x, dtype=jnp.float64) for x in (number, radius, density)]
    if any(not value.ndim or value.shape[-1] != 7 for value in arrays):
        raise ValueError("number, radius and density need seven trailing modes")
    number, radius, density = jnp.broadcast_arrays(*arrays)
    temperature = jnp.asarray(temperature, dtype=jnp.float64)[..., None]
    pressure = jnp.asarray(pressure, dtype=jnp.float64)[..., None]
    active = (number > CMIN_AERNL) & (radius > CMINRAD) & (density > CMINRHO)
    # Benign arguments keep unselected empty-mode arithmetic finite.
    r = jnp.where(active, radius, 1.0)
    rho = jnp.where(active, density, 1.0)
    free_path = 2.251e-3 * temperature / pressure
    viscosity = 1.827e-4 * (temperature / 291.15)**0.74
    thermal = (BK / (6.0 * PI)) * temperature / viscosity
    volume = (4.0 * PI / 3.0) * r**3
    velocity_squared = (8.0 * BK / PI) * temperature / (volume * rho)
    knudsen = 0.5 * free_path / r
    diffusion = thermal * (1.0 + knudsen * 2.492 + knudsen * 0.84 * jnp.exp(-0.43 / knudsen)) / r
    distance = (8.0 / PI) * diffusion / jnp.sqrt(velocity_squared)
    diameter = 2.0 * r
    jump = (((diameter + distance)**3 - jnp.sqrt((diameter * diameter + distance * distance)**3))
            / (6.0 * r * distance) - diameter) * jnp.sqrt(2.0)
    first, second = jnp.asarray(COAG_MODE_PAIRS).T
    average_radius = jnp.finfo(jnp.float64).eps + 0.5 * (r[..., first] + r[..., second])
    velocity = jnp.sqrt(velocity_squared[..., first] + velocity_squared[..., second])
    average_diffusion = 0.5 * (diffusion[..., first] + diffusion[..., second])
    jump_pair = jnp.sqrt(jump[..., first]**2 + jump[..., second]**2)
    continuum = (16.0 * PI) * average_diffusion * average_radius
    resistance = (4.0 * average_diffusion / (velocity * average_radius)
                  + average_radius / (average_radius + 0.5 * jump_pair))
    return jnp.where(active[..., first] & active[..., second], continuum / resistance, 0.0)


def _decay(total, initial, a, b, formed, timestep):
    """Native analytical number solution and intra/inter-mode loss split."""
    inter = b >= ZMIN_COAG
    safe_b = jnp.where(inter, b, 1.0)
    safe_a = jnp.where(a != 0, a, 1.0)
    exponential = jnp.exp(-b * timestep)
    no_source = jnp.where(
        inter,
        jnp.minimum(safe_b * total * exponential / (total * a * (1.0 - exponential) + safe_b), total),
        total / (1.0 + total * a * timestep),
    )
    source = formed / timestep
    radical = jnp.sqrt(b * b + 4.0 * a * source)
    f2 = 2.0 * a * initial + b
    denominator = radical + f2
    f3 = jnp.exp(-radical * timestep) * (radical - f2) / jnp.where(denominator != 0, denominator, 1.0)
    with_inter = (radical * ((1.0 - f3) / (1.0 + f3)) - b) / 2.0 / safe_a
    r1 = jnp.sqrt(source / safe_a)
    exp_intra = jnp.exp(-r1 * 2.0 * a * timestep)
    # Preserve the native solution, including its signed denominator test.
    ratio = (initial + r1) / jnp.where(initial != r1, initial - r1, 1.0)
    bottom = ratio - exp_intra
    with_intra = jnp.where(bottom < ZMIN_COAG, total,
                          r1 * (ratio + exp_intra) / jnp.where(bottom != 0, bottom, 1.0))
    # The native formula is singular at initial == sqrt(source/a). That case
    # is outside the validated native domain; no alternate science is added.
    with_intra = jnp.where((source >= 1.0) & (initial == r1), jnp.nan, with_intra)
    new = jnp.where(source >= 1.0, jnp.minimum(jnp.where(inter, with_inter, with_intra), total), no_source)
    total_loss = total - new
    intra_rate = total * total * a
    inter_rate = total * b
    rate = intra_rate + inter_rate
    safe_rate = jnp.where(rate != 0, rate, 1.0)
    inter_loss = jnp.where(inter, total_loss * inter_rate / safe_rate, 0.0)
    intra_loss = jnp.where(inter, total_loss * intra_rate / safe_rate, total_loss)
    return new, intra_loss, inter_loss


def delcoa(sulfate, number, sulfate_delta, nucleated_number, a, b, fraction1, fraction2, timestep):
    """Integrate pure-sulfate M7 coagulation, returning mass, number, delta.

    Sulfate and delta have four trailing soluble modes [molecules/cm³];
    number has seven [cm⁻³], with the three insoluble entries zero. ``a`` and
    ``b`` have four entries [cm³/s and s⁻¹]; fractions have six entries in the
    native destination-minus-two order. Only delta[..., 0] is a supported
    incoming source (nucleated sulfate); nucleated_number is cm⁻³ per step.
    ``timestep`` is positive seconds. Ordered transfers match the Fortran.
    """
    sulfate, sulfate_delta = jnp.broadcast_arrays(
        jnp.asarray(sulfate, dtype=jnp.float64), jnp.asarray(sulfate_delta, dtype=jnp.float64))
    number = jnp.asarray(number, dtype=jnp.float64)
    nucleated_number = jnp.asarray(nucleated_number, dtype=jnp.float64)
    timestep = jnp.asarray(timestep, dtype=jnp.float64)
    a, b, fraction1, fraction2 = (jnp.asarray(value, dtype=jnp.float64)
                                for value in (a, b, fraction1, fraction2))
    for value, size in ((sulfate, 4), (number, 7), (a, 4), (b, 4), (fraction1, 6), (fraction2, 6)):
        if not value.ndim or value.shape[-1] != size:
            raise ValueError(f"Expected trailing mode dimension {size}")
    for mode in range(4):
        formed = nucleated_number if mode == 0 else jnp.zeros_like(nucleated_number)
        total = number[..., mode] + formed
        mass = sulfate[..., mode] + (sulfate_delta[..., 0] if mode == 0 else 0.0)
        sufficient = total > CMIN_AERNL
        active = sufficient & ((a[..., mode] >= ZMIN_COAG) | (b[..., mode] >= ZMIN_COAG))
        average = mass / jnp.where(sufficient, total, 1.0)
        new, intra, inter = _decay(total, number[..., mode], a[..., mode], b[..., mode], formed, timestep)
        updated_mass = jnp.where(active, (new + intra) * average, mass if mode == 0 else sulfate[..., mode])
        updated_number = jnp.where(active, new, total if mode == 0 else number[..., mode])
        sulfate = sulfate.at[..., mode].set(jnp.where(sufficient, updated_mass, sulfate[..., mode]))
        number = number.at[..., mode].set(jnp.where(sufficient, updated_number, number[..., mode]))
        if mode == 0:
            for target in range(1, 4):
                transfer = fraction1[..., target - 1] * inter * average
                transfer += fraction1[..., target + 2] * inter * average
                sulfate_delta = sulfate_delta.at[..., target].set(jnp.where(active, transfer, sulfate_delta[..., target]))
        elif mode == 1:
            for target in (2, 3):
                transfer = fraction2[..., target - 1] * inter * average
                transfer2 = fraction2[..., target + 2] * inter * average
                value = sulfate_delta[..., target] + transfer
                value = value + transfer2
                sulfate_delta = sulfate_delta.at[..., target].set(jnp.where(active, value, sulfate_delta[..., target]))
            value = sulfate_delta[..., 1] + fraction2[..., 3] * inter * average
            sulfate_delta = sulfate_delta.at[..., 1].set(jnp.where(active, value, sulfate_delta[..., 1]))
    sulfate = sulfate.at[..., 1:4].add(sulfate_delta[..., 1:4])
    return sulfate, number, sulfate_delta


def dnum(sulfate, number, temperature, pressure, radius, density,
         nucleated_number, sulfate_delta, timestep, *, coagulation=True):
    """Pure-sulfate ``m7_dnum`` with native units and trailing mode axes.

    Insoluble numbers must be zero; coating and mixed-species aging are outside
    this API. ``coagulation`` is a static switch when compiling with JAX.
    See :func:`coaset` and :func:`delcoa` for the input and return contracts.
    """
    number = jnp.asarray(number, dtype=jnp.float64)
    if not number.ndim or number.shape[-1] != 7:
        raise ValueError("number needs seven trailing modes")
    shape = number.shape[:-1]
    a, b = jnp.zeros(shape + (4,), dtype=jnp.float64), jnp.zeros(shape + (4,), dtype=jnp.float64)
    fraction1 = jnp.zeros(shape + (6,), dtype=jnp.float64)
    fraction2 = jnp.zeros(shape + (6,), dtype=jnp.float64)
    if coagulation:
        kernels = coaset(number, temperature, pressure, radius, density)
        a = a.at[..., :3].set(kernels[..., jnp.asarray([0, 7, 13])] / 2.0)
        fraction1 = kernels[..., 1:7] * number[..., 1:7]
        fraction2 = fraction2.at[..., 1:3].set(kernels[..., 8:10] * number[..., 2:4])
        fraction2 = fraction2.at[..., 4:6].set(kernels[..., 11:13] * number[..., 5:7])
        # Ordered sums follow the native statement order.
        b1 = fraction1[..., 0]
        for index in range(1, 6):
            b1 = b1 + fraction1[..., index]
        b2 = fraction2[..., 1]
        for index in range(2, 6):
            b2 = b2 + fraction2[..., index]
        b = b.at[..., 0].set(b1).at[..., 1].set(b2)
        fraction1 = fraction1 / jnp.where(b1 > 0, b1, 1.0)[..., None]
        fraction2 = fraction2 / jnp.where(b2 > 0, b2, 1.0)[..., None]
    return delcoa(sulfate, number, sulfate_delta, nucleated_number, a, b, fraction1, fraction2, timestep)
