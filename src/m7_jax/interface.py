"""Host tracer <-> native M7 unit conversions (mo_ham_subm.f90::ham_subm_interface).

mo_ham_subm.f90 is a BSD-3 HAM module that does not ship in this checkout (only
mo_ham_m7.F90 and its dependencies do - see AGENTS.md/README.md on scope). The
conversions below, including the literal 6.022e20 factor, are specified by the
#1017 lead directly (TASK.md), not read from a Fortran file here; keeping them
in their own module, isolated from the checked M7 core, makes that provenance
difference easy to see in review.

All functions are pure and take plain arrays; none touch jax.jit tracing
rules beyond ordinary arithmetic, so they work equally well eager or jitted.
"""
import jax.numpy as jnp

from .constants import SO4_MOLAR_MASS, HAM_SUBM_AVOGADRO

NUMBER_FACTOR = 1e-6  # cm3 -> m3, i.e. particles kg-1 air -> particles cm-3 given air_density in kg m-3
MASS_FACTOR = 1e9     # kg -> ug, i.e. kg kg-1 -> ug m-3 given air_density in kg m-3


def so4_mixing_ratio_to_native(mixing_ratio, air_density, molar_mass=SO4_MOLAR_MASS):
    """SO4/H2SO4 mass mixing ratio [kg kg-1 air] -> molecules cm-3.

    ``value = mixing_ratio * air_density * 6.022e20 / molar_mass``. Air
    density is in kg m-3. `molar_mass` defaults to the species-registry SO4
    weight (96.0631 g/mol, matching `kappa_hydration`); pass 98.0734 (H2SO4,
    matching `repartition_all`/the legacy sulfate-only path) if the host's
    tracer is a mixing ratio of H2SO4 rather than of the SO4 aerosol species.
    """
    return jnp.asarray(mixing_ratio)*jnp.asarray(air_density)*HAM_SUBM_AVOGADRO/molar_mass


def native_to_so4_mixing_ratio(value, air_density, molar_mass=SO4_MOLAR_MASS):
    """Inverse of `so4_mixing_ratio_to_native`: molecules cm-3 -> kg kg-1 air."""
    return jnp.asarray(value)*molar_mass/(jnp.asarray(air_density)*HAM_SUBM_AVOGADRO)


def mass_mixing_ratio_to_native(mixing_ratio, air_density):
    """Non-SO4 component mass mixing ratio [kg kg-1 air] -> ug m-3 (BC, OC, SS, DU).

    ``value = mixing_ratio * air_density * 1e9``, air density in kg m-3.
    """
    return jnp.asarray(mixing_ratio)*jnp.asarray(air_density)*MASS_FACTOR


def native_to_mass_mixing_ratio(value, air_density):
    """Inverse of `mass_mixing_ratio_to_native`: ug m-3 -> kg kg-1 air."""
    return jnp.asarray(value)/(jnp.asarray(air_density)*MASS_FACTOR)


def number_mixing_ratio_to_native(mixing_ratio, air_density):
    """Number mixing ratio [particles kg-1 air] -> cm-3.

    ``value = mixing_ratio * air_density * 1e-6``, air density in kg m-3.
    """
    return jnp.asarray(mixing_ratio)*jnp.asarray(air_density)*NUMBER_FACTOR


def native_to_number_mixing_ratio(value, air_density):
    """Inverse of `number_mixing_ratio_to_native`: cm-3 -> particles kg-1 air."""
    return jnp.asarray(value)/(jnp.asarray(air_density)*NUMBER_FACTOR)


def aerosol_mass_to_native(mixing_ratio, air_density, molar_mass=SO4_MOLAR_MASS):
    """Full (...,18) HAM aerocomp mass mixing ratio [kg kg-1 air] -> native
    units: molecules cm-3 for the four SO4 components (idx 0-3), ug m-3 for
    every other component (idx 4-17); see constants.COMPONENT_SPECIES.
    """
    mixing_ratio = jnp.asarray(mixing_ratio)
    so4 = so4_mixing_ratio_to_native(mixing_ratio[..., :4], air_density, molar_mass)
    other = mass_mixing_ratio_to_native(mixing_ratio[..., 4:], air_density)
    return jnp.concatenate((so4, other), axis=-1)


def native_to_aerosol_mass(value, air_density, molar_mass=SO4_MOLAR_MASS):
    """Inverse of `aerosol_mass_to_native`: native (...,18) -> kg kg-1 air."""
    value = jnp.asarray(value)
    so4 = native_to_so4_mixing_ratio(value[..., :4], air_density, molar_mass)
    other = native_to_mass_mixing_ratio(value[..., 4:], air_density)
    return jnp.concatenate((so4, other), axis=-1)
