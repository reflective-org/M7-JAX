# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Fortran-validated ECHAM-HAM M7 aerosol microphysics in JAX.

Enable jax_enable_x64 to reproduce the float64 Fortran; ``step_all`` also runs
in float32. Native units are molecules/cm3, particles/cm3, ug/m3 for non-sulfate
mass, radii cm, temperature K and pressure Pa.
"""
