# M7-JAX

A Python/JAX port of **M7 sulfuric-acid coagulation, condensation and binary
nucleation**, validated against the original Fortran routines and complete
pure-sulfate core call. This repository is independent of SALSA-JAX.

The supported state has four soluble lognormal modes (nucleation, Aitken,
accumulation, coarse) and three empty insoluble modes. It includes native
particle properties, both hydration options, modal redistribution and the
native process order. Other aerosol materials, organic/ion-mediated nucleation,
HAM coupling, radiation and the full box driver are outside this scope.

## Install and test

Use this repository's own virtual environment. GNU Fortran must be on `PATH`;
NetCDF Python reading uses SciPy, so the reference harness does not require
NetCDF Fortran libraries. The authentic 45 MB kappa lookup table is included.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
```

Run commands from this checkout. Enable JAX double precision before constructing
arrays. `requirements-dev.txt` records the tested environment; the editable
checkout includes the reference inputs used by command-line tools.

## Run the sulfate experiment

```sh
.venv/bin/python -m tools.run_m7_sulfate \
  --config configs/m7-sulfate.toml --output outputs/sulfate
```

Use a fresh output directory. The command runs **both JAX and checked/optimized
Fortran**, saves every timestep to NPZ and CSV, records source/compiler hashes,
and checks parity, physical state and sulfur budgets. It exits nonzero if a
comparison or budget gate fails, retaining the report for inspection.

The example starts with 100 and 900 particles/cm³ in the nucleation and Aitken
modes, dry count-median diameters 3 and 25 nm, 280 K, 101325 Pa, RH 50%, and
10⁸ H₂SO₄ molecules/cm³. Gas production is 10⁵ molecules/cm³/s; the gas evolves
rather than remaining constant. It runs 600 one-second steps. Other checked
configurations are in `configs/`.

## Python API

```python
import jax
jax.config.update('jax_enable_x64', True)
from m7_jax.model import initialize, step, STATIC_OPTIONS
from m7_jax.properties import load_kappa_table

state = initialize(dry_diameter_nm=(3., 25., 200., 2000.))
table = load_kappa_table('reference/input/lut_kappa.nc')
advance = jax.jit(step, static_argnames=STATIC_OPTIONS)
result = advance(state, 280., 101325., .5, 1., production=1e5, table=table)
assert bool(result.valid)
state = result.state
```

Independent APIs: `coagulation.coaset`/`dnum`,
`condensation.condensation_sink`/`produce_and_condense`, and
`nucleation.vehkamaeki`/`nucleate`. Use native concentrations in **cm⁻³**, radii
in **cm**, pressure in Pa, temperature in K and timestep in seconds. Leading
array dimensions represent independent cells. Host validation rejects insoluble
populations; check `StepResult.valid` inside repeated integration.

## Evidence and limits

**85 tests pass** in this checkout’s own virtual environment.
All six recorded trajectories pass every-step Fortran comparison and sulfur
budgets. Maximum relative discrepancy across their reported fields is
4.42×10⁻¹³; maximum relative sulfur residual is below 6×10⁻¹⁵. This establishes
compatibility for the tested sulfate scope, not independent physical accuracy.

- [Architecture, units and scientific limitations](docs/model.md)
- [Validation results, timestep refinement and CPU measurements](docs/validation.md)
- [Commit-sized implementation and verification plan](docs/PLAN.md)
- [Reference provenance](reference/manifest.json)

JAX is slower for the measured single-cell workloads. Native cancellation and
lookup discretization are preserved in float64. The float32 path of
`step_all` uses algebraically identical forms that avoid the cancellations
([model contract](docs/model.md#float32-precision-adaptations)). The Kazil/Lovejoy lookup option
(`nucleate_all`/`step_all`, `scheme=2`) is supported given its PARNUC table
(`HAM_INPUT_DIR`, not shipped -- see `docs/validation.md`); without one it is
rejected there too, and the independent `nucleate`/`step` APIs reject it
unconditionally. Coupled nucleation with condensation disabled is also
rejected because native M7 leaves its sink uninitialized; the independent
nucleation API accepts an explicit sink. No scientific Fortran formulas were
changed. Original source/license notices are retained in `reference/`.
