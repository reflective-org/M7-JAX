# M7-JAX

A Python/JAX port of the **ECHAM6.3-HAM2.3 M7 aerosol microphysics**, validated
against the original Fortran routines and the complete native `m7` call. This
repository is independent of SALSA-JAX.

`step_all` advances HAM's 18-component, seven-mode state: sulfate, black
carbon, organic carbon, sea salt and dust in the four soluble and three
insoluble lognormal modes, plus H₂SO₄ gas. It includes coagulation,
condensation, binary (Vehkamäki) and ion-mediated (Kazil–Lovejoy) sulfuric-acid
nucleation, organic nucleation, insoluble-to-soluble ageing, particle
properties, kappa hydration and modal redistribution in the native process
order, and the host-tracer conversions of HAM's `ham_subm_interface`. The
original pure-sulfate `step` (four soluble modes, empty insoluble modes, both
hydration options) remains. The rest of HAM (chemistry, emissions,
deposition, the box and GCM drivers), secondary organic aerosol and radiation
are outside this scope.

## Install

```sh
pip install m7-jax
```

The wheel includes the authentic 45 MB kappa lookup table
(`m7_jax.properties.default_kappa_table_path()`). The Fortran reference
harness, the recorded configurations and the test suite need a checkout of
this repository.

## Develop and test

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
from m7_jax.properties import default_kappa_table_path, load_kappa_table

state = initialize(dry_diameter_nm=(3., 25., 200., 2000.))
table = load_kappa_table(default_kappa_table_path())
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

**353 tests pass** on JAX 0.10.2, 0.11.0 and 0.12.1, with 8 real-PARNUC-table
variants skipped; **361 pass** when `HAM_INPUT_DIR` names the table.
All six recorded trajectories pass every-step Fortran comparison and sulfur
budgets. This establishes compatibility with the Fortran for the tested
configurations, not independent physical accuracy. The
[validation report](https://github.com/reflective-org/M7-JAX/blob/main/docs/validation.md) has the measured errors.

- [Architecture, units and scientific limitations](https://github.com/reflective-org/M7-JAX/blob/main/docs/model.md)
- [Validation results, timestep refinement and CPU measurements](https://github.com/reflective-org/M7-JAX/blob/main/docs/validation.md)
- [Commit-sized implementation and verification plan](https://github.com/reflective-org/M7-JAX/blob/main/docs/PLAN.md)
- [Reference provenance](https://github.com/reflective-org/M7-JAX/blob/main/reference/manifest.json)

JAX is slower for the measured single-cell workloads. Native cancellation and
lookup discretization are preserved in float64. The float32 path of
`step_all` uses algebraically identical forms that avoid the cancellations
([model contract](https://github.com/reflective-org/M7-JAX/blob/main/docs/model.md#float32-precision-adaptations)). The Kazil/Lovejoy lookup option
(`nucleate_all`/`step_all`, `scheme=2`) is supported given its PARNUC table
(`HAM_INPUT_DIR`, not shipped -- see `docs/validation.md`); without one it is
rejected there too, and the independent `nucleate`/`step` APIs reject it
unconditionally. Coupled nucleation with condensation disabled is also
rejected because native M7 leaves its sink uninitialized; the independent
nucleation API accepts an explicit sink. One native defect is corrected, as
the r7492 reference configuration itself corrects it (the insoluble-mode
radius, TUT-ISI/SALSA#6); every other native formula, including the known
defects listed in the [model contract](https://github.com/reflective-org/M7-JAX/blob/main/docs/model.md#native-defects-retained),
is unchanged. Original source/license notices are retained in `reference/`.

## License

M7-JAX is licensed under the [Apache License 2.0](https://github.com/reflective-org/M7-JAX/blob/main/LICENSE). The JAX kernels
and native oracle are derived from HAMMOZ M7 and also remain subject to the
[BSD 3-Clause License](https://github.com/reflective-org/M7-JAX/blob/main/LICENSES/BSD-3-Clause-HAMMOZ.txt); see [NOTICE](https://github.com/reflective-org/M7-JAX/blob/main/NOTICE).
Upstream sources and data under `reference/` are BSD 3-Clause; see NOTICE
for the two ICON constant modules that still carry older MPI-M headers.
