# M7 port: commit-sized plan

Scope: the complete ECHAM6.3-HAM2.3 M7 call on HAM's 18-component state. Tasks
M01-M18 delivered the pure-sulfate port (0.1.0); M19-M26 extend it to the full
state (0.2.0). Each task is independently reviewable; related small tasks may
share a focused commit. Acceptance does not extend to arbitrary atmospheric
configurations.

## 1. Preserve references and establish the environment

- **M01 — Reference snapshot:** copy unchanged M7 science/constants/metadata
  sources and authentic kappa data. Verify against pinned source hashes.
- **M02 — Package isolation:** create `m7_jax`, its own virtual environment,
  dependency snapshot and ignore rules. Verify imports from this checkout.
- **M03 — Native oracle:** expose private routines through a visibility overlay
  and supply explicit metadata. Verify checked/optimized equality.
- **M04 — Trajectory observer:** capture each original full call without changing
  formulas. Verify final output and manual replay are bitwise identical.

## 2. Port processes separately

- **M05 — Coefficients:** port the native 16-pair coagulation kernel. Verify
  active/empty modes and exact threshold semantics against Fortran.
- **M06 — Coagulation transfer:** port ordered sulfate number/mass updates.
  Verify sources, disabled coagulation and sulfur budgets.
- **M07 — Condensation sink:** port native transport and accommodation factors.
  Verify mode-wise and total sinks in eager and JIT execution.
- **M08 — Production/condensation:** port analytical integration and cloud
  allocation. Verify sink thresholds, gas depletion and sulfur inventory.
- **M09 — Binary rate:** port Vehkamaeki polynomial/clamps. Verify the 216-case
  temperature/RH/concentration matrix against original Fortran.
- **M10 — Nucleation insertion:** port native gas limitation and new-particle
  increments. Verify cloud branches and sulfur budgets; reject missing schemes.

## 3. Complete required dependencies and composition

- **M11 — Modal properties:** preserve virtual clusters and lognormal moments.
  Verify compiled constants, empty populations and both hydration options.
- **M12 — Redistribution:** port native normal-CDF branches and simultaneous
  upward transfer. Verify threshold/tail cases and conservation.
- **M13 — Coupled call:** compose the original process order. Verify the process
  switch matrix, both water schemes and 100 repeated calls.
- **M14 — Public contract:** validate state/environment and reject undefined
  native configurations. Verify NaN, negative and unsupported inputs.

## 4. Record reproducible acceptance

- **M15 — Configured runner:** save every timestep, provenance and failure
  verdicts. Verify saved units, hashes and protected output directories.
- **M16 — Trajectory matrix:** coupled, coagulation, condensation, strong cold
  nucleation, Jacobson and half-timestep runs. Require finite nonnegative state,
  every-step native parity and sulfur budgets; record actual measured errors.
- **M17 — Benchmarks:** measure checked-gated optimized Fortran and synchronized
  warm JAX for five processes, one/64 cells. Record costs without speed promises.
- **M18 — Documentation and delivery:** document units, assumptions, limitations,
  reproduction and results; run the full M7 suite in its own virtual environment,
  then commit and push to Reflective-org/M7-JAX.

## 5. Full HAM M7 state (0.2.0)

- **M19 — Species registry:** the 18 HAM aerocomp components and their modes.
  Verify the compiled metadata against the native registry.
- **M20 — Mixed properties:** `m7_averageproperties` and kappa hydration for
  every species. Verify against the oracle, including the r7492 insoluble
  radius correction (TUT-ISI/SALSA#6, [model contract](model.md#corrections-to-the-native-source)).
- **M21 — Mixed coagulation and ageing:** both `m7_delcoa` sections and
  `m7_concoag`/`m7_coat`. Verify transfers, coating thresholds and budgets.
- **M22 — Nucleation:** organic (HAMMOZ build) and Kazil–Lovejoy nucleation.
  Verify against the oracle; Kazil–Lovejoy on a synthetic table by default and
  on the real PARNUC table when `HAM_INPUT_DIR` names it.
- **M23 — Complete call:** `step_all` in the native `m7` order. Verify designed
  and random cases, every process switch, validity and per-species budgets.
- **M24 — Float32:** cancellation-free forms chosen by the working dtype.
  Verify against 50-digit references and the Fortran.
- **M25 — Reverse mode:** finite gradients in both dtypes and float64
  gradients equal to finite differences.
- **M26 — Host interface and packaging:** `ham_subm_interface` conversions and
  the kappa table as package data. Verify round trips and the wheel contents.

## Separate future work

Scientific repairs to native singularities, rounding conditioning, molecular
weight choices or redistribution require independent reference tests and
explicit scope; the native defects M7-JAX keeps are listed in the
[model contract](model.md#native-defects-retained). The rest of HAM and the
GCM coupling stay outside this port.
