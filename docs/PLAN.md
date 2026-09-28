# M7 sulfuric-acid port: commit-sized plan

Scope: sulfate coagulation, condensation, binary nucleation and their necessary
modal dependencies. Each task below is independently reviewable; related small
tasks may share a focused commit. The initial implementation is complete within
this scope. Acceptance does not extend to arbitrary atmospheric configurations.

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

## Separate future work

Scientific repairs to native singularities, rounding conditioning, molecular
weight choices or redistribution require independent reference tests and
explicit scope. Kazil/ion-mediated nucleation needs its authentic missing data.
Other species and HAM/driver integration remain outside this requested port.
