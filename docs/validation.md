# Validation and performance

Recorded 2026-09-28 using float64 CPU JAX 0.11.2, NumPy 2.5.3, Python 3.12.12 and GNU Fortran 16.1.0.

The final local checkout passed **85 tests in 45.99 seconds** using its own `.venv`
(`python -m pytest -q`). Generated trajectories are saved locally in
`outputs/m7-*-20260928/`; compact evidence is committed below.

## Numerical reference checks

The regression suite covers actual checked Fortran leaves, initialization constants, both hydration schemes, normal-CDF branches, modal transfer, all supported process-switch combinations and repeated full calls. The binary rate matrix contains 216 temperature/RH/gas cases. The trajectory observer is independently checked against unobserved native final states and manual replay. Checked/optimized native checkpoints are exact.

Process-specific tolerances are explicit in tests: transport/condensation 3×10⁻¹³, binary fit 10⁻¹¹, composed single-step state 3×10⁻¹¹, and repeated/recorded state 3×10⁻¹⁰. The recorded trajectory gate uses zero absolute tolerance, checks reference zeros separately, and requires relative sulfur budget residual ≤10⁻¹⁰. These gates are not scientific accuracy targets.

## Recorded trajectory matrix

All rows compare every timestep, not only final values. All have finite nonnegative state, empty insoluble modes, conserved sulfate+gas including source, and exact checked/optimized native results.

| Configuration | Steps × dt | Final number, cm⁻³ | Maximum relative JAX–Fortran discrepancy | Maximum sulfur residual, either implementation |
|---|---:|---:|---:|---:|
| cold-active | 1000 × 1.0 s | 3503866.480291 | 4.41e-13 | 5.6e-15 |
| condensation | 600 × 1.0 s | 1000.000000 | 1.36e-14 | 6.7e-16 |
| coupled | 600 × 1.0 s | 998.743266 | 2.53e-14 | 1.31e-15 |
| coagulation | 3600 × 1.0 s | 992.395750 | 8.99e-15 | 4.8e-15 |
| jacobson | 600 × 1.0 s | 998.609918 | 8.19e-15 | 1.08e-15 |
| half-timestep | 1200 × 0.5 s | 998.743292 | 1.49e-14 | 9.82e-16 |

The warm coupled run is weakly nucleating; the 240 K, 10⁹ molecules/cm³ cold case explicitly tests strong nucleation and vapor depletion. It is a numerical stress case, not a calibrated atmospheric scenario. All configurations are in `configs/` and embedded in the [trajectory evidence](evidence/m7-trajectories-2026-09-28.json).

Halving the warm-run timestep from 1 s to 0.5 s changes its final modal number distribution by 0.00802% in relative L1, sulfate by 0.000145%, and gas by 0.0000919%. Maximum number L1 change over equal physical times is 0.00993%. Both languages show the same sensitivity. This two-resolution check does not establish a convergence order.

## CPU process measurements

Five processes were timed for one cell and batches of 64, after checked/optimized equality and JAX parity checks. Each result uses the median of three trials of 2000 repetitions. JAX outputs are synchronized after every call; compilation/warmup is excluded. The recorded first-call field can include a JIT cache hit and is not guaranteed cold compilation.

Native values below are CPU time per batch, including state reset and output copy; JAX values are wall time including dispatch/synchronization. Native table loading, input/output serialization and compilation are excluded. These are scoped comparisons with differing wrapper overhead, not full-model speedups.

| Process | Cells | Native CPU µs/batch | JAX wall µs/batch |
|---|---:|---:|---:|
| coaset | 1 | 0.26 | 4.78 |
| condensation_sink | 1 | 0.17 | 5.49 |
| produce_and_condense | 1 | 0.05 | 7.17 |
| nucleate | 1 | 0.27 | 6.42 |
| dnum | 1 | 0.76 | 13.54 |
| coaset | 64 | 12.62 | 29.62 |
| condensation_sink | 64 | 10.73 | 13.66 |
| produce_and_condense | 64 | 3.34 | 37.10 |
| nucleate | 64 | 13.60 | 11.89 |
| dnum | 64 | 39.68 | 46.00 |

JAX is slower for every measured single-cell process. Batched nucleation alone slightly outperforms the measured native CPU time; no general acceleration is claimed. Reproduce with `.venv/bin/python -m benchmarks.benchmark_m7 --output outputs/benchmark.json`. See the [complete benchmark evidence](evidence/m7-process-benchmark.json) for raw trials, case/source hashes and gates.

### Retained small-transfer diagnostic

A one-second coagulation case fails the 3×10⁻¹² internal-transfer relative gate: a transfer increment differs by about 9.8×10⁻⁹ molecules/cm³, or 3.08×10⁻¹⁰ relative. Final sulfate/number are within about 2×10⁻¹⁵ relative. This cancellation-sensitive diagnostic is retained as a failure. Timed coagulation uses the independently validated 600-second case and does not change the tolerance. The measured tiny-transfer failure is distinct from the six accepted full-state trajectories.

## Remaining limits

This is a pure-sulfate core port with a controlled native metadata registry. It does not validate the complete HAM driver. Missing Kazil nucleation data, non-sulfate species and undefined native condensation-off/nucleation-on calls are explicitly outside acceptance. Shared native formulas, hydration lookup discretization, molecular-weight distinctions and coagulation singularities remain as documented in [the model contract](model.md).
