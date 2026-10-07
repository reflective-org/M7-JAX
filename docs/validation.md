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

## Kazil and Lovejoy (2007) nucleation, `nucleation_scheme=2` (jax-gcm#1017)

`nucl_kazil_lovejoy` (mo_ham_m7_nucl.F90:320-668) is a 5-D log-linear
interpolation of a tabulated particle-formation-rate lookup (the PARNUC
table, `ham_nucl_initialize`). The table is MPI-M/HAMMOZ **data**, not
source, and is not shipped with this package; point `HAM_INPUT_DIR` at a
directory holding it (ECHAM itself opens it as `parnuc.15H2SO4.nc`; the
staged distribution name `parnuc.15H2SO4.A0.total.nc` is also accepted) to
run the gated tests below -- they otherwise skip, naming the variable and
file. The oracle for this scheme is built with
`build_oracle(..., kazil_table_path=...)`, which compiles the real
`nucl_kazil_lovejoy` (and the module's own table arrays) verbatim against a
binary serialization of the table (`start_nucl`, mirroring `start_kappa`'s
existing pattern -- no NetCDF Fortran library dependency).

Recorded 2026-10-06, `HAM_INPUT_DIR` pointed at
`/data/climate-analytics-lab-shared/ECHAM_emissions_v0006/hammoz/T63/hammoz`
(`parnuc.15H2SO4.A0.total.nc`), same toolchain as above. 306 cases (6
designed edge points -- both table-boundary clamp/zero branches, every
corner of the RH/[H2SO4] "set-zero" test -- plus 300 uniform-random draws
over T 190-300 K, RH 5-100%, [H2SO4] 1e4-1e10 cm⁻³, ion-pair 0-50 cm⁻³ s⁻¹,
sink 1e-5-1e-1 s⁻¹):

| Check | Max relative error (nonzero) | Zero/clamp branches match |
|---|---:|---|
| `kazil_lovejoy` kernel vs `nucl_kazil_lovejoy` | 2.0e-16 | exact |
| `nucleate_all(scheme=2)` vs `m7_nuck` (gas) | 3.0e-16 | exact |
| `step_all(nucleation_scheme=2)` vs full `m7` | ≤1e-12 (test gate; 7 designed HAM-population cells × random ion-pair rate) | exact |

All three are at float64 round-off, well inside the task's 1e-12 gate; no
tolerance needed loosening. (`nucleate_all`'s *compiled* path additionally
needs a small `atol` floor on a handful of near-total-consumption cells --
ordinary jit-vs-eager floating-point non-associativity in the clamp/division
chain, not a port discrepancy; see the test's own comment.)

Reproduce (from this checkout, `.venv` activated or prefixed as below):

```
HAM_INPUT_DIR=/data/climate-analytics-lab-shared/ECHAM_emissions_v0006/hammoz/T63/hammoz \
  JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q tests/test_m7_nucleation.py tests/test_m7_step_all.py
```

The reader's own parsing/indexing (shapes, the `H2SO4`/`[H2SO4]` name
fallback, finite/monotonic-axis validation) is covered unconditionally
(no `HAM_INPUT_DIR` needed) by a tiny synthetic fixture built in the test
file itself (`*_synthetic_format_test*`, never used for a physics number):

```
.venv/bin/python -m pytest -q tests/test_m7_nucleation.py -k synthetic_format
```

## Float32 forward path

`step_all` runs in float32 with `jax_enable_x64` off, using the
cancellation-free forms listed in the
[model contract](model.md#float32-precision-adaptations). Measured
2026-10-07 on CPU with JAX 0.10.2 and 0.11.2. The figures below are the same
in both. Float32 outputs on random cells differ between the two versions by
≤ 2×10⁻⁵ relative, from XLA code generation.

- **Float64 is unchanged bit for bit.** The native forms are still the
  float64 forms. Across 46 output arrays (4000 random cells; `dnum_all` with
  coagulation on and off; `step_all` for five scheme/organic combinations)
  every element is identical to the pre-change code. The full parity suite
  passes unchanged.
- **Equivalence in float64.** With the cancellation-free forms forced in
  float64, the designed parity cases agree with the compiled Fortran to
  ≤ 7.8×10⁻¹⁰ (`step_all` cold-ut KI mode) and ≤ 3.6×10⁻⁸ (`nucleate_all`
  depletion). Those differences are the Fortran's own double-precision
  cancellation error. 50-digit evaluations of the native expressions agree with the
  cancellation-free forms to ≤ 1e-12. The native float64 forms miss the same
  references by up to 4×10⁻³: the mixed-particle count of a 1e-9 cm⁻³
  insoluble population (`tests/test_m7_float32_precision.py`).
- **The review case** (4.4e5 NS on 52 KI particles cm⁻³, 220 K): KS sulfate
  and number were −1022.18 molecules cm⁻³ and −0.0287 cm⁻³, in eager and JIT
  execution. They are now 199.12965 and 0.0055973367; float64 gives
  199.12989 and 0.0055973396.
- **Designed cases against the Fortran**, every scheme/organic combination
  and field: ≤ 6.4×10⁻⁴ relative. The softest case is "gas-only" under
  nucleation, which carries the float32 Vehkamaeki rate. The cold
  upper-troposphere KI mode, previously 0.71, is now ≤ 2.0×10⁻⁵. Per-species
  and sulfur budgets are within ≤ 1.2×10⁻⁷.
- **Randomized properties** (`tests/m7_case_space.py`, 4096 seeded cells per
  configuration over the designed-case space; `dnum_all` with coagulation on
  and off, and `step_all` for (0,0), (1,0), (1,1) and (1,2)):
  - every mass, number, mass delta and gas is finite and ≥ 0;
  - `valid` holds for every cell;
  - BC, OC, SS and DU budgets hold to 1e-6 relative + 1e-14 µg m⁻³ absolute,
    the native `cmin_aerml` pttn gate;
  - the sulfur budget holds to 1e-6.
  With the native forms in float32, all six configurations fail. On 4000
  cells the native forms gave 1137 negative `dnum_all` mass entries and
  422–461 negative `step_all` mass entries.

## Remaining limits

This is a pure-sulfate core port with a controlled native metadata registry. It does not validate the complete HAM driver. Kazil/Lovejoy nucleation (`nucleation_scheme=2`, above) is validated against the real PARNUC table when `HAM_INPUT_DIR` names it; non-sulfate species and undefined native condensation-off/nucleation-on calls remain explicitly outside acceptance. Shared native formulas, hydration lookup discretization, molecular-weight distinctions and coagulation singularities remain as documented in [the model contract](model.md).
