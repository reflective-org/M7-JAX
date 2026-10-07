# Sulfate M7 model contract

## State and process order

M7 represents lognormal populations, not sectional bins. Four soluble modes
are retained: nucleation, Aitken, accumulation and coarse. Fine-mode geometric
standard deviation is 1.59; coarse-mode width is 2.0. Number arrays retain the
native seven-mode layout; all three insoluble entries must be zero. Sulfate
has four components in molecules/cm³; gas H₂SO₄ is molecules/cm³. Other
species do not enter the public state.

`step` follows the original complete M7 call:

1. Mean particle masses and hydrated radii/densities.
2. Analytical gas production/condensation.
3. Binary nucleation and gas consumption.
4. Modal coagulation and particle insertion.
5. Recalculate particle properties.
6. Simultaneous upward modal redistribution.
7. Recalculate final properties.

Radii are count-median radii in cm; densities are g/cm³; water output is kg/m³.
Temperature, pressure and timestep use K, Pa and seconds. Cloud fraction and
RH are fractions. `initialize` accepts four count-median **diameters** in nm.
It integrates modal volume using the native lognormal moment conversion.
Calling `validate_state` checks host inputs; `StepResult.valid` checks input
eligibility and finite output state during compiled integration.

## Separately callable processes

| API | Result |
|---|---|
| `coagulation.coaset` | Native 16 collision-pair coefficients, cm³/s |
| `coagulation.dnum` | Sulfate, seven-mode number and sulfate transfer increments |
| `condensation.condensation_sink` | Total and seven modal sinks, s⁻¹ |
| `condensation.produce_and_condense` | Updated gas/sulfate and insoluble deposit diagnostic |
| `nucleation.vehkamaeki` | Binary rate, cm⁻³ s⁻¹, and sulfate molecules per critical cluster |
| `nucleation.nucleate` | Remaining gas, new-particle number and new sulfate increments |

The coagulation transfer API is sulfate-only even though the coefficient leaf
can evaluate all native pairs. Insoluble deposition is unreachable for valid
coupled states. Scalar environment inputs broadcast over leading cell axes.
Process switches are static JIT arguments. This is not a promise of smooth
derivatives across clipping, lookup indices or branch thresholds.

## Native behavior deliberately retained

- `water_scheme=1` uses the authentic version-2 kappa lookup and native nearest
  coordinate selection. Temperature endpoints are 233–313 K; RH coordinates
  span 0.3–0.9878263. Lookup index clamping remains; this is not a dry switch.
- `water_scheme=0` uses the native Jacobson sulfate parameterization. It clamps
  temperature below 240 K and RH into 0.05–0.90.
- Kappa hydration uses the species registry molecular weight 96.0631 g/mol;
  condensation, legacy hydration and repartition use 98.0734 g/mol. Sulfate
  density is 1.841 g/cm³. These native distinctions are not harmonized.
- Empty nucleation modes have a virtual critical cluster for geometry, without
  adding physical sulfate or particle number.
- Condensation integrates constant gas production analytically. Above the sink
  threshold, cloud condensate goes to the largest occupied mode. With the
  condensation switch off, the native core ignores its production argument.
- Native Vehkamaeki temperature/RH/gas clamps and cluster/rate limiters remain.
  The port does not substitute SALSA's nucleation growth/survival treatment.
- Modal redistribution uses the native cumulative-normal approximation and
  epsilon tail cutoff. It conserves sulfate/number while moving material
  upward; compatibility does not prove that transfer is physically accurate.

## Float32 precision adaptations

`step_all` and everything it calls compute in the working dtype of their
inputs. Several native expressions obtain a small net quantity as the
difference of two nearly equal totals. Evaluated in float64, as the compiled
Fortran does, they keep enough digits; in float32 they lose all of them and can
change sign, which gives negative particle counts and masses. For each such
expression `m7_jax.precision.cancellation_free(dtype)` selects a form that is
**identical in real arithmetic** but computes the net quantity directly. The
choice depends only on the static working dtype: float64 uses the native forms
bit for bit, so the float64 parity suite against the compiled r7492 Fortran
does not change, and every narrower dtype uses the cancellation-free forms.
These are precision adaptations, not scientific deviations.

| Native routine (mo_ham_m7.F90) | Native difference | Cancellation-free form |
|---|---|---|
| `m7_delcoa` 2.2.3, mixed particles `zanytnm` (4379-4389) | `zanytnt - (zanytni + zanytns)`: two totals of ~T for a net of ~c·S·I | `[cS/(2+cS)]·[2I/(2+cI)]·[(4+cT)/(2+cT)]`; `2I/(2+cI)` when I ≤ 1 |
| `m7_delcoa` 1.2/2.2.1, no-source decay (4101-4283) | `1 - exp(-b dt)`; particles lost `zansum - zaernt` | `-expm1(-b dt)`; loss `n·G(na+b)/(naG+b)`, or `n·(na dt)/(1+na dt)` intra-only |
| `m7_delcoa` 2.2.2a, nucleating solution (4302-4329) | `(R·zf4 - b)/(2a)`, which tends to `b - b` | `[n0(2ER + G(R-b)) + 2cG]/[(1+E)R + G·zf2]` with `R - b = 4ac/(R+b)` |
| `m7_delcoa` 2.2.2b, `zbot` (4344-4359) | `zf1 - ze1`, both near 1 | `2R1/(n0-R1) - expm1(-2aR1 dt)` |
| `m7_concoag` ageing (3777-3822) | `paerml - zcrit·pttn·1e12`, exactly the whole mode when every particle ages | `(basis - zcrit)·pttn·1e12`, basis = KI's section-1 particle equivalent or the mode's number |
| `m7_prod_cond` (2063-2254) | gas `(g - P/k)E + P/k`; condensate `g - gas_left + P dt` | `gE + (P/k)G`; `gG + (P/k)(x + expm1(-x))` |
| `m7_nuck` (2257-2465) | `zdelta = clear - clear_final`; gas left `g - zdelta` | `clip(consumed, 0, clear)`; `clear_final·(1 - cloud)` plus any gas outside the cloud-free part |

Here `E = exp(-x)`, `G = 1 - E = -expm1(-x)` for the relevant `x`. Each form is
a ratio or sum of non-negative terms, so it is non-negative by construction. No
clamp was added. Native `MIN`/`MAX` limiters are kept as written.

Two subtractions stay in every dtype. In each case no closed form free of
cancellation exists, and the residual error is one rounding of an input
quantity, which is never amplified:

- With a source, the particles lost (`zansum - zaernt`) carry the rounding of
  `zansum`. The native `MIN(zaernt, zansum)` keeps them non-negative. The
  difference-free form would need `x(1 + e^-x) - 2(1 - e^-x)`.
- In `x + expm1(-x)`, a sink so small that `k dt` is below about 1e-4 leaves
  the cloud-free condensate with an absolute error of one rounding of `P dt`.
  That error is relative to the step's production, not to the deposit. The
  result stays non-negative in floating point: `-x` is representable and the
  true value lies above it.

The following were audited and are not affected:
- `m7_dconc`: `1 - zfconm` is exact by Sterbenz's lemma, and the
  fractions it acts on lie in [0, 1].
- `m7_concoag`: `number - zcrit` is a genuine difference with
  `zcrit = MIN(number, ...)`, so it is sign-safe.
- `m7_coat`: products only.
- `zansq - ztloss`: at least 5% of `zansq` by the native 0.95 cap.
- The pure-sulfate `delcoa`/`step` path: always float64.

Measured effects are in the [validation report](validation.md#float32-forward-path).

## Explicit exclusions and known numerical limits

The independent `nucleate`/`step` APIs (pure-sulfate state) support only
schemes 0 (off) and 1 (Vehkamaeki binary); scheme 2 is rejected there
unconditionally. `nucleate_all`/`step_all` (the 18-component HAM state)
additionally support scheme 2 (Kazil and Lovejoy 2007) given the PARNUC
lookup table (`load_kazil_lovejoy_table`, from `HAM_INPUT_DIR` -- MPI-M/HAMMOZ
data, not shipped; see `docs/validation.md`); without a table, scheme 2
raises there too, naming what is missing. Organic activation/kinetic
nucleation is outside the independent APIs' sulfur-only scope and is also
compiled out of the standalone native caller.

The original coupled routine does not initialize its sink when condensation
is disabled. Coupled nucleation therefore requires condensation enabled.
For a nucleation-only calculation, call `nucleate` with an explicit sink.

Native coagulation's analytical number solution has a singular equality case
`initial_number == sqrt(source/a)` and, in float64, cancellation-sensitive
tiny transfers (float32 uses the forms above).
The singular native domain is not scientifically repaired here; the JAX result
is invalid there. Benchmarks retain a one-second transfer diagnostic exceeding
the strict internal-transfer tolerance, even though its final number and sulfate
agree near 2×10⁻¹⁵ relative. See the validation report for details.

## Native reference isolation

The original complete `mo_ham_m7.F90` is compiled with a visibility-only overlay.
The harness supplies a documented 18-component metadata registry (four sulfate
plus unused BC/OC/sea-salt/dust slots), extracted original initialization and
nucleation leaves, and lossless serialization of the authentic kappa table.
The seven-mode/four-soluble registry mirrors the checked source definitions.
No scientific routines are translated into the reference. This verifies the
core and leaves; it does not claim validation of the full HAM initialization,
tracer interface or box driver.
