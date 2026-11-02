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
  span 0.3–0.9878263. Lookup index clamping remains by default; this is not a dry
  switch (see the opt-in `unclamped_water` below).
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

## Opt-in fixes (default off = native)

- `unclamped_water=True` (`properties`, `step`; static option). With
  `water_scheme=1`, the native lookup clamps the RH index at the table's first
  coordinate (0.3) and temperature at 233 K, so at RH 0.05 and 220 K particles
  carry the water of the first table entry. That entry is the growth of RH 0.33,
  not 0.30: every entry at RH node *i* matches the kappa-Koehler growth at RH
  node *i*+1 (5 µm dry radius, Kelvin term negligible: median 0.02 %, against
  1 % at node *i*). With the flag on, states below either axis get
  the growth factor solved from the table's own kappa-Koehler form at the actual
  T and RH (`properties.kappa_kohler_growth`):
  RH = a_w(g) exp(A/(r_d g)), a_w = (g³−1)/(g³−1+κ),
  A = 2σ_w(T)M_w/(RTρ_w), σ_w = 0.0761 − 1.55e-4 (T − 273.15) N/m,
  M_w 18.015 g/mol, ρ_w 1000 kg/m³, κ the native volume-weighted value. This form
  reproduces the table, read at RH node *i*+1, within 0.5 % on every checked entry
  (median 0.07 %; `tests/test_m7_unclamped_water.py`). Inside the table range the
  native lookup is unchanged (bit-for-bit), and the solve caps RH at the table top
  as the lookup does. At 220 K the temperature clamp alone changes the 25 nm
  growth factor by 1e-4; the RH clamp changes it from 1.0782 to 1.0091.
  Reported upstream in TUT-ISI/SALSA#10 (minor bundle, item B).

## Explicit exclusions and known numerical limits

Only nucleation schemes 0 (off) and 1 (Vehkamaeki binary) are supported. Scheme
2 needs the absent Kazil/Lovejoy table; the harness fails explicitly if called.
Organic activation/kinetic nucleation is outside this sulfur-only scope and is
also compiled out of the standalone native caller.

The original coupled routine does not initialize its sink when condensation
is disabled. Coupled nucleation therefore requires condensation enabled.
For a nucleation-only calculation, call `nucleate` with an explicit sink.

Native coagulation's analytical number solution has a singular equality case
`initial_number == sqrt(source/a)` and cancellation-sensitive tiny transfers.
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
