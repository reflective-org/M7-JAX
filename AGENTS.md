# Repository Guidelines

## Scope and Layout

M7-JAX supports the full 18-component ECHAM-HAM M7 state (four soluble sulfate
modes plus BC, OC, sea-salt and dust in soluble and insoluble modes) and the
processes acting on it: coagulation with insoluble ageing, H2SO4 condensation,
binary (Vehkamaeki), Kazil/Lovejoy and organic nucleation, and the modal
properties, kappa hydration and redistribution they need. The pure-sulfate
`step` API remains for the four-soluble-mode case. Supported configuration:
`nwater=1` (kappa lookup) for the multi-species state, `nwater=0` for sulfate
only, `nsnucl` 0/1/2, `nonucl` 0/1/2, `nsoa=0`, and a float32 forward path.
Keep unrelated materials, HAM coupling beyond the M7 call and radiation outside
this scope. `src/m7_jax/` contains
JAX kernels; `reference/fortran/` preserves native sources; `reference/input/`
holds authentic lookup data. Tests live in `tests/`, reference drivers and
simulation commands in `tools/`, benchmarks in `benchmarks/`, configuration in
`configs/`, and recorded evidence in `docs/evidence/`.

## Development Commands

Use this checkout's `.venv`; never install into system Python. Install
`requirements-dev.txt`, then `pip install --no-deps -e .`.
GNU Fortran is mandatory for native comparisons.

- `.venv/bin/python -m pytest -q`: run native and JAX regression tests.
- `.venv/bin/python -m tools.run_m7_sulfate --config configs/m7-sulfate.toml --output outputs/example`: compare every trajectory step.
- `.venv/bin/python -m benchmarks.benchmark_m7 --help`: inspect benchmark options.

Use fresh output directories. Keep environments, build products, caches and
large generated trajectories out of commits. Commit compact evidence with
configuration, source/data hashes, compiler and numerical results.

## Code and Scientific Contracts

Use four-space Python indentation and descriptive snake_case names. Match
existing style; no formatter is prescribed. Enable JAX float64. Document units
and trailing array dimensions: sulfate four soluble modes, number seven modes
with empty insoluble entries for the sulfate API, the 18-component mass array
for the full state, and scalar gas per cell. Centralize scientific
constants and keep tunable experiment settings in configuration files.

Preserve native formulas and branch order during compatibility work. Treat
scientific changes as separately documented and tested work. Never replace a
missing lookup with an undocumented approximation.

## Workflow and Verification

State assumptions, keep changes focused, and use independent agents/worktrees
when useful. See `docs/PLAN.md` for commit-sized tasks. Reproduce defects before
fixing them; remove only code made unused by your changes. Tests use
`test_*.py`, actual checked Fortran, eager/JIT paths and sulfur budgets. No
coverage quota is set. Commit descriptive, focused changes. PRs must state the
supported configuration, checks, measured errors and remaining limitations.
