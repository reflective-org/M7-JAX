# Repository Guidelines

## Scope and Layout

M7-JAX supports sulfuric-acid coagulation, condensation, binary nucleation and
the modal properties, hydration and redistribution they need. Keep unrelated
materials, HAM coupling and radiation outside this scope. `src/m7_jax/` contains
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
with empty insoluble entries, and scalar gas per cell. Centralize scientific
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
