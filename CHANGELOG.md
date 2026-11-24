# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-24

The first release on PyPI (`pip install m7-jax`).

### Added
 - The full 18-component ECHAM-HAM M7 state: `step_all`, the complete native
   `m7` call with BC, OC, sea salt and dust, insoluble-to-soluble ageing,
   organic nucleation and Kazil-Lovejoy ion-mediated nucleation.
 - Float32 support through cancellation-free expression forms, and finite,
   finite-difference-checked float64 reverse-mode gradients.
 - A cell-minor structure-of-arrays path for GPUs and TPUs: `step_all_soa`
   with `SoAState`, the `m7_jax.interface.*_soa` converters, and
   `chunk_size`/`remat_chunks` scan chunking (`step_all` routes through it).
 - Host tracer conversions (`m7_jax.interface`) and the kappa lookup table as
   package data (the wheel is about 42 MB).
 - A GitHub Actions workflow that publishes to PyPI by trusted publishing when
   a GitHub release is published.

### Changed
 - The JAX floor is `jax>=0.10.2`, the oldest version tested; `requirements-dev.txt` pins the 0.11.2
   reference environment.

## [0.1.0] - 2026-10-13

Repository documentation and licensing; not published to PyPI.

### Added
 - CHANGELOG.md
 - CITATION.cff
 - CODE_OF_CONDUCT.md
 - CONTRIBUTING.md
 - CODEOWNERS
 - PULL_REQUEST_TEMPLATE.md

### Changed

- Updated the README to capture the updated license
- Updated the LICENSE to Apache 2.0

### Removed

- N/A
