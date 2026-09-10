# Changelog

Notable changes to MilliHertzQML. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Committed `.JuliaFormatter.toml` (default style, 92-column margin, spaced
  keyword arguments); the tree is formatted with it.
- Auxiliary environments: `scripts/Project.toml`, `bench/Project.toml`, and
  `test/Project.toml` consume the package by path through `[sources]` and
  self-activate; the root project declares only what `src/` loads.
- Static QA in the test suite: `Aqua.test_all`, `JET.test_package`, and
  the ExplicitImports checks (the module now imports every name
  explicitly); test randomness from `StableRNGs`; the classifier
  constructor accepts an `rng` keyword.
- End-to-end pipeline smoke test: generation, preprocessing, training,
  labeled and blind inference on a three-day configuration inside a
  temporary directory, driven by the new `[paths]` configuration section
  (`inputs`, `models`, `plots`, `results`) that every script honors.
- `CHANGELOG.md` and `CITATION.cff`.

### Changed
- CI: the blocking test job runs on Julia 1.12 (the compat floor the
  Manifests are resolved on) with coverage; a job on the current stable
  release is advisory; formatting and documentation-build jobs added.

## [0.1.0] — 2026-08-07

### Added
- Variational quantum classifier with data re-uploading (`Yao.jl`),
  binary cross-entropy loss, `Zygote.jl` gradients, `Flux.jl` Adam with
  exponential learning-rate decay and early stopping.
- Simulated LISA-like telemetry (HDF5 strain plus point-wise labels),
  sliding-window feature extraction, training and inference scripts with
  ROC-derived thresholds persisted per run, blind inference.
- Fail-fast validation on public interfaces and on configuration load;
  JLD2 model persistence; run provenance snapshots with a hardware
  fingerprint; path-independent scripts.
- Unit tests, `BenchmarkTools` benchmarks, Documenter manual, CI workflow.
