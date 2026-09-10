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
- `src/simulation.jl`: the sky-averaged sensitivity of Robson, Cornish &
  Liu (2019) with the selectable confusion fit (`lisa_noise_psd`,
  `instrument_psd`, `confusion_psd`), calibrated Gaussian-noise synthesis
  (`synthesize_noise`), the matched-filter SNR (`matched_filter_snr`,
  `scale_to_snr`), record-level high-pass and whitening (`highpass_record`,
  `whiten_record`), the normalized tapered periodogram
  (`tapered_periodogram`), and `place_signal!`.
- Event catalog CSV written by the simulator (merger sample and time,
  SNR, waveform parameters, covered and labeled ranges).
- `FeatureScaler`, `fit_scaler`, `encode_features`: per-feature quantile
  bounds fitted on the training partition, persisted in the JLD2 model
  artifact, and applied at inference; `load_model` returns the scaler.

### Changed
- Simulator: noise synthesized at physical strain amplitude against the
  corrected PSD (the previous confusion term vanished above 0.2 mHz and
  the instrument term lacked the 10/3 and transfer factors); every
  injected source is scaled to a matched-filter SNR (`snr_min`/`snr_max`
  for MBHBs, default [8, 50]; `gb_snr_*`, `emri_snr_*` for the
  background); MBHB waveforms are anchored on the coalescence sample and
  truncated to the record instead of being shifted; the label span is
  configurable (`label_before_sec`, `label_after_sec`); the inspiral
  duration is `min(mbhb_duration_days, 0.4 T)`; noise is synthesized
  from `noise_f_min_hz` (default 1e-5 Hz) upwards.
- Features: the record is high-passed (zero-phase, `highpass_cutoff_hz`,
  `highpass_order`) and whitened by the model PSD before windowing, and
  every window's Hann-tapered periodogram has unit mean for noise;
  PSD-normalized band powers, entropy normalized by `ln N`; band edges
  and the whitening fit are configuration keys (`[preprocessing]`); the
  fixed `FEATURE_SCALES`
  clamps are removed. `load_data`/`load_features` return raw features;
  encoding is the scaler's job. Feature column `log_psd_std` renamed
  `log_power_std`.
- Preprocessing refuses an output path that coincides with an input.
- Inference: model artifacts without a scaler are rejected; the
  sensitivity-versus-SNR figure bins the observed SNR range.
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
