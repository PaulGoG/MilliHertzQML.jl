# Changelog

Notable changes to MilliHertzQML. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `edge_margin` under `[preprocessing]` (window lengths, default 0; 10 in
  the Sangria configurations): the first and last windows of a record,
  where the circular high-pass and whitening filters ring, are dropped
  from the feature and label products; the sidecar records
  `first_window`, `edge_margin_windows`, and `n_windows_record`, and
  inference places every row at its record window index.
- `feature_set = "bands"`: the mean whitened power of every band between
  the ascending `band_edges_hz` of `[preprocessing]`, followed by the
  spectral entropy and the log power spread (`length(band_edges_hz) + 1`
  features; the default edges reproduce the whitened set exactly). The
  edges are recorded in the feature sidecar and honoured by the streaming
  detector; `feature_names` takes `n_bands`.
- Threaded training and inference: `batch_gradient` cuts the batch into
  consecutive chunks of `chunk_size` samples (default 4), evaluates one
  Zygote tape per chunk over the Julia threads, and reduces the chunk
  gradients in chunk order (deterministic, independent of the thread
  count; equal to the serial single-tape gradient up to accumulation
  rounding); `train_step!` takes `threaded`, the `[training] threaded`
  key (default `true`) selects the path, and the validation forward pass,
  the threshold fit, and inference score their rows over the same threads.
  `weighted_bce` and `sample_loss` factor the loss; `predict_probability`
  is the single functional forward pass of the circuit (the mutating
  `build_step` path is gone). Benchmarks of both paths in `bench/`.
- `threshold_sweep`: the event-level operating characteristic of a scored
  block (window precision, recall, and false-positive rate; events
  detected; false-alarm episodes per 30 days) at every candidate
  threshold, persisted by training as `threshold_sweep.csv` of the
  validation block and by labeled inference as the post-hoc sweep of the
  evaluated rows; the `figure_threshold_sweep` figure (`threshold_sweep`)
  with the operating point and the `far` target. `event_metrics` reports
  the window false-positive rate (`fpr`).
- Pipeline architecture: every stage is a typed library function
  (`generate_telemetry`, `label_truth_stream`, `preprocess_record`,
  `train_classifier`, `evaluate_classifier` under `src/stages/`) returning
  its artifacts; the scripts are thin dispatchers.
- `src/config.jl`: validated settings of every configuration section
  (`generation_settings`, `preprocessing_settings`, `model_settings`,
  `training_settings`, `inference_settings`, `ldc_settings`,
  `resource_settings`), path resolution against the package root.
- `src/provenance.jl`: git description and package version beside the
  hardware fingerprint in every snapshot (`write_toml`), overwrite-safe
  writing with DrWatson-style `_#k` backups (`backup_existing!`,
  `write_csv`), run identifiers, the `[resources]` memory guard with
  pre-flight estimates (`training_memory_estimate_gib`,
  `record_memory_estimate_gib`, `check_memory`), and the stage timer
  (`TIMER`, `report_timing`). DrWatson, TimerOutputs, and the provenance
  standard libraries become package dependencies.
- Preprocessing records a hash of every parameter that determines a
  feature product and reuses an identical product unless `--force`.
- Telemetry coupling to DeepSpaceTelemetry.jl (`src/telemetry.jl`,
  `ext/MilliHertzQMLDeepSpaceTelemetryExt.jl`): the run interface
  (`open_telemetry_run`, `run_geometry`, `list_batches`, `read_batch`,
  `arrival_events`, `run_state`) implemented over the producer's API, the
  batch-to-row geometry, the coverage set of delivered rows, the window
  scheduler, the streaming detector reproducing the batch conditioning on
  the delivered stretch around a window (`StreamingDetector`,
  `detector_from_run`), replay and live modes (`replay_run`,
  `follow_run`), and the alert-latency table (`alert_latency_table`);
  the payload export stage (`export_telemetry_payload`,
  `scripts/export_telemetry_payload.jl`) and the consumer script
  `scripts/infer_telemetry.jl`; the `[telemetry]` configuration section;
  the `figure_telemetry_alerts` figure; unit tests on an in-memory run and
  an integration test that runs a producer mission in a temporary root.
  DeepSpaceTelemetry is a weak dependency of the package and a pinned git
  source of the script and test environments.
- Publication figures as a CairoMakie package extension
  (`src/visualization.jl`, `ext/MilliHertzQMLCairoMakieExt.jl`): one
  theme at the 86 mm single-column width (Computer Modern, boxed axes,
  no titles, legend above the axes, Okabe–Ito colors, offset multiplier
  for strain amplitudes), one function per figure (`figure_training_history`,
  `figure_mission_trace`, `figure_roc`, `figure_sensitivity`,
  `figure_score_distribution`, `figure_telemetry_trace` with a whitened
  panel), and `save_figure` exporting PDF and 4× PNG with a provenance
  sidecar. Plots.jl is dropped from the script environment.
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
  SNR, masses and transition frequencies, covered and labeled ranges).
- `src/waveforms.jl`: the IMRPhenomA inspiral–merger–ringdown model
  (Ajith et al. 2008, Table I) generated in the frequency domain on the
  injection segment's sampling grid (`phenoma_parameters`,
  `phenoma_waveform`): coalescence placed through the phase's group
  delay, inspiral roll-on at the segment start, cosine taper below
  Nyquist. Replaces the phenomenological chirp (Nyquist aliasing,
  unrelated ringdown). Masses drawn log-uniformly in
  `[mbhb_total_mass_min, mbhb_total_mass_max]`, mass ratio in
  `[1, mbhb_mass_ratio_max]`.
- `label_span = "detectable" | "injection" | "fixed"`: by default the
  positive span is the union of the windows whose matched-filter SNR
  reaches `label_snr_threshold` (`detectable_span`), so labels cover the
  signal an optimal single-window filter can see.
- `FeatureScaler`, `fit_scaler`, `encode_features`: per-feature quantile
  bounds fitted on the training partition, persisted in the JLD2 model
  artifact, and applied at inference; `load_model` returns the scaler.
- `src/evaluation.jl`: `chronological_split` (contiguous training,
  validation, and test blocks with a buffer), `roc_curve`/`roc_auc`,
  `contiguous_runs`, `event_metrics` (window-level precision, recall,
  F1, balanced accuracy; event recall; false-alarm episodes outside the
  labeled spans per 30 mission days), and `select_threshold` (`far`,
  `fpr`, or `youden` criterion on the validation block).
- `loss_function`/`train_step!` accept `positive_weight`; training
  balances the classes by the negative-to-positive count ratio
  (`class_weight`).
- Preprocessing writes a geometry sidecar `<features>.toml` (window
  size, step, sampling rate, bands, source) that training and inference
  read through `feature_geometry`.
- Training writes `split.toml`, `threshold.toml` (fitted on the validation
  block), and `metrics.toml` (validation and test blocks); inference
  writes `metrics.toml` for labeled rows and accepts `--block
  validation | test`.
- `src/ldc.jl`: the analytic TDI noise PSD of the `ldc` package
  (`ldc_tdi_psd`, `ldc_confusion_psd`; equal arms, X/XY/A/E/T, TDI 1.5
  and 2, named noise levels), readers of the compound HDF5 TDI datasets
  and catalogs (`read_tdi`, `read_catalog`), `tdi_to_aet`, a
  median-averaged Welch estimate (`welch_psd`) with log-log
  interpolation (`interpolated_psd`), and truth-stream labeling
  (`windowed_snr`, `snr_peaks` with a merger threshold and a precursor
  rule for inspiral fluctuations, `detectable_spans`, `fixed_spans`,
  `span_labels`). HDF5 becomes a package dependency.
- `scripts/label_ldc.jl`: point-wise labels and an event table of an LDC
  product from its truth stream and catalog, or from a signal-only CSV
  for the blind set; `[ldc]` configuration section.
- Preprocessing selects the whitening PSD (`[preprocessing] psd = "model"
  | "ldc" | "welch" | "none"`) and the feature set (`feature_set =
  "whitened" | "paper"`), reads the sampling step from the file, and
  persists the Welch estimate beside the features.
- Sangria validation anchors in the test suite, gated on
  `MILLIHERTZQML_LDC_DIR`: the School-notebook SNR of catalog source 4
  (1883.5 against 1885.7) and a noise-only null test of the whitening
  chain.
- `config_sangria.toml` and `config_sangria_paper.toml`: the Sangria
  benchmark configurations (Welch-whitened features; the paper's
  raw-window feature set).

### Changed
- The `far` threshold criterion is the operating point of an alert
  trigger: candidates are scanned from the highest threshold downwards and
  the threshold is the lowest of the admissible range that starts at the
  top, where a candidate is admissible when its false-alarm episode rate
  does not exceed `target_far_per_30d` and its window false-positive rate
  (the alarm duty cycle on unlabeled windows) does not exceed
  `target_fpr`. The previous ascending scan accepted the permanently
  raised alarm — few long episodes — as soon as the target admitted a few
  episodes per month. The default `target_far_per_30d` is 3 (trigger
  level; a trigger costs a characterization pass, not an alert).
- Command-line interface: the configuration file is the first (positional)
  argument of every script and the single source of every parameter; the
  scripts keep only `--run-id`, `--test-mode`, and the location of
  external inputs (`--h5-file`, `--label-file`, `--truth-csv`,
  `--features`, `--labels`, `--model`, `--block`, `--output-prefix`,
  `--output`, `--force`). Numeric overrides are removed.
- Evaluation protocol: the random shuffle over overlapping windows is
  replaced by the chronological block split; early stopping and the
  decision threshold use the validation block only and the test block is
  scored once. Configuration keys `train_fraction`,
  `validation_fraction` (now 0.15), `class_weight`,
  `threshold_criterion`, `target_far_per_30d`, `target_fpr` under
  `[training]`; `[inference]` loses `target_fpr` and gains `block`.
- Inference no longer fits a threshold: it requires the `threshold.toml`
  written at training time.
- `confusion_psd` evaluates the Robson–Cornish–Liu foreground in log
  space, so the sensitivity stays finite far above the knee frequency
  (the direct product gave `0 × Inf` above a few hertz).
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
- Toolchain: every environment (root, `test/`, `scripts/`, `bench/`,
  `docs/`) is re-resolved on Julia 1.13.0, taking the newest compatible
  versions (Yao 0.9.3, Zygote 0.7.13, Flux 0.16.11, JLD2 0.6.6); the
  `[compat]` floor stays at 1.12.
- Inference: the ROC curve, its area, and the threshold selection are
  computed in `scripts/infer.jl` (`roc_points`, `roc_area`); a labeled
  set containing a single class is rejected with an `ArgumentError`.
- CI: the workflow runs on manual dispatch only until the repository is
  public (GitHub Actions minutes are unavailable on the private
  repository); the matrix tests the current stable release with coverage
  and the 1.12 compat floor; formatting and documentation-build jobs
  added. The README badges are removed for the same period.
- `predict` and `accuracy` documented.

### Removed
- `EvalMetrics` from the script environment: unmaintained since 2024-07
  and broken on Julia 1.13, where its unqualified `ispositive` collides
  with the new `Base.ispositive`.

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
