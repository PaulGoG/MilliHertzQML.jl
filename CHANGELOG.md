# Changelog

Notable changes to MilliHertzQML since its public release. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Features are encoded on `[0, π]` instead of `[0, 2π]`. The encoding
  gate `R_z` is 2π-periodic with `R_z(2π) = −I`, so the full period sent
  both ends of the scaler interval to the same state and a feature
  saturated above the training range scored as one at the noise floor: on
  the Sangria blind year five of the six coalescences fell below the
  threshold at the merger itself. The span is a configuration key
  (`[training] phase_span`, in units of π) persisted with the scaler;
  artifacts written before it existed load with `2π` and a warning. Every
  result on the benchmark page was produced under the old encoding and is
  labelled so; the retraining is pending. Models trained from now on are
  not comparable to the shipped ones.
- The decision threshold is fitted on the validation block by default
  (`threshold_block = "validation"`); pooling validation and test is an
  opt-in that the Sangria configurations declare. The pooled block never
  leaked into the model, but it made the test block's operating-point
  metrics in-sample, which the documentation had described as costing
  nothing.
- The alert protocol of the telemetry replay requires persistence: an
  alert is raised by the arrival that completes
  `[telemetry] alert_persistence` consecutive alarmed windows (three by
  default), and shorter runs are neither alerts nor charged as false-alarm
  episodes. The lead times of the first release rested on isolated one-
  or two-window alarms up to four days before the merger; under the
  persistence criterion no alert precedes its merger by more than twenty
  minutes, and the false-alarm rate of the year replay falls from 3.71 to
  1.16 episodes per 30 days under ground-causal whitening (2.56 to 0.74
  under the oracle PSD). The value is fixed on the calibration block of
  the training year: the smallest persistence under one false alert per
  30 days that keeps every calibration event alerted with a window of
  margin. `alert_latency_table` takes `persistence` and records it.
- `min_coverage` bounds the delivered fraction of a window's conditioning
  stretch; the window's own rows must all be on the ground. A window with
  rows of its own missing could previously be emitted by the scheduler
  and then skipped without record.
- The warning on a threshold fitted from few calibration episodes reads
  its count from `[training] min_fit_episodes`.
- The figures are built on the base layout — 900 × 600 pt single panel,
  26 pt type, 3 pt data lines — instead of an 86 mm column; the
  false-alarm rate, the decision threshold and a requested rate keep one
  colour each across every figure; the provenance sidecar records the
  canvas actually exported; a strain axis whose multiplier would be 10⁰
  or 10¹ folds it into the tick values.
- The configuration files live under `configs/` (`configs/default.toml`,
  `configs/sangria.toml`, `configs/sangria_paper.toml`,
  `configs/experiments/`); the scripts default to `configs/default.toml`.

### Added
- A constellation response for the simulator (`[generation] response =
  "lisa"`), as a package extension on CurvatureDistinguishability.jl: the
  record holds the A and E channels, each with independent noise at the
  Michelson-channel level (`R(f) S_n(f)`), the MBHB injections at physical
  amplitude for a luminosity distance drawn from
  `[mbhb_distance_min_gpc, mbhb_distance_max_gpc]` with isotropic sky
  position, inclination and polarization, projected frequency by frequency
  at the arrival time of each frequency on the antenna patterns, Doppler
  phase and transfer roll-off of the orbits; the background sources are
  projected in the time domain and scaled to a network SNR. Labels and the
  catalog SNR follow `label_channel` (A, E or network); the catalog records
  the extrinsic parameters and both channel SNRs; the HDF5 record carries
  `X`, `Y`, `Z` recombining to A, E and a vanishing T. The pre-processor
  whitens such records with `psd = "channel"`. `phenoma_spectrum`,
  `phenoma_series`, `phenoma_physical_amplitude` and
  `phenoma_arrival_delay` expose the pieces of the waveform; the
  sky-averaged generator is unchanged bit for bit.
- Ground-causal whitening for the replay (`[telemetry] psd_mode =
  "trailing"`, `TrailingWelch`): each window is whitened by the median
  Welch estimate of the delivered record behind its conditioning stretch,
  refreshed as the record advances, so that no data still in flight
  enters the estimate; the scored-window table records the last row
  behind each estimate (`psd_row`). The Sangria configuration replays
  under it (`psd_mode = "trailing"` in `configs/sangria.toml`), and the
  benchmark page quotes the
  causal replay — the same five coalescences at 1.16 false-alarm
  episodes per 30 days — with the oracle replay of the year-median
  sidecar PSD, 0.74, beside it as a bound: at the first event that PSD
  contains nine months of undelivered data.
- `threshold.toml` files written before the fitting block became
  configurable are read through `migrate_threshold_info!`, which renames
  their `validation_*` rates to `fit_*`.

### Fixed
- The benchmark page discloses the shipped model as the best of seven
  blind-year evaluations; counts five coalescences detected on their own,
  event 2 being only ever alarmed by event 1's windows; describes the
  classical baseline's refitted scaler and this pipeline's blind-year
  whitening symmetrically; attributes the inspiral-time alerts to the
  encoding; and defines every variable of its reproduction block.
- The LISA conventions document is cited by its public version (Baghi et
  al. 2026, arXiv:2603.22377) with its DOI.

## [1.0.0] — 2026-09-14

First public release: the four-stage pipeline (simulated telemetry or LDC
truth-stream labels, windowed whitened features, training, inference),
the Sangria benchmark with its provenance regenerated from a committed
tree, the telemetry coupling with replay and live modes, the
loss-tolerance and seed-spread studies, and the Documenter manual.
Distribution is by clone, `Pkg.develop`, or a `[sources]` entry pinned to
the tag; the package is not registered.
