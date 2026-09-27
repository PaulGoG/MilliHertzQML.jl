# Changelog

Notable changes to MilliHertzQML since its public release. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `figure_gap_study`: the levels of a delivery-gap study on the vertical
  axis, three panels beside them (windows scored of the reference, events
  detected, false alarms per 30 days), families separated.
- `welch_psd` accepts a vector of records and pools the segments of every
  record long enough to hold one, so that an estimate of a record with
  holes never spans a hole.
- The benchmark page's lossy-link section is rewritten from a
  seventeen-mission study on DeepSpaceTelemetry.jl 2.0.0 (scattered,
  bursty and retransmitted loss, link outages, generation gaps, recorder
  overflow, partial conditioning), replayed under causal whitening (the
  PSD estimated only from data already delivered to the ground station)
  and under the full-record PSD (the median Welch estimate of the entire
  blind year, which is available only after the whole record has been
  received).
- `smooth_psd` and `[preprocessing] psd_smoothing_dex` (0, off, by
  default): the Welch estimate that whitens a record, and the trailing
  estimate of a replay (`TrailingWelch` `smoothing_dex`), smoothed by a
  Gaussian in log-frequency. A 0.01-dex smoothing lowers the envelope of
  the whitening kernel at one window length from 21 % to 0.1 % of its
  peak; applied to the blind year with the selected model and its
  threshold unchanged it keeps five of five label spans at 4.78 instead
  of 1.57 false-alarm episodes per 30 days, so a model trained on
  smoothed features is required to use it.
- A test of the labelling stage on a synthetic truth stream; coverage
  now includes the stages run by the scripts of the smoke test and the
  three package extensions.

### Changed
- The trailing whitening estimate of a replay pools the Welch segments of
  every delivered run inside its span instead of using the last contiguous
  run only, trims `[telemetry] psd_edge_periods` cutoff periods of the
  record high-pass from both ends of every run before segmenting it (the
  high-pass rings at the ends of a run as it does at the ends of the batch
  record), and keeps its previous estimate while no run holds a segment
  beyond the trim. Behind a few permanent holes the contiguous record held
  one or two segments and the estimate was close to a raw periodogram: on
  30-day missions the false-alarm rate rose several-fold and at 0.3 % loss
  both coalescences went undetected, while the replay under the
  full-record PSD retained the reference rate. `TrailingWelch` takes `edge_rows`. The causal year
  replay of the selected model changes with it: 23 false-alarm episodes
  (1.90 per 30 days) instead of 17 (1.40), a difference within the
  Poisson uncertainty of either count; the same five coalescences are
  detected on their own; the alerts of events 1 and 4 move from 1.9 and
  7.5 hours after their mergers to 25.6 and 16.1 hours before them in
  data time, within the conditioning stretch that contains the merger.
- `configs/experiments/q8_b6.toml` carries the `[telemetry]` section of
  the replay of its model; `configs/sangria.toml` keeps one for its own
  model and for the payload export, with the whitening sidecar of its own
  features.

### Fixed
- A replay of a producer that discarded production (a generation gap, a
  recorder overflow) raised a `KeyError` on the first window after the
  gap: the delivered payload was indexed by batch index, which no longer
  tracks the payload rows after such a gap. It is keyed by the first row
  a batch holds, and a stretch is assembled from the blocks that cover it.
- The physics page states the ecliptic-frame sky and polarisation
  conventions of the constellation response with the equations of the
  conventions document they follow.
- `figure_telemetry_alerts` labels each alert with its data latency
  t_alarm − t_merger, the quantity of its marker position and of the
  benchmark tables; the labels carried the total latency, one hour more.
  The label of the lower of two neighbouring alerts is set beneath its
  marker, where the two overlapped, and the delivery trace is drawn
  translucent so that labels on it remain legible.
- The alarm-episode panel of the replay animation ticks from zero up to
  the final episode count (`count_ticks`); its top tick lay on the panel
  edge and collided with the zero of the probability panel above.
  The inner panels of the gap-study figure tick every level.

## [1.1.0] — 2026-09-26

First public version of the repository. The models are retrained on the half-period encoding and the benchmark
page is rewritten from those runs. The selected run is `q8_b6_pi`: the
configuration of the 1.0.0 tag, chosen again on the validation block
of the training year by a pre-registered rule, delivering all five blind
label spans at 1.57 false-alarm episodes per 30 days (fit 1.38, ROC area
0.807) against 2.47 (fit 2.20, 0.793) before. Every model of the grid, the
three further seeds and the paper-feature parity run were retrained and
scored once; the ranking between configurations lies inside the seed
spread and is reported as unresolved.

### Changed
- Features are encoded on `[0, π]` instead of `[0, 2π]`. The encoding
  gate `R_z` is 2π-periodic with `R_z(2π) = −I`, so the full period mapped
  both ends of the scaler interval to the same state and a feature
  saturated above the training range scored as one at the noise floor: on
  the Sangria blind year five of the six coalescences fell below the
  threshold at the merger itself. The span is a configuration key
  (`[training] phase_span`, in units of π) persisted with the scaler;
  artifacts written before it existed load with `2π` and a warning. The
  retrained grid is part of this release; the full-period runs of the
  1.0.0 tag remain only in the history of the page.
- The decision threshold is fitted on the validation block by default
  (`threshold_block = "validation"`); pooling validation and test is an
  opt-in that the Sangria configurations declare. The pooled block never
  leaked into the model, but it made the test block's operating-point
  metrics in-sample, which the documentation had described as having no
  effect.
- The alert protocol of the telemetry replay requires persistence: an
  alert is raised by the arrival that completes
  `[telemetry] alert_persistence` consecutive alarmed windows (three by
  default; two in the Sangria configuration, fitted on the retrained
  model's calibration block), and shorter runs are neither alerts nor
  counted as false-alarm episodes. The lead times of the 1.0.0 tag rested on isolated one-
  or two-window alarms up to four days before the merger; under the
  persistence criterion no alert precedes its merger by more than twenty
  minutes, and the false-alarm rate of the year replay falls from 3.71 to
  1.16 episodes per 30 days under causal whitening (2.56 to 0.74 under the
  full-record PSD) for the full-period model, and 6.35 to 1.40
  (full-record PSD 1.65 to 1.32) for the retrained one at its persistence
  of two.
  The value is fixed on the calibration block of the training year: the
  smallest persistence under one false alert per 30 days that keeps every
  calibration event alerted with a window of margin. `alert_latency_table`
  takes `persistence` and records it.
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
- The script and test environments pin DeepSpaceTelemetry.jl 2.0.0 and
  CurvatureDistinguishability.jl 2.0.1 (the commits of their release
  tags), and `[compat]` admits both producers at 1.x and 2.x.
  DeepSpaceTelemetry 2.0.0 leaves the run-directory contract the consumer
  reads unchanged (configuration snapshot, batch metadata and naming,
  segment loader, arrival log, lifecycle sentinels), and the integration
  test passes against it. CurvatureDistinguishability 2.0.0 brings the
  confusion fit of its noise PSD to the channel level; the extension never
  used that model, its channel PSD `R(f) S_n(f)` being defined here, so no
  number changes.
- The configuration files live under `configs/` (`configs/default.toml`,
  `configs/sangria.toml`, `configs/sangria_paper.toml`,
  `configs/experiments/`); the scripts default to `configs/default.toml`.

### Added
- Continuous integration on push, tag and pull request: the test suite on
  the compat floor (1.12) and on the current release with coverage upload,
  a formatting check, the manual deployed to GitHub Pages, and Dependabot
  for the Julia and GitHub Actions ecosystems.
- A constellation response for the simulator (`[generation] response =
  "lisa"`), as a package extension on CurvatureDistinguishability.jl: the
  record holds the A and E channels, each with independent noise at the
  Michelson-channel level (`R(f) S_n(f)`), the MBHB injections at physical
  amplitude for a luminosity distance drawn from
  `[mbhb_distance_min_gpc, mbhb_distance_max_gpc]` with isotropic sky
  position, inclination and polarisation, projected frequency by frequency
  at the arrival time of each frequency on the antenna patterns, Doppler
  phase and transfer roll-off of the orbits; the background sources are
  projected in the time domain and scaled to a network SNR. Labels and the
  catalogue SNR follow `label_channel` (A, E or network); the catalogue records
  the extrinsic parameters and both channel SNRs; the HDF5 record carries
  `X`, `Y`, `Z` recombining to A, E and a vanishing T. The pre-processor
  whitens such records with `psd = "channel"`. `phenoma_spectrum`,
  `phenoma_series`, `phenoma_physical_amplitude` and
  `phenoma_arrival_delay` expose the pieces of the waveform; the
  sky-averaged generator is unchanged bit for bit.
- Causal whitening for the replay (`[telemetry] psd_mode =
  "trailing"`, `TrailingWelch`): each window is whitened by the median
  Welch estimate of the delivered record behind its conditioning stretch,
  refreshed as the record advances, so that no data still in flight
  enters the estimate; the scored-window table records the last row
  behind each estimate (`psd_row`). The Sangria configuration replays
  under it (`psd_mode = "trailing"` in `configs/sangria.toml`), and the
  benchmark page quotes the causal replay — five coalescences alerted on
  their own at 1.40 false-alarm episodes per 30 days with the retrained
  model — with the replay under the full-record sidecar PSD, 1.32, beside
  it as an upper reference for the causal replay: at the first event that
  PSD contains nine months of undelivered data.
- `threshold.toml` files written before the fitting block became
  configurable are read through `migrate_threshold_info!`, which renames
  their `validation_*` rates to `fit_*`.

### Fixed
- The benchmark page selects the model on the training year by a
  pre-registered rule and reports every run; counts five coalescences
  detected on their own, event 2 being alarmed only by windows of the
  alarm cluster of event 1; describes the
  classical baseline's refitted scaler and this pipeline's blind-year
  whitening symmetrically; attributes the inspiral-time alerts to the
  encoding; and defines every variable of its reproduction block.
- The LISA conventions document is cited by its public version (Baghi et
  al. 2026, arXiv:2603.22377) with its DOI.

[Unreleased]: https://github.com/PaulGoG/MilliHertzQML.jl/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/PaulGoG/MilliHertzQML.jl/releases/tag/v1.1.0
