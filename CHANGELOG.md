# Changelog

Notable changes to MilliHertzQML since its public release. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- An estimator interface between the conditioning chain and the method
  applied to each window: `AbstractWindowEstimator`, the scalar
  `AbstractWindowScorer` with `window_score`, `score_label` and
  `score_bounds`, the memory trait `estimator_memory` (`Stateless`,
  `Stateful`), the spectral `FeatureMap` of a conditioned window, and
  `condition_window`. The classifier is one scorer, `VQCScorer`.
- Stateful estimators in a replay: windows are conditioned when they
  complete, held, and released in content order (`OrderedCommit`,
  `PendingWindow`); windows that can never be scored are declared by a
  `GapEvent` (`:lost`, `:undelivered`, `:horizon`) passed to the estimator
  (`estimator_gap!`), which is reset at the start of a replay
  (`reset_estimator!`). `replay_run` and `follow_run` take `order_horizon`
  and `late_policy`; `replay_state` returns the finalised replay
  (`finalize_replay!`), `gaps_table` its gaps; a stateful replay records
  `release_at` beside `complete_at`, and its alerts are timed by it
  (`scored_at`). A stateless replay is unchanged.

### Changed
- The package is built in three layers: `MilliHertzQML.StreamingInference`
  (domain-general: configuration, provenance, signal processing, features,
  evaluation, the estimator interface, the streamed replay),
  `MilliHertzQML.MilliHertzBase` (gravitational waves: noise model,
  response, waveforms, LDC products, whitening PSDs, the generation,
  pre-processing, labelling and payload-export stages), and the classifier.
  `using MilliHertzQML` exports every public name as before; internal names
  are reached through their layer (for example
  `MilliHertzQML.MilliHertzBase.L_ARM`).
- **Breaking:** `StreamingDetector` is generic over its scorer and holds
  the conditioning only (sampling rate, window geometry, whitening PSD,
  high-pass, context); the feature set and bands moved to the scorer's
  `FeatureMap` (`detector.scorer.features`). The constructor
  `StreamingDetector(model, scaler, threshold; ...)` is unchanged.
- **Breaking:** `synthesize_noise`, `whiten_record`, `matched_filter_snr`
  and `scale_to_snr` require the `psd` keyword; they no longer default to
  the LISA noise model (`psd = lisa_noise_psd` restores the former call).

- **Breaking:** `whitening_psd_from_sidecar`, and with it
  `detector_from_run` and `scripts/infer_telemetry.jl`, refuses a feature
  sidecar that lacks the PSD kind or the parameters of an analytic kind
  instead of substituting defaults. Sidecars written by 2.x for
  `psd = "model"`, `"channel"` or `"ldc"` lack them: regenerate the
  features, or add the keys the configuration of the product used.
  Products whitened by `"welch"` or `"none"` are unaffected.
- `WindowRecord` has a thirteenth field, `release_at`; the twelve-argument
  constructor remains. For a stateful scorer `process_event!` returns the
  windows its event released, which may have completed earlier.
- The identifier of a standalone inference run (`evaluate_classifier`
  without a run identifier) is derived from the SHA-256 of the absolute
  model path, so its output directory name differs from 2.x.
- Product identity: the parameter digest of a pre-processed product is the
  SHA-256 of the key-sorted TOML rendering of its parameters
  (`parameter_digest`, formerly `Base.hash`, which differs between Julia
  versions), and its inputs enter it by the SHA-256 of their content
  (`content_digest`) instead of path, size and modification time, so a
  touched or moved but identical input keeps the product. Feature and label
  sidecars carry a `[product]` table (`product_table`: kind, channels,
  schema, parents). Products made by 2.x have other digests and are
  recomputed on their next run (the previous files are kept as `_#k`).
  The input is hashed on every call, reuse included: about 17 s for the
  3 GB Sangria training product.

### Fixed
- A feature sidecar records the parameters of an analytic whitening PSD
  (`observation_years`; `ldc_model`, `ldc_tdi2`, `ldc_observation_years`),
  which 2.x read back with defaults whatever the product had been made
  with; `whitening_psd_from_sidecar` also rebuilds the `"channel"` kind,
  which it did not know.
- `rootrelative` took a sibling directory whose name extends the root's
  (`/ws/MilliHertzQMLx` against `/ws/MilliHertzQML`) for a path inside the
  root.

## [2.0.1] — 2026-09-30

The telemetry replay refuses runs of DeepSpaceTelemetry 2.0.1 and
earlier whose external payload the producer misplaced after a scheduled
generation gap or an emitter restart, and takes the rows of a batch from
the payload row that the producer stamps on it from 2.1.0. The two
generation-gap missions of the lossy-link study, recorded with the
affected version, are regenerated, and the benchmark rows built on them are
corrected.

### Added
- `THIRD_PARTY_NOTICES.md` with the MIT notice of the LISA Data Challenge
  toolbox, whose equal-arm analytic TDI noise model and Galactic-confusion
  fit `src/ldc.jl` ports.

### Changed
- The script and test environments pin DeepSpaceTelemetry.jl `v2.1.1`.

### Fixed
- Benchmark: the two generation-gap missions of the lossy-link study were
  regenerated with DeepSpaceTelemetry 2.1.1 and replayed (see the
  correction notes under 2.0.0 and 1.2.0). Fifteen minutes without data
  on day 3: 8.0 false-alarm episodes per 30 days (published 6.9), both
  coalescences alerted. Two hours without data, now centred on merger 1:
  neither coalescence is alerted by the selected model (published one of
  two), since its conditioning stretch around the hole reaches past
  merger 2; the smoothed model alerts both at every seed.
- The telemetry replay placed the payload of a DeepSpaceTelemetry run
  wherever the producer's content epoch put it. Producers up to 2.0.1 read
  an external payload sequentially, so after a scheduled generation gap or
  an emitter restart every batch carried rows other than those of its
  content epoch (DeepSpaceTelemetry 2.1.0 fixed this). The run adapter now
  refuses such runs, takes the rows of a batch from the `payload_row` that
  producers from 2.1.0 stamp on it and checks it against the content
  epoch, and refuses a run whose payload row 1 does not lie at the mission
  epoch.
- The physics page described every LDC product as first-generation TDI;
  Sangria is TDI 1.5 with equal arm lengths, Spritz TDI 2 with Keplerian
  orbits.

## [2.0.0] — 2026-09-30

*Correction, 30 September 2026: the two generation-gap missions of the
lossy-link study were recorded with DeepSpaceTelemetry 2.0.0, which
misplaced an external payload after a scheduled gap, and the two-hour gap
was placed 4.9 h before merger 1 instead of astride it; the missions were
regenerated with 2.1.1 (see Fixed under 2.0.1). The fix of the
payload rows listed below, keyed by the content epoch, holds only for runs
without a scheduled gap or emitter restart, or from DeepSpaceTelemetry
2.1.0 on.*

Alerts of a streamed replay are credited to a coalescence only from its
signal onset, the first window in which its signal reaches the labelling
SNR, and the smoothed-whitening configuration `q8_b6_s001`, now trained at
four seeds, becomes the streaming configuration. The new default refuses
event tables written without onsets, hence the major version (see
Changed). Release 1.2.0 credited the whole four-day label span and so
counted noise alarms up to 95 hours before a merger as early detections;
the corrected numbers are under Fixed.

### Added
- `signal_onsets` and the event-table column `signal_start_index`,
  written by the labelling stage for fixed label spans and by the
  generation stage for every injection: the first window, inside the span
  and past the preceding event's span, whose matched-filter SNR of the
  signal-only truth reaches `label_snr_threshold`.
- `[telemetry] alert_crediting` (`"signal"` by default, or `"label"`) and
  the `crediting` keyword of `alert_latency_table`; the alert table records
  the criterion in `alert_crediting`, and the alert figure and the replay
  animation shade the credited span (`span_label`).
- The streaming configuration trained at the three further seeds of the
  grid, with its year replays and lossy-link replays.

### Changed
- **Breaking.** Alerts are credited from the signal onset
  (`alert_crediting = "signal"` by default); alarm runs earlier in a label
  span count as false alarms. `alert_latency_table` and the telemetry
  replay refuse an event table with label spans but without onsets:
  regenerate it with `scripts/label_ldc.jl` or the generation stage, or set
  `[telemetry] alert_crediting = "label"` (`crediting = :label`) to
  reproduce the tables of 1.2.0.
- The persistence of an alert is the smallest value that brings the
  calibration block, credited from the signal onset and pooled over the
  trained seeds, below one false alert per 30 days; the margin required on
  every calibration event is no longer applied. `q8_b6` moves from two to
  three; `q8_b6_s001` keeps three.
- `q8_b6_s001` is the streaming configuration of the manual and the
  README, and the replay animation shows it.
- CSV compatibility admits 1.1 beside 0.10.

### Fixed
- The alert times published with 1.2.0. Credited from the signal onset,
  `q8_b6_s001` alerts five of the six Sangria coalescences 11 to 24 hours
  before their merger at 0.49 false-alarm episodes per 30 days (seed 9999;
  at the three further seeds six of six, 2 to 29 hours ahead, at 1.15 to
  1.56), where 1.2.0 reported 12 to 71 hours at 0.16; `q8_b6` alerts all
  six after their merger at 0.99 per 30 days, where 1.2.0 reported two
  alerts before it. The earliest alerts of 1.2.0 fell where the window SNR
  of the source is about 2.

## [1.2.0] — 2026-09-28

*Correction, 29 September 2026: the lead times quoted in this section
credit alerts from the start of the four-day label span; credited from the
signal onset they are 11 to 24 hours, at 0.49 false-alarm episodes per 30
days (see Fixed under 2.0.0).*

*Correction, 30 September 2026: the two generation-gap missions of the
lossy-link study introduced here misplaced the payload after the gap (a
DeepSpaceTelemetry 2.0.0 defect) and were regenerated with 2.1.1 (see
Fixed under 2.0.1).*

The streamed detector becomes usable on a real link. A Welch whitening
estimate smoothed in log-frequency shortens the stretch of data each
window needs from 410 batches to 50; the selected configuration retrained
on it (`q8_b6_s001`) alerts five of the six coalescences of the Sangria
blind year at the ground station 12 to 71 hours before their merger, at
0.16 false-alarm episodes per 30 days, and keeps scoring through
scattered batch loss. The causal whitening of a replay pools every
delivered run, the lossy-link study is redone on DeepSpaceTelemetry.jl
2.0.0 over seventeen missions, the grid is trained at four seeds, and the
README and the manual are rewritten, with the figures gathered on a new
Results page.

### Added
- `smooth_psd` and `[preprocessing] psd_smoothing_dex` (0, off, by
  default): the Welch estimate that whitens a record, and the trailing
  estimate of a replay (`TrailingWelch` `smoothing_dex`), smoothed by a
  Gaussian in log-frequency. A 0.01-dex smoothing lowers the envelope of
  the whitening kernel at one window length from 21 % to 0.1 % of its
  peak.
- `configs/experiments/q8_b6_s001.toml`: the selected configuration with
  the Welch estimate smoothed by 0.01 dex, two window lengths of
  conditioning context (from a scan of streamed against batch scores) and
  a persistence of three (from the calibration rule). Its model delivers
  5/5 label spans at 2.97 false-alarm episodes per 30 days on the
  completed blind year; five of six coalescences alerted on their own,
  12 to 71 hours before the merger at the ground station, at 0.16 per 30
  days in the causal year replay; and both coalescences alerted in each of
  the seventeen lossy missions, with 81 % of the record scored at 0.43 %
  permanent loss.
- The lossy-link study of the benchmark page: seventeen 30-day missions on
  DeepSpaceTelemetry.jl 2.0.0 (scattered, bursty and retransmitted loss,
  link outages, generation gaps, recorder overflow, partial conditioning),
  replayed under causal whitening and under the full-record PSD.
- The Sangria grid trained at four seeds (18 further runs): the four- and
  six-band configurations recover all five blind events in every run, the
  two-band ones in two of eight; apart from 2000-sample windows the
  false-alarm rates do not differ beyond the seed scatter; applied seed by
  seed, the selection rule chooses three different configurations.
- Figures: `figure_gap_study` (the levels of a delivery-gap study with the
  windows scored, the events detected and the false alarms) and
  `figure_grid_seeds` (a model grid over initialisation seeds).
- `welch_psd` accepts a vector of records and pools the segments of every
  record long enough to hold one, so that an estimate never spans a hole.
- A Results page of the manual with every figure and animation and their
  captions.
- A test of the labelling stage on a synthetic truth stream; coverage
  includes the stages run by the scripts of the smoke test and the three
  package extensions.

### Changed
- The trailing whitening estimate of a replay pools the Welch segments of
  every delivered run inside its span instead of using the last contiguous
  run only, trims `[telemetry] psd_edge_periods` cutoff periods of the
  record high-pass from both ends of every run, and keeps its previous
  estimate while no run holds a segment beyond the trim. Behind a few
  permanent holes the contiguous record held one or two segments and the
  estimate was close to a raw periodogram. The causal year replay of the
  selected model changes with it: 23 false-alarm episodes (1.90 per 30
  days) instead of 17 (1.40), within the Poisson uncertainty of either
  count; the same five coalescences are detected on their own; the alerts
  of events 1 and 4 move from 1.9 and 7.5 hours after their mergers to
  25.6 and 16.1 hours before them at the ground station.
- The README is condensed to an overview (results table, the streamed
  replay of the smoothed model and the replay animation, setup, entry
  points, status, limitations); its usage details moved to the manual. The
  manual was revised as a whole: summary tables on the home and benchmark
  pages, the telemetry page divided into subsections, the log-frequency
  smoothing and the non-stationary foreground on the physics page, model
  selection and training throughput as open items on the architecture
  page.
- `configs/experiments/q8_b6.toml` carries the `[telemetry]` section of
  the replay of its model; `configs/sangria.toml` keeps one for its own
  model and for the payload export.
- Figure labels: "Labelled span", "Initialisation seed", "Selected run",
  and "Window availability" for the band of `complete_at − content_end`,
  the wait of a window for its conditioning stretch plus the downlink
  delay, formerly "Delivery".

### Fixed
- A replay of a producer that discarded production (a generation gap, a
  recorder overflow) raised a `KeyError` on the first window after the
  gap: the delivered payload was indexed by batch index, which no longer
  tracks the payload rows after such a gap. It is keyed by the first row
  a batch holds.
- `figure_telemetry_alerts` labelled each alert with the total latency,
  one hour more than its marker and the tables, and let labels of
  neighbouring alerts overlap. Labels give the alert time minus the merger
  time and are placed by their extent (`alert_label_placement`).
- The alarm-episode panel of the replay animation ticks up to the final
  episode count (`count_ticks`); its top tick collided with the panel
  above. The inner panels of the gap-study figure tick every level, and
  the rate axis of the grid figure is ticked at 1, 2 and 5 per decade.
- The manual stated the early alerts of the causal year replay in data
  time and said they reach the ground after the merger; the alert time is
  the ground arrival of the batch that completes the alert. The held-out
  block of the training year spans 109 days, not 110, and the conditioning
  lag is not irreducible.
- The physics page states the ecliptic-frame sky and polarisation
  conventions of the constellation response with the equations of the
  conventions document they follow.

## [1.1.0] — 2026-09-26

*Correction, 30 September 2026: the alert results quoted in this section
credit alerts over the four-day label span. Credited from the signal onset
(see 2.0.0), the causal replay alerts all six coalescences on their own,
each after its merger, at 1.57 false-alarm episodes per 30 days, and the
lossy-link study of this release alerts neither coalescence at 0.14 %
permanent loss and one at 0.45 %; the v1.1.0 manual has been rebuilt with
these numbers.*

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

[Unreleased]: https://github.com/PaulGoG/MilliHertzQML.jl/compare/v2.0.1...HEAD
[2.0.1]: https://github.com/PaulGoG/MilliHertzQML.jl/releases/tag/v2.0.1
[2.0.0]: https://github.com/PaulGoG/MilliHertzQML.jl/releases/tag/v2.0.0
[1.2.0]: https://github.com/PaulGoG/MilliHertzQML.jl/releases/tag/v1.2.0
[1.1.0]: https://github.com/PaulGoG/MilliHertzQML.jl/releases/tag/v1.1.0
