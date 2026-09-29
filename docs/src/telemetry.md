# Telemetry coupling

## Overview

The classifier is coupled to the telemetry producer through files in both directions; neither package loads the other.

- **Upstream (this package to the producer).** The payload-export stage writes the A channel of an HDF5 TDI product as a single-column CSV and a scenario fragment beside it. The producer's external-data mode reads the CSV gaplessly into segments and batches according to the fragment (`[physics] data_source = "external"`, `external_data_path`, `sample_rate`, `segment_duration_sec`, `batch_size`), stamps the mission clock from `[simulation] start_sim_time`, and carries the coalescence markers (`[[events.markers]]`, each a `time` and a `label`) into its own event stream.
- **Downstream (the producer to this package).** The producer emits a run directory of batches; the replay stage of this package follows that directory, reassembles the sliding windows as batches arrive, scores them with a trained model, and raises alerts.

The two halves share one time base, fixed by the fragment: payload row ``r`` is the sample at `start_sim_time` ``+ (r - 1)/f_s``. Every quantity the producer or the replay reports in mission time maps back to a payload row, and through the row to the sample of the original product and to the point-wise labels.

## Payload export

`export_telemetry_payload(config; h5_file, tdi_group, catalog, output_prefix)` (script `scripts/export_telemetry_payload.jl`) reads the Michelson variables of the product (`read_tdi`), forms ``A = (Z - X)/\sqrt{2}`` (`tdi_to_aet`), and writes under the `[paths] inputs` root:

- `<output_prefix>_payload.csv` — one column `Amplitude`, single precision, one row per sample; the file the producer ingests.
- `<output_prefix>_scenario.toml` — the fragment: `[physics]` (`data_source = "external"`, `external_data_path` relative to the package root, `sample_rate` ``= 1/\Delta t`` of the file, `segment_duration_sec`, `batch_size`); `[simulation]` (`start_sim_time`); `[[events.markers]]` from the event catalogue when one is given, one entry `{time, label = "mbhb_<id>"}` per event, the time being `start_sim_time` plus the coalescence time measured from the first record sample, rounded to the millisecond; `[[labels]]` with the `label_start_index`/`label_end_index` span of every event when the catalogue carries these columns (the simulator catalogue always does; the LDC event table does for `label_span = "fixed"`), so that the point-wise labels can be rebuilt on the consumer side; `[payload]` (source, group, row count, catalog, rows per batch, number of complete batches); and the git and hardware provenance every snapshot carries.

The catalogue is either the simulator's `<stem>_events.csv` (columns `event_id`, `t_c_sec`) or the LDC event table of the labelling stage (`event`, `merger_time_s`). A catalogue whose coalescences fall outside the record, or whose label spans exceed it, is rejected as not belonging to the product.

**Time base.** With sampling frequency ``f_s`` and mission start ``t_0`` = `start_sim_time`,

```math
\text{row } r \;\leftrightarrow\; t_0 + \frac{r - 1}{f_s},
\qquad
\text{batch } k \;\leftrightarrow\; \text{rows } [(k - 1)P + 1,\; kP],
\qquad
P = f_s \, T_\mathrm{seg} \, B,
```

where ``T_\mathrm{seg}`` is `segment_duration_sec` and ``B`` is `batch_size`. A segment must hold a whole number of samples (``f_s T_\mathrm{seg} \in \mathbb{N}``), the record must hold at least one full batch (``n_\mathrm{rows} \ge P``), and the stage refuses to start otherwise; `samples_per_batch(sample_rate, segment_duration_sec, batch_size)` evaluates ``P``.

**Default geometry.** At ``f_s = 0.2`` Hz the defaults ``T_\mathrm{seg} = 50`` s and ``B = 10`` give ``P = 100`` rows per batch, which is exactly the window step of the pre-processor (`[preprocessing] step_size = 100`): every batch advances the sliding window by one step, a window of 1000 samples spans ten consecutive batches, and the ``k``-th window of the feature table becomes evaluable when batch ``k + 9`` has arrived. The marker of an event coalescing at row ``r_c`` falls in batch ``\lceil r_c / P \rceil``.

**Configuration.** The `[telemetry]` section holds `segment_duration_sec` (> 0, default 50), `batch_size` (>= 1, default 10), `start_sim_time` (ISO-8601 date-time string, default `"2035-01-01T00:00:00"`), and `output_prefix` (default `"telemetry"`); the product and group default to `[preprocessing] h5_file` and `tdi_group`.

```bash
julia scripts/export_telemetry_payload.jl configs/default.toml \
    --h5-file data/inputs/simulated_telemetry_complex.h5 \
    --catalog data/inputs/simulated_telemetry_complex_events.csv \
    --output-prefix telemetry_sim
```

## Replay and alerts

`scripts/infer_telemetry.jl` replays a producer run directory, or follows
it live with `--live`, and writes `telemetry_windows.csv` (one row per
scored window), `alert_latency.csv` (one row per event of `--events`), a
snapshot of the `[telemetry]` settings, and the alert figure:

```bash
julia scripts/infer_telemetry.jl configs/default.toml --run-dir <DeepSpaceTelemetry run directory> \
    --model models/run_<RUN_ID>/gw_model.jld2 \
    --events data/inputs/simulated_telemetry_complex_events.csv --run-id coupling01
```

### Run interface

The consumer never writes into a producer run directory. It reads the
directory through the producer's own API, in the package extension that
loads with DeepSpaceTelemetry.jl (`open_telemetry_run`), so that a change
of the run-directory contract surfaces as a version bump, checked against
`[telemetry] producer_compat`, or as an API error rather than as silent
format drift. The run exposes its geometry (`run_geometry`: sampling
rate, segment length, batch size, samples per batch, mission epoch,
producer version), its batches (`list_batches`: delivered, lost or pruned,
each with the payload rows it covers), the payload of a delivered batch
(`read_batch`), the arrival feed in mission-time order (`arrival_events`),
and the lifecycle state (`run_state`). The rows of a batch are taken from
the content epoch the producer stamps on it rather than from its stored
index, since the two differ once the producer discards production;
`list_batches` warns when they do.

### Scheduling and conditioning

Replaying the feed (`replay_run`) maintains the coverage of delivered rows
as a set of disjoint intervals (`Coverage`), so the order of arrivals,
live batches first and the archive backfilled newest-first, does not
matter. A lost or pruned batch is a permanent hole, widened by
`tdi_gap_dilation_sec` on each side, that later arrivals never fill.

A window is scored once, at the mission time of the arrival that completes
its **conditioning stretch**: its own rows widened by `context_windows`
window lengths on each side and cut to the record. The stretch must be
delivered to at least `min_coverage`, and the window's own rows entirely,
so a hole inside a window is never scored across (`WindowScheduler`,
`conditioning_rows`). The score uses the conditioning of the batch
pipeline: the delivered stretch is high-passed and whitened as a whole,
and the window is cut from it (`StreamingDetector`, built by
`detector_from_run` from the model artifact, its threshold, and the
sidecar of the feature table it was trained on). The first windows of a
record, and any isolated window, are scored on shorter context and are
edge-affected, as in the batch pipeline.

### Whitening PSD

`[telemetry] psd_mode` selects the PSD that whitens each stretch.

- **`"sidecar"`: a fixed PSD**, that of the feature sidecar `psd_sidecar`
  when one is configured and the PSD persisted with the training run
  otherwise. In the benchmark the sidecar supplies the full-record PSD of
  the blind year, the median Welch estimate of the entire blind year,
  which is available only after the whole record has been received.
- **`"trailing"`: the causal estimate of `TrailingWelch`**, the median
  Welch spectrum of every delivered run inside the last
  `psd_trailing_days` behind the window's conditioning stretch. Each run
  is high-passed as every stretch is and trimmed by `psd_edge_periods`
  cutoff periods at both ends, where the high-pass rings, and the segments
  (`psd_segment_length`) of all runs are pooled, so that none spans a
  delivery hole. The estimate is redone whenever that record has advanced
  by `psd_refresh_days`, kept from the previous estimate while no run
  holds a segment, and replaced by the training run's own PSD only before
  any estimate exists. When `[preprocessing] psd_smoothing_dex` is
  positive, it is smoothed in log-frequency by the width that smoothed the
  batch estimate behind the model's training features.

The trailing estimate is causal in that only data already delivered to
the ground station enter it, unlike a sidecar PSD of a whole record, which
for a streamed mission includes data not yet delivered. The scored-window
table records the last row behind each estimate (`psd_row`, 0 for the
static PSD).

### Choosing the PSD and the context

Both settings must be chosen for the record being scored rather than left
at their defaults. A whitening PSD calibrates the record it whitens:
whitening one observation record with the PSD estimated from another
mis-scales every band power wherever the noise is not stationary between
them, and the annually modulated Galactic foreground makes that the
normal case. On the Sangria blind year, whitening with the training run's
own PSD removes every detection.

`context_windows` has to reach past the conditioning kernel, the impulse
response of the high-pass followed by whitening. That kernel decays as a
power law rather than exponentially whenever the whitening PSD resolves
sharp spectral features such as the TDI transfer nulls. With the Sangria
Welch estimate the streamed scores match the batch pipeline only from
about twenty window lengths of context upward, and not monotonically
below that; with the same estimate smoothed by 0.01 dex in log-frequency
they match from one window length on (see the benchmark page). The
agreement has to be measured for the whitening in use rather than
assumed. The look-ahead is a property of the filter, not a tuning choice:
the whitening is zero-phase, so a window's conditioned features depend on
the data after it as much as on the data before it, and a detector that
scores each window as soon as its own samples are delivered detects two
of the five Sangria blind events where a centred stretch detects all
five; sixty-four window lengths of past context do not remove this
deficit. An alert therefore carries a conditioning lag of
`context_windows` window lengths, 1.16 days at twenty and 2.8 hours with
the smoothed whitening of `configs/experiments/q8_b6_s001.toml`, on top of
the delivery latency of the ground segment.

### Alerts

The table of scored windows records, per window, its rows and their
mission times, the completing arrival and its time, the coverage, the
score, the decision, and the inference wall time; windows beyond the rows
of the ingested payload, which the producer records in its snapshot and
pads with zeros afterwards, are not scored.

`alert_latency_table` reduces it per event of an event table
(`merger_time_s` or the simulator's `t_c_sec`, optional label spans). An
alert is a run of `alert_persistence` consecutive alarmed windows,
consecutive in window index whatever the order of arrival, raised by the
arrival that completes the run; with a persistence of one, every alarmed
window is an alert.

An alert is credited to an event only from the event's **signal onset**
(`[telemetry] alert_crediting = "signal"`, the default): the first window
of the signal-only truth stream, inside the label span and past the
preceding event's span, whose matched-filter SNR reaches the labelling
threshold (`label_snr_threshold`, 5). The labelling and generation stages
write it as `signal_start_index` beside the label span. The credited span
runs from the onset to the end of the label span; an alarm earlier in the
label span falls where no single window of the source reaches the
threshold, is not attributable to it, and counts as a false alarm. The label span itself, four days before to 27 minutes after the
merger for the Sangria labels, is the training convention of Isfan et al.
[IsfanEtAl2025](@cite) and is much wider than the stretch in which a coalescence is visible: on
the Sangria blind year the onsets lie 7 to 35 hours before the merger.
Crediting the whole label span (`alert_crediting = "label"`) counts noise
alarms of its first days as early detections; it reproduces the alert
tables of releases up to 1.2.0. An event table with label spans
but without onsets is refused under signal crediting.

For every event the table gives the earliest alert touching its credited
span and:

- `t_alarm`, the **alert time**: the arrival at the ground station of the
  batch that completes the alert, which includes the conditioning lag and
  the downlink delay;
- `latency_data_h`, the alert time minus the merger time, negative when
  the alert reaches the ground before the coalescence;
- `latency_total_h`, the same with the ground processing budget
  `processing_latency_hours` added;
- whether the event was detected, and whether its alert is shared with
  another event whose credited span overlaps;
- the false-alarm episodes per 30 days of scored data: runs of alarmed
  windows outside every credited span that reach the persistence. Shorter
  runs raise no alert and are not counted;
- `alert_persistence` and `alert_crediting`, the criteria applied.

The benchmark page counts an event whose only alert is shared as not
detected on its own. In live mode (`follow_run`, `[telemetry] mode =
"live"` or `--live`) the feed is polled every `poll_interval_sec` and
consumed beyond the events already processed until the run reaches a
terminal sentinel and the feed is drained; the two modes share the same
state machine (`process_event!`) and give identical tables.

### Gaps and figures

The coupling excludes delivery holes from scoring rather than scoring
across them, follows a producer that discarded production, and pools the
causal PSD estimate over the delivered runs around the holes; no masked
or Lomb–Scargle features are implemented. The effect of the exclusion is
measured on the [benchmark page](benchmark.md) over seventeen missions:
scattered permanent loss removes the windows whose conditioning stretch
contains a hole, so the tolerable loss rate is set by the length of the
stretch; bursts, retransmission, outages and gaps in the data itself
remove far less.

`figure_telemetry_alerts` stacks the score trace, with the threshold, the
alarmed windows and the credited spans, over the availability latency of
every window (the wait for its conditioning stretch plus the downlink
delay) and the alert time of every detected event. `animate_mission_replay`
sweeps the same replay across four panels in the order in which the ground
received the windows; the dotted rule is the ground clock, and its
distance from the edge of the data is the availability latency.

![A year of telemetry replay: coverage, classifier score against the threshold with the labelled spans, cumulative alarm episodes, and window availability](assets/mission_replay.gif)

## Testing

The unit tests exercise the consumer on an in-memory run (`MemoryTelemetryRun`): geometry, coverage algebra under out-of-order arrival, window scheduling, erosion of a lost batch, replay against a direct evaluation on the delivered stretch, live-mode drainage, and the alert table. The integration test runs a one-day DeepSpaceTelemetry mission at 0.2 Hz on an exported payload in a temporary run root (the producer's `DATA_ROOT` redirected) and replays it through the extension, so a producer upgrade is checked by the suite itself. DeepSpaceTelemetry.jl is a pinned git source of the script and test environments and a weak dependency of the package.
