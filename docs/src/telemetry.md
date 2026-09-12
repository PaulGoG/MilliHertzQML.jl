# Telemetry coupling

## Overview

The classifier is coupled to the telemetry producer through files in both directions; neither package loads the other.

- **Upstream (this package to the producer).** The payload-export stage writes the A channel of an HDF5 TDI product as a single-column CSV and a scenario fragment beside it. The producer's external-data mode reads the CSV gaplessly into segments and batches according to the fragment (`[physics] data_source = "external"`, `external_data_path`, `sample_rate`, `segment_duration_sec`, `batch_size`), stamps the mission clock from `[simulation] start_sim_time`, and carries the coalescence markers (`[[events.markers]]`, each a `time` and a `label`) into its own event stream.
- **Downstream (the producer to this package).** The producer emits a run directory of batches; the replay stage of this package follows that directory, reassembles the sliding windows as batches arrive, scores them with a trained model, and raises alerts.

The two halves share one time base, fixed by the fragment: payload row ``r`` is the sample at `start_sim_time` ``+ (r - 1)/f_s``. Every quantity the producer or the replay reports in mission time maps back to a payload row, and through the row to the sample of the original product and to the point-wise labels.

## Payload export

`export_telemetry_payload(config; h5_file, tdi_group, catalog, output_prefix)` (script `scripts/export_telemetry_payload.jl`) reads the Michelson variables of the product (`read_tdi`), forms ``A = (Z - X)/\sqrt{2}`` (`tdi_to_aet`), and writes under the `[paths] inputs` root:

- `<output_prefix>_payload.csv` — one column `Amplitude`, single precision, one row per sample; the file the producer ingests.
- `<output_prefix>_scenario.toml` — the fragment: `[physics]` (`data_source = "external"`, `external_data_path` relative to the package root, `sample_rate` ``= 1/\Delta t`` of the file, `segment_duration_sec`, `batch_size`); `[simulation]` (`start_sim_time`); `[[events.markers]]` from the event catalog when one is given, one entry `{time, label = "mbhb_<id>"}` per event, the time being `start_sim_time` plus the coalescence time measured from the first record sample, rounded to the millisecond; `[[labels]]` with the `label_start_index`/`label_end_index` span of every event when the catalog carries these columns (the simulator catalog always does; the LDC event table does for `label_span = "fixed"`), so that the point-wise labels can be rebuilt on the consumer side; `[payload]` (source, group, row count, catalog, rows per batch, number of complete batches); and the git and hardware provenance every snapshot carries.

The catalog is either the simulator's `<stem>_events.csv` (columns `event_id`, `t_c_sec`) or the LDC event table of the labeling stage (`event`, `merger_time_s`). A catalog whose coalescences fall outside the record, or whose label spans exceed it, is rejected as not belonging to the product.

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
julia scripts/export_telemetry_payload.jl config.toml \
    --h5-file data/inputs/simulated_telemetry_complex.h5 \
    --catalog data/inputs/simulated_telemetry_complex_events.csv \
    --output-prefix telemetry_sim
```

## Replay and alerts

The consumer never writes into a producer run directory; it reads it through the producer's own API in the package extension that loads with DeepSpaceTelemetry.jl (`open_telemetry_run`), so that a change of the run-directory contract surfaces as a version bump — checked against `[telemetry] producer_compat` — or an API error rather than as silent format drift. The run exposes its geometry (`run_geometry`: sampling rate, segment length, batch size, samples per batch, mission epoch, producer version), its batches (`list_batches`: delivered, lost, or pruned, each with the payload rows it covers), the payload of a delivered batch (`read_batch`), the arrival feed in mission-time order (`arrival_events`), and the lifecycle state (`run_state`).

Replaying the feed (`replay_run`) maintains the coverage of delivered rows as a set of disjoint intervals (`Coverage`), so that the order of arrivals — live batches first, then the archive backfilled newest-first — does not matter; a lost or pruned batch is a permanent hole, widened by `tdi_gap_dilation_sec` on each side, that later arrivals never fill. Whenever a batch arrives, the windows that intersect it and are now delivered to at least `min_coverage` of their samples are scored once (`WindowScheduler`), at the mission time of the arrival that completed them. The score comes from the same conditioning as the batch pipeline: the delivered stretch around the window — up to `context_windows` window lengths on each side — is high-passed and whitened by the PSD of `psd_sidecar` when one is configured and by the PSD persisted with the training run otherwise, and the window is cut from it (`StreamingDetector`, built by `detector_from_run` from the model artifact, its threshold, and the sidecar of the feature table it was trained on). The first windows of a record, and any isolated window, are scored on shorter context and are edge-affected, as in the batch pipeline.

Both settings have to be chosen against the record, not left at their defaults. A whitening PSD calibrates the record being scored: whitening one observation record with the PSD estimated from another mis-scales every band power wherever the noise is not stationary between them, and the annually modulated Galactic foreground makes that the normal case rather than the exceptional one — on the Sangria blind year the training run's own PSD costs every detection. `context_windows` has to reach past the conditioning kernel, which decays as a power law rather than exponentially whenever the whitening PSD resolves sharp spectral features such as the TDI transfer nulls; with the Sangria Welch estimate the streamed scores match the batch pipeline only from about twenty window lengths of context upward, and not monotonically below that, so the agreement has to be measured rather than assumed.

The table of scored windows records, per window, its rows and their mission times, the completing arrival and its time, the coverage, the score, the decision, and the inference wall time; windows beyond the rows of the ingested payload (which the producer records in its snapshot and pads with zeros afterwards) are not scored. `alert_latency_table` reduces it per event of an event table (`merger_time_s` or the simulator's `t_c_sec`, optional label spans): the first alarmed window overlapping the event's label span, its completion time as the alert time, the data latency (alert time minus merger time — negative when the inspiral is alarmed inside the label span before the coalescence), the total latency with the `processing_latency_hours` budget, whether the event was detected, and the number of alarmed windows outside every span as false-alarm episodes per 30 days of scored data. In live mode (`follow_run`, `[telemetry] mode = "live"` or `--live`), the feed is polled every `poll_interval_sec` and consumed beyond the events already processed until the run reaches a terminal sentinel and the feed is drained; the two modes share the same state machine (`process_event!`) and give identical tables.

The figure `figure_telemetry_alerts` stacks the score trace with the threshold, the alarmed windows, and the labeled spans over the ground-availability latency of every window, with the alert latency of each detected event annotated. Scope of the first coupling: lossless delivery without generation gaps; holes are excluded from scoring rather than handled (masked or Lomb–Scargle features are deferred).

## Testing

The unit tests exercise the consumer on an in-memory run (`MemoryTelemetryRun`): geometry, coverage algebra under out-of-order arrival, window scheduling, erosion of a lost batch, replay against a direct evaluation on the delivered stretch, live-mode drainage, and the alert table. The integration test runs a one-day DeepSpaceTelemetry mission at 0.2 Hz on an exported payload in a temporary run root (the producer's `DATA_ROOT` redirected) and replays it through the extension, so a producer upgrade is checked by the suite itself. DeepSpaceTelemetry.jl is a pinned git source of the script and test environments and a weak dependency of the package.
