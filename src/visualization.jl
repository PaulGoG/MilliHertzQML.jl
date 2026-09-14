# src/visualization.jl — publication figures of the pipeline. The functions
# are implemented by the package extension MilliHertzQMLCairoMakieExt,
# which loads together with CairoMakie; the core package stays free of a
# plotting dependency.

"""
    FIGURE_WIDTH_MM

Printed width of a single-column figure [mm]; figures are designed at this
size and enter a manuscript without rescaling.
"""
const FIGURE_WIDTH_MM = 86.0

"""
    FIGURE_COLORS

Semantic colors shared by every figure of the project (Okabe–Ito palette):
`data` (classifier output, time series), `label` (labeled spans), `threshold`,
`training`, `validation`, `noise` and `signal` (score distributions), `fit`.
"""
const FIGURE_COLORS = (
    data = "#0072B2",
    label = "#E69F00",
    threshold = "#000000",
    training = "#0072B2",
    validation = "#D55E00",
    noise = "#999999",
    signal = "#D55E00",
    fit = "#009E73",
)

"""
    figure_theme(; width_mm = FIGURE_WIDTH_MM, height_mm = 0.68 * width_mm, fontsize = 8)

Makie theme of the project's publication figures: Computer Modern fonts,
boxed axes with inward ticks, dashed low-opacity grid, no minor ticks, and
a figure size of `width_mm` × `height_mm` in typographic points (1 mm =
72/25.4 pt), so that a PDF export enters a manuscript at its native
width. Requires CairoMakie to be loaded.
"""
function figure_theme end

"""
    save_figure(figure, stem; run_id = "", formats = ("pdf", "png"), px_per_unit = 4)

Export `figure` as `<stem>.pdf` (vector) and `<stem>.png` (raster at
`px_per_unit` times the design size) and write the provenance sidecar
`<stem>.toml` (run identifier, git description, hardware fingerprint,
time). Existing files are backed up first. Returns the vector of written
paths. Requires CairoMakie to be loaded.
"""
function save_figure end

"""
    animation_theme(; width_mm = 180.0, height_mm = 0.72 * width_mm, fontsize = 9)

Screen counterpart of [`figure_theme`](@ref) for the animations: the same
fonts, colors, and axis discipline on the wider canvas, with the larger
type, margins, and strokes a GIF is read at. Requires CairoMakie to be
loaded.
"""
function animation_theme end

"""
    save_animation(render, stem; run_id = "") -> String

Write the animation `<stem>.gif` by calling `render(path)` with that path
and write the provenance sidecar `<stem>.toml` (run identifier, git
description, hardware fingerprint, time), as [`save_figure`](@ref) does for
a static figure. An existing GIF is backed up first. Returns the written
path. Requires CairoMakie to be loaded.

# Example

```julia
save_animation(joinpath(plot_dir, "training_history"); run_id = run_id) do path
    animate_training_history(history, path)
end
```
"""
function save_animation end

"""
    animate_training_history(history, path; framerate = 5, hold_frames = 10,
                             width_mm = 180.0, px_per_unit = 2) -> String

Animated counterpart of [`figure_training_history`](@ref): the two stacked
panels sharing the epoch axis, revealed one epoch per frame at fixed axis
limits, held for `hold_frames` frames at the end. The epoch of least
validation loss — the checkpoint whose weights the run ships — is marked in
both panels once the sweep reaches it. `history` holds the vectors
`epochs`, `train_loss`, `val_loss`, and `val_acc`; `path` must name a GIF,
the only file written. Returns `path`. Requires CairoMakie to be loaded.
"""
function animate_training_history end

"""
    animate_mission_replay(windows, threshold, path; epoch, label_spans = nothing,
                           n_frames = 200, framerate = 20, hold_frames = 20,
                           max_points = 6000, width_mm = 180.0,
                           px_per_unit = 2) -> String

Four stacked panels of a telemetry replay on a shared mission-time axis
[days since `epoch`, by default the content end of the first window]:
window coverage, classifier score with the decision `threshold` as a dashed
rule, the alarmed windows marked, and the labeled spans (`label_spans`,
pairs of `DateTime`) shaded, the count of alarm episodes accumulated along
mission time, and the ground latency of every window (`complete_at −
content_end` [h]).

The windows are revealed in arrival order, sorted by `complete_at`, not in
mission-time order: a pass delivers its backlog newest first and a window
becomes evaluable only once the conditioning stretch around it has landed,
so the trace fills in wherever the ground has learned something rather than
from left to right. A dotted rule marks the ground clock, whose distance to
the data edge is the delivery latency. The sweep takes about `n_frames`
frames and the traces are decimated to about `max_points` windows, the
alarmed ones always kept. `windows` is the table of `replay_run`, `path`
must name a GIF, the only file written. Returns `path`. Requires CairoMakie
to be loaded.
"""
function animate_mission_replay end

"""
    figure_training_history(history) -> Figure

Two stacked panels sharing the epoch axis: training and validation loss
(solid and dashed), and validation accuracy. `history` holds the vectors
`epochs`, `train_loss`, `val_loss`, `val_acc`. Requires CairoMakie.
"""
function figure_training_history end

"""
    figure_mission_trace(days, probabilities, threshold; labels = nothing,
                         max_points = 5000) -> Figure

Classifier output against mission time [days] with the decision threshold
and, when `labels` is given, the labeled spans as shaded bands. Long
traces are decimated to about `max_points` samples. Requires CairoMakie.
"""
function figure_mission_trace end

"""
    figure_roc(fpr, tpr, auc) -> Figure

Receiver operating characteristic with the chance diagonal and the area
under the curve stated in the legend. Requires CairoMakie.
"""
function figure_roc end

"""
    figure_threshold_sweep(sweep, threshold; target_far_per_30d = nothing,
                           operating_point = nothing) -> Figure

Event-level operating characteristic of a scored block: two stacked
panels over the decision threshold, the event recall (solid) with the
window recall (dashed), and the false-alarm episodes per 30 days on a
logarithmic axis (thresholds without a false alarm are left blank). The
operating point `threshold` is marked and the `target_far_per_30d` of the
`far` criterion drawn when given. `operating_point`, anything carrying
`n_detected`, `n_events`, and `false_alarms_per_30d`, puts the metrics of
the applied threshold in the legend; they cannot be read off `sweep`,
whose candidates are score quantiles and are sparse in the far tail.
Without it the legend states the threshold alone. `sweep` is the table of
[`threshold_sweep`](@ref). Requires CairoMakie.
"""
function figure_threshold_sweep end

"""
    figure_sensitivity(snrs, labels, decisions; n_bins = 8) -> Figure

Fraction of positive windows detected per matched-filter SNR bin, with
the number of windows per bin annotated. Returns `nothing` when no
positive window exists. Requires CairoMakie.
"""
function figure_sensitivity end

"""
    figure_score_distribution(probabilities, threshold; labels = nothing, n_bins = 50) -> Figure

Histogram of the classifier scores, split into noise and labeled windows
when `labels` is given, with the decision threshold. Requires CairoMakie.
"""
function figure_score_distribution end

"""
    figure_telemetry_alerts(windows, threshold; epoch, label_spans = nothing,
                            latencies = nothing) -> Figure

Two stacked panels on a shared mission-time axis [days since `epoch`] for
the windows table of a replay: the classifier score of every window at its
content end with the decision threshold, alarmed windows marked, and the
labeled spans (`label_spans`, pairs of `DateTime`) shaded; below, the
ground-availability latency of every window (`complete_at − content_end`
[h]) with the alert latencies of the detected events (`latencies`, the
table of `alert_latency_table`) annotated. Requires CairoMakie.
"""
function figure_telemetry_alerts end

"""
    figure_loss_survival(p_loss, scored_fraction, events_detected, n_events;
                         stretch_batches, model = true) -> Figure

Two stacked panels against the permanent per-batch loss probability of a
delivery channel [%], on a logarithmic axis: the windows a replay could
score as a fraction of the lossless mission, and the coalescences it still
detected of `n_events`.

A window is scored only when its whole conditioning stretch has been
delivered, so survival requires `stretch_batches` consecutive batches to
arrive. `model` adds the independent-batch estimate
`(1 - p)^stretch_batches` and marks `1 / stretch_batches`, the loss rate
at which the mean spacing of losses equals the stretch. Requires
CairoMakie.
"""
function figure_loss_survival end

"""
    figure_seed_spread(seeds, thresholds, far_per_30d, events_detected, n_events;
                       baseline_seed = nothing, target_far = nothing) -> Figure

Two stacked panels over the initialization seeds of repeated training
runs: the decision threshold each run fitted, and the false-alarm rate it
then delivered on the blind record, annotated with the events each run
recovered of `n_events`. `baseline_seed` marks the run the package ships
and `target_far` the rate the threshold criterion asked for. Requires
CairoMakie.
"""
function figure_seed_spread end

"""
    figure_telemetry_trace(t_days, strain, labels; max_points = 5000, whitened = nothing) -> Figure

Simulated strain record against mission time with the labeled spans as
shaded bands, an axis offset multiplier for the small strain amplitudes,
and, when `whitened` is given, a second panel with the whitened record.
Requires CairoMakie.
"""
function figure_telemetry_trace end
