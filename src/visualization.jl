# src/visualization.jl — publication figures of the pipeline. The functions
# are implemented by the package extension MilliHertzQMLCairoMakieExt,
# which loads together with CairoMakie; the core package stays free of a
# plotting dependency.

"""
    FIGURE_SIZE

Canvas of a single-panel figure in typographic points, `(width, height)`:
the base layout every figure scales from. Each further stacked main panel
adds [`PANEL_HEIGHT`](@ref), each auxiliary strip (a counter or residual
panel) [`STRIP_HEIGHT`](@ref); see [`figure_size`](@ref).
"""
const FIGURE_SIZE = (900, 600)

"""
    PANEL_HEIGHT

Height [pt] added to [`FIGURE_SIZE`](@ref) by each stacked main panel beyond
the first.
"""
const PANEL_HEIGHT = 350

"""
    STRIP_HEIGHT

Height [pt] added to [`FIGURE_SIZE`](@ref) by each auxiliary strip, a
counter or residual panel beneath the main panels.
"""
const STRIP_HEIGHT = 180

"""
    FIGURE_FONTSIZE

Type size [pt] of axis labels and legends.
"""
const FIGURE_FONTSIZE = 26

"""
    TICK_FONTSIZE

Type size [pt] of tick labels.
"""
const TICK_FONTSIZE = 22

"""
    ANNOTATION_FONTSIZE

Type size [pt] of in-axis annotations.
"""
const ANNOTATION_FONTSIZE = 21

"""
    figure_size(main_panels = 1, strips = 0) -> Tuple{Int,Int}

Canvas [pt] of a figure with `main_panels` stacked main panels and `strips` auxiliary strips: `FIGURE_SIZE` heightened by `PANEL_HEIGHT` per panel beyond the first and `STRIP_HEIGHT` per strip.
"""
function figure_size(main_panels::Integer = 1, strips::Integer = 0)
    main_panels >= 1 || throw(ArgumentError("a figure has at least one main panel."))
    strips >= 0 || throw(ArgumentError("strips must be non-negative."))
    return (
        FIGURE_SIZE[1],
        FIGURE_SIZE[2] + (main_panels - 1) * PANEL_HEIGHT + strips * STRIP_HEIGHT,
    )
end

"""
    FIGURE_COLORS

Semantic colours shared by every figure of the project (Okabe–Ito palette),
each with a single use: `data` (classifier output, time series, measured
points), `label` (labelled spans), `threshold` (the decision threshold,
black, dashed in every figure), `training` and `validation` (the two
blocks), `noise` and `signal` (score distributions, and the detected-event
markers), `fit` (fitted or model curves), `false_alarm` (the false-alarm
rate wherever it is drawn), `target` (a requested or reference rate: the
`far` target, the `1 / stretch` rule).
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
    false_alarm = "#CC79A7",
    target = "#56B4E9",
)

"""
    FIGURE_STROKES

Darker same-hue counterparts of [`FIGURE_COLORS`](@ref), under the same
keys, for marker outlines and histogram bar edges.
"""
const FIGURE_STROKES = (
    data = "#004B78",
    label = "#9A6A00",
    threshold = "#000000",
    training = "#004B78",
    validation = "#8E3F00",
    noise = "#666666",
    signal = "#8E3F00",
    fit = "#006A4E",
    false_alarm = "#8A5070",
    target = "#2F7DA0",
)

"""
    figure_theme(; size = FIGURE_SIZE, fontsize = FIGURE_FONTSIZE)

Makie theme of the base layout: Computer Modern fonts, boxed axes with
inward ticks, dashed low-opacity grid, no minor ticks, 3 pt data lines,
14 pt markers with a 1.5 pt stroke, and the canvas `size` in typographic
points, so that a PDF export enters a document at native size. Taller
canvases come from [`figure_size`](@ref). Requires CairoMakie to be loaded.
"""
function figure_theme end

"""
    save_figure(figure, stem; run_id = "", formats = ("pdf", "png"), px_per_unit = 4)

Export `figure` as `<stem>.pdf` (vector) and `<stem>.png` (raster at
`px_per_unit` times the design size) and write the provenance sidecar
`<stem>.toml` (run identifier, the canvas actually exported `size_pt`
[pt], `px_per_unit`, git description, hardware fingerprint, time).
Existing files are backed up first. Returns the vector of written paths.
Requires CairoMakie to be loaded.
"""
function save_figure end

"""
    animation_theme(; size = FIGURE_SIZE, fontsize = FIGURE_FONTSIZE)

Theme of the animations: [`figure_theme`](@ref) unchanged, on a canvas
snapped to an even whole number of typographic points so that every frame
renders at exactly the size declared to the encoder. Requires CairoMakie
to be loaded.
"""
function animation_theme end

"""
    save_animation(render, stem; run_id = "") -> String

Write the animation `<stem>.gif` by calling `render(path)` with that path
and write the provenance sidecar `<stem>.toml` (run identifier, `frame_px`,
the pixel size of the written GIF, `px_per_unit`, file size, git
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
                             size = figure_size(2), px_per_unit = 2) -> String

Animated counterpart of [`figure_training_history`](@ref): the two stacked
panels sharing the epoch axis on a canvas of `size` [pt], revealed one
epoch per frame at fixed axis limits, held for `hold_frames` frames at the
end. The epoch of least
validation loss — the checkpoint whose weights the run saves — is marked in
both panels once the sweep reaches it. `history` holds the vectors
`epochs`, `train_loss`, `val_loss`, and `val_acc`; `path` must name a GIF,
the only file written. Returns `path`. Requires CairoMakie to be loaded.
"""
function animate_training_history end

"""
    animate_mission_replay(windows, threshold, path; epoch, label_spans = nothing,
                           span_label = "Labelled span", n_frames = 200, framerate = 20, hold_frames = 20,
                           max_points = 6000, size = figure_size(2, 2),
                           px_per_unit = 2) -> String

Four stacked panels of a telemetry replay, two main panels and two strips
on a canvas of `size` [pt], on a shared mission-time axis
[days since `epoch`, by default the content end of the first window]:
window coverage, classifier score with the decision `threshold` as a dashed
rule, the alarmed windows marked, and the spans alerts are credited to
(`label_spans`, pairs of `DateTime`, named `span_label` in the legend)
shaded, the count of alarm episodes accumulated along
mission time, and the availability latency of every window (`complete_at −
content_end` [h]: the wait for its conditioning stretch and the downlink
delay of the batch that completes it).

The windows are revealed in arrival order, sorted by `complete_at`, not in
mission-time order: a pass delivers its backlog newest first and a window
becomes evaluable only once the conditioning stretch around it has landed,
so the trace fills in wherever windows have become evaluable rather than
from left to right. A dotted rule marks the ground clock, whose distance to
the data edge is the availability latency. The sweep takes about `n_frames`
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
and, when `labels` is given, the labelled spans as shaded bands. Long
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

Histogram of the classifier scores, split into noise and labelled windows
when `labels` is given, with the decision threshold. Requires CairoMakie.
"""
function figure_score_distribution end

"""
    figure_telemetry_alerts(windows, threshold; epoch, label_spans = nothing,
                            latencies = nothing, span_label = "Labelled span") -> Figure

Two stacked panels on a shared mission-time axis [days since `epoch`] for
the windows table of a replay: the classifier score of every window at its
content end with the decision threshold, alarmed windows marked, and the
spans alerts are credited to (`label_spans`, pairs of `DateTime`, named
`span_label` in the legend) shaded; below, the
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

Two stacked panels over the initialisation seeds of repeated training
runs: the decision threshold each run fitted, and the false-alarm rate it
then delivered on the blind record, annotated with the events each run
recovered of `n_events`. `baseline_seed` marks the selected run and `target_far` the rate the threshold criterion asked for. Requires
CairoMakie.
"""
function figure_seed_spread end

"""
    figure_gap_study(levels, families, scored_fraction, events_detected, n_events,
                     false_alarms_per_30d; reference_far = nothing) -> Figure

Three panels side by side over the levels of a delivery-gap study, one row
per level with the level names on the vertical axis in the given order
(first level on top) and a dashed separator between consecutive families:
the windows a replay scored as a fraction of the reference mission, the
coalescences it detected of `n_events`, and its false-alarm episodes per
30 days. A `NaN` rate (a replay that scored nothing) leaves its row empty
in the third panel; `reference_far` draws the reference's rate there as a
labelled guide. Requires CairoMakie.
"""
function figure_gap_study end

"""
    figure_grid_seeds(names, validation_episodes, blind_far_per_30d, events_detected,
                      n_events; seeds, selected = nothing, target_far = nothing) -> Figure

Two panels side by side over the configurations of a model grid trained
at several initialisation seeds, one row per configuration (`names`, the
first on top) and one column of the matrices per seed (`seeds`): the
false-alarm episodes of every run on its validation block, the selection
statistic, and its false-alarm rate per 30 days on the blind record, on a
logarithmic axis when every rate is positive. Seeds are told apart by
marker shape; a run that missed a blind event of `n_events` is drawn open;
a vertical bar marks the median of every configuration. `selected` shades
the row of the selected configuration and `target_far` draws the requested
rate. Entries that are not finite (a run without results) are left out.
Requires CairoMakie.
"""
function figure_grid_seeds end

"""
    figure_telemetry_trace(t_days, strain, labels; max_points = 5000, whitened = nothing) -> Figure

Simulated strain record against mission time with the labelled spans as
shaded bands, an axis offset multiplier for the small strain amplitudes,
and, when `whitened` is given, a second panel with the whitened record.
Requires CairoMakie.
"""
function figure_telemetry_trace end
