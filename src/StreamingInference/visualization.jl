# Figure interface of the domain-general layer (implemented by the CairoMakie
# extension): the theme, geometry and palette of every figure, and the
# evaluation, score and replay figures.

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

# Figure toolkit shared by the figures of every layer: legend style, tick
# labelling of logarithmic axes, decimation of long traces, and the frame
# checks of the animations. Public, not exported; `top_legend!` and
# `label_bands!` are implemented by the CairoMakie extension.

"""
    LEGEND_STYLE

Keyword arguments of the horizontal legend placed above the axes.
"""
const LEGEND_STYLE = (
    orientation = :horizontal,
    tellheight = true,
    tellwidth = false,
    framevisible = false,
    padding = (0, 0, 0, 0),
    rowgap = 4,
    colgap = 16,
    patchsize = (40, 14),
    merge = true,
    titlefont = :bold,
)

"""
    decimation(n, max_points) -> StepRange

Index range keeping about `max_points` of `n` samples.
"""
decimation(n::Integer, max_points::Integer) = 1:max(1, cld(n, max_points)):n

"""
    decade_label(k, m = 1) -> String

Plain-decimal tick label of ``m \\times 10^k`` for a single-digit mantissa
`m`: `1`, `10`, `100`, `0.1`, `0.01`; `20`, `0.2`, `0.05`.
"""
function decade_label(k::Integer, m::Integer = 1)
    1 <= m <= 9 || throw(ArgumentError("m = $m; the mantissa must be a single digit."))
    return k >= 0 ? string(m * 10^k) : "0." * repeat("0", -k - 1) * string(m)
end

"""
    log_ticks(lo, hi) -> (values, labels)

Tick values of a logarithmic axis spanning `[lo, hi]`: the decades inside
the range, with the 2× and 5× intermediates added when fewer than two
decades fall inside, labelled as plain decimals.
"""
function log_ticks(lo::Real, hi::Real)
    0 < lo <= hi || throw(ArgumentError("a logarithmic range needs 0 < lo <= hi."))
    k_lo = floor(Int, log10(lo))
    k_hi = ceil(Int, log10(hi))
    decades = [k for k in k_lo:k_hi if lo <= 10.0^k <= hi]
    if length(decades) >= 2
        return 10.0 .^ decades, decade_label.(decades)
    end
    values = Float64[]
    labels = String[]
    for k in k_lo:k_hi, m in (1, 2, 5)
        v = m * 10.0^k
        lo <= v <= hi || continue
        push!(values, v)
        push!(labels, decade_label(k, m))
    end
    return values, labels
end

"""
    dense_log_ticks(lo, hi) -> (values, labels)

Ticks of a logarithmic axis spanning `[lo, hi]` at the 1×, 2× and 5×
multiples of every decade, for axes that span two or three decades; above
nine such ticks it falls back to [`log_ticks`](@ref).
"""
function dense_log_ticks(lo::Real, hi::Real)
    0 < lo <= hi || throw(ArgumentError("a logarithmic range needs 0 < lo <= hi."))
    values = Float64[]
    labels = String[]
    for k in floor(Int, log10(lo)):ceil(Int, log10(hi)), m in (1, 2, 5)
        v = m * 10.0^k
        lo <= v <= hi || continue
        push!(values, v)
        push!(labels, decade_label(k, m))
    end
    return length(values) <= 9 ? (values, labels) : log_ticks(lo, hi)
end

"""
    ANIMATION_PX_PER_UNIT

Raster scale of an animation in pixels per typographic point. A GIF is read
on screen, so two pixels per point on the 900 pt canvas (1800 px wide) is
its resolution, against four for the printed figures. Whole numbers only,
see [`check_frame_scale`](@ref).
"""
const ANIMATION_PX_PER_UNIT = 2

"""
    check_frame_scale(px_per_unit)

Throw unless `px_per_unit` is a whole number at least one. A fractional
raster scale renders frames that do not match the size declared to the
video encoder, which appears as a band of noise along their top edge.
"""
function check_frame_scale(px_per_unit::Real)
    isinteger(px_per_unit) && px_per_unit >= 1 || throw(
        ArgumentError("px_per_unit = $px_per_unit; expected a whole number of at least 1."),
    )
    return nothing
end

"""
    check_gif_path(path)

Throw unless `path` names a GIF; the animations of the project are written
as GIF.
"""
function check_gif_path(path::AbstractString)
    endswith(lowercase(path), ".gif") ||
        throw(ArgumentError("path = $(repr(path)); an animation is written as .gif."))
    return nothing
end

"""
    frame_schedule(n, n_frames, hold_frames) -> Vector{Int}

Sweep of about `n_frames` states out of `n`, always ending on `n`, followed
by `hold_frames` repetitions of the last state. The sweep starts at two
states, since a single one draws no line segment.
"""
function frame_schedule(n::Integer, n_frames::Integer, hold_frames::Integer)
    n >= 2 || throw(ArgumentError("n must be at least 2."))
    n_frames >= 2 || throw(ArgumentError("n_frames must be at least 2."))
    hold_frames >= 0 || throw(ArgumentError("hold_frames must not be negative."))
    sweep = unique(round.(Int, range(2, n; length = min(n_frames, n - 1))))
    return vcat(sweep, fill(n, hold_frames))
end

"""
    top_legend!(figure, axis; nbanks = 1)

Horizontal legend of the labelled series of `axis` above the axes, in the
first row of the figure layout, in `nbanks` rows. Requires CairoMakie.
"""
function top_legend! end

"""
    label_bands!(axis, x, labels)

Shaded band over every contiguous run of positive `labels` along `x`; the
first band carries the legend entry. Requires CairoMakie.
"""
function label_bands! end
