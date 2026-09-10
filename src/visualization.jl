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
    figure_telemetry_trace(t_days, strain, labels; max_points = 5000, whitened = nothing) -> Figure

Simulated strain record against mission time with the labeled spans as
shaded bands, an axis offset multiplier for the small strain amplitudes,
and, when `whitened` is given, a second panel with the whitened record.
Requires CairoMakie.
"""
function figure_telemetry_trace end
