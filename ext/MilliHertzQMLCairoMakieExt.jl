# ext/MilliHertzQMLCairoMakieExt.jl — publication figures of the pipeline,
# loaded together with CairoMakie. Every figure is built on the base layout
# (900 × 600 pt single panel; each further stacked panel adds 350 pt, each
# auxiliary strip 180 pt), shares the project theme (Computer Modern, boxed
# axes, no titles, legend on top), encodes series families by colour and
# roles by line style, and is exported as vector PDF plus a 4× raster with
# a provenance sidecar.
module MilliHertzQMLCairoMakieExt

using CairoMakie: CairoMakie
using CairoMakie.Makie: Makie, Figure, Axis, Legend, Theme, with_theme
using CairoMakie.Makie: lines!, hlines!, vlines!, vspan!, hist!, scatterlines!, text!
using CairoMakie.Makie: linkxaxes!, hidexdecorations!, rowgap!, xlims!, ylims!, save
using CairoMakie.Makie.MathTeXEngine: texfont
using CairoMakie.Makie: LaTeXStrings
using MilliHertzQML
using MilliHertzQML.StreamingInference: FIGURE_FONTSIZE, TICK_FONTSIZE, ANNOTATION_FONTSIZE
using MilliHertzQML:
    FIGURE_SIZE, FIGURE_STROKES, figure_size, FIGURE_COLORS, backup_existing!, write_toml
using MilliHertzQML: contiguous_runs
using CairoMakie.Makie: scatter!, stairs!, Observable, @lift, Point2f, record
using CairoMakie.Makie: rowsize!, Auto, LinearTicks
using CairoMakie.Makie: widths
using CairoMakie.Makie: linkyaxes!, hideydecorations!, colgap!, hspan!
using DataFrames: DataFrame, nrow
using Statistics: median
using Dates: Dates, DateTime
import MilliHertzQML:
    figure_theme,
    save_figure,
    figure_training_history,
    figure_mission_trace,
    figure_roc,
    figure_threshold_sweep,
    figure_sensitivity,
    figure_score_distribution,
    figure_telemetry_trace,
    figure_telemetry_alerts,
    animation_theme,
    save_animation,
    animate_training_history,
    animate_mission_replay,
    figure_loss_survival,
    figure_seed_spread,
    figure_gap_study,
    figure_grid_seeds

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

function figure_theme(; size = FIGURE_SIZE, fontsize::Real = FIGURE_FONTSIZE)
    size[1] > 0 && size[2] > 0 && fontsize > 0 ||
        throw(ArgumentError("figure dimensions must be positive."))
    return Theme(
        size = size,
        fonts = (;
            regular = texfont(:text),
            bold = texfont(:bold),
            italic = texfont(:italic),
        ),
        fontsize = fontsize,
        figure_padding = (10, 26, 10, 10),   # room for a tick label centred on the right spine
        linewidth = 3,
        markersize = 14,
        Axis = (
            spinewidth = 1.5,
            xticklabelsize = TICK_FONTSIZE,
            yticklabelsize = TICK_FONTSIZE,
            xgridstyle = :dash,
            ygridstyle = :dash,
            xgridcolor = (:grey, 0.12),
            ygridcolor = (:grey, 0.12),
            xminorticksvisible = false,
            yminorticksvisible = false,
            xtickalign = 1,
            ytickalign = 1,
            xticksize = 6,
            yticksize = 6,
            xticklabelpad = 8,
            yticklabelpad = 8,
            xlabelpadding = 8,
            ylabelpadding = 8,
        ),
        Scatter = (strokewidth = 1.5,),
        Legend = (framevisible = false, orientation = :horizontal, titlefont = :bold),
    )
end

function save_figure(
    figure::Figure,
    stem::AbstractString;
    run_id::AbstractString = "",
    formats = ("pdf", "png"),
    px_per_unit::Real = 4,
)
    isempty(formats) && throw(ArgumentError("at least one export format is required."))
    mkpath(dirname(stem))
    written = String[]
    for format in formats
        format in ("pdf", "png", "svg") ||
            throw(ArgumentError("format = $(repr(format)); expected pdf, png, or svg."))
        path = "$stem.$format"
        backup_existing!(path)
        if format == "png"
            save(path, figure; px_per_unit = px_per_unit)
        else
            save(path, figure)
        end
        push!(written, path)
    end
    w, h = round.(Int, widths(figure.scene.viewport[]))
    write_toml(
        "$stem.toml",
        Dict{String,Any}(
            "figure" => Dict{String,Any}(
                "run_id" => run_id,
                "files" => basename.(written),
                "size_pt" => [w, h],
                "px_per_unit" => px_per_unit,
            ),
        ),
    )
    return written
end

"""
    decimation(n, max_points) -> StepRange

Index range keeping about `max_points` of `n` samples.
"""
decimation(n::Integer, max_points::Integer) = 1:max(1, cld(n, max_points)):n

"""
    top_legend!(figure, axis; nbanks = 1)

Horizontal legend of the labelled series of `axis` above the axes, in the
first row of the figure layout, in `nbanks` rows.
"""
function top_legend!(figure::Figure, axis::Axis; nbanks::Integer = 1)
    Legend(figure[0, 1], axis; LEGEND_STYLE..., nbanks = nbanks)
    rowgap!(figure.layout, 10)
    return nothing
end

"""
    label_bands!(axis, x, labels)

Shaded band over every contiguous run of positive `labels` along `x`; the
first band carries the legend entry.
"""
function label_bands!(
    axis::Axis,
    x::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
)
    length(x) == length(labels) ||
        throw(DimensionMismatch("$(length(x)) abscissae for $(length(labels)) labels."))
    for (k, run) in enumerate(contiguous_runs(labels .== 1))
        vspan!(
            axis,
            x[first(run)],
            x[last(run)];
            color = (FIGURE_COLORS.label, 0.25),
            label = k == 1 ? "Labelled span" : nothing,
        )
    end
    return nothing
end

function figure_training_history(history::NamedTuple)
    epochs = collect(history.epochs)
    isempty(epochs) && throw(ArgumentError("the training history is empty."))
    return with_theme(figure_theme(; size = figure_size(2))) do
        figure = Figure()
        ax_loss = Axis(figure[1, 1]; ylabel = "Loss")
        lines!(
            ax_loss,
            epochs,
            history.train_loss;
            color = FIGURE_COLORS.training,
            label = "Training",
        )
        lines!(
            ax_loss,
            epochs,
            history.val_loss;
            color = FIGURE_COLORS.validation,
            linestyle = :dash,
            label = "Validation",
        )
        ax_acc = Axis(figure[2, 1]; xlabel = "Epoch", ylabel = "Validation accuracy")
        lines!(ax_acc, epochs, history.val_acc; color = FIGURE_COLORS.validation)
        linkxaxes!(ax_loss, ax_acc)
        hidexdecorations!(ax_loss; grid = false, ticks = false)
        pad = length(epochs) > 1 ? 0.3 : 1.0
        xlims!(ax_acc, first(epochs) - pad, last(epochs) + pad)
        # Epochs are integers: ticks at a stride that keeps about eight labels
        stride = max(1, round(Int, (last(epochs) - first(epochs) + 1) / 8))
        ax_acc.xticks = first(epochs):stride:last(epochs)
        top_legend!(figure, ax_loss)
        rowgap!(figure.layout, 10)
        figure
    end
end

function figure_mission_trace(
    days::AbstractVector{<:Real},
    probabilities::AbstractVector{<:Real},
    threshold::Real;
    labels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    max_points::Integer = 5000,
)
    n = length(probabilities)
    n == length(days) ||
        throw(DimensionMismatch("$(length(days)) times for $n probabilities."))
    n >= 1 || throw(ArgumentError("the trace is empty."))
    idx = decimation(n, max_points)
    return with_theme(figure_theme(; size = figure_size(1))) do
        figure = Figure()
        axis =
            Axis(figure[1, 1]; xlabel = "Mission time [days]", ylabel = "MBHB probability")
        labels === nothing || label_bands!(axis, days, labels)
        lines!(
            axis,
            days[idx],
            probabilities[idx];
            color = FIGURE_COLORS.data,
            linewidth = 2,
            label = "Classifier output",
        )
        hlines!(
            axis,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
            linewidth = 1.5,
            label = "Threshold $(round(threshold; digits = 3))",
        )
        xlims!(axis, days[1], days[end] == days[1] ? days[1] + 1 : days[end])
        ylims!(axis, 0, 1)
        top_legend!(figure, axis; nbanks = 2)
        figure
    end
end

function figure_roc(fpr::AbstractVector{<:Real}, tpr::AbstractVector{<:Real}, auc::Real)
    length(fpr) == length(tpr) || throw(DimensionMismatch("fpr and tpr differ in length."))
    # Both rates are commensurate, so the axis is square: the canvas keeps
    # the base height and takes only the width the square axis needs.
    return with_theme(figure_theme(; size = (FIGURE_SIZE[2] + 20, FIGURE_SIZE[2]))) do
        figure = Figure()
        axis = Axis(
            figure[1, 1];
            xlabel = "False-positive rate",
            ylabel = "True-positive rate",
            aspect = 1,
        )
        lines!(
            axis,
            [0.0, 1.0],
            [0.0, 1.0];
            color = FIGURE_COLORS.noise,
            linestyle = :dot,
            linewidth = 1.5,
            label = "Chance",
        )
        lines!(
            axis,
            fpr,
            tpr;
            color = FIGURE_COLORS.data,
            label = "VQC, AUC $(round(auc; digits = 3))",
        )
        xlims!(axis, -0.01, 1.01)
        ylims!(axis, -0.01, 1.01)
        top_legend!(figure, axis)
        figure
    end
end

"""
    count_ticks(n) -> Vector{Int}

Ticks `0, s, 2s, …` up to the count `n` for a counting axis, the step `s`
taken from the 1–2–5 sequence as the smallest that gives at most five
ticks. The ticks stop at `n`, so an axis whose limit lies above `n` keeps
its top tick clear of the frame, and of the tick labels of a panel stacked
above it.
"""
function count_ticks(n::Integer)
    n >= 0 || throw(ArgumentError("the count must be non-negative, got $n."))
    n == 0 && return [0]
    steps = sort(vec([m * 10^e for m in (1, 2, 5), e in 0:15]))
    step = steps[findfirst(s -> fld(n, s) + 1 <= 5, steps)]
    return collect(0:step:n)
end

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
    compact(x; digits = 2) -> String

`x` with `digits` decimals, or as an integer when it is one.
"""
function compact(x::Real; digits::Integer = 2)
    return isfinite(x) && x == round(x) ? string(round(Int, x)) :
           string(round(Float64(x); digits = digits))
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

function figure_threshold_sweep(
    sweep::DataFrame,
    threshold::Real;
    target_far_per_30d::Union{Nothing,Real} = nothing,
    operating_point = nothing,
)
    operating_point === nothing ||
        all(
            k -> hasproperty(operating_point, k),
            (:n_detected, :n_events, :false_alarms_per_30d),
        ) ||
        throw(
            ArgumentError(
                "operating_point must carry n_detected, n_events, and false_alarms_per_30d.",
            ),
        )
    nrow(sweep) >= 1 || throw(ArgumentError("the sweep table is empty."))
    for column in (
        "threshold",
        "recall",
        "event_recall",
        "false_alarms_per_30d",
        "n_events",
        "n_detected",
    )
        column in names(sweep) ||
            throw(ArgumentError("the sweep table lacks the column $column."))
    end
    thresholds = Float64.(sweep.threshold)
    finite = findall(isfinite, thresholds)
    isempty(finite) && throw(ArgumentError("the sweep table holds no finite threshold."))
    order = finite[sortperm(thresholds[finite])]
    θ = thresholds[order]
    event_recall = Float64.(sweep.event_recall[order])
    window_recall = Float64.(sweep.recall[order])
    far = Float64.(sweep.false_alarms_per_30d[order])
    has_far = any(x -> x > 0, far)
    # Thresholds without a false alarm are blank on the logarithmic axis
    far_log = [x > 0 ? x : NaN for x in far]
    lo, hi = extrema(θ)
    pad = 0.02 * max(hi - lo, 1e-3)
    # The candidates are score quantiles and are sparse in the far tail, so
    # no row of the sweep reports the applied threshold faithfully: the
    # nearest candidate lies on either side of it and the next one above can
    # be far above. The counts therefore come from the caller, which holds
    # the metrics of the threshold it applied; without them the legend
    # states the threshold alone.
    operating = if !isfinite(threshold)
        ""
    elseif operating_point === nothing
        "Threshold $(round(threshold; digits = 3))"
    else
        "Threshold $(round(threshold; digits = 3)): " *
        "$(operating_point.n_detected)/$(operating_point.n_events) events, " *
        "$(compact(operating_point.false_alarms_per_30d)) per 30 d"
    end
    return with_theme(figure_theme(; size = figure_size(2))) do
        figure = Figure()
        ax_recall = Axis(figure[1, 1]; ylabel = "Recall")
        lines!(ax_recall, θ, event_recall; color = FIGURE_COLORS.data, label = "Events")
        lines!(
            ax_recall,
            θ,
            window_recall;
            color = FIGURE_COLORS.data,
            linestyle = :dash,
            label = "Windows",
        )
        isfinite(threshold) && vlines!(
            ax_recall,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
            linewidth = 1.5,
            label = operating,
        )
        ylims!(ax_recall, -0.03, 1.03)
        ax_far = if has_far
            f_lo = minimum(x for x in far if x > 0) / 1.5
            f_hi = maximum(far) * 1.5
            if target_far_per_30d !== nothing && target_far_per_30d > 0
                f_lo = min(f_lo, target_far_per_30d / 1.5)
                f_hi = max(f_hi, target_far_per_30d * 1.5)
            end
            axis = Axis(
                figure[2, 1];
                xlabel = "Decision threshold",
                ylabel = "False alarms per 30 d",
                yscale = log10,
                yticks = log_ticks(f_lo, f_hi),
            )
            lines!(axis, θ, far_log; color = FIGURE_COLORS.false_alarm)
            ylims!(axis, f_lo, f_hi)
            axis
        else
            axis = Axis(
                figure[2, 1];
                xlabel = "Decision threshold",
                ylabel = "False alarms per 30 d",
            )
            text!(
                axis,
                0.5,
                0.5;
                text = "No false-alarm episode at any threshold",
                space = :relative,
                align = (:center, :center),
                fontsize = ANNOTATION_FONTSIZE,
                color = FIGURE_COLORS.false_alarm,
            )
            ylims!(
                axis,
                0,
                target_far_per_30d === nothing ? 1.0 : max(1.0, 1.3 * target_far_per_30d),
            )
            axis
        end
        if target_far_per_30d !== nothing && (!has_far || target_far_per_30d > 0)
            hlines!(
                ax_far,
                [target_far_per_30d];
                color = FIGURE_COLORS.target,
                linestyle = :dash,
                linewidth = 1.5,
            )
            # Left of centre, where the false-alarm curve runs far above the
            # target: the right end is where the fitted threshold's vertical
            # rule crosses the line.
            text!(
                ax_far,
                lo + 0.35 * (hi - lo),
                Float64(target_far_per_30d);
                text = "Target $(compact(target_far_per_30d)) per 30 d",
                align = (:left, :bottom),
                offset = (0, 6),
                fontsize = ANNOTATION_FONTSIZE,
                color = FIGURE_COLORS.target,
            )
        end
        isfinite(threshold) && vlines!(
            ax_far,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
            linewidth = 1.5,
        )
        linkxaxes!(ax_recall, ax_far)
        hidexdecorations!(ax_recall; grid = false, ticks = false)
        xlims!(ax_far, lo - pad, hi + pad)
        # One legend row per entry: the operating-point statement is long
        top_legend!(figure, ax_recall; nbanks = isfinite(threshold) ? 3 : 2)
        rowgap!(figure.layout, 10)
        figure
    end
end

function figure_sensitivity(
    snrs::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
    decisions::AbstractVector{<:Integer};
    n_bins::Integer = 8,
)
    length(snrs) == length(labels) == length(decisions) ||
        throw(DimensionMismatch("snrs, labels, and decisions differ in length."))
    n_bins >= 1 || throw(ArgumentError("n_bins must be positive."))
    positive = labels .== 1
    any(positive) || return nothing
    snr_pos = snrs[positive]
    lo, hi = extrema(snr_pos)
    edges = range(lo, hi + 1e-6 * max(hi, 1); length = n_bins + 1)
    centers = Float64[]
    rates = Float64[]
    counts = Int[]
    for k in 1:n_bins
        mask = positive .& (snrs .>= edges[k]) .& (snrs .< edges[k+1])
        any(mask) || continue
        push!(centers, (edges[k] + edges[k+1]) / 2)
        push!(rates, count(decisions[mask] .== 1) / count(mask))
        push!(counts, count(mask))
    end
    return with_theme(figure_theme(; size = figure_size(1))) do
        figure = Figure()
        axis =
            Axis(figure[1, 1]; xlabel = "Matched-filter SNR", ylabel = "Detected fraction")
        scatterlines!(
            axis,
            centers,
            rates;
            color = FIGURE_COLORS.data,
            strokewidth = 1.5,
            strokecolor = FIGURE_STROKES.data,
        )
        text!(
            axis,
            centers,
            rates .+ 0.05;
            text = string.(counts),
            align = (:center, :bottom),
            fontsize = ANNOTATION_FONTSIZE,
            color = FIGURE_COLORS.data,
        )
        text!(
            axis,
            0.02,
            0.97;
            text = "Numbers: labeled windows per SNR bin",
            space = :relative,
            align = (:left, :top),
            fontsize = ANNOTATION_FONTSIZE,
            color = FIGURE_COLORS.data,
        )
        ylims!(axis, 0, 1.18)
        span = hi - lo
        xlims!(axis, lo - 0.05 * max(span, 1), hi + 0.05 * max(span, 1))
        figure
    end
end

function figure_score_distribution(
    probabilities::AbstractVector{<:Real},
    threshold::Real;
    labels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    n_bins::Integer = 50,
)
    isempty(probabilities) && throw(ArgumentError("no scores to plot."))
    labels === nothing ||
        length(labels) == length(probabilities) ||
        throw(DimensionMismatch("labels and probabilities differ in length."))
    n_bins >= 1 || throw(ArgumentError("n_bins must be positive."))
    bins = collect(range(0.0, 1.0; length = n_bins + 1))
    return with_theme(figure_theme(; size = figure_size(1))) do
        figure = Figure()
        axis = Axis(figure[1, 1]; xlabel = "Classifier score", ylabel = "Windows")
        if labels === nothing
            hist!(
                axis,
                probabilities;
                bins = bins,
                color = (FIGURE_COLORS.noise, 0.6),
                strokecolor = FIGURE_STROKES.noise,
                strokewidth = 1.5,
                label = "All windows",
            )
        else
            hist!(
                axis,
                probabilities[labels .== 0];
                bins = bins,
                color = (FIGURE_COLORS.noise, 0.6),
                strokecolor = FIGURE_STROKES.noise,
                strokewidth = 1.5,
                label = "Noise windows",
            )
            hist!(
                axis,
                probabilities[labels .== 1];
                bins = bins,
                color = (FIGURE_COLORS.signal, 0.5),
                strokecolor = FIGURE_STROKES.signal,
                strokewidth = 1.5,
                label = "Labelled windows",
            )
        end
        vlines!(
            axis,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
            linewidth = 1.5,
            label = "Threshold $(round(threshold; digits = 3))",
        )
        xlims!(axis, 0, 1)
        top_legend!(figure, axis; nbanks = 2)
        figure
    end
end

function figure_telemetry_trace(
    t_days::AbstractVector{<:Real},
    strain::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer};
    max_points::Integer = 5000,
    whitened::Union{Nothing,AbstractVector{<:Real}} = nothing,
)
    n = length(strain)
    n == length(t_days) == length(labels) ||
        throw(DimensionMismatch("t_days, strain, and labels differ in length."))
    n >= 2 || throw(ArgumentError("the record must hold at least 2 samples."))
    whitened === nothing ||
        length(whitened) == n ||
        throw(DimensionMismatch("the whitened record differs in length."))
    idx = decimation(n, max_points)
    amplitude = maximum(abs, strain)
    exponent = amplitude > 0 ? floor(Int, log10(amplitude)) : 0
    exponent in (0, 1) && (exponent = 0)  # 10⁰ and 10¹ multipliers fold into the tick values
    scale = 10.0^exponent
    two_panels = whitened !== nothing
    return with_theme(
        figure_theme(; size = two_panels ? figure_size(2) : figure_size(1)),
    ) do
        figure = Figure()
        ax_strain = Axis(
            figure[1, 1];
            ylabel = exponent == 0 ? "Strain" :
                     LaTeXStrings.latexstring("\\mathrm{Strain}\\ [10^{$exponent}]"),
        )
        label_bands!(ax_strain, t_days, labels)
        lines!(
            ax_strain,
            t_days[idx],
            strain[idx] ./ scale;
            color = FIGURE_COLORS.data,
            linewidth = 2,
            label = "Simulated record",
        )
        xlims!(ax_strain, t_days[1], t_days[end])
        if two_panels
            ax_white =
                Axis(figure[2, 1]; xlabel = "Mission time [days]", ylabel = "Whitened")
            label_bands!(ax_white, t_days, labels)
            lines!(
                ax_white,
                t_days[idx],
                whitened[idx];
                color = FIGURE_COLORS.data,
                linewidth = 2,
            )
            linkxaxes!(ax_strain, ax_white)
            hidexdecorations!(ax_strain; grid = false, ticks = false)
            rowgap!(figure.layout, 10)
        else
            ax_strain.xlabel = "Mission time [days]"
        end
        top_legend!(figure, ax_strain)
        figure
    end
end

"""
    alert_label_placement(x, y, labels) -> Vector{Symbol}

Position of each alert label relative to its marker, one of `:above_right`,
`:above_left`, `:below_right`, `:below_left`, for markers at axis-fraction
coordinates `x`, `y` (in `[0, 1]`) carrying the texts `labels`. Labels are
placed in order of `x`; each takes the first position whose box stays
inside the axis and clear of every marker and of the labels already placed,
else `:none` (the caller then widens the axis or draws it above and to the
right). The boxes are estimated from the character count at
the annotation size over the lower axis of the two-panel layout
(about 780 × 370 pt); the offsets are those of the drawn labels (9 pt
horizontally, 6 pt vertically).
"""
function alert_label_placement(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    labels::AbstractVector{<:AbstractString},
)
    length(x) == length(y) == length(labels) ||
        throw(DimensionMismatch("x, y and labels must have equal lengths."))
    W, H = 780.0, 370.0
    dx, dy = 9 / W, 6 / H
    h = (ANNOTATION_FONTSIZE + 2) / H
    mx, my = 12 / W, 12 / H                     # marker half-extent with its stroke
    overlaps(a, b) = a[1] < b[2] && b[1] < a[2] && a[3] < b[4] && b[3] < a[4]
    markers = [(x[j] - mx, x[j] + mx, y[j] - my, y[j] + my) for j in eachindex(x)]
    placed = Tuple{Float64,Float64,Float64,Float64}[]
    placement = fill(:none, length(x))
    for i in sortperm(collect(x))
        w = 0.55 * ANNOTATION_FONTSIZE * length(labels[i]) / W
        for p in (:above_right, :above_left, :below_right, :below_left)
            right = p in (:above_right, :below_right)
            above = p in (:above_right, :above_left)
            x0 = right ? x[i] + dx : x[i] - dx - w
            y0 = above ? y[i] + dy : y[i] - dy - h
            box = (x0, x0 + w, y0, y0 + h)
            inside = 0 <= box[1] && box[2] <= 1 && 0 <= box[3] && box[4] <= 1
            clear =
                !any(j -> j != i && overlaps(box, markers[j]), eachindex(x)) &&
                !any(b -> overlaps(box, b), placed)
            if inside && clear
                placement[i] = p
                break
            end
        end
        p = placement[i] == :none ? :above_right : placement[i]
        w = 0.55 * ANNOTATION_FONTSIZE * length(labels[i]) / W
        x0 = p in (:above_right, :below_right) ? x[i] + dx : x[i] - dx - w
        y0 = p in (:above_right, :above_left) ? y[i] + dy : y[i] - dy - h
        push!(placed, (x0, x0 + w, y0, y0 + h))
    end
    return placement
end

"""
    days_since(epoch, t) -> Float64

Mission time `t` in days after `epoch`.
"""
days_since(epoch::DateTime, t::DateTime) = Dates.value(t - epoch) / 8.64e7

function figure_telemetry_alerts(
    windows::DataFrame,
    threshold::Real;
    epoch::DateTime,
    label_spans::Union{Nothing,AbstractVector{<:Tuple{DateTime,DateTime}}} = nothing,
    latencies::Union{Nothing,DataFrame} = nothing,
    span_label::AbstractString = "Labelled span",
)
    nrow(windows) >= 1 || throw(ArgumentError("the windows table is empty."))
    t_days = [days_since(epoch, t) for t in windows.content_end]
    scores = Float64.(windows.score)
    latency_h = [
        Dates.value(a - c) / 3.6e6 for
        (a, c) in zip(scored_at(windows), windows.content_end)
    ]
    # Windows complete out of order: draw them in content-time order
    order = sortperm(t_days)
    t_days = t_days[order]
    scores = scores[order]
    latency_h = latency_h[order]
    alarmed = findall(==(1), Int.(windows.decision)[order])
    return with_theme(figure_theme(; size = figure_size(2))) do
        figure = Figure()
        ax_score = Axis(figure[1, 1]; ylabel = "MBHB probability")
        span_handle = nothing
        if label_spans !== nothing
            for (a, b) in label_spans
                p = vspan!(
                    ax_score,
                    days_since(epoch, a),
                    days_since(epoch, b);
                    color = (FIGURE_COLORS.label, 0.25),
                )
                span_handle === nothing && (span_handle = p)
            end
        end
        score_handle =
            lines!(ax_score, t_days, scores; color = FIGURE_COLORS.data, linewidth = 2)
        alarm_handle =
            isempty(alarmed) ? nothing :
            scatter!(
                ax_score,
                t_days[alarmed],
                scores[alarmed];
                color = FIGURE_COLORS.signal,
                markersize = 8,   # hundreds of alarmed windows; the base marker would merge them
                strokewidth = 1,
                strokecolor = FIGURE_STROKES.signal,
            )
        threshold_handle = hlines!(
            ax_score,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
            linewidth = 1.5,
        )
        ylims!(ax_score, 0, 1)
        # The lower panel carries two different latencies against the same
        # mission time: the availability latency of every scored window (the
        # wait for its conditioning stretch and the downlink delay), and, per
        # event, the alert time measured from the coalescence — negative when
        # the inspiral is alarmed before the merger.
        ax_lat = Axis(figure[2, 1]; xlabel = "Mission time [days]", ylabel = "Latency [h]")
        # The availability latency cycles with the downlink schedule and fills a band
        # at this scale; it is drawn translucent so that alert labels on it
        # remain legible
        availability_handle = lines!(
            ax_lat,
            t_days,
            latency_h;
            color = (FIGURE_COLORS.fit, 0.35),
            linewidth = 2,
        )
        merger_handle = hlines!(
            ax_lat,
            [0.0];
            color = FIGURE_COLORS.noise,
            linestyle = :dot,
            linewidth = 1.5,
        )
        alert_handle = nothing
        alert_x = Float64[]
        alert_y = Float64[]
        if latencies !== nothing
            hits = latencies[latencies.detected .== true, :]
            alert_x = [days_since(epoch, t) for t in hits.t_alarm]
            # Data latency t_alarm − t_merger, the quantity of the benchmark
            # tables; the processing budget is not added
            alert_y =
                [Dates.value(a - m) / 3.6e6 for (a, m) in zip(hits.t_alarm, hits.t_merger)]
            isempty(alert_x) || (
                alert_handle = scatter!(
                    ax_lat,
                    alert_x,
                    alert_y;
                    color = FIGURE_COLORS.signal,
                    marker = :diamond,
                    markersize = 18,
                    strokewidth = 1.5,
                    strokecolor = FIGURE_STROKES.signal,
                )
            )
        end
        linkxaxes!(ax_score, ax_lat)
        hidexdecorations!(ax_score; grid = false, ticks = false)
        lo, hi = extrema(t_days)
        xlims!(ax_lat, lo, hi == lo ? lo + 1 : hi)
        y_lo, y_hi = extrema(vcat(latency_h, alert_y))
        x_span = hi == lo ? 1.0 : hi - lo
        y_span = max(y_hi - y_lo, 1.0)
        alert_labels = map(alert_y) do y
            r = round(y; digits = 1)
            r < 0 ? "−$(-r) h" : "$(abs(r)) h"
        end
        # Each label takes the first position clear of the other markers and
        # labels; the lower limit leaves room for a label set beneath its
        # marker only when one is
        function placement_for(bottom)
            limits = (y_lo - bottom * y_span, y_hi + 0.12 * y_span)
            xf = (alert_x .- lo) ./ x_span
            yf = (alert_y .- limits[1]) ./ (limits[2] - limits[1])
            return alert_label_placement(xf, yf, alert_labels), limits
        end
        placement, limits = placement_for(0.06)
        if any(p -> !(p in (:above_right, :above_left)), placement)
            placement, limits = placement_for(0.14)
        end
        placement = replace(placement, :none => :above_right)
        for (x, y, label, p) in zip(alert_x, alert_y, alert_labels, placement)
            right = p in (:above_right, :below_right)
            above = p in (:above_right, :above_left)
            position = (
                align = (right ? :left : :right, above ? :bottom : :top),
                offset = (right ? 9 : -9, above ? 6 : -6),
            )
            # A white outline beneath the label keeps it legible on the
            # availability trace
            for (color, strokewidth) in ((:white, 3), (FIGURE_COLORS.signal, 0))
                text!(
                    ax_lat,
                    x,
                    y;
                    text = label,
                    position...,
                    fontsize = ANNOTATION_FONTSIZE,
                    color = color,
                    strokecolor = :white,
                    strokewidth = strokewidth,
                )
            end
        end
        ylims!(ax_lat, limits...)
        handles = Any[]
        labels = String[]
        for (h, l) in (
            (span_handle, span_label),
            (score_handle, "Classifier output"),
            (alarm_handle, "Alarm"),
            (threshold_handle, "Threshold $(round(threshold; digits = 3))"),
            (availability_handle, "Window availability"),
            (alert_handle, "Alert time"),
            (merger_handle, "Merger"),
        )
            h === nothing && continue
            push!(handles, h)
            push!(labels, l)
        end
        Legend(figure[0, 1], handles, labels; LEGEND_STYLE..., nbanks = 3)
        rowgap!(figure.layout, 10)
        figure
    end
end

# Animations. A GIF shares the base layout and theme of the figures: fonts,
# colours, boxed axes, and the legend on top. Axis limits are fixed over the
# whole sweep, so nothing rescales between frames.

"""
    ANIMATION_PX_PER_UNIT

Raster scale of an animation in pixels per typographic point. A GIF is read
on screen, so two pixels per point on the 900 pt canvas (1800 px wide) is
its resolution, against four for the printed figures. Whole numbers only,
see [`check_frame_scale`](@ref).
"""
const ANIMATION_PX_PER_UNIT = 2

function animation_theme(; size = FIGURE_SIZE, fontsize::Real = FIGURE_FONTSIZE)
    theme = figure_theme(; size = size, fontsize = fontsize)
    # The canvas is snapped to an even whole number of typographic points. A
    # fractional design size renders to a surface whose extent differs from
    # the frame size declared to the encoder, and the mismatch shows as a band
    # of noise along the top of every frame; with an even size and a whole
    # `px_per_unit` the rendered frame is exactly the declared one.
    theme.size = map(x -> 2.0 * max(1, round(Int, x / 2)), theme.size[])
    return theme
end

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

function save_animation(render, stem::AbstractString; run_id::AbstractString = "")
    mkpath(dirname(stem))
    path = "$stem.gif"
    backup_existing!(path)
    render(path)
    isfile(path) || error("the renderer wrote no file at $path.")
    # Logical screen size of the GIF: little-endian 16-bit width and height
    # at byte offsets 6 and 8
    w, h = open(path) do io
        seek(io, 6)
        (Int(read(io, UInt16)), Int(read(io, UInt16)))
    end
    write_toml(
        "$stem.toml",
        Dict{String,Any}(
            "animation" => Dict{String,Any}(
                "run_id" => run_id,
                "files" => [basename(path)],
                "frame_px" => [w, h],
                "px_per_unit" => ANIMATION_PX_PER_UNIT,
                "bytes" => filesize(path),
            ),
        ),
    )
    return path
end

function animate_training_history(
    history::NamedTuple,
    path::AbstractString;
    framerate::Integer = 5,
    hold_frames::Integer = 10,
    size = figure_size(2),
    px_per_unit::Real = ANIMATION_PX_PER_UNIT,
)
    check_gif_path(path)
    check_frame_scale(px_per_unit)
    framerate >= 1 || throw(ArgumentError("framerate must be positive."))
    epochs = Float64.(collect(history.epochs))
    n = length(epochs)
    n >= 2 || throw(ArgumentError("the training history holds fewer than two epochs."))
    train = Float64.(collect(history.train_loss))
    val = Float64.(collect(history.val_loss))
    acc = Float64.(collect(history.val_acc))
    length(train) == length(val) == length(acc) == n ||
        throw(DimensionMismatch("the history columns differ in length."))
    # The epoch of least validation loss is the checkpoint the run saves
    checkpoint = argmin(val)
    loss_lo, loss_hi = extrema(vcat(train, val))
    loss_span = max(loss_hi - loss_lo, 1e-12)
    acc_lo, acc_hi = extrema(acc)
    acc_span = max(acc_hi - acc_lo, 1e-12)
    x_pad = 0.02 * (n - 1)
    # Ticks at a stride that keeps about eight labels, as in the static figure
    stride = max(1, round(Int, n / 8))
    on_right = epochs[checkpoint] > (epochs[1] + epochs[end]) / 2
    return with_theme(animation_theme(; size = size)) do
        figure = Figure()
        ax_loss = Axis(figure[1, 1]; ylabel = "Loss")
        ax_acc = Axis(figure[2, 1]; xlabel = "Epoch", ylabel = "Validation accuracy")
        linkxaxes!(ax_loss, ax_acc)
        hidexdecorations!(ax_loss; grid = false, ticks = false)
        xlims!(ax_acc, epochs[1] - x_pad, epochs[end] + x_pad)
        # Headroom above the data for the annotations, which sit at the top
        ylims!(ax_loss, loss_lo - 0.06 * loss_span, loss_hi + 0.24 * loss_span)
        ylims!(ax_acc, acc_lo - 0.08 * acc_span, acc_hi + 0.26 * acc_span)
        ax_acc.xticks = round(Int, epochs[1]):stride:round(Int, epochs[end])

        k = Observable(1)
        seen = @lift(1:($k))
        lines!(
            ax_loss,
            @lift(epochs[$seen]),
            @lift(train[$seen]);
            color = FIGURE_COLORS.training,
            label = "Training",
        )
        lines!(
            ax_loss,
            @lift(epochs[$seen]),
            @lift(val[$seen]);
            color = FIGURE_COLORS.validation,
            linestyle = :dash,
            label = "Validation",
        )
        lines!(
            ax_acc,
            @lift(epochs[$seen]),
            @lift(acc[$seen]);
            color = FIGURE_COLORS.validation,
        )
        # The checkpoint appears once the sweep reaches it, in both panels
        mark = @lift($k >= checkpoint ? [epochs[checkpoint]] : Float64[])
        for ax in (ax_loss, ax_acc)
            vlines!(
                ax,
                mark;
                color = FIGURE_COLORS.threshold,
                linestyle = :dot,
                linewidth = 1.5,
            )
        end
        text!(
            ax_loss,
            epochs[checkpoint],
            loss_hi + 0.24 * loss_span;
            text = @lift(
                $k >= checkpoint ?
                "Checkpoint: epoch $(round(Int, epochs[checkpoint])), " *
                "validation loss $(round(val[checkpoint]; digits = 4))" : ""
            ),
            align = (on_right ? :right : :left, :top),
            offset = (on_right ? -15 : 15, -15),
            fontsize = ANNOTATION_FONTSIZE,
            color = FIGURE_COLORS.threshold,
        )
        text!(
            ax_acc,
            0.99,
            0.96;
            text = @lift("Epoch $(round(Int, epochs[$k])) of $(round(Int, epochs[end]))"),
            space = :relative,
            align = (:right, :top),
            fontsize = ANNOTATION_FONTSIZE,
            color = FIGURE_COLORS.threshold,
        )
        top_legend!(figure, ax_loss)
        rowgap!(figure.layout, 8)
        record(
            figure,
            path,
            frame_schedule(n, n, hold_frames);
            framerate = framerate,
            px_per_unit = px_per_unit,
        ) do i
            k[] = i
        end
        path
    end
end

function animate_mission_replay(
    windows::DataFrame,
    threshold::Real,
    path::AbstractString;
    epoch::DateTime = minimum(windows.content_end),
    label_spans::Union{Nothing,AbstractVector{<:Tuple{DateTime,DateTime}}} = nothing,
    span_label::AbstractString = "Labelled span",
    n_frames::Integer = 200,
    framerate::Integer = 20,
    hold_frames::Integer = 20,
    max_points::Integer = 6000,
    size = figure_size(2, 2),
    px_per_unit::Real = ANIMATION_PX_PER_UNIT,
)
    check_gif_path(path)
    check_frame_scale(px_per_unit)
    framerate >= 1 || throw(ArgumentError("framerate must be positive."))
    max_points >= 2 || throw(ArgumentError("max_points must be at least 2."))
    n = nrow(windows)
    n >= 2 || throw(ArgumentError("the replay holds fewer than two windows."))
    for column in ("content_end", "complete_at", "coverage", "score", "decision")
        column in names(windows) ||
            throw(ArgumentError("the windows table lacks the column $column."))
    end
    # Mission order is the axis of every panel; permuting once makes window
    # adjacency, which defines an alarm episode, index adjacency.
    days = [days_since(epoch, t) for t in windows.content_end]
    perm = sortperm(days)
    content_day = days[perm]
    scored = scored_at(windows)
    arrival_day = [days_since(epoch, scored[i]) for i in perm]
    latency_h = 24 .* (arrival_day .- content_day)
    coverage = Float64.(windows.coverage[perm])
    score = Float64.(windows.score[perm])
    alarm = Int.(windows.decision[perm]) .== 1
    alarm_idx = findall(alarm)
    # Reveal order: the order in which the windows became evaluable. A pass
    # delivers its backlog newest first and a window becomes evaluable only
    # once the conditioning stretch around it has landed, so an arrival prefix
    # need not be a prefix in mission time.
    arrival_order = sortperm(arrival_day)
    reveal_rank = Vector{Int}(undef, n)
    reveal_rank[arrival_order] = 1:n
    # The traces are decimated, the alarmed windows always kept
    show_idx = sort(unique(vcat(collect(decimation(n, max_points)), n, alarm_idx)))
    show_day = content_day[show_idx]
    alarm_day = content_day[alarm_idx]
    n_episodes = max(count(alarm .& .!vcat(false, alarm[1:(end-1)])), 1)
    t_lo, t_hi = extrema(content_day)
    t_pad = 0.01 * max(t_hi - t_lo, 1e-6)
    lat_lo, lat_hi = extrema(latency_h)
    lat_pad = 0.08 * max(lat_hi - lat_lo, 1e-3)
    full_coverage = all(>=(1.0), coverage)
    return with_theme(animation_theme(; size = size)) do
        figure = Figure()
        # Row 1 is the legend; explicit rows keep the panel sizes unambiguous
        ax_cov = Axis(figure[2, 1]; ylabel = "Coverage", yticks = [0.0, 0.5, 1.0])
        ax_score = Axis(figure[3, 1]; ylabel = "MBHB probability")
        ax_episode =
            Axis(figure[4, 1]; ylabel = "Alarm episodes", yticks = count_ticks(n_episodes))
        ax_lat = Axis(
            figure[5, 1];
            xlabel = "Mission time [days]",
            ylabel = "Window availability [h]",
            xticks = LinearTicks(7),
        )
        panels = (ax_cov, ax_score, ax_episode, ax_lat)
        linkxaxes!(panels...)
        for ax in panels[1:3]
            hidexdecorations!(ax; grid = false, ticks = false)
        end
        xlims!(ax_lat, t_lo - t_pad, t_hi + t_pad)
        ylims!(ax_cov, -0.12, 1.22)
        ylims!(ax_score, 0, 1)
        ylims!(ax_episode, -0.04 * n_episodes, 1.12 * n_episodes)
        ylims!(ax_lat, lat_lo - lat_pad, lat_hi + lat_pad)

        cov_y = Observable(fill(NaN, length(show_idx)))
        score_y = Observable(fill(NaN, length(show_idx)))
        lat_y = Observable(fill(NaN, length(show_idx)))
        alarm_y = Observable(fill(NaN, length(alarm_idx)))
        episode_points = Observable([Point2f(t_lo, 0), Point2f(t_lo, 0)])
        clock_x = Observable([t_lo])
        progress = Observable("")
        reveal! =
            k -> begin
                received = reveal_rank .<= k
                cov_y[] = [received[i] ? coverage[i] : NaN for i in show_idx]
                score_y[] = [received[i] ? score[i] : NaN for i in show_idx]
                lat_y[] = [received[i] ? latency_h[i] : NaN for i in show_idx]
                alarm_y[] = [received[i] ? score[i] : NaN for i in alarm_idx]
                # Episodes among the windows received so far: a run of alarmed
                # windows adjacent in mission index counts once, and an arrival
                # that fills the hole between two runs merges them.
                raised = received .& alarm
                starts = findall(raised .& .!vcat(false, raised[1:(end-1)]))
                edge_lo, edge_hi = extrema(content_day[received])
                points = Vector{Point2f}(undef, length(starts) + 2)
                points[1] = Point2f(edge_lo, 0)
                for (j, i) in enumerate(starts)
                    points[j+1] = Point2f(content_day[i], j)
                end
                points[end] = Point2f(edge_hi, length(starts))
                episode_points[] = points
                # The ground clock is the arrival time of the newest window
                # received; its distance to the data edge is the latency.
                clock = arrival_day[arrival_order[k]]
                clock_x[] = [clock]
                progress[] =
                    "Ground clock: day $(compact(clock; digits = 1))\n" *
                    "Windows received: $k of $n"
                return nothing
            end
        reveal!(1)

        span_handle = nothing
        if label_spans !== nothing
            for (a, b) in label_spans
                p = vspan!(
                    ax_score,
                    days_since(epoch, a),
                    days_since(epoch, b);
                    color = (FIGURE_COLORS.label, 0.25),
                )
                span_handle === nothing && (span_handle = p)
            end
        end
        clock_handle = nothing
        for ax in panels
            p = vlines!(
                ax,
                clock_x;
                color = FIGURE_COLORS.threshold,
                linestyle = :dot,
                linewidth = 1.5,
            )
            clock_handle === nothing && (clock_handle = p)
        end
        lines!(ax_cov, show_day, cov_y; color = FIGURE_COLORS.data, linewidth = 2)
        full_coverage && text!(
            ax_cov,
            0.012,
            0.06;
            text = "Every window fully covered",
            space = :relative,
            align = (:left, :bottom),
            fontsize = ANNOTATION_FONTSIZE,
            color = FIGURE_COLORS.data,
        )
        score_handle =
            lines!(ax_score, show_day, score_y; color = FIGURE_COLORS.data, linewidth = 2)
        alarm_handle = scatter!(
            ax_score,
            alarm_day,
            alarm_y;
            color = FIGURE_COLORS.signal,
            strokewidth = 1.5,
            strokecolor = FIGURE_STROKES.signal,
        )
        threshold_handle = hlines!(
            ax_score,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
            linewidth = 1.5,
        )
        stairs!(ax_episode, episode_points; step = :post, color = FIGURE_COLORS.signal)
        text!(
            ax_episode,
            0.012,
            0.96;
            text = progress,
            space = :relative,
            align = (:left, :top),
            fontsize = ANNOTATION_FONTSIZE,
            color = FIGURE_COLORS.threshold,
        )
        # A year of daily passes packs the latency sawtooth into a few pixels
        # per period, so the trace is drawn light enough to read as a band
        lines!(ax_lat, show_day, lat_y; color = (FIGURE_COLORS.fit, 0.85), linewidth = 2)

        handles = Any[]
        labels = String[]
        for (h, l) in (
            (span_handle, span_label),
            (score_handle, "Classifier output"),
            (alarm_handle, "Alarm"),
            (threshold_handle, "Threshold $(round(threshold; digits = 3))"),
            (clock_handle, "Ground clock"),
        )
            h === nothing && continue
            push!(handles, h)
            push!(labels, l)
        end
        Legend(figure[1, 1], handles, labels; LEGEND_STYLE..., nbanks = 2)
        rowgap!(figure.layout, 10)
        # The coverage and episode panels are strips: one flat trace and one
        # counter, at half the height of the score and latency panels.
        rowsize!(figure.layout, 2, Auto(0.5))
        rowsize!(figure.layout, 4, Auto(0.5))
        record(
            figure,
            path,
            frame_schedule(n, n_frames, hold_frames);
            framerate = framerate,
            px_per_unit = px_per_unit,
        ) do k
            reveal!(k)
        end
        path
    end
end

"""
    log_decimal_ticks(lo, hi) -> (values, labels)

Decade and 3× intermediate ticks of the closed interval `[lo, hi]`,
labelled as plain decimals. A logarithmic axis of a few decades reads as
decimals rather than as powers, and never as a fractional exponent.
"""
function log_decimal_ticks(lo::Real, hi::Real)
    values = Float64[]
    for d in floor(Int, log10(lo)):ceil(Int, log10(hi)), m in (1.0, 3.0)
        v = m * 10.0^d
        lo <= v <= hi && push!(values, v)
    end
    labels = map(values) do v
        v >= 1 ? string(round(Int, v)) : rstrip(rstrip(string(round(v; digits = 4)), '0'), '.')
    end
    return (values, labels)
end

function figure_loss_survival(
    p_loss::AbstractVector{<:Real},
    scored_fraction::AbstractVector{<:Real},
    events_detected::AbstractVector{<:Integer},
    n_events::Integer;
    stretch_batches::Integer,
    model::Bool = true,
)
    n = length(p_loss)
    n == length(scored_fraction) == length(events_detected) || throw(
        DimensionMismatch("the loss, survival and detection vectors differ in length."),
    )
    n >= 1 || throw(ArgumentError("the sweep is empty."))
    all(>(0), p_loss) || throw(
        ArgumentError("a logarithmic loss axis admits no zero; omit the lossless run."),
    )
    stretch_batches >= 1 || throw(ArgumentError("the stretch must be at least one batch."))

    percent = 100 .* p_loss
    knee = 100 / stretch_batches      # losses spaced one stretch apart
    lo, hi = extrema(percent)
    grid = exp10.(range(log10(0.6 * min(lo, knee)), log10(1.6 * hi); length = 200))
    return with_theme(figure_theme(; size = figure_size(1, 1))) do
        figure = Figure()
        ticks = log_decimal_ticks(first(grid), last(grid))
        ax_survival = Axis(
            figure[1, 1];
            ylabel = "Windows scored, of lossless",
            xscale = log10,
            xticks = ticks,
            yticks = 0:0.25:1,
        )
        ax_events = Axis(
            figure[2, 1];
            xlabel = "Permanent batch loss [%]",
            ylabel = "Events detected",
            xscale = log10,
            xticks = ticks,
            yticks = 0:1:n_events,
        )
        linkxaxes!(ax_survival, ax_events)
        hidexdecorations!(ax_survival; grid = false, ticks = false)
        xlims!(ax_events, first(grid), last(grid))
        ylims!(ax_survival, -0.05, 1.1)
        ylims!(ax_events, -0.3, n_events + 0.4)

        if model
            lines!(
                ax_survival,
                grid,
                (1 .- grid ./ 100) .^ stretch_batches;
                color = FIGURE_COLORS.fit,
                linestyle = :dash,
                label = "Independent-batch estimate",
            )
        end
        # The rate at which the mean spacing of losses equals the stretch:
        # above it a stretch without a lost batch is rare, not typical.
        vlines!(
            ax_survival,
            [knee];
            color = FIGURE_COLORS.target,
            linestyle = :dot,
            linewidth = 1.5,
        )
        text!(
            ax_survival,
            knee,
            1.06;
            text = "1 / stretch",
            align = (:left, :top),
            offset = (6, 0),
            fontsize = ANNOTATION_FONTSIZE,
            color = FIGURE_COLORS.target,
        )
        scatterlines!(
            ax_survival,
            percent,
            scored_fraction;
            color = FIGURE_COLORS.data,
            strokewidth = 1.5,
            strokecolor = FIGURE_STROKES.data,
            label = "Measured",
        )
        scatterlines!(
            ax_events,
            percent,
            Float64.(events_detected);
            color = FIGURE_COLORS.signal,
            strokewidth = 1.5,
            strokecolor = FIGURE_STROKES.signal,
        )
        top_legend!(figure, ax_survival)
        rowgap!(figure.layout, 10)
        figure
    end
end

function figure_seed_spread(
    seeds::AbstractVector{<:Integer},
    thresholds::AbstractVector{<:Real},
    far_per_30d::AbstractVector{<:Real},
    events_detected::AbstractVector{<:Integer},
    n_events::Integer;
    baseline_seed::Union{Nothing,Integer} = nothing,
    target_far::Union{Nothing,Real} = nothing,
)
    n = length(seeds)
    n == length(thresholds) == length(far_per_30d) == length(events_detected) ||
        throw(DimensionMismatch("the seed, threshold, rate and detection vectors differ."))
    n >= 2 || throw(ArgumentError("a spread needs at least two runs."))

    x = collect(1:n)
    selected = baseline_seed === nothing ? Int[] : findall(==(baseline_seed), seeds)
    return with_theme(figure_theme(; size = figure_size(2))) do
        figure = Figure()
        ax_thr = Axis(figure[1, 1]; ylabel = "Fitted threshold")
        ax_far = Axis(
            figure[2, 1];
            xlabel = "Initialisation seed",
            ylabel = "False alarms / 30 d",
        )
        linkxaxes!(ax_thr, ax_far)
        hidexdecorations!(ax_thr; grid = false, ticks = false)
        ax_far.xticks = (x, string.(seeds))
        xlims!(ax_far, 0.5, n + 0.5)
        # Headroom above whichever is higher, the requested rate or the
        # highest delivered rate, so that neither the rule nor the count meets the frame.
        far_top =
            target_far === nothing ? maximum(far_per_30d) :
            max(target_far, maximum(far_per_30d))
        far_low = minimum(far_per_30d)
        ylims!(
            ax_far,
            far_low - 0.12 * (far_top - far_low),
            far_top + 0.22 * (far_top - far_low),
        )

        if target_far !== nothing
            hlines!(
                ax_far,
                [target_far];
                color = FIGURE_COLORS.target,
                linestyle = :dash,
                linewidth = 1.5,
                label = "Requested rate",
            )
        end
        scatter!(
            ax_thr,
            x,
            thresholds;
            color = FIGURE_COLORS.data,
            strokewidth = 1.5,
            strokecolor = FIGURE_STROKES.data,
        )
        scatter!(
            ax_far,
            x,
            far_per_30d;
            color = FIGURE_COLORS.false_alarm,
            strokewidth = 1.5,
            strokecolor = FIGURE_STROKES.false_alarm,
            label = "Delivered rate",
        )
        if !isempty(selected)
            scatter!(
                ax_thr,
                x[selected],
                thresholds[selected];
                color = FIGURE_COLORS.signal,
                markersize = 20,
                marker = :diamond,
                strokewidth = 1.5,
                strokecolor = FIGURE_STROKES.signal,
            )
            scatter!(
                ax_far,
                x[selected],
                far_per_30d[selected];
                color = FIGURE_COLORS.signal,
                markersize = 20,
                marker = :diamond,
                strokewidth = 1.5,
                strokecolor = FIGURE_STROKES.signal,
                label = "Selected run",
            )
        end
        # Every run recovered the same events, so the count is stated once
        # rather than repeated over each marker; it sits in the headroom
        # above the highest rate, clear of the markers.
        recovered =
            all(==(n_events), events_detected) ? "$n_events of $n_events events" :
            "$(minimum(events_detected))–$(maximum(events_detected)) of $n_events events"
        text!(
            ax_far,
            0.5 + 0.05 * n,
            far_top + 0.20 * (far_top - far_low);
            text = recovered,
            align = (:left, :top),
            fontsize = ANNOTATION_FONTSIZE,
            color = FIGURE_COLORS.signal,
        )
        top_legend!(figure, ax_far)
        rowgap!(figure.layout, 10)
        figure
    end
end

function figure_gap_study(
    levels::AbstractVector{<:AbstractString},
    families::AbstractVector{<:AbstractString},
    scored_fraction::AbstractVector{<:Real},
    events_detected::AbstractVector{<:Integer},
    n_events::Integer,
    false_alarms_per_30d::AbstractVector{<:Real};
    reference_far::Union{Nothing,Real} = nothing,
)
    n = length(levels)
    lengths = (
        length(families),
        length(scored_fraction),
        length(events_detected),
        length(false_alarms_per_30d),
    )
    all(==(n), lengths) || throw(
        DimensionMismatch(
            "the level, family, survival, detection and false-alarm vectors " *
            "differ in length.",
        ),
    )
    n >= 1 || throw(ArgumentError("the study is empty."))
    n_events >= 1 || throw(ArgumentError("n_events must be at least 1."))
    all(>=(0), scored_fraction) || throw(ArgumentError("a scored fraction is negative."))
    all(e -> 0 <= e <= n_events, events_detected) ||
        throw(ArgumentError("events detected exceed n_events."))

    y = collect(n:-1:1)      # the first level at the top
    return with_theme(figure_theme(; size = (1200, max(600, 220 + 36 * n)))) do
        figure = Figure()
        ax_windows = Axis(
            figure[1, 1];
            xlabel = "Windows scored, of reference",
            yticks = (y, String.(levels)),
            xticks = 0:0.25:1,
        )
        # The inner panels tick every level as the first does; their labels
        # are hidden below
        ax_events = Axis(
            figure[1, 2];
            xlabel = "Events detected",
            xticks = 0:1:n_events,
            yticks = y,
        )
        ax_far = Axis(figure[1, 3]; xlabel = "False alarms per 30 d", yticks = y)
        linkyaxes!(ax_windows, ax_events, ax_far)
        hideydecorations!(ax_events; grid = false, ticks = false)
        hideydecorations!(ax_far; grid = false, ticks = false)
        ylims!(ax_windows, 0.4, n + 0.6)
        xlims!(ax_windows, -0.05, 1.12)
        xlims!(ax_events, -0.4, n_events + 0.4)

        # A dashed rule between consecutive families of levels, across all panels
        for i in 2:n
            families[i] == families[i-1] && continue
            for ax in (ax_windows, ax_events, ax_far)
                hlines!(
                    ax,
                    [y[i] + 0.5];
                    color = (:grey, 0.5),
                    linestyle = :dash,
                    linewidth = 1.5,
                )
            end
        end

        vlines!(
            ax_windows,
            [1.0];
            color = FIGURE_COLORS.target,
            linestyle = :dot,
            linewidth = 1.5,
        )
        scatter!(
            ax_windows,
            Float64.(scored_fraction),
            y;
            color = FIGURE_COLORS.data,
            strokewidth = 1.5,
            strokecolor = FIGURE_STROKES.data,
        )
        scatter!(
            ax_events,
            Float64.(events_detected),
            y;
            color = FIGURE_COLORS.signal,
            strokewidth = 1.5,
            strokecolor = FIGURE_STROKES.signal,
        )

        # A replay that scored nothing has no rate; its row stays empty.
        finite = isfinite.(false_alarms_per_30d)
        top = maximum(
            vcat(
                Float64.(false_alarms_per_30d[finite]),
                reference_far === nothing ? Float64[] : [Float64(reference_far)],
            );
            init = 0.0,
        )
        xlims!(ax_far, 0.0, top > 0 ? 1.15 * top : 1.0)
        if reference_far !== nothing
            vlines!(
                ax_far,
                [reference_far];
                color = FIGURE_COLORS.target,
                linestyle = :dash,
                linewidth = 1.5,
            )
            text!(
                ax_far,
                Float64(reference_far),
                n + 0.55;
                text = "Reference",
                align = (:left, :top),
                offset = (6, -2),
                fontsize = ANNOTATION_FONTSIZE,
                color = FIGURE_COLORS.target,
            )
        end
        if any(finite)
            scatter!(
                ax_far,
                Float64.(false_alarms_per_30d[finite]),
                y[finite];
                color = FIGURE_COLORS.false_alarm,
                strokewidth = 1.5,
                strokecolor = FIGURE_STROKES.false_alarm,
            )
        end
        colgap!(figure.layout, 12)
        figure
    end
end

function figure_grid_seeds(
    names::AbstractVector{<:AbstractString},
    validation_episodes::AbstractMatrix{<:Real},
    blind_far_per_30d::AbstractMatrix{<:Real},
    events_detected::AbstractMatrix{<:Real},
    n_events::Integer;
    seeds::AbstractVector{<:Integer} = collect(1:size(validation_episodes, 2)),
    selected::Union{Nothing,AbstractString} = nothing,
    target_far::Union{Nothing,Real} = nothing,
)
    n, m = size(validation_episodes)
    size(blind_far_per_30d) == (n, m) && size(events_detected) == (n, m) || throw(
        DimensionMismatch(
            "the validation, false-alarm and detection matrices differ in size.",
        ),
    )
    length(names) == n ||
        throw(DimensionMismatch("one name per row of the matrices is required."))
    length(seeds) == m ||
        throw(DimensionMismatch("one seed per column of the matrices is required."))
    n >= 1 && m >= 1 || throw(ArgumentError("the grid is empty."))
    n_events >= 1 || throw(ArgumentError("n_events must be at least 1."))
    present = isfinite.(blind_far_per_30d) .& isfinite.(validation_episodes)
    any(present) || throw(ArgumentError("no run of the grid has results."))
    all(e -> 0 <= e <= n_events, events_detected[present]) ||
        throw(ArgumentError("events detected exceed n_events."))
    all(>=(0), blind_far_per_30d[present]) ||
        throw(ArgumentError("a false-alarm rate is negative."))
    selected === nothing ||
        selected in names ||
        throw(ArgumentError("the selected configuration $selected is not a row."))

    y = collect(n:-1:1)      # the first configuration at the top
    jitter = m == 1 ? [0.0] : collect(range(-0.24, 0.24; length = m))
    markers = (:circle, :rect, :diamond, :utriangle, :dtriangle, :star5)
    rates = Float64.(blind_far_per_30d[present])
    logscale = all(>(0), rates)
    return with_theme(figure_theme(; size = (1200, max(600, 260 + 52 * n)))) do
        figure = Figure()
        ax_val = Axis(
            figure[1, 1];
            xlabel = "Validation false-alarm episodes",
            yticks = (y, String.(names)),
        )
        ax_far = if logscale
            lo, hi = extrema(rates)
            lo, hi = lo / 1.4, hi * 1.4
            Axis(
                figure[1, 2];
                xlabel = "Blind false alarms per 30 d",
                xscale = log10,
                xticks = dense_log_ticks(lo, hi),
                yticks = y,
            )
        else
            Axis(figure[1, 2]; xlabel = "Blind false alarms per 30 d", yticks = y)
        end
        linkyaxes!(ax_val, ax_far)
        hideydecorations!(ax_far; grid = false, ticks = false)
        ylims!(ax_val, 0.4, n + 0.6)
        logscale && xlims!(ax_far, lo, hi)
        v_hi = maximum(validation_episodes[present])
        xlims!(ax_val, -0.05 * max(v_hi, 1), 1.08 * max(v_hi, 1))

        if selected !== nothing
            r = y[findfirst(==(selected), names)]
            for ax in (ax_val, ax_far)
                hspan!(ax, r - 0.46, r + 0.46; color = (:grey, 0.15))
            end
        end
        target_handle = nothing
        if target_far !== nothing
            target_handle = vlines!(
                ax_far,
                [target_far];
                color = FIGURE_COLORS.target,
                linestyle = :dash,
                linewidth = 1.5,
            )
        end
        seed_handles = Any[]
        missed_handle = nothing
        for j in 1:m
            rows = findall(present[:, j])
            isempty(rows) && continue
            marker = markers[mod1(j, length(markers))]
            yj = y[rows] .+ jitter[j]
            h = scatter!(
                ax_val,
                Float64.(validation_episodes[rows, j]),
                yj;
                marker = marker,
                color = FIGURE_COLORS.false_alarm,
                strokewidth = 1.5,
                strokecolor = FIGURE_STROKES.false_alarm,
            )
            push!(seed_handles, (h, "Seed $(seeds[j])"))
            # A run that missed a blind event is drawn open in the blind panel
            complete = [events_detected[i, j] == n_events for i in rows]
            scatter!(
                ax_far,
                Float64.(blind_far_per_30d[rows[complete], j]),
                yj[complete];
                marker = marker,
                color = FIGURE_COLORS.false_alarm,
                strokewidth = 1.5,
                strokecolor = FIGURE_STROKES.false_alarm,
            )
            if !all(complete)
                p = scatter!(
                    ax_far,
                    Float64.(blind_far_per_30d[rows[.!complete], j]),
                    yj[.!complete];
                    marker = marker,
                    color = :white,
                    strokewidth = 1.5,
                    strokecolor = FIGURE_STROKES.false_alarm,
                )
                missed_handle === nothing && (missed_handle = p)
            end
        end
        # The median over the seeds of every configuration
        median_handle = nothing
        for (ax, values) in ((ax_val, validation_episodes), (ax_far, blind_far_per_30d))
            xs = Float64[]
            ys = Float64[]
            for i in 1:n
                v = Float64.(values[i, present[i, :]])
                isempty(v) && continue
                push!(xs, median(v))
                push!(ys, y[i])
            end
            median_handle = scatter!(
                ax,
                xs,
                ys;
                marker = :vline,
                markersize = 30,
                color = FIGURE_COLORS.threshold,
            )
        end
        handles = Any[first.(seed_handles)...]
        labels = String[last.(seed_handles)...]
        push!(handles, median_handle)
        push!(labels, "Median")
        if missed_handle !== nothing
            push!(handles, missed_handle)
            push!(labels, "Blind event missed")
        end
        if target_handle !== nothing
            push!(handles, target_handle)
            push!(labels, "Requested rate")
        end
        Legend(figure[0, 1:2], handles, labels; LEGEND_STYLE..., nbanks = 2)
        colgap!(figure.layout, 12)
        rowgap!(figure.layout, 10)
        figure
    end
end

end # module
