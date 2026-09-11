# ext/MilliHertzQMLCairoMakieExt.jl — publication figures of the pipeline,
# loaded together with CairoMakie. Every figure is designed at the printed
# single-column width, shares the project theme (Computer Modern, boxed
# axes, no titles, legend on top), encodes series families by color and
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
using MilliHertzQML: FIGURE_WIDTH_MM, FIGURE_COLORS, backup_existing!, write_toml
using MilliHertzQML: contiguous_runs
using CairoMakie.Makie: scatter!
using DataFrames: DataFrame, nrow
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
    figure_telemetry_alerts

"""
    PT_PER_MM

Typographic points per millimetre; Makie's PDF export uses points as its
unit, so a figure of `FIGURE_WIDTH_MM * PT_PER_MM` points prints at the
declared width.
"""
const PT_PER_MM = 72 / 25.4

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
    rowgap = 0,
    colgap = 8,
    patchsize = (12, 5),
    merge = true,
)

function figure_theme(;
    width_mm::Real = FIGURE_WIDTH_MM,
    height_mm::Real = 0.68 * width_mm,
    fontsize::Real = 8,
)
    width_mm > 0 && height_mm > 0 ||
        throw(ArgumentError("figure dimensions must be positive."))
    return Theme(
        size = (width_mm * PT_PER_MM, height_mm * PT_PER_MM),
        fonts = (;
            regular = texfont(:text),
            bold = texfont(:bold),
            italic = texfont(:italic),
        ),
        fontsize = fontsize,
        figure_padding = (2, 5, 2, 2),
        linewidth = 1.0,
        Axis = (
            xgridstyle = :dash,
            ygridstyle = :dash,
            xgridcolor = (:grey, 0.12),
            ygridcolor = (:grey, 0.12),
            xminorticksvisible = false,
            yminorticksvisible = false,
            xtickalign = 1,
            ytickalign = 1,
            xticksize = 3,
            yticksize = 3,
            spinewidth = 0.6,
            xticklabelpad = 2,
            yticklabelpad = 2,
            xlabelpadding = 2,
            ylabelpadding = 3,
        ),
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
    write_toml(
        "$stem.toml",
        Dict{String,Any}(
            "figure" => Dict{String,Any}(
                "run_id" => run_id,
                "files" => basename.(written),
                "width_mm" => FIGURE_WIDTH_MM,
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

Horizontal legend of the labeled series of `axis` above the axes, in the
first row of the figure layout, in `nbanks` rows.
"""
function top_legend!(figure::Figure, axis::Axis; nbanks::Integer = 1)
    Legend(figure[0, 1], axis; LEGEND_STYLE..., nbanks = nbanks)
    rowgap!(figure.layout, 3)
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
            label = k == 1 ? "Labeled span" : nothing,
        )
    end
    return nothing
end

function figure_training_history(history::NamedTuple)
    epochs = collect(history.epochs)
    isempty(epochs) && throw(ArgumentError("the training history is empty."))
    return with_theme(figure_theme(; height_mm = 0.85 * FIGURE_WIDTH_MM)) do
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
        rowgap!(figure.layout, 4)
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
    return with_theme(figure_theme()) do
        figure = Figure()
        axis =
            Axis(figure[1, 1]; xlabel = "Mission time [days]", ylabel = "MBHB probability")
        labels === nothing || label_bands!(axis, days, labels)
        lines!(
            axis,
            days[idx],
            probabilities[idx];
            color = FIGURE_COLORS.data,
            linewidth = 0.7,
            label = "Classifier output",
        )
        hlines!(
            axis,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
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
    return with_theme(figure_theme(; height_mm = 0.95 * FIGURE_WIDTH_MM)) do
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
            color = FIGURE_COLORS.threshold,
            linestyle = :dot,
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
    decade_label(k) -> String

Plain-decimal tick label of ``10^k``: `1`, `10`, `100`, `0.1`, `0.01`.
"""
decade_label(k::Integer) = k >= 0 ? string(10^k) : "0." * repeat("0", -k - 1) * "1"

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
decades fall inside, labeled as plain decimals.
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
        push!(labels, m == 1 ? decade_label(k) : string(m) * decade_label(k)[2:end])
    end
    return values, labels
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
    return with_theme(figure_theme(; height_mm = 0.95 * FIGURE_WIDTH_MM)) do
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
            lines!(axis, θ, far_log; color = FIGURE_COLORS.signal)
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
                fontsize = 7,
                color = FIGURE_COLORS.signal,
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
                color = FIGURE_COLORS.threshold,
                linestyle = :dot,
            )
            text!(
                ax_far,
                hi,
                Float64(target_far_per_30d);
                text = "Target $(compact(target_far_per_30d)) per 30 d",
                align = (:right, :bottom),
                offset = (0, 2),
                fontsize = 7,
                color = FIGURE_COLORS.threshold,
            )
        end
        isfinite(threshold) &&
            vlines!(ax_far, [threshold]; color = FIGURE_COLORS.threshold, linestyle = :dash)
        linkxaxes!(ax_recall, ax_far)
        hidexdecorations!(ax_recall; grid = false, ticks = false)
        xlims!(ax_far, lo - pad, hi + pad)
        # One legend row per entry: the operating-point statement is long
        top_legend!(figure, ax_recall; nbanks = isfinite(threshold) ? 3 : 2)
        rowgap!(figure.layout, 4)
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
    return with_theme(figure_theme()) do
        figure = Figure()
        axis =
            Axis(figure[1, 1]; xlabel = "Matched-filter SNR", ylabel = "Detected fraction")
        scatterlines!(axis, centers, rates; color = FIGURE_COLORS.data, markersize = 6)
        text!(
            axis,
            centers,
            rates .+ 0.05;
            text = string.(counts),
            align = (:center, :bottom),
            fontsize = 7,
            color = FIGURE_COLORS.data,
        )
        text!(
            axis,
            0.02,
            0.97;
            text = "Numbers: labeled windows per SNR bin",
            space = :relative,
            align = (:left, :top),
            fontsize = 7,
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
    return with_theme(figure_theme()) do
        figure = Figure()
        axis = Axis(figure[1, 1]; xlabel = "Classifier score", ylabel = "Windows")
        if labels === nothing
            hist!(
                axis,
                probabilities;
                bins = bins,
                color = (FIGURE_COLORS.noise, 0.6),
                strokecolor = FIGURE_COLORS.noise,
                strokewidth = 0.4,
                label = "All windows",
            )
        else
            hist!(
                axis,
                probabilities[labels .== 0];
                bins = bins,
                color = (FIGURE_COLORS.noise, 0.6),
                strokecolor = FIGURE_COLORS.noise,
                strokewidth = 0.4,
                label = "Noise windows",
            )
            hist!(
                axis,
                probabilities[labels .== 1];
                bins = bins,
                color = (FIGURE_COLORS.signal, 0.5),
                strokecolor = FIGURE_COLORS.signal,
                strokewidth = 0.4,
                label = "Labeled windows",
            )
        end
        vlines!(
            axis,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
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
    scale = 10.0^exponent
    two_panels = whitened !== nothing
    return with_theme(
        figure_theme(;
            height_mm = two_panels ? 0.95 * FIGURE_WIDTH_MM : 0.68 * FIGURE_WIDTH_MM,
        ),
    ) do
        figure = Figure()
        ax_strain = Axis(
            figure[1, 1];
            ylabel = LaTeXStrings.latexstring("\\mathrm{Strain}\\ [10^{$exponent}]"),
        )
        label_bands!(ax_strain, t_days, labels)
        lines!(
            ax_strain,
            t_days[idx],
            strain[idx] ./ scale;
            color = FIGURE_COLORS.data,
            linewidth = 0.5,
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
                linewidth = 0.5,
            )
            linkxaxes!(ax_strain, ax_white)
            hidexdecorations!(ax_strain; grid = false, ticks = false)
            rowgap!(figure.layout, 4)
        else
            ax_strain.xlabel = "Mission time [days]"
        end
        top_legend!(figure, ax_strain)
        figure
    end
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
)
    nrow(windows) >= 1 || throw(ArgumentError("the windows table is empty."))
    t_days = [days_since(epoch, t) for t in windows.content_end]
    scores = Float64.(windows.score)
    latency_h = [
        Dates.value(a - c) / 3.6e6 for
        (a, c) in zip(windows.complete_at, windows.content_end)
    ]
    # Windows complete out of order: draw them in content-time order
    order = sortperm(t_days)
    t_days = t_days[order]
    scores = scores[order]
    latency_h = latency_h[order]
    alarmed = findall(==(1), Int.(windows.decision)[order])
    return with_theme(figure_theme(; height_mm = 0.95 * FIGURE_WIDTH_MM)) do
        figure = Figure()
        ax_score = Axis(figure[1, 1]; ylabel = "MBHB probability")
        if label_spans !== nothing
            for (k, (a, b)) in enumerate(label_spans)
                vspan!(
                    ax_score,
                    days_since(epoch, a),
                    days_since(epoch, b);
                    color = (FIGURE_COLORS.label, 0.25),
                    label = k == 1 ? "Labeled span" : nothing,
                )
            end
        end
        lines!(
            ax_score,
            t_days,
            scores;
            color = FIGURE_COLORS.data,
            linewidth = 0.7,
            label = "Classifier output",
        )
        isempty(alarmed) || scatter!(
            ax_score,
            t_days[alarmed],
            scores[alarmed];
            color = FIGURE_COLORS.signal,
            markersize = 4,
            label = "Alarm",
        )
        hlines!(
            ax_score,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
            label = "Threshold $(round(threshold; digits = 3))",
        )
        ylims!(ax_score, 0, 1)
        ax_lat = Axis(
            figure[2, 1];
            xlabel = "Mission time [days]",
            ylabel = "Ground latency [h]",
        )
        lines!(ax_lat, t_days, latency_h; color = FIGURE_COLORS.fit, linewidth = 0.7)
        if latencies !== nothing
            for row in eachrow(latencies)
                row.detected || continue
                x = days_since(epoch, row.t_alarm)
                y = Dates.value(row.t_alarm - row.t_merger) / 3.6e6
                scatter!(ax_lat, [x], [y]; color = FIGURE_COLORS.signal, markersize = 5)
                text!(
                    ax_lat,
                    x,
                    y;
                    text = "$(round(row.latency_total_h; digits = 1)) h",
                    align = (:left, :bottom),
                    offset = (3, 2),
                    fontsize = 7,
                    color = FIGURE_COLORS.signal,
                )
            end
        end
        linkxaxes!(ax_score, ax_lat)
        hidexdecorations!(ax_score; grid = false, ticks = false)
        lo, hi = extrema(t_days)
        xlims!(ax_lat, lo, hi == lo ? lo + 1 : hi)
        Legend(figure[0, 1], ax_score; LEGEND_STYLE..., nbanks = 2)
        rowgap!(figure.layout, 4)
        figure
    end
end

end # module
