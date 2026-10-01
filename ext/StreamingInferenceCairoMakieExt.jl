# ext/StreamingInferenceCairoMakieExt.jl — figures of the domain-general
# layer, loaded together with CairoMakie: the theme every figure of the
# pipeline shares, the PDF/PNG and GIF exports with their provenance
# sidecars, and the evaluation, score and replay figures. Every figure is
# built on the base layout (900 × 600 pt single panel; each further stacked
# panel adds 350 pt, each auxiliary strip 180 pt), uses Computer Modern,
# boxed axes, no titles and the legend on top, and encodes series families
# by colour and roles by line style.
module StreamingInferenceCairoMakieExt

using CairoMakie.Makie: Figure, Axis, Legend, Theme, with_theme
using CairoMakie.Makie: lines!, hlines!, vlines!, vspan!, hist!, scatterlines!, text!
using CairoMakie.Makie: linkxaxes!, hidexdecorations!, rowgap!, xlims!, ylims!, save
using CairoMakie.Makie.MathTeXEngine: texfont
using CairoMakie.Makie: scatter!, stairs!, Observable, Point2f, record
using CairoMakie.Makie: rowsize!, Auto, LinearTicks, widths
using DataFrames: DataFrame, nrow
using Dates: Dates, DateTime
using MilliHertzQML.StreamingInference:
    FIGURE_SIZE, FIGURE_STROKES, figure_size, FIGURE_COLORS, backup_existing!, write_toml
using MilliHertzQML.StreamingInference: contiguous_runs
using MilliHertzQML.StreamingInference: FIGURE_FONTSIZE, TICK_FONTSIZE, ANNOTATION_FONTSIZE
using MilliHertzQML.StreamingInference: LEGEND_STYLE, ANIMATION_PX_PER_UNIT, decimation
using MilliHertzQML.StreamingInference: log_ticks, check_frame_scale, check_gif_path
using MilliHertzQML.StreamingInference: frame_schedule, scored_at
import MilliHertzQML.StreamingInference:
    figure_theme,
    save_figure,
    top_legend!,
    label_bands!,
    figure_roc,
    figure_threshold_sweep,
    figure_sensitivity,
    figure_score_distribution,
    figure_telemetry_alerts,
    animation_theme,
    save_animation,
    animate_mission_replay

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

function top_legend!(figure::Figure, axis::Axis; nbanks::Integer = 1)
    Legend(figure[0, 1], axis; LEGEND_STYLE..., nbanks = nbanks)
    rowgap!(figure.layout, 10)
    return nothing
end

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
    compact(x; digits = 2) -> String

`x` with `digits` decimals, or as an integer when it is one.
"""
function compact(x::Real; digits::Integer = 2)
    return isfinite(x) && x == round(x) ? string(round(Int, x)) :
           string(round(Float64(x); digits = digits))
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

end # module
