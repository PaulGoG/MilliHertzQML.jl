# ext/MilliHertzQMLCairoMakieExt.jl — figures of the classifier and its
# studies, loaded together with CairoMakie: training history and its
# animation, loss survival, seed spread, gap study and model grid. Theme,
# legend, exports and the shared tick and frame helpers come from the
# figures of the domain-general layer.
module MilliHertzQMLCairoMakieExt

using CairoMakie.Makie: Figure, Axis, Legend, with_theme
using CairoMakie.Makie: lines!, hlines!, vlines!, scatterlines!, text!, scatter!, hspan!
using CairoMakie.Makie: linkxaxes!, hidexdecorations!, rowgap!, xlims!, ylims!
using CairoMakie.Makie: linkyaxes!, hideydecorations!, colgap!, Observable, @lift, record
using Statistics: median
using MilliHertzQML.StreamingInference: FIGURE_STROKES, figure_size, FIGURE_COLORS
using MilliHertzQML.StreamingInference: ANNOTATION_FONTSIZE, LEGEND_STYLE
using MilliHertzQML.StreamingInference: ANIMATION_PX_PER_UNIT, dense_log_ticks
using MilliHertzQML.StreamingInference: check_frame_scale, check_gif_path, frame_schedule
using MilliHertzQML.StreamingInference: figure_theme, animation_theme, top_legend!
import MilliHertzQML:
    figure_training_history,
    animate_training_history,
    figure_loss_survival,
    figure_seed_spread,
    figure_gap_study,
    figure_grid_seeds

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
