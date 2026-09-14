# scripts/animate.jl — dispatcher of the animations: the training history of
# a training run and the streaming replay of a telemetry results directory.
# Both are written as GIF with a provenance sidecar into the run's plot
# directory; the animations themselves come from the CairoMakie extension.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse
using CairoMakie: CairoMakie
using DataFrames: DataFrame, nrow
using Dates: Dates, DateTime, Millisecond

function parse_commandline()
    s = ArgParseSettings(
        description = "Animate a MilliHertzQML training run or telemetry replay",
    )
    @add_arg_table s begin
        "config"
        help = "Path to the configuration file"
        required = false
        default = joinpath(project_root(), "config.toml")
        "--run-id"
        help = "Run identifier: the training run <models>/run_<id> and the telemetry results <results>/run_<id> are animated when they exist"
        default = ""
        "--training-run"
        help = "Training run directory holding history.csv (overrides --run-id)"
        default = nothing
        "--telemetry-run"
        help = "Telemetry results directory holding telemetry_windows.csv (overrides --run-id)"
        default = nothing
        "--out-dir"
        help = "Directory of the animations (default: the plot directory of the run)"
        default = nothing
    end
    return parse_args(s)
end

"""
    run_identifier(dir) -> String

Run identifier of a run directory: its name without the `run_` prefix.
"""
run_identifier(dir::AbstractString) =
    replace(basename(rstrip(abspath(dir), '/')), r"^run_" => "")

"""
    training_animation(run_dir, out_dir, run_id) -> String

Animated training history of the run directory `run_dir` (which holds
`history.csv`) as `<out_dir>/training_history.gif` with a provenance
sidecar. Returns the written path.
"""
function training_animation(
    run_dir::AbstractString,
    out_dir::AbstractString,
    run_id::AbstractString,
)
    table = MilliHertzQML.CSV.read(joinpath(run_dir, "history.csv"), DataFrame)
    history = (
        epochs = table.epochs,
        train_loss = table.train_loss,
        val_loss = table.val_loss,
        val_acc = table.val_acc,
    )
    return save_animation(joinpath(out_dir, "training_history"); run_id = run_id) do path
        animate_training_history(history, path)
    end
end

"""
    replay_geometry(windows) -> NamedTuple

Mission epoch and sample interval [s] of a scored-window table: the
interval from the extent of the first window, the epoch from the time of
its first row. The table describes its own geometry, so an animation of a
replay needs neither the run directory nor the producer.
"""
function replay_geometry(windows::DataFrame)
    nrow(windows) >= 1 || throw(ArgumentError("the windows table is empty."))
    row = first(eachrow(windows))
    row.row_end > row.row_start ||
        throw(ArgumentError("window $(row.window) spans a single row."))
    interval =
        Dates.value(row.content_end - row.content_start) / 1000 /
        (row.row_end - row.row_start)
    epoch =
        row.content_start - Millisecond(round(Int, 1000 * interval * (row.row_start - 1)))
    return (epoch = epoch, sample_interval = interval)
end

"""
    replay_spans(results_dir, geometry) -> Union{Nothing, Vector{Tuple{DateTime,DateTime}}}

Labeled spans of a replay, in mission time: the payload rows
`label_start_index` and `label_end_index` of the event table recorded in
`config_telemetry.toml`, or, when that table is out of reach, the hour
before every merger of `alert_latency.csv`. `nothing` when the results
directory holds neither.
"""
function replay_spans(results_dir::AbstractString, geometry::NamedTuple)
    settings = section(
        MilliHertzQML.TOML.parsefile(joinpath(results_dir, "config_telemetry.toml")),
        "telemetry",
    )
    events_path = cfgget(settings, "events_csv", ""; type = String)
    row_time(index) =
        geometry.epoch +
        Millisecond(round(Int, 1000 * geometry.sample_interval * (index - 1)))
    if !isempty(events_path) && isfile(resolvepath(events_path))
        events = MilliHertzQML.CSV.read(resolvepath(events_path), DataFrame)
        if all(c -> c in names(events), ("label_start_index", "label_end_index"))
            return [
                (row_time(Int(r.label_start_index)), row_time(Int(r.label_end_index))) for
                r in eachrow(events)
            ]
        end
    end
    latency_path = joinpath(results_dir, "alert_latency.csv")
    isfile(latency_path) || return nothing
    latencies = MilliHertzQML.CSV.read(latency_path, DataFrame)
    nrow(latencies) == 0 && return nothing
    # Without the event table the span is unknown; the hour before the
    # merger is the convention of the alert figure.
    mergers = [t isa DateTime ? t : DateTime(t) for t in latencies.t_merger]
    return [(t - Dates.Hour(1), t) for t in mergers]
end

"""
    replay_animation(results_dir, out_dir, run_id) -> String

Animated streaming replay of the telemetry results directory `results_dir`
(which holds `telemetry_windows.csv` and `config_telemetry.toml`) as
`<out_dir>/mission_replay.gif` with a provenance sidecar. Returns the
written path.
"""
function replay_animation(
    results_dir::AbstractString,
    out_dir::AbstractString,
    run_id::AbstractString,
)
    windows =
        MilliHertzQML.CSV.read(joinpath(results_dir, "telemetry_windows.csv"), DataFrame)
    settings = section(
        MilliHertzQML.TOML.parsefile(joinpath(results_dir, "config_telemetry.toml")),
        "telemetry",
    )
    threshold = cfgget(settings, "threshold", NaN; type = Float64)
    isfinite(threshold) ||
        throw(ArgumentError("config_telemetry.toml of $results_dir records no threshold."))
    geometry = replay_geometry(windows)
    spans = replay_spans(results_dir, geometry)
    return save_animation(joinpath(out_dir, "mission_replay"); run_id = run_id) do path
        animate_mission_replay(
            windows,
            threshold,
            path;
            epoch = geometry.epoch,
            label_spans = spans,
        )
    end
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    paths = pipeline_paths(config)
    run_id = args["run-id"]
    training_dir = override(
        args["training-run"],
        isempty(run_id) ? "" : joinpath(paths.models, "run_$run_id"),
    )
    telemetry_dir = override(
        args["telemetry-run"],
        isempty(run_id) ? "" : joinpath(paths.results, "run_$run_id"),
    )
    written = String[]
    for (dir, file, animate) in (
        (training_dir, "history.csv", training_animation),
        (telemetry_dir, "telemetry_windows.csv", replay_animation),
    )
        isempty(dir) && continue
        dir = resolvepath(dir)
        isfile(joinpath(dir, file)) || continue
        id = isempty(run_id) ? run_identifier(dir) : run_id
        out_dir = override(args["out-dir"], joinpath(paths.plots, "run_$id"))
        mkpath(out_dir)
        path = animate(dir, out_dir, id)
        @info "animation written" path bytes = filesize(path)
        push!(written, path)
    end
    isempty(written) && throw(
        ArgumentError(
            "no animation to draw: pass --run-id, --training-run, or --telemetry-run " *
            "for a directory holding history.csv or telemetry_windows.csv.",
        ),
    )
    println("Animations: $(join(written, ", "))")
    report_timing()
    return nothing
end

main()
