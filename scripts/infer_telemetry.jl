# scripts/infer_telemetry.jl — dispatcher of the telemetry coupling: replay
# (or follow live) a DeepSpaceTelemetry run directory with a trained
# classifier, write the scored-window table and the alert-latency table,
# and draw the alert figure. The consumer only reads the run directory.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse
using CairoMakie: CairoMakie
using DeepSpaceTelemetry: DeepSpaceTelemetry
using DataFrames: DataFrame, nrow
using Dates: Dates, DateTime, Millisecond

function parse_commandline()
    s = ArgParseSettings(
        description = "Score a DeepSpaceTelemetry run directory as it arrives",
    )
    @add_arg_table s begin
        "config"
        help = "Path to the configuration file"
        required = false
        default = joinpath(project_root(), "config.toml")
        "--run-dir"
        help = "DeepSpaceTelemetry run directory (overrides [telemetry] run_dir)"
        default = nothing
        "--model"
        help = "Path to the trained model (.jld2) whose directory holds threshold.toml and config.toml"
        required = true
        "--run-id"
        help = "Run identifier of the outputs (default: generated)"
        default = ""
        "--events"
        help = "Event table CSV (merger_time_s and label spans) for the alert-latency table"
        default = nothing
        "--live"
        help = "Follow the arrival feed until the run ends instead of replaying it"
        action = :store_true
    end
    return parse_args(s)
end

"""
    label_span_times(events, geometry) -> Vector{Tuple{DateTime,DateTime}}

Mission-time intervals of the label spans of an event table (payload rows
`label_start_index`/`label_end_index`), or one window around each merger
when the table carries no spans.
"""
function label_span_times(events::DataFrame, geometry::RunGeometry)
    spans = Tuple{DateTime,DateTime}[]
    has_span = "label_start_index" in names(events) && "label_end_index" in names(events)
    times = event_merger_times(events)
    for (i, row) in enumerate(eachrow(events))
        if has_span
            push!(
                spans,
                (
                    row_time(geometry, Int(row.label_start_index)),
                    row_time(geometry, Int(row.label_end_index)),
                ),
            )
        else
            t = geometry.start_sim_time + Millisecond(round(Int, 1000 * times[i]))
            push!(spans, (t - Dates.Hour(1), t))
        end
    end
    return spans
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    settings = telemetry_settings(config)
    paths = pipeline_paths(config)
    run_dir = override(args["run-dir"], settings.run_dir)
    isempty(run_dir) &&
        throw(ArgumentError("no run directory: pass --run-dir or set [telemetry] run_dir."))
    run_dir = resolvepath(run_dir)
    run_id = isempty(args["run-id"]) ? new_run_id() : args["run-id"]
    results_dir = joinpath(paths.results, "run_$run_id")
    plot_dir = joinpath(paths.plots, "run_$run_id")
    startswith(abspath(results_dir), abspath(run_dir)) &&
        throw(ArgumentError("the output directory must lie outside the run directory."))
    mkpath(results_dir)
    mkpath(plot_dir)

    detector = detector_from_run(
        resolvepath(args["model"]);
        context_windows = settings.context_windows,
    )
    run = open_telemetry_run(run_dir; producer_compat = settings.producer_compat)
    geometry = run_geometry(run)
    @info "telemetry run opened" run_dir state = run_state(run) sample_rate =
        geometry.sample_rate points_per_batch = geometry.points_per_batch producer =
        geometry.package_version
    live = args["live"] || settings.mode == "live"
    n_alarms = Ref(0)
    on_window =
        record -> begin
            record.decision == 1 && (n_alarms[] += 1)
            record.window % 100 == 0 && @info "scored" window = record.window score =
                round(record.score; digits = 3) complete_at = record.complete_at alarms =
                n_alarms[]
        end
    windows = begin
        if live
            follow_run(
                run,
                detector;
                poll_interval_sec = settings.poll_interval_sec,
                min_coverage = settings.min_coverage,
                tdi_gap_dilation_sec = settings.tdi_gap_dilation_sec,
                on_window = on_window,
            )
        else
            replay_run(
                run,
                detector;
                min_coverage = settings.min_coverage,
                tdi_gap_dilation_sec = settings.tdi_gap_dilation_sec,
                on_window = on_window,
            )
        end
    end
    write_csv(joinpath(results_dir, "telemetry_windows.csv"), windows)
    @info "windows scored" n_windows = nrow(windows) alarms = n_alarms[]

    events_path = override(args["events"], settings.events_csv)
    latencies = nothing
    spans = nothing
    if !isempty(events_path)
        events = MilliHertzQML.CSV.read(resolvepath(events_path), DataFrame)
        latencies = alert_latency_table(
            windows,
            events,
            geometry;
            processing_latency_hours = settings.processing_latency_hours,
        )
        write_csv(joinpath(results_dir, "alert_latency.csv"), latencies)
        spans = label_span_times(events, geometry)
        for row in eachrow(latencies)
            if row.detected
                @info "event detected" label = row.label latency_data_h =
                    round(row.latency_data_h; digits = 2) latency_total_h =
                    round(row.latency_total_h; digits = 2)
            else
                @info "event missed" label = row.label
            end
        end
    end
    write_toml(
        joinpath(results_dir, "config_telemetry.toml"),
        Dict{String,Any}(
            "telemetry" => Dict{String,Any}(
                "run_dir" => run_dir,
                "producer_version" => geometry.package_version,
                "run_state" => String(run_state(run)),
                "mode" => live ? "live" : "replay",
                "model" => rootrelative(resolvepath(args["model"])),
                "threshold" => Float64(detector.threshold),
                "min_coverage" => settings.min_coverage,
                "tdi_gap_dilation_sec" => settings.tdi_gap_dilation_sec,
                "context_windows" => settings.context_windows,
                "processing_latency_hours" => settings.processing_latency_hours,
                "events_csv" =>
                    isempty(events_path) ? "" : rootrelative(resolvepath(events_path)),
                "run_id" => run_id,
                "n_windows" => nrow(windows),
                "n_alarms" => n_alarms[],
            ),
        ),
    )
    if nrow(windows) > 0
        save_figure(
            figure_telemetry_alerts(
                windows,
                detector.threshold;
                epoch = geometry.start_sim_time,
                label_spans = spans,
                latencies = latencies,
            ),
            joinpath(plot_dir, "telemetry_alerts");
            run_id = run_id,
        )
    end
    println("Telemetry results in $results_dir; figures in $plot_dir")
    report_timing()
    return nothing
end

main()
