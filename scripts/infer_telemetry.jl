# scripts/infer_telemetry.jl — dispatcher of the telemetry replay stage:
# command line and the alert figure. The stage itself is
# `replay_classifier` in the package: it replays (or follows live) a
# DeepSpaceTelemetry run directory with a trained classifier and writes the
# scored-window table, the alert-latency table and the snapshot.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse
using CairoMakie: CairoMakie
using DeepSpaceTelemetryReader: DeepSpaceTelemetryReader
using DataFrames: nrow
using StreamingInference: credited_span_label

function parse_commandline()
    s = ArgParseSettings(
        description = "Score a DeepSpaceTelemetry run directory as it arrives",
    )
    @add_arg_table s begin
        "config"
        help = "Path to the configuration file"
        required = false
        default = DEFAULT_CONFIG
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
        help = "Event table CSV (merger_time_s, label spans and signal onsets) for the alert-latency table"
        default = nothing
        "--tdi-file"
        help = "TDI record (HDF5) the payload of the mission was exported from; serves the channels of a model trained on several when the run does not carry them as payload columns"
        default = nothing
        "--tdi-group"
        help = "TDI group of --tdi-file (default: the component groups of [preprocessing], tdi_components or tdi_group)"
        default = nothing
        "--link-volume-channels"
        help = "Number of channels the production rate of the mission was scaled for (recorded in the snapshot)"
        arg_type = Int
        default = 1
        "--tdi-first-row"
        help = "Row of --tdi-file that the first payload row of the mission holds (a mission over a later stretch of the record)"
        arg_type = Int
        default = 1
        "--live"
        help = "Follow the arrival feed until the run ends instead of replaying it"
        action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    result = replay_classifier(
        config;
        model = args["model"],
        run_id = isempty(args["run-id"]) ? new_run_id() : args["run-id"],
        run_dir = args["run-dir"],
        events = args["events"],
        tdi_file = args["tdi-file"],
        tdi_group = args["tdi-group"],
        tdi_first_row = args["tdi-first-row"],
        live = args["live"],
        link_volume_channels = args["link-volume-channels"],
    )
    if nrow(result.windows) > 0
        save_figure(
            figure_telemetry_alerts(
                result.windows,
                result.threshold;
                epoch = result.geometry.start_sim_time,
                label_spans = result.spans,
                latencies = result.latencies,
                span_label = credited_span_label(result.crediting),
                event_label = "Merger",
                CLASSIFIER_SCORE_AXIS...,
            ),
            joinpath(result.plot_dir, "telemetry_alerts");
            run_id = result.run_id,
        )
    end
    println("Telemetry results in $(result.results_dir); figures in $(result.plot_dir)")
    report_timing()
    return nothing
end

with_pipeline_root(main, config_root(config_argument(DEFAULT_CONFIG)))
