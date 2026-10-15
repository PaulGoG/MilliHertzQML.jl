# scripts/calibrate.jl — dispatcher of the calibration stage: command line.
# The stage itself is `calibrate_classifier` in the package.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        description = "Calibrate a trained MilliHertzQML classifier on a new record's own noise",
    )
    @add_arg_table s begin
        "config"
        help = "Path to the configuration file"
        required = false
        default = DEFAULT_CONFIG
        "--run-id"
        help = "Run identifier of the calibration run, the directory of its model and threshold"
        default = ""
        "--model"
        help = "Path to the trained model (.jld2) of the training run"
        default = nothing
        "--features"
        help = "Feature table of the new record; default [inference] features"
        default = nothing
        "--labels"
        help = "Window labels of the new record; absent for a background record"
        default = nothing
        "--scaler"
        help = "Scaler rule: match, refit or keep; default [calibration] scaler"
        default = nothing
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    result = calibrate_classifier(
        config;
        run_id = args["run-id"],
        model = args["model"],
        features = args["features"],
        labels = args["labels"],
        scaler = args["scaler"],
    )
    println("Calibration run written to $(result.run_dir)")
    println(
        "Threshold $(result.threshold) (parent $(result.threshold_info["parent_threshold"]))",
    )
    println(
        "Evaluation block: rows $(first(result.blocks.evaluation))–" *
        "$(last(result.blocks.evaluation)); score it with scripts/infer.jl " *
        "--run-id $(result.run_id) --block evaluation",
    )
    report_timing()
    return nothing
end

with_pipeline_root(main, config_root(config_argument()))
