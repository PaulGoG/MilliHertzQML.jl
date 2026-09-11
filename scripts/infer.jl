# scripts/infer.jl — dispatcher of the inference stage: command line and
# diagnostic figures. The stage itself is `evaluate_classifier` in the
# package; the figures come from the CairoMakie extension.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse
using CairoMakie: CairoMakie

function parse_commandline()
    s = ArgParseSettings(
        description = "Run inference with a trained MilliHertzQML classifier",
    )
    @add_arg_table s begin
        "config"
        help = "Path to the configuration file"
        required = false
        default = joinpath(project_root(), "config.toml")
        "--run-id"
        help = "Run ID of the training run (locates the model unless --model is given) and of the outputs"
        default = ""
        "--model"
        help = "Path to the trained model (.jld2)"
        default = nothing
        "--features"
        help = "Path to the feature CSV"
        default = nothing
        "--labels"
        help = "Path to the label CSV; an empty string (or a missing file) selects blind inference"
        default = nothing
        "--block"
        help = "Rows to evaluate when the features are the training table: all, validation, or test"
        default = nothing
    end
    return parse_args(s)
end

"""
    plot_diagnostics(result, plot_dir)

Diagnostic figures of an inference result in `plot_dir`, each as PDF and
PNG with a provenance sidecar: the mission trace of the classifier output
with the threshold (`mission_trace`) and the score distribution
(`probability_distribution`); with labels also the ROC curve
(`roc_curve`), the event-level operating characteristic with the applied
threshold (`threshold_sweep`), and the detection sensitivity versus
matched-filter SNR (`detection_sensitivity_snr`).
"""
function plot_diagnostics(result::NamedTuple, plot_dir::AbstractString)
    run_id = result.run_id
    save_figure(
        figure_mission_trace(
            result.days,
            result.probabilities,
            result.threshold;
            labels = result.labels,
        ),
        joinpath(plot_dir, "mission_trace");
        run_id = run_id,
    )
    save_figure(
        figure_score_distribution(
            result.probabilities,
            result.threshold;
            labels = result.labels,
        ),
        joinpath(plot_dir, "probability_distribution");
        run_id = run_id,
    )
    if result.labels !== nothing
        save_figure(
            figure_roc(result.roc.fpr, result.roc.tpr, result.auc),
            joinpath(plot_dir, "roc_curve");
            run_id = run_id,
        )
        info = result.threshold_info
        save_figure(
            figure_threshold_sweep(
                result.sweep,
                result.threshold;
                target_far_per_30d = get(info, "criterion", "") == "far" ?
                                     info["target_far_per_30d"] : nothing,
                operating_point = (
                    n_detected = result.metrics["n_detected"],
                    n_events = result.metrics["n_events"],
                    false_alarms_per_30d = result.metrics["false_alarms_per_30d"],
                ),
            ),
            joinpath(plot_dir, "threshold_sweep");
            run_id = run_id,
        )
        sensitivity = figure_sensitivity(result.snrs, result.labels, result.decisions)
        sensitivity === nothing || save_figure(
            sensitivity,
            joinpath(plot_dir, "detection_sensitivity_snr");
            run_id = run_id,
        )
    end
    return plot_dir
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    result = evaluate_classifier(
        config;
        run_id = args["run-id"],
        model = args["model"],
        features = args["features"],
        labels = args["labels"],
        block = args["block"],
    )
    println("Generating diagnostics...")
    plot_diagnostics(result, result.plot_dir)
    println("Diagnostics saved to $(result.plot_dir); results in $(result.results_dir)")
    report_timing()
    return nothing
end

main()
