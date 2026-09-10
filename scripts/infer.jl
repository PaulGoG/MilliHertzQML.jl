# scripts/infer.jl — dispatcher of the inference stage: command line and
# diagnostic figures. The stage itself is `evaluate_classifier` in the
# package.

ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "common.jl"))

using ArgParse, Plots

Plots.default(
    dpi = 600,
    frame = :box,
    fontfamily = "Computer Modern",
    grid = true,
    gridalpha = 0.2,
    minorgrid = false,
    margin = 5Plots.mm,
)

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

Diagnostic figures of an inference result in `plot_dir`: the mission
trace of the classifier output with the threshold
(`mission_trace_days.png`) and the score distribution
(`probability_distribution.png`); with labels also the ROC curve
(`roc_curve.png`) and the detection sensitivity versus matched-filter SNR
(`detection_sensitivity_snr.png`).
"""
function plot_diagnostics(result::NamedTuple, plot_dir::AbstractString)
    probs = result.probabilities
    days = result.days
    threshold = result.threshold
    y_true = result.labels
    has_labels = y_true !== nothing
    n = length(probs)
    idx = 1:max(1, div(n, 5000)):n

    p1 = plot(
        days[idx],
        probs[idx],
        xlabel = "Mission time [days]",
        ylabel = "MBHB probability",
        lw = 1.0,
        color = :darkred,
        label = "Classifier output",
        alpha = 0.8,
    )
    if has_labels
        plot!(
            p1,
            days[idx],
            y_true[idx] .* 0.5,
            st = :step,
            color = :blue,
            alpha = 0.2,
            label = "Labeled span",
            fill = (0, 0.2, :blue),
        )
    end
    hline!(
        p1,
        [threshold],
        color = :black,
        ls = :dash,
        label = "Threshold ($(round(threshold, digits = 2)))",
        lw = 1.5,
    )
    savefig(joinpath(plot_dir, "mission_trace_days.png"))

    if has_labels
        p_roc = plot(
            result.roc.fpr,
            result.roc.tpr,
            xlabel = "False-positive rate",
            ylabel = "True-positive rate",
            lw = 2,
            color = :purple,
            label = "VQC (AUC = $(round(result.auc, digits = 3)))",
        )
        plot!(p_roc, [0, 1], [0, 1], color = :black, ls = :dash, label = "Random")
        savefig(joinpath(plot_dir, "roc_curve.png"))

        snrs = result.snrs
        decisions = result.decisions
        snr_pos = snrs[y_true .== 1]
        if !isempty(snr_pos)
            edges = range(minimum(snr_pos), maximum(snr_pos) + 1e-6; length = 9)
            centers = Float64[]
            rates = Float64[]
            for i in 1:(length(edges)-1)
                mask = (snrs .>= edges[i]) .& (snrs .< edges[i+1]) .& (y_true .== 1)
                if any(mask)
                    push!(centers, (edges[i] + edges[i+1]) / 2)
                    push!(rates, count(decisions[mask] .== 1) / count(mask))
                end
            end
            plot(
                centers,
                rates,
                xlabel = "Matched-filter SNR",
                ylabel = "True-positive rate",
                marker = :circle,
                lw = 2,
                color = :green,
                legend = false,
                ylims = (0, 1.05),
            )
            savefig(joinpath(plot_dir, "detection_sensitivity_snr.png"))
        end

        p3 = histogram(
            probs[y_true .== 0],
            bins = 50,
            label = "Noise windows",
            alpha = 0.5,
            color = :gray,
        )
        histogram!(
            p3,
            probs[y_true .== 1],
            bins = 50,
            label = "MBHB windows",
            alpha = 0.5,
            color = :red,
            xlabel = "Probability score",
            ylabel = "Count",
        )
    else
        p3 = histogram(
            probs,
            bins = 50,
            label = "All windows",
            alpha = 0.6,
            color = :gray,
            xlabel = "Probability score",
            ylabel = "Count",
        )
    end
    vline!(p3, [threshold], color = :black, ls = :dash, label = "Threshold", lw = 1.5)
    savefig(joinpath(plot_dir, "probability_distribution.png"))
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
