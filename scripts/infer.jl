ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "common.jl"))

using MilliHertzQML, CSV, DataFrames, Plots, ArgParse, TOML, Dates, EvalMetrics

function parse_commandline()
    s = ArgParseSettings(description = "Run Inference with the MilliHertzQML VQC")
    @add_arg_table s begin
        "--config"
            help = "Path to the configuration file"
            default = joinpath(dirname(@__DIR__), "config.toml")
        "--features"
            help = "Path to the inference features CSV"
            default = nothing
        "--labels"
            help = "Path to the inference labels CSV. Pass an empty string (or omit the file) for blind inference; blind mode requires a threshold persisted by a previous labeled run."
            default = nothing
        "--model"
            help = "Path to the trained model (.jld2)"
            default = nothing
        "--run-id"
            help = "The Run ID used during training (e.g., 'a1b2c3d4'). Required to locate the model unless --model is given."
            default = ""
        "--step-size"
            help = "Step size used during pre-processing (to calculate real mission time)"
            arg_type = Int
            default = nothing
        "--sample-rate"
            help = "Sample rate of original telemetry in Hz"
            arg_type = Float64
            default = nothing
        "--target-fpr"
            help = "Target maximum False Positive Rate (e.g., 0.05 for 5% max false alarms). If 0.0, uses Youden's J index."
            arg_type = Float64
            default = nothing
    end
    return parse_args(s)
end

"""
    select_threshold(y_true, probs, target_fpr)

Select the decision threshold from the ROC curve: the highest-TPR threshold
satisfying `fpr <= target_fpr`, or the Youden's J maximizer when
`target_fpr == 0`. Returns `(threshold, index, tpr, fpr, criterion)`.
"""
function select_threshold(y_true, probs, target_fpr)
    thresholds_arr = thresholds(probs)
    tpr_arr = true_positive_rate(y_true, probs, thresholds_arr)
    fpr_arr = false_positive_rate(y_true, probs, thresholds_arr)
    if target_fpr > 0.0
        valid_idx = findall(fpr_arr .<= target_fpr)
        opt_idx = isempty(valid_idx) ? 1 : valid_idx[argmax(tpr_arr[valid_idx])]
        criterion = "target_fpr"
    else
        opt_idx = argmax(tpr_arr .- fpr_arr)
        criterion = "youden_j"
    end
    return thresholds_arr[opt_idx], opt_idx, tpr_arr, fpr_arr, criterion
end

function main()
    parsed_args = parse_commandline()

    # 1. Load and validate TOML configuration
    config_file = load_config(parsed_args["config"])
    infer_cfg = get(config_file, "inference", Dict{String, Any}())

    # 2. Harmonize CLI with TOML defaults (CLI takes precedence)
    run_id = parsed_args["run-id"]
    features_path = override(parsed_args["features"],
        cfgget(infer_cfg, "features", "data/inputs/inference_features.csv"; type = String))
    labels_path = override(parsed_args["labels"],
        cfgget(infer_cfg, "labels", "data/inputs/inference_labels.csv"; type = String))
    step_size = override(parsed_args["step-size"],
        cfgget(infer_cfg, "step_size", 100; type = Int, min = 1))
    sample_rate = override(parsed_args["sample-rate"],
        cfgget(infer_cfg, "sample_rate", 0.2; type = Float64, min = 1e-6))
    target_fpr = override(parsed_args["target-fpr"],
        cfgget(infer_cfg, "target_fpr", 0.05; type = Float64, min = 0.0, max = 1.0))

    features_path = resolvepath(features_path)
    has_labels = !isempty(labels_path) && isfile(resolvepath(labels_path))
    has_labels && (labels_path = resolvepath(labels_path))

    (parsed_args["model"] !== nothing || !isempty(run_id)) || throw(ArgumentError(
        "no model specified: provide --model <path> or --run-id <id> of a training run."))
    model_path = parsed_args["model"] !== nothing ? resolvepath(parsed_args["model"]) :
        joinpath(PROJECT_ROOT, "models", "run_$run_id", "gw_model.jld2")

    if isempty(run_id)
        run_id = "standalone_" * string(hash(model_path))[1:6]
    end

    plot_dir = joinpath(PROJECT_ROOT, "data", "outputs", "plots", "run_$run_id")
    res_dir = joinpath(PROJECT_ROOT, "data", "outputs", "results", "run_$run_id")
    mkpath(plot_dir)
    mkpath(res_dir)

    # Configuration snapshot for provenance
    final_config = Dict(
        "inference" => Dict(
            "features" => rootrelative(features_path),
            "labels" => has_labels ? rootrelative(labels_path) : "",
            "model" => rootrelative(model_path),
            "step_size" => step_size,
            "sample_rate" => sample_rate,
            "target_fpr" => target_fpr,
            "run_id" => run_id,
            "blind" => !has_labels
        ),
        "hardware" => hardware_fingerprint(),
    )
    open(joinpath(res_dir, "config_infer.toml"), "w") do io
        TOML.print(io, final_config)
    end

    # Plotting setup (High DPI / Boxed)
    default(dpi=600, frame=:box, fontfamily="Computer Modern", grid=true, gridalpha=0.2, minorgrid=false, margin=5Plots.mm)

    # 3. Load model and data
    model, model_meta = load_model(model_path)
    println("\n[INFER] Model loaded from $model_path")

    local y_true, snrs
    if has_labels
        X, y_true, df_meta = load_data(features_path, labels_path)
        snrs = "SNR" in names(df_meta) ? df_meta[:, :SNR] : zeros(Float32, size(X, 1))
    else
        println("[INFER] Blind mode: no labels; using the persisted decision threshold.")
        X = load_features(features_path)
    end

    # 4. Forward pass over all windows
    num_samples = size(X, 1)
    println("Analyzing $num_samples samples...")
    probs = Float32[]
    for i in 1:num_samples
        push!(probs, predict_probability(model, @view(X[i, :])))
        if i % max(1, div(num_samples, 10)) == 0
            println("  Progress: $(round(Int, i/num_samples*100))%")
        end
    end

    # 5. Decision threshold: fit from labels, or load the persisted value
    thresh_file = joinpath(dirname(model_path), "threshold.toml")
    local opt_thresh, opt_idx, tpr_arr, fpr_arr, auc_score
    if has_labels
        roc = roccurve(y_true, probs)
        auc_score = auc_trapezoidal(roc...)
        opt_thresh, opt_idx, tpr_arr, fpr_arr, criterion = select_threshold(y_true, probs, target_fpr)

        open(thresh_file, "w") do io
            TOML.print(io, Dict("threshold" => Dict(
                "value" => Float64(opt_thresh),
                "criterion" => criterion,
                "target_fpr" => target_fpr,
                "auc" => Float64(auc_score),
                "fitted_on" => rootrelative(features_path),
                "fitted_at" => string(Dates.now())
            )))
        end
        println("\n[FINISH] ROC AUC Score: $(round(auc_score, digits=4))")
        println("[FINISH] Threshold ($criterion): $(round(opt_thresh, digits=4)) — persisted to $thresh_file")

        final_acc = sum((probs .>= opt_thresh) .== y_true) / num_samples
        println("[FINISH] Final Inference Accuracy (Opt. Thresh): $(round(final_acc, digits=4))")
    else
        isfile(thresh_file) || error(
            "Blind inference requires a persisted threshold at $thresh_file. " *
            "Run labeled inference for this model first.")
        tcfg = TOML.parsefile(thresh_file)["threshold"]
        opt_thresh = Float32(tcfg["value"])
        println("[INFER] Loaded threshold $(round(opt_thresh, digits=4)) " *
                "(criterion: $(tcfg["criterion"]), fitted at $(tcfg["fitted_at"]))")
    end

    # 6. Persist per-window scores and decisions
    results = DataFrame(Probability = probs, Detection = Int.(probs .>= opt_thresh))
    if has_labels
        results.Label = y_true
        results.SNR = snrs
    end
    CSV.write(joinpath(res_dir, "inference_probabilities.csv"), results)

    # 7. Diagnostic plots
    println("[PLOT] Generating diagnostics...")

    # Mission trace (always available)
    step_duration_secs = step_size / sample_rate
    days = (1:num_samples) .* step_duration_secs ./ (24 * 3600)
    mask_zoom = days .<= 30.0
    ds = max(1, Int(floor(sum(mask_zoom) / 1000)))
    idx_ds = (1:ds:sum(mask_zoom))

    p1 = plot(days[mask_zoom][idx_ds], probs[mask_zoom][idx_ds], title="LISA Mission Detection Trace (30 Days)",
              xlabel="Mission Time [Days]", ylabel="MBHB Probability",
              lw=1.0, color=:darkred, label="QNN Output", alpha=0.8)
    if has_labels
        plot!(p1, days[mask_zoom][idx_ds], y_true[mask_zoom][idx_ds] .* 0.5, st=:step, color=:blue, alpha=0.2, label="True Event Window", fill=(0, 0.2, :blue))
    end
    hline!(p1, [opt_thresh], color=:black, ls=:dash, label="Threshold ($(round(opt_thresh, digits=2)))", lw=1.5)
    savefig(joinpath(plot_dir, "mission_trace_days.png"))
    println("  - Mission trace saved.")

    if has_labels
        # ROC curve
        p_roc = plot(fpr_arr, tpr_arr, title="Receiver Operating Characteristic (ROC)",
                     xlabel="False Positive Rate (FPR)", ylabel="True Positive Rate (TPR)",
                     lw=2, color=:purple, label="VQC (AUC = $(round(auc_score, digits=3)))")
        plot!(p_roc, [0, 1], [0, 1], color=:black, ls=:dash, label="Random Guess")
        scatter!(p_roc, [fpr_arr[opt_idx]], [tpr_arr[opt_idx]], color=:red, markersize=5, label="Optimal Cutoff ($(round(opt_thresh, digits=2)))")
        savefig(joinpath(plot_dir, "roc_curve.png"))
        println("  - ROC curve saved.")

        # Detection sensitivity vs SNR
        snr_bins = 2.0:0.5:8.0
        bin_centers = Float64[]
        bin_accs = Float64[]
        for i in 1:(length(snr_bins)-1)
            low, high = snr_bins[i], snr_bins[i+1]
            mask = (snrs .>= low) .& (snrs .< high) .& (y_true .== 1)
            if sum(mask) > 0
                push!(bin_centers, (low + high)/2)
                push!(bin_accs, sum(probs[mask] .>= opt_thresh) / sum(mask))
            end
        end
        p2 = plot(bin_centers, bin_accs, title="Detection Efficiency vs Signal Strength",
                  xlabel="Signal-to-Noise Ratio (SNR)", ylabel="True Positive Rate (TPR)",
                  marker=:circle, lw=2, color=:green, legend=false, ylims=(0, 1.05))
        savefig(joinpath(plot_dir, "detection_sensitivity_snr.png"))
        println("  - Sensitivity plot saved.")

        # Class-conditional score distributions
        p3 = histogram(probs[y_true .== 0], bins=50, label="Pure Noise/Forest", alpha=0.5, color=:gray)
        histogram!(p3, probs[y_true .== 1], bins=50, label="MBHB Events", alpha=0.5, color=:red,
                   title="Score Distribution: Separation Power", xlabel="Probability Score", ylabel="Count")
        vline!(p3, [opt_thresh], color=:black, ls=:dash, label="Opt. Threshold", lw=1.5)
        savefig(joinpath(plot_dir, "probability_distribution.png"))
        println("  - Probability distribution saved.")
    else
        # Blind mode: single score distribution with the applied threshold
        p3 = histogram(probs, bins=50, label="All windows", alpha=0.6, color=:gray,
                       title="Score Distribution (Blind)", xlabel="Probability Score", ylabel="Count")
        vline!(p3, [opt_thresh], color=:black, ls=:dash, label="Applied Threshold", lw=1.5)
        savefig(joinpath(plot_dir, "probability_distribution.png"))
        println("  - Probability distribution saved.")
    end

    println("[SUCCESS] Diagnostic plots saved to '$plot_dir'")
end

main()
