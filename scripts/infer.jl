ENV["GKSwstype"] = "100"
using Pkg
Pkg.activate("QuantumGW", io=devnull)
push!(LOAD_PATH, "QuantumGW/src")

using Serialization, QuantumGW, CSV, DataFrames, Plots, Statistics, ArgParse, TOML, EvalMetrics

function parse_commandline()
    s = ArgParseSettings(description = "Run Inference with the QuantumGW VQC")
    @add_arg_table s begin
        "--config"
            help = "Path to the configuration file"
            default = "QuantumGW/config.toml"
        "--features"
            help = "Path to the inference features CSV"
            default = nothing
        "--labels"
            help = "Path to the inference labels CSV (used for validation)"
            default = nothing
        "--model"
            help = "Path to the trained model (.jls)"
            default = nothing
        "--run-id"
            help = "The Run ID used during training (e.g., 'a1b2c3d4'). Required to locate the correct model if not explicitly provided."
            default = ""
        "--use-real-data"
            help = "Automatically use the pre-processed real telemetry data (telemetry_blind_features.csv)"
            action = :store_true
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
            default = 0.05
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    # 1. Load TOML
    config_file = isfile(parsed_args["config"]) ? TOML.parsefile(parsed_args["config"]) : Dict{String, Any}()
    infer_cfg = get(config_file, "inference", Dict{String, Any}())

    # 2. Harmonize CLI with TOML Defaults
    run_id = parsed_args["run-id"]
    features_path = parsed_args["features"] !== nothing ? parsed_args["features"] : get(infer_cfg, "features", "QuantumGW/data/inputs/inference_features.csv")
    labels_path = parsed_args["labels"] !== nothing ? parsed_args["labels"] : get(infer_cfg, "labels", "QuantumGW/data/inputs/inference_labels.csv")
    step_size = parsed_args["step-size"] !== nothing ? parsed_args["step-size"] : get(infer_cfg, "step_size", 100)
    sample_rate = parsed_args["sample-rate"] !== nothing ? parsed_args["sample-rate"] : get(infer_cfg, "sample_rate", 0.2)
    target_fpr = parsed_args["target-fpr"] !== nothing ? parsed_args["target-fpr"] : get(infer_cfg, "target_fpr", 0.05)

    if parsed_args["use-real-data"]
        features_path = "QuantumGW/data/inputs/telemetry_blind_features.csv"
    end

    model_path = parsed_args["model"] !== nothing ? parsed_args["model"] : (isempty(run_id) ? "QuantumGW/models/gw_model.jls" : "QuantumGW/models/run_$run_id/gw_model.jls")

    if isempty(run_id)
        # If no run-id provided, infer one from the model path or generate a standalone one
        run_id = "standalone_" * string(hash(model_path))[1:6]
    end

    # Save Configuration Snapshot
    plot_dir = "QuantumGW/data/outputs/plots/run_$run_id"
    res_dir = "QuantumGW/data/outputs/results/run_$run_id"
    mkpath(plot_dir)
    mkpath(res_dir)

    final_config = Dict(
        "inference" => Dict(
            "features" => features_path,
            "labels" => labels_path,
            "model" => model_path,
            "step_size" => step_size,
            "sample_rate" => sample_rate,
            "run_id" => run_id,
            "use_real_data" => parsed_args["use-real-data"]
        )
    )
    open(joinpath(res_dir, "config_infer.toml"), "w") do io
        TOML.print(io, final_config)
    end

    # Plotting setup (High DPI / Boxed)
    default(dpi=600, frame=:box, fontfamily="Computer Modern", grid=true, gridalpha=0.2, minorgrid=false, margin=5Plots.mm)

    # 1. Load Model
    model = deserialize(model_path)
    println("\n[INFER] Model loaded from $model_path")

    # 2. Load Inference Data (X, y, df)
    X, y_true, df_meta = load_data(features_path, labels_path)
    snrs = "SNR" in names(df_meta) ? df_meta[:, :SNR] : zeros(Float32, size(X, 1))

    # Transpose for fast, contiguous memory access during pointwise inference
    X_t = copy(X')
    X_fast = X_t'

    # 3. Analyze Samples
    println("Analyzing $(size(X, 1)) samples...")
    probs = Float32[]
    num_samples = size(X, 1)

    for i in 1:num_samples
        p = predict_probability(model, @view(X_fast[i, :]))
        push!(probs, p)
        if i % max(1, div(num_samples, 10)) == 0
            println("  Progress: $(round(Int, i/num_samples*100))%")
        end
    end

    # 4. Optimal Threshold Calculation (ROC Analysis)
    roc = roccurve(y_true, probs)
    auc_score = auc_trapezoidal(roc...)
    thresholds_arr = thresholds(probs)
    tpr_arr = true_positive_rate(y_true, probs, thresholds_arr)
    fpr_arr = false_positive_rate(y_true, probs, thresholds_arr)

    if target_fpr > 0.0
        # Mission-driven threshold: Find the highest TPR where FPR <= target_fpr
        valid_idx = findall(fpr_arr .<= target_fpr)
        if isempty(valid_idx)
            opt_idx = 1 # Fallback to strictest threshold if impossible
        else
            opt_idx = valid_idx[argmax(tpr_arr[valid_idx])]
        end
        opt_thresh = thresholds_arr[opt_idx]
        println("\n[FINISH] ROC AUC Score: $(round(auc_score, digits=4))")
        println("[FINISH] Calculated Threshold for ≤ $(target_fpr*100)% FPR: $(round(opt_thresh, digits=4))")
    else
        # Youden's J = TPR - FPR. Maximize this to find optimal mathematical threshold.
        j_scores = tpr_arr .- fpr_arr
        opt_idx = argmax(j_scores)
        opt_thresh = thresholds_arr[opt_idx]
        println("\n[FINISH] ROC AUC Score: $(round(auc_score, digits=4))")
        println("[FINISH] Calculated Youden's J Threshold: $(round(opt_thresh, digits=4))")
    end

    # Calculate final accuracy using the OPTIMAL threshold
    y_pred_opt = probs .>= opt_thresh
    final_acc = sum(y_pred_opt .== y_true) / num_samples

    println("[FINISH] Final Inference Accuracy (Opt. Thresh): $(round(final_acc, digits=4))")

    # 5. Save Results
    CSV.write(joinpath(res_dir, "inference_probabilities.csv"), DataFrame(Probability=probs, Label=y_true, SNR=snrs))

    # 6. HUMAN-READABLE PLOTS
    println("[PLOT] Generating human-readable diagnostics...")

    # --- Plot 1: ROC Curve ---
    p_roc = plot(fpr_arr, tpr_arr, title="Receiver Operating Characteristic (ROC)", 
                 xlabel="False Positive Rate (FPR)", ylabel="True Positive Rate (TPR)", 
                 lw=2, color=:purple, label="VQC (AUC = $(round(auc_score, digits=3)))")
    plot!(p_roc, [0, 1], [0, 1], color=:black, ls=:dash, label="Random Guess")
    scatter!(p_roc, [fpr_arr[opt_idx]], [tpr_arr[opt_idx]], color=:red, markersize=5, label="Optimal Cutoff ($(round(opt_thresh, digits=2)))")
    savefig(joinpath(plot_dir, "roc_curve.png"))
    println("  - ROC curve saved.")

    # --- Plot 2: Mission Trace ---
    step_duration_secs = step_size / sample_rate
    days = (1:num_samples) .* step_duration_secs ./ (24 * 3600)

    mask_zoom = days .<= 30.0 # View the first 30 days
    ds = max(1, Int(floor(sum(mask_zoom) / 1000))) # Dynamic downsampling to ~1000 points for the plot
    idx_ds = (1:ds:sum(mask_zoom))

    p1 = plot(days[mask_zoom][idx_ds], probs[mask_zoom][idx_ds], title="LISA Mission Detection Trace (30 Days)", 
              xlabel="Mission Time [Days]", ylabel="MBHB Probability", 
              lw=1.0, color=:darkred, label="QNN Output", alpha=0.8)
    plot!(p1, days[mask_zoom][idx_ds], y_true[mask_zoom][idx_ds] .* 0.5, st=:step, color=:blue, alpha=0.2, label="True Event Window", fill=(0, 0.2, :blue))
    hline!(p1, [opt_thresh], color=:black, ls=:dash, label="Optimal Threshold ($(round(opt_thresh, digits=2)))", lw=1.5)
    savefig(joinpath(plot_dir, "mission_trace_days.png"))
    println("  - Mission trace saved.")

    # --- Plot 3: Detection Sensitivity vs SNR ---
    snr_bins = 2.0:0.5:8.0
    bin_centers = []
    bin_accs = []
    for i in 1:(length(snr_bins)-1)
        low, high = snr_bins[i], snr_bins[i+1]
        mask = (snrs .>= low) .& (snrs .< high) .& (y_true .== 1)
        if sum(mask) > 0
            # Use optimal threshold for sensitivity counting
            acc = sum(probs[mask] .>= opt_thresh) / sum(mask)
            push!(bin_centers, (low + high)/2)
            push!(bin_accs, acc)
        end
    end
    p2 = plot(bin_centers, bin_accs, title="Detection Efficiency vs Signal Strength", 
              xlabel="Signal-to-Noise Ratio (SNR)", ylabel="True Positive Rate (TPR)", 
              marker=:circle, lw=2, color=:green, legend=false, ylims=(0, 1.05))
    savefig(joinpath(plot_dir, "detection_sensitivity_snr.png"))
    println("  - Sensitivity plot saved.")

    # --- Plot 4: Probability Distribution (Human Readable) ---
    p3 = histogram(probs[y_true .== 0], bins=50, label="Pure Noise/Forest", alpha=0.5, color=:gray)
    histogram!(p3, probs[y_true .== 1], bins=50, label="MBHB Events", alpha=0.5, color=:red, 
               title="Score Distribution: Separation Power", xlabel="Probability Score", ylabel="Count")
    vline!(p3, [opt_thresh], color=:black, ls=:dash, label="Opt. Threshold", lw=1.5)
    savefig(joinpath(plot_dir, "probability_distribution.png"))
    println("  - Probability distribution saved.")

    println("[SUCCESS] Diagnostic plots saved to '\$plot_dir/'")
end

main()