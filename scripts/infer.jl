ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "common.jl"))

using MilliHertzQML, CSV, DataFrames, Plots, ArgParse, TOML, Dates, Printf

function parse_commandline()
    s = ArgParseSettings(
        description = "Run inference with a trained MilliHertzQML classifier",
    )
    @add_arg_table s begin
        "--config"
        help = "Path to the configuration file"
        default = joinpath(dirname(@__DIR__), "config.toml")
        "--features"
        help = "Path to the feature CSV"
        default = nothing
        "--labels"
        help = "Path to the label CSV; an empty string (or a missing file) selects blind inference"
        default = nothing
        "--model"
        help = "Path to the trained model (.jld2)"
        default = nothing
        "--run-id"
        help = "Run ID of the training run (locates the model unless --model is given) and of the outputs"
        default = ""
        "--block"
        help = "Rows to evaluate when the features are the training table: all, validation, or test"
        default = nothing
        "--step-size"
        help = "Window step in samples (used only without a feature sidecar)"
        arg_type = Int
        default = nothing
        "--sample-rate"
        help = "Sampling frequency in Hz (used only without a feature sidecar)"
        arg_type = Float64
        default = nothing
    end
    return parse_args(s)
end

metrics_dict(m::NamedTuple) = Dict{String,Any}(String(k) => v for (k, v) in pairs(m))

function main()
    parsed_args = parse_commandline()

    # 1. Load and validate the TOML configuration
    config_file = load_config(parsed_args["config"])
    paths = pipeline_paths(config_file)
    infer_cfg = get(config_file, "inference", Dict{String,Any}())
    pre_cfg = get(config_file, "preprocessing", Dict{String,Any}())

    # 2. Harmonize CLI with TOML defaults (CLI takes precedence)
    run_id = parsed_args["run-id"]
    features_path = resolvepath(
        override(
            parsed_args["features"],
            cfgget(
                infer_cfg,
                "features",
                "data/inputs/inference_features.csv";
                type = String,
            ),
        ),
    )
    labels_path = override(
        parsed_args["labels"],
        cfgget(infer_cfg, "labels", "data/inputs/inference_labels.csv"; type = String),
    )
    block = override(
        parsed_args["block"],
        cfgget(
            infer_cfg,
            "block",
            "all";
            type = String,
            choices = ("all", "validation", "test"),
        ),
    )
    block in ("all", "validation", "test") ||
        throw(ArgumentError("--block $block; expected all, validation, or test."))
    has_labels = !isempty(labels_path) && isfile(resolvepath(labels_path))
    has_labels && (labels_path = resolvepath(labels_path))

    (parsed_args["model"] !== nothing || !isempty(run_id)) || throw(
        ArgumentError(
            "no model specified: provide --model <path> or --run-id <id> of a training run.",
        ),
    )
    model_path =
        parsed_args["model"] !== nothing ? resolvepath(parsed_args["model"]) :
        joinpath(paths.models, "run_$run_id", "gw_model.jld2")
    model_dir = dirname(model_path)
    isempty(run_id) && (run_id = "standalone_" * string(hash(model_path))[1:6])

    geometry =
        if parsed_args["step-size"] !== nothing || parsed_args["sample-rate"] !== nothing
            (
                step_size = override(
                    parsed_args["step-size"],
                    cfgget(infer_cfg, "step_size", 100; type = Int, min = 1),
                ),
                sample_rate = override(
                    parsed_args["sample-rate"],
                    cfgget(infer_cfg, "sample_rate", 0.2; type = Float64, min = 1e-6),
                ),
            )
        else
            g = feature_geometry(features_path, pre_cfg)
            (step_size = g.step_size, sample_rate = g.sample_rate)
        end

    plot_dir = joinpath(paths.plots, "run_$run_id")
    res_dir = joinpath(paths.results, "run_$run_id")
    mkpath(plot_dir)
    mkpath(res_dir)

    # 3. Model, scaler, and the threshold fitted at training time
    model, model_meta, scaler = load_model(model_path)
    scaler === nothing && error(
        "the model artifact $model_path carries no feature scaler; " *
        "retrain it with the current pipeline.",
    )
    thresh_file = joinpath(model_dir, "threshold.toml")
    isfile(thresh_file) || error(
        "no decision threshold at $thresh_file; the threshold is fitted on the " *
        "validation block by scripts/train.jl.",
    )
    tcfg = TOML.parsefile(thresh_file)["threshold"]
    threshold = Float32(tcfg["value"])
    println("\n[INFER] Model loaded from $model_path")
    println(
        "[INFER] Threshold $(round(threshold, digits = 4)) (criterion: $(tcfg["criterion"]), " *
        "fitted at $(tcfg["fitted_at"]))",
    )

    # 4. Data, restricted to a block of the training table when requested
    local y_true, snrs
    if has_labels
        X_raw, y_true, df_meta = load_data(features_path, labels_path)
        snrs = "SNR" in names(df_meta) ? df_meta[:, :SNR] : zeros(Float32, size(X_raw, 1))
    else
        println("[INFER] Blind mode: no labels.")
        X_raw = load_features(features_path)
    end
    row_offset = 0
    if block != "all"
        split_file = joinpath(model_dir, "split.toml")
        isfile(split_file) || error(
            "--block $block requires the split.toml written by scripts/train.jl in $model_dir.",
        )
        split = TOML.parsefile(split_file)["split"]
        split["n_windows"] == size(X_raw, 1) || error(
            "--block $block applies to the training table of $(split["n_windows"]) windows; " *
            "the given features hold $(size(X_raw, 1)).",
        )
        lo, hi = split[block]
        rows = lo:hi
        row_offset = lo - 1
        X_raw = X_raw[rows, :]
        if has_labels
            y_true = y_true[rows]
            snrs = snrs[rows]
        end
        println("[INFER] Evaluating the $block block: windows $lo-$hi.")
    end
    X = encode_features(scaler, X_raw)

    # 5. Forward pass over all windows
    num_samples = size(X, 1)
    println("Analyzing $num_samples windows...")
    probs = Float32[]
    for i in 1:num_samples
        push!(probs, predict_probability(model, @view(X[i, :])))
        if i % max(1, div(num_samples, 10)) == 0
            println("  Progress: $(round(Int, i / num_samples * 100))%")
        end
    end
    decisions = Int.(probs .>= threshold)

    # 6. Persist per-window scores and decisions, the snapshot, and metrics
    results = DataFrame(
        Window = row_offset .+ (1:num_samples),
        Probability = probs,
        Detection = decisions,
    )
    if has_labels
        results.Label = y_true
        results.SNR = snrs
    end
    CSV.write(joinpath(res_dir, "inference_probabilities.csv"), results)

    final_config = Dict(
        "inference" => Dict(
            "features" => rootrelative(features_path),
            "labels" => has_labels ? rootrelative(labels_path) : "",
            "model" => rootrelative(model_path),
            "block" => block,
            "step_size" => geometry.step_size,
            "sample_rate" => geometry.sample_rate,
            "threshold" => Float64(threshold),
            "run_id" => run_id,
            "blind" => !has_labels,
        ),
        "hardware" => hardware_fingerprint(),
    )
    open(joinpath(res_dir, "config_infer.toml"), "w") do io
        TOML.print(io, final_config)
    end

    local auc_score, fpr_arr, tpr_arr
    if has_labels
        fpr_arr, tpr_arr, _ = roc_curve(y_true, probs)
        auc_score = roc_auc(fpr_arr, tpr_arr)
        m = event_metrics(
            decisions,
            y_true;
            step_size = geometry.step_size,
            sample_rate = geometry.sample_rate,
        )
        open(joinpath(res_dir, "metrics.toml"), "w") do io
            TOML.print(
                io,
                Dict(
                    "metrics" => merge(
                        metrics_dict(m),
                        Dict(
                            "auc" => auc_score,
                            "n_windows" => num_samples,
                            "threshold" => Float64(threshold),
                            "block" => block,
                        ),
                    ),
                ),
            )
        end
        @printf(
            "\n[RESULT] AUC %.3f | precision %.3f | recall %.3f | F1 %.3f | balanced accuracy %.3f\n",
            auc_score,
            m.precision,
            m.recall,
            m.f1,
            m.balanced_accuracy
        )
        @printf(
            "[RESULT] events %d, detected %d | false-alarm episodes %d over %.1f days = %.2f per 30 d\n",
            m.n_events,
            m.n_detected,
            m.n_false_alarm_episodes,
            m.observation_days,
            m.false_alarms_per_30d
        )
    end

    # 7. Diagnostic figures
    println("[PLOT] Generating diagnostics...")
    Plots.default(
        dpi = 600,
        frame = :box,
        fontfamily = "Computer Modern",
        grid = true,
        gridalpha = 0.2,
        minorgrid = false,
        margin = 5Plots.mm,
    )
    step_duration_secs = geometry.step_size / geometry.sample_rate
    days = (row_offset .+ (1:num_samples)) .* step_duration_secs ./ 86400
    ds = max(1, div(num_samples, 5000))
    idx = 1:ds:num_samples
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
            fpr_arr,
            tpr_arr,
            xlabel = "False-positive rate",
            ylabel = "True-positive rate",
            lw = 2,
            color = :purple,
            label = "VQC (AUC = $(round(auc_score, digits = 3)))",
        )
        plot!(p_roc, [0, 1], [0, 1], color = :black, ls = :dash, label = "Random")
        savefig(joinpath(plot_dir, "roc_curve.png"))

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
            p2 = plot(
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
        vline!(p3, [threshold], color = :black, ls = :dash, label = "Threshold", lw = 1.5)
        savefig(joinpath(plot_dir, "probability_distribution.png"))
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
        vline!(p3, [threshold], color = :black, ls = :dash, label = "Threshold", lw = 1.5)
        savefig(joinpath(plot_dir, "probability_distribution.png"))
    end

    println("[SUCCESS] Diagnostics saved to '$plot_dir'; results in '$res_dir'")
end

main()
