ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "common.jl"))

using Random, MilliHertzQML, Flux, MLUtils, Plots, UnicodePlots
using Logging, LoggingExtras, Printf, ArgParse, UUIDs, TOML, Dates

# Publication-ready plotting setup
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
        description = "Train the MilliHertzQML variational quantum classifier",
    )
    @add_arg_table s begin
        "--config"
        help = "Path to the configuration file"
        default = joinpath(dirname(@__DIR__), "config.toml")
        "--train-features"
        help = "Path to the feature CSV (chronologically ordered windows)"
        default = nothing
        "--train-labels"
        help = "Path to the label CSV"
        default = nothing
        "--test-mode"
        help = "Run in test mode (sample and epoch caps from [training]) for fast validation"
        action = :store_true
        "--epochs"
        help = "Maximum number of epochs to train"
        arg_type = Int
        default = nothing
        "--batch-size"
        help = "Batch size for the Adam optimizer"
        arg_type = Int
        default = nothing
        "--run-id"
        help = "Optional custom run ID (default: auto-generated UUID)"
        default = ""
    end
    return parse_args(s)
end

function format_duration(seconds)
    h = floor(Int, seconds / 3600)
    m = floor(Int, (seconds % 3600) / 60)
    s = floor(Int, seconds % 60)
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

function update_dashboard(run_id, test_mode, epoch, total_epochs, lr, history, elapsed)
    print("\033[2J")
    print("\033[H")
    println(
        "================================================================================",
    )
    println(
        "  MilliHertzQML TRAINING DASHBOARD | Run ID: $run_id | Mode: $(test_mode ? "TEST" : "FULL")",
    )
    println(
        "================================================================================",
    )
    elapsed_str = format_duration(elapsed)
    avg_per_epoch = elapsed / epoch
    eta_str = format_duration((total_epochs - epoch) * avg_per_epoch)
    @printf(
        "  Epoch: %3d/%3d | LR: %.5f | Elapsed: %s | ETA: %s\n",
        epoch,
        total_epochs,
        lr,
        elapsed_str,
        eta_str
    )
    println(
        "--------------------------------------------------------------------------------",
    )
    if length(history.train_loss) > 1
        p_loss = lineplot(
            history.epochs,
            history.train_loss,
            title = "Loss",
            name = "Train",
            color = :blue,
            width = 60,
            height = 10,
        )
        lineplot!(
            p_loss,
            history.epochs,
            history.val_loss,
            name = "Validation",
            color = :red,
        )
        println(p_loss)
        p_acc = lineplot(
            history.epochs,
            history.val_acc,
            title = "Validation accuracy",
            color = :green,
            width = 60,
            height = 10,
            ylim = (0, 1),
        )
        println(p_acc)
    else
        println("\n  [Waiting for more data to plot...]\n")
    end
    @printf(
        "  Current -> Train loss: %.4f | Val loss: %.4f | Val acc: %.4f\n",
        history.train_loss[end],
        history.val_loss[end],
        history.val_acc[end]
    )
    println(
        "================================================================================",
    )
end

"""
    predict_all(model, X) -> Vector{Float32}

Classifier probability of every row of the encoded feature matrix `X`.
"""
function predict_all(model, X)
    return Float32[predict_probability(model, @view(X[i, :])) for i in 1:size(X, 1)]
end

metrics_dict(m::NamedTuple) = Dict{String,Any}(String(k) => v for (k, v) in pairs(m))

function main()
    parsed_args = parse_commandline()

    # 1. Load and validate the TOML configuration
    config_file = load_config(parsed_args["config"])
    train_cfg = get(config_file, "training", Dict{String,Any}())
    model_cfg = get(config_file, "model", Dict{String,Any}())
    pre_cfg = get(config_file, "preprocessing", Dict{String,Any}())
    paths = pipeline_paths(config_file)

    # 2. Harmonize CLI with TOML defaults (CLI takes precedence)
    test_mode = parsed_args["test-mode"]
    test_mode_samples = cfgget(train_cfg, "test_mode_samples", 5000; type = Int, min = 1)
    test_mode_epochs = cfgget(train_cfg, "test_mode_epochs", 20; type = Int, min = 1)
    max_epochs =
        test_mode ? test_mode_epochs :
        override(
            parsed_args["epochs"],
            cfgget(train_cfg, "epochs", 100; type = Int, min = 1),
        )
    batch_size = override(
        parsed_args["batch-size"],
        cfgget(train_cfg, "batch_size", 32; type = Int, min = 1),
    )
    n_qubits = cfgget(model_cfg, "n_qubits", 4; type = Int, min = 2, max = 24)
    n_layers = cfgget(model_cfg, "n_layers", 4; type = Int, min = 1)
    initial_lr = cfgget(train_cfg, "learning_rate", 0.01; type = Float64, min = 1e-8)
    lr_decay = cfgget(train_cfg, "lr_decay", 0.95; type = Float64, min = 1e-3, max = 1.0)
    patience = cfgget(train_cfg, "patience", 12; type = Int, min = 1)
    train_fraction =
        cfgget(train_cfg, "train_fraction", 0.7; type = Float64, min = 0.05, max = 0.95)
    val_fraction = cfgget(
        train_cfg,
        "validation_fraction",
        0.15;
        type = Float64,
        min = 0.01,
        max = 0.5,
    )
    train_fraction + val_fraction < 1 || throw(
        ArgumentError(
            "train_fraction + validation_fraction = $(train_fraction + val_fraction); " *
            "must leave a test block.",
        ),
    )
    class_weight = cfgget(
        train_cfg,
        "class_weight",
        "balanced";
        type = String,
        choices = ("balanced", "none"),
    )
    threshold_criterion = cfgget(
        train_cfg,
        "threshold_criterion",
        "far";
        type = String,
        choices = ("far", "fpr", "youden"),
    )
    target_far = cfgget(train_cfg, "target_far_per_30d", 1.0; type = Float64, min = 0.0)
    target_fpr = cfgget(train_cfg, "target_fpr", 0.05; type = Float64, min = 0.0, max = 1.0)
    scaler_quantiles =
        cfgget(train_cfg, "scaler_quantiles", [0.005, 0.995]; type = AbstractVector)
    (
        length(scaler_quantiles) == 2 &&
        all(q -> q isa Real, scaler_quantiles) &&
        0 <= scaler_quantiles[1] < scaler_quantiles[2] <= 1
    ) || throw(
        ArgumentError(
            "configuration key `scaler_quantiles` = $(repr(scaler_quantiles)); " *
            "expected two ascending values in [0, 1].",
        ),
    )
    scaler_quantiles = (Float64(scaler_quantiles[1]), Float64(scaler_quantiles[2]))
    train_features_path = resolvepath(
        override(
            parsed_args["train-features"],
            cfgget(
                train_cfg,
                "train_features",
                "data/inputs/train_features.csv";
                type = String,
            ),
        ),
    )
    train_labels_path = resolvepath(
        override(
            parsed_args["train-labels"],
            cfgget(
                train_cfg,
                "train_labels",
                "data/inputs/train_labels.csv";
                type = String,
            ),
        ),
    )
    seed = cfgget(train_cfg, "seed", 42; type = Int)
    Random.seed!(seed)

    # --- Run directory and provenance snapshot ---
    run_id = isempty(parsed_args["run-id"]) ? string(uuid4())[1:8] : parsed_args["run-id"]
    run_dir = joinpath(paths.models, "run_$run_id")
    plot_dir = joinpath(paths.plots, "run_$run_id")
    mkpath(run_dir)
    mkpath(plot_dir)

    geometry = feature_geometry(train_features_path, pre_cfg)
    final_config = Dict(
        "model" => Dict("n_qubits" => n_qubits, "n_layers" => n_layers),
        "training" => Dict(
            "epochs" => max_epochs,
            "batch_size" => batch_size,
            "learning_rate" => initial_lr,
            "lr_decay" => lr_decay,
            "patience" => patience,
            "train_fraction" => train_fraction,
            "validation_fraction" => val_fraction,
            "class_weight" => class_weight,
            "threshold_criterion" => threshold_criterion,
            "target_far_per_30d" => target_far,
            "target_fpr" => target_fpr,
            "scaler_quantiles" => collect(scaler_quantiles),
            "train_features" => rootrelative(train_features_path),
            "train_labels" => rootrelative(train_labels_path),
            "test_mode" => test_mode,
            "run_id" => run_id,
            "seed" => seed,
        ),
        "features" => Dict(
            "window_size" => geometry.window_size,
            "step_size" => geometry.step_size,
            "sample_rate" => geometry.sample_rate,
        ),
        "hardware" => hardware_fingerprint(),
    )
    open(joinpath(run_dir, "config.toml"), "w") do io
        TOML.print(io, final_config)
    end

    file_logger = FileLogger(joinpath(run_dir, "training.log"))
    global_logger(TeeLogger(global_logger(), file_logger))

    println(
        "\n================================================================================",
    )
    println("  STARTING NEW TRAINING RUN | ID: [ $run_id ]")
    println(
        "================================================================================",
    )
    X_raw, y_raw, _ = load_data(train_features_path, train_labels_path)
    size(X_raw, 2) == n_qubits || throw(
        DimensionMismatch(
            "feature dimension $(size(X_raw, 2)) does not match n_qubits = $n_qubits.",
        ),
    )
    if test_mode
        n_keep = min(test_mode_samples, size(X_raw, 1))
        println(">>> TEST MODE: keeping the first $n_keep windows.")
        X_raw = X_raw[1:n_keep, :]
        y_raw = y_raw[1:n_keep]
    end

    # --- Chronological block split with a one-window buffer ---
    buffer = cld(geometry.window_size, geometry.step_size)
    blocks = chronological_split(
        size(X_raw, 1);
        train_fraction = train_fraction,
        validation_fraction = val_fraction,
        buffer = buffer,
    )
    open(joinpath(run_dir, "split.toml"), "w") do io
        TOML.print(
            io,
            Dict(
                "split" => Dict(
                    "n_windows" => size(X_raw, 1),
                    "buffer_windows" => buffer,
                    "train" => [first(blocks.train), last(blocks.train)],
                    "validation" => [first(blocks.validation), last(blocks.validation)],
                    "test" => [first(blocks.test), last(blocks.test)],
                    "features" => rootrelative(train_features_path),
                ),
            ),
        )
    end
    @info "Chronological split" train = blocks.train validation = blocks.validation test =
        blocks.test buffer = buffer

    # --- Feature scaler fitted on the training block only ---
    scaler = fit_scaler(X_raw[blocks.train, :]; quantiles = scaler_quantiles)
    X_train = encode_features(scaler, X_raw[blocks.train, :])
    X_val = encode_features(scaler, X_raw[blocks.validation, :])
    X_test = encode_features(scaler, X_raw[blocks.test, :])
    y_train = y_raw[blocks.train]
    y_val = y_raw[blocks.validation]
    y_test = y_raw[blocks.test]

    n_pos = count(==(1), y_train)
    n_neg = length(y_train) - n_pos
    positive_weight = 1.0
    if class_weight == "balanced"
        if n_pos == 0
            @warn "no positive window in the training block; class weighting disabled."
        else
            positive_weight = n_neg / n_pos
        end
    end
    @info "Training block" windows = length(y_train) positive = n_pos positive_weight =
        positive_weight

    # Pre-transpose for DataLoader (features x samples)
    X_train_t = copy(X_train')
    X_val_t = copy(X_val')

    # --- Model & optimizer ---
    model = VariationalQuantumClassifier(n_qubits, n_layers)
    opt_state = Flux.setup(Adam(initial_lr), model.params)
    train_loader = DataLoader((X_train_t, y_train), batchsize = batch_size, shuffle = true)
    val_loader = DataLoader((X_val_t, y_val), batchsize = batch_size, shuffle = false)
    history =
        (epochs = Int[], train_loss = Float32[], val_loss = Float32[], val_acc = Float32[])

    best_val_loss = Inf32
    epochs_no_improve = 0
    start_time = time()
    best_path = joinpath(run_dir, "gw_model_best.jld2")
    @info "Training started" batch_size = batch_size max_epochs = max_epochs n_qubits =
        n_qubits n_layers = n_layers

    for epoch in 1:max_epochs
        current_lr = initial_lr * (lr_decay^(epoch - 1))
        Flux.adjust!(opt_state, current_lr)

        epoch_train_loss = 0.0f0
        for (Xbatch_t, ybatch) in train_loader
            epoch_train_loss += train_step!(
                model,
                opt_state,
                Xbatch_t',
                ybatch;
                positive_weight = positive_weight,
            )
        end
        avg_train_loss = epoch_train_loss / length(train_loader)

        epoch_val_loss = 0.0f0
        val_correct = 0.0f0
        for (Xval_t, yval) in val_loader
            epoch_val_loss +=
                loss_function(model, Xval_t', yval; positive_weight = positive_weight) *
                length(yval)
            val_correct += accuracy(model, Xval_t', yval) * length(yval)
        end
        avg_val_loss = epoch_val_loss / length(y_val)
        avg_val_acc = val_correct / length(y_val)

        push!(history.epochs, epoch)
        push!(history.train_loss, avg_train_loss)
        push!(history.val_loss, avg_val_loss)
        push!(history.val_acc, avg_val_acc)
        elapsed = time() - start_time
        update_dashboard(run_id, test_mode, epoch, max_epochs, current_lr, history, elapsed)
        @info "Epoch complete" epoch = epoch lr = current_lr train_loss = avg_train_loss val_loss =
            avg_val_loss val_acc = avg_val_acc elapsed = elapsed

        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_model(
                best_path,
                model;
                metadata = Dict(
                    "run_id" => run_id,
                    "seed" => seed,
                    "epoch" => epoch,
                    "val_loss" => avg_val_loss,
                    "config" => final_config,
                ),
                scaler = scaler,
            )
        else
            epochs_no_improve += 1
        end
        if epochs_no_improve >= patience
            @info "Early stopping" best_val_loss = best_val_loss
            break
        end
    end

    if isfile(best_path)
        model, best_meta, best_scaler = load_model(best_path)
        save_model(
            joinpath(run_dir, "gw_model.jld2"),
            model;
            metadata = best_meta,
            scaler = best_scaler,
        )
    end

    # --- Decision threshold fitted on the validation block only ---
    println(
        "\n[FINISH] Training complete. Fitting the decision threshold on the validation block...",
    )
    probs_val = predict_all(model, X_val)
    threshold, info = select_threshold(
        y_val,
        probs_val;
        criterion = threshold_criterion,
        target_far_per_30d = target_far,
        target_fpr = target_fpr,
        step_size = geometry.step_size,
        sample_rate = geometry.sample_rate,
    )
    fpr_val, tpr_val, _ = roc_curve(y_val, probs_val)
    info["auc"] = roc_auc(fpr_val, tpr_val)
    info["value"] = threshold
    info["fitted_on"] = rootrelative(train_features_path) * " (validation block)"
    info["fitted_at"] = string(Dates.now())
    open(joinpath(run_dir, "threshold.toml"), "w") do io
        TOML.print(io, Dict("threshold" => info))
    end
    @info "Threshold" value = threshold criterion = info["criterion"] validation_auc =
        info["auc"]

    # --- The isolated test block, evaluated once ---
    probs_test = predict_all(model, X_test)
    m_val = event_metrics(
        Int.(probs_val .>= threshold),
        y_val;
        step_size = geometry.step_size,
        sample_rate = geometry.sample_rate,
    )
    m_test = event_metrics(
        Int.(probs_test .>= threshold),
        y_test;
        step_size = geometry.step_size,
        sample_rate = geometry.sample_rate,
    )
    fpr_test, tpr_test, _ = roc_curve(y_test, probs_test)
    metrics = Dict(
        "validation" => merge(
            metrics_dict(m_val),
            Dict(
                "auc" => info["auc"],
                "n_windows" => length(y_val),
                "threshold" => threshold,
            ),
        ),
        "test" => merge(
            metrics_dict(m_test),
            Dict(
                "auc" => roc_auc(fpr_test, tpr_test),
                "n_windows" => length(y_test),
                "threshold" => threshold,
            ),
        ),
    )
    open(joinpath(run_dir, "metrics.toml"), "w") do io
        TOML.print(io, metrics)
    end
    println(
        "\n  Block        AUC      Prec.   Recall  F1      Bal.acc  Events  Detected  FA/30 d",
    )
    for (name, m, auc) in
        (("validation", m_val, info["auc"]), ("test", m_test, metrics["test"]["auc"]))
        @printf(
            "  %-11s  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6d  %8d  %7.2f\n",
            name,
            auc,
            m.precision,
            m.recall,
            m.f1,
            m.balanced_accuracy,
            m.n_events,
            m.n_detected,
            m.false_alarms_per_30d
        )
    end
    @info "Test block" auc = metrics["test"]["auc"] precision = m_test.precision recall =
        m_test.recall event_recall = m_test.event_recall false_alarms_per_30d =
        m_test.false_alarms_per_30d

    p1 = plot(
        history.train_loss,
        label = "Training loss",
        xlabel = "Epoch",
        ylabel = "Loss",
        lw = 2,
    )
    plot!(p1, history.val_loss, label = "Validation loss", lw = 2, linestyle = :dash)
    p2 = Plots.plot(
        history.val_acc,
        xlabel = "Epoch",
        ylabel = "Validation accuracy",
        lw = 2,
        color = :green,
        legend = false,
    )
    Plots.plot(p1, p2, layout = (2, 1), size = (1000, 800))
    Plots.savefig(joinpath(plot_dir, "training_metrics.png"))

    println("[SUCCESS] Training results and logs saved to: $run_dir")
end

main()
