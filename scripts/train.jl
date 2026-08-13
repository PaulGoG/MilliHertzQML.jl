ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "common.jl"))

using Random, MilliHertzQML, Flux, MLUtils, Plots, UnicodePlots
using Logging, LoggingExtras, Printf, ArgParse, UUIDs, TOML

# Publication-ready plotting setup
Plots.default(dpi=600, frame=:box, fontfamily="Computer Modern", grid=true, gridalpha=0.2, minorgrid=false, margin=5Plots.mm)

function parse_commandline()
    s = ArgParseSettings(description = "Train the MilliHertzQML Variational Quantum Classifier")
    @add_arg_table s begin
        "--config"
            help = "Path to the configuration file"
            default = joinpath(dirname(@__DIR__), "config.toml")
        "--train-features"
            help = "Path to the training features CSV"
            default = nothing
        "--train-labels"
            help = "Path to the training labels CSV"
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
            help = "Optional custom Run ID (default: auto-generated UUID)"
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
    # Clear terminal
    print("\033[2J")
    print("\033[H")

    println("================================================================================")
    println("  MilliHertzQML TRAINING DASHBOARD | Run ID: $run_id | Mode: $(test_mode ? "TEST" : "FULL")")
    println("================================================================================")

    # Time Tracking
    elapsed_str = format_duration(elapsed)
    avg_per_epoch = elapsed / epoch
    eta = (total_epochs - epoch) * avg_per_epoch
    eta_str = format_duration(eta)

    @printf("  Epoch: %3d/%3d | LR: %.5f | Elapsed: %s | ETA: %s\n", epoch, total_epochs, lr, elapsed_str, eta_str)
    println("--------------------------------------------------------------------------------")

    if length(history.train_loss) > 1
        # Loss Plot
        p_loss = lineplot(history.epochs, history.train_loss, title="Loss Convergence", name="Train", color=:blue, width=60, height=10)
        lineplot!(p_loss, history.epochs, history.val_loss, name="Val", color=:red)
        println(p_loss)

        # Accuracy Plot
        p_acc = lineplot(history.epochs, history.val_acc, title="Validation Accuracy", color=:green, width=60, height=10, ylim=(0, 1))
        println(p_acc)
    else
        println("\n  [Waiting for more data to plot...]\n")
    end

    @printf("  Current Stats -> Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f\n",
            history.train_loss[end], history.val_loss[end], history.val_acc[end])
    println("================================================================================")
end

function main()
    parsed_args = parse_commandline()

    # 1. Load and validate TOML configuration
    config_file = load_config(parsed_args["config"])
    train_cfg = get(config_file, "training", Dict{String, Any}())
    model_cfg = get(config_file, "model", Dict{String, Any}())

    # 2. Harmonize CLI with TOML defaults (CLI takes precedence)
    test_mode = parsed_args["test-mode"]
    test_mode_samples = cfgget(train_cfg, "test_mode_samples", 5000; type = Int, min = 1)
    test_mode_epochs = cfgget(train_cfg, "test_mode_epochs", 20; type = Int, min = 1)
    max_epochs = test_mode ? test_mode_epochs :
        override(parsed_args["epochs"], cfgget(train_cfg, "epochs", 100; type = Int, min = 1))
    batch_size = override(parsed_args["batch-size"],
        cfgget(train_cfg, "batch_size", 32; type = Int, min = 1))

    n_qubits = cfgget(model_cfg, "n_qubits", 4; type = Int, min = 2, max = 24)
    n_layers = cfgget(model_cfg, "n_layers", 4; type = Int, min = 1)
    initial_lr = cfgget(train_cfg, "learning_rate", 0.01; type = Float64, min = 1e-8)
    lr_decay = cfgget(train_cfg, "lr_decay", 0.95; type = Float64, min = 1e-3, max = 1.0)
    patience = cfgget(train_cfg, "patience", 12; type = Int, min = 1)
    val_fraction = cfgget(train_cfg, "validation_fraction", 0.1;
                          type = Float64, min = 0.01, max = 0.5)

    train_features_path = resolvepath(override(parsed_args["train-features"],
        cfgget(train_cfg, "train_features", "data/inputs/train_features.csv"; type = String)))
    train_labels_path = resolvepath(override(parsed_args["train-labels"],
        cfgget(train_cfg, "train_labels", "data/inputs/train_labels.csv"; type = String)))

    seed = cfgget(train_cfg, "seed", 42; type = Int)
    Random.seed!(seed)

    # --- Run ID & Directory Setup ---
    run_id = isempty(parsed_args["run-id"]) ? string(uuid4())[1:8] : parsed_args["run-id"]
    run_dir = joinpath(PROJECT_ROOT, "models", "run_$run_id")
    plot_dir = joinpath(PROJECT_ROOT, "data", "outputs", "plots", "run_$run_id")
    mkpath(run_dir)
    mkpath(plot_dir)

    # --- Save Configuration Snapshot for Reproducibility ---
    final_config = Dict(
        "model" => Dict(
            "n_qubits" => n_qubits,
            "n_layers" => n_layers,
        ),
        "training" => Dict(
            "epochs" => max_epochs,
            "batch_size" => batch_size,
            "learning_rate" => initial_lr,
            "lr_decay" => lr_decay,
            "patience" => patience,
            "validation_fraction" => val_fraction,
            "train_features" => rootrelative(train_features_path),
            "train_labels" => rootrelative(train_labels_path),
            "test_mode" => test_mode,
            "run_id" => run_id,
            "seed" => seed,
        ),
    )
    open(joinpath(run_dir, "config.toml"), "w") do io
        TOML.print(io, final_config)
    end

    # --- Setup Logging ---
    file_logger = FileLogger(joinpath(run_dir, "training.log"))
    global_logger(TeeLogger(global_logger(), file_logger))

    println("\n================================================================================")
    println("  STARTING NEW TRAINING RUN | ID: [ $run_id ]")
    println("================================================================================")
    X_raw, y_raw, _ = load_data(train_features_path, train_labels_path)
    size(X_raw, 2) == n_qubits || throw(DimensionMismatch(
        "feature dimension $(size(X_raw, 2)) does not match n_qubits = $n_qubits."))

    if test_mode
        println(">>> TEST MODE: capping dataset at $test_mode_samples samples.")
        idx_subset = randperm(size(X_raw, 1))[1:min(test_mode_samples, size(X_raw, 1))]
        X_raw = X_raw[idx_subset, :]
        y_raw = y_raw[idx_subset]
    end

    # Split (random shuffle; validation also serves as the reported test set —
    # a known evaluation deficiency scheduled for remediation)
    n_samples = size(X_raw, 1)
    split_idx = Int(floor((1 - val_fraction) * n_samples))
    indices = shuffle(1:n_samples)
    train_idx = indices[1:split_idx]
    val_idx = indices[split_idx+1:end]

    X_train, y_train = X_raw[train_idx, :], y_raw[train_idx]
    X_val, y_val = X_raw[val_idx, :], y_raw[val_idx]

    # Pre-transpose for DataLoader (features x samples) to avoid allocations
    X_train_t = copy(X_train')
    X_val_t = copy(X_val')

    # --- Model & Optimizer ---
    model = VariationalQuantumClassifier(n_qubits, n_layers)
    opt_state = Flux.setup(Adam(initial_lr), model.params)

    train_loader = DataLoader((X_train_t, y_train), batchsize=batch_size, shuffle=true)
    val_loader = DataLoader((X_val_t, y_val), batchsize=batch_size, shuffle=false)

    history = (epochs = Int[], train_loss = Float32[], val_loss = Float32[], val_acc = Float32[])

    # --- Training Loop ---
    best_val_loss = Inf32
    epochs_no_improve = 0
    start_time = time()

    @info "Training started" n_samples=n_samples batch_size=batch_size max_epochs=max_epochs n_qubits=n_qubits n_layers=n_layers

    for epoch in 1:max_epochs
        current_lr = initial_lr * (lr_decay ^ (epoch - 1))
        Flux.adjust!(opt_state, current_lr)

        # Training
        epoch_train_loss = 0.0f0
        for (Xbatch_t, ybatch) in train_loader
            l = train_step!(model, opt_state, Xbatch_t', ybatch)
            epoch_train_loss += l
        end
        avg_train_loss = epoch_train_loss / length(train_loader)

        # Validation (loss and accuracy both weighted per sample)
        epoch_val_loss = 0.0f0
        val_correct = 0.0f0
        for (Xval_t, yval) in val_loader
            epoch_val_loss += loss_function(model, Xval_t', yval) * length(yval)
            val_correct += accuracy(model, Xval_t', yval) * length(yval)
        end
        avg_val_loss = epoch_val_loss / length(y_val)
        avg_val_acc = val_correct / length(y_val)

        push!(history.epochs, epoch)
        push!(history.train_loss, avg_train_loss)
        push!(history.val_loss, avg_val_loss)
        push!(history.val_acc, avg_val_acc)

        elapsed = time() - start_time

        # Update terminal dashboard
        update_dashboard(run_id, test_mode, epoch, max_epochs, current_lr, history, elapsed)

        # Log to file
        @info "Epoch complete" epoch=epoch lr=current_lr train_loss=avg_train_loss val_loss=avg_val_loss val_acc=avg_val_acc elapsed=elapsed

        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_model(joinpath(run_dir, "gw_model_best.jld2"), model;
                       metadata = Dict("run_id" => run_id, "seed" => seed,
                                       "epoch" => epoch, "val_loss" => avg_val_loss,
                                       "config" => final_config))
        else
            epochs_no_improve += 1
        end

        if epochs_no_improve >= patience
            @info "Early stopping" best_val_loss=best_val_loss
            break
        end
    end

    if isfile(joinpath(run_dir, "gw_model_best.jld2"))
        model, best_meta = load_model(joinpath(run_dir, "gw_model_best.jld2"))
        save_model(joinpath(run_dir, "gw_model.jld2"), model; metadata = best_meta)
    end

    println("\n[FINISH] Training Complete. Final Evaluation...")
    final_val_acc = accuracy(model, X_val_t', y_val)
    @info "Final result" val_accuracy=final_val_acc

    # Final plot to file
    p1 = plot(history.train_loss, label="Train Loss", title="VQC Loss Convergence", xlabel="Epoch", ylabel="Loss", lw=2)
    plot!(p1, history.val_loss, label="Val Loss", lw=2, linestyle=:dash)
    p2 = Plots.plot(history.val_acc, title="VQC Validation Accuracy", xlabel="Epoch", ylabel="Accuracy", lw=2, color=:green, legend=false)
    Plots.plot(p1, p2, layout=(2,1), size=(1000, 800))
    Plots.savefig(joinpath(plot_dir, "training_metrics.png"))

    println("[SUCCESS] Training results and logs saved to: $run_dir")
end

main()
