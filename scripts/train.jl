ENV["GKSwstype"] = "100"
using Pkg
Pkg.activate(dirname(@__DIR__); io = devnull)
Pkg.instantiate(; io = devnull)

using Random, QuantumGW, Flux, MLUtils, Statistics, CSV, DataFrames, Plots, Dates
using UnicodePlots, Logging, LoggingExtras, Printf, ArgParse, UUIDs, TOML

const PROJECT_ROOT = dirname(@__DIR__)
resolvepath(p) = isabspath(p) ? p : joinpath(PROJECT_ROOT, p)

# Publication-ready plotting setup
Plots.default(dpi=600, frame=:box, fontfamily="Computer Modern", grid=true, gridalpha=0.2, minorgrid=false, margin=5Plots.mm)

function parse_commandline()
    s = ArgParseSettings(description = "Train the QuantumGW Variational Quantum Classifier")
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
            help = "Run in test mode (5,000 samples, 20 epochs) for fast validation"
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
        "--use-real-data"
            help = "Automatically use the pre-processed real telemetry data (telemetry_train_features.csv)"
            action = :store_true
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

# 1. Load TOML
config_file = isfile(parsed_args["config"]) ? TOML.parsefile(parsed_args["config"]) : Dict{String, Any}()
train_cfg = get(config_file, "training", Dict{String, Any}())

# 2. Harmonize CLI with TOML Defaults
TEST_MODE = parsed_args["test-mode"]
MAX_EPOCHS = TEST_MODE ? 20 : (parsed_args["epochs"] !== nothing ? parsed_args["epochs"] : get(train_cfg, "epochs", 100))
BATCH_SIZE = parsed_args["batch-size"] !== nothing ? parsed_args["batch-size"] : get(train_cfg, "batch_size", 32)

train_features_path = resolvepath(parsed_args["train-features"] !== nothing ? parsed_args["train-features"] : get(train_cfg, "train_features", "data/inputs/train_features.csv"))
train_labels_path = resolvepath(parsed_args["train-labels"] !== nothing ? parsed_args["train-labels"] : get(train_cfg, "train_labels", "data/inputs/train_labels.csv"))

if parsed_args["use-real-data"]
    train_features_path = joinpath(PROJECT_ROOT, "data", "inputs", "telemetry_train_features.csv")
    train_labels_path = joinpath(PROJECT_ROOT, "data", "inputs", "telemetry_train_labels.csv")
end

SEED = get(train_cfg, "seed", 42)
Random.seed!(SEED)

# --- Run ID & Directory Setup ---
run_id = isempty(parsed_args["run-id"]) ? string(uuid4())[1:8] : parsed_args["run-id"]
run_dir = joinpath(PROJECT_ROOT, "models", "run_$run_id")
plot_dir = joinpath(PROJECT_ROOT, "data", "outputs", "plots", "run_$run_id")
mkpath(run_dir)
mkpath(plot_dir)

# --- Save Configuration Snapshot for Reproducibility ---
final_config = Dict(
    "training" => Dict(
        "epochs" => MAX_EPOCHS,
        "batch_size" => BATCH_SIZE,
        "train_features" => train_features_path,
        "train_labels" => train_labels_path,
        "test_mode" => TEST_MODE,
        "use_real_data" => parsed_args["use-real-data"],
        "run_id" => run_id,
        "seed" => SEED
    )
)
open(joinpath(run_dir, "config.toml"), "w") do io
    TOML.print(io, final_config)
end

# --- Setup Logging ---
file_logger = FileLogger(joinpath(run_dir, "training.log"))
global_logger(TeeLogger(global_logger(), file_logger))

println("\n================================================================================")
println("  🚀 STARTING NEW TRAINING RUN | ID: [ $run_id ]")
println("================================================================================")
X_raw, y_raw, _ = load_data(train_features_path, train_labels_path)

if TEST_MODE
    println(">>> TEST MODE ENABLED: Slicing dataset to 5,000 samples for speed.")
    idx_subset = randperm(size(X_raw, 1))[1:min(5000, size(X_raw, 1))]
    X_raw = X_raw[idx_subset, :]
    y_raw = y_raw[idx_subset]
end

# Split
n_samples = size(X_raw, 1)
split_idx = Int(floor(0.9 * n_samples))
indices = shuffle(1:n_samples)
train_idx = indices[1:split_idx]
test_idx = indices[split_idx+1:end]

X_train, y_train = X_raw[train_idx, :], y_raw[train_idx]
X_test, y_test = X_raw[test_idx, :], y_raw[test_idx]

# Pre-transpose for DataLoader (features x samples) to avoid allocations
X_train_t = copy(X_train')
X_test_t = copy(X_test')

# --- Model & Opt ---
n_qubits = 4
n_layers = 4
model = VariationalQuantumClassifier(n_qubits, n_layers)
initial_lr = 0.01
opt_state = Flux.setup(Adam(initial_lr), model.params)

batch_size = BATCH_SIZE
train_loader = DataLoader((X_train_t, y_train), batchsize=batch_size, shuffle=true)
val_loader = DataLoader((X_test_t, y_test), batchsize=batch_size, shuffle=false)

# --- Metrics Containers ---
history_train_loss = Float32[]
history_val_loss = Float32[]
history_val_acc = Float32[]
history_epochs = Int[]

function format_duration(seconds)
    h = floor(Int, seconds / 3600)
    m = floor(Int, (seconds % 3600) / 60)
    s = floor(Int, seconds % 60)
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

# --- Dashboard Function ---
function update_dashboard(epoch, lr, train_loss, val_loss, val_acc, elapsed, total_epochs)
    # Clear terminal
    print("\033[2J")
    print("\033[H")

    println("================================================================================")
    println("  QuantumGW TRAINING DASHBOARD | Run ID: $run_id | Mode: $(TEST_MODE ? "TEST" : "FULL")")
    println("================================================================================")

    # Time Tracking
    elapsed_str = format_duration(elapsed)
    avg_per_epoch = elapsed / epoch
    eta = (total_epochs - epoch) * avg_per_epoch
    eta_str = format_duration(eta)

    @printf("  Epoch: %3d/%3d | LR: %.5f | Elapsed: %s | ETA: %s\n", epoch, total_epochs, lr, elapsed_str, eta_str)
    println("--------------------------------------------------------------------------------")

    if length(history_train_loss) > 1
        # Loss Plot
        p_loss = lineplot(history_epochs, history_train_loss, title="Loss Convergence", name="Train", color=:blue, width=60, height=10)
        lineplot!(p_loss, history_epochs, history_val_loss, name="Val", color=:red)
        println(p_loss)

        # Accuracy Plot
        p_acc = lineplot(history_epochs, history_val_acc, title="Validation Accuracy", color=:green, width=60, height=10, ylim=(0, 1))
        println(p_acc)
    else
        println("\n  [Waiting for more data to plot...]\n")
    end

    @printf("  Current Stats -> Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f\n", train_loss, val_loss, val_acc)
    println("================================================================================")
end

# --- Training Loop ---
patience = 12
best_val_loss = Inf32
epochs_no_improve = 0
lr_decay = 0.95

start_time = time()

@info "Training started" n_samples=n_samples batch_size=batch_size max_epochs=MAX_EPOCHS

for epoch in 1:MAX_EPOCHS
    current_lr = initial_lr * (lr_decay ^ (epoch - 1))
    Flux.adjust!(opt_state, current_lr)

    # Training
    epoch_train_loss = 0.0f0
    for (Xbatch_t, ybatch) in train_loader
        l = train_step!(model, opt_state, Xbatch_t', ybatch)
        epoch_train_loss += l
    end
    avg_train_loss = epoch_train_loss / length(train_loader)

    # Validation (loss averaged per batch, accuracy weighted per sample)
    epoch_val_loss = 0.0f0
    val_correct = 0.0f0
    for (Xval_t, yval) in val_loader
        epoch_val_loss += loss_function(model, Xval_t', yval)
        val_correct += accuracy(model, Xval_t', yval) * length(yval)
    end
    avg_val_loss = epoch_val_loss / length(val_loader)
    avg_val_acc = val_correct / length(y_test)

    push!(history_train_loss, avg_train_loss)
    push!(history_val_loss, avg_val_loss)
    push!(history_val_acc, avg_val_acc)
    push!(history_epochs, epoch)

    elapsed = time() - start_time

    # Update REPL Visual
    update_dashboard(epoch, current_lr, avg_train_loss, avg_val_loss, avg_val_acc, elapsed, MAX_EPOCHS)

    # Log to file
    @info "Epoch complete" epoch=epoch lr=current_lr train_loss=avg_train_loss val_loss=avg_val_loss val_acc=avg_val_acc elapsed=elapsed

    if avg_val_loss < best_val_loss
        global best_val_loss = avg_val_loss
        global epochs_no_improve = 0
        save_model(joinpath(run_dir, "gw_model_best.jld2"), model;
                   metadata = Dict("run_id" => run_id, "seed" => SEED,
                                   "epoch" => epoch, "val_loss" => avg_val_loss,
                                   "config" => final_config))
    else
        global epochs_no_improve += 1
    end

    if epochs_no_improve >= patience
        @info "Early stopping" best_val_loss=best_val_loss
        break
    end
end

if isfile(joinpath(run_dir, "gw_model_best.jld2"))
    best_meta = Dict{String, Any}()
    model, best_meta = load_model(joinpath(run_dir, "gw_model_best.jld2"))
    save_model(joinpath(run_dir, "gw_model.jld2"), model; metadata = best_meta)
end

println("\n[FINISH] Training Complete. Final Evaluation...")
final_test_acc = accuracy(model, X_test_t', y_test)
@info "Final result" test_accuracy=final_test_acc

# Final plot to file
p1 = plot(history_train_loss, label="Train Loss", title="VQC Loss Convergence", xlabel="Epoch", ylabel="Loss", lw=2)
plot!(p1, history_val_loss, label="Val Loss", lw=2, linestyle=:dash)
p2 = Plots.plot(history_val_acc, title="VQC Validation Accuracy", xlabel="Epoch", ylabel="Accuracy", lw=2, color=:green, legend=false)
Plots.plot(p1, p2, layout=(2,1), size=(1000, 800))
Plots.savefig(joinpath(plot_dir, "training_metrics.png"))

println("[SUCCESS] Training results and logs saved to: $run_dir")