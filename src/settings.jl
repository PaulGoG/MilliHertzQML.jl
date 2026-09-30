# Settings of the classifier: the circuit, its training, and the memory
# estimate of a training run.

"""
    model_settings(config) -> NamedTuple

Validated `[model]` parameters of the classifier.
"""
function model_settings(config::AbstractDict)
    m = section(config, "model")
    return (
        n_qubits = cfgget(m, "n_qubits", 4; type = Int, min = 2, max = 24),
        n_layers = cfgget(m, "n_layers", 4; type = Int, min = 1),
    )
end

"""
    training_settings(config) -> NamedTuple

Validated `[training]` parameters: inputs, optimiser, chronological blocks,
class weighting, threshold criterion and fitting block, scaler quantiles and
phase-encoding span, test-mode caps, threading.
"""
function training_settings(config::AbstractDict)
    t = section(config, "training")
    train_fraction =
        cfgget(t, "train_fraction", 0.7; type = Float64, min = 0.05, max = 0.95)
    validation_fraction =
        cfgget(t, "validation_fraction", 0.15; type = Float64, min = 0.01, max = 0.5)
    train_fraction + validation_fraction < 1 || throw(
        ArgumentError(
            "train_fraction + validation_fraction = $(train_fraction + validation_fraction); " *
            "must leave a test block.",
        ),
    )
    quantiles = cfgget(t, "scaler_quantiles", [0.005, 0.995]; type = AbstractVector)
    (
        length(quantiles) == 2 &&
        all(q -> q isa Real, quantiles) &&
        0 <= quantiles[1] < quantiles[2] <= 1
    ) || throw(
        ArgumentError(
            "configuration key `scaler_quantiles` = $(repr(quantiles)); " *
            "expected two ascending values in [0, 1].",
        ),
    )
    return (
        train_features = resolvepath(
            cfgget(t, "train_features", "data/inputs/train_features.csv"; type = String),
        ),
        train_labels = resolvepath(
            cfgget(t, "train_labels", "data/inputs/train_labels.csv"; type = String),
        ),
        epochs = cfgget(t, "epochs", 100; type = Int, min = 1),
        batch_size = cfgget(t, "batch_size", 32; type = Int, min = 1),
        learning_rate = cfgget(t, "learning_rate", 0.01; type = Float64, min = 1e-8),
        lr_decay = cfgget(t, "lr_decay", 0.95; type = Float64, min = 1e-3, max = 1.0),
        patience = cfgget(t, "patience", 12; type = Int, min = 1),
        train_fraction = train_fraction,
        validation_fraction = validation_fraction,
        class_weight = cfgget(
            t,
            "class_weight",
            "balanced";
            type = String,
            choices = ("balanced", "none"),
        ),
        threshold_criterion = cfgget(
            t,
            "threshold_criterion",
            "far";
            type = String,
            choices = ("far", "fpr", "youden"),
        ),
        threshold_block = cfgget(
            t,
            "threshold_block",
            "validation";
            type = String,
            choices = ("validation", "held_out"),
        ),
        target_far_per_30d = cfgget(
            t,
            "target_far_per_30d",
            3.0;
            type = Float64,
            min = 0.0,
        ),
        target_fpr = cfgget(t, "target_fpr", 0.05; type = Float64, min = 0.0, max = 1.0),
        min_fit_episodes = cfgget(t, "min_fit_episodes", 5; type = Int, min = 0),
        scaler_quantiles = (Float64(quantiles[1]), Float64(quantiles[2])),
        phase_span = cfgget(t, "phase_span", 1.0; type = Float64, min = 1e-3, max = 2.0),
        test_mode_samples = cfgget(t, "test_mode_samples", 5000; type = Int, min = 1),
        test_mode_epochs = cfgget(t, "test_mode_epochs", 20; type = Int, min = 1),
        threaded = cfgget(t, "threaded", true; type = Bool),
        seed = cfgget(t, "seed", 42; type = Int),
    )
end

"""
    training_memory_estimate_gib(n_qubits, n_layers, batch_size) -> Float64

Pre-flight estimate of the memory of one training step: the statevector of
``2^{n}`` complex single-precision amplitudes is copied at every gate
application under the automatic-differentiation tape, ``n_\\mathrm{qubits}
(2 + 2) + n_\\mathrm{qubits}`` gates per layer (feature map, rotations,
CNOT ring), for every sample of the batch, plus the same again for the
adjoint pass.
"""
function training_memory_estimate_gib(
    n_qubits::Integer,
    n_layers::Integer,
    batch_size::Integer,
)
    (n_qubits >= 1 && n_layers >= 1 && batch_size >= 1) ||
        throw(ArgumentError("n_qubits, n_layers, and batch_size must be positive."))
    statevector_bytes = 2.0^n_qubits * 8
    gates_per_layer = 5 * n_qubits
    return 2 * batch_size * n_layers * gates_per_layer * statevector_bytes / 2^30
end
