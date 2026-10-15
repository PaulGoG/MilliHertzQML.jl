# Settings of the classifier: the circuit, its training, and the memory
# estimate of a training run.

"""
    model_settings(config) -> NamedTuple

Validated `[model]` parameters of the classifier: the register size and
depth of the circuit, the `kind` of classifier ([`MODEL_KINDS`](@ref)) and
the `hidden_units` of the `"perceptron"`; `n_qubits` is the feature
dimension of every kind.
"""
function model_settings(config::AbstractDict)
    m = section(config, "model")
    return (
        n_qubits = cfgget(m, "n_qubits", 4; type = Int, min = 2, max = 24),
        n_layers = cfgget(m, "n_layers", 4; type = Int, min = 1),
        kind = cfgget(m, "kind", "circuit"; type = String, choices = MODEL_KINDS),
        hidden_units = cfgget(m, "hidden_units", 6; type = Int, min = 1),
    )
end

"""
    training_settings(config) -> NamedTuple

Validated `[training]` parameters: inputs, optimiser, chronological blocks,
class weighting, threshold criterion and fitting block, scaler quantiles and
phase-encoding span, test-mode caps, threading, gradient method.
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
        gradient_method = cfgget(
            t,
            "gradient_method",
            "adjoint";
            type = String,
            choices = ("adjoint", "zygote"),
        ),
        seed = cfgget(t, "seed", 42; type = Int),
    )
end

"""
    CALIBRATION_SCALERS

The scaler rules of the calibration stage: `"match"` encodes the record's
noise where the training noise was by its quantiles, `"refit"` applies the
training rule — `[training] scaler_quantiles` on the calibration block —
and `"keep"` keeps the parent model's scaler and refits the threshold only.
"""
const CALIBRATION_SCALERS = ("match", "refit", "keep")

"""
    calibration_settings(config) -> NamedTuple

Validated `[calibration]` parameters of the calibration of a trained
classifier on a new record ([`calibrate_classifier`](@ref)): the
chronological `block_fraction` of the record's windows that fits the
scaler and the threshold (in [0.05, 0.95], default 0.3), the `scaler` rule
(`"match"`, `"refit"`, `"keep"`; [`CALIBRATION_SCALERS`](@ref)), the
`noise_quantiles` matched under `"match"` (two ascending values in [0, 1],
default `[0.5, 0.99]`), the reference product of the noise the parent
model was trained on, `reference_features` and `reference_labels` (empty,
the default: `[training] train_features` and `train_labels`), and
`tail_candidates` (>= 0, default 200), the candidate thresholds added to
the upper tail of the calibration block's scores
([`threshold_sweep`](@ref)), without which a block without signal cannot
reach a false-alarm target below the quantile step of the grid. The
threshold criterion, its targets, `min_fit_episodes`, `scaler_quantiles`
and `phase_span` are those of `[training]`.
"""
function calibration_settings(config::AbstractDict)
    c = section(config, "calibration")
    quantiles = cfgget(c, "noise_quantiles", [0.5, 0.99]; type = AbstractVector)
    (
        length(quantiles) == 2 &&
        all(q -> q isa Real, quantiles) &&
        0 <= quantiles[1] < quantiles[2] <= 1
    ) || throw(
        ArgumentError(
            "configuration key `noise_quantiles` = $(repr(quantiles)); " *
            "expected two ascending values in [0, 1].",
        ),
    )
    reference_features = cfgget(c, "reference_features", ""; type = String)
    reference_labels = cfgget(c, "reference_labels", ""; type = String)
    return (
        block_fraction = cfgget(
            c,
            "block_fraction",
            0.3;
            type = Float64,
            min = 0.05,
            max = 0.95,
        ),
        scaler = cfgget(c, "scaler", "match"; type = String, choices = CALIBRATION_SCALERS),
        noise_quantiles = (Float64(quantiles[1]), Float64(quantiles[2])),
        reference_features = isempty(reference_features) ? "" :
                             resolvepath(reference_features),
        reference_labels = isempty(reference_labels) ? "" : resolvepath(reference_labels),
        tail_candidates = cfgget(c, "tail_candidates", 200; type = Int, min = 0),
    )
end

"""
    training_memory_estimate_gib(n_qubits, n_layers, batch_size;
                                 gradient_method = "adjoint") -> Float64

Pre-flight estimate of the memory of one training step. A statevector
holds ``2^{n}`` complex single-precision amplitudes.

- `"adjoint"`: every task owns a workspace of two registers, whatever the
  depth, and there is at most one task per chunk of
  [`GRADIENT_CHUNK`](@ref) samples.
- `"zygote"`: the statevector is copied at every gate application under
  the tape, ``n_\\mathrm{qubits} (2 + 2) + n_\\mathrm{qubits}`` gates per
  layer (feature map, rotations, CNOT ring), for every sample of the batch,
  plus the same again for the reverse pass.
"""
function training_memory_estimate_gib(
    n_qubits::Integer,
    n_layers::Integer,
    batch_size::Integer;
    gradient_method::AbstractString = "adjoint",
)
    (n_qubits >= 1 && n_layers >= 1 && batch_size >= 1) ||
        throw(ArgumentError("n_qubits, n_layers, and batch_size must be positive."))
    gradient_method in ("adjoint", "zygote") || throw(
        ArgumentError(
            "gradient_method = \"$gradient_method\"; expected \"adjoint\" or \"zygote\".",
        ),
    )
    statevector_bytes = 2.0^n_qubits * 8
    gradient_method == "adjoint" &&
        return 2 * cld(batch_size, GRADIENT_CHUNK) * statevector_bytes / 2^30
    gates_per_layer = 5 * n_qubits
    return 2 * batch_size * n_layers * gates_per_layer * statevector_bytes / 2^30
end
