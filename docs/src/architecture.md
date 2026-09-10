# Quantum Architecture & Training

## Circuit

The classifier is a four-qubit VQC with data re-uploading: the encoding block is interleaved before every trainable layer rather than applied once at state preparation.

- **Encoding block:** Hadamard followed by ``R_z(x_i)`` on qubit ``i``, where ``x_i \in [0, 2\pi]`` is the ``i``-th normalized feature. The number of qubits equals the feature dimension (four).
- **Trainable layer:** hardware-efficient ansatz — ``R_y`` and ``R_z`` rotations on every qubit followed by a ring of CNOT gates. Each layer holds ``2 n_\mathrm{qubits}`` parameters; the default configuration uses four layers.
- **Measurement:** the Pauli-``Z`` expectation is averaged over all qubits and mapped to a class probability ``p = (1 - \langle Z \rangle)/2``.

Parameters are initialized from a zero-mean normal distribution with standard deviation 0.5.

## Evaluation protocol

Sliding windows overlap by 90 %, so neighbouring windows are nearly identical and any random split would place copies of the same signal on both sides. `scripts/train.jl` therefore partitions the chronologically ordered windows into three contiguous blocks — training, validation, and test, holding `train_fraction`, `validation_fraction`, and the remainder of the windows (defaults 0.7, 0.15, 0.15) — separated by a buffer of one window length (`window_size / step_size` windows) so that no window straddles two blocks (`chronological_split`). The block ranges are written to `split.toml` in the run directory; the feature scaler is fitted on the training block alone.

- The validation block drives early stopping and the decision threshold; the test block is scored once, after training, with the fitted threshold.
- Window labels are positive whenever a labeled sample falls inside the window (a few per cent to a few tens of per cent of the windows, depending on the label span). The binary cross-entropy weights the positive term by the negative-to-positive count ratio of the training block (`class_weight = "balanced"`) unless disabled.
- Metrics are reported at window level (precision, recall, ``F_1``, balanced accuracy, ROC AUC) and at event level (`event_metrics`): an event is a contiguous run of positive labels and counts as detected when any window inside it is alarmed; a false-alarm episode is a contiguous run of alarmed windows outside the labeled spans; the operational false-alarm rate is the number of such episodes per 30 mission days of the evaluated block.

## Training

- Class-weighted binary cross-entropy over mini-batches of the training block, differentiated end-to-end with `Zygote.jl`. The forward pass in `loss_function` is functional: parameter slices are dispatched into freshly constructed blocks so that no global circuit state is mutated under the AD tracer.
- `Flux.Adam` with exponential learning-rate decay and patience-based early stopping on the validation loss; the best model is persisted per run as a JLD2 artifact holding the parameter vector, hyperparameters, the feature scaler, and run metadata (the circuit is rebuilt on load).
- Quantum registers use `ComplexF32` to match the `Float32` parameter vector.

## Decision threshold

After training, `select_threshold` fits the threshold on the validation block by the `threshold_criterion` of `[training]`:

- `far` (default): the lowest threshold — hence the highest recall — whose false-alarm episode rate does not exceed `target_far_per_30d`. On short validation blocks a single episode already exceeds one per 30 days, so the criterion is conservative there; it is meant for mission-scale records.
- `fpr`: the lowest threshold whose window-level false-positive rate does not exceed `target_fpr`.
- `youden`: the maximizer of ``\mathrm{TPR} - \mathrm{FPR}``; it needs both classes in the validation block and falls back to `fpr` with a warning otherwise.

Candidates are the quantiles of the validation scores. The threshold, the criterion actually applied, the validation rates at the threshold, and the validation AUC are persisted as `threshold.toml` next to the model; `metrics.toml` holds the validation- and test-block metrics of the run.

## Inference

`scripts/infer.jl` loads the model, the scaler, and the persisted threshold, scores a feature table, and writes per-window probabilities and decisions. The window geometry comes from the sidecar `<features>.toml` written by `scripts/preprocess_ldc.jl` (CLI overrides exist for tables without one). With labels it reports the window- and event-level metrics of the evaluated rows to `metrics.toml`; `--block validation` or `--block test` restricts the evaluation to one block of the training table through the run's `split.toml`. Blind inference (`--labels ""`) needs no labels. Figures: the mission trace with the threshold, the ROC curve, detection sensitivity versus matched-filter SNR, and the score distributions.

## Known Methodological Deficiencies

Retained here so the documentation reflects the code as it stands; remediation is planned.

1. **Configuration coverage.** Model, optimizer, feature-band, scaler, split, and threshold parameters are exposed in `config.toml` and validated on load; memory-safety thresholds and a pre-run register-size estimate are not yet enforced.
2. **Single evaluation record.** The blocks are cut from one simulated record, so the test block carries the events of one realization; event-level statistics on the Sangria data set are pending.
