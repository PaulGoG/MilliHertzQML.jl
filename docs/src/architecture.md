# Quantum Architecture & Training

## Circuit

The classifier is a four-qubit VQC with data re-uploading: the encoding block is interleaved before every trainable layer rather than applied once at state preparation.

- **Encoding block:** Hadamard followed by ``R_z(x_i)`` on qubit ``i``, where ``x_i \in [0, 2\pi]`` is the ``i``-th normalized feature. The number of qubits equals the feature dimension (four).
- **Trainable layer:** hardware-efficient ansatz — ``R_y`` and ``R_z`` rotations on every qubit followed by a ring of CNOT gates. Each layer holds ``2 n_\mathrm{qubits}`` parameters; the default configuration uses four layers.
- **Measurement:** the Pauli-``Z`` expectation is averaged over all qubits and mapped to a class probability ``p = (1 - \langle Z \rangle)/2``.

Parameters are initialized from a zero-mean normal distribution with standard deviation 0.5.

## Training

- Binary cross-entropy loss over mini-batches, differentiated end-to-end with `Zygote.jl`. The forward pass in `loss_function` is functional: parameter slices are dispatched into freshly constructed blocks so that no global circuit state is mutated under the AD tracer.
- `Flux.Adam` with exponential learning-rate decay and patience-based early stopping; the best model by validation loss is persisted per run as a JLD2 artifact holding the parameter vector, hyperparameters, and run metadata (the circuit is rebuilt on load).
- Quantum registers use `ComplexF32` to match the `Float32` parameter vector.

## Inference and Thresholding

`scripts/infer.jl` computes classifier probabilities over all windows, the ROC curve, and its AUC (`EvalMetrics.jl`). The decision threshold is selected either by a target false-positive-rate constraint (`--target-fpr`) or by Youden's J statistic, and is persisted as `threshold.toml` next to the model artifact. Blind inference loads the persisted threshold and requires no labels.

## Known Methodological Deficiencies

Retained here so the documentation reflects the code as it stands; remediation is planned.

1. **Evaluation leakage.** The train/validation split is a random shuffle over sliding windows with 90 % overlap, so nearly identical windows appear on both sides of the split; the validation set drives early stopping and is also reported as the test result.
2. **Threshold selection.** The ROC-derived threshold is fitted on the same data on which accuracy is subsequently reported; it should be fitted on a held-out validation set instead.
3. **Class imbalance.** Positive windows are of order 10 % of the data; accuracy at a fixed threshold is reported without precision/recall or false-alarm-rate context.
4. **Configuration coverage.** Model and optimizer hyperparameters are exposed in `config.toml` (`[model]`, `[training]`) and validated on load, but the feature-extraction constants (analysis-band edges and the `FEATURE_SCALES` clamps in `src/data.jl`) remain hardcoded.
