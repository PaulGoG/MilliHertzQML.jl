# Quantum Architecture & Training

## Circuit

The classifier is a four-qubit VQC with data re-uploading: the encoding block is interleaved before every trainable layer rather than applied once at state preparation.

- **Encoding block:** Hadamard followed by ``R_z(x_i)`` on qubit ``i``, where ``x_i \in [0, 2\pi]`` is the ``i``-th normalized feature. The number of qubits equals the feature dimension (four).
- **Trainable layer:** hardware-efficient ansatz — ``R_y`` and ``R_z`` rotations on every qubit followed by a ring of CNOT gates. Each layer holds ``2 n_\mathrm{qubits}`` parameters; the default configuration uses four layers.
- **Measurement:** the Pauli-``Z`` expectation is averaged over all qubits and mapped to a class probability ``p = (1 - \langle Z \rangle)/2``.

Parameters are initialized from a zero-mean normal distribution with standard deviation 0.5.

## Evaluation protocol

Sliding windows overlap by 90 %, so neighbouring windows are nearly identical and any random split would place copies of the same signal on both sides. The training stage therefore partitions the chronologically ordered windows into three contiguous blocks — training, validation, and test, holding `train_fraction`, `validation_fraction`, and the remainder of the windows (defaults 0.7, 0.15, 0.15) — separated by a buffer of one window length (`window_size / step_size` windows) so that no window straddles two blocks (`chronological_split`). The block ranges are written to `split.toml` in the run directory; the feature scaler is fitted on the training block alone.

- The validation block drives early stopping and the decision threshold; the test block is scored once, after training, with the fitted threshold.
- Window labels are positive whenever a labeled sample falls inside the window (a few per cent to a few tens of per cent of the windows, depending on the label span). The binary cross-entropy weights the positive term by the negative-to-positive count ratio of the training block (`class_weight = "balanced"`) unless disabled.
- Metrics are reported at window level (precision, recall, ``F_1``, balanced accuracy, ROC AUC) and at event level (`event_metrics`): an event is a contiguous run of positive labels and counts as detected when any window inside it is alarmed; a false-alarm episode is a contiguous run of alarmed windows outside the labeled spans; the operational false-alarm rate is the number of such episodes per 30 mission days of the evaluated block.

## Training

- Class-weighted binary cross-entropy over mini-batches of the training block, differentiated end-to-end with `Zygote.jl`. The forward pass in `loss_function` is functional: parameter slices are dispatched into freshly constructed blocks so that no global circuit state is mutated under the AD tracer.
- `Flux.Adam` with exponential learning-rate decay and patience-based early stopping on the validation loss; the best model is persisted per run as a JLD2 artifact holding the parameter vector, hyperparameters, the feature scaler, and run metadata (the circuit is rebuilt on load).
- Quantum registers use `ComplexF32` to match the `Float32` parameter vector.

## Decision threshold

After training, `threshold_sweep` evaluates the window- and event-level statistics of the validation block at every candidate threshold (the quantiles of the validation scores) — the event-level operating characteristic of the block, persisted as `threshold_sweep.csv` and drawn as the `threshold_sweep` figure (event and window recall, and false-alarm episodes per 30 days, against the threshold). `select_threshold` then fits the threshold on it by the `threshold_criterion` of `[training]`:

- `far` (default): the operating point of an alert trigger. A candidate is admissible when its false-alarm episode rate does not exceed `target_far_per_30d` and its window false-positive rate — the alarm duty cycle on unlabeled windows — does not exceed `target_fpr`. Candidates are scanned from the highest downwards and the threshold is the lowest candidate of the admissible range that starts at the top, i.e. the highest recall reachable while alarms remain short and isolated. The scan direction matters because the episode count is not monotone in the threshold: as the threshold falls, spurious episodes first multiply and then merge into a permanently raised alarm charged with only a few long episodes, which an ascending scan would accept as soon as the target admits a few episodes per month. The default target of three episodes per 30 days is the trigger-level policy: a trigger costs a characterization pass of the low-latency chain, not an alert, and the observed rate on a block of ``T`` days is a Poisson count of mean ``3T/30``, so the block must span a couple of months for the fit to be meaningful — on the eight-day sanity records the criterion degenerates to the duty-cycle guard.
- `fpr`: the lowest threshold whose window-level false-positive rate does not exceed `target_fpr`.
- `youden`: the maximizer of ``\mathrm{TPR} - \mathrm{FPR}``; it needs both classes in the validation block and falls back to `fpr` with a warning otherwise.

The threshold, the criterion actually applied, the targets, the validation rates at the threshold, and the validation AUC are persisted as `threshold.toml` next to the model; `metrics.toml` holds the validation- and test-block metrics of the run.

## Inference

The inference stage loads the model, the scaler, and the persisted threshold, scores a feature table, and writes per-window probabilities and decisions. The window geometry comes from the sidecar `<features>.toml` written by the pre-processor. With labels it reports the window- and event-level metrics of the evaluated rows to `metrics.toml` and their operating characteristic to `threshold_sweep.csv` — a post-hoc diagnostic of where the persisted threshold sits on the evaluated data; the threshold itself is never refitted at inference. `--block validation` or `--block test` restricts the evaluation to one block of the training table through the run's `split.toml`. Blind inference (`--labels ""`) needs no labels. Figures: the mission trace with the threshold, the ROC curve, the operating characteristic with the applied threshold, detection sensitivity versus matched-filter SNR, and the score distributions.

## Pipeline architecture

The pipeline is a library with thin command-line entry points. Every stage is a typed, documented function of the package taking the parsed TOML configuration and returning a named tuple of its artifacts and results, so that it can be called from a script, a test, or another package alike:

| Stage | Function | Script |
|---|---|---|
| Telemetry simulation | `generate_telemetry(config; run_id, output)` | `scripts/generate_data.jl` |
| Truth-stream labels (LDC) | `label_truth_stream(config; h5_file, truth_csv, output_prefix)` | `scripts/label_ldc.jl` |
| Window features | `preprocess_record(config; h5_file, tdi_group, label_file, output_prefix, force)` | `scripts/preprocess_ldc.jl` |
| Training | `train_classifier(config; run_id, test_mode, on_epoch)` | `scripts/train.jl` |
| Inference | `evaluate_classifier(config; run_id, model, features, labels, block)` | `scripts/infer.jl` |
| Payload export (telemetry) | `export_telemetry_payload(config; h5_file, tdi_group, catalog, output_prefix)` | `scripts/export_telemetry_payload.jl` |
| Telemetry replay | `replay_run` / `follow_run` on `open_telemetry_run(run_dir)` with `detector_from_run(model)` | `scripts/infer_telemetry.jl` |

- **Configuration.** The TOML file is the single source of truth (`src/config.jl`): each section is read through a settings function (`generation_settings`, `preprocessing_settings`, `model_settings`, `training_settings`, `inference_settings`, `ldc_settings`, `resource_settings`) that validates types, bounds, and enumerated choices on load and fails with the offending key. The scripts add only a run identifier, a test-mode switch, and the location of external inputs (`julia scripts/<stage>.jl [config.toml] [--run-id ID] ...`); relative paths resolve against the package root whichever environment is active.
- **Provenance.** Every snapshot written by a stage (`write_toml`) carries the hardware fingerprint, the git description of the package tree with its dirty flag and the package version, and the wall-clock time; model artifacts carry the same in their metadata. Existing files are never overwritten: `write_toml` and `write_csv` move a previous file to `<stem>_#k<ext>` first, the `safesave` convention of DrWatson. Feature products record a hash of every parameter that determines them, and preprocessing reuses an identical product unless forced.
- **Resource guard.** `[resources]` holds `max_memory_gib` and `warn_memory_gib`. Before allocating, a stage estimates its memory — for training, the ``2^{n}`` complex single-precision statevector copied per gate application under the automatic-differentiation tape for every sample of the batch and every layer, forward and adjoint (`training_memory_estimate_gib`); for record processing, a few record-length arrays (`record_memory_estimate_gib`) — and refuses to start above the maximum or warns above the warning level (`check_memory`).
- **Timing.** Stages accumulate wall time and allocations in a package-wide `TimerOutput`; the scripts print its table at the end (`report_timing`).
- **Figures.** The figure functions are declared in the package (`src/visualization.jl`) and implemented by the CairoMakie extension, so the core library carries no plotting dependency. One theme serves every figure: the 86 mm single-column width in typographic points, Computer Modern through MathTeXEngine, boxed axes with inward ticks and a faint dashed grid, no titles, a horizontal legend above the axes, series families in Okabe–Ito colors with roles in line style (data solid, thresholds dashed, chance dotted), and an axis offset multiplier for strain amplitudes. `save_figure` writes the vector PDF, a 4× PNG, and a provenance sidecar per figure, backing up earlier exports.

## Known Methodological Deficiencies

Retained here so the documentation reflects the code as it stands; remediation is planned.

1. **Single evaluation record.** The blocks are cut from one record, so the test block carries the events of one realization; the Sangria blind set is the independent evaluation.
2. **Serial training loop.** Gradients are evaluated sample by sample on one thread; mission-scale training runs for hours on a workstation. Threaded batch gradients are deferred until the benchmark demands them.
