# MilliHertzQML

MilliHertzQML is a Julia pipeline for the detection of massive black hole binary (MBHB) coalescences in simulated LISA telemetry using a variational quantum classifier (VQC) with data re-uploading. Quantum circuits are simulated with `Yao.jl`; training uses `Zygote.jl` automatic differentiation and `Flux.jl` optimisers.

On the LISA Data Challenge 2a "Sangria" blind year, the eight-qubit model
(`configs/experiments/q8_b6.toml`: 8 qubits, 4 re-uploading layers, six
sub-mHz band powers, run `q8_b6_pi`) detects **all five labelled MBHB
events at 1.57 false-alarm episodes per 30 mission days**, from a decision
threshold fitted on the pooled held-out block of the *training* year —
validation and test together, 110 days — and applied without adjustment;
the fit predicted 1.38. The configuration and seed were chosen on the
validation block of the training year by a selection rule fixed before the
blind year was scored, and the [benchmark page](benchmark.md)
reports every run of the grid beside it.
The spread under re-initialisation alone ranges from 1.57 to 8.74 per 30
days across four seeds of the same configuration, and the single seed of
every other configuration lies inside that spread on the selection
statistic, so the ranking between configurations is not established. The
blind year is whitened by the full-record PSD, the median Welch estimate
of the entire blind year, which is available only after the whole record
has been received; this is admissible for a completed record and
non-causal for a streamed one.

![Classifier output over the Sangria blind year](assets/benchmark_mission_trace.png)

The pipeline comprises four stages, each a library function (`generate_telemetry`, `preprocess_record`, `train_classifier`, `evaluate_classifier`, plus `label_truth_stream` for LDC products) behind a thin script that takes the configuration file as its first argument (`julia scripts/<stage>.jl configs/default.toml [--run-id ID] ...`); the TOML file is the single source of every parameter and is validated on load:

1. `scripts/generate_data.jl` — simulates continuous milliHertz telemetry at physical strain amplitude (Robson–Cornish–Liu noise [RobsonCornishLiu2019](@cite), resolvable galactic binaries and EMRIs, IMRPhenomA MBHB injections [AjithEtAl2008](@cite) at a prescribed matched-filter SNR), written to HDF5 with point-wise labels and an event catalogue. For an LDC product, `scripts/label_ldc.jl` derives the point-wise labels from the truth stream instead.
2. `scripts/preprocess_ldc.jl` — extracts a spectral feature vector (four features by default, one per band plus two under `feature_set = "bands"`) per sliding window of the A channel, whitened by the strain model, the LDC TDI noise model, or a Welch estimate of the record, and writes the window-geometry sidecar.
3. `scripts/train.jl` — splits the windows chronologically into training, validation, and test blocks; trains the VQC with Adam, exponential learning-rate decay, and early stopping on the validation block; fits the decision threshold on the calibration block (`threshold_block`, the validation block by default); scores the test block once at window and event level.
4. `scripts/infer.jl` — applies the persisted threshold to a feature table (or to one block of the training table), reports window- and event-level metrics when labels are present, and produces diagnostic figures.

Scripts resolve relative paths against the project root and may be invoked from any working directory; RNG seeds come from the configuration. Every artifact carries git and hardware provenance, existing files are backed up rather than overwritten, and each stage checks its memory estimate against `[resources]` before allocating (see [Quantum Architecture](architecture.md), "Pipeline architecture").

See [Physics & Data](physics.md) for the simulation and feature models (including known deficiencies), [Quantum Architecture](architecture.md) for the circuit and training design, [Telemetry Coupling](telemetry.md) for the payload export and the streaming replay, [Sangria Benchmark](benchmark.md) for the results on the LDC blind year, and the [API Reference](api.md) for docstrings.
