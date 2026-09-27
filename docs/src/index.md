# MilliHertzQML

MilliHertzQML is a Julia pipeline for the detection of massive black hole binary (MBHB) coalescences in simulated LISA telemetry using a variational quantum classifier (VQC) with data re-uploading. Quantum circuits are simulated with `Yao.jl`; training uses `Zygote.jl` automatic differentiation and `Flux.jl` optimisers.

The pipeline comprises four stages, each a library function (`generate_telemetry`, `preprocess_record`, `train_classifier`, `evaluate_classifier`, plus `label_truth_stream` for LDC products) behind a thin script that takes the configuration file as its first argument (`julia scripts/<stage>.jl configs/default.toml [--run-id ID] ...`); the TOML file is the single source of every parameter and is validated on load:

1. `scripts/generate_data.jl` — simulates continuous milliHertz telemetry at physical strain amplitude (Robson–Cornish–Liu noise [RobsonCornishLiu2019](@cite), resolvable galactic binaries and EMRIs, IMRPhenomA MBHB injections [AjithEtAl2008](@cite) at a prescribed matched-filter SNR), written to HDF5 with point-wise labels and an event catalogue. For an LDC product, `scripts/label_ldc.jl` derives the point-wise labels from the truth stream instead.
2. `scripts/preprocess_ldc.jl` — extracts a four-dimensional spectral feature vector per sliding window of the A channel, whitened by the strain model, the LDC TDI noise model, or a Welch estimate of the record, and writes the window-geometry sidecar.
3. `scripts/train.jl` — splits the windows chronologically into training, validation, and test blocks; trains the VQC with Adam, exponential learning-rate decay, and early stopping on the validation block; fits the decision threshold on the calibration block (`threshold_block`, the validation block by default); scores the test block once at window and event level.
4. `scripts/infer.jl` — applies the persisted threshold to a feature table (or to one block of the training table), reports window- and event-level metrics when labels are present, and produces diagnostic figures.

Scripts resolve relative paths against the project root and may be invoked from any working directory; RNG seeds come from the configuration. Every artefact carries git and hardware provenance, existing files are backed up rather than overwritten, and each stage checks its memory estimate against `[resources]` before allocating (see [Quantum Architecture](architecture.md), "Pipeline architecture").

See [Physics & Data](physics.md) for the simulation and feature models (including known deficiencies), [Quantum Architecture](architecture.md) for the circuit and training design, and the [API Reference](api.md) for docstrings.
