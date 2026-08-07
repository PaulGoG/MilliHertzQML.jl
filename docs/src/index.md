# MilliHertzQML

MilliHertzQML is a Julia pipeline for the detection of massive black hole binary (MBHB) coalescences in simulated LISA telemetry using a variational quantum classifier (VQC) with data re-uploading. Quantum circuits are simulated with `Yao.jl`; training uses `Zygote.jl` automatic differentiation and `Flux.jl` optimizers.

The pipeline comprises four stages, each an executable script driven by `config.toml` with CLI overrides:

1. `scripts/generate_data.jl` — simulates continuous milliHertz telemetry with a galactic-binary/EMRI background and injected MBHB waveforms, written to HDF5 with point-wise labels.
2. `scripts/preprocess_ldc.jl` — extracts a four-dimensional spectral feature vector per sliding window.
3. `scripts/train.jl` — trains the VQC with Adam, exponential learning-rate decay, and early stopping.
4. `scripts/infer.jl` — evaluates the classifier, selects a decision threshold from the ROC curve, and produces diagnostic figures.

Scripts resolve relative paths against the project root and may be invoked from any working directory; RNG seeds come from the configuration.

See [Physics & Data](physics.md) for the simulation and feature models (including known deficiencies), [Quantum Architecture](architecture.md) for the circuit and training design, and the [API Reference](api.md) for docstrings.
