# MilliHertzQML

A variational quantum classifier (VQC) with data re-uploading for the detection of massive black hole binary (MBHB) coalescences in simulated LISA telemetry. The quantum circuit is simulated with `Yao.jl`; optimization uses `Zygote.jl` gradients and `Flux.jl` optimizers. The pipeline follows the classification approach of Isfan et al., *Class. Quantum Grav.* (2025), DOI: 10.1088/1361-6382/ae1787, replacing the original Python/PennyLane prototype with a Julia implementation.

## File Structure

```text
MilliHertzQML/
├── src/
│   ├── MilliHertzQML.jl        # Module definition and exports
│   ├── model.jl            # VQC struct, ansatz and feature-map construction
│   ├── training.jl         # Forward pass, BCE loss, gradient step
│   ├── data.jl             # Feature extraction and normalization
│   └── persistence.jl      # JLD2 model save/load (parameters + hyperparameters)
├── scripts/
│   ├── generate_data.jl    # Simulated continuous LISA telemetry (HDF5 + labels)
│   ├── preprocess_ldc.jl   # Sliding-window feature extraction (HDF5 -> CSV)
│   ├── train.jl            # Training loop with early stopping and terminal dashboard
│   └── infer.jl            # Inference, ROC thresholding, diagnostic figures
├── test/
│   └── runtests.jl         # Unit tests (model, gradients, features, normalization)
├── benchmarks/
│   └── benchmarks.jl       # BenchmarkTools performance measurements
├── docs/                   # Documenter.jl sources (build/ is generated, not tracked)
├── data/
│   ├── inputs/             # Generated telemetry and feature CSVs (not tracked)
│   └── outputs/            # Per-run plots and results (not tracked)
├── models/                 # Per-run model checkpoints (not tracked)
├── config.toml             # Pipeline defaults; overridden by CLI flags
├── Project.toml            # Package manifest
└── Manifest.toml           # Pinned dependency versions (tracked)
```

## Environment

Julia ≥ 1.12. The manifest is authoritative:

```bash
julia --project=MilliHertzQML -e 'using Pkg; Pkg.instantiate()'
```

## Usage

Scripts resolve relative paths against the project root and may be invoked from any working directory. Configuration defaults come from `config.toml`; CLI flags override them. RNG seeds are set from the configuration. Each run is assigned a run identifier under which models (JLD2), plots, logs, and a configuration snapshot are stored.

```bash
# 1. Simulate continuous telemetry (HDF5 strain + point-wise label CSV)
julia MilliHertzQML/scripts/generate_data.jl --days 30.0

# 2. Sliding-window feature extraction
julia MilliHertzQML/scripts/preprocess_ldc.jl \
    --h5-file MilliHertzQML/data/inputs/simulated_telemetry.h5 \
    --label-file MilliHertzQML/data/inputs/simulated_telemetry_labels.csv \
    --output-prefix telemetry_sim

# 3. Training (Adam, exponential learning-rate decay, early stopping)
julia MilliHertzQML/scripts/train.jl \
    --train-features MilliHertzQML/data/inputs/telemetry_sim_features.csv \
    --train-labels MilliHertzQML/data/inputs/telemetry_sim_labels.csv --epochs 50

# 4. Inference and diagnostics (ROC, mission trace, sensitivity, score distributions)
julia MilliHertzQML/scripts/infer.jl \
    --features MilliHertzQML/data/inputs/telemetry_sim_features.csv \
    --labels MilliHertzQML/data/inputs/telemetry_sim_labels.csv --run-id <RUN_ID>
```

`--test-mode` restricts training to 5000 samples and 20 epochs for rapid validation.

Labeled inference fits the decision threshold from the ROC curve and persists it as `threshold.toml` next to the model. Blind inference (`--labels ""` or `--use-real-data`) requires that persisted threshold and produces per-window scores and decisions without labels.

## Testing and Benchmarks

```bash
julia MilliHertzQML/test/runtests.jl        # unit tests
julia MilliHertzQML/benchmarks/benchmarks.jl
```

Documentation builds with Documenter.jl:

```bash
julia --project=MilliHertzQML/docs MilliHertzQML/docs/make.jl
# open MilliHertzQML/docs/build/index.html
```

## Component Status

| Component | State |
|---|---|
| Core library (`src/`) | Functional; unit tests pass |
| Telemetry simulator | Functional and seeded, with known physics defects (confusion-noise spectrum, injection SNR definition, chirp aliasing near merger, injection/label alignment for early merger times) |
| Feature extraction | Functional on simulator output only; fixed normalization scales are incompatible with physical-strain-amplitude LDC data |
| Training script | Runs end-to-end (verified); evaluation protocol still leaks information (random split over overlapping windows, validation set reused as test set) |
| Inference script | Runs with labeled and blind data; threshold persisted per run; threshold is still fitted on the evaluated dataset |
| Documentation | Tracks the current state; remediation of the remaining defects is planned |

The defects listed here are documented in detail, together with the remediation plan, in the workspace notes (outside this repository).
