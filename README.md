# MilliHertzQML.jl

[![CI](https://github.com/PaulGoG/MilliHertzQML.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/PaulGoG/MilliHertzQML.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/PaulGoG/MilliHertzQML.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PaulGoG/MilliHertzQML.jl)

Quantum machine learning for gravitational-wave detection in the milliHertz band. A variational quantum classifier (VQC) with data re-uploading detects massive black hole binary (MBHB) coalescences in simulated LISA-like telemetry. Quantum circuits are simulated with `Yao.jl`; optimization uses `Zygote.jl` gradients and `Flux.jl` optimizers. The classification approach follows Isfan et al., *Class. Quantum Grav.* (2025), DOI: 10.1088/1361-6382/ae1787, replacing the original Python/PennyLane prototype with a Julia implementation.

## File Structure

```text
MilliHertzQML/
├── src/
│   ├── MilliHertzQML.jl    # Module definition and exports
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
│   └── runtests.jl         # Unit tests (model, gradients, features, validation, persistence)
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

## Installation

Julia ≥ 1.12 is required. From a clone of this repository:

```bash
git clone git@github.com:PaulGoG/MilliHertzQML.jl.git
cd MilliHertzQML.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The tracked `Manifest.toml` pins the exact dependency versions.

## Usage

All commands below run from the repository root; scripts resolve relative paths against the project root and may equally be invoked from any working directory. Configuration defaults come from `config.toml`; CLI flags override them. RNG seeds are set from the configuration. Each run is assigned a run identifier under which models (JLD2), plots, logs, and a configuration snapshot are stored.

```bash
# 1. Simulate continuous telemetry (HDF5 strain + point-wise label CSV)
julia scripts/generate_data.jl --days 30.0

# 2. Sliding-window feature extraction
julia scripts/preprocess_ldc.jl \
    --h5-file data/inputs/simulated_telemetry_complex.h5 \
    --label-file data/inputs/simulated_telemetry_complex_labels.csv \
    --output-prefix telemetry_sim

# 3. Training (Adam, exponential learning-rate decay, early stopping)
julia scripts/train.jl \
    --train-features data/inputs/telemetry_sim_features.csv \
    --train-labels data/inputs/telemetry_sim_labels.csv --epochs 50

# 4. Inference and diagnostics (ROC, mission trace, sensitivity, score distributions)
julia scripts/infer.jl \
    --features data/inputs/telemetry_sim_features.csv \
    --labels data/inputs/telemetry_sim_labels.csv --run-id <RUN_ID>
```

`--test-mode` restricts training to 5000 samples and 20 epochs for rapid validation.

Labeled inference fits the decision threshold from the ROC curve and persists it as `threshold.toml` next to the model. Blind inference (`--labels ""` or `--use-real-data`) requires that persisted threshold and produces per-window scores and decisions without labels.

## Testing and Benchmarks

```bash
julia --project -e 'using Pkg; Pkg.test()'   # unit tests
julia benchmarks/benchmarks.jl               # performance measurements
```

Documentation builds with Documenter.jl:

```bash
julia --project=docs docs/make.jl
# open docs/build/index.html
```

## Component Status

| Component | State |
|---|---|
| Core library (`src/`) | Functional; unit tests pass; fail-fast input validation on public interfaces |
| Telemetry simulator | Functional and seeded, with known physics defects (confusion-noise spectrum, injection SNR definition, chirp aliasing near merger, injection/label alignment for early merger times) |
| Feature extraction | Functional on simulator output only; fixed normalization scales are incompatible with physical-strain-amplitude LDC data |
| Training script | Runs end-to-end (verified); evaluation protocol still leaks information (random split over overlapping windows, validation set reused as test set) |
| Inference script | Runs with labeled and blind data; threshold persisted per run; threshold is still fitted on the evaluated dataset |
| Documentation | Tracks the current state; remediation of the remaining defects is planned |

Version 0.1.x is a pre-release: the physics defects listed above are documented in `docs/src/physics.md` and scheduled for remediation before any science use.

## License

MIT — see `LICENSE`.
