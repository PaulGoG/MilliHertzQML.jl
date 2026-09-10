# MilliHertzQML.jl

[![CI](https://github.com/PaulGoG/MilliHertzQML.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/PaulGoG/MilliHertzQML.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/PaulGoG/MilliHertzQML.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PaulGoG/MilliHertzQML.jl)

Quantum machine learning for gravitational-wave detection in the milliHertz band. A variational quantum classifier (VQC) with data re-uploading detects massive black hole binary (MBHB) coalescences in simulated LISA-like telemetry. Quantum circuits are simulated with `Yao.jl`; optimization uses `Zygote.jl` gradients and `Flux.jl` optimizers. The classification approach follows Isfan et al., *Class. Quantum Grav.* **42** 225001 (2025), DOI: 10.1088/1361-6382/ae1787, replacing the original Python/Qiskit implementation with a Julia one.

## File Structure

```text
MilliHertzQML/
├── src/
│   ├── MilliHertzQML.jl    # Module definition and exports
│   ├── model.jl            # VQC struct, ansatz and feature-map construction
│   ├── training.jl         # Forward pass, BCE loss, gradient step
│   ├── simulation.jl       # Noise model (Robson–Cornish–Liu 2019), synthesis, matched-filter SNR, whitening
│   ├── data.jl             # Whitened window features, train-fitted feature scaler, CSV loading
│   └── persistence.jl      # JLD2 model save/load (parameters, hyperparameters, feature scaler)
├── scripts/
│   ├── Project.toml        # Script environment (package consumed by path); Manifest committed
│   ├── common.jl           # Shared preamble: activation, paths, validated config access
│   ├── generate_data.jl    # Simulated continuous LISA telemetry (HDF5 + labels + event catalog)
│   ├── preprocess_ldc.jl   # Sliding-window whitened feature extraction (HDF5 -> CSV)
│   ├── train.jl            # Training loop with early stopping and terminal dashboard
│   └── infer.jl            # Inference, ROC thresholding, diagnostic figures
├── test/
│   ├── Project.toml        # Test environment (package consumed by path); Manifest committed
│   └── runtests.jl         # Static QA (Aqua, JET, ExplicitImports), unit tests, pipeline smoke test
├── bench/
│   ├── Project.toml        # Benchmark environment (package consumed by path); Manifest committed
│   └── benchmarks.jl       # BenchmarkTools performance measurements
├── docs/
│   ├── Project.toml        # Documentation environment (package consumed by path)
│   ├── make.jl             # Documenter.jl build script
│   └── src/                # Manual pages (build/ is generated, not tracked)
├── data/
│   ├── inputs/             # Generated telemetry and feature CSVs (not tracked)
│   └── outputs/            # Per-run plots and results (not tracked)
├── models/                 # Per-run model checkpoints (not tracked)
├── .github/workflows/CI.yml # Test matrix, formatting check, documentation build
├── .JuliaFormatter.toml    # Committed formatter configuration
├── CHANGELOG.md            # Notable changes (Keep a Changelog format)
├── CITATION.cff            # Citation metadata
├── config.toml             # Pipeline defaults; overridden by CLI flags
├── Project.toml            # Package metadata: only the dependencies of src/
└── Manifest.toml           # Pinned dependency versions (tracked)
```

```

## Installation

Julia 1.12 is the supported release: the committed `Manifest.toml` files are
resolved on it and it is the `[compat]` floor. Newer releases are exercised
by an advisory CI job only. With [juliaup](https://github.com/JuliaLang/juliaup),
`juliaup add 1.12` installs it and `julia +1.12` selects it. From a clone
of this repository:

```bash
git clone git@github.com:PaulGoG/MilliHertzQML.jl.git
cd MilliHertzQML.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The scripts, tests, benchmarks, and documentation each carry their own
environment (`scripts/`, `test/`, `bench/`, `docs/`) that consumes the
package by path and activates itself, so the step above is optional for
them; the first invocation of each environment resolves and precompiles
it.

## Usage

All commands below run from the repository root; scripts resolve relative paths against the project root and may equally be invoked from any working directory. Configuration defaults come from `config.toml` (including model and optimizer hyperparameters under `[model]` and `[training]`, and the output roots under `[paths]`) and are validated on load; CLI flags override them. RNG seeds are set from the configuration. Each run is assigned a run identifier under which models (JLD2), plots, logs, and a configuration snapshot are stored.

```bash
# 1. Simulate continuous telemetry (HDF5 strain + point-wise labels + event catalog)
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

`--test-mode` restricts training to `test_mode_samples` samples and `test_mode_epochs` epochs (from `[training]`) for rapid validation.

Labeled inference fits the decision threshold from the ROC curve and persists it as `threshold.toml` next to the model. Blind inference (`--labels ""`) requires that persisted threshold and produces per-window scores and decisions without labels.

## Testing and Benchmarks

```bash
julia --project -e 'using Pkg; Pkg.test()'   # static QA, unit tests, pipeline smoke test (equivalently: julia test/runtests.jl)
julia bench/benchmarks.jl                    # performance measurements
```

Documentation builds with Documenter.jl (`docs/make.jl` activates its own environment):

```bash
julia docs/make.jl
# open docs/build/index.html
```

## Component Status

| Component | State |
|---|---|
| Core library (`src/`) | Functional; unit tests pass; fail-fast input validation on public interfaces |
| Telemetry simulator | Functional and seeded; Robson–Cornish–Liu (2019) noise at physical amplitude, injections scaled to a matched-filter SNR and anchored on the coalescence sample; the phenomenological MBHB waveform (Nyquist aliasing near merger, no mass-consistent ringdown) awaits the closed-form IMR model |
| Feature extraction | PSD-whitened, amplitude- and window-length-independent features; scaler fitted on the training partition and persisted with the model |
| Training script | Runs end-to-end (verified); evaluation protocol still leaks information (random split over overlapping windows, validation set reused as test set) |
| Inference script | Runs with labeled and blind data; threshold persisted per run; threshold is still fitted on the evaluated dataset |
| Documentation | Tracks the current state; remediation of the remaining defects is planned |

Version 0.1.x is a pre-release: the remaining physics and evaluation deficiencies are documented in `docs/src/physics.md` and `docs/src/architecture.md` and scheduled for remediation before any science use.

## License

MIT — see `LICENSE`.
