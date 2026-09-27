# MilliHertzQML.jl

[![CI](https://github.com/PaulGoG/MilliHertzQML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PaulGoG/MilliHertzQML.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/PaulGoG/MilliHertzQML.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PaulGoG/MilliHertzQML.jl)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://PaulGoG.github.io/MilliHertzQML.jl/stable/)
[![Release](https://img.shields.io/github/v/release/PaulGoG/MilliHertzQML.jl?label=release)](https://github.com/PaulGoG/MilliHertzQML.jl/releases)
[![Julia](https://img.shields.io/badge/julia-%E2%89%A5%201.12-9558B2)](https://julialang.org/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Quantum machine learning for gravitational-wave detection in the
milliHertz band. A variational quantum classifier (VQC) with data
re-uploading detects massive black hole binary (MBHB) coalescences in
LISA-like telemetry, both in a completed record and streamed through a
simulated telemetry mission as the data reaches the ground. The circuits
are simulated with `Yao.jl`; training uses `Zygote.jl` gradients and
`Flux.jl` optimisers. The classification approach follows Isfan et al.,
*Class. Quantum Grav.* **42** 225001 (2025), DOI
[10.1088/1361-6382/ae1787](https://doi.org/10.1088/1361-6382/ae1787),
reimplemented in Julia in place of the original Python and Qiskit code.

On the LISA Data Challenge 2a "Sangria" blind year the selected model
detects all five labelled MBHB events at 1.57 false alarms per 30 days;
retrained on a smoothed whitening spectrum, it alerts five of six
coalescences 12 to 71 hours before their merger in a causal replay of the
year, at 0.16 false alarms per 30 days. The [results](#results) and the
[benchmark page](https://PaulGoG.github.io/MilliHertzQML.jl/stable/benchmark/)
state both with their protocol and limits.

## File structure

```
MilliHertzQML.jl/
├── activate.jl          # activates and instantiates the root environment
├── configs/             # default, Sangria, and experiment configurations
├── src/                 # package: physics, model, stages, telemetry coupling
├── ext/                 # CairoMakie, DeepSpaceTelemetry and CurvatureDistinguishability extensions
├── scripts/             # entry points, own environment
├── test/                # suite with static QA, own environment
├── bench/               # benchmarks, own environment
├── docs/                # Documenter manual, own environment
├── data/                # inputs and outputs (not tracked)
└── models/              # per-run model directories (not tracked)
```

The full tree is at the end of this page.

## Environment

The package is developed on Julia 1.13, the current stable release, and
supports Julia 1.12 and later; continuous integration runs the suite on
both at every push. `Manifest.toml` files are not tracked: the
environments resolve from `Project.toml` and its `[compat]` bounds. With
[juliaup](https://github.com/JuliaLang/juliaup), `juliaup update` keeps
the release channel current.

The package is not registered in the General registry. A downstream
environment adds it by URL, pinned to a release tag,

```julia
using Pkg
Pkg.add(url = "https://github.com/PaulGoG/MilliHertzQML.jl", rev = "v1.1.0")
```

or consumes a clone by path, through `Pkg.develop(path = ...)` or a
`[sources]` entry. From a clone of this repository:

```bash
git clone https://github.com/PaulGoG/MilliHertzQML.jl.git
cd MilliHertzQML.jl
julia activate.jl
```

Every environment carries an `activate.jl` that activates and instantiates
it: `julia -i activate.jl` opens a session in the root environment, and
`julia -i test/activate.jl` (likewise `scripts/`, `docs/`, `bench/`) in an
auxiliary one. The scripts, tests, benchmarks and documentation each have
their own environment, which consumes the package by path and activates
itself, so the step above is optional for them; the first invocation of
each environment resolves and precompiles it.

The script and test environments pin two unregistered packages of mine as
git sources at the commit of a release tag:
[DeepSpaceTelemetry.jl](https://github.com/PaulGoG/DeepSpaceTelemetry.jl)
`v2.0.0`, the telemetry producer, whose
[manual](https://PaulGoG.github.io/DeepSpaceTelemetry.jl/stable/)
documents the run directory this package reads, and
[CurvatureDistinguishability.jl](https://github.com/PaulGoG/CurvatureDistinguishability.jl)
`v2.0.1`, for the constellation response of the simulator. On a machine
whose git configuration rewrites GitHub URLs to SSH, instantiate them with
`JULIA_PKG_USE_CLI_GIT=true`, so that the package manager uses the
command-line git client and its agent.

## Entry points

```bash
julia activate.jl                                   # instantiate the root environment
julia scripts/generate_data.jl configs/default.toml --run-id sim01
julia scripts/preprocess_ldc.jl configs/default.toml \
    --h5-file data/inputs/simulated_telemetry_complex.h5 \
    --label-file data/inputs/simulated_telemetry_complex_labels.csv \
    --output-prefix telemetry_sim
julia scripts/train.jl configs/default.toml --run-id <RUN_ID>
julia scripts/infer.jl configs/default.toml --run-id <RUN_ID> --block test
julia scripts/infer_telemetry.jl configs/default.toml --run-dir <RUN_DIR> \
    --model models/run_<RUN_ID>/gw_model.jld2       # replay a telemetry mission
julia test/runtests.jl                              # static QA, unit tests, pipeline smoke test
julia bench/benchmarks.jl                           # performance measurements
julia docs/make.jl                                  # manual, written to docs/build/
```

The manual is deployed at
[PaulGoG.github.io/MilliHertzQML.jl/stable](https://PaulGoG.github.io/MilliHertzQML.jl/stable/)
from the latest release and at
[`/dev`](https://PaulGoG.github.io/MilliHertzQML.jl/dev/) from `main`.

## Component status

| Component | State |
|---|---|
| Core library (`src/`) | Functional; unit tests pass; fail-fast validation on every public interface |
| Pipeline architecture | Every stage a typed library function behind a thin script; one TOML file per run, validated on load; git and hardware provenance in every snapshot; overwrite-safe writes; memory guard; stage-timing table |
| Telemetry simulator | Robson–Cornish–Liu noise at physical amplitude and IMRPhenomA injections scaled to a matched-filter SNR; optionally the A and E channels of the constellation through the CurvatureDistinguishability extension; no spins or higher modes |
| Feature extraction | Whitened band powers, spectral entropy and power spread per window; whitening by the strain model, the LDC TDI model, or a Welch estimate, optionally smoothed in log-frequency; scaler fitted on the training block and persisted with the model |
| LDC products | Native reader of the TDI datasets and catalogues; analytic TDI noise PSD; truth-stream labels; validated on Sangria against a reference SNR and a noise-only null test |
| Training and inference | Chronological blocks, class-weighted loss, threaded batch gradients, early stopping, threshold fitted on a calibration block by a false-alarm criterion, event-level metrics |
| Telemetry coupling | Payload export for DeepSpaceTelemetry; replay and live modes; causal whitening from delivered data; alert table with a persistence criterion; delivery holes excluded from scoring, outages, retransmission and generation gaps handled |
| Documentation | Manual with the Sangria benchmark and its figures; deficiencies listed on the physics and architecture pages |

## Pipeline

```mermaid
flowchart LR
  H["LDC Sangria TDI<br/>(HDF5)"] --> L[label_ldc.jl]
  H --> P[preprocess_ldc.jl]
  L --> P
  P -->|"features, labels, PSD"| T[train.jl]
  T -->|"weights + threshold"| I[infer.jl]
  T -->|"weights + threshold"| S[infer_telemetry.jl]
  P --> X[export_telemetry_payload.jl]
  X -->|"payload + scenario"| M(["DeepSpaceTelemetry<br/>mission"])
  M -->|"run directory"| S
  I --> R["Blind-year metrics"]
  S --> A["Alert times"]
```

The batch path trains and evaluates on a completed record. The streaming
path replays the same model against a telemetry mission and scores each
window once its conditioning stretch, the interval of the record around
the window over which it is high-pass filtered and whitened, has reached
the ground. The threshold is fitted once, in the batch path, and carried
unchanged into both.

Every script takes the configuration file as its first argument (default
`configs/default.toml`) and may be invoked from any working directory;
relative paths resolve against the repository root. The TOML file holds
every parameter: physical and numerical settings, output roots under
`[paths]`, memory thresholds under `[resources]`, and RNG seeds. The
command line adds only a run identifier, a test-mode switch, a seed
override, and the location of external inputs. Each run stores its model,
figures, logs, training history and configuration snapshot under its run
identifier.

## Usage

### Simulated telemetry

```bash
# 1. Simulate continuous telemetry (HDF5 strain, point-wise labels, event catalogue)
julia scripts/generate_data.jl configs/default.toml --run-id sim01

# 2. Sliding-window feature extraction
julia scripts/preprocess_ldc.jl configs/default.toml \
    --h5-file data/inputs/simulated_telemetry_complex.h5 \
    --label-file data/inputs/simulated_telemetry_complex_labels.csv \
    --output-prefix telemetry_sim

# 3. Training: chronological blocks, class-weighted loss, early stopping on
#    the validation block, threshold fitted on the calibration block
julia scripts/train.jl configs/default.toml --run-id <RUN_ID>

# 4. Inference with the persisted threshold and the diagnostic figures;
#    --block test restricts the evaluation to the test block
julia scripts/infer.jl configs/default.toml --run-id <RUN_ID> --block test
```

Training converges in a few dozen epochs and stops on the validation
block; `scripts/animate.jl` renders the history and a telemetry replay as
GIFs beside the static figures.

![Training and validation loss and the validation accuracy, epoch by epoch, with the selected checkpoint of the run](docs/src/assets/training_history.gif)

`--test-mode` restricts training to the first `test_mode_samples` windows
and `test_mode_epochs` epochs for a quick check. `--seed` overrides
`[training] seed` for one run and is recorded in the run's configuration
snapshot. Training and inference use every Julia thread of the session
(`julia -t auto`, or `JULIA_NUM_THREADS`); `threaded = false` under
`[training]` selects the serial path. Each stage is also a library
function (`generate_telemetry`, `label_truth_stream`, `preprocess_record`,
`train_classifier`, `evaluate_classifier`) taking the parsed configuration
and returning its artifacts.

### Sangria products

For an LDC product the labels come from the truth stream, and the
whitening PSD is a Welch estimate of the record (`[preprocessing] psd =
"welch"`) or the LDC analytic TDI model (`"ldc"`). The configurations of
the benchmark are in `configs/experiments/`: `q8_b6.toml` is the selected
one and `q8_b6_s001.toml` its smoothed-whitening variant;
`configs/sangria.toml` is the four-qubit baseline and
`configs/sangria_paper.toml` the parity run with the features of the
reference paper. The HDF5 products are passed on the command line:

```bash
# Labels from the truth stream of the training product and of the unblinded
# MBHB-only TDI of the blind year (columns t, X, Y, Z)
julia scripts/label_ldc.jl configs/sangria.toml --h5-file <LDC2_sangria_training_v2.h5>
julia scripts/label_ldc.jl configs/sangria.toml \
    --truth-csv <mbhb_unbl.csv> --output-prefix sangria_blind_points

# Features of both years, training, and the blind evaluation
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6.toml \
    --h5-file <LDC2_sangria_training_v2.h5> --label-file data/inputs/sangria_labels.csv
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6.toml \
    --h5-file <LDC2_sangria_blind_v2.h5> \
    --label-file data/inputs/sangria_blind_points_labels.csv --output-prefix sangria_b6_blind
julia scripts/train.jl configs/experiments/q8_b6.toml --run-id q8_b6_pi
julia scripts/infer.jl configs/experiments/q8_b6.toml --run-id q8_b6_pi
```

The validation anchors of the LDC reader and noise model run with the
test suite when `MILLIHERTZQML_LDC_DIR` names the directory holding
`LDC2_sangria_training_v2.h5`.

### Telemetry coupling

The coupling to the telemetry producer DeepSpaceTelemetry.jl is
file-based in both directions. Upstream, a product of this pipeline is
exported as the producer's external payload (one `Amplitude` column at
0.2 Hz) with a scenario fragment carrying the geometry (50 s segments, ten
per batch, so that one batch advances the window by one step), the
mission epoch, and the event catalogue as markers. Downstream, a producer
run directory is replayed, or followed live, through the producer's own
API: the consumer tracks the coverage of delivered rows, scores every
window once its conditioning stretch has arrived, with the conditioning
of the batch pipeline, and raises an alert when `alert_persistence`
consecutive windows are alarmed.

```bash
julia scripts/export_telemetry_payload.jl configs/default.toml \
    --h5-file data/inputs/simulated_telemetry_complex.h5 \
    --catalog data/inputs/simulated_telemetry_complex_events.csv --output-prefix mission01
# ... run DeepSpaceTelemetry on a scenario merged from data/inputs/mission01_scenario.toml ...
julia scripts/infer_telemetry.jl configs/default.toml --run-dir <DeepSpaceTelemetry run directory> \
    --model models/run_<RUN_ID>/gw_model.jld2 \
    --events data/inputs/simulated_telemetry_complex_events.csv --run-id coupling01
```

`infer_telemetry.jl` writes `telemetry_windows.csv` (one row per scored
window, with the arrival that completed it and the inference wall time),
`alert_latency.csv` (per event: the alert, its ground arrival time
relative to the merger, the same with the ground processing budget, and
the false-alarm episodes per 30 days), a snapshot, and the alert figure.
The `[telemetry]` section of the configuration holds the geometry, the
coverage and erosion policy, the whitening PSD of the replay, the alert
persistence, the accepted producer version, and the processing budget.

### Constellation response

The simulator records, by default, one strain referred to the
sky-averaged sensitivity, with every source placed at a matched-filter
SNR. With `response = "lisa"` in `[generation]` it records the A and E
channels of the constellation instead: antenna patterns on the LISA
orbits, orbital Doppler phase and transfer roll-off from
CurvatureDistinguishability.jl, loaded as a package extension, with the
MBHB injections at physical amplitude for a luminosity distance drawn
from `[mbhb_distance_min_gpc, mbhb_distance_max_gpc]` and isotropic
orientation. Such records are whitened with `psd = "channel"` or
`"welch"`; the physics page of the manual states the conventions and
their limits.

### Run artifacts

Every snapshot a stage writes carries the hardware fingerprint, the git
description of the tree and the package version. Existing files are moved
to `<stem>_#k<ext>` backups instead of being overwritten, and
preprocessing reuses a feature product whose parameters have not changed
unless `--force` is given. Before allocating, a stage estimates its
memory against `[resources]` and refuses to start above
`max_memory_gib`.

Training writes `split.toml` (block ranges), `threshold.toml` (the
threshold fitted on the calibration block by `threshold_criterion`: by
default `far`, at most `target_far_per_30d` false-alarm episodes per 30
days and an alarm duty cycle of at most `target_fpr`, scanned from the
highest threshold down), `threshold_sweep.csv` (the operating
characteristic of the calibration block), and `metrics.toml` (window- and
event-level metrics of the validation and test blocks). Inference applies
the persisted threshold; blind inference (`--labels ""`) produces scores
and decisions without labels.

### Figures

Every figure shares one layout (a 900 × 600 pt single panel that grows by
350 pt per stacked panel) and one theme (Computer Modern, 26 pt type,
boxed axes, legend above the axes, Okabe–Ito colours, one colour per
quantity), and is exported as vector PDF, a 4× PNG and a provenance
sidecar (`<plots>/run_<id>/<figure>.{pdf,png,toml}`). The figure
functions live in a CairoMakie extension (`using CairoMakie` activates
them), so the core library carries no plotting dependency.

## Results

All results are on the LDC 2a "Sangria" data set, with the decision
threshold fitted on the training year and applied unchanged to the blind
year.

| | Selected configuration, `q8_b6` | Smoothed whitening, `q8_b6_s001` |
|---|---|---|
| Completed blind year: label spans, false alarms per 30 d | 5 of 5, **1.57** | 5 of 5, 2.97 |
| Streamed year, causal whitening: coalescences alerted on their own | 5 of 6 | 5 of 6 |
| Alerts reaching the ground more than an hour before the merger | 2, consistent with chance | **5, 12 to 71 h ahead** |
| Streamed year: false alarms per 30 d | 1.90 | **0.16** |
| Conditioning lag of every alert | 1.16 days | **2.8 hours** |
| Record scored at 0.43 % scattered batch loss | 9 % | **81 %** |

### The Sangria blind year

The selected model (`configs/experiments/q8_b6.toml`: 8 qubits, 4
re-uploading layers, six sub-mHz band powers, run `q8_b6_pi`) detects
**all five labelled MBHB events at 1.57 false-alarm episodes per 30
mission days**, from a threshold fitted on the pooled held-out block of
the training year, 109 days of validation and test, where the fit
predicted 1.38. The configuration and seed were chosen on the validation
block of the training year by a rule fixed before the blind year was
scored, and the benchmark page reports every run of the grid beside it.
The blind year is whitened by its full-record PSD, the median Welch
estimate of the entire year, which is available only after the whole
record has been received: admissible for a completed record, non-causal
for a streamed one.

![Classifier output over the Sangria blind year](docs/src/assets/benchmark_mission_trace.png)

Every run encodes its features on the half period ``[0, π]`` of the
``R_z`` gate, so that a feature saturated above the training range is
encoded as a state distinct from the noise floor. The selected model
exceeds the threshold in the merger bin of four of the six blind
coalescences; the detection of the two most saturated rests on the
inspiral excess of the preceding hours.

### The spread under re-initialisation

The selected configuration was trained at three further seeds, each run
refitting its own threshold. All four recover 5 of 5 events and two of
the four stay below the requested three false alarms per 30 days, but the
delivered rate spans 1.57 to 8.74 while the ROC area moves from 0.807 to
0.827 in the opposite direction. The recall is stable; the false-alarm
rate carries a factor-of-several uncertainty from initialisation alone,
larger than the differences between configurations of the grid, and the
seed with the lowest fitted threshold is the one whose operating point
does not transfer.

![Threshold each run fitted and the false-alarm rate it then delivered, over four initialisation seeds](docs/src/assets/benchmark_seed_spread.png)

### Telemetry replay

The blind year was replayed through a simulated year-long telemetry
mission with daily ground-station passes. Under causal whitening, with
the PSD estimated from the delivered record behind each window and
re-estimated daily, the selected model raises a sustained alert (two
consecutive alarmed windows, a persistence fixed on the calibration block
of the training year) for five of the six coalescences at 1.90
false-alarm episodes per 30 days. The alerts reach the ground station
between 26 hours before and 42 hours after the merger; at this rate 1.3
chance episodes are expected inside the label spans, so the two alerts
that precede their merger by more than an hour, 16 and 26 hours, are not
evidence of a detection of the inspiral. Event 2 is alarmed only within the cluster of alarms around the
merger of event 1. Whitened by the full-record PSD, a non-causal upper
reference, the replay counts 1.32 per 30 days.

![Classifier output and alarms over the year-long replay under causal whitening, with the alert time of every coalescence at the ground station against the latency at which each window becomes available for scoring](docs/src/assets/benchmark_telemetry_alerts_causal.png)

The replay is also rendered as an animation: four panels traverse the
year in the order in which the ground station received the windows. The
dotted line is the ground-station clock; its distance from the edge of
the received data is the availability latency, the wait for the
conditioning stretch plus the downlink delay.

![A year of telemetry replay under causal whitening: coverage, classifier score against the threshold with the labelled spans, cumulative alarm episodes, and window availability](docs/src/assets/mission_replay.gif)

### Effect of a lossy link

Seventeen 30-day missions over days 65 to 95 of the blind year, each
differing from a lossless reference in one property of the channel or of
the spacecraft, were replayed with the selected model. Scattered
permanent loss is the property that reduces the scored record: a window
is scored only once its whole conditioning stretch of 410 consecutive
batches has reached the ground, so the scorable fraction falls as
`(1 − p)^410` and collapses beyond about 0.2 %; above that rate, whether
a coalescence is still detected depends on where the holes fall. The same
losses clustered in bursts preserve most of the record, three
retransmission attempts on a 3 % channel leave no hole, a link outage of
up to three days delays the alerts without removing a window, and a gap
in the data itself removes the stretch around it and the event inside it.

![The seventeen missions and two consumer-side variants: windows scored as a fraction of the reference, coalescences detected, and false-alarm episodes per 30 days](docs/src/assets/benchmark_gap_study.png)

### Smoothed whitening

The conditioning stretch of twenty window lengths on each side is set by
the long impulse response of whitening with a raw Welch estimate.
`configs/experiments/q8_b6_s001.toml` smooths that estimate by 0.01 dex in
log-frequency and retrains the same model with the same seed and
threshold rule. Streamed and batch scores then agree from one window
length of context on, so the stretch falls to two window lengths, 50
batches instead of 410, and the conditioning lag from 1.16 days to 2.8
hours. On the completed blind year the smoothed model recovers the five
label spans at 2.97 false alarms per 30 days, within the spread of the
selected configuration under re-initialisation.

In the causal replay of the year, at the persistence of three that the
calibration rule gives for this model, five of the six coalescences are
alerted on their own, each 12 to 71 hours before its merger at the
ground station, at two false-alarm episodes in the year (0.16 per 30
days). About 0.1 chance episodes are expected inside the label spans, so
these alerts are a detection of the inspiral. In the seventeen lossy
missions the smoothed model alerts both coalescences in every mission at
0 to 2.5 false alarms per 30 days, and at 0.43 % permanent loss it still
scores 81 % of the record, where the selected configuration scores 9 %.
These results rest on one initialisation seed.

![Classifier output and alarms over the year-long replay of the smoothed-whitening model, with the alert time of every coalescence at the ground station](docs/src/assets/benchmark_telemetry_alerts_smoothed.png)

### Against other methods

Reproduced on the same years and encoding with its own raw-periodogram
features and four-qubit register (`configs/sangria_paper.toml`), the
published method delivers 31.7 false-alarm episodes per 30 days, against
1.57 for the band features, and clears the threshold on the highest-SNR
merger by a margin of 0.002; the paper reports no false-alarm rate. A
classical multilayer perceptron trained on the same challenge data
detects all six coalescences with no false alarm, with 29,569 parameters
against the 64 of the circuit. The comparison is indicative rather than
decided: the perceptron is quoted from the predictions released with it,
not reproduced, and both classifiers are evaluated on a record
renormalised towards the training one, the perceptron through a refitted
scaler and this pipeline through the full-record PSD of the blind year.

## Limitations

- **One blind realisation of five events.** The recall is 5 of 5 and the
  false-alarm rate is measured over 364 days, but five events do not
  measure a detection efficiency.
- **Few initialisations.** The selected configuration was trained at four
  seeds, whose delivered false-alarm rates span 1.57 to 8.74 per 30 days;
  the rest of the grid, and the smoothed-whitening configuration, at one.
- **Single channel.** Only A is used. E and T carry independent
  information and would allow a null-channel veto.
- **Scattered permanent loss must stay below about 0.2 % for the selected
  configuration.** The coupling discards windows whose conditioning
  stretch contains a delivery hole instead of scoring across it; the
  smoothed-whitening configuration, with a stretch of 50 batches instead
  of 410, still scores 81 % of the record at 0.43 %. The classifier has
  never been trained on gapped data.
- **The threshold comes from the same mission's earlier year.** A real
  chain would recalibrate as the mission proceeds; the transfer measured
  here spans one year, in one direction.
- **The encoding is mitigated, not solved.** On ``[0, π]`` a clamped
  feature is a state distinct from the floor, but every clamped feature
  is the same state whatever its magnitude; the two most saturated
  coalescences still score below the threshold in their merger bin.
- **The completed-record result is non-causal.** It whitens the blind year
  by the PSD of the whole year; the streamed results are quoted under
  causal whitening.

The physical and methodological deficiencies behind these, from the
scope of the waveform and noise models to the single evaluation record,
are listed on the [physics](docs/src/physics.md) and
[architecture](docs/src/architecture.md) pages.

## Full file tree

<details>
<summary>Full file tree</summary>

```text
MilliHertzQML.jl/
├── activate.jl             # Activates and instantiates the root environment
├── Project.toml            # Package metadata: only the dependencies of src/
├── src/
│   ├── MilliHertzQML.jl    # Module definition and exports
│   ├── config.jl           # TOML loading, validated key access, typed settings of every section, path resolution
│   ├── provenance.jl       # Run identifiers, hardware and git provenance, overwrite-safe writing, memory guard, stage timer
│   ├── stages/             # One typed stage function per pipeline step
│   │   ├── generation.jl   #   generate_telemetry: simulated continuous telemetry, labels, event catalogue
│   │   ├── labeling.jl     #   label_truth_stream: point-wise MBHB labels of an LDC product
│   │   ├── export_payload.jl #  export_telemetry_payload: A-channel payload and scenario fragment for the telemetry producer
│   │   ├── preprocessing.jl #  preprocess_record: whitening and window features with produce-or-load semantics
│   │   ├── training.jl     #   train_classifier: chronological blocks, training, threshold fitted on the calibration block
│   │   └── inference.jl    #   evaluate_classifier: scoring, event-level metrics
│   ├── model.jl            # VQC struct, ansatz and feature-map construction
│   ├── training.jl         # Forward pass, class-weighted BCE loss, threaded batch gradient
│   ├── evaluation.jl       # Chronological block split, ROC, calibration-block threshold, event-level metrics
│   ├── simulation.jl       # Noise model (Robson–Cornish–Liu 2019), synthesis, matched-filter SNR, whitening
│   ├── waveforms.jl        # IMRPhenomA inspiral–merger–ringdown waveform on the sampling grid
│   ├── response.jl         # Detector-response interface: sky-averaged response and the constellation hook the extension implements
│   ├── data.jl             # Window features, train-fitted feature scaler, CSV loading
│   ├── ldc.jl              # LDC TDI noise PSD, HDF5 readers, A/E/T, Welch PSD and its log-frequency smoothing, truth-stream labels
│   ├── visualization.jl    # Figure interface: theme, export with provenance, one function per figure
│   ├── telemetry.jl        # Telemetry coupling: run interface, coverage, scheduler, streaming detector, trailing PSD, replay, alert table
│   └── persistence.jl      # JLD2 model save and load: parameters, hyperparameters, feature scaler
├── ext/
│   ├── MilliHertzQMLCairoMakieExt.jl        # CairoMakie implementation of the figures and animations
│   ├── MilliHertzQMLCurvatureDistinguishabilityExt.jl # Constellation response of the simulator
│   └── MilliHertzQMLDeepSpaceTelemetryExt.jl # Run-directory adapter over the DeepSpaceTelemetry API
├── scripts/
│   ├── Project.toml        # Script environment (package by path, the two producers by git)
│   ├── activate.jl         # Activates and instantiates this environment
│   ├── common.jl           # Activation of the script environment
│   ├── generate_data.jl    # generate_telemetry and the trace figure
│   ├── label_ldc.jl        # label_truth_stream
│   ├── preprocess_ldc.jl   # preprocess_record
│   ├── train.jl            # train_classifier, terminal dashboard, file logger, training figure
│   ├── infer.jl            # evaluate_classifier and the diagnostic figures
│   ├── export_telemetry_payload.jl  # Payload CSV and scenario fragment for a DeepSpaceTelemetry mission
│   ├── infer_telemetry.jl  # Replay or follow a DeepSpaceTelemetry run: scored windows, alert table, figure
│   └── animate.jl          # GIF of a training history or of a telemetry replay, with a provenance sidecar
├── test/
│   ├── Project.toml        # Test environment (package by path, the two producers by git)
│   ├── activate.jl         # Activates and instantiates this environment
│   ├── runtests.jl         # Static QA (Aqua, JET, ExplicitImports), unit tests, figures, pipeline smoke test
│   ├── labeling_tests.jl   # Labelling stage on a synthetic truth stream
│   ├── telemetry_tests.jl  # Coupling core on an in-memory run
│   ├── telemetry_integration_tests.jl  # A DeepSpaceTelemetry mission replayed through the extension
│   ├── response_tests.jl   # Constellation response extension: patterns, channel PSD, catalogue, A/E record
│   └── export_payload_tests.jl         # Payload export stage
├── bench/
│   ├── Project.toml        # Benchmark environment (package by path)
│   ├── activate.jl         # Activates and instantiates this environment
│   └── benchmarks.jl       # BenchmarkTools performance measurements
├── docs/
│   ├── Project.toml        # Documentation environment (package by path)
│   ├── activate.jl         # Activates and instantiates this environment
│   ├── make.jl             # Documenter build and deployment
│   └── src/                # Manual pages, bibliography, and the figures they show (build/ is generated)
├── configs/
│   ├── default.toml        # Pipeline defaults (simulator)
│   ├── sangria.toml        # Sangria four-qubit baseline: Welch-whitened two-band features, truth-stream labels
│   ├── sangria_paper.toml  # Sangria parity run with the raw-window features of Isfan et al. (2025)
│   └── experiments/        # Sangria experiments: one configuration per model width, depth, band partition, and whitening variant
├── data/
│   ├── inputs/             # Telemetry records, feature and label CSVs (not tracked)
│   └── outputs/            # Per-run figures and results (not tracked)
├── models/                 # Per-run model directories (not tracked)
├── .github/
│   ├── workflows/CI.yml    # Tests on the compat floor and the current release, formatting check, coverage
│   ├── workflows/Documentation.yml  # Manual built on every push, deployed to GitHub Pages
│   └── dependabot.yml      # Weekly updates of the Julia and GitHub Actions dependencies
├── .JuliaFormatter.toml    # Formatter configuration
├── .gitignore
├── CHANGELOG.md            # Notable changes (Keep a Changelog format)
├── CITATION.cff            # Citation metadata
└── LICENSE                 # MIT
```

</details>

## How to cite

Cite the software through `CITATION.cff`, or with:

```bibtex
@software{Gogita_MilliHertzQML,
  author  = {Gogîță, Paul-Adrian},
  title   = {MilliHertzQML.jl},
  version = {1.1.0},
  year    = {2026},
  url     = {https://github.com/PaulGoG/MilliHertzQML.jl}
}
```

## License

MIT; see `LICENSE`.
