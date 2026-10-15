# MilliHertzQML

MilliHertzQML is a Julia pipeline for the detection of massive black hole
binary (MBHB) coalescences in LISA telemetry by a variational quantum
classifier (VQC) with data re-uploading. The circuits are simulated with
`Yao.jl` and differentiated by its reversible (adjoint) mode; the
optimiser comes from `Optimisers.jl`.
The classification approach follows Isfan et al. [IsfanEtAl2025](@cite).

The LISA Data Challenge (LDC) 2a "Sangria" data set holds one simulated
year for training and a second, "blind" year whose source catalogue was
withheld until evaluation. On the blind year, with decision thresholds
fitted on the training year and applied unchanged:

| | Selected configuration, `q8_b6` | Streaming configuration, `q8_b6_s001` |
|---|---|---|
| Completed blind year | 5 of 5 labelled events, 1.57 false alarms per 30 days | 5 of 5, 2.97 per 30 days |
| Streamed year, causal whitening | 6 of 6 coalescences alerted, one 1.6 hours before its merger and five after, 0.74 per 30 days | 5 of 6, each 11 to 29 hours before its merger, 0.41 per 30 days |
| Record scored at 0.43 % scattered batch loss | 9 % | 81 % |

Events 1 and 2 of the six coalescences merge 29.5 hours apart and share
one label span, so the completed blind year counts five labelled events.
A false alarm is an episode, a run of alarmed windows outside the spans in
which alerts are credited, counted per 30 days; in the streamed year it
must last at least three windows. The streamed-year alerts are preliminary
alerts of three consecutive alarmed windows, timed at the ground station.
The [Sangria Benchmark](benchmark.md) page defines every row under the
same table.

The selected configuration, eight qubits, four re-uploading layers and six
band powers between 0.3 and 40 mHz, was chosen on the training year by a
rule fixed before the blind year was scored. Trained at four seeds, every
configuration with four or six bands from 0.3 mHz upwards recovers all
five events in every run, and the false-alarm rates of the configurations
do not differ beyond the scatter between seeds, which for the selected one
spans 1.57 to 8.74 per 30 days. The completed record is whitened by the
full-record power spectral density (PSD) of the blind year, available only
after the whole record has been received; the streamed year is whitened
causally, from data already delivered to the ground station.

The streaming configuration is the selected one retrained on a Welch
whitening spectrum smoothed in log-frequency, which shortens the
conditioning lag, the stretch of data after a window that must be
delivered before the window can be scored, from 1.16 days to 2.8 hours. At each of four
initialisation seeds it alerts five or six of the six coalescences before
their merger, at 0.41 to 1.40 false alarms per 30 days. An alert counts
for a coalescence only from its signal onset, the first window in which
the signal reaches a norm of 3. The norm is that, over one 1000-sample
window, of the signal-only truth stream high-passed and whitened as the
record is: the optimal single-window signal-to-noise ratio against the
record's own noise. Releases up to 2.0.1 credited alerts by other onset
rules and replayed a mission paced by the wall clock; see
[Corrections to earlier releases](benchmark.md#Corrections-to-earlier-releases).
The [Sangria Benchmark](benchmark.md) page gives the protocol, every run,
and the limits of these results.

Two further results qualify the table. Trained on the features of the A
and E channels (`[tdi] channels = "AE"`), the streaming configuration
alerts the six coalescences before their mergers at each of the four
seeds, at 0.90 to 4.44 false alarms per 30 days
([The A and E network](benchmark.md#The-A-and-E-network)). And on the same
eight features, under the same protocol, a logistic regression with nine
parameters does at least as well as the circuit with 64: it alerts the six
coalescences at every seed, at 0.58 to 0.82 false alarms per 30 days on A
and 0.58 to 0.66 on A and E, and has the larger area under the receiver
operating characteristic (ROC) on the completed record. The detection
results of this package are those of its conditioning and features, for
which the variational circuit is one classifier among others; what its
feature map adds, measured on the glitch records of LDC-2b, is a
tolerance of instrumental transients of large amplitude that a logistic
regression on the cosine and sine of the same angles reproduces
([Classical controls on the same features](benchmark.md#Classical-controls-on-the-same-features)).

![Classifier output over the Sangria blind year](assets/benchmark_mission_trace.png)

## Pipeline

Every stage is a library function behind a thin script that takes the
configuration file as its first argument
(`julia scripts/<stage>.jl configs/default.toml [--run-id ID] ...`); the
TOML file is the single source of every parameter and is validated on
load.

**Batch path**, on a completed record:

1. `scripts/generate_data.jl` (`generate_telemetry`) simulates continuous
   milliHertz telemetry at physical strain amplitude: Robson–Cornish–Liu
   noise [RobsonCornishLiu2019](@cite), resolvable galactic binaries and
   extreme-mass-ratio inspirals (EMRIs), and IMRPhenomA MBHB injections
   [AjithEtAl2008](@cite) at a prescribed matched-filter signal-to-noise
   ratio (SNR), written to HDF5 with point-wise labels
   and an event catalogue. For an LDC product, `scripts/label_ldc.jl`
   (`label_truth_stream`) derives the labels from the truth stream
   instead.
2. `scripts/preprocess_ldc.jl` (`preprocess_record`) whitens the A channel,
   by the strain model, the LDC time-delay-interferometry (TDI) noise
   model, or a Welch estimate of the
   record, optionally smoothed in log-frequency, and extracts one feature
   vector per sliding window: band powers of the whitened spectrum, its
   entropy and its log power spread.
3. `scripts/train.jl` (`train_classifier`) splits the windows
   chronologically into training, validation and test blocks, trains the
   VQC with Adam and early stopping on the validation block, fits the
   decision threshold on the calibration block, and scores the test block
   once.
4. `scripts/infer.jl` (`evaluate_classifier`) applies the persisted
   threshold to a feature table and reports window- and event-level
   metrics with diagnostic figures.

**Streaming path**, on a telemetry mission:
`scripts/export_telemetry_payload.jl` writes a product as the payload of
the DeepSpaceTelemetry.jl producer, and `scripts/infer_telemetry.jl`
replays or follows the producer's run directory, scoring every window
once the stretch of data around it has reached the ground and timing the
alerts against the coalescences.

Every artifact carries git and hardware provenance, existing files are
backed up rather than overwritten, and each stage checks its memory
estimate against `[resources]` before allocating.

**Quick start**, on simulated telemetry:

```bash
julia scripts/generate_data.jl configs/default.toml --run-id sim01
julia scripts/preprocess_ldc.jl configs/default.toml \
    --h5-file data/inputs/simulated_telemetry_complex.h5 \
    --label-file data/inputs/simulated_telemetry_complex_labels.csv \
    --output-prefix telemetry_sim
julia scripts/train.jl configs/default.toml --run-id <RUN_ID>
julia scripts/infer.jl configs/default.toml --run-id <RUN_ID> --block test
```

The Sangria runs are reproduced by the commands at the end of the
[Sangria Benchmark](benchmark.md#Reproducing) page.

## Manual

- [Results](results.md): the figures and animations of the results, with
  short captions.
- [Physics & Data](physics.md): the noise model, the simulator, the
  constellation response, the features and the whitening, the LDC
  products, and the known physical deficiencies.
- [Quantum Architecture](architecture.md): the circuit, the evaluation
  protocol, training, the decision threshold, and the pipeline
  architecture.
- [Telemetry Coupling](telemetry.md): the payload export, the replay of a
  producer run, the whitening of a streamed record, and the alert table.
- [Sangria Benchmark](benchmark.md): the results on the LDC blind year,
  completed and streamed, over a lossy link, of the streaming
  configuration with smoothed whitening and on the A and E network, and
  the classical controls on the same features, with the glitch records of
  LDC-2b.
- [API Reference](api.md) and [References](references.md).
