# Sangria Benchmark

This page reports what the classifier does on the LISA Data Challenge 2a
"Sangria" data set: one year of simulated LISA telemetry for training and a
second, blind year for evaluation. It is the honest result, including the
respects in which a small classical network does better.

Headline: on the blind year the eight-qubit model detects **all five
labelled MBHB events at 2.47 false-alarm episodes per 30 mission days**,
from a decision threshold fitted on held-out data of the *training* year
and applied without adjustment. The threshold's own prediction for that
rate was 2.20, so the operating point transfers.

Three qualifications belong next to the headline rather than among the
caveats. The shipped model is the best of seven configurations that were
all scored on the blind year, so the 2.47 carries a selection effect; the
spread under re-initialisation alone, measured below, runs from 1.48 to
2.97. Every run on this page encodes its features on the full period
``[0, 2\pi]`` of the ``R_z`` gate, under which the two ends of the
encoding interval prepare the same state; the consequence — the
classifier is blind at the merger itself and fires on the inspiral — is
measured in the section on the encoding below. And the blind year is
whitened by its own Welch estimate, a label-free statistic of the record
under evaluation, which every streaming latency below inherits as a
year-long look-ahead. The package now encodes on ``[0, \pi]``, and the
year replay is reported under ground-causal whitening as well; the
retraining that will replace these numbers has not been made, and the
page reports what was measured.

## Data and labels

LDC2a Sangria provides two one-year records of TDI ``X``, ``Y``, ``Z`` at
0.2 Hz: a training product with its source catalog and truth streams, and
a blind product whose MBHB catalog is released separately. Both are read
natively (`read_tdi`, `read_catalog`) and reduced to the noise-orthogonal
A channel, ``A = (Z - X)/\sqrt{2}`` (`tdi_to_aet`).

Labels come from the truth stream rather than from the classifier's own
notion of a detection (`label_truth_stream`). A merger is a peak of the
windowed matched-filter SNR against the analytic Sangria TDI PSD, separated
from its neighbours by at least a day and reaching SNR 8; a window is
labelled positive when it overlaps the fixed span of four days before to
27 minutes after a coalescence. The training year carries 13 such events
and the blind year 6 coalescences that merge into **5 labelled runs**, two
of them falling within a day of each other.

The first and last ten window lengths of each record are dropped
(`edge_margin`), because the circular high-pass and whitening filters ring
there; 62,863 windows of 1000 samples at a stride of 100 remain, covering
363.8 days.

Features are the mean whitened power in six bands whose edges rise from
0.3 mHz, together with the spectral entropy and the log power spread —
eight numbers, one per qubit (`feature_set = "bands"`). The whitening PSD
is a Welch estimate of the record being processed.

## Protocol

The evaluation protocol matters more than usual here, so it is stated in
full.

**Blocks.** Windows overlap by 90 %, so any random split would put copies
of the same signal on both sides of it. The training year is cut
chronologically into training (70 %), validation (15 %) and test (15 %)
blocks separated by one window length (`chronological_split`). The feature
scaler is fitted on the training block alone, early stopping runs on the
validation block, and the **decision threshold is fitted on validation and
test pooled** — the held-out block, 110 days (`threshold_block = "held_out"`,
the opt-in the Sangria configurations declare; the package default is the
validation block). The blind year enters twice: label-free, through the
Welch estimate that whitens it, and at evaluation, where every
configuration of the grid was scored on it.

**Threshold.** The `far` criterion places the operating point of an alert
trigger: among the candidates whose false-alarm episode rate does not
exceed `target_far_per_30d` (3 per 30 days) and whose alarm duty cycle
does not exceed `target_fpr` (5 %), it takes the lowest of the admissible
range that starts at the highest threshold. The scan runs downwards
because the episode count is not monotone — as the threshold falls,
spurious episodes first multiply and then merge into one permanently
raised alarm that is charged as a handful of long episodes, which an
upward scan would accept.

**Why the pooled block.** The fitted rate is a Poisson count of the alarm
episodes its block charges, so its relative error is the inverse square
root of that count. Across the seven models below, a 55-day validation
block charges one to five episodes — `q6_b4` fits its operating point on a
single one — which is not enough to place a rate. Pooling validation with
test doubles the exposure. It does not leak into the model — the test
block is scored once after training and enters neither model selection
nor early stopping — but it costs the test block's independence: its
operating-point metrics are then in-sample, and the only independent
check of the operating point is the blind year. That is why the package
default is the validation block alone (`threshold_block = "validation"`)
and the pooled block is an opt-in for records that come with a separate
blind product, as this one does.

**Metrics.** An event is a contiguous run of positive labels and counts as
detected when any window inside it alarms. A false-alarm episode is a
contiguous run of alarmed windows outside every labelled span, and the
operational rate is episodes per 30 mission days. Window-level precision,
recall and ROC area are reported too, but the event-level pair is the
operating point.

**Encoding.** Every run on this page maps its features onto ``[0, 2\pi]``
before the ``R_z`` encoding gate. The runs predate the configuration key
that now fixes the span (`[training] phase_span`, in units of ``\pi``),
and their artifacts load with the full period they were trained on.
``R_z(2\pi) = -I`` is a global phase, so a feature clamped at the upper
scaler bound is encoded exactly as one at the lower bound, and by
continuity the response returns to its floor value as a feature
approaches the bound. Measured with the shipped model on the blind year,
in the six-hour bin holding each coalescence, from the persisted model
and feature table:

| Event | Merger-window SNR | Peak band power / scaler upper bound | Peak score in the bin |
|---|---|---|---|
| 1 | 598 | 129 | 0.738 |
| 2 | 690 | 106 | 0.708 |
| 3 | 1312 | 788 | 0.682 |
| 4 | 799 | 158 | 0.764 |
| 5 | 272 | 109 | **0.861** |
| 6 | 723 | 193 | 0.585 |

Five of the six coalescences score below the 0.826 threshold at the
merger; the only one above it is the quietest. The scores peak 12 to 36
hours earlier, where the band powers sit at 0.4 to 0.8 of the upper
bound. The detector is therefore one of intermediate sub-mHz excess, and
its efficiency is not monotone in signal strength. The package default is
now ``[0, \pi]`` (`phase_span = 1.0`), the widest interval whose two ends
are distinct states; the models on this page have not been retrained on
it, and reproducing them needs `phase_span = 2.0`.

## Results

Seven models, all trained on the same training year, all evaluated on the
blind year with the threshold carried over unchanged.

| Run | Qubits × layers, features | Epochs | Threshold | Blind events | **Blind FA / 30 d** | Blind AUC |
|---|---|---|---|---|---|---|
| `sangria02` | 4 × 4, 2 bands | 28 | 0.856 | 3 / 5 | 1.07 | 0.7911 |
| `q4_l6` | 4 × 6, 2 bands | 15 | 0.758 | **5 / 5** | 12.29 | 0.7840 |
| `q6_b4` | 6 × 4, 4 bands | 26 | 0.706 | **5 / 5** | 7.01 | 0.7997 |
| **`q8_b6`** | **8 × 4, 6 bands** | 35 | 0.826 | **5 / 5** | **2.47** | 0.7933 |
| `q8_b6_l6` | 8 × 6, 6 bands | 50 | 0.857 | **5 / 5** | 2.72 | 0.8046 |
| `q6_b4_w2000` | 6 × 4, 4 bands, 2000-sample windows | 18 | 0.714 | **5 / 5** | 27.55 | 0.8132 |
| `q6_b4_noweight` | 6 × 4, 4 bands, unweighted loss | 39 | 0.554 | **5 / 5** | 5.20 | **0.8205** |

Every model with a partitioned sub-mHz band set reaches full event recall;
the two-band four-qubit baseline does not. Of the seven, **only `q8_b6`
and `q8_b6_l6` meet the three-per-30-days target**. The table is seven
evaluations on the blind year and the shipped model is the best of them,
so its 2.47 is a minimum over seven draws; the difference to the 2.72 of
`q8_b6_l6` lies inside the 1.48 to 2.97 that re-initialisation alone
produces, so the six extra layers can be said to buy nothing measurable,
not to cost anything.

![Classifier output over the blind year](assets/benchmark_mission_trace.png)

The trace of `q8_b6` over the blind year shows what the protocol is up
against. The bulk of the noise score rises by half between day 30 and day
250 and falls back by day 350 — the Galactic foreground seen through the
constellation's rotating antenna pattern — while the five labelled spans
(beige) carry the peaks that clear the threshold. The threshold works
because it sits *above* the annual excursion, not because the events are
loud in absolute terms.

![Event recall and false alarms against the threshold](assets/benchmark_threshold_sweep.png)

The blind year's operating characteristic, drawn post hoc: event recall
holds at 1 across the whole range up to the fitted threshold, so the
operating point is limited by the false-alarm rate alone. The rate crosses
the three-per-30-days target just below 0.826, which is where the fit —
made on the training year, without sight of this curve — placed it.

### What the table is really showing

Order the same models by ROC area and the ranking nearly inverts. The two
best discriminators are the two worst operating points. The fitted
threshold orders the delivered rate better than the area does, though
with six single-seed models the rank correlation (−0.43) is nowhere near
significant and the raw thresholds of different circuits are not
commensurable; the table is shown for the pattern, not as a test:

| Run | Blind AUC | Fitted threshold | Blind FA / 30 d |
|---|---|---|---|
| `q6_b4_noweight` | **0.8205** | 0.554 | 5.20 |
| `q6_b4_w2000` | 0.8132 | 0.714 | **27.55** |
| `q8_b6_l6` | 0.8046 | 0.857 | 2.72 |
| `q6_b4` | 0.7997 | 0.706 | 7.01 |
| `q8_b6` | 0.7933 | 0.826 | **2.47** |
| `q4_l6` | 0.7840 | 0.758 | 12.29 |

The reason is a property of the data, not of the classifier. The
classifier's noise score is modulated over the year with a period of one
year and the same phase in both records: the Galaxy sits at a fixed sky
position, LISA's response to it is modulated by the constellation's
rotation, and a single year-median Welch estimate whitens the record
correctly only on average. The two years' noise distributions therefore
agree only in the far tail. Measuring the ratio of blind to training-year
noise-window exceedance at a given score:

| Model | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|---|
| `sangria02` | 1.91 | 2.07 | 1.85 | 1.17 | 0.53 |
| `q4_l6` | 2.26 | 2.67 | 3.48 | — | — |
| `q6_b4` | 2.91 | 3.21 | 8.66 | — | — |
| `q8_b6` | 1.79 | 1.66 | 1.51 | 1.22 | **1.00** |

(`—`: the training year has no noise window that high at all.)

Below about 0.80 the blind year produces one and a half to three times as
many exceedances as the training year; above it the two agree. A model
whose training-year noise tail does not reach 0.80 cannot have its
operating point placed there, so it is forced into the shoulder where the
years disagree, and it delivers roughly the factor by which they disagree.
The ROC area is a rank statistic over the whole distribution and is blind
to where the tail ends, which is why it selects the worst model of the
seven.

Six sub-mHz bands help twice over: they compress the annual excursion of
the noise tail (peak-to-trough of the 99.9 % quantile falls from 0.40 for
two bands to 0.22) and they lift the event peaks above it, so the
threshold can sit in the stable region. The mechanism is that a contrast
between two band powers that are modulated together is invariant to the
modulation, and a single wide band offers no contrast to form.

The mechanism — a contrast between band powers that are modulated
together is invariant to the modulation — is what the exceedance table
supports. Whether the ordering of the models generalises past this data
set is not something one record and six single-seed models can
establish. What does follow is the check itself: an operating point
measured on one observation record means nothing on the next unless it
sits where the two records' noise tails agree, and no amount of ROC area
substitutes for verifying that.

## Against the published method

Everything above uses whitened sub-mHz band powers, which the paper this
pipeline follows does not: Isfan et al. take the spectral entropy and the
mean, standard deviation and maximum of the raw window periodogram, with
no filtering and no whitening, on a four-qubit register. That
configuration ships as `configs/sangria_paper.toml` and was run on the same
two years, so the published result can be compared against a reproduction
of its own method rather than only against a different one.

The reproduction lands where the paper does. Isfan et al. report five of
six blind mergers, the missed one being the lowest-SNR source. The parity
run detects four of the five label spans — and the span it misses is the
one holding **event 5, the lowest-SNR merger of the blind year** at a
merger-window SNR of 272 against 598 to 1312 for the rest:

| Span | Events 1+2 | Event 3 | Event 4 | **Event 5** | Event 6 |
|---|---|---|---|---|---|
| Peak score, paper features | 0.740 | 0.664 | 0.693 | **0.550** | 0.805 |
| Peak score, band features | 0.905 | 0.885 | 0.889 | **0.892** | 0.923 |
| Threshold | 0.584 / 0.826 | | | | |

Under the paper's features the missed event peaks at 0.550 against a
fitted threshold of 0.584 — a marginal miss, which is consistent with the
paper finding it the one source its classifier could not reach. Under
band features the same event peaks at 0.892 against a threshold of 0.826
and is recovered with room to spare.

What the reproduction adds is the cost the paper does not report. It
publishes no false-alarm rate, so "five of six" has no operating point
attached to it. Fitted here by the same criterion as every other run in
this document, the paper's feature set delivers **17.7 false-alarm
episodes per 30 days** — 215 episodes over the blind year, an alarm every
1.7 days — against 2.47 for `q8_b6`. Its calibration also fails to
transfer: the fit predicted 0.275 per 30 days and the blind year returned
17.7, a factor of 64, on a held-out block whose AUC is 0.535, barely above
chance.

| | Paper features, 4 × 4 | Band features, 8 × 4 (`q8_b6`) |
|---|---|---|
| Trainable parameters | 32 | 64 |
| Events | 4 / 5 spans (misses the lowest-SNR merger) | **5 / 5** |
| False alarms / 30 d | 17.73 | **2.47** |
| Blind AUC | 0.635 | 0.793 |
| Held-out AUC | 0.535 | 0.682 |

So the conclusion is not that the quantum classifier was improved: the
register and the ansatz are doing what they did before. **The conditioning
is what buys the operating point.** Whitening the record and partitioning
the milliHertz decade into six bands recovers the event the published
method misses and cuts the false-alarm rate by a factor of seven, at the
same parameter budget the paper reports for its own circuit.

## Against a classical baseline

The GWEEP multilayer perceptron shipped with the same challenge material
detects **all five blind label spans — all six coalescences — with no
false-alarm episode at all**, alarming on 698 of 6,307,190 samples
(0.011 %) at a threshold of 0.5. It is a stronger result than this
classifier achieves, and it should be stated that way.

**It is quoted, not reproduced.** The numbers above come from scoring the
predictions shipped with the material through this project's event
protocol; the model was never retrained here. Reproducing it is out of
proportion to what it would settle: the preprocessing is a rolling Welch
spectral entropy at unit stride (`gweep_preproc.py`, window 1000), which
is 6.3 million transforms per channel-year and ten of them, and the
spectral-entropy series for X, Y and Z are not among the shipped files.

What the shipped artifacts do establish, by inspection:

- **The model.** `Flatten → Dense(128, relu) → Dense(128, relu) →
  Dropout(0.2) → Dense(1, sigmoid)` over a 10 × 10 input — ten timesteps
  of ten features — is **29,569 parameters**, against a few dozen for the
  variational circuit. That is the interesting comparison, not that the
  quantum model wins, which it does not.
- **The inputs.** Ten features: the raw projections X, Y, Z, A, E and the
  spectral entropy of each. Since `A = (Z − X)/√2` and
  `E = (X − 2Y + Z)/√6` are computed from X, Y and Z in that same script,
  the five projections span a three-dimensional space: the input is
  redundant by construction rather than wrong. The null channel
  `T = (X + Y + Z)/√3`, the standard instrumental-artifact monitor, is
  written as a comment and never used.
- **The scaling is not blind.** `gweep_preds.py` fits a fresh
  `StandardScaler` on the blind set rather than reusing the one fitted in
  training. It touches no labels and acts on the five raw projections
  only, but it is not cosmetic: between the two years the per-column means
  differ by 0.27 to 0.65 training standard deviations and the standard
  deviations themselves by up to 12 %.

That last point has to be weighed symmetrically. The operating-point
result above says the two observation years differ enough that a
threshold carried between them lands in a shoulder where their noise
distributions disagree threefold — that year-to-year shift is the hard
part of this problem. Refitting the scaler on the blind year absorbs part
of that shift, and so does the whitening of the variational classifier:
its band powers are normalised by the blind year's own Welch estimate,
band by band, which is likewise a label-free statistic of the record
under evaluation, and whitening it with the training year's estimate
instead detects nothing (see the telemetry section). Both classifiers
are therefore evaluated on a record renormalised towards the one they
trained on, by different means; the feature scaler of the variational
classifier, fitted on the training year, is the one statistic carried
across unchanged. How much of the zero false-alarm rate the refit buys
cannot be settled from the shipped files; it is a reason to read the
comparison as indicative rather than decided.

The VQC carries 8 qubits and 4 re-uploading layers, 64 parameters against
the perceptron's 29,569 — a ratio of 460 — on one projection against five
and four features against ten.

![Window-level receiver operating characteristic](assets/benchmark_roc_curve.png)

The window-level ROC of `q8_b6` on the blind year is shown for
completeness and should be read with the warning above: at 0.793 it is the
*fifth* best of the seven models and the one that transfers. The operating
point lives at a false-positive rate of ``5 \times 10^{-4}``, in the corner
of this plot where the curve carries almost no resolution — which is
precisely why the area under it is the wrong summary.

## Telemetry replay and alert latency

The benchmark above scores a finished record. The package also couples to
a telemetry producer (see [Telemetry Coupling](telemetry.md)), replaying a
year-long mission with a realistic downlink: daily ground-station passes,
1 % packet loss with retransmission, and an on-board recorder. Windows are
scored as their data arrives and alerts are timed against the true
coalescences.

Two findings from that exercise bound what this pipeline can contribute to
a low-latency alert chain, and both are properties of the conditioning
rather than of the classifier:

**The whitening is zero-phase, so it is not causal.** A window's
conditioned features depend on the data after it as much as on the data
before it. A detector that scores each window the moment its own samples
land finds two of the five blind events; sixty-four window lengths of past
context do not repair it. What restores the batch result is *centring* the
window in the stretch that is whitened around it — twenty window lengths
on each side reproduce the batch scores at a rank correlation of 0.997,
while thirty-two before and twenty after manage 0.860. An alert therefore
carries an irreducible conditioning lag of twenty window lengths, 1.16
days at these settings, before any ground-segment latency.

**The whitening PSD calibrates a record, not a model.** Whitening the
blind year with the PSD persisted from the training year costs every
detection — nought of five, at any context length — because the annually
modulated foreground makes the two records' noise genuinely different.
`psd_sidecar` supplies the PSD of the record under analysis. For a
streamed mission that PSD is an oracle: the year-median estimate that
whitens the window at the first event, on day 78, contains nine months
of data that have not reached the ground. The replay below was made with
it, so its latencies hold for a detector that knows the year's noise in
advance, not for a causal one. The same mission was therefore replayed a
second time under ground-causal whitening (`psd_mode = "trailing"`): each
window is whitened by the Welch median of the last 30 days of delivered
record behind its conditioning stretch, redone once a day, and by the
training year's PSD until one 65,536-sample segment is on the ground (the
first 446 windows). That replay is reported beside the first below.

With both settled, a full year of the blind record was replayed through a
simulated mission: 63,043 batches of 500 s, a 12-hour daily pass, 1 %
packet loss with retransmission, no permanent gaps. The consumer scored
62,834 windows and alarmed 169 of them.

![Replay of the blind year with alert latencies](assets/benchmark_telemetry_alerts.png)

What counts as an alert has to be stated, because it decides the
latencies. The first version of this page credited each event with the
earliest alarmed window overlapping its label span, whatever came after
it. Read that way, the replay alarms every coalescence and four of the
six alerts precede their merger by 1.9 to 2.7 days. Those four early
alerts are one or two isolated windows each — 75, 96 and 80 hours before
the merger, one of them at a score margin of 0.008 over the threshold —
and at the delivered 2.56 false-alarm episodes per 30 days about 1.7
chance episodes are expected inside five 96-hour spans. Five isolated
early alarms where 1.7 are expected is suggestive (Poisson
``p \approx 0.03``), but each of them is individually indistinguishable
from the false-alarm population and could not be acted on. The alert
protocol therefore requires persistence: an alert is raised by the
arrival that completes `alert_persistence` consecutive alarmed windows
(three, the shipped value), and shorter runs are neither alerts nor
charged as false alarms. Both readings of the same replay:

| Event | Merger (mission time) | Alert − merger [h], first alarmed window | Alert − merger [h], three consecutive | Three consecutive, causal PSD | Alert of its own |
|---|---|---|---|---|---|
| 1 | 2035-03-19T16:27 | −46.3 | **+2.2** | +22.4 | yes |
| 2 | 2035-03-20T21:59 | −55.2 | −27.3 | −7.1 | no: event 1's |
| 3 | 2035-08-12T15:40 | −0.3 | −0.3 | −0.3 | yes |
| 4 | 2035-09-17T07:48 | −64.4 | +7.5 | +7.5 | yes |
| 5 | 2035-09-21T21:39 | −51.3 | +17.9 | +41.7 | yes |
| 6 | 2035-11-10T00:11 | +15.1 | +15.1 | +15.1 | yes |
| False alarms / 30 d | | 2.56 | 0.74 | 1.16 | |

The data latencies exclude the one-hour processing budget, which
`latency_total_h` of the table adds. Under the persistence criterion no
alert precedes its merger by more than twenty minutes, and the sustained
alarms follow it by 2 to 18 hours: the 1.16-day conditioning lag and the
delivery latency of the lower panel, less the hours by which the inspiral
was already alarmed. Event 2 has no alert of its own under either
reading. Its merger follows event 1's by 29.5 hours, the two label spans
overlap, every alarmed window in its span belongs to one of event 1's
episodes, and no window within 43 hours of its merger reaches the
threshold. The replay therefore detects five coalescences, not six, and
the batch metric's "5 of 5 label spans" is unaffected only because the
two spans merge into one.

Under ground-causal whitening the same protocol alarms 214 windows
instead of 169, raises an alert of its own for the same five
coalescences, and charges 1.16 false-alarm episodes per 30 days. Events
3, 4 and 6 are alerted at the same hour as before; events 1 and 5 later,
22 and 42 hours after the merger. The window scores of the two replays
correlate at only 0.57, so the operating point survives the loss of the
year-long look-ahead while the scores themselves do not reproduce it;
the causal replay is the one a mission could run, and its numbers are
the ones to quote for latency.

![Replay of the blind year under ground-causal whitening](assets/benchmark_telemetry_alerts_causal.png)

Why the classifier fires on the inspiral rather than on the coalescence
is the encoding section above: at every merger the band powers exceed the
scaler bound by two to three orders of magnitude and are folded back onto
the floor, while the inspiral, at 0.4 to 0.8 of the bound, sits where the
response peaks. The early sensitivity is real to the extent the Poisson
excess says, but it is a property of the encoding rather than of the
signal, and the encoding has been changed.

The alarm tails after each merger and the record-edge transients have
one cause: the conditioning kernel of the record high-pass followed by
whitening decays as a power law rather than
exponentially, because the Welch estimate resolves sharp spectral features
such as the TDI transfer nulls. Its envelope is still 7 % of the peak
eight window lengths from the impulse and 99.9 % of its energy needs
thirty-two. The same kernel produces the record-edge transients that force
`edge_margin` and the post-merger alarm tails that account for every
charged false alarm at the operating point. Smoothing the PSD would
shorten it and is the natural next step.

## The spread under re-initialisation

The shipped configuration was retrained at three further seeds, each
refitting its own threshold on the pooled held-out block and applying it
to the blind year without adjustment:

| Seed | Fitted threshold | Fit predicted | Delivered FA / 30 d | Events | AUC |
|---|---|---|---|---|---|
| 1009 | 0.7941 | 2.750 | 2.969 | 5 of 5 | 0.7981 |
| 2027 | 0.8430 | 1.650 | 2.309 | 5 of 5 | 0.7949 |
| 3041 | 0.8134 | 1.375 | 1.484 | 5 of 5 | 0.7960 |
| 9999 (shipped) | 0.8256 | 2.200 | 2.474 | 5 of 5 | 0.7933 |

![Threshold each run fitted and the false-alarm rate it then delivered, over four initialisation seeds](assets/benchmark_seed_spread.png)

**Every realisation recovers all five events, and every one lands under
the three per 30 days the criterion asked for.** What moves is the
false-alarm rate, by a factor of two across 1.48 to 2.97, while the ROC
area barely moves at all — 0.7933 to 0.7981, a spread of half a per cent.
That is the same lesson the model grid taught, now measured within one
configuration rather than across several: the area under the curve is
nearly blind to what the operating point delivers, because the threshold
sits in the tail and the area is dominated by the bulk.

So the headline should be read as an event recall that is stable and a
false-alarm rate that carries a factor-of-two uncertainty from
initialisation alone — on top of the Poisson uncertainty of the episodes
themselves. Four realisations do not measure a distribution either; they
bound the scatter well enough to say the result is not an artefact of one
lucky initialisation.

## What a lossy link costs

The mission above delivered everything, so the hole handling was never
put under load. Five 30-day missions answer what it costs when the link
does not: the same payload window — days 65 to 95 of the blind year,
which carries coalescences 1 and 2 — and the same pass schedule, with a
channel that drops a fraction of transfers for good rather than
retransmitting them.

| Permanent batch loss | Batches lost | Windows scored | Alarm episodes | Events |
|---|---|---|---|---|
| none | 0 of 5155 | 4946 | 38 | 2 of 2 |
| 0.14 % | 7 | 2653 | 10 | 2 of 2 |
| 0.45 % | 23 | 563 | 2 | 0 of 2 |
| 1.01 % | 52 | 0 | 0 | 0 of 2 |
| 2.99 % | 154 | 0 | 0 | 0 of 2 |

![Windows a replay can score, and the events it still detects, against the permanent batch loss of the link](assets/benchmark_loss_survival.png)

The collapse is not proportional to the loss, and it is not a defect of
the hole handling. A window is scored only once its **entire conditioning
stretch** has reached the ground: `context_windows = 20` window lengths
on each side of a 1000-sample window, and one batch carries one window
step of 100 samples, so 410 consecutive batches must survive. Under an
independent per-batch loss `p` a given window therefore survives with
probability `(1 - p)^410`, drawn as the dashed curve.

The measured survival tracks that estimate while losses are sparse and
falls below it as they are not, because the expectation stops being the
right summary: surviving windows exist only inside a clean run of 410
batches, the mean spacing of losses is `1/p` batches, and once `p`
exceeds `1/410 ≈ 0.24 %` such a run is a rare event rather than a typical
one. The whole of the sweep is that one number. A link losing half a per
cent of its batches permanently delivers a detector that sees nothing,
while the same link losing them *temporarily* — retransmitted, as in the
year-long mission, where 648 retries cost nothing — is harmless.

Two consequences worth stating plainly. The operating requirement is on
**permanent** loss, and it is strict: about 0.2 %. And the smoothed
whitening PSD deferred to v1.1, which would shorten the conditioning
kernel, is not a refinement of this result but the precondition for
running on a link that loses anything at all; every reduction of the
stretch raises the tolerable loss rate in proportion.

## Caveats

- **One blind realisation of five events.** The event recall is 5 of 5 and
  the false-alarm rate is measured over 364 days, but five events do not
  measure a detection efficiency. Treat the recall as a result, not a rate.
- **One initialisation per model.** Every configuration in the grid runs
  at `seed = 9999`, so the differences between models are differences
  between single training runs. The shipped configuration has since been
  repeated at three further seeds (below); the grid has not.
- **Single channel.** Only A is used; E and T carry independent
  information and would also permit a null-channel veto.
- **No gaps in the headline.** The Sangria products are gapless, and the
  mission replayed above delivered all 63,043 batches: nothing was lost or
  pruned, so every number in the table before it describes a lossless
  link. What a lossy one costs is measured separately, above, on 30-day
  missions; the classifier itself has never been trained on data with
  gaps.
- **The threshold is fitted on the same mission's earlier year.** A real
  chain would recalibrate as the mission proceeds; the transfer measured
  here is over one year, in one direction.
- **The encoding folds saturated features onto the floor.** Every model
  here maps its features onto ``[0, 2\pi]``; five of the six blind
  coalescences score below the threshold at the merger itself (the
  encoding section). The package default is now ``[0, \pi]``; nothing on
  this page has been retrained on it.
- **The shipped model is the best of seven blind evaluations.** The grid
  was scored on the blind year and the best operating point ships, so the
  headline rate carries a selection effect of the order of the seed
  spread. The next round selects on the validation block and scores the
  blind year once per model.
- **The headline replay uses an oracle PSD.** The year-median Welch
  estimate of the blind year whitens every window of the first replay,
  including at the first event the nine months not yet delivered. The
  ground-causal replay beside it detects the same five coalescences at
  1.16 false alarms per 30 days, two of them later.
- **Five coalescences are detected on their own, not six.** Event 2 is
  only ever alarmed by event 1's windows.

## Reproducing

With `$LDC` pointing at the directory holding the Sangria HDF5 products
and `$UNBLINDED` at the CSV of the blind year's signal-only TDI (columns
`t`, `X`, `Y`, `Z`, released with the unblinded MBHB catalog):

```bash
# Labels for both years
julia scripts/label_ldc.jl configs/sangria.toml --h5-file $LDC/LDC2_sangria_training_v2.h5
julia scripts/label_ldc.jl configs/sangria.toml --truth-csv $UNBLINDED --output-prefix sangria_blind_points

# Band features for both years
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6.toml \
    --h5-file $LDC/LDC2_sangria_training_v2.h5 --label-file data/inputs/sangria_labels.csv
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6.toml \
    --h5-file $LDC/LDC2_sangria_blind_v2.h5 \
    --label-file data/inputs/sangria_blind_points_labels.csv \
    --output-prefix sangria_b6_blind

# Train and evaluate
julia scripts/train.jl configs/experiments/q8_b6.toml --run-id q8_b6
julia scripts/infer.jl configs/experiments/q8_b6.toml --run-id q8_b6
```

Training takes about 35 epochs at roughly half a minute each on 22 threads;
inference over the blind year takes a minute. The other six configurations
are in `configs/experiments/`. The runs on this page were trained before
the encoding span was a configuration key and load with the full period;
the experiment configurations now ship with `phase_span = 1.0`, so
reproducing the numbers above needs `phase_span = 2.0` in `[training]`.
