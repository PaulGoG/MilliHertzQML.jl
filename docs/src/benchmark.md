# Sangria Benchmark

This page reports what the classifier does on the LISA Data Challenge 2a
"Sangria" data set: one year of simulated LISA telemetry for training and a
second, blind year for evaluation. It is the honest result, including the
respects in which a small classical network does better.

Headline: on the blind year the eight-qubit model detects **all five
labelled MBHB events at 1.57 false-alarm episodes per 30 mission days**
(19 episodes over 364 days), from a decision threshold fitted on held-out
data of the *training* year and applied without adjustment. The
threshold's own prediction for that rate was 1.38, so the operating point
transfers. The model was chosen on the validation block of the training
year before the blind year was scored, and it is the same configuration
and seed that shipped with the first release, retrained on the
half-period encoding.

Three qualifications belong next to the headline rather than among the
caveats. The configuration was selected by a pre-registered statistic on
the training year, so the 1.57 carries no selection on the blind year; but
the spread under re-initialisation alone, measured below, runs from 1.57
to 8.74, and every other configuration of the grid falls inside that
spread, so the ranking *between* configurations is not resolved. Every
run on this page encodes its features on the half period ``[0, \pi]`` of
the ``R_z`` gate; the merger windows of the loud coalescences, which the
full-period encoding folded onto the noise floor, now score above the
threshold for four of the six, and still not for the two most saturated
(the encoding section). And the blind year is whitened by its own Welch
estimate, a label-free statistic of the record under evaluation —
legitimate for a finished record and an oracle for a streamed one, which
is why the telemetry replay below is quoted under ground-causal
whitening, with the oracle replay beside it as a bound.

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
Welch estimate that whitens it, and at evaluation, where every model is
scored once.

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
root of that count. Across the grid below, a 55-day validation block
charges four to ten episodes, which is not enough to place a rate.
Pooling validation with test doubles the exposure. It does not leak into
the model — the test block is scored once after training and enters
neither model selection nor early stopping — but it costs the test
block's independence: its operating-point metrics are then in-sample, and
the only independent check of the operating point is the blind year. That
is why the package default is the validation block alone
(`threshold_block = "validation"`) and the pooled block is an opt-in for
records that come with a separate blind product, as this one does.

**Selection.** Which configuration and seed ship was fixed on the training
year, by a rule written down before any blind score existed: the
false-alarm episode count at the fitted threshold on the validation block,
with every event of the block recovered as a constraint; counts within
one Poisson standard deviation of each other are ties, broken by the
validation ROC area and then by the smaller circuit. The blind year is
then scored once per model and every model is reported, whatever its
blind rank.

**Metrics.** An event is a contiguous run of positive labels and counts as
detected when any window inside it alarms. A false-alarm episode is a
contiguous run of alarmed windows outside every labelled span, and the
operational rate is episodes per 30 mission days. Window-level precision,
recall and ROC area are reported too, but the event-level pair is the
operating point.

**Encoding.** Every run on this page maps its features onto ``[0, \pi]``
before the ``R_z`` encoding gate (`[training] phase_span = 1.0`). The
first release used the full period ``[0, 2\pi]``: ``R_z(2\pi) = -I`` is a
global phase, so a feature clamped at the upper scaler bound was encoded
exactly as one at the lower bound, and by continuity the response
returned to its floor value as a feature approached the bound. At the
coalescence itself the band powers exceed the scaler bound by two to
three orders of magnitude, and under the full period five of the six
blind coalescences scored below the threshold in the six-hour bin holding
the merger. Measured with the retrained model, from the persisted model
and feature table, in that same bin:

| Event | Merger-window SNR | Features at the upper bound (mean per window) | Peak band power / scaler upper bound | Peak score, ``[0, 2\pi]`` model (threshold 0.826) | Peak score, ``[0, \pi]`` model (threshold 0.737) |
|---|---|---|---|---|---|
| 1 | 598 | 1.1 | 129 | 0.738 | **0.852** |
| 2 | 690 | 1.1 | 106 | 0.708 | **0.795** |
| 3 | 1312 | 2.7 | 788 | 0.682 | 0.703 |
| 4 | 799 | 1.2 | 158 | 0.764 | **0.849** |
| 5 | 272 | 0.9 | 109 | **0.861** | **0.913** |
| 6 | 723 | 1.2 | 193 | 0.585 | 0.730 |

The half period lifts the merger bin above the threshold for four of the
six coalescences. The two that stay below are the two most saturated:
event 3, the loudest, has on average 2.7 of its six band powers clamped at
the bound in that bin, and event 6 1.2. A clamped feature is now a state
distinct from the floor, but every clamped feature is the *same* state
whatever its magnitude, and a window with several bands clamped at once is
a pattern the training year offers few examples of. The scores still peak
12 to 30 hours before the merger (0.91 to 0.95 for every event), where the
band powers sit at 0.4 to 0.8 of the bound: the detector remains one of
intermediate sub-mHz excess, with a merger response that is now mostly,
not entirely, above threshold.

## Results

Ten runs, all trained on the same training year with the same protocol,
all evaluated once on the blind year with the threshold carried over
unchanged. The validation column is the selection statistic; the blind
columns were computed after the selection was recorded.

| Run | Qubits × layers, features | Epochs | Threshold | Validation episodes | Blind events | **Blind FA / 30 d** | Blind AUC |
|---|---|---|---|---|---|---|---|
| `sangria_pi` | 4 × 4, 2 bands | 37 | 0.816 | 9 | **5 / 5** | 2.64 | 0.8101 |
| `q4_l6_pi` | 4 × 6, 2 bands | 37 | 0.844 | 10 | 4 / 5 | 1.98 | 0.8139 |
| `q6_b4_pi` | 6 × 4, 4 bands | 24 | 0.765 | 8 | **5 / 5** | 5.53 | 0.7919 |
| `q6_b4_noweight_pi` | 6 × 4, 4 bands, unweighted loss | 50 | 0.426 | 7 | **5 / 5** | 9.81 | 0.8115 |
| `q6_b4_w2000_pi` | 6 × 4, 4 bands, 2000-sample windows | 36 | 0.618 | 10 | **5 / 5** | 26.39 | 0.8071 |
| **`q8_b6_pi`** | **8 × 4, 6 bands** | 30 | 0.737 | **4** | **5 / 5** | **1.57** | 0.8065 |
| `q8_b6_pi_s1009` | 8 × 4, 6 bands, seed 1009 | 20 | 0.740 | 10 | **5 / 5** | 2.06 | 0.8110 |
| `q8_b6_pi_s2027` | 8 × 4, 6 bands, seed 2027 | 36 | 0.643 | 10 | **5 / 5** | 8.74 | **0.8274** |
| `q8_b6_pi_s3041` | 8 × 4, 6 bands, seed 3041 | 29 | 0.764 | 7 | **5 / 5** | 3.30 | 0.8129 |
| `q8_b6_l6_pi` | 8 × 6, 6 bands | 50 | 0.716 | 9 | **5 / 5** | 4.78 | 0.8268 |

Every run but one recovers all five label spans — under the full-period
encoding the two-band four-qubit baseline found three of them, and now
finds five at 2.64 per 30 days. Four runs meet the three-per-30-days
target on the blind year: `q8_b6_pi`, its seed 1009, `sangria_pi`, and
`q4_l6_pi`, which misses an event. The selection rule picked `q8_b6_pi`
with four validation episodes against seven for the runner-up, three
apart, outside the tie band of two, and it is also the best operating
point on the blind year; that agreement is a result, not a guarantee.
The same configuration under the full period delivered 2.47 at a ROC
area of 0.793; the retrained one delivers 1.57 at 0.807, with a threshold
that moved from 0.826 to 0.737.

![Classifier output over the blind year](assets/benchmark_mission_trace.png)

The trace of `q8_b6_pi` over the blind year shows what the protocol is up
against. The bulk of the noise score rises between day 30 and day 250 and
falls back by day 350 — the Galactic foreground seen through the
constellation's rotating antenna pattern — while the five labelled spans
(beige) carry the peaks that clear the threshold. The threshold works
because it sits above the annual excursion, not because the events are
loud in absolute terms.

![Event recall and false alarms against the threshold](assets/benchmark_threshold_sweep.png)

The blind year's operating characteristic, drawn post hoc: event recall
holds at 1 across the whole range up to the fitted threshold, so the
operating point is limited by the false-alarm rate alone, and the fit —
made on the training year, without sight of this curve — placed it where
the rate crosses the target with room to spare.

### What the table is really showing

Order the same runs by ROC area and the ranking inverts. The shipped model
is ninth of ten by area and first by operating point; the best area,
seed 2027 of the same configuration, delivers 8.74 false alarms per 30
days:

| Run | Blind AUC | Fitted threshold | Blind FA / 30 d |
|---|---|---|---|
| `q8_b6_pi_s2027` | **0.8274** | 0.643 | 8.74 |
| `q8_b6_l6_pi` | 0.8268 | 0.716 | 4.78 |
| `q4_l6_pi` | 0.8139 | 0.844 | 1.98 (4 / 5) |
| `q8_b6_pi_s3041` | 0.8129 | 0.764 | 3.30 |
| `q6_b4_noweight_pi` | 0.8115 | 0.426 | 9.81 |
| `q8_b6_pi_s1009` | 0.8110 | 0.740 | 2.06 |
| `sangria_pi` | 0.8101 | 0.816 | 2.64 |
| `q6_b4_w2000_pi` | 0.8071 | 0.618 | **26.39** |
| **`q8_b6_pi`** | 0.8065 | 0.737 | **1.57** |
| `q6_b4_pi` | 0.7919 | 0.765 | 5.53 |

The reason is a property of the data, not of the classifier. The
classifier's noise score is modulated over the year with a period of one
year and the same phase in both records: the Galaxy sits at a fixed sky
position, LISA's response to it is modulated by the constellation's
rotation, and a single year-median Welch estimate whitens the record
correctly only on average. The two years' noise distributions therefore
agree only in the tail. Measuring the ratio of blind to training-year
noise-window exceedance at a given score, for the two runs whose
training-year scores were also computed:

| Model | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|---|---|---|
| `sangria_pi` (threshold 0.816) | 1.60 | 1.90 | 2.02 | 1.74 | 1.10 | 0.70 | 0.54 |
| `q8_b6_pi` (threshold 0.737) | 1.51 | 1.36 | 1.11 | 0.93 | 0.98 | 1.06 | — |

(`—`: the blind year has no noise window that high.)

Below about 0.65 the blind year produces one and a half to two times as
many noise exceedances as the training year; the shipped model's
threshold sits at 0.737, where the ratio is within a few per cent of one,
and its fitted rate transfers (1.38 predicted, 1.57 delivered). The
two-band baseline's tail crosses unity only above 0.75 and its threshold
lands at 0.816, in the region where the blind year is *quieter* than the
training year, which is why it transfers as well. A model whose threshold
falls in the shoulder delivers roughly the factor by which the years
disagree there. The ROC area is a rank statistic over the whole
distribution and is blind to where the tail ends, which is why it favours
the runs with the worst operating points.

Six sub-mHz bands help twice over: they compress the annual excursion of
the noise tail and they lift the event peaks above it, so the threshold
can sit in the stable region. The mechanism is that a contrast between two
band powers that are modulated together is invariant to the modulation,
and a single wide band offers no contrast to form. Whether the ordering of
the configurations generalises past this data set is not something one
record and single-seed runs can establish — the seed spread of the shipped
configuration on the selection statistic, four to ten episodes, covers
every other configuration's single seed. What does follow is the check
itself: an operating point measured on one observation record means
nothing on the next unless it sits where the two records' noise tails
agree, and no amount of ROC area substitutes for verifying that.

## Against the published method

Everything above uses whitened sub-mHz band powers, which the paper this
pipeline follows does not: Isfan et al. take the spectral entropy and the
mean, standard deviation and maximum of the raw window periodogram, with
no filtering and no whitening, on a four-qubit register. That
configuration ships as `configs/sangria_paper.toml` and was retrained on
the same two years and the same encoding (`sangria_paper_pi`), so the
published result can be compared against a reproduction of its own method
rather than only against a different one.

Isfan et al. report five of six blind mergers, the missed one being the
lowest-SNR source. The retrained parity run reaches all five label spans,
one of them by a margin of 0.002 in score — and that span holds **event
3, the loudest merger of the blind year**, at a merger-window SNR of 1312
against 272 for the quietest:

| Span | Events 1+2 | **Event 3** | Event 4 | Event 5 | Event 6 |
|---|---|---|---|---|---|
| Peak score, paper features | 0.626 | **0.559** | 0.652 | 0.595 | 0.729 |
| Peak score, band features | 0.949 | 0.930 | 0.944 | 0.913 | 0.923 |
| Threshold | 0.557 / 0.737 | | | | |

Under the paper's features the loudest event peaks at 0.559 against a
fitted threshold of 0.557: the raw periodogram moments saturate hardest
where the signal is strongest, and without whitening the feature scaler
clamps them. Under band features the same event peaks at 0.930 against a
threshold of 0.737. (Under the full-period encoding the parity run missed
the quietest event instead, which is where the paper's own miss lies;
the encoding moved the marginal case from the quiet end to the loud end.)

What the reproduction adds is the cost the paper does not report. It
publishes no false-alarm rate, so "five of six" has no operating point
attached to it. Fitted here by the same criterion as every other run in
this document, the paper's feature set delivers **31.7 false-alarm
episodes per 30 days** — 385 episodes over the blind year, an alarm
almost every day — against 1.57 for `q8_b6_pi`. Its calibration also
fails to transfer: the fit predicted 1.38 per 30 days and the blind year
returned 31.7, a factor of 23, on a held-out block whose AUC is 0.561,
barely above chance.

| | Paper features, 4 × 4 | Band features, 8 × 4 (`q8_b6_pi`) |
|---|---|---|
| Trainable parameters | 32 | 64 |
| Events | 5 / 5 spans (one at a margin of 0.002) | **5 / 5** |
| False alarms / 30 d | 31.7 | **1.57** |
| Blind AUC | 0.666 | 0.807 |
| Held-out AUC | 0.561 | 0.702 |

So the conclusion is not that the quantum classifier was improved: the
register and the ansatz are doing what they did before. **The conditioning
is what buys the operating point.** Whitening the record and partitioning
the milliHertz decade into six bands cuts the false-alarm rate by a factor
of twenty and takes the loudest event from the edge of the threshold to
well above it, at twice the parameter budget the paper reports for its
own circuit.

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
distributions disagree twofold — that year-to-year shift is the hard part
of this problem. Refitting the scaler on the blind year absorbs part of
that shift, and so does the whitening of the variational classifier: its
band powers are normalised by the blind year's own Welch estimate, band
by band, which is likewise a label-free statistic of the record under
evaluation, and whitening it with the training year's estimate instead
detects nothing (see the telemetry section). Both classifiers are
therefore evaluated on a record renormalised towards the one they trained
on, by different means; the feature scaler of the variational classifier,
fitted on the training year, is the one statistic carried across
unchanged. How much of the zero false-alarm rate the refit buys cannot be
settled from the shipped files; it is a reason to read the comparison as
indicative rather than decided.

The VQC carries 8 qubits and 4 re-uploading layers, 64 parameters against
the perceptron's 29,569 — a ratio of 460 — on one projection against five
and four features against ten.

![Window-level receiver operating characteristic](assets/benchmark_roc_curve.png)

The window-level ROC of `q8_b6_pi` on the blind year is shown for
completeness and should be read with the warning above: at 0.807 it is the
*ninth* best of the ten runs and the one that transfers. The operating
point lives at a false-positive rate of a few ``10^{-4}``, in the corner
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
A streamed mission therefore has to estimate the PSD of the record it is
scoring from the part of it that has reached the ground. The replay
quoted here does so (`psd_mode = "trailing"`, the setting of
`configs/sangria.toml`): each window is whitened by the Welch median of
the last 30 days of delivered record behind its conditioning stretch,
redone once a day, and by the training year's PSD until one 65,536-sample
segment is on the ground (the first 446 windows). The estimate the batch
benchmark uses instead — `psd_sidecar`, the year-median Welch spectrum of
the whole blind year — is an oracle for a streamed mission: at the first
event, on day 78, it contains nine months of data still in flight. The
same mission was replayed under it as well, and that replay is reported
beside the causal one as the bound a detector would reach if it knew the
year's noise in advance.

With both settled, a full year of the blind record was replayed through a
simulated mission: 63,043 batches of 500 s, a 12-hour daily pass, 1 %
packet loss with retransmission, no permanent gaps. Under ground-causal
whitening the consumer scored 62,834 windows and alarmed 629 of them;
under the oracle PSD, 572. The retrained model alarms about three times
as many windows as the full-period one did, in longer runs: the same
inspiral excess that lifts the merger bins keeps the score above the
threshold for hours around each coalescence.

![Replay of the blind year under ground-causal whitening, with alert latencies](assets/benchmark_telemetry_alerts_causal.png)

What counts as an alert has to be stated, because it decides the
latencies. The first release credited each event with the earliest
alarmed window overlapping its label span, whatever came after it. Read
that way, the causal replay alarms four coalescences 1.1 to 1.9 days
before their merger, on runs of one window, at 6.35 false-alarm episodes
per 30 days — a rate at which about four chance episodes are expected
inside the five 96-hour label spans, so the early alarms are what chance
gives. Each of them is a single window, individually indistinguishable
from the false-alarm population and impossible to act on. The alert
protocol therefore requires persistence: an alert is raised by the
arrival that completes `alert_persistence` consecutive alarmed windows,
and shorter runs are neither alerts nor charged as false alarms.

The value is fixed on the training year, not on this replay, by a rule
written down with the selection: the smallest persistence that brings the
calibration block under one false alert per 30 days — the rate below
which the LIGO–Virgo–KAGRA public alerts count as significant (Chaudhary
et al. 2024), taken as the standard because the Definition Study sets
none for its low-latency alert pipelines — that keeps every calibration
event alerted with a window of margin, and whose wait stays inside half
of the Definition Study's one-hour processing budget. On the calibration
block of the retrained model (validation and test, 109 days) the
classifier charges five alarm episodes of one to seven windows, 1.38 per
30 days, and alarms its four events for 3, 7, 13 and 46 consecutive
windows. Two consecutive windows is the value that satisfies all three
conditions: it brings the block to 0.83 per 30 days, and three would leave
the shortest event run without margin. The first release, whose model
alarmed its calibration events for at least four windows, shipped with
three. Both replays, read at that persistence and at the first alarmed
window:

| Event | Merger (mission time) | Causal PSD, two consecutive: alert − merger [h] | Causal PSD, first alarmed window | Oracle PSD, two consecutive | Oracle PSD, first alarmed window | Alert of its own |
|---|---|---|---|---|---|---|
| 1 | 2035-03-19T16:27 | **+1.9** | −25.6 | +1.9 | −25.6 | yes |
| 2 | 2035-03-20T21:59 | −27.6 | −27.7 | −27.6 | −27.7 | no: event 1's |
| 3 | 2035-08-12T15:40 | **−0.3** | −46.0 | −0.3 | −0.3 | yes |
| 4 | 2035-09-17T07:48 | **+7.5** | −40.5 | −64.4 | −64.4 | yes |
| 5 | 2035-09-21T21:39 | **+41.7** | +41.7 | −51.3 | −54.5 | yes |
| 6 | 2035-11-10T00:11 | **+15.1** | +15.1 | +15.1 | +15.1 | yes |
| False alarms / 30 d | | **1.40** | 6.35 | 1.32 | 1.65 | |

The data latencies exclude the one-hour processing budget, which
`latency_total_h` of the table adds. Under the persistence criterion and
causal whitening no alert precedes its merger by more than twenty
minutes, and the sustained alarms follow it by 2 to 42 hours: the 1.16-day
conditioning lag and the delivery latency of the lower panel, less the
hours by which the inspiral was already alarmed. Event 2 has no alert of
its own under either whitening. Its merger follows event 1's by 29.5
hours, the two label spans overlap, and every alarmed window in its span
belongs to one of event 1's episodes. The replay therefore detects five
coalescences, not six, and the batch metric's "5 of 5 label spans" is
unaffected only because the two spans merge into one.

The oracle PSD changes two alerts: events 4 and 5 are alerted 64 and 51
hours *before* their mergers, on runs of two windows in the inspiral that
the causal whitening does not produce, and 0.08 false alarms per 30 days
are saved. Two early alerts where 0.9 chance episodes are expected inside
the spans is not evidence of anything (Poisson ``p \approx 0.2``); the
year-long look-ahead sharpens the inspiral excess, and a mission does not
have it. At a persistence of three the causal replay would charge 0.74
episodes per 30 days at the same latencies, but the shortest alarm run of
a calibration event is exactly three windows, so three leaves no margin
and the protocol keeps two.

![Replay of the blind year under the oracle year-median PSD](assets/benchmark_telemetry_alerts.png)

The alarm tails after each merger and the record-edge transients have
one cause: the conditioning kernel of the record high-pass followed by
whitening decays as a power law rather than exponentially, because the
Welch estimate resolves sharp spectral features such as the TDI transfer
nulls. Its envelope is still 7 % of the peak eight window lengths from
the impulse and 99.9 % of its energy needs thirty-two. The same kernel
produces the record-edge transients that force `edge_margin` and the
post-merger alarm tails that account for most of the charged false alarms
at the operating point. Smoothing the PSD would shorten it and is the
natural next step.

## The spread under re-initialisation

The shipped configuration was trained at three further seeds inside the
same grid, each refitting its own threshold on the pooled held-out block
and applying it to the blind year without adjustment:

| Seed | Fitted threshold | Fit predicted | Delivered FA / 30 d | Events | AUC |
|---|---|---|---|---|---|
| 1009 | 0.7398 | 2.750 | 2.062 | 5 of 5 | 0.8110 |
| 2027 | 0.6432 | 2.750 | 8.741 | 5 of 5 | 0.8274 |
| 3041 | 0.7637 | 2.475 | 3.299 | 5 of 5 | 0.8129 |
| 9999 (shipped) | 0.7366 | 1.375 | 1.567 | 5 of 5 | 0.8065 |

![Threshold each run fitted and the false-alarm rate it then delivered, over four initialisation seeds](assets/benchmark_seed_spread.png)

**Every realisation recovers all five events; two of the four land under
the three per 30 days the criterion asked for.** What moves is the
false-alarm rate, by a factor of five and a half across 1.57 to 8.74,
while the ROC area moves by two and a half per cent — and in the opposite
direction: the seed with the highest area has the worst operating point.
The outlier is the seed whose threshold fitted lowest, 0.643, in the
shoulder where the two years' noise tails disagree. That is the same
lesson the model grid taught, now measured within one configuration
rather than across several: the area under the curve is nearly blind to
what the operating point delivers, because the threshold sits in the tail
and the area is dominated by the bulk.

So the headline should be read as an event recall that is stable and a
false-alarm rate that carries a factor-of-several uncertainty from
initialisation alone — on top of the Poisson uncertainty of the episodes
themselves. The shipped seed was chosen on the training year, where it is
also the best of the four (four validation episodes against seven and
ten); that the ranking held on the blind year is worth exactly one
observation. Four realisations do not measure a distribution; they bound
the scatter well enough to say that a threshold fitted in the shoulder
does not transfer, whichever seed produced it.

## What a lossy link costs

The mission above delivered everything, so the hole handling was never
put under load. Five 30-day missions answer what it costs when the link
does not: the same payload window — days 65 to 95 of the blind year,
which carries coalescences 1 and 2 — and the same pass schedule, with a
channel that drops a fraction of transfers for good rather than
retransmitting them.

| Permanent batch loss | Batches lost | Windows scored | Alarmed windows | Events |
|---|---|---|---|---|
| none | 0 of 5155 | 4946 | 307 | 2 of 2 |
| 0.14 % | 7 | 2653 | 113 | 2 of 2 |
| 0.45 % | 23 | 563 | 14 | 2 of 2 |
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
one. The whole of the sweep is that one number. At half a per cent the
replay scores a tenth of the windows and, with the retrained model's
longer alarm runs, still raises an alert on both coalescences; at one per
cent it scores none. A link losing one per cent of its batches
permanently delivers a detector that sees nothing, while the same link
losing them *temporarily* — retransmitted, as in the year-long mission,
where 648 retries cost nothing — is harmless.

Two consequences worth stating plainly. The operating requirement is on
**permanent** loss, and it is strict: about half a per cent for an alert
on a loud pair of coalescences, a fifth of a per cent for scoring most of
the record. And the smoothed
whitening PSD deferred to v1.2, which would shorten the conditioning
kernel, is not a refinement of this result but the precondition for
running on a link that loses anything at all; every reduction of the
stretch raises the tolerable loss rate in proportion.

## Caveats

- **One blind realisation of five events.** The event recall is 5 of 5 and
  the false-alarm rate is measured over 364 days, but five events do not
  measure a detection efficiency. Treat the recall as a result, not a rate.
- **One initialisation per configuration, four for the shipped one.** The
  grid runs at `seed = 9999`; the shipped configuration was repeated at
  three further seeds, whose blind rates span 1.57 to 8.74 per 30 days.
  Every other configuration's single seed lies inside the seed spread of
  the shipped one on the selection statistic, so the configuration
  ranking is not established.
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
- **The encoding is mitigated, not solved.** On ``[0, \pi]`` four of the
  six coalescences clear the threshold in their merger bin; the two most
  saturated do not, and every alert still rests on the inspiral excess of
  the hours before the merger.
- **The oracle replay is a bound, not a result.** The year-median Welch
  estimate of the blind year whitens every window of the comparison
  replay, including at the first event the nine months not yet delivered;
  the latencies quoted are those of the ground-causal replay, which a
  mission could run.
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

# Train and evaluate the shipped run; the seeds add --seed 1009 | 2027 | 3041
julia scripts/train.jl configs/experiments/q8_b6.toml --run-id q8_b6_pi
julia scripts/infer.jl configs/experiments/q8_b6.toml --run-id q8_b6_pi

# Replay a producer mission of the blind year: causal as configured, oracle
# with psd_mode = "sidecar" in [telemetry]
julia scripts/infer_telemetry.jl configs/sangria.toml --run-dir <DeepSpaceTelemetry run> \
    --model models/run_q8_b6_pi/gw_model.jld2 \
    --events data/inputs/sangria_blind_points_events.csv --run-id telemetry_year_pi
```

Training runs to early stopping in 20 to 50 epochs at roughly a minute
and a half each on 16 threads; inference over the blind year takes a
minute. The other configurations are in `configs/experiments/` and
`configs/sangria.toml`; every run on this page was trained with the
committed `phase_span = 1.0`. The batch gradient is chunked at a fixed
size and reduced in chunk order, so the thread count does not enter the
result: runs of the same seed at 8 and 16 threads reproduced each other
bit for bit.
