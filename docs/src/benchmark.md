# Sangria benchmark

This page reports the results of the classifier on the LISA Data Challenge
(LDC) 2a "Sangria" data set: one year of simulated LISA telemetry for
training and a second, blind year for evaluation, whose source catalogue
was withheld until evaluation. The blind year is scored twice:
as a completed record, and streamed through a simulated telemetry mission
in which every window is scored once the data around it has reached the
ground.

| | Selected configuration, `q8_b6` | Streaming configuration, `q8_b6_s001` |
|---|---|---|
| Completed blind year: label spans, false alarms per 30 d | 5 of 5, **1.57** | 5 of 5, 2.97 |
| Streamed year, causal whitening: coalescences alerted before the merger | 1 of 6, 1.6 h ahead; the other five after it | **5 of 6** (6 of 6 at three further seeds) |
| Alert time at the ground station | 1.6 h before to 40 h after the merger | **11 to 29 h before** |
| Streamed year: false alarms per 30 d | 0.74 | **0.41** (0.99 to 1.40 at the further seeds) |
| Conditioning lag of every alert | 1.16 days | **2.8 hours** |
| Record scored at 0.43 % scattered batch loss | 9 % | **81 %** |

The rows of the table read as follows.

- **Label spans.** The blind year holds six coalescences. Events 1 and 2
  merge 29.5 hours apart and share one label span, so the rows of the
  completed blind year count five label spans.
- **Signal onset and credited span.** The signal onset of a coalescence is
  the first window in which its signal reaches a norm of 3. The norm is
  taken over one 1000-sample window of the signal-only truth stream,
  high-passed and whitened as the record is: it is the optimal
  single-window signal-to-noise ratio (SNR) against the record's own noise,
  called below the window SNR against the record's noise. The credited
  span runs from the signal onset to the end of the label span.
- **False alarm.** An episode, not a window. On the completed record, a
  run of alarmed windows outside every label span, per 30 days of the
  record. In the streamed year, a run of at least three consecutive
  alarmed windows outside every credited span, per 30 days of scored data.
- **Alert time.** The arrival at the ground station of the batch that
  completes the alert, minus the merger time. It includes the conditioning
  lag and the downlink delay. It excludes the one-hour ground-processing
  budget.
- **Conditioning lag.** The stretch of data after a window that must be
  delivered before the window can be scored. Each window is whitened in a
  stretch of twenty window lengths on each side in the selected
  configuration, and of two in the streaming one.
- **Record scored.** Measured on a 30-day mission whose link loses 0.43 %
  of the batches permanently (nominal Bernoulli loss 0.5 %), relative to
  the lossless mission.
- **Preliminary alert and confirmation.** The streamed-year alerts are
  preliminary alerts, raised by three consecutive alarmed windows. A
  confirmation needs eleven (see the confirmation under
  [Telemetry replay and alert latency](#Telemetry-replay-and-alert-latency)).
  The streaming configuration confirms three to five of the six
  coalescences before their merger at its four seeds; the selected
  configuration confirms none before the merger.

The selected configuration, eight qubits, four re-uploading layers and six
band powers between 0.3 and 40 mHz (band edges 0.3, 0.6, 1, 2, 4, 10 and
40 mHz), was chosen on the training year by a rule fixed before
the blind year was scored. On the completed blind year it detects all five
labelled events at 1.57 false-alarm episodes per 30 mission days (19
episodes over 364 days), from a decision threshold fitted on held-out data
of the training year and applied without adjustment; the fit predicted
1.38, so the operating point transfers. Trained at four seeds, the configurations with
four or six bands from 0.3 mHz upwards recover all five events in every run, the
two-band ones in two of eight; apart from 2000-sample windows, their
false-alarm rates do not differ beyond the scatter between seeds, which
for the selected configuration spans 1.57 to 8.74 per 30 days.

The completed record is whitened by its full-record power spectral density
(PSD), the median Welch
estimate of the entire blind year: a label-free statistic, admissible for a
completed record but available only after the whole record has been
received. The streamed results are therefore quoted under causal
whitening, in which the PSD is estimated only from data already delivered
to the ground station; the stream whitened by the full-record PSD is
reported beside it as an upper reference.

Smoothing the Welch estimate in log-frequency shortens the kernel of the
whitening filter and with it the stretch of data every window needs. The
model retrained on smoothed features, `q8_b6_s001`, is the streaming
configuration: at each of four initialisation seeds it alerts five or six
of the six coalescences before their merger in the causal stream, and it
keeps scoring on a link that loses data. An alert is credited to a
coalescence only inside its credited span, from its signal onset on.
Releases up to 2.0.1 credited alerts by other onset rules and replayed a
mission paced by the wall clock; see [Corrections to earlier
releases](#Corrections-to-earlier-releases). Every run on this
page encodes its features on the half period ``[0, \pi]`` of the ``R_z``
gate (see the encoding paragraph of the protocol).

## Data and labels

LDC2a Sangria provides two one-year records of the time-delay-interferometry
(TDI) variables ``X``, ``Y``, ``Z`` at
0.2 Hz: a training product with its source catalogue and truth streams, and
a blind product whose MBHB catalogue is released separately. Both are read
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

The interpretation of the results depends on the evaluation protocol,
which is therefore stated in full.

**Blocks.** Windows overlap by 90 %, so any random split would place copies
of the same signal in both partitions. The training year is cut
chronologically into training (70 %), validation (15 %) and test (15 %)
blocks separated by one window length (`chronological_split`). The feature
scaler is fitted on the training block alone, early stopping runs on the
validation block, and the **decision threshold is fitted on validation and
test pooled** — the held-out block, 109 days (`threshold_block = "held_out"`,
the opt-in the Sangria configurations declare; the package default is the
validation block). The blind year enters twice: label-free, through the
Welch estimate that whitens it, and at evaluation, where every model is
scored once.

**Threshold.** The `far` criterion places the operating point of an alert
trigger: among the candidates whose false-alarm episode rate does not
exceed `target_far_per_30d` (3 per 30 days) and whose alarm duty cycle
does not exceed `target_fpr` (5 %), it takes the lowest of the admissible
range that starts at the highest threshold. The scan runs downwards
because the episode count is not monotone: as the threshold falls,
spurious episodes first increase in number and then merge into one
continuously raised alarm that is counted as a few long episodes, which an
upward scan would accept.

**Rationale for the pooled block.** The fitted rate rests on a Poisson
count of the alarm episodes in its block, so its relative error is the
inverse square root of that count. Across the grid below, four to ten
episodes are counted on a 55-day validation block, which is insufficient
to determine a rate. Pooling validation with test doubles the exposure.
Pooling does not leak information into the model — the test block is
scored once after training and enters neither model selection nor early
stopping — but it removes the independence of the test block: its
operating-point metrics are then in-sample, and the only independent
check of the operating point is the blind year. For this reason the
package default is the validation block alone
(`threshold_block = "validation"`) and the pooled block is an opt-in for
records that come with a separate blind product, as this one does.

**Selection.** The selected configuration and seed were determined on the
training year by a rule fixed before any blind score existed: the
false-alarm episode count at the fitted threshold on the validation block,
with every event of the block recovered as a constraint; counts within
one Poisson standard deviation of each other are ties, broken by the area
under the receiver operating characteristic (ROC area, AUC) of the
validation block and then by the smaller circuit. The blind year is
then scored once per model and every model is reported, whatever its
blind rank.

**Metrics.** An event is a contiguous run of positive labels and counts as
detected when any window inside it alarms. A false-alarm episode is a
contiguous run of alarmed windows outside every labelled span, and the
operational rate is episodes per 30 mission days. Window-level precision,
recall and ROC area are reported too, but the event-level pair defines the
operating point.

**Encoding.** Every run on this page maps its features onto ``[0, \pi]``
before the ``R_z`` encoding gate (`[training] phase_span = 1.0`). On the
full period ``[0, 2\pi]``, ``R_z(2\pi) = -I`` is a global phase, so a
feature clamped at the upper scaler bound is encoded exactly as one at
the lower bound, and by continuity the response returns to its floor
value as a feature approaches the bound. At the
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

The half period raises the score in the merger bin above the threshold for
four of the six coalescences. The two that remain below are the two most
saturated: event 3, the coalescence with the highest SNR, has on average
2.7 of its six band powers clamped at the bound in that bin, and event 6
has 1.2. A clamped feature is now a state distinct from the floor, but
every clamped feature is the *same* state whatever its magnitude, and a
window with several bands clamped simultaneously is a pattern of which the
training year contains few examples. The scores still peak 12 to 30 hours
before the merger (0.91 to 0.95 for every event), where the band powers
lie at 0.4 to 0.8 of the bound: the classifier remains a detector of
intermediate band-power excess, with a merger response that is now mostly,
but not entirely, above the threshold. On the full period the selected
configuration delivers 2.47 false alarms per 30 days on the blind year at
a ROC area of 0.793 and a threshold of 0.826, against 1.57 at 0.807 and
0.737 on the half period; the two-band four-qubit baseline recovers three
of the five label spans, against five at 2.64 per 30 days on the half
period.

## Results

Ten runs were trained on the same training year with the same protocol
and evaluated once on the blind year, with the threshold carried over
unchanged. The validation column is the selection statistic; the blind
columns were computed after the selection was recorded. Configuration
names give the qubits (`q8`: eight), the band powers (`b6`: six) or layers
(`l6`), `_s001` the 0.01-dex smoothing of the Welch estimate, `_ae` the A
and E network; a run name adds `_pi` for the encoding on ``[0, \pi]`` and
`_s<seed>` for a further initialisation seed.

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

Every run but one recovers all five label spans. Four runs meet the three-per-30-days
target on the blind year: `q8_b6_pi`, its seed 1009, `sangria_pi`, and
`q4_l6_pi`, which misses an event. The selection rule chose `q8_b6_pi`
with four validation episodes against seven for the second-ranked run, a
difference of three, outside the tie band of two; it is also the best
operating point on the blind year, an agreement that is observed here and
not guaranteed by the rule.

![Classifier output over the blind year](assets/benchmark_mission_trace.png)

The trace of `q8_b6_pi` over the blind year shows the annual variation of
the noise score against which the threshold is set. The bulk of the noise
score rises between day 200 and day 300 and falls back by day 350 — the
Galactic foreground seen through the
constellation's rotating antenna pattern — while the five labelled spans
(beige) carry the peaks that clear the threshold. The threshold is
effective because it lies above the annual excursion, not because the
events are of high amplitude in absolute terms.

![Event recall and false alarms against the threshold](assets/benchmark_threshold_sweep.png)

The operating characteristic of the blind year, computed post hoc: event
recall remains 1 over the whole range up to the fitted threshold, so the
operating point is limited by the false-alarm rate alone, and the fit —
made on the training year, without access to this curve — placed it with a
margin on the admissible side of the point at which the rate crosses the
target.

### What the table shows

Ordering the same runs by ROC area inverts the ranking. The selected
model is ninth of ten by area and first by operating point; the run with
the largest area, seed 2027 of the same configuration, delivers 8.74
false alarms per 30 days:

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
agree only in the tail. The ratio of blind to training-year noise-window
exceedance at a given score, for the two runs whose training-year scores
were also computed, is:

| Model | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|---|---|---|
| `sangria_pi` (threshold 0.816) | 1.60 | 1.90 | 2.02 | 1.74 | 1.10 | 0.70 | 0.54 |
| `q8_b6_pi` (threshold 0.737) | 1.51 | 1.36 | 1.11 | 0.93 | 0.98 | 1.06 | — |

(`—`: no noise window of the blind year reaches that score.)

Below about 0.65 the blind year produces one and a half to two times as
many noise exceedances as the training year; the threshold of the selected
model lies at 0.737, where the ratio is within a few per cent of one, and
its fitted rate transfers (1.38 predicted, 1.57 delivered). The ratio of
the two-band baseline crosses unity only above 0.75, and its threshold lies
at 0.816, in the region where the blind year produces *fewer* noise
exceedances than the training year, which is why its rate transfers as
well. A model whose threshold falls in the shoulder delivers its fitted
rate multiplied by roughly the factor by which the two years disagree
there. The ROC area is a rank statistic over the whole distribution and is
insensitive to where the tail ends, which is why it favours the runs with
the worst operating points.

Six bands, two of them below 1 mHz, have two effects: they reduce the annual excursion of
the noise tail and they raise the event peaks above it, so that the
threshold can lie in the stable region. The mechanism is that a contrast
between two band powers that are modulated together is invariant under
the modulation, whereas a single wide band provides no contrast. The
grid trained at four seeds, below, confirms the recall of the band
partitions and shows that the rates of the configurations do not differ
beyond the scatter between seeds. What does follow is
the need for the check itself: an operating point measured on one
observation record does not carry over to the next unless it lies where
the noise tails of the two records agree, and a large ROC area does not
substitute for verifying this.

## The spread under re-initialisation

The selected configuration was trained at three further seeds inside the
same grid, each refitting its own threshold on the pooled held-out block
and applying it to the blind year without adjustment:

| Seed | Fitted threshold | Fit predicted | Delivered FA / 30 d | Events | AUC |
|---|---|---|---|---|---|
| 1009 | 0.7398 | 2.750 | 2.062 | 5 of 5 | 0.8110 |
| 2027 | 0.6432 | 2.750 | 8.741 | 5 of 5 | 0.8274 |
| 3041 | 0.7637 | 2.475 | 3.299 | 5 of 5 | 0.8129 |
| 9999 (selected) | 0.7366 | 1.375 | 1.567 | 5 of 5 | 0.8065 |

![Threshold each run fitted and the false-alarm rate it then delivered, over four initialisation seeds](assets/benchmark_seed_spread.png)

**Every realisation recovers all five events; two of the four remain
below the three per 30 days that the criterion requested.** The quantity
that varies is the false-alarm rate, by a factor of five and a half from
1.57 to 8.74, while the ROC area varies by two and a half per cent, in
the opposite direction: the seed with the largest area has the worst
operating point. The outlier is the seed with the lowest fitted threshold,
0.643, which lies in the shoulder where the noise tails of the two years
disagree. This is the finding of the model grid, now measured within one
configuration rather than across several: the area under the curve is
nearly insensitive to the rate delivered at the operating point, because
the threshold lies in the tail and the area is dominated by the bulk of
the distribution.

The principal result should therefore be read as an event recall that is
stable and a false-alarm rate that carries a factor-of-several uncertainty from
initialisation alone, in addition to the Poisson uncertainty of the
episodes themselves. The selected seed was chosen on the training year,
where it is also the best of the four (four validation episodes against
seven and ten); that the ranking held on the blind year constitutes a
single observation. Four realisations do not measure a distribution; they
constrain the scatter sufficiently to show that a threshold fitted in the
shoulder does not transfer, irrespective of the seed that produced it.

### The grid over four seeds

Every other configuration of the grid was then trained at the same three
further seeds, each run refitting its own threshold on the pooled
held-out block, 28 runs in all:

| Configuration | Validation episodes, median (range) | Blind false alarms / 30 d, median (range) | Runs with 5 of 5 events |
|---|---|---|---|
| **`q8_b6`** (selected) | 8.5 (4 to 10) | 2.68 (1.57 to 8.74) | 4 of 4 |
| `q8_b6_l6` | 9.5 (9 to 10) | 4.62 (4.04 to 5.53) | 4 of 4 |
| `q6_b4` | 6.5 (1 to 8) | 3.38 (0.49 to 10.64) | 4 of 4 |
| `q6_b4_noweight` | 7.0 (5 to 7) | 5.48 (2.23 to 9.81) | 4 of 4 |
| `q6_b4_w2000` | 9.5 (8 to 10) | 25.98 (24.66 to 26.39) | 4 of 4 |
| `q4_l6` | 8.0 (6 to 10) | 1.69 (1.24 to 4.70) | 1 of 4 |
| `sangria` | 9.0 (8 to 10) | 1.73 (1.24 to 2.64) | 1 of 4 |

![The configurations of the grid at four initialisation seeds: validation false-alarm episodes and blind false-alarm rate of every run](assets/benchmark_grid_seeds.png)

Three results follow. **The partition of the spectrum decides the
recall:** the configurations with four or six bands from 0.3 mHz upwards recover all
five events in all 20 of their runs, the two-band configurations in 2 of
their 8 runs, 33 of 40 events. **The false-alarm rate differs between
configurations only through the 2000-sample windows:** over all seven the
Kruskal–Wallis statistic of the blind log rates is 15.0 (permutation
``p = 0.006``), but `q6_b4_w2000` delivers about 26 episodes per 30 days at
every seed, and without it the rates do not differ beyond the scatter
between seeds (``p = 0.23``; among the four band configurations that
always recover five events, ``p = 0.76``). **The selection statistic
cannot rank the configurations:** applied seed by seed, the rule selects
`q8_b6` at seeds 9999 and 3041, `q6_b4` at seed 2027 (0.49 per 30 days on
the blind year) and `q6_b4_w2000` at seed 1009 (25.8), where four
configurations tie on validation episodes and the tie-break by validation
ROC area chooses the worst operating point, as the inversion of ROC area
and operating point above predicts. The validation block counts 1 to 10
episodes per run, and those counts differ between configurations only
marginally (``p = 0.04``).

The selected run therefore owes its 1.57 per 30 days to its configuration
and its seed together. The configuration recovers every event at every
seed with a median of 2.68, the lowest median among the configurations
that always recover five events, but not distinguishable from `q6_b4` on
four seeds each. The tables are built by `alert_latency`-independent
metrics of each run: the validation block of its training and the blind
year of its inference.

## Against the published method

Everything above uses whitened band powers between 0.3 and 40 mHz, which the paper this
pipeline follows does not: Isfan et al. [IsfanEtAl2025](@cite) take the spectral entropy and the
mean, standard deviation and maximum of the raw window periodogram, with
no filtering and no whitening, on a four-qubit register. That
configuration is provided as `configs/sangria_paper.toml` and was retrained on
the same two years and the same encoding (`sangria_paper_pi`), so the
published result can be compared against a reproduction of its own method
rather than only against a different one.

Isfan et al. [IsfanEtAl2025](@cite) report five of six blind mergers, the missed one being the
lowest-SNR source. The retrained parity run recovers all five label spans,
one of them by a margin of 0.002 in score; that span contains **event
3, the highest-SNR merger of the blind year**, at a merger-window SNR of
1312 against 272 for the lowest-SNR one:

| Span | Events 1+2 | **Event 3** | Event 4 | Event 5 | Event 6 |
|---|---|---|---|---|---|
| Peak score, paper features | 0.626 | **0.559** | 0.652 | 0.595 | 0.729 |
| Peak score, band features | 0.949 | 0.930 | 0.944 | 0.913 | 0.923 |
| Threshold | 0.557 / 0.737 | | | | |

Under the paper's features the highest-SNR event peaks at 0.559 against a
fitted threshold of 0.557: the raw periodogram moments saturate most
strongly where the signal is strongest, and without whitening the feature
scaler clamps them. Under band features the same event peaks at 0.930
against a threshold of 0.737. (On the full period ``[0, 2\pi]`` the parity
run misses the lowest-SNR event instead, the event the paper also misses:
the encoding decides whether the marginal case lies at the low-SNR or at
the high-SNR end.)

The reproduction adds the quantity the paper does not report. The paper
publishes no false-alarm rate, so "five of six" has no operating point
attached to it. Fitted here by the same criterion as every other run in
this document, the paper's feature set delivers **31.7 false-alarm
episodes per 30 days** — 385 episodes over the blind year, about one per
day — against 1.57 for `q8_b6_pi`. Its calibration does not transfer
either: the fit predicted 1.38 per 30 days and the blind year returned
31.7, a factor of 23, on a held-out block whose AUC is 0.561, close to the
chance level.

| | Paper features, 4 × 4 | Band features, 8 × 4 (`q8_b6_pi`) |
|---|---|---|
| Trainable parameters | 32 | 64 |
| Events | 5 / 5 spans (one at a margin of 0.002) | **5 / 5** |
| False alarms / 30 d | 31.7 | **1.57** |
| Blind AUC | 0.666 | 0.807 |
| Held-out AUC | 0.561 | 0.702 |

The improvement therefore does not lie in the quantum classifier: the
register and the ansatz perform the same function in both runs. **The
operating point results from the conditioning of the data.** Whitening
the record and partitioning the milliHertz decade into six bands reduces
the false-alarm rate by a factor of twenty and raises the highest-SNR
event from the edge of the threshold to well above it, at twice the
parameter count the paper reports for its own circuit.

## Against a classical baseline

A classical multilayer perceptron trained on the same challenge data, whose
blind-year predictions I could score, detects **all five blind label spans — all six coalescences — with no
false-alarm episode**, alarming on 698 of 6,307,190 samples
(0.011 %) at a threshold of 0.5. It is a stronger result than this
classifier achieves.

**It is quoted, not reproduced.** The numbers above come from scoring those
predictions through this project's event protocol; the model was never retrained here. A reproduction would
require an effort out of proportion to what it would settle: the
preprocessing is a rolling Welch spectral entropy at unit stride (its
preprocessing script; window 1000), which amounts to 6.3 million
transforms for each of ten channel-years, and the spectral-entropy series
for X, Y and Z were not available.

Inspection of its artifacts establishes the following:

- **The model.** `Flatten → Dense(128, relu) → Dense(128, relu) →
  Dropout(0.2) → Dense(1, sigmoid)` over a 10 × 10 input — ten timesteps
  of ten features — has **29,569 parameters**, against 64 for the
  variational circuit; the quantum model does not achieve the better
  detection result.
- **The inputs.** Ten features: the raw projections X, Y, Z, A, E and the
  spectral entropy of each. Since `A = (Z − X)/√2` and
  `E = (X − 2Y + Z)/√6` are computed from X, Y and Z in that same script,
  the five projections span a three-dimensional space: the input is
  redundant by construction rather than erroneous. The null channel
  `T = (X + Y + Z)/√3`, the standard instrumental-artifact monitor,
  appears only in a comment and is not used.
- **The scaling is not blind.** Its prediction script fits a fresh
  `StandardScaler` on the blind set rather than reusing the one fitted in
  training. It uses no labels and acts on the five raw projections
  only, but its effect is not negligible: between the two years the per-column means
  differ by 0.27 to 0.65 training standard deviations and the standard
  deviations themselves by up to 12 %.

This point applies symmetrically to both classifiers. The operating-point
analysis above shows that the two observation years differ enough for a
threshold carried between them to fall in a shoulder where their noise
distributions disagree twofold; this year-to-year shift is the principal
difficulty of the problem. Refitting the scaler on the blind year absorbs
part of that shift, and so does the whitening of the variational
classifier: its band powers are normalised by the full-record PSD of the
blind year, band by band, which is likewise a label-free statistic of the
record under evaluation, and whitening the blind year with the estimate
of the training year instead yields no detection (see the telemetry
section). Both classifiers are therefore evaluated on a record
renormalised towards the one they trained on, by different means; the
feature scaler of the variational classifier, fitted on the training year,
is the one statistic carried across unchanged. The contribution of the
refit to the zero false-alarm rate cannot be determined from the released
files, and for this reason the comparison is indicative rather than
decided.

The VQC has 8 qubits and 4 re-uploading layers and 64 parameters against
the 29,569 of the perceptron (a ratio of 460), and it uses one projection
against five and eight features against ten.

![Window-level receiver operating characteristic](assets/benchmark_roc_curve.png)

The window-level ROC of `q8_b6_pi` on the blind year is shown for
completeness and should be read with the warning above: at 0.807 it is the
*ninth* best of the ten runs and the one that transfers. The operating
point lies at a false-positive rate of a few ``10^{-4}``, in the corner
of this plot where the curve has almost no resolution, which is why the
area under it is not an adequate summary.

## Telemetry replay and alert latency

The benchmark above scores a finished record. The package also couples to
a telemetry producer (see [Telemetry Coupling](telemetry.md)), replaying a
year-long mission with a realistic downlink: daily ground-station passes,
1 % packet loss with retransmission, and an on-board recorder. Windows are
scored as their data arrives and alerts are timed against the true
coalescences.

Two findings from this replay limit the contribution of this pipeline to
a low-latency alert chain; both are properties of the conditioning rather
than of the classifier:

**The whitening is zero-phase, so it is not causal.** A window's
conditioned features depend on the data after it as much as on the data
before it. A detector that scores each window as soon as its own samples
have been received detects two of the five blind events; sixty-four window
lengths of past context do not remedy this. The batch result is restored
by *centring* the window in the stretch that is whitened around it, its
conditioning stretch: twenty window lengths on each side reproduce the
batch scores at a rank correlation of 0.997, whereas thirty-two before and
twenty after reach 0.860. An alert therefore
carries a conditioning lag of twenty window lengths, 1.16
days at these settings, before any ground-segment latency; the smoothed
whitening of the section below shortens it to 2.8 hours.

**The whitening PSD calibrates a record, not a model.** Whitening the
blind year with the PSD persisted from the training year removes every
detection (none of five, at any context length), because the annually
modulated foreground makes the noise of the two records differ.
A streamed mission therefore has to estimate the PSD of the record it is
scoring from the part of it that has reached the ground. The replay
quoted here does so (`psd_mode = "trailing"`, the setting of
`configs/experiments/q8_b6.toml`): each window is whitened by the median Welch estimate of
the delivered record in the last 30 days behind its conditioning stretch,
in which the segments of every delivered run in that interval are pooled
after three cutoff periods of the record high-pass have been trimmed from
each end of the run; the estimate is recomputed once a day, and the PSD of
the training year is used until one 65,536-sample segment beyond that trim
has been delivered (the first 470 windows). The estimate the batch
benchmark uses instead, the full-record PSD (`psd_sidecar`), is non-causal
for a streamed mission: at the first event, on day 78, it contains nine
months of data not yet delivered. The same mission was also replayed under
the full-record PSD; this non-causal replay is reported beside the causal
one as an upper reference for the causal replay, the result a detector
would obtain with prior knowledge of the noise of the whole year.

With both points settled, a full year of the blind record was replayed
through a simulated mission: 63,043 batches of 500 s, a 12-hour daily
pass, 1 % packet loss with retransmission, no permanent gaps, run by
DeepSpaceTelemetry.jl 2.6.2 on its logical (tick) clock, so that the
mission is reproducible to the byte from its scenario and seed. Under
causal whitening the consumer scored 62,834 windows and alarmed 647 of
them; under the full-record PSD it alarmed 572. The alarms come in runs
lasting hours around each coalescence, raised by the inspiral excess.

![Replay of the blind year under causal whitening, with alert latencies](assets/benchmark_telemetry_alerts_causal.png)

The lower panel shows, for every window, the latency with which it became
available for scoring, the wait for its conditioning stretch (1.16 days)
plus the downlink delay of the batch that completes it (up to 18 hours),
and, for every coalescence, the alert time relative to the merger.

The definition of an alert determines the latencies and is therefore
stated explicitly.

**Crediting.** An alarm is credited to a coalescence only where the
signal of the coalescence can be seen. The label span, four days before
to 27 minutes after the merger, is the training convention of Isfan et
al. [IsfanEtAl2025](@cite) and is far wider than that: the matched-filter SNR of the signal-only
A channel in a single 1000-sample window, against the analytic Sangria
PSD without the Galactic confusion noise and therefore an upper bound,
stays below 3 up to 40 hours before each merger and first reaches 5
between 35 and 7 hours before it.

| Event | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Window SNR against the instrument PSD, 74 h before the merger | 2.0 | event 1 | 1.5 | 1.5 | 0.3 | 1.4 |
| Window SNR against the instrument PSD, 50 h before the merger | 2.8 | event 1 | 2.3 | 2.5 | 0.6 | 2.1 |
| First window at SNR 5 against the instrument PSD [h before the merger] | 34.8 | 29.1 | 27.8 | 32.2 | 7.4 | 27.4 |
| Signal onset, first window at a norm of 3 against the record's noise [h before the merger] | 37.1 | 29.1 | 38.1 | 35.7 | 9.8 | 33.2 |

Event 2 merges 29.5 hours after event 1, whose signal dominates the truth
stream until then, so the onset of event 2 is sought past the span of
event 1. An alert is credited from the onset to the end of the label span
(`alert_crediting = "signal"`; the labelling stage writes the onset as
`signal_start_index`), and alarm runs earlier in the label span count as
false alarms.

Releases up to 2.0.1 credited alerts by other onset rules and replayed a
mission paced by the wall clock; see [Corrections to earlier
releases](#Corrections-to-earlier-releases).

**Persistence.** An isolated alarmed window cannot be distinguished from
the false-alarm population, so an alert is raised by the arrival that
completes `alert_persistence` consecutive alarmed windows; shorter runs
are neither alerts nor counted as false alarms. The value is fixed on the
training year, not on this replay: the smallest persistence that brings
the calibration block, credited from the signal onsets of the training
year and pooled over the trained seeds, below one false alert per 30 days
— the rate below which the LIGO–Virgo–KAGRA public alerts count as
significant (Chaudhary et al. 2024) [ChaudharyEtAl2024](@cite), taken as
the standard because the LISA Definition Study Report
[ColpiEtAl2024](@cite) sets none for its low-latency alert pipelines —
with a waiting time within half of the Definition Study Report's one-hour
processing budget. On the calibration block of the selected model
(validation and test, 109 days) two consecutive windows leave 1.10 false
alerts per 30 days and three 0.55: the persistence is three. A further
criterion, that every calibration event keep an alert with a window of
margin, would fix two; with four calibration events it rests on a single
short alarm run, and it is not applied. At three, the shortest alarmed
calibration event, of three windows, keeps its alert.

**Confirmation.** Windows of 1000 samples every 100 put each sample in ten
consecutive windows, so a transient shorter than a step can alarm ten
consecutive windows and raise an alert at any persistence up to ten; a
persistence of three cannot tell such a transient from a signal. Raising
the persistence beyond ten removes most alerts of instrumental glitches,
but the alarms of a classifier of band powers on an early inspiral come in
short runs as well, so it also removes the early alerts of the
coalescences (on this year the alert of event 1 by the streaming
configuration moves from 23 to 5 hours before the merger). The alert of
the persistence rule is therefore kept as a preliminary alert, and a
second tier confirms it: `alert_confirmation_persistence = 11`, the
arrival that completes eleven consecutive alarmed windows touching the
credited span, which needs alarms on windows sharing no sample. The value
follows from the window geometry and is not fitted; the calibration block,
which holds no glitches, cannot set it.

On the year mission no confirmation falls outside a credited span, for any
configuration, seed or whitening. The streaming configuration confirms
three to five of the six coalescences before their merger, the A and E
network five or six. The selected configuration confirms five
coalescences 3 to 40 hours after their merger and not event 3. Over the
seeds of the streaming configuration and of the A and E network:

| Event | Confirmation before the merger [h] |
|---|---|
| 1 | about 5.3 (alerted 23 to 29 hours before it) |
| 2 | 8 to 11 |
| 3 | 2 to 4 (24.5 at one seed) |
| 4 | 12 to 21 (after the merger at one seed of A) |
| 5 | 2 to 3 at three seeds of the A and E network; after the merger otherwise |
| 6 | 10 to 13 |

On the LDC-2b year with glitches, without coalescences, the confirmation
removes most of the glitch alerts of the streaming configuration that the
angle encoding lets through (see [Instrumental
glitches](#Instrumental-glitches)); at seeds 9999, 1009, 2027 and 3041:

| Mode | Preliminary alerts | Confirmations |
|---|---|---|
| A | 19, 14, 0, 0 | 5, 1, 0, 0 |
| A and E | 18, 12, 0, 14 | 7, 6, 0, 0 |

Both replays at that persistence, credited from the signal onset; alert
time minus merger time [h]:

| Event | Merger (mission time) | Signal onset | Causal PSD | Full-record PSD |
|---|---|---|---|---|
| 1 | 2035-03-19T16:27 | −37.1 | +1.9 | +2.1 |
| 2 | 2035-03-20T21:59 | −29.1 | +15.9 | +15.9 |
| 3 | 2035-08-12T15:40 | −38.1 | −1.6 | −0.4 |
| 4 | 2035-09-17T07:48 | −35.7 | +6.0 | +6.0 |
| 5 | 2035-09-21T21:39 | −9.8 | +40.2 | +40.2 |
| 6 | 2035-11-10T00:11 | −33.2 | +13.7 | +13.7 |
| False-alarm episodes (per 30 d) | | | **9 (0.74)** | 8 (0.66) |

The alert time is the arrival at the ground station of the batch that
completes the alert, so it contains the conditioning lag and the downlink
delay; the latencies exclude the one-hour processing budget, which
`latency_total_h` of the table adds. Under causal whitening every
coalescence is alerted on its own, five of them after their merger, 1.9 to
40 hours later, and event 3 1.6 hours before it. A window becomes
available for scoring only once the 1.16 days of data after it have been
delivered, so even a window at the signal onset, 29 to 38 hours before the
merger for five of the six coalescences, is scored at the earliest around
the merger. The full-record PSD alerts the same coalescences, event 3 1.2
hours later and the others at the same times, at 0.66 false-alarm
episodes per 30 days.

![Replay of the blind year under the full-record PSD](assets/benchmark_telemetry_alerts.png)

The alarm tails after each merger and the record-edge transients have
one cause: the conditioning kernel of the record high-pass followed by
whitening decays as a power law rather than exponentially, because the
Welch estimate resolves sharp spectral features such as the TDI transfer
nulls. Its envelope is still 7 % of the peak eight window lengths from
the impulse and 99.9 % of its energy is contained within thirty-two. The
same kernel produces the record-edge transients that necessitate
`edge_margin` and the post-merger alarm tails that account for most of the
counted false alarms at the operating point. Smoothing the Welch estimate by 0.01 dex in
log-frequency (`[preprocessing] psd_smoothing_dex`) lowers the envelope
at one window length from 21 % to 0.1 % of the peak, because the length
of the kernel is set by the estimator's line-to-line scatter and the
sharp TDI transfer notch rather than by the shape of the spectrum. Applied
to the blind year with the selected model and its threshold unchanged, it
keeps five of five label spans but raises the false-alarm rate from 1.57
to 4.78 episodes per 30 days (ROC area 0.800 against 0.807): the fitted
threshold does not transfer to features whitened differently, and the
shorter kernel can be used only by a model trained on smoothed features.

## Effect of a lossy link

The year-long mission delivered every batch, so the handling of delivery
holes was not exercised. Seventeen 30-day missions on DeepSpaceTelemetry.jl
2.0.0 (the two generation-gap missions on 2.1.1) measure the effect of a
link that does lose data. They share the same payload window — days 65
to 95 of the blind year, which contains coalescences 1 and 2 — and the
same pass schedule; each mission differs
from a lossless reference in one property of the channel or of the
spacecraft and was replayed once by the selected model under the causal
whitening of `configs/experiments/q8_b6.toml` (persistence 3,
`context_windows = 20`, alerts credited from the signal onset) and, for
five of them, under the full-record PSD of the whole blind year. The
reference replay scores 4,945 windows, alerts both coalescences, each after
its merger, and counts 3.1 false-alarm episodes per 30 days; under the
full-record PSD it counts 1.0. The difference arises from the initial
interval of the trailing estimate on a 30-day record, before one Welch
segment of delivered data exists (over the year the causal replay
delivers 0.74 against 0.66 under the full-record PSD), and it is the
scale against which the rows below are read.

![The seventeen missions and two consumer-side variants: windows scored as a fraction of the reference, coalescences detected, false-alarm episodes per 30 days under causal whitening](assets/benchmark_gap_study.png)

| Channel or spacecraft | Batches lost | Windows scored | Events | False alarms per 30 d | Full-record PSD |
|---|---|---|---|---|---|
| lossless reference | 0 | 4945 | 2 of 2 | 3.1 | 1.0, 2 of 2 |
| scattered permanent loss, 0.04 % realised | 2 | 4285 | 2 of 2 | 13.3 | |
| scattered permanent loss, 0.08 % | 4 | 3772 | 2 of 2 | 6.9 | 1.4, 2 of 2 |
| scattered permanent loss, 0.19 % | 10 | 2120 | 0 of 2 | 12.2 | 2.4, 2 of 2 |
| scattered permanent loss, 0.43 % | 22 | 426 | 2 of 2 | 12.2 | |
| bursty loss, mean 0.3 %, bursts of 4 | 21 | 3485 | 2 of 2 | 4.5 | 1.5, 2 of 2 |
| bursty loss, mean 0.3 %, bursts of 20 | 9 | 4475 | 2 of 2 | 3.5 | |
| bursty loss, mean 0.3 %, bursts of 50 | 0 | 4945 | 2 of 2 | 3.1 | |
| 3 % loss, one retransmission | 6 | 2475 | 0 of 2 | 6.3 | 0.0, 2 of 2 |
| 3 % loss, three retransmissions | 0 | 4946 | 2 of 2 | 3.1 | |
| link outage 6 h, day 3 | 0 | 4946 | 2 of 2 | 3.1 | |
| link outage 24 h, day 3 | 0 | 4946 | 2 of 2 | 3.1 | |
| link outage 72 h, day 3 | 0 | 4946 | 2 of 2 | 9.4 | |
| link outage 24 h astride merger 1 | 0 | 4946 | 2 of 2 | 2.1 | |
| no data produced for 15 min, day 3 | 0 | 4533 | 2 of 2 | 8.0 | |
| no data produced for 2 h astride merger 1 | 0 | 4521 | 0 of 2 | 3.4 | |
| two-day recorder with a 72 h outage | 0 | 4277 | 2 of 2 | 3.6 | |
| 0.19 % loss, stretch admitted at 75 % delivered | 10 | 5028 | 2 of 2 | 4.1 | |
| 0.19 % loss, stretch admitted at 50 % delivered | 10 | 5030 | 2 of 2 | 2.1 | |

**Scattered permanent loss is the property that reduces the scored
record, and its effect is not proportional to the loss.** A window is
scored only once its **entire conditioning stretch** has reached the
ground: `context_windows = 20` window lengths on each side of a
1000-sample window, and one batch carries one window step of 100 samples,
so 410 consecutive batches must all be delivered. Under an independent
per-batch loss `p` a window therefore survives with probability
`(1 - p)^410`, drawn as the dashed curve below against the loss the
channel realised; the measured survival follows it while losses are
sparse and falls below it once the mean spacing of losses, `1/p` batches,
approaches the stretch: above `1/410 ≈ 0.24 %` an uninterrupted run of
410 batches is rare rather than typical. Whether a coalescence is detected
depends on where the holes fall, not on the rate alone: both are alerted
at 0.04, 0.08 and 0.43 % and neither at 0.19 %, where the replay under
the full-record PSD still alerts both, a day after each merger, on the
few windows remaining around them.

![Windows a replay can score, and the events it still detects, against the permanent batch loss the channel realised](assets/benchmark_loss_survival.png)

**The same losses, clustered in bursts, remove far fewer windows.**
Twenty-one batches lost in bursts of mean length four remove 30 % of the
windows; twenty-two scattered losses remove 91 %. One burst of nine
removes 10 %. The survival is set by the number of holes, each of which
excludes the windows whose conditioning stretch contains it, and not by
the number of batches.

**Retransmission removes the effect.** A 3 % channel with one retry leaves
six permanent holes, and the replay scores half the windows and detects
neither coalescence; with three retries it leaves none, and the replay
reproduces the reference to within one window.

**A link outage increases the latency and removes no window.** The
recorder holds the backlog, and every window is scored once its stretch
has been delivered.
An outage of 24 h astride merger 1 delays its alert to 73 h after the
merger — the data reached the ground after the outage and its backlog —
and the second alert to 20 h after merger 2. The delivery order after an
outage does change the causal estimate, which is a function of what has
been delivered: the 72 h outage replay differs from the reference in every
score (9.4 false alarms per 30 days, among them an alarm run 50 h before
merger 1), although it received the same batches.

**A generation gap removes the stretch around it, and with the selected
model's long stretch a neighbouring coalescence too.** Fifteen minutes
without data on day 3 remove 412 windows, the stretch around the hole;
the hole also enters the 30-day trailing estimate of every later window
and shifts the later scores by 0.02 in the median, so that the replay
counts 8.0 false alarms per 30 days against 3.1 and alerts event 1 2.7 h
before its merger instead of 1.8 h after. Two hours without data centred
on merger 1 remove 424 windows and both coalescences: the stretch of
twenty window lengths (27.8 h) on each side of the hole reaches past
merger 2, 29.5 h later, and contains the windows that carried both alerts
of the reference. The smoothed model of the next section, whose stretch
is two window lengths on each side, alerts both. A two-day recorder that overflows
during a 72 h outage discards 260 batches of production and removes 668
windows, and both events are still detected.

The two generation-gap missions were regenerated on 30 September 2026;
see [Corrections to earlier releases](#Corrections-to-earlier-releases).

**Admitting a partially delivered stretch scores every window and brings
the alerts forward.** With the stretch admitted at 75 or 50 % delivered
the replay scores 5,028 and 5,030 windows (more than the reference,
because edge windows now qualify) at 4.1 and 2.1 false alarms per 30
days, and alerts both coalescences, event 1 6.7 and 23.2 h before its
merger, after its signal onset, on windows conditioned on a partial
stretch.

The operating requirement is on **scattered permanent** loss, and it is
strict: about a fifth of a per cent for scoring most of the record, and
above that rate the detection of a given coalescence depends on where the
holes fall. Every reduction of the conditioning stretch raises the
tolerable loss rate in proportion: with the Welch PSD of the selected
configuration the impulse response of the high-pass followed by whitening
keeps 99.9 % of its energy only within 55 window lengths, whereas the same
PSD smoothed by 0.01 dex in log-frequency keeps it within half a window;
the line structure of the estimator and the exact TDI notch, not the
shape of the spectrum, determine the kernel. A smoothed whitening PSD is
therefore a precondition for operation on a link that loses any data,
rather than a refinement of this result; the model trained on the
smoothed whitening is reported in the section [Smoothed
whitening](#Smoothed-whitening).

## Smoothed whitening

The conditioning stretch of twenty window lengths, and with it the
conditioning lag of 1.16 days and the tolerance to scattered loss, is set
by the kernel of the record high-pass followed by whitening, whose length
comes from the line structure of the Welch estimate. The configuration
`configs/experiments/q8_b6_s001.toml` is the selected configuration with
the Welch estimate smoothed by 0.01 dex in log-frequency, for the training
and blind features and for the trailing estimate of the replay. Its model
`q8_b6_s001_pi` was trained on the smoothed features with the seed and the
threshold rule of the selected run, and the blind year was scored once; it
was then trained at the three further seeds of the grid. It is the
streaming configuration of the package.

| | Selected, `q8_b6_pi` | Smoothed, `q8_b6_s001_pi` |
|---|---|---|
| Validation events, episodes | 3/3, 4 | 3/3, 5 |
| Test-block event | 1/1 | 0/1 |
| Blind label spans | 5/5 | 5/5 |
| Blind false-alarm episodes (per 30 d) | 19 (1.57) | 36 (2.97) |
| Blind ROC area | 0.807 | 0.782 |

In the batch benchmark the smoothed model recovers the five label spans at
2.97 false-alarm episodes per 30 days, within the range of 1.57 to 8.74
that re-initialisation alone produces for the selected configuration; with
five validation episodes against four and the test-block event missed, the
selection rule of the grid would not have preferred it. Its advantage is
in the streamed setting. At the four seeds:

| Seed | Epochs | Threshold | Blind label spans | Blind false-alarm episodes (per 30 d) | ROC area |
|---|---|---|---|---|---|
| 9999 | 16 | 0.630 | 5/5 | 36 (2.97) | 0.782 |
| 1009 | 13 | 0.624 | 5/5 | 48 (3.96) | 0.779 |
| 2027 | 15 | 0.626 | 5/5 | 82 (6.76) | 0.783 |
| 3041 | 17 | 0.638 | 5/5 | 48 (3.96) | 0.786 |

**Conditioning length.** Every fifth blind-year window was streamed from
a stretch of *c* window lengths on each side, whitened by the full-record
PSD, and compared with its batch score:

| *c* | Stretch [samples] | Median \|Δ\| | 99th percentile \|Δ\| | ROC area | Rank correlation |
|---|---|---|---|---|---|
| 1 | 3,000 | 0.0004 | 0.0049 | 0.782 | 1.000 |
| 2 | 5,000 | 0.0002 | 0.0033 | 0.784 | 1.000 |
| 4 | 9,000 | 0.0001 | 0.0017 | 0.783 | 1.000 |
| 20 | 41,000 | 0.0000 | 0.0001 | 0.783 | 1.000 |

The streamed scores agree with the batch scores from one window length
on; the selected model needs twenty for a rank correlation of 0.997 and a
99th percentile of 0.022, and its agreement is not monotone below. At the
three further seeds two window lengths give a rank correlation of 1.000
and a 99th percentile of 0.0029 to 0.0036. The configuration uses two
window lengths: a window needs 50 delivered batches instead of 410, and
the conditioning lag falls from 1.16 days to 2.8 hours.

**Persistence.** By the rule of the telemetry section, the calibration
blocks of the four seeds, credited from the signal onset and pooled, count
1.17 false alerts per 30 days at two consecutive windows and 0.76 at
three: the persistence is three. Seed by seed the rule would give three, two, four
and two (seeds 9999, 1009, 2027 and 3041). At three, two of
the twelve alarmed calibration events, each alarmed for two windows (seeds
1009 and 3041), are not alerted, and each seed leaves one of its four
calibration events unalarmed.

**Year replay.** The lossless year-long mission of the previous sections,
replayed with the four runs at persistence three and credited from the
signal onset; alert time minus merger time [h], with the alert time the
arrival at the ground station of the batch that completes the alert:

| Event | Signal onset | Seed 9999 | Seed 1009 | Seed 2027 | Seed 3041 |
|---|---|---|---|---|---|
| 1 | −37.1 | **−22.8** | −22.8 | −28.6 | −22.8 |
| 2 | −29.1 | **−11.5** | −11.5 | −11.5 | −11.3 |
| 3 | −38.1 | **−28.7** | −28.7 | −28.7 | −29.3 |
| 4 | −35.7 | **−21.3** | −21.4 | −21.5 | −21.4 |
| 5 | −9.8 | not alerted | −2.1 | −2.1 | −2.1 |
| 6 | −33.2 | **−12.0** | −12.1 | −12.0 | −12.0 |
| False-alarm episodes (per 30 d) | | **5 (0.41)** | 12 (0.99) | 17 (1.40) | 15 (1.23) |
| Full-record PSD: alerted before the merger; false alarms per 30 d | | 5, 0.41 | 5, 1.48 | 5, 2.39 | 6, 1.32 |

![Replay of the blind year under causal whitening by the streaming configuration, seed 9999](assets/benchmark_telemetry_alerts_smoothed.png)

Under causal whitening every seed alerts events 1 to 4 and 6 on their own
and before their merger, 11 to 29 hours ahead at the ground station and
8 to 21 hours after the signal onset. Event 5, the weakest coalescence,
whose norm reaches 3 only 9.8 hours before its merger, is alerted
2.1 hours before it at three seeds and not at all at seed 9999. The
false-alarm rate spans 0.41 to 1.40 episodes per 30 days; the credited
spans cover 7.7 days of the year, in which about 0.1 to 0.36 chance
episodes are expected at these rates, and every alert follows the signal
onset of its coalescence. The alert times agree between the seeds to
within six hours where they alert; the false-alarm rate does not, and it is
the quantity the reference seed understates. Under the full-record PSD the same
coalescences are alerted, event 5 after its merger or not at all at three
seeds, at 0.41 to 2.39 per 30 days. The score baseline of the smoothed
model follows the annual modulation of the noise level, which that of the
selected model does not: under the full-record PSD it falls towards 0.1 at
the beginning and at the end of the year and approaches the threshold
between days 200 and 300; the trailing estimate follows part of this
modulation.

**Lossy link.** The seventeen 30-day missions of the previous section,
replayed with the four smoothed runs at persistence three and credited
from the signal onset; seed 9999 against the selected model:

| Mission, nominal loss | Permanent loss, realised | Windows scored, selected | Windows scored, smoothed | Coalescences alerted, selected | Coalescences alerted, smoothed |
|---|---|---|---|---|---|
| Lossless reference | 0 | 4,945 | 5,125 | 2, after the mergers | 2, before the mergers |
| Bernoulli 0.1 % | 0.04 % | 4,285 (0.87) | 5,024 (0.98) | 2 | 2 |
| Bernoulli 0.2 % | 0.08 % | 3,772 (0.76) | 4,926 (0.96) | 2 | 2 |
| Bernoulli 0.3 % | 0.19 % | 2,120 (0.43) | 4,636 (0.90) | 0 | 2 |
| Bernoulli 0.5 % | 0.43 % | 426 (0.09) | 4,142 (0.81) | 2 | 2 |

The fraction of the reference record the smoothed model scores follows the
independent-batch estimate `(1 − p)^50` (0.98, 0.96, 0.91, 0.81). Over all
seventeen missions every seed alerts both coalescences on their own and
before their merger, event 1 6 to 29 hours and event 2 10 to 11 hours
ahead, at 0 to 7.1 false-alarm episodes per 30 days (means of 0.65 to
2.49 per seed), where the selected model counts 2.1 to 13.3, alerts after
the merger, and misses at least one coalescence in three missions. The
false alarms fall because the long kernel of the unsmoothed whitening
produced the alarm tails after each merger. The replay results are those
of the `gapreplay_<mission>_s001[_s<seed>]` runs.

The selected configuration of the batch benchmark is unchanged. For
streamed operation and for a link that loses data, the smoothed
configuration, at four seeds, removes the two limitations of the selected
one, the conditioning lag and the loss rate above which windows can no
longer be scored, and it is the streaming configuration of the package.

## The A and E network

The channel E carries, on equal arms, the other noise-orthogonal view of a
source: the squared signal-to-noise ratios of A and E add, and how a
source divides between them depends on its sky position and polarisation.
In the channel mode `[tdi] channels = "AE"` the two channels are whitened
each by its own PSD estimate and the classifier receives, of every one of
its eight features, the value farthest towards a signal among the two —
the larger band power and power spread, the smaller entropy — so the
circuit, the labels and the protocol are those of the A mode. The
alternative combination, the features of the averaged periodogram
(`channel_combination = "mean"`), halves the excess power of a source that
one channel sees while the noise spread falls only by ``\sqrt 2``; on the
completed blind record it delays the alert of event 4, which A carries
almost alone (peak window SNR 1,127 in A against 213 in E), by 12 to 21
hours, which the extreme of the features does not. The configurations
`configs/experiments/q8_b6_ae.toml` and `q8_b6_s001_ae.toml` are the
selected and the streaming configuration in this mode. Each was trained at
the four seeds with its threshold rule, and the blind year was scored once
per run.

| Configuration | Seed | Epochs | Threshold | Blind label spans | Blind false-alarm episodes (per 30 d) | ROC area |
|---|---|---|---|---|---|---|
| `q8_b6_ae` | 9999 | 39 | 0.718 | 5/5 | 33 (2.72) | 0.812 |
| | 1009 | 25 | 0.715 | 5/5 | 32 (2.64) | 0.813 |
| | 2027 | 28 | 0.726 | 5/5 | 27 (2.23) | 0.804 |
| | 3041 | 29 | 0.738 | 5/5 | 32 (2.64) | 0.785 |
| `q8_b6_s001_ae` | 9999 | 23 | 0.612 | 5/5 | 40 (3.30) | 0.767 |
| | 1009 | 26 | 0.593 | 5/5 | 96 (7.92) | 0.783 |
| | 2027 | 15 | 0.632 | 5/5 | 104 (8.58) | 0.749 |
| | 3041 | 13 | 0.613 | 5/5 | 82 (6.76) | 0.753 |

On the completed record the second channel changes little: every run
recovers the five label spans; the false-alarm rate of `q8_b6_ae` lies
between 2.23 and 2.72 episodes per 30 days, inside the range of 1.57 to
8.74 of the A configuration and with a smaller spread over the seeds, and
that of `q8_b6_s001_ae` between 3.30 and 8.58 against 2.97 to 6.76.

**Signal onsets.** The labelling stage writes, beside the onset of A, the
onset of the network: the first window at which the root of the summed
squared norms of A and E reaches the threshold of 3
(`signal_start_index_ae`). On the six blind coalescences it precedes the
onset of A by 0.1, 0.0, 10.4, 0.5, 11.8 and 15.7 hours; for event 5, the
weakest in A (peak window SNR 86), the network SNR peaks at 227 and the
onset moves from 9.8 to 21.6 hours before the merger. Alerts of a model of
this mode are credited from the onset of the network.

**Year replay.** The lossless year-long mission, run again with both
channels as the two columns of its payload (below), replayed with the four
runs of `q8_b6_s001_ae` under causal whitening (one trailing estimate per
channel), at persistence three; alert time minus merger time [h] at the
ground station:

| Event | Onset of the network | Seed 9999 | Seed 1009 | Seed 2027 | Seed 3041 |
|---|---|---|---|---|---|
| 1 | −37.2 | −26.4 | −29.4 | −26.4 | −26.5 |
| 2 | −29.1 | −11.5 | −11.5 | −11.5 | −11.5 |
| 3 | −48.5 | −26.7 | −28.7 | −28.8 | −28.8 |
| 4 | −36.2 | −21.4 | −21.4 | −21.4 | −21.4 |
| 5 | −21.6 | −4.5 | −4.5 | −4.6 | −4.6 |
| 6 | −48.9 | −13.8 | −13.8 | −12.6 | −13.2 |
| False-alarm episodes (per 30 d) | | 11 (0.90) | 54 (4.44) | 17 (1.40) | 18 (1.48) |

Every seed alerts the six coalescences before their mergers; the A
configuration misses event 5 at the reference seed. Event 5 is alerted 4.5
to 4.6 hours ahead against 2.1, event 6 0.6 to 1.8 hours earlier, event 1
3.6 to 6.6 hours earlier at three seeds and 2.2 hours later at one; events
2 and 4 are alerted within 0.2 hours of the A configuration, and event 3
at the same time or up to 2.0 hours later. The false-alarm rate is
0.90 to 1.48 episodes per 30 days at three seeds, about the range of the A
configuration (0.41 to 1.40), and 4.44 at seed 1009, whose rate on the
completed record is also among the highest. Four seeds do not rank the
two modes by false-alarm rate.

**The link.** The year mission of this section carries both channels as
the two columns of one payload (`[tdi] channels = "AE"` in the export;
`physics.external_data_columns = ["A", "E"]` of the producer): one batch,
one transfer and one loss draw for both, so that the replay reads A and E
of every delivered batch from the run itself. Run from the same scenario
and seed as the one-column mission of the A configuration, it delivers
the same batches at the same instants — the arrival feed, the declared
holes and the delivery delays of the two runs are identical byte for
byte — and the A configuration replayed on it scores every window as on
the one-column mission. The delivery of a batch is common to its channels,
as it is on LISA: the combinations A, E and T are formed on the ground
from the same telemetry, about 75 kbit/s for the whole constellation,
science data and housekeeping included [ColpiEtAl2024](@cite); the volume
on the link does not depend on the number of combinations an analysis
reads, and a lost packet is lost to all of them. The missions of the
lossy-link study carried the A channel alone; their replays by this mode
serve E from the TDI record, after checking batch by batch that the A
column of the record is the payload the mission delivered
(`scripts/infer_telemetry.jl --tdi-file`, `ScheduledRecordRun`).

**A saturated link.** A mission whose production is doubled against the
same link is therefore not the scenario of a second channel but of a link
that cannot carry its production, and was run as such: the two-column
mission at 150 kbit/s against the 230 kbit/s of the pass of 12 hours,
132 batches per pass against 173 produced per day. Over the year 76.2 %
of the batches reach the ground; the on-board recorder of 30 days is full
on day 125, and from then on the data between about 02:00 and 07:55 of
every day are never recorded, as before that the data between 19:00 and
01:40 never leave the recorder; 75.4 % of the year is on the ground within
24 hours of its measurement, against 99.96 % at the nominal rate. The
merger of event 2 is in a batch that is never delivered, and the merger of
event 4 with the 5.3 hours before it is never recorded. Replayed on this
mission (`--link-volume-channels 2`), the four runs of either mode score
47 % of the windows of the year:

| | Nominal rate, A | Doubled rate, A | Nominal rate, A and E | Doubled rate, A and E |
|---|---|---|---|---|
| Coalescences alerted before the merger, by seed | 5, 6, 6, 6 | 5, 6, 5, 6 | 6, 6, 6, 6 | 6, 6, 6, 6 |
| False-alarm episodes per 30 d of scored record | 0.41 to 1.40 | 0.53 to 2.98 | 0.90 to 4.44 | 0.35 to 2.28 |
| Alert before the merger [h], seed 1009, events 1 to 6 | 22.8, 11.5, 28.7, 21.4, 2.1, 12.1 | 22.7, 9.0, 24.2, 16.3, 2.2, 8.8 | 29.4, 11.5, 28.7, 21.4, 4.5, 13.8 | 26.2, 9.0, 24.2, 15.9, 5.3, 8.8 |
| Coalescences confirmed before the merger, by seed | 3, 5, 5, 5 | 2, 3, 3, 2 | 5, 6, 6, 6 | 2, 5, 4, 2 |

The alerts survive: the models of the A and E network alert the six
coalescences at every seed, up to 9 hours later than on the nominal link,
and those of A five or six. The confirmations do not: eleven consecutive
windows need eleven consecutive scored windows, and the daily hole cuts
every alarm run that reaches it, so two to five coalescences are confirmed
before their merger where the nominal link confirms three to six; no
confirmation falls outside a credited span. What the saturated link loses
is the data, a quarter of the year and two of the six mergers among it,
and with them the requirement that data reach the ground within 24
hours.

**Lossy link.** The seventeen 30-day missions of the lossy-link section,
replayed with the four runs of `q8_b6_s001_ae`; their payload is the A
channel of days 65 to 95, so the record is served from its row 1,123,201
(`--tdi-first-row`). The delivery is that of the A replays and the windows
scored are the same, mission by mission. Every seed alerts both
coalescences before their mergers in every mission: event 1 22.5 to 29.3
hours ahead, and 5.5 hours in the mission with the recorder of two days at
one seed, where two of the A runs alert at 5.6 hours; event 2 9.5 to 11.6
hours ahead, as the A runs do. The false-alarm rate over these thirty days
is 0 to 3.1 episodes per 30 days at two seeds, 1.0 to 8.1 at a third, and
2.5 to 12.3 at seed 1009 (means of 1.12, 7.15, 2.14 and 1.49 for seeds
9999, 1009, 2027 and 3041, against 1.05, 0.65, 2.49 and 1.64 for A): the
run with the highest rate over the year has the highest here. The replay
results are those of the `gapreplay_<mission>_s001_ae[_s<seed>]` runs.

The A and E network is a mode of the package; the streaming configuration
is unchanged. Its gain on this year is the weakest coalescence, alerted at
every seed and two hours earlier, and it costs nothing in the tolerance to
a lossy link; its false-alarm rate depends on the seed more than that of
the A configuration.

## Classical controls on the same features

The perceptron quoted above differs from the circuit in its inputs, its
preprocessing and its size, so that comparison says little about the
circuit itself. The controls of this section differ from it in the
classifier alone. `[model] kind = "logistic"` puts a logistic regression,
`kind = "perceptron"` a network with one hidden layer of six tanh units,
and `kind = "harmonic"` a logistic regression on the cosine and sine of the
encoded angles — the bounded, periodic feature map of the circuit, without
the circuit — in the place of the circuit: on the same eight features after
the same scaler, initialised from the same distribution, trained by the
same stage — blocks, weighted cross-entropy, optimiser, early stopping —
with the threshold fitted by the same rule on the same block, the blind
year scored once, and the year mission replayed under the same telemetry
settings. The logistic regression has 9 parameters, the harmonic control
17 and the network 61, against the 64 of the circuit. The configurations
are `configs/experiments/q8_b6_s001_logistic.toml`,
`q8_b6_s001_perceptron.toml` and `q8_b6_s001_harmonic.toml` for the
streaming configuration and the `_ae_` triplet for the A and E network;
each was run at the four seeds. Ranges over the seeds:

| Streaming configuration, `q8_b6_s001` | Circuit (64) | Logistic regression (9) | Network of six (61) | Harmonic (17) |
|---|---|---|---|---|
| Validation ROC area | 0.627 to 0.694 | 0.748 to 0.752 | 0.714 to 0.746 | 0.659 to 0.721 |
| Blind ROC area | 0.779 to 0.786 | 0.825 to 0.827 | 0.808 to 0.823 | 0.796 to 0.823 |
| Blind window recall | 0.085 to 0.155 | 0.212 to 0.213 | 0.219 to 0.224 | 0.212 to 0.225 |
| Blind label spans | 5/5 | 5/5 | 5/5 | 5/5 |
| Blind false-alarm episodes per 30 d | 2.97 to 6.76 | 3.30 to 3.46 | 3.30 to 4.78 | 3.30 to 6.76 |
| Streamed year: coalescences alerted before the merger, by seed | 5, 6, 6, 6 | 6, 6, 6, 6 | 6, 6, 6, 6 | 5, 6, 6, 6 |
| Streamed year: false alarms per 30 d | 0.41 to 1.40 | 0.58 to 0.82 | 1.32 to 1.89 | 0.82 to 1.89 |
| Streamed year: alert of event 2 before its merger [h] | 11.3 to 11.5 | 26.2 | 26.2 | 26.2 |
| Streamed year: coalescences confirmed before the merger, by seed | 3, 5, 5, 5 | 5, 5, 5, 5 | 5, 5, 5, 5 | 5, 5, 5, 5 |

| A and E network, `q8_b6_s001_ae` | Circuit (64) | Logistic regression (9) | Network of six (61) | Harmonic (17) |
|---|---|---|---|---|
| Validation ROC area | 0.648 to 0.746 | 0.772 to 0.773 | 0.744 to 0.776 | 0.702 to 0.734 |
| Blind ROC area | 0.749 to 0.783 | 0.805 to 0.806 | 0.796 to 0.807 | 0.776 to 0.799 |
| Blind window recall | 0.127 to 0.165 | 0.222 to 0.223 | 0.223 to 0.228 | 0.221 to 0.232 |
| Blind label spans | 5/5 | 5/5 | 5/5 | 5/5 |
| Blind false-alarm episodes per 30 d | 3.30 to 8.58 | 3.96 to 4.04 | 3.71 to 4.29 | 5.20 to 6.27 |
| Streamed year: coalescences alerted before the merger, by seed | 6, 6, 6, 6 | 6, 6, 6, 6 | 6, 6, 6, 6 | 6, 6, 6, 6 |
| Streamed year: false alarms per 30 d | 0.90 to 4.44 | 0.58 to 0.66 | 0.58 to 1.56 | 1.07 to 1.65 |
| Streamed year: alert of event 2 before its merger [h] | 11.5 | 26.2 | 26.2 | 26.2 |
| Streamed year: alert of event 5 before its merger [h] | 4.5 to 4.6 | 7.5 to 7.7 | 4.5 to 7.5 | 4.5 |
| Streamed year: coalescences confirmed before the merger, by seed | 5, 6, 6, 6 | 6, 6, 6, 6 | 6, 6, 6, 6 | 6, 6, 6, 6 |

On these features the circuit adds nothing to a linear classifier. With
nine parameters the logistic regression has the larger ROC area at every
seed, on the validation block and on the blind year, and a window recall
of 0.21 to 0.22 against 0.09 to 0.17 at a comparable rate of false-alarm
episodes. In the streamed year it alerts the six coalescences before
their mergers at every seed and in both modes, at 0.58 to 0.82 and 0.58
to 0.66 false alarms per 30 days, where the circuit counts 0.41 to 1.40
and 0.90 to 4.44 and misses event 5 at one seed; it alerts event 2 26.2
hours before its merger, the circuit 11.3 to 11.5 hours, and the other
alert times agree within six hours. Its runs differ between the seeds in the third digit,
since its loss is convex. The network of six behaves as the circuit does:
the same alerts, and a false-alarm rate that scatters over the seeds. The
harmonic control pays for the circuit's feature map as the circuit does —
a validation ROC area that scatters with the seed, 3.3 to 6.8 blind
episodes per 30 days — and still has the larger blind ROC area; it alerts
event 2 at 26.2 hours as the other controls do.

**The other configurations.** The same three controls were trained on every
configuration of the grid (`configs/experiments/<config>_logistic.toml`,
`_perceptron.toml`, `_harmonic.toml`; four seeds each; `q8_b6_l6` shares
the features of `q8_b6`), the blind year scored once per run and the
alerts of the completed record credited from the signal onset at
persistence 3:

| Configuration | Model (parameters) | Blind ROC area | Blind spans (seeds 9999, 1009, 2027, 3041) | Blind episodes per 30 d | Alerts before the merger (of 6) | Alert false alarms per 30 d |
|---|---|---|---|---|---|---|
| `q4_l6` | circuit (48) | 0.800–0.814 | 4, 3, 4, 5 of 5 | 1.24–4.70 | 4, 3, 3, 6 | 0.49–2.97 |
|  | logistic (5) | 0.802–0.804 | 5 of 5 at every seed | 2.06–2.23 | 6 at every seed | 1.81–1.90 |
|  | network (37) | 0.822–0.823 | 5 of 5 at every seed | 1.57–2.06 | 6 at every seed | 1.48–1.65 |
|  | harmonic (9) | 0.815–0.817 | 5 of 5 at every seed | 1.98–4.21 | 6 at every seed | 1.57–2.56 |
| `q6_b4` | circuit (48) | 0.784–0.807 | 5 of 5 at every seed | 0.49–10.64 | 6, 6, 6, 4 | 0.08–7.17 |
|  | logistic (7) | 0.806–0.807 | 5 of 5 at every seed | 4.62–4.70 | 6 at every seed | 2.23 |
|  | network (49) | 0.829–0.830 | 5 of 5 at every seed | 3.46–3.63 | 6 at every seed | 1.32–1.48 |
|  | harmonic (13) | 0.819–0.821 | 5 of 5 at every seed | 5.28–6.02 | 6 at every seed | 2.23–2.56 |
| `q6_b4_noweight` | circuit (48) | 0.811–0.820 | 5 of 5 at every seed | 2.23–9.81 | 6 at every seed | 1.40–6.76 |
|  | logistic (7) | 0.798–0.799 | 5 of 5 at every seed | 4.78–4.95 | 6 at every seed | 2.06 |
|  | network (49) | 0.829–0.830 | 5 of 5 at every seed | 3.38–3.55 | 6 at every seed | 1.40 |
|  | harmonic (13) | 0.816–0.817 | 5 of 5 at every seed | 4.70–5.44 | 6 at every seed | 1.98–2.47 |
| `q6_b4_w2000` | circuit (48) | 0.804–0.807 | 5 of 5 at every seed | 24.66–26.39 | 6 at every seed | 26.06–28.05 |
|  | logistic (7) | 0.843–0.844 | 5 of 5 at every seed | 10.51–11.75 | 6 at every seed | 10.76–11.91 |
|  | network (49) | 0.851–0.854 | 5 of 5 at every seed | 3.48–4.05 | 6 at every seed | 2.81–3.06 |
|  | harmonic (13) | 0.848–0.850 | 5 of 5 at every seed | 9.27–12.74 | 6 at every seed | 8.94–12.66 |
| `q8_b6` | circuit (64) | 0.807–0.827 | 5 of 5 at every seed | 1.57–8.74 | 6 at every seed | 0.66–4.62 |
|  | logistic (9) | 0.809–0.813 | 5 of 5 at every seed | 3.30–3.71 | 6 at every seed | 1.81–2.06 |
|  | network (61) | 0.831–0.832 | 5 of 5 at every seed | 2.80–2.97 | 6 at every seed | 1.15–1.48 |
|  | harmonic (17) | 0.824–0.831 | 5 of 5 at every seed | 2.47–3.71 | 6 at every seed | 1.24–1.98 |
| `q8_b6_ae` | circuit (64) | 0.785–0.813 | 5 of 5 at every seed | 2.23–2.72 | 6 at every seed | 0.58–1.15 |
|  | logistic (9) | 0.805–0.806 | 5 of 5 at every seed | 3.05–3.22 | 6 at every seed | 1.40–1.57 |
|  | network (61) | 0.820–0.826 | 5 of 5 at every seed | 2.14–2.23 | 6 at every seed | 0.58–0.74 |
|  | harmonic (17) | 0.818–0.819 | 5 of 5 at every seed | 2.56–3.30 | 6 at every seed | 0.82–0.99 |

On the configurations that whiten without the smoothing, the comparison is
closer than on the streaming configuration. The logistic regression lies
within the scatter of the circuit's seeds in blind ROC area (`q8_b6`: 0.809
to 0.813 against 0.807 to 0.827) and below every run of the circuit in one
configuration (`q6_b4_noweight`: 0.798 to 0.799 against 0.811 to 0.820);
the network of six has the larger area in every configuration (0.820 to
0.854 against 0.784 to 0.827), the harmonic control in all but
`q6_b4_noweight`. Every control recovers the five label spans and alerts
the six coalescences at every seed of every configuration, which the
circuit does not in `q4_l6` (three to five spans, three to six alerts) and
at one seed of `q6_b4` (four alerts). The false-alarm rate of the circuit
scatters around those of the controls, single runs lying below them and
others far above: for `q8_b6`, 1.57 to 8.74 episodes per 30 days against
3.30 to 3.71 for the logistic regression, 2.80 to 2.97 for the network and
2.47 to 3.71 for the harmonic control, the selected run being the one at
1.57. No configuration separates the circuit from its controls beyond that
scatter, and the clear advantage of the logistic regression is that of the
smoothed features.

### Instrumental glitches

The LDC-2b (Spritz) records carry what Sangria
lacks: glitches fitted to LISA Pathfinder, gaps of several hours marked as
`NaN`, second-generation TDI on Keplerian orbits. Three records are
released: a year of 36 verification binaries with Poisson glitches
(`vgb`; no coalescence), and two months of 31 days each holding one
coalescence, of high SNR (`mbhb1`, three short glitches) and of low SNR (`mbhb2`, a
long glitch at the merger). The configurations
`configs/experiments/q8_b6_s001_spritz.toml` and `q8_b6_s001_ae_spritz.toml`
run them through the stages of this package: the pre-processing conditions
every stretch between two gaps on its own and pools the Welch estimate over
the stretches; the labelling reads the truth stream `sky/tdi` and the
catalogue `sky/cat` and credits alerts from the onset of the signal
against the record's own noise (`[ldc] psd_model = "record"`: the truth
stream conditioned as the record is, high-passed and whitened by the
record's PSD, the first window in which the whitened signal reaches a
norm of 3); the inference scores the products with the runs trained on
Sangria, circuit and controls, at the thresholds those runs fitted,
without retraining. An alert is a run of three alarmed windows consecutive
in the record; on `vgb` every alert is false. Seeds 9999, 1009, 2027 and
3041:

| Mode | Model | `vgb`: false alerts per 30 d | `mbhb1`: alert before the merger [h] | `mbhb1`: false alerts | `mbhb2`: alert before the merger [h] | `mbhb2`: false alerts |
|---|---|---|---|---|---|---|
| A | circuit | 1.78, 1.31, 0.00, 0.00 | 26.3, 15.6, –, 26.3 | 0, 1, 0, 0 | – at every seed | 1, 0, 0, 0 |
| A | logistic | 2.90, 2.90, 2.81, 2.72 | 82.4, 82.4, 67.8, 67.8 | 1 at every seed | 1.7 at every seed | 1 at every seed |
| A | network | 3.09, 3.09, 3.09, 3.37 | 82.4 at every seed | 1 at every seed | 1.7 at every seed | 1 at every seed |
| A | harmonic | 2.72, 2.72, 2.62, 2.43 | 67.8, 67.8, 67.8, 63.3 | 1 at every seed | 1.6, 1.6, 1.6, 0.3 | 1 at every seed |
| AE | circuit | 1.69, 1.12, 0.00, 1.31 | 32.6, 38.5, –, 26.3 | 0 at every seed | 0.9, 0.9, –, 0.4 | 1, 0, 0, 0 |
| AE | logistic | 3.56, 3.65, 3.65, 3.65 | 82.4 at every seed | 1 at every seed | 1.7 at every seed | 1 at every seed |
| AE | network | 3.65, 3.65, 3.84, 3.75 | 67.8, 67.8, 82.4, 82.4 | 1 at every seed | 1.7 at every seed | 1 at every seed |
| AE | harmonic | 3.28, 3.56, 3.56, 3.47 | 67.8, 67.8, 67.8, 63.3 | 1 at every seed | 1.1, 1.6, 1.6, 1.1 | 1 at every seed |

The high-SNR coalescence exceeds the onset threshold throughout its four-day
label span, so every alert inside the span is credited to it; the signal
onset of the low-SNR one lies 4.9 hours before its merger on A and 7.2 on the
A and E network. Every false alert of the year lies on a glitch, and without the
glitches there is none; the glitch power is red and resembles an early
inspiral to band-power features. The monotone controls give the windows
of largest band power the highest score: they alert the high-SNR coalescence 63 to 82 hours
ahead and the low-SNR one at every seed, and they alert on 2.4 to 3.8
glitches per 30 days. The circuit's response is bounded and periodic in every feature,
so a window far outside its training range does not receive the top
score: it alerts on 0 to 1.8 glitches per 30 days — the order of its
false-alarm rate on Sangria — while alerting the high-SNR coalescence 16 to 39
hours ahead at three seeds and the low-SNR one at the network's three; one
seed in four is silent on every Spritz record. The comparison that places
the models at the same rate of glitch alerts, the threshold of every run
moved to where `vgb` raises at most a given number per 30 days (the
controls by their logit, since their probabilities saturate on
windows of large band power), reads:

| Glitch false alerts per 30 d | Mode | Circuit | Harmonic (17) | Logistic regression (9) | Network of six (61) |
|---|---|---|---|---|---|
| ≤ 2 | A | 26.3, 19.8, 9.2, 26.5 | 30.2, 39.0, 36.6, 13.1 | 11.5, 11.3, 12.0, 12.0 | 12.3, 11.7, 12.3, 11.3 |
| ≤ 1 | A | –, none, –, 26.5 | 12.3, 36.3, 36.5, 7.8 | none | 7.4, –, none, – |
| ≤ 0.5 | A | –, none, –, 26.5 | 7.8 at every seed | none | –, –, none, – |
| ≤ 2 | AE | 38.4, 38.7, –, 26.3 | 25.3, 38.7, 36.5, 25.1 | 7.4 at every seed | 12.3, 11.5, 15.5, 11.0 |
| ≤ 1 | AE | –, 36.5, –, 26.3 | 7.8 at every seed | 7.4 at every seed | 7.4, –, none, – |
| ≤ 0.5 | AE | –, 32.3, –, 26.3 | 7.8, 6.9, 7.8, 7.8 | – | 7.4, –, none, – |

(alert of `mbhb1` before its merger in hours; "none": no threshold brings
the run to that rate; "–": not alerted at that threshold). At a matched
rate of at most 2 glitch alerts per 30 days the circuit alerts the high-SNR
coalescence 9 to 39 hours ahead at the seeds that respond, where the
logistic regression gives 7 to 12 hours and the network of six at most
15.5 hours; at lower rates the logistic regression on A cannot be brought
to the rate at all. The harmonic
control, with the circuit's feature map and no circuit, does as well or
better at two per 30 days and keeps an alert of 7 to 8 hours at every seed
down to 0.5 per 30 days, where the circuit alerts at one or two seeds. The
tolerance of transients of large amplitude beyond the scaler's range is therefore a property of the
angle encoding, measured here on one simulated year of glitches and two
coalescences, with scalers and thresholds fitted on Sangria noise whose
upper-band features lie above those of Spritz.

The detection results of this page are those of the conditioning and of
the eight features — the record high-pass, the whitening by a measured
PSD, the whitened band powers between 0.3 and 40 mHz — for which the variational circuit is one
classifier among others, neither the smallest nor the most stable under
re-initialisation. What speaks for it is its feature map: bounded and
periodic in every feature, it does not hand the top score to the transient
of largest amplitude, and at a matched false-alarm rate on clean data it is level
with its controls. What cannot be said is that the entangling circuit adds
detection performance over a classical model with the same encoding, or an
economy of parameters: nine classical ones suffice on these inputs, and
seventeen reproduce the glitch tolerance. The best streamed result on this
page is that of the logistic regression on the features of the A and E
network: six of six coalescences at every seed, alerted 7.5 to 28.7 hours
before their mergers at 0.58 to 0.66 false alarms per 30 days, and
confirmed before every merger.

## Caveats

- **One blind realisation of five events.** The event recall is 5 of 5 and
  the false-alarm rate is measured over 364 days, but five events do not
  measure a detection efficiency. Treat the recall as a result, not a rate.
- **Four initialisations per configuration.** They establish the recall
  of the band partitions and the rate of the 2000-sample windows; they do
  not rank the other configurations by false-alarm rate, and the
  selection statistic of the validation block, 1 to 10 episodes, cannot.
- **Two channels at most.** A alone, or A and E; the T channel carries no
  feature of this package: on arms of unequal length it responds to the
  signal below 1 mHz, and above it its band powers did not separate the
  glitches of LDC-2b from noise.
- **No gaps in the principal result.** The Sangria products are gapless, and the
  mission replayed above delivered all 63,043 batches: nothing was lost or
  pruned, so every number in the table before it describes a lossless
  link. The effect of a lossy link is measured separately, above, on 30-day
  missions; the classifier is trained on gapless records, and a record
  with gaps is scored stretch by stretch, as the LDC-2b records are in the
  section on the controls.
- **The threshold is fitted on the same mission's earlier year.** A real
  chain would recalibrate as the mission proceeds; the transfer measured
  here is over one year, in one direction.
- **The encoding is mitigated, not solved.** On ``[0, \pi]`` four of the
  six coalescences clear the threshold in their merger bin; the two most
  saturated do not, and every alert still rests on the inspiral excess of
  the hours before the merger.
- **The replay whitened by the full-record PSD is non-causal.** The
  full-record PSD whitens every window of the comparison replay and, at
  the first event, includes the nine months not yet delivered; that replay
  is an upper reference for the causal replay. The latencies quoted are
  those of the causal replay, which a mission could run.
- **Signal onsets from the combined truth stream.** The onset of a
  coalescence is where the sum of all MBHB signals first reaches a norm
  of 3 against the record's own noise, sought past the span of the
  preceding coalescence. An alert
  after it may still have been raised by a neighbouring source, and a
  matched filter sees a signal before a classifier of band powers can: the
  onset bounds the earliest creditable alert rather than locating it.
- **Four seeds of the streaming configuration.** They agree on the alerts
  of five coalescences and differ on event 5 and on the false-alarm rate,
  0.41 to 1.40 per 30 days; seed by seed, the calibration rule would set
  the persistence between two and four, and the configuration keeps the
  pooled value of three.
- **The glitch results rest on one simulated year and two coalescences.**
  The Sangria scalers and thresholds were applied to Spritz noise, whose
  upper-band features lie below the Sangria ones; the comparison at a
  matched rate of glitch alerts moves the thresholds after the fact and is
  not a selection.

## Corrections to earlier releases

Values of the earlier release or mission are given in parentheses.

**Correction to release 1.2.0.** Release 1.2.0 credited the whole label
span and thereby counted alarm runs 60 to 95 hours before a merger as
early detections: the alerts of the selected model 25.6 and 16.1 hours
before the mergers of events 1 and 4, and the lead of 71 hours on event 1
of the smoothed model (section [Smoothed whitening](#Smoothed-whitening)).
Those runs lie where the window SNR of the source against the instrument
PSD is about 2, and those of the smoothed model recur at the same mission
times at every initialisation seed, because every seed sees the same
noise. The text of 1.2.0 judged the early alerts of the selected model
consistent with chance but those of the smoothed model significant against
its false-alarm rate. The second judgement was wrong on two counts: that
rate is not uniform over the year, since the smoothed model's score
baseline approaches the threshold between days 200 and 300, where events 4
and 5 fall; and a comparison of rates cannot show that a particular alarm
was raised by the signal, which the window SNR can. The tables and figures
on this page credit from the signal onset.

The two alerts of the selected model on events 1 and 4 rest on windows
that end 67 and 63 hours before the merger, where the window SNR of the
source against the instrument PSD is about 2. Credited from the onset, the
same causal replay alerts event 2 on its own, 16 hours after its merger,
where 1.2.0 found only the alarms of event 1 in its span. Under the
full-record PSD, label-span crediting had alerted events 4 and 5 64 and 51
hours before their mergers, on runs of two windows where the window SNR of
the source against the instrument PSD is below 2. On the lossy link, the
one alert that release 1.2.0 counted with 3 % loss and one retransmission
preceded the signal onset. The alarms 53 and 76 h before merger 1, which
release 1.2.0 counted as detections in the missions that admit the
stretch at 75 or 50 % delivered, precede the signal onset and are false
alarms.

**Correction, 30 September 2026.** The two generation-gap missions first
published with releases 1.2.0 and 2.0.0 were recorded with
DeepSpaceTelemetry.jl 2.0.0, which in external-payload mode resumed the
payload where it had stopped after a scheduled gap while stamping the
following batches with the epoch of the gap's end: every later sample was
replayed 1.99 h after its content time in the mission with the two-hour
gap, and 15 min after it in the mission with the fifteen-minute gap, and
merger 1 reached the replay outside its credited span. The two-hour gap
had, moreover, been placed 4.9 h before merger 1 instead of astride it.
Both missions were regenerated with DeepSpaceTelemetry.jl 2.1.1, which
addresses the payload by content time, and the rows of the table of
[Effect of a lossy link](#Effect-of-a-lossy-link) replace the published
ones (15 min: 6.9 false alarms per 30 days; 2 h: one coalescence of two,
the other said to have fallen in the gap). The replay now refuses runs of
the affected producer versions. The other fifteen missions contain no
scheduled gap and are unaffected.

**Correction, 12 October 2026.** Releases 2.0.0 to 2.0.1 defined the
signal onset as the first window in which the raw truth stream reaches a
matched-filter SNR of 5 against the analytic instrument PSD of the LDC
noise model, without the Galactic foreground. The noise the detector
faces is the record's own: every feature is whitened by the measured PSD
of the record, which exceeds the instrument model by a factor of 18 to 35
at 1 to 5 mHz on the Sangria blind year. Against that noise the signal
reaches a norm of 3 to 5 at the onsets of those releases, and the first
alerts of the high-SNR coalescences fall where it reaches about 4. The onset
is now the first window in which the truth stream, high-passed and
whitened as the record is, reaches a norm of 3 (`[ldc] psd_model =
"record"`, `label_snr_threshold = 3`); on this year the onsets move from
7 to 35 hours to 10 to 38 hours before the merger (the onset table of
[Telemetry replay and alert latency](#Telemetry-replay-and-alert-latency)).
Every alert table of this page was re-credited under the new rule from its
stored window table. The alerts credited to the coalescences are those of
releases 2.0.0 to 2.0.1 to the tenth of an hour, except event 3, whose
earlier alert now lies inside the span: the A runs of the streaming
configuration alert it 27.8 to 28.7 hours ahead (previously 24.3 to 24.8
hours before the merger), both replays of the selected configuration 0.3
hours ahead (previously 0.1 and 0.6 hours after the merger). The
false-alarm rates are lower by up to 0.3 per 30 days where the longer
spans absorb alarms the earlier onset left outside: the selected run of
the streaming configuration moves from 0.49 to 0.33 per 30 days (these are
the values of the wall-clock mission; see the rerun on the tick clock
below). The calibration counts of the persistence rule of the telemetry
section are those of the training year's onsets under the new rule
(previously 1.38 and 0.83 false alerts per 30 days at two and three
consecutive windows for the selected model, 1.38 and 0.96 for the pooled
seeds of the streaming configuration); the persistence it selects is
unchanged. The 30-day missions of the lossy-link study are unchanged.
Amended 15 October 2026: the former onsets lie at a norm of 3 to 5, not 1,
and event 3 of the selected configuration is also re-credited.

**Rerun on the tick clock, 15 October 2026.** The year mission of releases
1.2.0 to 2.0.1 was run by DeepSpaceTelemetry.jl 1.2.0-DEV paced against
the wall clock at a speed-up of 7200, that is 7200 mission seconds per
wall-clock second, so that the instants at which its batches arrived
depended on the load of the host and cannot be reproduced. It was run
again with DeepSpaceTelemetry.jl 2.6.2 on the producer's logical clock,
from the same scenario and seed. On the logical clock the producer
advances mission time by its own events, independent of the host, so a
run is reproducible from its scenario and seed. Every streamed-year table
and figure of this page is now that of the new mission. It delivers the
same 63,043 batches, at instants that differ from those of the first
mission by a median of 4 minutes, with 99 % within 1.5 hours. Under the
full-record PSD, and for the selected configuration under causal
whitening, every window scores as previously and only the alert times
move with the arrivals, by up to 1.8 hours: the selected configuration
alerts event 3 1.6 hours before its merger (previously 0.3 hours before
the merger) and the five others 1.9 to 40 hours after it (previously 1.9
to 42). The trailing estimate of the streaming configuration sees the
record in a different order of arrival, so 2 to 6 of its 63,014 decisions
per seed change, and with them one false-alarm episode at two seeds: 0.41,
0.99, 1.40 and 1.23 per 30 days at the four seeds (previously 0.33, 0.99,
1.32 and 1.23), the same coalescences alerted before their merger, each
within 0.9 hours of its former alert time. The A and E network was rerun
as a two-column mission (section [The A and E
network](#The-A-and-E-network)): 0.90, 4.44, 1.40 and 1.48 per 30 days
(previously 0.99, 4.69, 1.32 and 1.40), alert times within 1.3 hours. The
classical controls (section [Classical controls on the same
features](#Classical-controls-on-the-same-features)) were replayed on the
same missions: alert times within 1.7 hours, false-alarm rates within
three episodes of the year per seed; the logistic regression on A and E
counts two or three more at every seed, 0.58 to 0.66 per 30 days
(previously 0.41 to 0.49).

## Reproducing

With `$LDC` pointing at the directory holding the Sangria HDF5 products
and `$UNBLINDED` at the CSV of the blind year's signal-only TDI (columns
`t`, `X`, `Y`, `Z`, released with the unblinded MBHB catalogue):

```bash
# Labels, event tables and signal onsets for both years
julia scripts/label_ldc.jl configs/sangria.toml --h5-file $LDC/LDC2_sangria_training_v2.h5
julia scripts/label_ldc.jl configs/sangria.toml --truth-csv $UNBLINDED \
    --record-file $LDC/LDC2_sangria_blind_v2.h5 --output-prefix sangria_blind_points

# Band features for both years
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6.toml \
    --h5-file $LDC/LDC2_sangria_training_v2.h5 --label-file data/inputs/sangria_labels.csv
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6.toml \
    --h5-file $LDC/LDC2_sangria_blind_v2.h5 \
    --label-file data/inputs/sangria_blind_points_labels.csv \
    --output-prefix sangria_b6_blind

# Train and evaluate the selected run; the further seeds add --seed 1009 | 2027 | 3041
julia scripts/train.jl configs/experiments/q8_b6.toml --run-id q8_b6_pi
julia scripts/infer.jl configs/experiments/q8_b6.toml --run-id q8_b6_pi

# Export the blind year as a telemetry payload, run a DeepSpaceTelemetry mission on
# the scenario fragment it writes (the year of this page: the link of the telemetry
# section, 365 days on the producer's tick clock, simulation.mode = "tick"), and
# replay the mission: causal as configured, full-record PSD with psd_mode = "sidecar"
# in [telemetry]
julia scripts/export_telemetry_payload.jl configs/experiments/q8_b6.toml \
    --h5-file $LDC/LDC2_sangria_blind_v2.h5 \
    --catalog data/inputs/sangria_blind_points_events.csv --output-prefix sangria_blind
julia scripts/infer_telemetry.jl configs/experiments/q8_b6.toml --run-dir <DeepSpaceTelemetry run> \
    --model models/run_q8_b6_pi/gw_model.jld2 \
    --events data/inputs/sangria_blind_points_events.csv --run-id telemetry_year_pi

# The smoothed-whitening configuration: the same steps with its own feature prefix
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6_s001.toml \
    --h5-file $LDC/LDC2_sangria_training_v2.h5 --label-file data/inputs/sangria_labels.csv
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6_s001.toml \
    --h5-file $LDC/LDC2_sangria_blind_v2.h5 \
    --label-file data/inputs/sangria_blind_points_labels.csv \
    --output-prefix sangria_b6s001_blind
julia scripts/train.jl configs/experiments/q8_b6_s001.toml --run-id q8_b6_s001_pi
julia scripts/infer.jl configs/experiments/q8_b6_s001.toml --run-id q8_b6_s001_pi
julia scripts/infer_telemetry.jl configs/experiments/q8_b6_s001.toml --run-dir <DeepSpaceTelemetry run> \
    --model models/run_q8_b6_s001_pi/gw_model.jld2 \
    --events data/inputs/sangria_blind_points_events.csv --run-id telemetry_year_s001_pi
# The A and E network: products of the mode (suffix _ae), onsets of the network; the
# export of the mode writes the two-column payload A, E, whose mission the replay reads
# both channels from (a one-column mission takes --tdi-file and the TDI record instead)
julia scripts/label_ldc.jl configs/experiments/q8_b6_s001_ae.toml --h5-file $LDC/LDC2_sangria_training_v2.h5
julia scripts/label_ldc.jl configs/experiments/q8_b6_s001_ae.toml --truth-csv $UNBLINDED \
    --record-file $LDC/LDC2_sangria_blind_v2.h5 --output-prefix sangria_ae_blind_points
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6_s001_ae.toml \
    --h5-file $LDC/LDC2_sangria_training_v2.h5 --label-file data/inputs/sangria_labels.csv
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6_s001_ae.toml \
    --h5-file $LDC/LDC2_sangria_blind_v2.h5 \
    --label-file data/inputs/sangria_blind_points_labels.csv \
    --output-prefix sangria_b6s001_blind
julia scripts/train.jl configs/experiments/q8_b6_s001_ae.toml --run-id q8_b6_s001_ae_pi
julia scripts/infer.jl configs/experiments/q8_b6_s001_ae.toml --run-id q8_b6_s001_ae_pi
julia scripts/export_telemetry_payload.jl configs/experiments/q8_b6_s001_ae.toml \
    --h5-file $LDC/LDC2_sangria_blind_v2.h5 \
    --catalog data/inputs/sangria_blind_points_events.csv --output-prefix sangria_blind
julia scripts/infer_telemetry.jl configs/experiments/q8_b6_s001_ae.toml --run-dir <two-column run> \
    --model models/run_q8_b6_s001_ae_pi/gw_model.jld2 \
    --events data/inputs/sangria_ae_blind_points_events.csv --run-id telemetry_year_s001_ae_pi

# The classical controls: the same products, [model] kind in the configuration
# (q8_b6_s001_logistic | _perceptron | _harmonic and the _ae_ triplet; every
# configuration of the grid has the same three companions)
julia scripts/train.jl configs/experiments/q8_b6_s001_logistic.toml --run-id q8_b6_s001_logistic_pi
julia scripts/infer.jl configs/experiments/q8_b6_s001_logistic.toml --run-id q8_b6_s001_logistic_pi
julia scripts/infer_telemetry.jl configs/experiments/q8_b6_s001_logistic.toml --run-dir <DeepSpaceTelemetry run> \
    --model models/run_q8_b6_s001_logistic_pi/gw_model.jld2 \
    --events data/inputs/sangria_blind_points_events.csv --run-id telemetry_year_s001_logistic_pi

# The LDC-2b (Spritz) records, $SPRITZ pointing at their directory: labels and onsets
# against the record's own noise, the product conditioned stretch by stretch, and a
# Sangria run applied at its own threshold (mbhb2 and vgb with their own
# --output-prefix and --features; the _ae_spritz configuration for the network)
julia scripts/label_ldc.jl configs/experiments/q8_b6_s001_spritz.toml \
    --h5-file $SPRITZ/LDC2_spritz_mbhb1_training_v2.h5
julia scripts/preprocess_ldc.jl configs/experiments/q8_b6_s001_spritz.toml \
    --h5-file $SPRITZ/LDC2_spritz_mbhb1_training_v2.h5 --label-file data/inputs/spritz_mbhb1_labels.csv
julia scripts/infer.jl configs/experiments/q8_b6_s001_spritz.toml --run-id spritz_mbhb1_s001_pi \
    --model models/run_q8_b6_s001_pi/gw_model.jld2 --labels ""
```

Training runs to early stopping in 16 to 50 epochs; an epoch of the
eight-qubit model takes about three seconds on 16 threads, and inference
over the blind year about a minute. The runs of this page were trained
under the Zygote tape, at two to four minutes per epoch; retrained with
the adjoint gradient, the default, the two configurations
reproduce them at every seed (see [Training](architecture.md#Training)). The other configurations are in `configs/experiments/` and
`configs/sangria.toml`; every run on this page was trained with the
committed `phase_span = 1.0`. The batch gradient is chunked at a fixed
size and reduced in chunk order, so the thread count does not enter the
result: runs of the same seed at 8 and 16 threads reproduced each other
bit for bit.
