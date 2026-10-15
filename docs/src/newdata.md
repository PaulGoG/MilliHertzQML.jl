# New data sets

This page describes how a classifier trained on one record is applied to
another: a later stretch of the same mission, a record of another data
challenge, or a background record without coalescences.

## Why an operating point does not transfer

A training run carries two fitted quantities beside the parameters of the
classifier. The feature scaler maps the per-feature quantiles of the
training partition onto the encoding interval ``[0, \phi]``; the decision
threshold is fitted on the calibration block to a false-alarm target
(`[training] threshold_criterion`, `target_far_per_30d`). Both are
statements about the training noise. Applied to a record whose noise
differs, the scaler places that noise elsewhere on the encoding interval,
and the threshold no longer corresponds to the false-alarm rate it was
fitted to.

The LDC-2b (Spritz) records show the effect. The scaler fitted on Sangria
placed the Spritz noise near its lower bound in three of the eight
features: the whitened band powers of Spritz lie below those of Sangria in
the upper bands, where the Galactic foreground of Spritz has no forest of
resolved binaries. One run in four fell silent on every record
([Instrumental glitches](benchmark.md#Instrumental-glitches)). A new
record therefore needs the scaler and the threshold fitted on its own
noise before its alerts can be read. The parameters of the classifier
stay those of the training run.

## The procedure

1. **Describe the record** in a configuration under `configs/experiments/`:
   the HDF5 object of the record (`[preprocessing] tdi_group`), or, for a
   product shipped as component streams, the list of components summed
   into the record (`[preprocessing] tdi_components`); the sampling rate
   (`[preprocessing] sample_rate`); the TDI generation (`[ldc] tdi2`); the
   output prefix of the products (`[preprocessing] output_prefix`); and
   the `[calibration]` section. The record itself is passed on the command
   line with `--h5-file`.
2. **Label the record** if it carries coalescences, from its truth stream,
   with the signal onsets against the record's own noise:

   ```bash
   julia scripts/label_ldc.jl configs/experiments/<config>.toml --h5-file <record>.h5
   ```

   A record without a catalogue of coalescences is a background record;
   every alarm on it is false, and this step is skipped.
3. **Pre-process the record.** The features are whitened by the Welch
   estimate of the record itself, so the conditioning is already the
   record's own:

   ```bash
   julia scripts/preprocess_ldc.jl configs/experiments/<config>.toml --h5-file <record>.h5 \
       [--label-file data/inputs/<record>_labels.csv]
   ```

4. **Score the record as it comes**, with the scaler and threshold of the
   training run. This is the transfer check, the baseline against which
   the calibration is judged:

   ```bash
   julia scripts/infer.jl configs/experiments/<config>.toml --run-id <ID> \
       --model models/run_<RUN_ID>/gw_model.jld2 \
       --features data/inputs/<record>_features.csv --labels ""
   ```

5. **Calibrate.** The first `[calibration] block_fraction` of the record's
   windows (0.3 by default) is the calibration block; on it the stage fits
   the scaler by the rule `[calibration] scaler` (`match`, the default;
   `refit`; `keep`) and then the threshold, under the criterion and target
   of `[training]`:

   ```bash
   julia scripts/calibrate.jl configs/experiments/<config>.toml --run-id <CAL> \
       --model models/run_<RUN_ID>/gw_model.jld2 \
       --features data/inputs/<record>_features.csv \
       [--labels data/inputs/<record>_labels.csv]
   ```

6. **Score the evaluation block**, the rest of the record after a buffer of
   one window length, at the new operating point (with
   `--labels data/inputs/<record>_labels.csv` for a labelled record):

   ```bash
   julia scripts/infer.jl configs/experiments/<config>.toml --run-id <CAL> \
       --features data/inputs/<record>_features.csv --labels "" --block evaluation
   ```

7. **Report**, on the evaluation block alone: the alerts, runs of
   `[telemetry] alert_persistence` alarmed windows consecutive in the
   record, and the confirmations, runs of
   `alert_confirmation_persistence` (eleven) alarmed windows, the first and
   last of which share no sample, both per 30 days; the alert time of every
   labelled coalescence relative to its merger; the realised false-alarm
   rate against the one fitted on the calibration block
   (`fit_false_alarms_per_30d` in `calibration.toml`); and the number of
   false-alarm episodes that fit rests on (`fit_false_alarm_episodes`),
   whose inverse square root is its relative error. The stage warns when
   that number falls below `[training] min_fit_episodes`.

The threshold is fitted over the candidate grid of the training stage,
`n_candidates` quantiles of the block's scores, extended by
`[calibration] tail_candidates` (200 by default) candidates log-spaced
into the upper tail down to one window. On a block without signal the
quantile step, a quarter of a percent of the windows, spans the whole
range of false-alarm rates an alert trigger is fitted to: the next
candidate below the maximum score already alarms more windows than the
target admits, and the fit would rest on a single episode at the maximum.
The tail candidates make the rates between one episode and that step
reachable.

The calibration block and the evaluation block are disjoint: the
operating point is fitted on the first and judged on the second, as the
training run fits its threshold on the validation block and judges it on
the test block.

## The scaler rules

**`match`** (the default, `match_scaler`) encodes the noise of the new
record where the training noise was encoded. For every feature ``j`` two
quantiles ``p_1 < p_2`` of the noise windows are taken
(`[calibration] noise_quantiles`, by default the median and the 0.99
quantile): ``q_\mathrm{ref}(p)`` of the noise windows of the parent's
training block, and ``q_\mathrm{new}(p)`` of the noise windows of the
calibration block. Under the parent's scaler, with bounds
``[\ell_j, u_j]`` and encoding span ``\phi``, the training quantiles
receive the angles

```math
a(p) = \phi\,\frac{q_\mathrm{ref}(p) - \ell_j}{u_j - \ell_j},
```

and the new bounds are those of the affine map that sends
``q_\mathrm{new}(p_1)`` and ``q_\mathrm{new}(p_2)`` to the same angles:

```math
u_j' - \ell_j' = \phi\,\frac{q_\mathrm{new}(p_2) - q_\mathrm{new}(p_1)}{a(p_2) - a(p_1)},
\qquad
\ell_j' = q_\mathrm{new}(p_1) - a(p_1)\,\frac{u_j' - \ell_j'}{\phi}.
```

If the new noise is an affine image of the training noise, feature by
feature, the map recovers it exactly, and the noise windows of the new
record are encoded as those of the training block were. A signal window
of the new record then lies above its noise as a training signal lay
above the training noise. Only the noise windows enter the fit: the
labelled positives of a labelled record are excluded from both
quantiles. The reference product is `[calibration] reference_features`
and `reference_labels`, by default the training product of `[training]`,
and its training block is read from the `split.toml` of the parent run.

**`refit`** applies the training rule itself, the quantiles
`[training] scaler_quantiles` of the calibration block. That rule was
fitted on a training block holding a mixture of noise and signal windows,
so it reproduces the training encoding only when the calibration block
holds a comparable mixture, as on a labelled record with coalescences; on
a background record its upper bound is a quantile of the noise alone.

**`keep`** keeps the parent's scaler and refits the threshold only. It
measures what the threshold alone repairs.

## The calibration run directory

`<models>/run_<CAL>` holds:

- `gw_model.jld2`: the parent's parameters with the new scaler; its
  metadata gains a `calibration` entry (run identifier, parent model,
  feature table, scaler rule);
- `threshold.toml`: the threshold fitted on the calibration block, with
  the criterion, the targets and the rates of the block at the threshold
  (`fit_*`), `block = "calibration"`, the parent's threshold
  (`parent_threshold`) and model (`parent_model`);
- `threshold_sweep.csv`: the event-level operating characteristic of the
  calibration block;
- `split.toml`: the row ranges `calibration` and `evaluation`, the
  `buffer_windows` between them, `n_windows` and the feature table, read
  by `scripts/infer.jl --block`;
- `calibration.toml`: the fitted quantities, the bounds of the parent's
  and of the new scaler, the counts of noise windows in the reference and
  calibration blocks, both thresholds, and the fitted false-alarm rate with
  its episode count;
- `config.toml`: the snapshot of the `[model]` of the parent, the window
  geometry and channels of the record, and the `[calibration]` and
  threshold rules applied, with provenance; and `manifest_snapshot.toml`.

The stage refuses the product of a record with gaps, since its blocks
take consecutive rows as consecutive windows, and a feature table of
another channel set than the model was trained on.

## Worked cases

### Spritz

The LDC-2b records were scored as they came, with the scalers and
thresholds of the Sangria runs, and the comparison at a matched rate of
glitch alerts moved the threshold of every run post hoc to where the
year of verification binaries raises a given number of alerts per 30
days ([Instrumental glitches](benchmark.md#Instrumental-glitches)). The
calibration stage is the configuration-driven form of that step, with the
scaler refitted as well and the operating point judged on a block it was
not fitted on.

### Yorsh

The LDC-1b (Yorsh) record holds two years sampled at 5 s, in
second-generation TDI, without gaps or glitches. It is shipped as
thirteen component streams, which the pre-processing sums into the
record (`[preprocessing] tdi_components`): instrument noise, confusion
noise, eight stellar-origin binary black holes and three
extreme-mass-ratio inspirals. It holds no massive black hole binary, so
it is a background record for this detector and every alert on it is
false: its sources are continuous, and the four that end inside the
record (two of the stellar-origin binaries leave the band at days 579
and 683, two of the inspirals plunge at days 183 and 439) stay orders of
magnitude below the noise of a single window. The configurations
`configs/experiments/q8_b6_s001_yorsh.toml` and
`q8_b6_s001_ae_yorsh.toml` run it under the streaming configuration, on
the A channel and on the A and E network; the pre-processing of the
composite record takes about two and a half minutes per mode and gives
126,021 windows (729 days).

**The transfer check.** At the Sangria operating point the circuit is
silent: no alarmed window in 729 days on A at any seed, none on the
network at three seeds and 22 isolated ones at the fourth. The controls
raise between 0 and 0.2 alerts per 30 days (the network of six the most).
The cause is the one measured on Spritz. The Yorsh noise whitens to band
powers of about 1 with a spread of 0.09 dex, while the Sangria training
noise carries excess power in the bands above 1 mHz (medians 1.2 to 2.3)
that the smoothed whitening leaves in the features; under the Sangria
scaler the median of the Yorsh noise lies between a tenth and a sixth of
the encoding interval in those bands, where the Sangria noise lay between
a quarter and a third, and its 0.99 quantile stays below the middle where
Sangria's reached the top. Nothing enters the region the circuit learned
to alarm on.

**Calibration.** Each of the 32 runs was calibrated on the first 30 % of
the record (219 days) under the three scaler rules at the training target
of 3 false-alarm episodes per 30 days, and scored on the evaluation block
(510 days). Every fit reached the target (21 episodes, 2.88 per 30 days).
With the noise matched by its quantiles, the fitted threshold of the
circuit returns to the Sangria one to four decimals (0.6305 against
0.6305 on A at the first seed; the controls within 0.01 to 0.05): the
Yorsh noise, encoded where the Sangria noise was, produces the same score
tail at the same threshold. The kept scaler needs the threshold moved far
down (0.39 for the same run), and the refitted scaler far up (0.74); the
refit disables the logistic regression altogether, whose probabilities
saturate once the noise is stretched over the whole interval.

On the evaluation block, alerts (runs of three alarmed windows) and
confirmations (eleven) per 30 days, by seed, at the matched operating
point:

| Mode | Model | As it comes: alerts | Matched: episodes | Matched: alerts | Matched: confirmations |
|---|---|---|---|---|---|
| A | circuit | 0, 0, 0, 0 | 3.5, 4.1, 4.8, 4.5 | 0.35, 0.35, 0.65, 0.65 | 0 at every seed |
| A | logistic | 0.12, 0.12, 0, 0 | 4.9, 5.0, 4.8, 4.5 | 1.00, 1.06, 1.00, 1.00 | 0 |
| A | network | 0.24, 0.18, 0.18, 0.24 | 5.2, 4.9, 5.0, 4.4 | 1.47, 1.23, 1.18, 1.00 | 0 |
| A | harmonic | 0, 0.06, 0, 0 | 5.2, 4.6, 5.1, 5.9 | 1.00, 1.06, 1.23, 1.29 | 0 |
| AE | circuit | 0, 0, 0, 0 | 5.5, 4.7, 4.1, 6.5 | 0.82, 0.71, 0.41, 0.47 | 0 |
| AE | logistic | 0.06 at every seed | 4.8, 5.0, 4.9, 5.2 | 1.00, 1.06, 1.06, 1.06 | 0 |
| AE | network | 0.06, 0.06, 0.12, 0.18 | 4.4, 5.5, 5.2, 4.7 | 0.82, 0.82, 1.06, 1.00 | 0 |
| AE | harmonic | 0, 0, 0, 0 | 4.6, 5.4, 4.8, 4.8 | 0.94, 1.06, 0.88, 0.94 | 0 |

The realised episode rate on the evaluation block, 3.5 to 6.5 per 30
days, exceeds the fitted 2.88: the tail of the Yorsh noise is not
stationary. Counted in 30-day bins at a fixed threshold, the episodes
come in stretches about half a year apart (days 60 to 120, 240 to 300,
420 to 510 and 600 to 660), the second year's stronger than the first's,
so that the first 219 days hold 20 episodes where the remaining 510 hold
80 to 90. A calibration block shorter than the modulation, or a
threshold held fixed across it, therefore misstates the rate by a factor
of about two; the streamed replay, which re-estimates its whitening from
the trailing month, is the form that follows such a modulation. The
persistence tiers absorb the excess: at the matched operating point the
circuit raises 0.35 to 0.82 alerts per 30 days on the two channel sets,
the controls 0.8 to 1.5, and no run of the 96 confirms a single alert in
510 days. The refitted scaler gives the circuit 0 to 0.12 alerts per 30
days at 1.5 to 2.8 episodes, the kept scaler 0.3 to 1.2 alerts at 4 to 6
episodes.

Two numbers of this record carry over to the procedure. A background
record cannot be calibrated on the quantile grid alone: the candidates
added to the tail are what let the fit reach a target below a quarter of
a percent of the windows. And the matched scaler, not the threshold, is
what transfers the operating point: it recovers the Sangria threshold on
a record whose raw features sit elsewhere, and the threshold alone, kept
or moved, does not.
