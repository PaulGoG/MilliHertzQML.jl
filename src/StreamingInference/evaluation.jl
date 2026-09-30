# Chronological partitioning, ROC analysis, decision
# thresholds fitted on a held-out calibration block, and event-level
# detection metrics with the operational false-alarm rate.

"""
    chronological_split(n; train_fraction = 0.7, validation_fraction = 0.15, buffer = 0)

Partition `n` chronologically ordered windows into contiguous training,
validation, and test blocks holding `train_fraction`, `validation_fraction`,
and the remaining fraction of the windows, separated by `buffer` unused
windows so that overlapping windows never straddle two blocks. Returns a
named tuple of `UnitRange{Int}` (`train`, `validation`, `test`); every
block must be non-empty.
"""
function chronological_split(
    n::Integer;
    train_fraction::Real = 0.7,
    validation_fraction::Real = 0.15,
    buffer::Integer = 0,
)
    (0 < train_fraction < 1 && 0 < validation_fraction < 1) ||
        throw(ArgumentError("fractions must lie in (0, 1)."))
    train_fraction + validation_fraction < 1 || throw(
        ArgumentError(
            "train_fraction + validation_fraction = $(train_fraction + validation_fraction); " *
            "must leave a test block.",
        ),
    )
    buffer >= 0 || throw(ArgumentError("buffer = $buffer; must be non-negative."))
    n_train = floor(Int, train_fraction * n)
    n_val = floor(Int, validation_fraction * n)
    train = 1:n_train
    validation = (n_train+buffer+1):(n_train+buffer+n_val)
    test = (last(validation)+buffer+1):n
    (isempty(train) || isempty(validation) || isempty(test)) && throw(
        ArgumentError(
            "n = $n windows with buffer = $buffer leave an empty block " *
            "(train $train, validation $validation, test $test).",
        ),
    )
    return (train = train, validation = validation, test = test)
end

"""
    threshold_rows(blocks, threshold_block) -> UnitRange{Int}

Rows of the training table on which the decision threshold is fitted,
given the `blocks` of [`chronological_split`](@ref):

- `"validation"`: the validation block alone, the classical split.
- `"held_out"`: validation and test together, including the buffer
  windows between them so that the range stays contiguous in mission
  time and an alarm episode spanning the join is counted once.

The pooled block doubles the mission time behind the fitted false-alarm
rate, whose relative error scales as the inverse square root of the
episodes it counts; a 55-day block routinely counts fewer than five.
The test block carries no information the fit could leak, being scored
once after training and entering neither model selection nor early
stopping, but it ceases to be an independent check of the operating
point — that role belongs to a separate observation record.
"""
function threshold_rows(blocks::NamedTuple, threshold_block::AbstractString)
    threshold_block in ("validation", "held_out") || throw(
        ArgumentError(
            "threshold_block = $(repr(threshold_block)); expected validation or held_out.",
        ),
    )
    return threshold_block == "validation" ? blocks.validation :
           (first(blocks.validation):last(blocks.test))
end

"""
    roc_curve(y, scores) -> (fpr, tpr, thresholds)

Receiver operating characteristic of binary labels `y` (0/1) under the
decision `score >= threshold`, evaluated at every distinct score in
descending order and at `+Inf` (no alarms). `fpr` and `tpr` are
non-decreasing; a class without members yields `NaN` rates.
"""
function roc_curve(y::AbstractVector{<:Integer}, scores::AbstractVector{<:Real})
    length(y) == length(scores) ||
        throw(DimensionMismatch("$(length(y)) labels for $(length(scores)) scores."))
    order = sortperm(scores; rev = true)
    n_pos = count(==(1), y)
    n_neg = length(y) - n_pos
    thresholds = Float64[Inf]
    fpr = Float64[0.0]
    tpr = Float64[0.0]
    tp = 0
    fp = 0
    i = 1
    while i <= length(order)
        s = scores[order[i]]
        while i <= length(order) && scores[order[i]] == s
            y[order[i]] == 1 ? (tp += 1) : (fp += 1)
            i += 1
        end
        push!(thresholds, Float64(s))
        push!(fpr, n_neg == 0 ? NaN : fp / n_neg)
        push!(tpr, n_pos == 0 ? NaN : tp / n_pos)
    end
    return fpr, tpr, thresholds
end

"""
    roc_auc(fpr, tpr)

Area under the ROC curve by the trapezoidal rule; `NaN` when a rate is
undefined.
"""
function roc_auc(fpr::AbstractVector{<:Real}, tpr::AbstractVector{<:Real})
    length(fpr) == length(tpr) || throw(DimensionMismatch("fpr and tpr differ in length."))
    (any(isnan, fpr) || any(isnan, tpr)) && return NaN
    area = 0.0
    for k in 2:length(fpr)
        area += (fpr[k] - fpr[k-1]) * (tpr[k] + tpr[k-1]) / 2
    end
    return area
end

"""
    contiguous_runs(mask) -> Vector{UnitRange{Int}}

Maximal runs of `true` in `mask`, in order.
"""
function contiguous_runs(mask::AbstractVector{Bool})
    runs = UnitRange{Int}[]
    start = 0
    for (i, m) in enumerate(mask)
        if m && start == 0
            start = i
        elseif !m && start != 0
            push!(runs, start:(i-1))
            start = 0
        end
    end
    start != 0 && push!(runs, start:length(mask))
    return runs
end

"""
    event_metrics(decisions, labels; step_size, sample_rate) -> NamedTuple

Window-level and event-level detection statistics of binary `decisions`
against binary `labels` over chronologically ordered windows advanced by
`step_size` samples at `sample_rate` [Hz]:

- `precision`, `recall`, `fpr` (the false-positive rate, i.e. the alarm
  duty cycle on unlabelled windows), `f1`, `balanced_accuracy` at window
  level (`NaN` where undefined);
- `n_events`: contiguous runs of positive labels; `n_detected`: events
  with at least one alarm inside their run; `event_recall`;
- `n_false_alarm_episodes`: contiguous runs of alarmed windows outside
  the labelled spans (an alarm that covers an event and extends beyond it
  contributes its unlabelled excess, so a permanently raised alarm is not
  free of false alarms); `observation_days`; `false_alarms_per_30d`, the
  operational false-alarm rate.
"""
function event_metrics(
    decisions::AbstractVector{<:Integer},
    labels::AbstractVector{<:Integer};
    step_size::Integer,
    sample_rate::Real,
)
    n = length(labels)
    n == length(decisions) ||
        throw(DimensionMismatch("$(length(decisions)) decisions for $n labels."))
    (step_size >= 1 && sample_rate > 0) ||
        throw(ArgumentError("step_size and sample_rate must be positive."))
    d = decisions .== 1
    l = labels .== 1
    tp = count(d .& l)
    fp = count(d .& .!l)
    fn = count(.!d .& l)
    tn = count(.!d .& .!l)
    precision = tp + fp == 0 ? NaN : tp / (tp + fp)
    recall = tp + fn == 0 ? NaN : tp / (tp + fn)
    specificity = tn + fp == 0 ? NaN : tn / (tn + fp)
    f1 =
        (isnan(precision) || isnan(recall) || precision + recall == 0) ? NaN :
        2 * precision * recall / (precision + recall)
    balanced_accuracy =
        (isnan(recall) || isnan(specificity)) ? NaN : (recall + specificity) / 2

    events = contiguous_runs(l)
    n_detected = count(r -> any(view(d, r)), events)
    n_false_alarm_episodes = length(contiguous_runs(d .& .!l))
    observation_days = n * step_size / sample_rate / 86400
    return (
        precision = precision,
        recall = recall,
        fpr = tn + fp == 0 ? NaN : fp / (tn + fp),
        f1 = f1,
        balanced_accuracy = balanced_accuracy,
        n_events = length(events),
        n_detected = n_detected,
        event_recall = isempty(events) ? NaN : n_detected / length(events),
        n_false_alarm_episodes = n_false_alarm_episodes,
        observation_days = observation_days,
        false_alarms_per_30d = observation_days == 0 ? NaN :
                               n_false_alarm_episodes / observation_days * 30,
    )
end

"""
    threshold_sweep(y, scores; step_size, sample_rate, n_candidates = 400) -> DataFrame

Window- and event-level detection statistics ([`event_metrics`](@ref)) of
the decision `score >= threshold` at every candidate threshold, the
`n_candidates` quantiles of `scores` in ascending order: one row per
candidate with `threshold`, `precision`, `recall`, `fpr`, `n_events`,
`n_detected`, `event_recall`, `n_false_alarm_episodes`, and
`false_alarms_per_30d`. The table is the event-level operating
characteristic on which [`select_threshold`](@ref) fits the decision
threshold; the stages persist it as `threshold_sweep.csv`.
"""
function threshold_sweep(
    y::AbstractVector{<:Integer},
    scores::AbstractVector{<:Real};
    step_size::Integer,
    sample_rate::Real,
    n_candidates::Integer = 400,
)
    length(y) == length(scores) ||
        throw(DimensionMismatch("$(length(y)) labels for $(length(scores)) scores."))
    isempty(y) && throw(ArgumentError("no windows to sweep."))
    n_candidates >= 2 || throw(ArgumentError("n_candidates = $n_candidates; at least 2."))
    candidates = unique(quantile(scores, range(0, 1; length = n_candidates)))
    sort!(candidates)
    sweep = DataFrame(
        threshold = Float64[],
        precision = Float64[],
        recall = Float64[],
        fpr = Float64[],
        n_events = Int[],
        n_detected = Int[],
        event_recall = Float64[],
        n_false_alarm_episodes = Int[],
        false_alarms_per_30d = Float64[],
    )
    for c in candidates
        m = event_metrics(
            Int.(scores .>= c),
            y;
            step_size = step_size,
            sample_rate = sample_rate,
        )
        push!(
            sweep,
            (
                Float64(c),
                m.precision,
                m.recall,
                m.fpr,
                m.n_events,
                m.n_detected,
                m.event_recall,
                m.n_false_alarm_episodes,
                m.false_alarms_per_30d,
            ),
        )
    end
    return sweep
end

"""
    select_threshold(y, scores; criterion = "far", target_far_per_30d = 3.0,
                     target_fpr = 0.05, step_size, sample_rate, n_candidates = 400)
        -> (threshold, info)

Decision threshold fitted on a calibration block of labels `y` and `scores`
from the event-level operating characteristic [`threshold_sweep`](@ref);
[`threshold_rows`](@ref) selects the rows of that block:

- `"far"`: the operating point of an alert trigger. A candidate is
  admissible when its false-alarm episode rate does not exceed
  `target_far_per_30d` and its window false-positive rate (the alarm
  duty cycle on unlabelled windows) does not exceed `target_fpr`. The
  candidates are scanned from the highest downwards and the threshold is
  the lowest candidate of the admissible range that starts at the top —
  the highest recall reachable while alarms remain short and isolated.
  The scan direction matters because the episode count is not monotone
  in the threshold: as the threshold falls, spurious episodes first
  multiply and then merge into a permanently raised alarm counted as a
  few long episodes, which an ascending scan would accept.
- `"fpr"`: the lowest threshold whose window false-positive rate does not
  exceed `target_fpr` (monotone, so the scan direction is immaterial).
- `"youden"`: the maximiser of `tpr - fpr` (requires both classes).

When the block holds no positive window, `"youden"` falls back to `"fpr"`
with a warning; the other criteria depend on negatives only. A block
without negatives makes every candidate admissible for `"far"` and
`"fpr"`. `info` records the criterion applied, the targets, the rates on
the fitting block at the threshold (`fit_*`), and the block size. Among
them `fit_false_alarm_episodes` is the count the fitted rate rests on:
its inverse square root is the relative error of that rate, and an
operating point placed on a handful of episodes does not transfer to
another record.
"""
function select_threshold(
    y::AbstractVector{<:Integer},
    scores::AbstractVector{<:Real};
    criterion::AbstractString = "far",
    target_far_per_30d::Real = 3.0,
    target_fpr::Real = 0.05,
    step_size::Integer,
    sample_rate::Real,
    n_candidates::Integer = 400,
)
    length(y) == length(scores) ||
        throw(DimensionMismatch("$(length(y)) labels for $(length(scores)) scores."))
    criterion in ("far", "fpr", "youden") || throw(
        ArgumentError("criterion = $(repr(criterion)); expected far, fpr, or youden."),
    )
    isempty(y) && throw(ArgumentError("the threshold fitting block is empty."))
    n_pos = count(==(1), y)
    applied = criterion
    if criterion == "youden" && (n_pos == 0 || n_pos == length(y))
        @warn "Youden's J is undefined with a single class in the fitting block; " *
              "falling back to the false-positive-rate criterion." n_positive = n_pos
        applied = "fpr"
    end
    threshold = Inf
    if applied == "youden"
        fpr, tpr, thresholds = roc_curve(y, scores)
        threshold = thresholds[argmax(tpr .- fpr)]
    else
        sweep = threshold_sweep(
            y,
            scores;
            step_size = step_size,
            sample_rate = sample_rate,
            n_candidates = n_candidates,
        )
        fpr_admissible(row) = isnan(row.fpr) || row.fpr <= target_fpr
        if applied == "far"
            # Descending scan: stop at the first candidate that violates a
            # target; the previous one is the operating point.
            for i in nrow(sweep):-1:1
                row = sweep[i, :]
                (row.false_alarms_per_30d <= target_far_per_30d && fpr_admissible(row)) ||
                    break
                threshold = row.threshold
            end
        else
            i = findfirst(fpr_admissible, eachrow(sweep))
            i === nothing || (threshold = sweep.threshold[i])
        end
        threshold == Inf && @warn "no candidate threshold meets the $applied constraint; " *
              "alarms are disabled (threshold = Inf)."
    end
    m = event_metrics(
        Int.(scores .>= threshold),
        y;
        step_size = step_size,
        sample_rate = sample_rate,
    )
    info = Dict{String,Any}(
        "criterion" => applied,
        "requested_criterion" => criterion,
        "target_far_per_30d" => Float64(target_far_per_30d),
        "target_fpr" => Float64(target_fpr),
        "fit_windows" => length(y),
        "fit_positive_windows" => n_pos,
        "fit_recall" => m.recall,
        "fit_precision" => m.precision,
        "fit_fpr" => m.fpr,
        "fit_false_alarms_per_30d" => m.false_alarms_per_30d,
        "fit_false_alarm_episodes" => m.n_false_alarm_episodes,
        "fit_observation_days" => m.observation_days,
    )
    return Float64(threshold), info
end
