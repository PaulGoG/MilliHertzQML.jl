# src/evaluation.jl — chronological partitioning, ROC analysis, decision
# thresholds fitted on the validation block, and event-level detection
# metrics with the operational false-alarm rate.

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

- `precision`, `recall`, `f1`, `balanced_accuracy` at window level
  (`NaN` where undefined);
- `n_events`: contiguous runs of positive labels; `n_detected`: events
  with at least one alarm inside their run; `event_recall`;
- `n_false_alarm_episodes`: contiguous runs of alarmed windows outside
  the labeled spans (an alarm that covers an event and extends beyond it
  contributes its unlabeled excess, so a permanently raised alarm is not
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
    select_threshold(y, scores; criterion = "far", target_far_per_30d = 1.0,
                     target_fpr = 0.05, step_size, sample_rate, n_candidates = 400)
        -> (threshold, info)

Decision threshold fitted on a validation block of labels `y` and `scores`:

- `"far"`: the lowest threshold (highest recall) whose false-alarm
  episode rate ([`event_metrics`](@ref)) does not exceed
  `target_far_per_30d`;
- `"fpr"`: the lowest threshold whose window-level false-positive rate
  does not exceed `target_fpr`;
- `"youden"`: the maximizer of `tpr - fpr` (requires both classes).

Candidate thresholds are the `n_candidates` quantiles of the scores. When
the block holds no positive window, `"youden"` falls back to `"fpr"` with
a warning; the other criteria depend on negatives only. `info` records
the criterion applied, the validation rates at the threshold, and the
block size.
"""
function select_threshold(
    y::AbstractVector{<:Integer},
    scores::AbstractVector{<:Real};
    criterion::AbstractString = "far",
    target_far_per_30d::Real = 1.0,
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
    isempty(y) && throw(ArgumentError("the validation block is empty."))
    n_pos = count(==(1), y)
    applied = criterion
    if criterion == "youden" && (n_pos == 0 || n_pos == length(y))
        @warn "Youden's J is undefined with a single class in the validation block; " *
              "falling back to the false-positive-rate criterion." n_positive = n_pos
        applied = "fpr"
    end
    candidates = unique(quantile(scores, range(0, 1; length = n_candidates)))
    sort!(candidates)
    n_neg = length(y) - n_pos
    threshold = Inf
    if applied == "youden"
        fpr, tpr, thresholds = roc_curve(y, scores)
        threshold = thresholds[argmax(tpr .- fpr)]
    else
        # Ascending scan: the first admissible candidate is the lowest
        # threshold, hence the highest recall, meeting the constraint.
        for c in candidates
            alarms = scores .>= c
            if applied == "far"
                m = event_metrics(
                    Int.(alarms),
                    y;
                    step_size = step_size,
                    sample_rate = sample_rate,
                )
                admissible = m.false_alarms_per_30d <= target_far_per_30d
            else
                fp = count(alarms .& (y .== 0))
                admissible = n_neg == 0 || fp / n_neg <= target_fpr
            end
            if admissible
                threshold = c
                break
            end
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
        "validation_windows" => length(y),
        "validation_positive_windows" => n_pos,
        "validation_recall" => m.recall,
        "validation_precision" => m.precision,
        "validation_false_alarms_per_30d" => m.false_alarms_per_30d,
        "validation_observation_days" => m.observation_days,
    )
    return Float64(threshold), info
end
