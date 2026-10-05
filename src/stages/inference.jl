# src/stages/inference.jl — inference stage: scoring of a feature table
# with a trained classifier and its validation-fitted threshold, optional
# restriction to one block of the training table, per-window results, and
# window- and event-level metrics when labels are available. The script
# scripts/infer.jl adds only the diagnostic figures.

"""
    inference_geometry(features_path, config) -> NamedTuple

Window `step_size` [samples], `sample_rate` [Hz], and `first_window` (the
record window index of the table's first row) of a feature table: from
its sidecar TOML when present ([`feature_geometry`](@ref)), otherwise
from the `[inference]` section with a warning (`first_window = 1`).
"""
function inference_geometry(features_path::AbstractString, config::AbstractDict)
    sidecar = replace(features_path, r"\.csv$" => ".toml")
    if isfile(sidecar)
        g = feature_geometry(features_path, config)
        return (
            step_size = g.step_size,
            sample_rate = g.sample_rate,
            first_window = g.first_window,
        )
    end
    @warn "no feature sidecar at $sidecar; window geometry taken from [inference]."
    inf = inference_settings(config)
    return (step_size = inf.step_size, sample_rate = inf.sample_rate, first_window = 1)
end

"""
    gapped_product(features_path) -> Bool

Whether a feature table is the product of a record with gaps: its sidecar
lists the stretches between them, and its rows are not consecutive windows
of the record ([`window_indices`](@ref)).
"""
function gapped_product(features_path::AbstractString)
    sidecar = replace(features_path, r"\.csv$" => ".toml")
    isfile(sidecar) || return false
    return haskey(get(TOML.parsefile(sidecar), "features", Dict{String,Any}()), "stretches")
end

"""
    run_channels(model_dir) -> String

Channel mode of the features a run was trained on, from its `config.toml`
snapshot; `"A"` for a run trained before the modes existed.
"""
function run_channels(model_dir::AbstractString)
    path = joinpath(model_dir, "config.toml")
    isfile(path) || return "A"
    features = get(TOML.parsefile(path), "features", Dict{String,Any}())
    return String(get(features, "channels", "A"))
end

"""
    THRESHOLD_FIT_KEYS

Rates of the fitting block recorded in `threshold.toml` under `fit_<key>`;
files written before the fitting block became configurable hold them
under `validation_<key>`.
"""
const THRESHOLD_FIT_KEYS = (
    "windows",
    "positive_windows",
    "recall",
    "precision",
    "fpr",
    "false_alarms_per_30d",
    "false_alarm_episodes",
    "observation_days",
)

"""
    migrate_threshold_info!(info) -> info

Rename the `validation_*` rate keys of a `threshold.toml` written before
the fitting block became configurable to their current `fit_*` names,
wherever the new name is absent, so that an artifact of either vintage
reads alike; the renamed keys are logged.
"""
function migrate_threshold_info!(info::AbstractDict)
    renamed = String[]
    for k in THRESHOLD_FIT_KEYS
        old, new = "validation_$k", "fit_$k"
        if haskey(info, old) && !haskey(info, new)
            info[new] = info[old]
            delete!(info, old)
            push!(renamed, old)
        end
    end
    isempty(renamed) ||
        @info "threshold.toml predates the configurable fitting block; validation_* keys read as fit_*" renamed
    return info
end

"""
    load_threshold(model_dir) -> (threshold::Float32, info::Dict)

Decision threshold persisted in `threshold.toml` of a training run
directory; the file is required. Older files are read through
[`migrate_threshold_info!`](@ref).
"""
function load_threshold(model_dir::AbstractString)
    path = joinpath(model_dir, "threshold.toml")
    isfile(path) || throw(
        ArgumentError(
            "no decision threshold at $path; the threshold is fitted on the validation " *
            "block by the training stage.",
        ),
    )
    info = migrate_threshold_info!(TOML.parsefile(path)["threshold"])
    haskey(info, "value") ||
        throw(ArgumentError("threshold.toml at $path carries no `value` key."))
    return Float32(info["value"]), info
end

"""
    block_rows(model_dir, block, n_windows) -> UnitRange{Int}

Row range of `block` (`"validation"` or `"test"`) of the training table
from the `split.toml` of the run in `model_dir`; the table must hold the
`n_windows` of the split.
"""
function block_rows(model_dir::AbstractString, block::AbstractString, n_windows::Integer)
    path = joinpath(model_dir, "split.toml")
    isfile(path) || throw(
        ArgumentError(
            "block = $block requires the split.toml of the training run in $model_dir.",
        ),
    )
    split = TOML.parsefile(path)["split"]
    split["n_windows"] == n_windows || throw(
        DimensionMismatch(
            "block = $block applies to the training table of $(split["n_windows"]) " *
            "windows; the given features hold $n_windows.",
        ),
    )
    lo, hi = split[block]
    return Int(lo):Int(hi)
end

"""
    evaluate_classifier(config; run_id = "", model = nothing, features = nothing,
                        labels = nothing, block = nothing) -> NamedTuple

Inference stage driven by the `[inference]` and `[paths]` sections of
`config`; the keyword arguments override the corresponding configuration
keys when given. The model is `model` (a `.jld2` written by
[`save_model`](@ref)) or `gw_model.jld2` of the training run `run_id`;
its directory must hold the `threshold.toml` of the training stage. An
empty `labels` (or an absent label file) selects blind inference.
`block = "validation"` or `"test"` restricts the evaluation to one block of
the training table through the run's `split.toml`.

The features may be the product of a record with gaps
([`preprocess_record`](@ref)): the mission time of a row is then that of
its record window, and events and false-alarm episodes end at a gap
([`event_metrics`](@ref) with the window indices of the sidecar). The
post-hoc threshold sweep still takes the rows as consecutive.

Artifacts in `<results>/run_<run_id>`: `inference_probabilities.csv`
(`Window`, `Probability`, `Detection`, and with labels `Label`, `SNR`),
the snapshot `config_infer.toml` (section `inference`, plus provenance),
and with labels `metrics.toml` (section `metrics`, the
[`event_metrics`](@ref) fields with `auc`, `n_windows`, `threshold`,
`block`) and `threshold_sweep.csv`, the event-level operating
characteristic of the evaluated rows ([`threshold_sweep`](@ref)); the
persisted threshold is applied unchanged, the sweep is a post-hoc
diagnostic.

Returns `(run_id, results_dir, plot_dir, probabilities, decisions, labels,
snrs, threshold, threshold_info, metrics, roc, auc, sweep, days,
geometry)`: `threshold_info` is the dictionary of the run's
`threshold.toml`; `labels`, `snrs`, `metrics`, `roc = (fpr, tpr)`, `auc`,
and `sweep` are `nothing` in blind mode; `days` is the mission time [days]
of every evaluated window; `geometry` holds `step_size` and `sample_rate`.
"""
function evaluate_classifier(
    config::AbstractDict;
    run_id::AbstractString = "",
    model::Union{Nothing,AbstractString} = nothing,
    features::Union{Nothing,AbstractString} = nothing,
    labels::Union{Nothing,AbstractString} = nothing,
    block::Union{Nothing,AbstractString} = nothing,
)
    return @timeit TIMER "inference" begin
        paths = pipeline_paths(config)
        inf = inference_settings(config)
        features_path = features === nothing ? inf.features : resolvepath(features)
        isfile(features_path) ||
            throw(ArgumentError("feature table not found: $features_path"))
        label_spec = labels === nothing ? inf.labels : labels
        block_name = block === nothing ? inf.block : String(block)
        block_name in ("all", "validation", "test") || throw(
            ArgumentError(
                "block = $(repr(block_name)); expected all, validation, or test.",
            ),
        )
        has_labels = !isempty(label_spec) && isfile(resolvepath(label_spec))
        labels_path = has_labels ? resolvepath(label_spec) : ""

        (model !== nothing || !isempty(run_id)) || throw(
            ArgumentError(
                "no model specified: give `model` or the `run_id` of a training run.",
            ),
        )
        model_path =
            model !== nothing ? resolvepath(model) :
            joinpath(paths.models, "run_$run_id", "gw_model.jld2")
        isfile(model_path) || throw(ArgumentError("model artifact not found: $model_path"))
        model_dir = dirname(model_path)
        output_id =
            isempty(run_id) ?
            "standalone_" * parameter_digest(Dict("model" => abspath(model_path)))[1:6] :
            String(run_id)
        geometry = inference_geometry(features_path, config)
        # The record window of every row, where the record has gaps
        record_windows =
            gapped_product(features_path) ? window_indices(features_path) : nothing

        plot_dir = joinpath(paths.plots, "run_$output_id")
        results_dir = joinpath(paths.results, "run_$output_id")
        mkpath(plot_dir)
        mkpath(results_dir)

        # Model, scaler, and the threshold fitted at training time.
        classifier, _, scaler = load_model(model_path)
        scaler === nothing && throw(
            ArgumentError(
                "the model artifact $model_path carries no feature scaler; retrain it " *
                "with the current pipeline.",
            ),
        )
        threshold, threshold_info = load_threshold(model_dir)
        trained_on = run_channels(model_dir)
        scored_on = recorded_channels(features_path; default = "A")
        trained_on == scored_on || throw(
            ArgumentError(
                "the model of $model_dir was trained on features of the channels " *
                "$trained_on; the table $features_path holds those of $scored_on.",
            ),
        )
        @info "model loaded" model_path = model_path threshold = threshold criterion =
            get(threshold_info, "criterion", "unknown") fitted_at =
            get(threshold_info, "fitted_at", "unknown")

        # Data, restricted to a block of the training table when requested.
        if has_labels
            X_raw, y_true, df_labels = load_data(features_path, labels_path)
            snrs =
                "SNR" in names(df_labels) ? Vector{Float64}(df_labels[:, :SNR]) :
                zeros(Float64, size(X_raw, 1))
        else
            @info "blind mode: no labels"
            X_raw = load_features(features_path)
            y_true = nothing
            snrs = nothing
        end
        row_offset = 0
        if block_name != "all"
            rows = block_rows(model_dir, block_name, size(X_raw, 1))
            row_offset = first(rows) - 1
            X_raw = X_raw[rows, :]
            if has_labels
                y_true = y_true[rows]
                snrs = snrs[rows]
            end
            @info "evaluating the $block_name block" windows = rows
        end
        X = encode_features(scaler, X_raw)

        # Forward pass over all windows.
        n_windows = size(X, 1)
        println("Analyzing $n_windows windows...")
        probabilities = predict_all(classifier, X; progress = true)
        decisions = Int.(probabilities .>= threshold)
        windows = row_offset .+ (1:n_windows)
        step_duration = geometry.step_size / geometry.sample_rate
        # Mission time of a row: its record window index (the table may
        # start after an edge margin) times the step
        days = Float64[
            (
                record_windows === nothing ? geometry.first_window - 1 + w :
                record_windows[w]
            ) * step_duration / 86400 for w in windows
        ]

        # Per-window scores and decisions, the snapshot, and the metrics.
        results = DataFrame(
            Window = collect(windows),
            Probability = probabilities,
            Detection = decisions,
        )
        if has_labels
            results.Label = y_true
            results.SNR = snrs
        end
        write_csv(joinpath(results_dir, "inference_probabilities.csv"), results)
        write_toml(
            joinpath(results_dir, "config_infer.toml"),
            Dict{String,Any}(
                "inference" => Dict{String,Any}(
                    "features" => provenance_path(features_path),
                    "labels" => has_labels ? provenance_path(labels_path) : "",
                    "model" => provenance_path(model_path),
                    "block" => block_name,
                    "step_size" => geometry.step_size,
                    "sample_rate" => geometry.sample_rate,
                    "threshold" => Float64(threshold),
                    "run_id" => output_id,
                    "blind" => !has_labels,
                ),
            ),
        )
        snapshot_manifest(results_dir)

        metrics = nothing
        roc = nothing
        auc = nothing
        sweep = nothing
        if has_labels
            fpr, tpr, _ = roc_curve(y_true, probabilities)
            roc = (fpr = fpr, tpr = tpr)
            auc = roc_auc(fpr, tpr)
            m = event_metrics(
                decisions,
                y_true;
                step_size = geometry.step_size,
                sample_rate = geometry.sample_rate,
                windows = record_windows === nothing ? nothing : record_windows[windows],
            )
            metrics = merge(
                metrics_dict(m),
                Dict{String,Any}(
                    "auc" => auc,
                    "n_windows" => n_windows,
                    "threshold" => Float64(threshold),
                    "block" => block_name,
                ),
            )
            write_toml(
                joinpath(results_dir, "metrics.toml"),
                Dict{String,Any}("metrics" => metrics),
            )
            sweep = threshold_sweep(
                y_true,
                probabilities;
                step_size = geometry.step_size,
                sample_rate = geometry.sample_rate,
            )
            write_csv(joinpath(results_dir, "threshold_sweep.csv"), sweep)
            @info "window-level metrics" auc = auc precision = m.precision recall = m.recall f1 =
                m.f1 balanced_accuracy = m.balanced_accuracy
            @info "event-level metrics" n_events = m.n_events n_detected = m.n_detected n_false_alarm_episodes =
                m.n_false_alarm_episodes observation_days = m.observation_days false_alarms_per_30d =
                m.false_alarms_per_30d
        end
        @info "inference artifacts written" results_dir = results_dir

        (
            run_id = output_id,
            results_dir = results_dir,
            plot_dir = plot_dir,
            probabilities = probabilities,
            decisions = decisions,
            labels = y_true,
            snrs = snrs,
            threshold = threshold,
            threshold_info = threshold_info,
            metrics = metrics,
            roc = roc,
            auc = auc,
            sweep = sweep,
            days = days,
            geometry = geometry,
        )
    end
end
