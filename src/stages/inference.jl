# src/stages/inference.jl — inference stage: scoring of a feature table
# with a trained classifier and the threshold its training or calibration
# stage fitted, optional restriction to one block of the run's split,
# per-window results, and window- and event-level metrics when labels are
# available. The script scripts/infer.jl adds only the diagnostic figures.

"""
    check_channels(model_dir, features_path) -> String

The channel set of the run in `model_dir` ([`run_channels`](@ref)),
checked against that of the feature table at `features_path`
([`recorded_channels`](@ref)); a run snapshot or a table that records no
channel set is one of the mode `"A"`, the only mode before the modes were
recorded. Throws an `ArgumentError` when the two differ.
"""
function check_channels(model_dir::AbstractString, features_path::AbstractString)
    trained_on = run_channels(model_dir; default = "A")
    scored_on = recorded_channels(features_path; default = "A")
    trained_on == scored_on || throw(
        ArgumentError(
            "the model of $model_dir was trained on features of the channels " *
            "$trained_on; the table $features_path holds those of $scored_on.",
        ),
    )
    return trained_on
end

"""
    evaluate_classifier(config; run_id = "", model = nothing, features = nothing,
                        labels = nothing, block = nothing) -> NamedTuple

Inference stage driven by the `[inference]` and `[paths]` sections of
`config`; the keyword arguments override the corresponding configuration
keys when given. The model is `model` (a `.jld2` written by
[`save_model`](@ref)) or `gw_model.jld2` of the training run `run_id`;
its directory must hold the `threshold.toml` of the training or
calibration stage. An empty `labels` (or an absent label file) selects
blind inference. `block` restricts the evaluation to one block of the
split the run persisted in `split.toml`: `"validation"` or `"test"` of a
training run, `"calibration"` or `"evaluation"` of a calibration run
([`calibrate_classifier`](@ref)).

The window geometry is that of the table's sidecar
([`feature_geometry`](@ref)), or of `[inference]` with a warning for a
table without one. The features may be the product of a record with gaps
([`preprocess_record`](@ref)): the mission time of a row is then that of
its record window, and events and false-alarm episodes end at a gap, in
the metrics and in the threshold sweep alike ([`event_metrics`](@ref) and
[`threshold_sweep`](@ref) with the window indices of the sidecar).

Artifacts in `<results>/run_<run_id>`: `inference_probabilities.csv`
(`Window`, `Probability`, `Detection`, and with labels `Label`, `SNR`),
the snapshot `config_infer.toml` (section `inference`, plus provenance),
and with labels `metrics.toml` (section `metrics`, the
[`event_metrics`](@ref) fields with `auc`, `n_windows`, `threshold`,
`block`) and `threshold_sweep.csv`, the event-level operating
characteristic of the evaluated rows ([`threshold_sweep`](@ref)); the
persisted threshold is applied unchanged, the sweep is a post-hoc
diagnostic.

Returns `(run_id, results_dir, plot_dir, classifier, probabilities,
decisions, labels, snrs, threshold, threshold_info, metrics, roc, auc,
sweep, days, geometry)`: `classifier` is the loaded model;
`threshold_info` is the dictionary of the run's `threshold.toml`;
`labels`, `snrs`, `metrics`, `roc = (fpr, tpr)`, `auc`, and `sweep` are
`nothing` in blind mode; `days` is the mission time [days] of every
evaluated window; `geometry` is that of [`feature_geometry`](@ref).
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
        geometry = feature_geometry(features_path, config; fallback = "inference")
        # The record window of every row, where the record has gaps
        record_windows =
            gapped_product(features_path) ? window_indices(features_path) : nothing

        plot_dir = joinpath(paths.plots, "run_$output_id")
        results_dir = joinpath(paths.results, "run_$output_id")
        mkpath(plot_dir)
        mkpath(results_dir)

        # Model, scaler, and the threshold fitted at training time.
        classifier, _, stored_scaler = load_model(model_path)
        scaler = require_scaler(stored_scaler, model_path)
        threshold, threshold_info = load_threshold(model_dir)
        check_channels(model_dir, features_path)
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
            # The record windows of the rows, where the record has gaps
            scored_windows = record_windows === nothing ? nothing : record_windows[windows]
            m, auc, metrics = block_metrics(
                probabilities,
                y_true,
                threshold,
                geometry;
                windows = scored_windows,
            )
            metrics["block"] = block_name
            write_toml(
                joinpath(results_dir, "metrics.toml"),
                Dict{String,Any}("metrics" => metrics),
            )
            sweep = threshold_sweep(
                y_true,
                probabilities;
                step_size = geometry.step_size,
                sample_rate = geometry.sample_rate,
                windows = scored_windows,
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
            classifier = classifier,
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
