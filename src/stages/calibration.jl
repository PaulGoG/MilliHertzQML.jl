# src/stages/calibration.jl — calibration stage: the scaler and the
# decision threshold of a trained classifier refitted on the noise of a new
# record, on a chronological calibration block of that record, with its
# parameters unchanged; the rest of the record is left as an evaluation
# block for the inference stage. The script scripts/calibrate.jl adds only
# the command line.

"""
    calibrate_classifier(config; run_id, model, features = nothing, labels = nothing,
                         scaler = nothing) -> NamedTuple

Calibration stage: a trained classifier `model` (the `gw_model.jld2` of a
training run, whose directory holds its `threshold.toml` and `split.toml`)
is given the scaler and the decision threshold of a new record, from the
record's own noise, under the `[calibration]` and `[training]` rules of
`config` ([`calibration_settings`](@ref), [`training_settings`](@ref)).
The record is the feature table `features` (default `[inference]
features`), with the point-wise window labels `labels` when the record
carries coalescences; without labels every window is taken as noise (a
background record). `scaler` overrides `[calibration] scaler`.

1. the first `block_fraction` of the rows, in time order, is the
   calibration block; after a buffer of `window_size ÷ step_size` windows
   (as the training split leaves between its blocks), the rest is the
   evaluation block, scored by [`evaluate_classifier`](@ref) with `block =
   "evaluation"`;
2. the scaler: `"match"`, the noise windows of the calibration block
   encoded where the noise windows of the parent's training block were
   ([`match_scaler`](@ref) at `noise_quantiles`; the reference product is
   `reference_features`/`reference_labels`, else the training product of
   `[training]`, and the training block is that of the parent's
   `split.toml`); `"refit"`, [`fit_scaler`](@ref) on the calibration block
   at `[training] scaler_quantiles`; `"keep"`, the parent's scaler;
3. the threshold, fitted on the calibration block by
   [`select_threshold`](@ref) under the training criterion and targets,
   over the training grid of candidates extended by `tail_candidates`
   into the upper tail of the block's scores (a block without signal
   cannot otherwise reach a target below the quantile step), with the
   operating characteristic of the block ([`threshold_sweep`](@ref)) in
   `threshold_sweep.csv`; a fit resting on fewer than `min_fit_episodes`
   false-alarm episodes is warned about, as in training.

The run directory `<models>/run_<run_id>` holds `gw_model.jld2` (the
parent's parameters with the new scaler; its metadata gains the
`calibration` entry), `threshold.toml` (`block = "calibration"`,
`parent_threshold`, `parent_model`), `split.toml` (`calibration`,
`evaluation`, `buffer_windows`, `n_windows`, `features`),
`calibration.toml` (the fitted quantities: both scalers' bounds, the
counts of noise windows, both thresholds, the fitted false-alarm rate),
the snapshot `config.toml` (sections `model`, `features`, `calibration`,
plus provenance) and `manifest_snapshot.toml`. A product of a record with
gaps is refused (the blocks take consecutive rows as consecutive windows),
as is a table of another channel set than the model was trained on.

Returns `(run_id, run_dir, model_path, threshold, threshold_info, scaler,
parent_scaler, blocks, sweep, elapsed)`: `blocks` is
`(calibration = 1:n_cal, evaluation = ...)`, `sweep` the table of
`threshold_sweep.csv`, `elapsed` the wall time in seconds.
"""
function calibrate_classifier(
    config::AbstractDict;
    run_id::AbstractString,
    model::Union{Nothing,AbstractString} = nothing,
    features::Union{Nothing,AbstractString} = nothing,
    labels::Union{Nothing,AbstractString} = nothing,
    scaler::Union{Nothing,AbstractString} = nothing,
)
    return @timeit TIMER "calibration" begin
        start_time = time()
        isempty(run_id) && throw(ArgumentError("run_id must not be empty."))
        model === nothing && throw(
            ArgumentError(
                "calibration needs the model artifact `model` of a training run.",
            ),
        )
        model_path = resolvepath(model)
        isfile(model_path) || throw(ArgumentError("model artifact not found: $model_path"))
        parent_dir = dirname(model_path)

        paths = pipeline_paths(config)
        cal = calibration_settings(config)
        trn = training_settings(config)
        rule = scaler === nothing ? cal.scaler : String(scaler)
        rule in CALIBRATION_SCALERS || throw(
            ArgumentError(
                "scaler = $(repr(rule)); expected one of " *
                join(("\"$s\"" for s in CALIBRATION_SCALERS), ", ") *
                ".",
            ),
        )

        features_path =
            features === nothing ? inference_settings(config).features :
            resolvepath(features)
        isfile(features_path) ||
            throw(ArgumentError("feature table not found: $features_path"))
        # The blocks take consecutive rows as consecutive windows
        gapped_product(features_path) && throw(
            ArgumentError(
                "the feature table $features_path is the product of a record with " *
                "gaps; calibration on such a product is not supported.",
            ),
        )
        labels_path = labels === nothing || isempty(labels) ? "" : resolvepath(labels)
        has_labels = !isempty(labels_path)
        has_labels &&
            !isfile(labels_path) &&
            throw(ArgumentError("label file not found: $labels_path"))
        trained_on = run_channels(parent_dir)
        scored_on = recorded_channels(features_path; default = "A")
        trained_on == scored_on || throw(
            ArgumentError(
                "the model of $parent_dir was trained on features of the channels " *
                "$trained_on; the table $features_path holds those of $scored_on.",
            ),
        )

        # The parent: parameters, scaler and threshold of the training run
        classifier, parent_metadata, parent_scaler = load_model(model_path)
        parent_scaler === nothing && throw(
            ArgumentError(
                "the model artifact $model_path carries no feature scaler; retrain it " *
                "with the current pipeline.",
            ),
        )
        # The parent's threshold as the training stage stored it (double precision)
        _, parent_info = load_threshold(parent_dir)
        parent_threshold = Float64(parent_info["value"])

        # The record; without labels every window is noise
        if has_labels
            X_raw, y, _ = load_data(features_path, labels_path)
        else
            X_raw = load_features(features_path)
            y = zeros(Int, size(X_raw, 1))
        end
        size(X_raw, 2) == length(parent_scaler.lower) || throw(
            DimensionMismatch(
                "the feature table has $(size(X_raw, 2)) columns; the model's scaler " *
                "expects $(length(parent_scaler.lower)).",
            ),
        )

        # Chronological blocks with the buffer of the training split
        geometry = feature_geometry(features_path, config)
        n = size(X_raw, 1)
        buffer = cld(geometry.window_size, geometry.step_size)
        n_cal = floor(Int, cal.block_fraction * n)
        calibration = 1:n_cal
        evaluation = (n_cal+buffer+1):n
        (isempty(calibration) || isempty(evaluation)) && throw(
            ArgumentError(
                "n = $n windows with block_fraction = $(cal.block_fraction) and " *
                "buffer = $buffer leave an empty block (calibration $calibration, " *
                "evaluation $evaluation).",
            ),
        )

        run_dir = joinpath(paths.models, "run_$run_id")
        mkpath(run_dir)
        @info "calibration run" run_id run_dir parent = model_path features = features_path blind =
            !has_labels rule

        cal_noise = [i for i in calibration if y[i] == 0]
        isempty(cal_noise) &&
            throw(ArgumentError("the calibration block holds no noise window."))

        # The scaler of the record
        if rule == "keep"
            new_scaler = parent_scaler
            reference_count = 0
            reference_path = ""
        elseif rule == "refit"
            new_scaler = fit_scaler(
                X_raw[calibration, :];
                quantiles = trn.scaler_quantiles,
                phase_span = parent_scaler.phase_span,
            )
            reference_count = 0
            reference_path = ""
        else
            reference_path =
                isempty(cal.reference_features) ? trn.train_features :
                cal.reference_features
            reference_labels =
                isempty(cal.reference_labels) ? trn.train_labels : cal.reference_labels
            for path in (reference_path, reference_labels)
                isfile(path) || throw(ArgumentError("reference product not found: $path"))
            end
            X_ref, y_ref, _ = load_data(reference_path, reference_labels)
            split_path = joinpath(parent_dir, "split.toml")
            isfile(split_path) || throw(
                ArgumentError(
                    "scaler = \"match\" needs the split.toml of the training run in " *
                    "$parent_dir.",
                ),
            )
            split = TOML.parsefile(split_path)["split"]
            split["n_windows"] == size(X_ref, 1) || throw(
                DimensionMismatch(
                    "the split of $parent_dir covers $(split["n_windows"]) windows; the " *
                    "reference table $reference_path holds $(size(X_ref, 1)).",
                ),
            )
            train_rows = Int(split["train"][1]):Int(split["train"][2])
            ref_noise = [i for i in train_rows if y_ref[i] == 0]
            isempty(ref_noise) && throw(
                ArgumentError(
                    "the training block of the reference product holds no noise window.",
                ),
            )
            new_scaler = match_scaler(
                parent_scaler,
                X_ref[ref_noise, :],
                X_raw[cal_noise, :];
                quantiles = cal.noise_quantiles,
            )
            reference_count = length(ref_noise)
        end
        @info "scaler" rule reference_noise_windows = reference_count calibration_noise_windows =
            length(cal_noise)

        # The decision threshold fitted on the calibration block
        X_cal = encode_features(new_scaler, X_raw[calibration, :])
        probs_cal = predict_all(classifier, X_cal; threaded = trn.threaded)
        y_cal = y[calibration]
        threshold, info = select_threshold(
            y_cal,
            probs_cal;
            criterion = trn.threshold_criterion,
            target_far_per_30d = trn.target_far_per_30d,
            target_fpr = trn.target_fpr,
            step_size = geometry.step_size,
            sample_rate = geometry.sample_rate,
            tail_candidates = cal.tail_candidates,
        )
        sweep = threshold_sweep(
            y_cal,
            probs_cal;
            step_size = geometry.step_size,
            sample_rate = geometry.sample_rate,
            tail_candidates = cal.tail_candidates,
        )
        write_csv(joinpath(run_dir, "threshold_sweep.csv"), sweep)
        info["value"] = threshold
        info["block"] = "calibration"
        info["fitted_on"] = rootrelative(features_path) * " (calibration block)"
        info["fitted_at"] = string(Dates.now())
        info["parent_threshold"] = parent_threshold
        info["parent_model"] = provenance_path(model_path)
        write_toml(
            joinpath(run_dir, "threshold.toml"),
            Dict{String,Any}("threshold" => info),
        )
        n_episodes = info["fit_false_alarm_episodes"]
        n_episodes < trn.min_fit_episodes && @warn "the fitted false-alarm rate rests " *
              "on $n_episodes false-alarm episodes, below min_fit_episodes = " *
              "$(trn.min_fit_episodes); its relative Poisson error is " *
              (
                  n_episodes == 0 ? "unbounded" :
                  "about $(round(Int, 100 / sqrt(n_episodes))) %"
              ) *
              " and the operating point may not transfer to another record." block = "calibration"
        @info "threshold" value = threshold parent = parent_threshold criterion =
            info["criterion"] episodes = info["fit_false_alarm_episodes"]

        # The calibrated model: the parent's parameters with the new scaler
        model_out = joinpath(run_dir, "gw_model.jld2")
        backup_existing!(model_out)
        metadata = Dict{String,Any}(parent_metadata)
        metadata["calibration"] = Dict{String,Any}(
            "run_id" => String(run_id),
            "parent_model" => provenance_path(model_path),
            "features" => provenance_path(features_path),
            "scaler" => rule,
        )
        save_model(model_out, classifier; metadata = metadata, scaler = new_scaler)

        write_toml(
            joinpath(run_dir, "split.toml"),
            Dict{String,Any}(
                "split" => Dict{String,Any}(
                    "n_windows" => n,
                    "buffer_windows" => buffer,
                    "calibration" => [first(calibration), last(calibration)],
                    "evaluation" => [first(evaluation), last(evaluation)],
                    "features" => rootrelative(features_path),
                ),
            ),
        )
        write_toml(
            joinpath(run_dir, "calibration.toml"),
            Dict{String,Any}(
                "calibration" => Dict{String,Any}(
                    "scaler" => rule,
                    "noise_quantiles" => collect(cal.noise_quantiles),
                    "block_fraction" => cal.block_fraction,
                    "tail_candidates" => cal.tail_candidates,
                    "calibration_rows" => [first(calibration), last(calibration)],
                    "evaluation_rows" => [first(evaluation), last(evaluation)],
                    "buffer_windows" => buffer,
                    "blind" => !has_labels,
                    "calibration_noise_windows" => length(cal_noise),
                    "reference_noise_windows" => reference_count,
                    "reference_features" =>
                        isempty(reference_path) ? "" : provenance_path(reference_path),
                    "parent_scaler_lower" => Float64.(parent_scaler.lower),
                    "parent_scaler_upper" => Float64.(parent_scaler.upper),
                    "scaler_lower" => Float64.(new_scaler.lower),
                    "scaler_upper" => Float64.(new_scaler.upper),
                    "phase_span" => Float64(new_scaler.phase_span),
                    "parent_threshold" => parent_threshold,
                    "threshold" => Float64(threshold),
                    "fit_false_alarms_per_30d" => info["fit_false_alarms_per_30d"],
                    "fit_false_alarm_episodes" => info["fit_false_alarm_episodes"],
                    "parent_model" => provenance_path(model_path),
                    "features" => provenance_path(features_path),
                    "labels" => has_labels ? provenance_path(labels_path) : "",
                ),
            ),
        )

        # The configuration snapshot: the parent's model, the record's
        # geometry, and the rules of this calibration
        parent_config = joinpath(parent_dir, "config.toml")
        model_table =
            isfile(parent_config) ?
            get(TOML.parsefile(parent_config), "model", Dict{String,Any}()) :
            Dict{String,Any}()
        snapshot = Dict{String,Any}(
            "model" => model_table,
            "features" => Dict{String,Any}(
                "channels" => trained_on,
                "window_size" => geometry.window_size,
                "step_size" => geometry.step_size,
                "sample_rate" => geometry.sample_rate,
            ),
            "calibration" => Dict{String,Any}(
                "run_id" => String(run_id),
                "block_fraction" => cal.block_fraction,
                "scaler" => rule,
                "noise_quantiles" => collect(cal.noise_quantiles),
                "tail_candidates" => cal.tail_candidates,
                "reference_features" =>
                    isempty(reference_path) ? "" : rootrelative(reference_path),
                "threshold_criterion" => trn.threshold_criterion,
                "target_far_per_30d" => trn.target_far_per_30d,
                "target_fpr" => trn.target_fpr,
                "min_fit_episodes" => trn.min_fit_episodes,
                "scaler_quantiles" => collect(trn.scaler_quantiles),
                "parent_model" => provenance_path(model_path),
                "features" => provenance_path(features_path),
                "labels" => has_labels ? provenance_path(labels_path) : "",
                "threaded" => trn.threaded,
            ),
        )
        write_toml(joinpath(run_dir, "config.toml"), snapshot)
        snapshot_manifest(run_dir)
        @info "calibration artifacts written" run_dir

        (
            run_id = String(run_id),
            run_dir = run_dir,
            model_path = model_out,
            threshold = threshold,
            threshold_info = info,
            scaler = new_scaler,
            parent_scaler = parent_scaler,
            blocks = (calibration = calibration, evaluation = evaluation),
            sweep = sweep,
            elapsed = time() - start_time,
        )
    end
end
