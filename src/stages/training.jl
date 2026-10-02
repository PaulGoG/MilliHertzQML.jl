# src/stages/training.jl — training stage: chronological block split,
# class-weighted training with early stopping on the validation block, a
# decision threshold fitted on the calibration block, and a single scoring
# of the test block. The script scripts/train.jl adds only the terminal
# dashboard, the file logger, and the training-history figure.

"""
    predict_all(model, X; progress = false, threaded = Threads.nthreads() > 1)
        -> Vector{Float32}

Classifier probability of every row of the encoded feature matrix `X`
(samples × features), evaluated in place on one [`CircuitWorkspace`](@ref)
per task and over the Julia threads when `threaded`, the tasks drawing
blocks of rows from a shared counter (the rows are independent, so the
result is the same either way, and equal to [`predict_probability`](@ref)
row by row); with `progress`, a line is printed after every tenth of the
rows.
"""
function predict_all(
    model::VariationalQuantumClassifier,
    X::AbstractMatrix{<:Real};
    progress::Bool = false,
    threaded::Bool = Threads.nthreads() > 1,
)
    n = size(X, 1)
    probabilities = Vector{Float32}(undef, n)
    n_tasks = threaded ? clamp(cld(n, SCORING_BLOCK), 1, Threads.nthreads()) : 1
    workspaces = [CircuitWorkspace(model) for _ in 1:n_tasks]
    tenth = max(1, cld(n, 10))
    for lo in 1:tenth:n
        hi = min(n, lo + tenth - 1)
        next_row = Threads.Atomic{Int}(lo)
        if n_tasks == 1
            score_rows!(probabilities, workspaces[1], X, next_row, hi)
        else
            @sync for t in 1:n_tasks
                Threads.@spawn score_rows!(probabilities, workspaces[t], X, next_row, hi)
            end
        end
        progress && println("  Progress: $(round(Int, hi / n * 100)) %")
    end
    return probabilities
end

"""
Rows a task of [`predict_all`](@ref) draws at a time from the shared counter.
"""
const SCORING_BLOCK = 16

# Scores blocks of rows drawn from the shared counter until row `hi` is passed
function score_rows!(probabilities, workspace, X, next_row, hi)
    while true
        first_row = Threads.atomic_add!(next_row, SCORING_BLOCK)
        first_row > hi && break
        for i in first_row:min(hi, first_row+SCORING_BLOCK-1)
            probabilities[i] = predict_probability!(workspace, @view(X[i, :]))
        end
    end
    return nothing
end

"""
    metrics_dict(m) -> Dict{String, Any}

The fields of an [`event_metrics`](@ref) named tuple keyed by name, as
persisted in the `metrics.toml` snapshots.
"""
metrics_dict(m::NamedTuple) = Dict{String,Any}(String(k) => v for (k, v) in pairs(m))

"""
    fixed(x, digits, width) -> String

`x` in fixed-point notation with `digits` decimals, right-aligned in
`width` characters; non-finite values print as such.
"""
function fixed(x::Real, digits::Integer, width::Integer)
    s = string(isfinite(x) ? round(Float64(x); digits = digits) : Float64(x))
    if isfinite(x) && !occursin('e', s)
        head, tail = split(s, '.')
        s = head * "." * rpad(tail, digits, '0')
    end
    return lpad(s, width)
end

"""
    print_block_summary(io, rows)

Console table of the window- and event-level detection statistics of the
evaluated blocks; `rows` holds `(name, metrics, auc)` triples with
`metrics` from [`event_metrics`](@ref).
"""
function print_block_summary(io::IO, rows)
    println(io)
    println(
        io,
        "  Block        AUC      Prec.   Recall  F1      Bal.acc  Events  Detected  FA/30 d",
    )
    for (name, m, auc) in rows
        println(
            io,
            "  ",
            rpad(name, 11),
            "  ",
            fixed(auc, 3, 6),
            "  ",
            fixed(m.precision, 3, 6),
            "  ",
            fixed(m.recall, 3, 6),
            "  ",
            fixed(m.f1, 3, 6),
            "  ",
            fixed(m.balanced_accuracy, 3, 6),
            "  ",
            lpad(m.n_events, 6),
            "  ",
            lpad(m.n_detected, 8),
            "  ",
            fixed(m.false_alarms_per_30d, 2, 7),
        )
    end
    return nothing
end

"""
    block_metrics(probabilities, labels, threshold, geometry) -> (metrics, auc, dict)

Event-level metrics of a scored block at the decision `threshold`, its ROC
area, and the merged dictionary persisted in `metrics.toml`
(`n_windows`, `threshold`, `auc` added to the [`event_metrics`](@ref)
fields).
"""
function block_metrics(
    probabilities::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
    threshold::Real,
    geometry::NamedTuple,
)
    m = event_metrics(
        Int.(probabilities .>= threshold),
        labels;
        step_size = geometry.step_size,
        sample_rate = geometry.sample_rate,
    )
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc = roc_auc(fpr, tpr)
    d = merge(
        metrics_dict(m),
        Dict{String,Any}(
            "auc" => auc,
            "n_windows" => length(labels),
            "threshold" => Float64(threshold),
        ),
    )
    return m, auc, d
end

"""
    epoch_validation(model, X, y; positive_weight, threaded = Threads.nthreads() > 1)
        -> (loss, accuracy)

Mean weighted binary cross-entropy ([`weighted_bce`](@ref)) and accuracy
at the 0.5 decision of `model` over the rows of `X` with labels `y`, from
one forward pass ([`predict_all`](@ref)).
"""
function epoch_validation(
    model::VariationalQuantumClassifier,
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer};
    positive_weight::Real,
    threaded::Bool = Threads.nthreads() > 1,
)
    n = length(y)
    size(X, 1) == n || throw(DimensionMismatch("$(size(X, 1)) rows for $n labels."))
    n >= 1 || throw(ArgumentError("the validation block is empty."))
    p = predict_all(model, X; threaded = threaded)
    loss = 0.0f0
    correct = 0
    for i in 1:n
        loss += weighted_bce(p[i], y[i]; positive_weight = positive_weight)
        correct += (p[i] > 0.5f0) == (y[i] == 1)
    end
    return loss / Float32(n), Float32(correct / n)
end

"""
    train_classifier(config; run_id = new_run_id(), test_mode = false, on_epoch = nothing)
        -> NamedTuple

Training stage driven by the `[model]`, `[training]`, `[paths]`, and
`[resources]` sections of `config`:

1. memory pre-flight of one training step against the `[resources]`
   thresholds;
2. run directory `<models>/run_<run_id>` with the configuration snapshot
   `config.toml` (sections `model`, `training`, `features`, plus
   provenance) and a copy of the resolved Manifest
   (`manifest_snapshot.toml`, [`snapshot_manifest`](@ref));
3. feature and label tables of `[training]`, capped to
   `test_mode_samples` windows and `test_mode_epochs` epochs under
   `test_mode`;
4. chronological split into training, validation, and test blocks with a
   one-window buffer ([`chronological_split`](@ref)), persisted in
   `split.toml`; feature scaler fitted on the training block only;
5. Adam with exponential learning-rate decay, the positive class weighted
   by the negative-to-positive count ratio under
   `class_weight = "balanced"`, batch gradients by `gradient_method`
   ([`batch_gradient`](@ref)) and forward passes over the Julia threads
   when `threaded`, early
   stopping on the validation loss with the `patience` of the
   configuration; the best epoch is kept in
   `gw_model_best.jld2` and copied to `gw_model.jld2`;
6. decision threshold fitted on the calibration block selected by
   `threshold_block` ([`threshold_rows`](@ref), [`select_threshold`](@ref)),
   persisted in `threshold.toml`, with the event-level operating
   characteristic of that block ([`threshold_sweep`](@ref)) in
   `threshold_sweep.csv`;
7. the test block scored once; window- and event-level metrics of both
   blocks in `metrics.toml` and printed as a table. Under the default
   `threshold_block = "held_out"` the test block is part of the
   calibration set and its metrics are not an independent check of the
   operating point; a separate observation record is.

`on_epoch`, when given, is called after every epoch as
`on_epoch(epoch, max_epochs, learning_rate, history, elapsed_seconds)`.

Returns `(run_id, run_dir, model_path, threshold, threshold_info, sweep,
metrics, history, blocks, elapsed)`: `threshold_info` and `metrics` are the
dictionaries of `threshold.toml` and `metrics.toml` (sections
`"validation"` and `"test"`), `sweep` the table of `threshold_sweep.csv`,
`history` the per-epoch vectors `epochs`, `train_loss`, `val_loss`,
`val_acc`, `blocks` the split ranges, and `elapsed` the wall time of the
stage in seconds.
"""
function train_classifier(
    config::AbstractDict;
    run_id::AbstractString = new_run_id(),
    test_mode::Bool = false,
    on_epoch = nothing,
)
    isempty(run_id) && throw(ArgumentError("run_id must not be empty."))
    return @timeit TIMER "training" begin
        start_time = time()
        paths = pipeline_paths(config)
        mdl = model_settings(config)
        trn = training_settings(config)
        resources = resource_settings(config)
        max_epochs = test_mode ? trn.test_mode_epochs : trn.epochs
        check_memory(
            training_memory_estimate_gib(
                mdl.n_qubits,
                mdl.n_layers,
                trn.batch_size;
                gradient_method = trn.gradient_method,
            ),
            resources;
            stage = "training",
        )
        Random.seed!(trn.seed)

        run_dir = joinpath(paths.models, "run_$run_id")
        mkpath(run_dir)
        geometry = feature_geometry(trn.train_features, config)
        # The product is authoritative: a configuration naming another
        # channel mode than its feature table records is refused
        channels = recorded_channels(trn.train_features)
        channels == tdi_settings(config).channels || throw(
            ArgumentError(
                "[tdi] channels = \"$(tdi_settings(config).channels)\", but the feature " *
                "table $(trn.train_features) holds the channels $channels.",
            ),
        )
        snapshot = Dict{String,Any}(
            "model" => Dict{String,Any}(
                "n_qubits" => mdl.n_qubits,
                "n_layers" => mdl.n_layers,
            ),
            "training" => Dict{String,Any}(
                "epochs" => max_epochs,
                "batch_size" => trn.batch_size,
                "learning_rate" => trn.learning_rate,
                "lr_decay" => trn.lr_decay,
                "patience" => trn.patience,
                "train_fraction" => trn.train_fraction,
                "validation_fraction" => trn.validation_fraction,
                "class_weight" => trn.class_weight,
                "threshold_criterion" => trn.threshold_criterion,
                "target_far_per_30d" => trn.target_far_per_30d,
                "target_fpr" => trn.target_fpr,
                "scaler_quantiles" => collect(trn.scaler_quantiles),
                "phase_span" => trn.phase_span,
                "threshold_block" => trn.threshold_block,
                "min_fit_episodes" => trn.min_fit_episodes,
                "train_features" => rootrelative(trn.train_features),
                "train_labels" => rootrelative(trn.train_labels),
                "test_mode" => test_mode,
                "threaded" => trn.threaded,
                "gradient_method" => trn.gradient_method,
                "run_id" => run_id,
                "seed" => trn.seed,
            ),
            "features" => Dict{String,Any}(
                "channels" => channels,
                "window_size" => geometry.window_size,
                "step_size" => geometry.step_size,
                "sample_rate" => geometry.sample_rate,
            ),
        )
        write_toml(joinpath(run_dir, "config.toml"), snapshot)
        snapshot_manifest(run_dir)
        @info "training run" run_id = run_id run_dir = run_dir test_mode = test_mode

        X_raw, y_raw, _ = load_data(trn.train_features, trn.train_labels)
        size(X_raw, 2) == mdl.n_qubits || throw(
            DimensionMismatch(
                "feature dimension $(size(X_raw, 2)) does not match n_qubits = $(mdl.n_qubits).",
            ),
        )
        if test_mode
            n_keep = min(trn.test_mode_samples, size(X_raw, 1))
            @info "test mode: keeping the first $n_keep windows"
            X_raw = X_raw[1:n_keep, :]
            y_raw = y_raw[1:n_keep]
        end

        # Chronological block split with a one-window buffer.
        buffer = cld(geometry.window_size, geometry.step_size)
        blocks = chronological_split(
            size(X_raw, 1);
            train_fraction = trn.train_fraction,
            validation_fraction = trn.validation_fraction,
            buffer = buffer,
        )
        write_toml(
            joinpath(run_dir, "split.toml"),
            Dict{String,Any}(
                "split" => Dict{String,Any}(
                    "n_windows" => size(X_raw, 1),
                    "buffer_windows" => buffer,
                    "train" => [first(blocks.train), last(blocks.train)],
                    "validation" => [first(blocks.validation), last(blocks.validation)],
                    "test" => [first(blocks.test), last(blocks.test)],
                    "features" => rootrelative(trn.train_features),
                ),
            ),
        )
        @info "chronological split" train = blocks.train validation = blocks.validation test =
            blocks.test buffer = buffer

        # Feature scaler fitted on the training block only.
        scaler = fit_scaler(
            X_raw[blocks.train, :];
            quantiles = trn.scaler_quantiles,
            phase_span = trn.phase_span * π,
        )
        X_train = encode_features(scaler, X_raw[blocks.train, :])
        X_val = encode_features(scaler, X_raw[blocks.validation, :])
        X_test = encode_features(scaler, X_raw[blocks.test, :])
        y_train = y_raw[blocks.train]
        y_val = y_raw[blocks.validation]
        y_test = y_raw[blocks.test]

        n_pos = count(==(1), y_train)
        n_neg = length(y_train) - n_pos
        positive_weight = 1.0
        if trn.class_weight == "balanced"
            if n_pos == 0
                @warn "no positive window in the training block; class weighting disabled."
            else
                positive_weight = n_neg / n_pos
            end
        end
        @info "training block" windows = length(y_train) positive = n_pos positive_weight =
            positive_weight

        n_train = length(y_train)
        n_batches = cld(n_train, trn.batch_size)

        model = VariationalQuantumClassifier(mdl.n_qubits, mdl.n_layers)
        opt_state = Optimisers.setup(Optimisers.Adam(trn.learning_rate), model.params)
        history = (
            epochs = Int[],
            train_loss = Float32[],
            val_loss = Float32[],
            val_acc = Float32[],
        )
        best_path = joinpath(run_dir, "gw_model_best.jld2")
        model_path = joinpath(run_dir, "gw_model.jld2")
        backup_existing!(best_path)
        backup_existing!(model_path)
        best_val_loss = Inf32
        epochs_no_improve = 0
        loop_start = time()
        threaded = trn.threaded && Threads.nthreads() > 1
        method = Symbol(trn.gradient_method)
        workspaces =
            method == :adjoint ?
            [
                CircuitWorkspace(model) for
                _ in 1:gradient_tasks(trn.batch_size; threaded = threaded)
            ] : nothing
        @info "training started" batch_size = trn.batch_size max_epochs = max_epochs n_qubits =
            mdl.n_qubits n_layers = mdl.n_layers gradient_method = trn.gradient_method threaded =
            threaded threads = Threads.nthreads()

        for epoch in 1:max_epochs
            current_lr = trn.learning_rate * trn.lr_decay^(epoch - 1)
            Optimisers.adjust!(opt_state, current_lr)

            # One random permutation of the training rows per epoch, cut into
            # consecutive mini-batches; the last one may be short
            order = randperm(n_train)
            epoch_train_loss = 0.0f0
            for b in 1:n_batches
                rows = @view(order[((b-1)*trn.batch_size+1):min(n_train, b*trn.batch_size)])
                epoch_train_loss += train_step!(
                    model,
                    opt_state,
                    X_train[rows, :],
                    y_train[rows];
                    positive_weight = positive_weight,
                    threaded = threaded,
                    method = method,
                    workspaces = workspaces,
                )
            end
            avg_train_loss = epoch_train_loss / n_batches
            avg_val_loss, avg_val_acc = epoch_validation(
                model,
                X_val,
                y_val;
                positive_weight = positive_weight,
                threaded = threaded,
            )

            push!(history.epochs, epoch)
            push!(history.train_loss, avg_train_loss)
            push!(history.val_loss, avg_val_loss)
            push!(history.val_acc, avg_val_acc)
            elapsed = time() - loop_start
            on_epoch === nothing ||
                on_epoch(epoch, max_epochs, current_lr, history, elapsed)
            @info "epoch complete" epoch = epoch lr = current_lr train_loss = avg_train_loss val_loss =
                avg_val_loss val_acc = avg_val_acc elapsed = elapsed

            if avg_val_loss < best_val_loss
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                save_model(
                    best_path,
                    model;
                    metadata = Dict{String,Any}(
                        "run_id" => run_id,
                        "seed" => trn.seed,
                        "epoch" => epoch,
                        "val_loss" => avg_val_loss,
                        "config" => snapshot,
                        "provenance" => provenance(),
                    ),
                    scaler = scaler,
                )
            else
                epochs_no_improve += 1
            end
            if epochs_no_improve >= trn.patience
                @info "early stopping" epoch = epoch best_val_loss = best_val_loss
                break
            end
        end

        isfile(best_path) || error(
            "no epoch improved the validation loss (best_val_loss = $best_val_loss); " *
            "no model checkpoint was written to $run_dir.",
        )
        model, best_meta, best_scaler = load_model(best_path)
        save_model(model_path, model; metadata = best_meta, scaler = best_scaler)

        # The per-epoch history as a table, so the training figure can be
        # redrawn — or animated — from the run instead of from a retraining.
        write_csv(joinpath(run_dir, "history.csv"), history)

        # Decision threshold fitted on the calibration block.
        fit_rows = threshold_rows(blocks, trn.threshold_block)
        block_label =
            trn.threshold_block == "validation" ? "validation block" :
            "held-out block (validation and test)"
        @info "training complete; fitting the decision threshold" block = block_label rows =
            fit_rows
        X_fit =
            fit_rows == blocks.validation ? X_val :
            encode_features(scaler, X_raw[fit_rows, :])
        y_fit = fit_rows == blocks.validation ? y_val : y_raw[fit_rows]
        probs_fit = predict_all(model, X_fit; threaded = threaded)
        threshold, info = select_threshold(
            y_fit,
            probs_fit;
            criterion = trn.threshold_criterion,
            target_far_per_30d = trn.target_far_per_30d,
            target_fpr = trn.target_fpr,
            step_size = geometry.step_size,
            sample_rate = geometry.sample_rate,
        )
        probs_val =
            fit_rows == blocks.validation ? probs_fit :
            predict_all(model, X_val; threaded = threaded)
        m_val, auc_val, metrics_val = block_metrics(probs_val, y_val, threshold, geometry)
        sweep = threshold_sweep(
            y_fit,
            probs_fit;
            step_size = geometry.step_size,
            sample_rate = geometry.sample_rate,
        )
        write_csv(joinpath(run_dir, "threshold_sweep.csv"), sweep)
        info["auc"] = auc_val
        info["value"] = threshold
        info["block"] = trn.threshold_block
        info["fitted_on"] = rootrelative(trn.train_features) * " ($block_label)"
        info["fitted_at"] = string(Dates.now())
        write_toml(
            joinpath(run_dir, "threshold.toml"),
            Dict{String,Any}("threshold" => info),
        )
        @info "threshold" value = threshold criterion = info["criterion"] block =
            trn.threshold_block episodes = info["fit_false_alarm_episodes"] validation_auc =
            auc_val
        n_episodes = info["fit_false_alarm_episodes"]
        n_episodes < trn.min_fit_episodes && @warn "the fitted false-alarm rate rests " *
              "on $n_episodes false-alarm episodes, below min_fit_episodes = " *
              "$(trn.min_fit_episodes); its relative Poisson error is " *
              (
                  n_episodes == 0 ? "unbounded" :
                  "about $(round(Int, 100 / sqrt(n_episodes))) %"
              ) *
              " and the operating point may not transfer to another record." block =
            trn.threshold_block

        # The test block, scored once; part of the calibration set under
        # `threshold_block = "held_out"`, independent of it otherwise.
        probs_test = predict_all(model, X_test; threaded = threaded)
        m_test, auc_test, metrics_test =
            block_metrics(probs_test, y_test, threshold, geometry)
        metrics = Dict{String,Any}("validation" => metrics_val, "test" => metrics_test)
        write_toml(joinpath(run_dir, "metrics.toml"), metrics)
        print_block_summary(
            stdout,
            (("validation", m_val, auc_val), ("test", m_test, auc_test)),
        )
        @info "test block" auc = auc_test precision = m_test.precision recall =
            m_test.recall event_recall = m_test.event_recall false_alarms_per_30d =
            m_test.false_alarms_per_30d
        @info "training artifacts written" run_dir = run_dir

        (
            run_id = String(run_id),
            run_dir = run_dir,
            model_path = model_path,
            threshold = threshold,
            threshold_info = info,
            sweep = sweep,
            metrics = metrics,
            history = history,
            blocks = blocks,
            elapsed = time() - start_time,
        )
    end
end
