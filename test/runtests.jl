# The suite runs in its own environment (test/Project.toml, with the package
# consumed by path through [sources]) and loads MilliHertzQML as a real
# package, so static QA resolves the package identity and `Pkg.test` agrees
# with a direct `julia test/runtests.jl` invocation.
include(joinpath(@__DIR__, "activate.jl"))

using Test
using Statistics, Random, TOML, Dates
using CSV, DataFrames
using StableRNGs
using Aqua, JET, ExplicitImports
using MilliHertzQML
using Yao, Optimisers, Zygote
using FiniteDiff: FiniteDiff
using CairoMakie: CairoMakie
using DeepSpaceTelemetry: DeepSpaceTelemetry

const PROJECT_ROOT = dirname(@__DIR__)
# The pipeline root of the whole run, child processes included: a sandboxed
# test environment (Pkg.test) lies outside the repository
ENV["STREAMINGINFERENCE_ROOT"] = PROJECT_ROOT

@testset "Static QA (Aqua)" begin
    # Scripts, benchmarks, and tests carry their own environments, so the
    # package dependency graph is exactly what src/ loads and the stale-deps
    # check runs unexempted. The persistent-tasks check precompiles a wrapper
    # package against the live registry; it is gated off CI, where it fails
    # for environmental reasons (see the sibling package's record), and runs
    # locally.
    Aqua.test_all(MilliHertzQML; persistent_tasks = get(ENV, "CI", "") != "true")
end

@testset "Static QA (ExplicitImports)" begin
    @test ExplicitImports.check_no_stale_explicit_imports(MilliHertzQML) === nothing
    @test ExplicitImports.check_no_implicit_imports(MilliHertzQML) === nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(MilliHertzQML) === nothing
    @test ExplicitImports.check_all_explicit_imports_are_public(MilliHertzQML) === nothing
    @test ExplicitImports.check_all_qualified_accesses_via_owners(MilliHertzQML) === nothing
    # `Optimisers.adjust!` is documented by Optimisers.jl, which declares no
    # public names beyond its exports
    @test ExplicitImports.check_all_qualified_accesses_are_public(
        MilliHertzQML;
        ignore = (:adjust!,),
    ) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(MilliHertzQML) === nothing
end

@testset "Static QA (JET)" begin
    # Reports are restricted to this package; its dependencies, the two
    # layers included, are analysed but not reported against.
    JET.test_package(MilliHertzQML; target_modules = (MilliHertzQML,))
end

@testset "MilliHertzQML Tests (Multi-Qubit VQC)" begin
    rng = StableRNG(1234)

    @testset "Model Initialization" begin
        n_qubits = 4
        n_layers = 2
        model = VariationalQuantumClassifier(n_qubits, n_layers; rng = rng)
        @test length(model.params) == n_layers * 8 # 2 params per qubit (Ry, Rz) per layer
        @test model.n_qubits == 4
        @test length(model.ansatz_layers) == n_layers
        # Seeded construction is reproducible
        @test VariationalQuantumClassifier(4, 2; rng = StableRNG(7)).params ==
              VariationalQuantumClassifier(4, 2; rng = StableRNG(7)).params
    end

    @testset "Forward Pass Logic" begin
        model = VariationalQuantumClassifier(4, 2; rng = rng)
        x = rand(rng, Float32, 4)
        p = predict_probability(model, x)
        @test 0.0 <= p <= 1.0
    end

    @testset "Training & Gradients" begin
        model = VariationalQuantumClassifier(4, 2; rng = rng)
        X_batch = rand(rng, Float32, 4, 4)
        y_batch = [0, 1, 0, 1]

        opt_state = Optimisers.setup(Optimisers.Adam(0.1), model.params)
        l_init = loss_function(model, X_batch, y_batch)

        grads = Zygote.gradient(model) do m
            loss_function(m, X_batch, y_batch)
        end
        @test any(abs.(grads[1].params) .> 0.0)

        for _ in 1:20
            train_step!(model, opt_state, X_batch, y_batch)
        end

        l_final = loss_function(model, X_batch, y_batch)
        @test l_final < l_init

        # The batch loss is the mean of the sample losses; the weighted
        # cross-entropy is the plain one at unit weight
        @test weighted_bce(0.5, 1) ≈ log(2) atol = 1e-6
        @test weighted_bce(0.5, 0) ≈ log(2) atol = 1e-6
        @test weighted_bce(0.5, 1; positive_weight = 3) ≈ 3 * log(2) atol = 1e-6
        @test weighted_bce(1.0, 1) < 1e-6 && weighted_bce(0.0, 1) > 10
        @test sum(sample_loss(model, X_batch[k, :], y_batch[k]) for k in 1:4) / 4 ≈
              loss_function(model, X_batch, y_batch)

        # The Zygote reference, threaded: same loss, the gradient of the
        # serial tape up to accumulation rounding, deterministic across calls
        X_big = rand(rng, Float32, 24, 4)
        y_big = rand(rng, 0:1, 24)
        tape(m, threaded) = batch_gradient(
            m,
            X_big,
            y_big;
            positive_weight = 2.0,
            threaded = threaded,
            method = :zygote,
        )
        l_serial, g_serial = tape(model, false)
        l_threads, g_threads = tape(model, true)
        @test l_serial ≈ loss_function(model, X_big, y_big; positive_weight = 2.0)
        @test l_threads ≈ l_serial rtol = 1e-5
        @test length(g_threads) == length(model.params)
        @test isapprox(g_threads, g_serial; rtol = 1e-4, atol = 1e-6)
        @test g_threads == tape(model, true)[2]
        @test g_serial == Zygote.gradient(
            m -> loss_function(m, X_big, y_big; positive_weight = 2.0),
            model,
        )[1].params
        @test_throws ArgumentError batch_gradient(model, zeros(Float32, 0, 4), Int[])
        # One training step from the same state along either path lands on
        # the same parameters to floating-point tolerance. The R_z rotations
        # of the last layer commute with the Z measurement (pulled back
        # through the CNOT ring, every measured Z_k is a product of Z's), so
        # their gradient vanishes and Adam's normalisation turns rounding
        # noise into an arbitrary step: they are pinned at zero gradient and
        # excluded from the comparison.
        m_serial = VariationalQuantumClassifier(4, 2; rng = StableRNG(5))
        m_threads = VariationalQuantumClassifier(4, 2; rng = StableRNG(5))
        g_ref =
            batch_gradient(m_serial, X_big, y_big; threaded = false, method = :zygote)[2]
        @test all(abs.(g_ref[(end-3):end]) .< 1e-6)
        @test count(abs.(g_ref) .> 1e-5) >= 8
        live = abs.(g_ref) .> 1e-5
        train_step!(
            m_serial,
            Optimisers.setup(Optimisers.Adam(0.1), m_serial.params),
            X_big,
            y_big;
            threaded = false,
            method = :zygote,
        )
        train_step!(
            m_threads,
            Optimisers.setup(Optimisers.Adam(0.1), m_threads.params),
            X_big,
            y_big;
            threaded = true,
            method = :zygote,
        )
        @test isapprox(
            m_serial.params[live],
            m_threads.params[live];
            rtol = 1e-4,
            atol = 1e-6,
        )

        # Threaded forward passes reproduce the serial ones exactly
        p_serial = MilliHertzQML.predict_all(model, X_big; threaded = false)
        p_threads = MilliHertzQML.predict_all(model, X_big; threaded = true)
        @test p_threads == p_serial
        @test p_serial == [predict_probability(model, X_big[k, :]) for k in 1:24]
        loss_val, acc_val = MilliHertzQML.epoch_validation(
            model,
            X_big,
            y_big;
            positive_weight = 2.0,
            threaded = true,
        )
        @test loss_val ≈ loss_function(model, X_big, y_big; positive_weight = 2.0) rtol =
            1e-5
        @test acc_val ≈ accuracy(model, X_big, y_big)
        @test_throws DimensionMismatch MilliHertzQML.epoch_validation(
            model,
            X_big,
            y_big[1:5];
            positive_weight = 1.0,
        )
    end

    @testset "Feature Extraction" begin
        # Pure 2 mHz sine wave (milliHertz regime)
        fs = 0.2
        t = range(0, 5000, length = 1000)
        signal = sin.(2π * 0.002 * t)

        p_low, p_high, ent, std_psd = extract_features(signal, fs)

        @test p_low > p_high # 2 mHz lies in the low band (1-5 mHz)
        @test ent > 0.0
        @test std_psd isa Float32
    end

    @testset "Data Loading" begin
        mktempdir() do dir
            feat_path = joinpath(dir, "test_feats.csv")
            lab_path = joinpath(dir, "test_labs.csv")
            df_f = DataFrame(
                p_low = [0.5, 3.0, 1.0],
                p_high = [1.0, 0.9, 1.1],
                spectral_entropy = [0.95, 0.6, 0.9],
                log_power_std = [0.0, 0.8, 0.1],
            )
            CSV.write(feat_path, df_f)
            CSV.write(lab_path, DataFrame(Label = [0, 1, 0], SNR = [0.0, 12.0, 0.0]))

            X, y, df_l = load_data(feat_path, lab_path)
            @test X == Matrix{Float32}(df_f)
            @test y == [0, 1, 0]
            @test df_l.SNR == [0.0, 12.0, 0.0]
            # The label-free path yields the identical raw matrix
            @test load_features(feat_path) == X

            CSV.write(lab_path, DataFrame(Label = [0, 1]))
            @test_throws DimensionMismatch load_data(feat_path, lab_path)
        end
    end

    @testset "Input Validation" begin
        @test_throws ArgumentError VariationalQuantumClassifier(1, 2)
        @test_throws ArgumentError VariationalQuantumClassifier(4, 0)

        model = VariationalQuantumClassifier(4, 2; rng = rng)
        @test_throws DimensionMismatch predict_probability(model, rand(rng, Float32, 3))
        @test_throws DimensionMismatch loss_function(
            model,
            rand(rng, Float32, 4, 3),
            [0, 1, 0, 1],
        )
        @test_throws DimensionMismatch loss_function(
            model,
            rand(rng, Float32, 4, 4),
            [0, 1],
        )

        # 16 samples at 0.2 Hz resolve no bins inside the 1-5 mHz band
        @test_throws ArgumentError extract_features(randn(rng, 16), 0.2)
    end

    @testset "Feature Edge Cases" begin
        # A constant (zero) signal yields finite features
        feats = extract_features(zeros(Float64, 1000), 0.2)
        @test all(isfinite, feats)
    end

    @testset "Model Persistence" begin
        mktempdir() do dir
            model = VariationalQuantumClassifier(4, 3; rng = rng)
            x = rand(rng, Float32, 4)
            p_ref = predict_probability(model, x)
            scaler = FeatureScaler([0.0, 0.0, 0.0, -1.0], [3.0, 3.0, 1.0, 1.0])

            path = joinpath(dir, "model.jld2")
            meta_in = Dict("run_id" => "test", "seed" => 1234)
            save_model(path, model; metadata = meta_in, scaler = scaler)

            loaded, meta_out, scaler_out = load_model(path)
            @test loaded.n_qubits == model.n_qubits
            @test loaded.n_layers == model.n_layers
            @test loaded.params == model.params
            @test meta_out["run_id"] == "test"
            @test meta_out["seed"] == 1234
            @test scaler_out.lower == scaler.lower && scaler_out.upper == scaler.upper
            @test scaler_out.phase_span == scaler.phase_span == Float32(π)
            @test isapprox(predict_probability(loaded, x), p_ref; atol = 1e-6)
            # The span is persisted with the bounds
            wide = FeatureScaler(scaler.lower, scaler.upper; phase_span = 2π)
            save_model(joinpath(dir, "wide.jld2"), model; scaler = wide)
            @test load_model(joinpath(dir, "wide.jld2"))[3].phase_span == Float32(2π)
            # An artifact written before the span was persisted restores the
            # full period its scaler was trained with, and says so
            legacy = joinpath(dir, "legacy.jld2")
            MilliHertzQML.JLD2.jldsave(
                legacy;
                n_qubits = model.n_qubits,
                n_layers = model.n_layers,
                params = model.params,
                scaler_lower = scaler.lower,
                scaler_upper = scaler.upper,
                metadata = Dict{String,Any}("run_id" => "old"),
            )
            _, meta_legacy, scaler_legacy =
                @test_logs (:warn, r"predates the persisted phase-encoding span") load_model(
                    legacy,
                )
            @test meta_legacy["run_id"] == "old"
            @test scaler_legacy.phase_span == Float32(2π)

            # Artifacts without a scaler load with `nothing`
            bare = joinpath(dir, "bare.jld2")
            save_model(bare, model)
            @test load_model(bare)[3] === nothing
        end
    end
end

include("circuit_tests.jl")

@testset "Feature scaler" begin
    rng = StableRNG(3)
    X = randn(rng, 500, 4) .* [1.0 10.0 0.1 100.0] .+ [0.0 5.0 0.5 -50.0]
    scaler = fit_scaler(X; quantiles = (0.01, 0.99))
    @test scaler.phase_span == Float32(π)
    E = encode_features(scaler, X)
    @test size(E) == size(X)
    @test eltype(E) == Float32
    @test all(0 .<= E .<= Float32(π))
    @test minimum(E) == 0.0f0 && isapprox(maximum(E), π; atol = 1e-5)
    # The bounds map to the interval ends; outside values clamp
    @test encode_features(scaler, reshape(scaler.lower, 1, :)) == zeros(Float32, 1, 4)
    @test all(
        isapprox.(encode_features(scaler, reshape(scaler.upper, 1, :)), π; atol = 1e-5),
    )
    @test all(encode_features(scaler, fill(1e9, 1, 4)) .≈ Float32(π))
    # The full period is admissible for artifacts trained on it, and folds
    # the two clamp ends onto one state: R_z(2π) = −I is a global phase, so
    # a saturated feature scores exactly as one at the floor
    legacy = fit_scaler(X; quantiles = (0.01, 0.99), phase_span = 2π)
    @test legacy.phase_span == Float32(2π)
    @test isapprox(maximum(encode_features(legacy, X)), 2π; atol = 1e-5)
    model = VariationalQuantumClassifier(4, 2; rng = rng)
    x_floor = fill(0.0f0, 4)
    x_wide = vec(encode_features(legacy, fill(1e9, 1, 4)))
    x_half = vec(encode_features(scaler, fill(1e9, 1, 4)))
    @test isapprox(
        predict_probability(model, x_wide),
        predict_probability(model, x_floor);
        atol = 1e-5,
    )
    @test abs(predict_probability(model, x_half) - predict_probability(model, x_floor)) >
          1e-3
    @test_throws ArgumentError FeatureScaler([0.0], [1.0]; phase_span = 3π)
    @test_throws ArgumentError FeatureScaler([0.0], [1.0]; phase_span = 0.0)
    @test_throws ArgumentError fit_scaler(hcat(X, ones(500)))
    @test_throws ArgumentError fit_scaler(X; quantiles = (0.9, 0.1))
    @test_throws DimensionMismatch encode_features(scaler, X[:, 1:3])
    @test_throws ArgumentError FeatureScaler([0.0, 1.0], [1.0, 1.0])
end

@testset "Threshold file migration" begin
    info = Dict{String,Any}(
        "value" => 0.5,
        "validation_recall" => 1.0,
        "validation_windows" => 10,
        "fit_fpr" => 0.0,
        "validation_fpr" => 0.1,
    )
    @test_logs (:info, r"predates the configurable fitting block") MilliHertzQML.migrate_threshold_info!(
        info,
    )
    @test info["fit_recall"] == 1.0 && info["fit_windows"] == 10
    @test !haskey(info, "validation_recall") && !haskey(info, "validation_windows")
    # A key already present under its current name is kept as it is
    @test info["fit_fpr"] == 0.0 && info["validation_fpr"] == 0.1
    @test_logs MilliHertzQML.migrate_threshold_info!(Dict{String,Any}("value" => 0.5))
    mktempdir() do dir
        open(joinpath(dir, "threshold.toml"), "w") do io
            TOML.print(
                io,
                Dict(
                    "threshold" =>
                        Dict("value" => 0.42, "validation_false_alarms_per_30d" => 1.5),
                ),
            )
        end
        thr, old = MilliHertzQML.load_threshold(dir)
        @test thr == 0.42f0 && old["fit_false_alarms_per_30d"] == 1.5
    end
end

@testset "Class-weighted loss" begin
    # Weight 1 is the plain BCE; a larger weight raises the loss of every
    # imperfectly scored positive
    rng = StableRNG(11)
    model = VariationalQuantumClassifier(4, 2; rng = rng)
    Xw = rand(rng, Float32, 6, 4) .* Float32(2π)
    yw = [1, 0, 1, 0, 0, 0]
    @test loss_function(model, Xw, yw; positive_weight = 1) == loss_function(model, Xw, yw)
    @test loss_function(model, Xw, yw; positive_weight = 3) > loss_function(model, Xw, yw)
    opt_state = Optimisers.setup(Optimisers.Adam(0.01), model.params)
    @test isfinite(train_step!(model, opt_state, Xw, yw; positive_weight = 2.0))
end

@testset "Configuration and provenance (classifier)" begin
    # A configuration of the repository has the repository as its pipeline root
    @test config_root(joinpath(PROJECT_ROOT, "configs", "experiments", "q8_b6.toml")) ==
          PROJECT_ROOT
    # An empty configuration yields the documented, validated defaults
    empty = Dict{String,Any}()
    @test model_settings(empty).n_qubits == 4
    @test training_settings(empty).threshold_criterion == "far"
    @test training_settings(empty).threshold_block == "validation"
    @test training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("threshold_block" => "held_out")),
    ).threshold_block == "held_out"
    @test_throws ArgumentError training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("threshold_block" => "test")),
    )
    @test training_settings(empty).phase_span == 1.0
    @test training_settings(empty).min_fit_episodes == 5
    @test training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("phase_span" => 2.0)),
    ).phase_span == 2.0
    @test_throws ArgumentError training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("phase_span" => 2.5)),
    )
    @test_throws ArgumentError training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("min_fit_episodes" => -1)),
    )
    @test_throws ArgumentError training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("train_fraction" => 0.9)),
    )
    # The provenance of a run records the version of this package, the
    # pipeline at the root
    g = git_provenance()
    @test haskey(g, "git_commit") &&
          g["package_version"] == string(pkgversion(MilliHertzQML))
end

@testset "Figures (CairoMakie extension)" begin
    @test Base.get_extension(MilliHertzQML, :MilliHertzQMLCairoMakieExt) !== nothing
    history = (
        epochs = collect(1:14),
        train_loss = 1.0 .- 0.03 .* (1:14),
        val_loss = 1.05 .- 0.02 .* (1:14),
        val_acc = 0.6 .+ 0.02 .* (1:14),
    )
    @test figure_training_history(history) isa CairoMakie.Figure
    @test_throws ArgumentError figure_training_history((
        epochs = Int[],
        train_loss = Float32[],
        val_loss = Float32[],
        val_acc = Float32[],
    ))
    # The sidecar records the canvas actually exported: two stacked panels
    # for the seed spread, the wide canvas for the studies
    mktempdir() do dir
        save_figure(
            figure_seed_spread([1, 2], [0.5, 0.6], [1.0, 2.0], [2, 2], 2),
            joinpath(dir, "wide"),
        )
        @test TOML.parsefile(joinpath(dir, "wide.toml"))["figure"]["size_pt"] == [900, 950]
        gap = figure_gap_study(
            ["Reference", "Loss 0.3 %", "Outage 24 h"],
            ["reference", "loss", "outage"],
            [1.0, 0.44, 0.99],
            [2, 2, 2],
            2,
            [18.9, 104.2, NaN];
            reference_far = 18.9,
        )
        save_figure(gap, joinpath(dir, "gap"))
        @test TOML.parsefile(joinpath(dir, "gap.toml"))["figure"]["size_pt"] == [1200, 600]
        @test_throws DimensionMismatch figure_gap_study(
            ["a"],
            ["f"],
            [1.0, 0.5],
            [2],
            2,
            [1.0],
        )
        @test_throws ArgumentError figure_gap_study(["a"], ["f"], [1.0], [3], 2, [1.0])
        # Grid over seeds: a missing run (NaN) is left out, a run that
        # missed a blind event is drawn open, and the rates of a positive
        # grid go on a logarithmic axis.
        grid = figure_grid_seeds(
            ["q8", "q6", "q4"],
            [4 10 NaN; 8 7 9; 9 10 12],
            [1.57 2.06 NaN; 5.5 9.8 4.0; 2.6 1.98 3.3],
            [5 5 0; 5 5 5; 5 4 5],
            5;
            seeds = [9999, 1009, 2027],
            selected = "q8",
            target_far = 3.0,
        )
        save_figure(grid, joinpath(dir, "grid"))
        @test TOML.parsefile(joinpath(dir, "grid.toml"))["figure"]["size_pt"] == [1200, 600]
        @test_throws DimensionMismatch figure_grid_seeds(
            ["a"],
            ones(1, 2),
            ones(1, 3),
            ones(Int, 1, 2),
            5,
        )
        @test_throws ArgumentError figure_grid_seeds(
            ["a"],
            ones(1, 1),
            ones(1, 1),
            fill(6, 1, 1),
            5,
        )
        @test_throws ArgumentError figure_grid_seeds(
            ["a"],
            ones(1, 1),
            ones(1, 1),
            ones(Int, 1, 1),
            5;
            selected = "b",
        )
    end
end

@testset "Animations (CairoMakie extension)" begin
    history = (
        epochs = collect(1:5),
        train_loss = [1.0, 0.8, 0.7, 0.65, 0.62],
        val_loss = [1.1, 0.9, 0.85, 0.88, 0.87],
        val_acc = [0.5, 0.6, 0.68, 0.66, 0.7],
    )
    mktempdir() do dir
        path = animate_training_history(
            history,
            joinpath(dir, "training_history.gif");
            framerate = 4,
            hold_frames = 0,
        )
        @test isfile(path) && filesize(path) > 1000
        # An animation is written as GIF and needs two epochs at least
        @test_throws ArgumentError animate_training_history(
            history,
            joinpath(dir, "training_history.mp4"),
        )
        @test_throws ArgumentError animate_training_history(
            (epochs = [1], train_loss = [1.0], val_loss = [1.0], val_acc = [0.5]),
            joinpath(dir, "replay.gif"),
        )
    end
end

@testset "Channel modes (classifier)" begin
    # A feature table of the A and E network, trained on and scored only
    # under a configuration and a model of the same mode
    @test MilliHertzQML.run_channels(joinpath(tempdir(), "absent_run")) == "A"
    mktempdir() do dir
        rng = StableRNG(17)
        n = 600
        labels = zeros(Int, n)
        labels[250:300] .= 1
        table = DataFrame(rand(rng, Float32, n, 4), feature_names(:whitened))
        table.p_low .+= 3.0f0 .* labels
        function product(stem, channels)
            path = joinpath(dir, stem * "_features.csv")
            CSV.write(path, table)
            CSV.write(joinpath(dir, stem * "_labels.csv"), DataFrame(Label = labels))
            sidecar = Dict{String,Any}(
                "features" => Dict{String,Any}(
                    "window_size" => 1000,
                    "step_size" => 100,
                    "sample_rate" => 0.2,
                    "psd" => "none",
                    "feature_set" => "whitened",
                ),
            )
            if channels !== nothing
                sidecar["product"] = Dict{String,Any}("channels" => channels)
                sidecar["features"]["channels"] = channels
                channels == "A" || (sidecar["features"]["channel_combination"] = "max")
            end
            open(io -> TOML.print(io, sidecar), joinpath(dir, stem * "_features.toml"), "w")
            return path
        end
        pair = product("pair", "AE")
        single = product("single", "A")
        legacy = product("legacy", nothing)
        config(features, mode) = Dict{String,Any}(
            "paths" => Dict{String,Any}(
                "inputs" => dir,
                "models" => joinpath(dir, "models"),
                "plots" => joinpath(dir, "plots"),
                "results" => joinpath(dir, "results"),
            ),
            "tdi" => Dict{String,Any}("channels" => mode),
            "model" => Dict{String,Any}("n_qubits" => 4, "n_layers" => 1),
            "training" => Dict{String,Any}(
                "train_features" => features,
                "train_labels" => replace(features, "_features" => "_labels"),
                "epochs" => 1,
                "batch_size" => 32,
                "threshold_criterion" => "youden",
            ),
        )
        # The product decides: a configuration of another mode is refused
        @test_throws ArgumentError train_classifier(config(pair, "A"); run_id = "refused")
        @test_throws ArgumentError train_classifier(
            config(single, "AE");
            run_id = "refused",
        )
        # A table that records no channel set is one of the mode A
        @test_throws ArgumentError train_classifier(
            config(legacy, "AE");
            run_id = "refused",
        )
        run = train_classifier(config(pair, "AE"); run_id = "pair")
        @test MilliHertzQML.run_channels(run.run_dir) == "AE"
        @test TOML.parsefile(joinpath(run.run_dir, "config.toml"))["features"]["channels"] ==
              "AE"
        # Scoring: features of the mode the model was trained on, or none
        scored = evaluate_classifier(
            config(pair, "AE");
            run_id = "pair",
            features = pair,
            labels = replace(pair, "_features" => "_labels"),
        )
        @test length(scored.probabilities) == n
        @test_throws ArgumentError evaluate_classifier(
            config(pair, "AE");
            run_id = "pair",
            features = single,
        )
        @test_throws ArgumentError evaluate_classifier(
            config(pair, "AE");
            run_id = "pair",
            features = legacy,
        )

        # The detector of the run: the combination of its features, no
        # whitening for this product, and a replay of two delivered channels
        detector = detector_from_run(run.model_path; context_windows = 2)
        @test detector.scorer.features.combination == :max
        @test detector.psd === nothing
        @test_throws ArgumentError detector_from_run(
            run.model_path;
            psd_sidecar = replace(single, r"\.csv$" => ".toml"),
        )
        fs = 0.2
        geometry = RunGeometry(fs, 50.0, 10, Dates.DateTime(2035, 1, 1))
        payload = randn(rng, Float32, 3000, 2)
        arrivals = [
            ArrivalEvent(
                geometry.start_sim_time + Dates.Minute(10 * k),
                "LIVE_batch_$k",
                :ingested,
                1,
            ) for k in 1:30
        ]
        windows = replay_run(MemoryTelemetryRun(geometry, payload, arrivals), detector)
        @test nrow(windows) == 21 && all(0 .<= windows.score .<= 1)
        # A window against a direct evaluation on its conditioning stretch
        w = windows[11, :]
        lo, hi = max(1, w.row_start - 2000), min(3000, w.row_end + 2000)
        stretch = payload[lo:hi, :]
        offset = w.row_start - lo + 1
        @test w.score == score_window(detector, stretch, offset)
        conditioned = condition_window(detector, stretch, offset)
        @test size(conditioned) == (1000, 2)
        features = extract_features(conditioned, fs; combination = :max)
        model, _, scaler = load_model(run.model_path)
        @test w.score == predict_probability(
            model,
            vec(encode_features(scaler, reshape(collect(Float32.(features)), 1, :))),
        )
        # The delivery of a single-channel run applied to the two-channel record
        mission = MemoryTelemetryRun(geometry, payload[:, 1], arrivals)
        @test replay_run(ScheduledRecordRun(mission, payload), detector).score ==
              windows.score
    end
    @test tdi_settings(Dict{String,Any}()).channels == "A"
    @test channel_suffix("AE") == "_ae"
    events = DataFrame(signal_start_index = [5], signal_start_index_ae = [3])
    @test mode_events(events, "AE").signal_start_index == [3]
end

@testset "Classical controls" begin
    σ(z) = 1 / (1 + exp(-z))
    @test MilliHertzQML.control_parameter_count(8, 0) == 9
    @test MilliHertzQML.control_parameter_count(8, 6) == 61
    @test MilliHertzQML.control_parameter_count(8, 9) == 91
    @test_throws ArgumentError ClassicalControl(0, 0)
    @test_throws ArgumentError ClassicalControl(2, -1)
    @test_throws ArgumentError ClassicalControl(2, 0; span = 0)
    @test_throws DimensionMismatch ClassicalControl(2, 0, π, zeros(Float32, 4))
    @test ClassicalControl(8, 6; rng = StableRNG(1)) isa AbstractClassifier
    @test VariationalQuantumClassifier(2, 1; rng = StableRNG(1)) isa AbstractClassifier

    # The models against their formulas in matrix form, on features in [0, π]
    x = Float32[π/2, π/4]
    u = Float64.(x) ./ π
    logistic = ClassicalControl(2, 0, π, Float32[1, -2, 0.5])
    @test predict_probability(logistic, x) ≈ σ(1 * u[1] - 2 * u[2] + 0.5) atol = 1e-6
    @test predict_probability(logistic, x) ≈ 0.6224593312018546 atol = 1e-6
    W₁ = [1.0 2.0; 3.0 4.0]
    b₁ = [0.1, -0.2]
    w₂ = [0.5, -1.5]
    network = ClassicalControl(2, 2, π, Float32.(vcat(vec(W₁), b₁, w₂, 0.3)))
    @test predict_probability(network, x) ≈ σ(w₂' * tanh.(W₁ * u .+ b₁) + 0.3) atol = 1e-6
    # The span scales the input: twice the span, half the argument
    wide = ClassicalControl(2, 0, 2π, Float32[1, -2, 0.5])
    @test predict_probability(wide, x) ≈ σ(0.5 * (u[1] - 2 * u[2]) + 0.5) atol = 1e-6
    @test_throws DimensionMismatch predict_probability(logistic, Float32[0.1, 0.2, 0.3])

    rng = StableRNG(2026)
    X = Float32.(π .* rand(rng, 24, 3))
    y = Int.(rand(rng, 24) .< 0.4)
    for hidden in (0, 4)
        model = ClassicalControl(3, hidden; rng = rng)
        p = MilliHertzQML.predict_all(model, X)
        @test p isa Vector{Float32} && length(p) == 24
        @test p == [predict_probability(model, X[i, :]) for i in 1:24]
        @test MilliHertzQML.predict_all(model, X; threaded = true) == p
        @test_throws DimensionMismatch MilliHertzQML.predict_all(model, X[:, 1:2])

        # Mean weighted cross-entropy and its gradient against central
        # differences of the loss in double precision
        function mean_loss(θ)
            trial = ClassicalControl(3, hidden, π, Float32.(θ))
            q = Float64.(MilliHertzQML.predict_all(trial, X))
            return -sum(2.5 .* y .* log.(q) .+ (1 .- y) .* log.(1 .- q)) / 24
        end
        loss, gradient = batch_gradient(model, X, y; positive_weight = 2.5)
        @test loss isa Float32 && gradient isa Vector{Float32}
        @test length(gradient) == length(model.params)
        @test loss ≈ mean_loss(model.params) rtol = 1e-5
        reference = FiniteDiff.finite_difference_gradient(
            mean_loss,
            Float64.(model.params),
            Val(:central);
            absstep = 1e-2,
        )
        @test gradient ≈ reference rtol = 2e-2 atol = 2e-3
        # The keywords of the circuit's step are accepted
        @test batch_gradient(model, X, y; positive_weight = 2.5, threaded = true)[2] ==
              gradient
        @test_throws DimensionMismatch batch_gradient(model, X, y[1:10])

        # A step returns the loss before the update and lowers it afterwards
        state = Optimisers.setup(Optimisers.Adam(0.05), model.params)
        before = train_step!(model, state, X, y; positive_weight = 2.5)
        @test before == loss
        for _ in 1:50
            train_step!(model, state, X, y; positive_weight = 2.5)
        end
        @test batch_gradient(model, X, y; positive_weight = 2.5)[1] < before
    end

    # Persistence: the kind is recorded for a control and absent for a circuit
    mktempdir() do dir
        scaler = FeatureScaler([0.0, 0.0, -1.0], [3.0, 1.0, 1.0])
        control = ClassicalControl(3, 4; rng = StableRNG(3))
        path = save_model(joinpath(dir, "control.jld2"), control; scaler = scaler)
        loaded, meta, loaded_scaler = load_model(path)
        @test loaded isa ClassicalControl
        @test (loaded.n_features, loaded.hidden, loaded.span) == (3, 4, Float32(π))
        @test loaded.params == control.params && meta isa AbstractDict
        @test loaded_scaler.lower == scaler.lower && loaded_scaler.upper == scaler.upper
        circuit = VariationalQuantumClassifier(3, 1; rng = StableRNG(3))
        circuit_path = save_model(joinpath(dir, "circuit.jld2"), circuit; scaler = scaler)
        @test !haskey(MilliHertzQML.JLD2.load(circuit_path), "kind")
        @test load_model(circuit_path)[1] isa VariationalQuantumClassifier
    end

    # The classifier of the [model] section, drawn from the seeded default stream
    scaler = FeatureScaler(zeros(4), ones(4); phase_span = π / 2)
    settings(kind) =
        model_settings(Dict{String,Any}("model" => Dict{String,Any}("kind" => kind)))
    @test model_settings(Dict{String,Any}()).kind == "circuit"
    @test model_settings(Dict{String,Any}()).hidden_units == 6
    @test MODEL_KINDS == ("circuit", "logistic", "perceptron")
    @test_throws ArgumentError settings("forest")
    Random.seed!(11)
    built = build_classifier(settings("circuit"), scaler)
    Random.seed!(11)
    @test built isa VariationalQuantumClassifier &&
          built.params == VariationalQuantumClassifier(4, 4).params
    logistic = build_classifier(settings("logistic"), scaler)
    @test logistic isa ClassicalControl && logistic.hidden == 0
    @test logistic.n_features == 4 && logistic.span == Float32(π / 2)
    perceptron = build_classifier(settings("perceptron"), scaler)
    @test perceptron.hidden == 6 && length(perceptron.params) == 6 * (4 + 2) + 1
    @test VQCScorer === ClassifierScorer{VariationalQuantumClassifier}

    # Through the stages: training, scoring, and the detector of the run
    mktempdir() do dir
        rng = StableRNG(17)
        n = 600
        labels = zeros(Int, n)
        labels[250:300] .= 1
        table = DataFrame(rand(rng, Float32, n, 4), feature_names(:whitened))
        table.p_low .+= 3.0f0 .* labels
        features_path = joinpath(dir, "toy_features.csv")
        CSV.write(features_path, table)
        CSV.write(joinpath(dir, "toy_labels.csv"), DataFrame(Label = labels))
        open(joinpath(dir, "toy_features.toml"), "w") do io
            TOML.print(
                io,
                Dict{String,Any}(
                    "features" => Dict{String,Any}(
                        "window_size" => 1000,
                        "step_size" => 100,
                        "sample_rate" => 0.2,
                        "psd" => "none",
                        "feature_set" => "whitened",
                    ),
                ),
            )
        end
        config(kind) = Dict{String,Any}(
            "paths" => Dict{String,Any}(
                "inputs" => dir,
                "models" => joinpath(dir, "models"),
                "plots" => joinpath(dir, "plots"),
                "results" => joinpath(dir, "results"),
            ),
            "model" => Dict{String,Any}(
                "n_qubits" => 4,
                "n_layers" => 1,
                "kind" => kind,
                "hidden_units" => 3,
            ),
            "training" => Dict{String,Any}(
                "train_features" => features_path,
                "train_labels" => joinpath(dir, "toy_labels.csv"),
                "epochs" => 3,
                "batch_size" => 32,
                "threshold_criterion" => "youden",
            ),
        )
        for (kind, hidden) in (("logistic", 0), ("perceptron", 3))
            run = train_classifier(config(kind); run_id = kind)
            model, _, scaler = load_model(run.model_path)
            @test model isa ClassicalControl && model.hidden == hidden
            @test model.span == scaler.phase_span
            snapshot = TOML.parsefile(joinpath(run.run_dir, "config.toml"))["model"]
            @test snapshot["kind"] == kind
            @test haskey(snapshot, "hidden_units") == (kind == "perceptron")
            @test length(run.history.epochs) >= 1 && isfinite(run.threshold)
            scored = evaluate_classifier(
                config(kind);
                run_id = kind,
                features = features_path,
                labels = joinpath(dir, "toy_labels.csv"),
            )
            @test scored.probabilities == MilliHertzQML.predict_all(
                model,
                encode_features(scaler, Matrix{Float32}(table)),
            )
            # The detector scores a window as the model scores its features
            detector = detector_from_run(run.model_path; context_windows = 1)
            @test detector.scorer isa ClassifierScorer{ClassicalControl}
            window = randn(rng, 1000)
            f = extract_features(window, 0.2)
            @test window_score(detector.scorer, window, 0.2) == predict_probability(
                model,
                vec(encode_features(scaler, reshape(collect(Float32.(f)), 1, :))),
            )
        end
        # The circuit through the same builder: the kind is in its snapshot
        run = train_classifier(config("circuit"); run_id = "circuit")
        @test load_model(run.model_path)[1] isa VariationalQuantumClassifier
        @test TOML.parsefile(joinpath(run.run_dir, "config.toml"))["model"]["kind"] ==
              "circuit"
    end
end

include("telemetry_tests.jl")
include("telemetry_integration_tests.jl")

# Validation anchors on the LDC Sangria training product. They run only when
# MILLIHERTZQML_LDC_DIR names a directory holding LDC2_sangria_training_v2.h5
# (a 3 GB download from Zenodo record 7132178).
const SANGRIA_FILE =
    joinpath(get(ENV, "MILLIHERTZQML_LDC_DIR", ""), "LDC2_sangria_training_v2.h5")
if isfile(SANGRIA_FILE)
    @testset "Sangria anchors" begin
        catalog = read_catalog(SANGRIA_FILE)
        @test nrow(catalog) == 15
        @test isapprox(catalog.CoalescenceTime[5], 11526944.9; atol = 1.0)

        # A reference evaluation quotes an optimal A-channel
        # SNR of 1885.7 for catalogue row 4 (0-based) against the SciRDv1 noise
        # model. Its neighbour (row 3, merging 3.1 d earlier) is excluded by a
        # segment starting 3 d before the merger; the segment ends are tapered.
        truth = read_tdi(SANGRIA_FILE; group = "sky/mbhb/tdi")
        fs = 1 / truth.dt
        A, _, _ = tdi_to_aet(truth.X, truth.Y, truth.Z)
        tc = catalog.CoalescenceTime[5]
        ic = round(Int, (tc - truth.t[1]) * fs) + 1
        seg = A[(ic-round(Int, 3*86400*fs)):(ic+round(Int, 7200*fs))]
        m = 100
        for k in 1:m
            w = 0.5 * (1 - cos(π * (k - 1) / m))
            seg[k] *= w
            seg[end-k+1] *= w
        end
        ρ = matched_filter_snr(
            seg,
            fs;
            psd = f -> ldc_tdi_psd(f; channel = :A, model = "SciRDv1"),
        )
        @test isapprox(ρ, 1885.7; rtol = 0.03)

        # Noise-only null test: the observed record minus every truth stream
        # is instrument noise, whose A-channel PSD follows the "sangria" model
        # to within the simulator's filters; whitened by that model and
        # high-passed, its windows have unit band powers.
        obs = read_tdi(SANGRIA_FILE; group = "obs/tdi")
        residual, _, _ = tdi_to_aet(obs.X, obs.Y, obs.Z)
        residual .-= A
        for source in ("dgb", "igb", "vgb")
            sky = read_tdi(SANGRIA_FILE; group = "sky/$source/tdi")
            residual .-= tdi_to_aet(sky.X, sky.Y, sky.Z)[1]
        end
        psd_sangria = f -> ldc_tdi_psd(f; channel = :A, model = "sangria")
        fw, sw = welch_psd(residual, fs; segment_length = 65536)
        band = (fw .>= 3e-4) .& (fw .<= 5e-2)
        ratio = sw[band] ./ psd_sangria.(fw[band])
        @test 0.8 < median(ratio) < 1.25
        white = whiten_record(
            highpass_record(residual, fs; cutoff = 5e-4, order = 8),
            fs;
            psd = psd_sangria,
        )
        # The equal-arm analytic PSD vanishes at the TDI null f = c/(2L) ≈ 60 mHz,
        # where the data keep a finite floor, so the analytic whitening is only
        # checked below the null; the Welch estimate whitens the full band.
        starts = round.(Int, range(1, length(white) - 1000; length = 400))
        feats = [
            extract_features(view(white, s:(s+999)), fs; high_band = (5e-3, 4e-2)) for
            s in starts
        ]
        @test isapprox(mean(first.(feats)), 1.0; atol = 0.25)
        @test isapprox(mean(getindex.(feats, 2)), 1.0; atol = 0.25)
        white_welch = whiten_record(
            highpass_record(residual, fs; cutoff = 5e-4, order = 8),
            fs;
            psd = interpolated_psd(fw, sw),
        )
        feats_welch = [extract_features(view(white_welch, s:(s+999)), fs) for s in starts]
        @test isapprox(mean(first.(feats_welch)), 1.0; atol = 0.15)
        @test isapprox(mean(getindex.(feats_welch, 2)), 1.0; atol = 0.15)
    end
else
    @info "Sangria anchors skipped: set MILLIHERTZQML_LDC_DIR to the directory holding LDC2_sangria_training_v2.h5"
end

@testset "Pipeline smoke test" begin
    # The four scripts run as child processes on a three-day configuration
    # whose [paths] section points into a temporary directory; nothing is
    # written into the project tree. Each script activates the scripts
    # environment itself.
    # The command of this process, so that the stages run by the scripts
    # inherit its code-coverage setting
    julia = Base.julia_cmd()
    scripts = joinpath(PROJECT_ROOT, "scripts")
    mktempdir() do dir
        cfg = TOML.parsefile(joinpath(PROJECT_ROOT, "configs", "default.toml"))
        inputs = joinpath(dir, "inputs")
        cfg["paths"] = Dict(
            "inputs" => inputs,
            "models" => joinpath(dir, "models"),
            "plots" => joinpath(dir, "plots"),
            "results" => joinpath(dir, "results"),
        )
        h5 = joinpath(inputs, "smoke_telemetry.h5")
        raw_labels = replace(h5, ".h5" => "_labels.csv")
        merge!(
            cfg["generation"],
            Dict(
                "days" => 3.0,
                "n_mbhb" => 2,
                "n_gbs" => 5,
                "n_emris" => 1,
                "output" => h5,
            ),
        )
        cfg["preprocessing"]["output_prefix"] = "smoke"
        cfg["preprocessing"]["edge_margin"] = 1.0
        feats = joinpath(inputs, "smoke_features.csv")
        labs = joinpath(inputs, "smoke_labels.csv")
        merge!(
            cfg["training"],
            Dict(
                "train_features" => feats,
                "train_labels" => labs,
                "epochs" => 2,
                "batch_size" => 32,
                "patience" => 2,
            ),
        )
        merge!(cfg["inference"], Dict("features" => feats, "labels" => labs))
        cfg_path = joinpath(dir, "config.toml")
        open(cfg_path, "w") do io
            TOML.print(io, cfg)
        end
        log = joinpath(dir, "pipeline.log")

        function stage(script, args...)
            cmd = `$julia --startup-file=no $(joinpath(scripts, script)) $cfg_path $(collect(args))`
            ok = success(pipeline(cmd; stdout = log, stderr = log, append = true))
            ok || println(read(log, String))
            @test ok
            return ok
        end

        fs = cfg["generation"]["fs"]
        n_total = round(Int, 3.0 * 24 * 3600 * fs)
        n_windows =
            div(
                n_total - cfg["preprocessing"]["window_size"],
                cfg["preprocessing"]["step_size"],
            ) + 1
        margin = round(
            Int,
            cfg["preprocessing"]["edge_margin"] * cfg["preprocessing"]["window_size"] /
            cfg["preprocessing"]["step_size"],
        )
        n_kept = n_windows - 2 * margin

        stage("generate_data.jl", "--run-id", "smoke") || return
        @test isfile(h5)
        @test nrow(CSV.read(raw_labels, DataFrame)) == n_total
        events = CSV.read(replace(h5, ".h5" => "_events.csv"), DataFrame)
        @test nrow(events) == 2
        @test all(8.0 .<= events.snr .<= 50.0)
        @test all(events.start_index .<= events.merger_index .<= events.end_index)
        @test all(1e5 .<= events.total_mass_msun .<= 1e7)
        @test all(events.label_start_index .>= events.start_index .- 999)
        @test all(events.label_end_index .<= events.end_index .+ 999)
        # Detectable spans start at the signal onset itself
        @test all(
            r ->
                r.label_start_index > r.label_end_index ||
                r.signal_start_index == r.label_start_index,
            eachrow(events),
        )

        stage("preprocess_ldc.jl", "--h5-file", h5, "--label-file", raw_labels) || return
        @test nrow(CSV.read(feats, DataFrame)) == n_kept
        @test nrow(CSV.read(labs, DataFrame)) == n_kept
        sidecar = TOML.parsefile(replace(feats, ".csv" => ".toml"))["features"]
        @test sidecar["window_size"] == cfg["preprocessing"]["window_size"]
        @test sidecar["step_size"] == cfg["preprocessing"]["step_size"]
        @test sidecar["sample_rate"] == fs
        @test sidecar["n_windows"] == n_kept
        @test sidecar["first_window"] == margin + 1 &&
              sidecar["edge_margin_windows"] == margin
        @test sidecar["n_windows_record"] == n_windows && margin == 10
        # The analytic whitening PSD is rebuilt from the sidecar alone
        @test sidecar["psd"] == "model" && haskey(sidecar, "observation_years")
        @test whitening_psd_from_sidecar(replace(feats, ".csv" => ".toml"))(2e-3) ==
              lisa_noise_psd(2e-3; observation_years = sidecar["observation_years"])

        stage("train.jl", "--run-id", "smoke") || return
        run_dir = joinpath(dir, "models", "run_smoke")
        model_path = joinpath(run_dir, "gw_model.jld2")
        @test isfile(model_path)
        @test isfile(joinpath(run_dir, "config.toml"))
        @test isfile(joinpath(run_dir, "manifest_snapshot.toml"))
        # Chronological blocks with a one-window buffer
        blocks = TOML.parsefile(joinpath(run_dir, "split.toml"))["split"]
        buffer = cld(cfg["preprocessing"]["window_size"], cfg["preprocessing"]["step_size"])
        @test blocks["n_windows"] == n_kept
        @test blocks["buffer_windows"] == buffer
        @test blocks["train"] == [1, floor(Int, 0.7 * n_kept)]
        @test blocks["validation"][1] == blocks["train"][2] + buffer + 1
        @test blocks["test"][1] == blocks["validation"][2] + buffer + 1
        @test blocks["test"][2] == n_kept
        n_test = blocks["test"][2] - blocks["test"][1] + 1
        # Threshold fitted on the calibration block; metrics of both blocks
        thr = TOML.parsefile(joinpath(run_dir, "threshold.toml"))["threshold"]
        @test haskey(thr, "value") && haskey(thr, "criterion") && haskey(thr, "auc")
        @test thr["criterion"] in ("far", "fpr")
        @test thr["block"] == "validation"
        @test thr["fit_windows"] == blocks["validation"][2] - blocks["validation"][1] + 1
        @test haskey(thr, "fit_false_alarm_episodes")
        metrics = TOML.parsefile(joinpath(run_dir, "metrics.toml"))
        @test haskey(metrics, "validation") && haskey(metrics, "test")
        @test isfinite(metrics["test"]["false_alarms_per_30d"])
        @test metrics["test"]["n_windows"] == n_test
        @test metrics["validation"]["threshold"] == thr["value"]
        # The operating characteristic of the calibration block and its figure
        sweep = CSV.read(joinpath(run_dir, "threshold_sweep.csv"), DataFrame)
        @test issorted(sweep.threshold) && "false_alarms_per_30d" in names(sweep)
        @test allequal(sweep.n_events)
        @test first(sweep.n_events) >= metrics["validation"]["n_events"]
        @test isfile(joinpath(dir, "plots", "run_smoke", "threshold_sweep.pdf"))

        stage("infer.jl", "--run-id", "smoke") || return
        probs = CSV.read(
            joinpath(dir, "results", "run_smoke", "inference_probabilities.csv"),
            DataFrame,
        )
        @test nrow(probs) == n_kept
        @test all(0 .<= probs.Probability .<= 1)
        @test probs.Window == 1:n_kept
        infer_metrics =
            TOML.parsefile(joinpath(dir, "results", "run_smoke", "metrics.toml"))["metrics"]
        @test infer_metrics["n_windows"] == n_kept
        @test infer_metrics["threshold"] == thr["value"]
        @test haskey(infer_metrics, "fpr")
        @test isfile(joinpath(dir, "results", "run_smoke", "threshold_sweep.csv"))
        @test isfile(joinpath(dir, "plots", "run_smoke", "threshold_sweep.pdf"))

        # The test block alone, through the run's split.toml
        stage(
            "infer.jl",
            "--model",
            model_path,
            "--run-id",
            "smoke_test",
            "--block",
            "test",
        ) || return
        block_probs = CSV.read(
            joinpath(dir, "results", "run_smoke_test", "inference_probabilities.csv"),
            DataFrame,
        )
        @test nrow(block_probs) == n_test
        @test first(block_probs.Window) == blocks["test"][1]
        block_metrics =
            TOML.parsefile(joinpath(dir, "results", "run_smoke_test", "metrics.toml"))["metrics"]
        @test block_metrics["block"] == "test"
        @test block_metrics["n_windows"] == n_test

        stage(
            "infer.jl",
            "--model",
            model_path,
            "--run-id",
            "smoke_blind",
            "--labels",
            "",
        ) || return
        blind =
            TOML.parsefile(joinpath(dir, "results", "run_smoke_blind", "config_infer.toml"))
        @test blind["inference"]["blind"] == true
        blind_probs = CSV.read(
            joinpath(dir, "results", "run_smoke_blind", "inference_probabilities.csv"),
            DataFrame,
        )
        @test nrow(blind_probs) == n_kept
        @test !isfile(joinpath(dir, "results", "run_smoke_blind", "threshold_sweep.csv"))

        # The project tree received nothing
        @test !isdir(joinpath(PROJECT_ROOT, "models", "run_smoke"))
        @test !isfile(joinpath(PROJECT_ROOT, "data", "inputs", "smoke_features.csv"))
    end
end
