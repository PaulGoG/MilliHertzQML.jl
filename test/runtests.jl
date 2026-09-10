# The suite runs in its own environment (test/Project.toml, with the package
# consumed by path through [sources]) and loads MilliHertzQML as a real
# package, so static QA resolves the package identity and `Pkg.test` agrees
# with a direct `julia test/runtests.jl` invocation.
using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using Test
using Statistics, Random, TOML
using CSV, DataFrames
using StableRNGs
using Aqua, JET, ExplicitImports
using MilliHertzQML
using Yao, Flux, Zygote

const PROJECT_ROOT = dirname(@__DIR__)

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
end

@testset "Static QA (JET)" begin
    # Reports are restricted to this package; dependencies are analyzed but
    # not reported against.
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

        opt_state = Flux.setup(Adam(0.1), model.params)
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

    @testset "Data Normalization" begin
        mktempdir() do dir
            feat_path = joinpath(dir, "test_feats.csv")
            lab_path = joinpath(dir, "test_labs.csv")

            df_f = DataFrame(
                PLow = [0.0, 50.0],
                PHigh = [0.0, 50.0],
                Ent = [0.0, 10.0],
                Std = [0.0, 7.0],
            )
            df_l = DataFrame(Label = [0, 1])

            CSV.write(feat_path, df_f)
            CSV.write(lab_path, df_l)

            X, y = load_data(feat_path, lab_path)

            # Scaling to [0, 2π]
            @test all(X .>= 0.0)
            @test all(X .<= 2π + 1e-5)
            @test isapprox(X[2, 1], 2π, atol = 1e-5) # 50.0 maps to 2π

            # The label-free path yields the identical feature matrix
            @test load_features(feat_path) == X
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

        # Out-of-range feature values clamp to the encoding bounds [0, 2π]
        mktempdir() do dir
            feat_path = joinpath(dir, "f.csv")
            lab_path = joinpath(dir, "l.csv")
            CSV.write(
                feat_path,
                DataFrame(
                    PLow = [-5.0, 500.0],
                    PHigh = [-1.0, 100.0],
                    Ent = [-2.0, 50.0],
                    Std = [-3.0, 20.0],
                ),
            )
            CSV.write(lab_path, DataFrame(Label = [0, 1]))
            X, _ = load_data(feat_path, lab_path)
            @test all(X[1, :] .== 0.0f0)
            @test all(isapprox.(X[2, :], Float32(2π); atol = 1e-5))
        end
    end

    @testset "Model Persistence" begin
        mktempdir() do dir
            model = VariationalQuantumClassifier(4, 3; rng = rng)
            x = rand(rng, Float32, 4)
            p_ref = predict_probability(model, x)

            path = joinpath(dir, "model.jld2")
            meta_in = Dict("run_id" => "test", "seed" => 1234)
            save_model(path, model; metadata = meta_in)

            loaded, meta_out = load_model(path)
            @test loaded.n_qubits == model.n_qubits
            @test loaded.n_layers == model.n_layers
            @test loaded.params == model.params
            @test meta_out["run_id"] == "test"
            @test meta_out["seed"] == 1234
            @test isapprox(predict_probability(loaded, x), p_ref; atol = 1e-6)
        end
    end
end

@testset "Pipeline smoke test" begin
    # The four scripts run as child processes on a three-day configuration
    # whose [paths] section points into a temporary directory; nothing is
    # written into the project tree. Each script activates the scripts
    # environment itself.
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    scripts = joinpath(PROJECT_ROOT, "scripts")
    mktempdir() do dir
        cfg = TOML.parsefile(joinpath(PROJECT_ROOT, "config.toml"))
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
            cmd = `$julia --startup-file=no $(joinpath(scripts, script)) --config $cfg_path $(collect(args))`
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

        stage("generate_data.jl", "--run-id", "smoke") || return
        @test isfile(h5)
        @test nrow(CSV.read(raw_labels, DataFrame)) == n_total

        stage("preprocess_ldc.jl", "--h5-file", h5, "--label-file", raw_labels) || return
        @test nrow(CSV.read(feats, DataFrame)) == n_windows
        @test nrow(CSV.read(labs, DataFrame)) == n_windows

        stage("train.jl", "--run-id", "smoke") || return
        model_path = joinpath(dir, "models", "run_smoke", "gw_model.jld2")
        @test isfile(model_path)
        @test isfile(joinpath(dir, "models", "run_smoke", "config.toml"))

        stage("infer.jl", "--run-id", "smoke") || return
        probs = CSV.read(
            joinpath(dir, "results", "run_smoke", "inference_probabilities.csv"),
            DataFrame,
        )
        @test nrow(probs) == n_windows
        @test all(0 .<= probs.Probability .<= 1)
        @test isfile(joinpath(dir, "models", "run_smoke", "threshold.toml"))

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
        @test nrow(blind_probs) == n_windows

        # The project tree received nothing
        @test !isdir(joinpath(PROJECT_ROOT, "models", "run_smoke"))
        @test !isfile(joinpath(PROJECT_ROOT, "data", "inputs", "smoke_features.csv"))
    end
end
