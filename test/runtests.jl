using Test
using MilliHertzQML
using Yao
using Flux
using Statistics
using Random
using Zygote
using DataFrames
using CSV

Random.seed!(1234)

@testset "MilliHertzQML Tests (Multi-Qubit VQC)" begin
    # 1. Initialization
    @testset "Model Initialization" begin
        n_qubits = 4
        n_layers = 2
        model = VariationalQuantumClassifier(n_qubits, n_layers)
        @test length(model.params) == n_layers * 8 # 2 params per qubit (Ry, Rz) per layer
        @test model.n_qubits == 4
        @test length(model.ansatz_layers) == n_layers
    end

    # 2. Forward Pass & Probability Logic
    @testset "Forward Pass Logic" begin
        model = VariationalQuantumClassifier(4, 2)
        x = rand(Float32, 4)
        p = predict_probability(model, x)

        @test 0.0 <= p <= 1.0
    end

    # 3. Training & Gradient Tracking
    @testset "Training & Gradients" begin
        model = VariationalQuantumClassifier(4, 2)
        X_batch = rand(Float32, 4, 4)
        y_batch = [0, 1, 0, 1]

        opt_state = Flux.setup(Adam(0.1), model.params)
        l_init = loss_function(model, X_batch, y_batch)

        # Test that gradient is not zero
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

    # 4. Feature Extraction (Physical Logic)
    @testset "Feature Extraction" begin
        # Pure 2mHz Sine Wave (MilliHertz Regime)
        fs = 0.2
        t = range(0, 5000, length = 1000) # 5000 seconds
        signal = sin.(2π * 0.002 * t) # 2 mHz sine wave

        p_low, p_high, ent, std_psd = extract_features(signal, fs)

        @test p_low > p_high # 2mHz is in low band (1-5mHz)
        @test ent > 0.0
        @test std_psd isa Float32
    end

    # 5. Data Loading & Normalization
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

            # Check scaling to [0, 2π]
            @test all(X .>= 0.0)
            @test all(X .<= 2π + 1e-5)
            @test isapprox(X[2, 1], 2π, atol = 1e-5) # 50.0 should map to 2π

            # Label-free path must produce the identical feature matrix
            @test load_features(feat_path) == X
        end
    end

    # 6. Input Validation (fail-fast interfaces)
    @testset "Input Validation" begin
        @test_throws ArgumentError VariationalQuantumClassifier(1, 2)
        @test_throws ArgumentError VariationalQuantumClassifier(4, 0)

        model = VariationalQuantumClassifier(4, 2)
        @test_throws DimensionMismatch predict_probability(model, rand(Float32, 3))
        @test_throws DimensionMismatch loss_function(
            model,
            rand(Float32, 4, 3),
            [0, 1, 0, 1],
        )
        @test_throws DimensionMismatch loss_function(model, rand(Float32, 4, 4), [0, 1])

        # 16 samples at 0.2 Hz resolve no bins inside the 1-5 mHz band
        @test_throws ArgumentError extract_features(randn(16), 0.2)
    end

    # 7. Feature Extraction Edge Cases
    @testset "Feature Edge Cases" begin
        # Constant (zero) signal must yield finite features, no NaN/Inf
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

    # 8. Model Persistence (JLD2 round trip)
    @testset "Model Persistence" begin
        mktempdir() do dir
            model = VariationalQuantumClassifier(4, 3)
            x = rand(Float32, 4)
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
