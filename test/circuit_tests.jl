# The circuit evaluated in place and its adjoint gradient, against the
# non-mutating forward pass, the Zygote tape, and central differences of a
# double-precision copy of the circuit written out gate by gate.

"""
Mean weighted cross-entropy of the classifier in double precision, from the
gate sequence itself: per layer ``H`` and ``R_z(x_i)`` on every qubit, then
``R_y`` and ``R_z`` rotations (parameters in that order) and the CNOT ring.
"""
function reference_loss(θ::AbstractVector{Float64}, features, labels, weight, n, n_layers)
    per_layer = length(θ) ÷ n_layers
    total = 0.0
    for k in axes(features, 1)
        reg = zero_state(ComplexF64, n)
        for layer in 1:n_layers
            offset = (layer - 1) * per_layer
            for i in 1:n
                apply!(reg, put(n, i => H))
                apply!(reg, put(n, i => Rz(Float64(features[k, i]))))
            end
            for i in 1:n
                apply!(reg, put(n, i => Ry(θ[offset+i])))
            end
            for i in 1:n
                apply!(reg, put(n, i => Rz(θ[offset+n+i])))
            end
            for i in 1:n
                apply!(reg, control(n, i, (i % n) + 1 => Yao.X))
            end
        end
        z = sum(real(expect(put(n, i => Z), reg)) for i in 1:n) / n
        p = clamp((1 - z) / 2, 1e-7, 1 - 1e-7)
        total -= weight * labels[k] * log(p) + (1 - labels[k]) * log(1 - p)
    end
    return total / size(features, 1)
end

@testset "In-place circuit and adjoint gradient" begin
    rng = StableRNG(2026)

    @testset "Forward pass" begin
        for (n, n_layers) in ((2, 1), (4, 3), (8, 4))
            model = VariationalQuantumClassifier(n, n_layers; rng = rng)
            features = Float32(π) .* rand(rng, Float32, 12, n)
            workspace = CircuitWorkspace(model)
            reference = [predict_probability(model, features[k, :]) for k in 1:12]
            # Bit for bit, and again on the used workspace
            @test [predict_probability!(workspace, @view(features[k, :])) for k in 1:12] == reference
            @test predict_probability!(workspace, features[1, :]) == reference[1]
            @test MilliHertzQML.predict_all(model, features; threaded = false) == reference
            @test MilliHertzQML.predict_all(model, features; threaded = true) == reference
            # New parameters reach the circuit through load_parameters!
            model.params .= 0.5f0 .* randn(rng, Float32, length(model.params))
            load_parameters!(workspace, model.params)
            @test predict_probability!(workspace, features[1, :]) ==
                  predict_probability(model, features[1, :])
        end
        model = VariationalQuantumClassifier(4, 2; rng = rng)
        workspace = CircuitWorkspace(model)
        @test @inferred(predict_probability!(workspace, zeros(Float32, 4))) isa Float32
        @test_throws DimensionMismatch predict_probability!(workspace, zeros(Float32, 3))
        @test_throws DimensionMismatch load_parameters!(workspace, zeros(Float32, 3))
        @test MilliHertzQML.predict_all(model, zeros(Float32, 0, 4)) == Float32[]
    end

    @testset "Mean-Z observable" begin
        for n in (2, 5)
            operator = Matrix(mat(ComplexF32, sum(put(n, i => Z) for i in 1:n) / n))
            @test MilliHertzQML.mean_z_diagonal(n) ≈ [real(operator[k, k]) for k in 1:(2^n)]
        end
    end

    @testset "Loss derivative" begin
        for y in (0, 1), p in (0.03f0, 0.5f0, 0.97f0), weight in (1, 7.5)
            tape = Zygote.gradient(q -> weighted_bce(q, y; positive_weight = weight), p)[1]
            @test MilliHertzQML.weighted_bce_derivative(p, y; positive_weight = weight) ≈
                  tape rtol = 1e-6
        end
        # Outside the clamp the loss is flat
        @test MilliHertzQML.weighted_bce_derivative(0.0f0, 1) == 0
        @test MilliHertzQML.weighted_bce_derivative(1.0f0, 0; positive_weight = 4) == 0
    end

    @testset "Batch gradient" begin
        for (n, n_layers) in ((2, 1), (4, 2), (6, 3))
            model = VariationalQuantumClassifier(n, n_layers; rng = rng)
            features = Float32(π) .* rand(rng, Float32, 22, n)
            labels = rand(rng, 0:1, 22)
            weight = 3.5
            gradient(; kwargs...) = batch_gradient(
                model,
                features,
                labels;
                positive_weight = weight,
                threaded = false,
                kwargs...,
            )
            loss_adjoint, g_adjoint = gradient()
            loss_tape, g_tape = gradient(; method = :zygote)
            @test length(g_adjoint) == length(model.params)
            @test loss_adjoint ≈ loss_tape rtol = 1e-6
            @test isapprox(g_adjoint, g_tape; rtol = 5e-5)
            g_differences = FiniteDiff.finite_difference_gradient(
                θ -> reference_loss(θ, features, labels, weight, n, n_layers),
                Float64.(model.params),
                Val(:central),
            )
            @test loss_adjoint ≈ reference_loss(
                Float64.(model.params),
                features,
                labels,
                weight,
                n,
                n_layers,
            ) rtol = 1e-5
            @test isapprox(g_adjoint, g_differences; rtol = 5e-5)
            # Threads and a caller's workspaces change no bit
            @test gradient(; threaded = true) == (loss_adjoint, g_adjoint)
            pool = [CircuitWorkspace(model) for _ in 1:6]
            @test gradient(; workspaces = pool, threaded = true) ==
                  (loss_adjoint, g_adjoint)
            @test_throws ArgumentError gradient(; workspaces = pool[1:5])
            # The chunk size changes the accumulation order only
            @test isapprox(gradient(; chunk_size = 1)[2], g_adjoint; rtol = 1e-5)
            @test isapprox(gradient(; chunk_size = 22)[2], g_adjoint; rtol = 1e-5)
            # The workspaces follow the parameters of the model
            model.params .+= 0.1f0
            @test gradient(; workspaces = pool) == gradient()
            @test gradient()[2] != g_adjoint
            @test_throws ArgumentError gradient(; method = :forward)
        end
    end

    @testset "Training step" begin
        features = rand(rng, Float32, 24, 4)
        labels = rand(rng, 0:1, 24)
        m_adjoint = VariationalQuantumClassifier(4, 2; rng = StableRNG(5))
        m_tape = VariationalQuantumClassifier(4, 2; rng = StableRNG(5))
        # The last-layer R_z rotations commute with the measurement: their
        # gradient is rounding noise, which Adam turns into an arbitrary step
        g_ref =
            batch_gradient(m_tape, features, labels; method = :zygote, threaded = false)[2]
        live = abs.(g_ref) .> 1e-5
        @test count(live) >= 8
        train_step!(m_adjoint, Flux.setup(Adam(0.1), m_adjoint.params), features, labels)
        train_step!(
            m_tape,
            Flux.setup(Adam(0.1), m_tape.params),
            features,
            labels;
            method = :zygote,
            threaded = false,
        )
        @test isapprox(
            m_adjoint.params[live],
            m_tape.params[live];
            rtol = 1e-4,
            atol = 1e-6,
        )

        pool = [CircuitWorkspace(m_adjoint) for _ in 1:6]
        opt_state = Flux.setup(Adam(0.1), m_adjoint.params)
        loss_initial = loss_function(m_adjoint, features, labels)
        for _ in 1:20
            train_step!(m_adjoint, opt_state, features, labels; workspaces = pool)
        end
        @test loss_function(m_adjoint, features, labels) < loss_initial
    end

    @testset "Settings and memory estimate" begin
        training(table) = training_settings(Dict{String,Any}("training" => table))
        @test training(Dict{String,Any}()).gradient_method == "adjoint"
        @test training(Dict{String,Any}("gradient_method" => "zygote")).gradient_method ==
              "zygote"
        @test_throws ArgumentError training(
            Dict{String,Any}("gradient_method" => "forward"),
        )
        # Adjoint: three statevectors per chunk of four samples, whatever the depth
        @test training_memory_estimate_gib(4, 4, 64) == 3 * 16 * 2^4 * 8 / 2^30
        @test training_memory_estimate_gib(5, 9, 64) ==
              2 * training_memory_estimate_gib(4, 4, 64)
        # Tape: one more qubit doubles the statevector and adds a fifth of the gates
        tape(n, layers, batch) =
            training_memory_estimate_gib(n, layers, batch; gradient_method = "zygote")
        @test tape(5, 4, 64) == 2.5 * tape(4, 4, 64)
        @test tape(4, 4, 128) == 2 * tape(4, 4, 64)
        @test tape(8, 4, 64) > 100 * training_memory_estimate_gib(8, 4, 64)
        @test_throws ArgumentError training_memory_estimate_gib(
            4,
            4,
            64;
            gradient_method = "forward",
        )
    end
end
