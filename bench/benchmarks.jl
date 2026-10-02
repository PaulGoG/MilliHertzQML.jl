# Benchmarks of the stages that cost time in training and scoring, on the
# selected circuit (8 qubits, 4 re-uploading layers, mini-batches of 64)
# with synthetic inputs: forward pass, batch gradient by both methods, a
# training step, the scoring of a block of windows, the features of one
# window, and the cost against the register size.
#
#   julia -t auto bench/benchmarks.jl
include(joinpath(@__DIR__, "activate.jl"))
using BenchmarkTools
using MilliHertzQML
using Optimisers: Optimisers
using Random: Xoshiro
using LinearAlgebra: BLAS
using Printf: @printf

BLAS.set_num_threads(1)

const N_QUBITS, N_LAYERS, BATCH = 8, 4, 64

function inputs(rng, n_qubits, n_layers, n_samples)
    model = VariationalQuantumClassifier(n_qubits, n_layers; rng = rng)
    X = Float32(π) .* rand(rng, Float32, n_samples, n_qubits)
    y = rand(rng, 0:1, n_samples)
    return model, X, y
end

function run_benchmarks()
    rng = Xoshiro(2026)
    model, X, y = inputs(rng, N_QUBITS, N_LAYERS, BATCH)
    workspace = CircuitWorkspace(model)
    x = X[1, :]
    threads = Threads.nthreads()
    @printf(
        "Circuit: %d qubits, %d layers, %d parameters; batch of %d; %d Julia threads\n",
        N_QUBITS,
        N_LAYERS,
        length(model.params),
        BATCH,
        threads
    )

    println("\n1. Forward pass, one sample")
    t_reference = @belapsed predict_probability($model, $x)
    t_inplace = @belapsed predict_probability!($workspace, $x)
    @printf("   non-mutating reference  %8.1f µs\n", 1e6 * t_reference)
    @printf("   in place                %8.1f µs\n", 1e6 * t_inplace)

    println("\n2. Batch gradient, per sample")
    pool = [CircuitWorkspace(model) for _ in 1:gradient_tasks(BATCH; threaded = true)]
    t_adjoint =
        @belapsed batch_gradient($model, $X, $y; threaded = false, workspaces = $pool)
    t_threads =
        @belapsed batch_gradient($model, $X, $y; threaded = true, workspaces = $pool)
    t_tape =
        @belapsed batch_gradient($model, $X, $y; threaded = false, method = :zygote) samples =
            3 evals = 1
    t_tape_threads =
        @belapsed batch_gradient($model, $X, $y; threaded = true, method = :zygote) samples =
            3 evals = 1
    bytes_adjoint =
        @allocated batch_gradient(model, X, y; threaded = false, workspaces = pool)
    bytes_tape = @allocated batch_gradient(model, X, y; threaded = false, method = :zygote)
    @printf(
        "   adjoint, serial         %8.3f ms  (%.1f forward passes, %.2f MiB per batch)\n",
        1e3 * t_adjoint / BATCH,
        t_adjoint / BATCH / t_inplace,
        bytes_adjoint / 2^20
    )
    @printf(
        "   adjoint, %2d threads     %8.3f ms  (speed-up %.1f)\n",
        threads,
        1e3 * t_threads / BATCH,
        t_adjoint / t_threads
    )
    @printf(
        "   Zygote tape, serial     %8.3f ms  (%.1f MiB per batch)\n",
        1e3 * t_tape / BATCH,
        bytes_tape / 2^20
    )
    @printf(
        "   Zygote tape, %2d threads %8.3f ms  (adjoint faster by %.0f)\n",
        threads,
        1e3 * t_tape_threads / BATCH,
        t_tape_threads / t_threads
    )

    println("\n3. Training step (batch gradient and Adam update)")
    opt_state = Optimisers.setup(Optimisers.Adam(0.01), model.params)
    t_step = @belapsed train_step!($model, $opt_state, $X, $y; workspaces = $pool)
    @printf(
        "   %8.2f ms per step; %.1f s per 688 steps (the training block of a Sangria year)\n",
        1e3 * t_step,
        688 * t_step
    )

    println("\n4. Scoring a block of 4,096 windows")
    block = Float32(π) .* rand(rng, Float32, 4096, N_QUBITS)
    t_serial =
        @belapsed MilliHertzQML.predict_all($model, $block; threaded = false) samples = 3
    t_parallel =
        @belapsed MilliHertzQML.predict_all($model, $block; threaded = true) samples = 3
    @printf(
        "   serial %8.0f windows/s | %d threads %8.0f windows/s\n",
        4096 / t_serial,
        threads,
        4096 / t_parallel
    )

    println("\n5. Features of one window of 1,000 samples (six bands)")
    window = randn(rng, 1000)
    edges = [3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 1e-2, 4e-2]
    t_features =
        @belapsed extract_features($window, 0.2; feature_set = :bands, band_edges = $edges)
    @printf("   %8.1f µs\n", 1e6 * t_features)

    println(
        "\n6. Register size, $N_LAYERS layers: forward pass and adjoint gradient per sample",
    )
    for n in (4, 6, 8, 10, 12, 14)
        m, Xn, yn = inputs(rng, n, N_LAYERS, 8)
        ws = CircuitWorkspace(m)
        xn = Xn[1, :]
        t_f = @belapsed predict_probability!($ws, $xn)
        t_g = @belapsed batch_gradient($m, $Xn, $yn; threaded = false, workspaces = $([ws]))
        @printf("   %2d qubits  %9.1f µs  %9.3f ms\n", n, 1e6 * t_f, 1e3 * t_g / 8)
    end
    return nothing
end

run_benchmarks()
