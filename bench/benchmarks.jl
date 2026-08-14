using Pkg
Pkg.activate(dirname(@__DIR__); io = devnull)
Pkg.instantiate(; io = devnull)
using BenchmarkTools
using MilliHertzQML
using Yao
using Flux
using Zygote
using Statistics
using Random

Random.seed!(42)

# Configuration
n_qubits = 4
n_layers = 4
model = VariationalQuantumClassifier(n_qubits, n_layers)
opt_state = Flux.setup(Adam(0.01), model.params)

# Batch data (10 samples, 4 features)
X_batch = rand(Float32, 10, 4)
y_batch = rand(0:1, 10)

println("\n--- Quantum Classifier (4 Features, 4 Qubits, 4 Layers) Benchmarks ---")

println("\n1. Forward Pass (Single Sample):")
@btime predict_probability($model, $(X_batch[1, :]))

println("\n2. Loss Evaluation (Batch size 10):")
@btime loss_function($model, $X_batch, $y_batch)

println("\n3. Gradient Calculation (Batch size 10):")
@btime Zygote.gradient($model) do m
    loss_function(m, $X_batch, $y_batch)
end

println("\n4. Full Training Step (Batch size 10):")
@btime train_step!($model, $opt_state, $X_batch, $y_batch)

println("\n--- EXPANDED SCIENTIFIC BENCHMARKS ---")

println("\n5. Feature Extraction (1024 sample time-series):")
sample_data = randn(Float32, 1024)
@btime extract_features($sample_data)

println("\n6. Model Scaling (Forward Pass):")
for q in [2, 4, 6]
    for l in [2, 4]
        m_scale = VariationalQuantumClassifier(q, l)
        x_scale = rand(Float32, q)
        t = @belapsed predict_probability($m_scale, $x_scale)
        println("  Qubits: $q, Layers: $l | Time: $(round(t*1e6, digits=2)) μs")
    end
end

println("\n7. Batch Throughput (Gradients):")
for bs in [16, 32, 64]
    X_bs = rand(Float32, bs, 4)
    y_bs = rand(0:1, bs)
    t = @belapsed Zygote.gradient(m -> loss_function(m, $X_bs, $y_bs), $model)
    println("  Batch Size: $bs | Samples/sec: $(round(bs/t, digits=2))")
end