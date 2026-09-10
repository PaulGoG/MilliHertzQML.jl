# src/model.jl

"""
    VariationalQuantumClassifier(n_qubits, n_layers, params, ansatz_layers)

A Variational Quantum Classifier (VQC) designed for Gravitational Wave detection.
Implements Data Re-uploading by interleaving feature maps and variational layers.
"""
mutable struct VariationalQuantumClassifier
    n_qubits::Int
    n_layers::Int
    params::Vector{Float32}
    ansatz_layers::Vector{AbstractBlock}
end

"""
    build_layer(n_qubits)

Constructs a single layer of a Hardware Efficient Ansatz using Ry and Rz rotations
followed by a ring of CNOT gates for entanglement.
"""
function build_layer(n_qubits)
    return chain(
        n_qubits,
        [put(i => Ry(0.0f0)) for i in 1:n_qubits]...,
        [put(i => Rz(0.0f0)) for i in 1:n_qubits]...,
        [control(i, (i % n_qubits) + 1 => X) for i in 1:n_qubits]...,
    )
end

"""
    VariationalQuantumClassifier(n_qubits=4, n_layers=2)

Constructor for the VQC. Initializes parameters from a zero-mean normal
distribution with standard deviation 0.5.
"""
function VariationalQuantumClassifier(n_qubits::Int = 4, n_layers::Int = 2)
    n_qubits >= 2 || throw(
        ArgumentError(
            "n_qubits = $n_qubits; at least 2 qubits are required for the entangling CNOT ring.",
        ),
    )
    n_layers >= 1 ||
        throw(ArgumentError("n_layers = $n_layers; at least 1 layer is required."))
    layers = [build_layer(n_qubits) for _ in 1:n_layers]

    n_params_per_layer = nparameters(layers[1])
    total_params = n_params_per_layer * n_layers

    # Initialize with Float32 for type stability
    p = randn(Float32, total_params) * 0.5f0

    idx = 1
    for l in layers
        np = nparameters(l)
        dispatch!(l, p[idx:(idx+np-1)])
        idx += np
    end

    return VariationalQuantumClassifier(n_qubits, n_layers, p, layers)
end

Functors.@functor VariationalQuantumClassifier (params,)

"""
    build_step(model, x, layer_idx)

Creates a single 'Data Re-uploading' block: [Feature Map] -> [Ansatz Layer].
The Feature Map uses a 1st order Pauli-Z expansion with Hadamard pre-rotation.
"""
function build_step(model::VariationalQuantumClassifier, x, layer_idx::Int)
    fm = chain(
        model.n_qubits,
        [put(i => H) for i in 1:model.n_qubits]...,
        [put(i => Rz(Float64(x[i]))) for i in 1:model.n_qubits]..., # Yao gates often internally prefer Float64 for angles
    )
    return chain(model.n_qubits, fm, model.ansatz_layers[layer_idx])
end
