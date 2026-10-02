# src/circuit.jl — the circuit of the classifier built once and evaluated in
# place: the forward pass on a preallocated register, and the gradient of the
# sample loss by Yao's reversible differentiation, which uncomputes the
# circuit gate by gate instead of recording a tape.

"""
Samples per chunk of a batch gradient; every chunk is one task and, under
the adjoint method, owns one [`CircuitWorkspace`](@ref).
"""
const GRADIENT_CHUNK = 4

"""
    encoding_block(n_qubits) -> ChainBlock

Feature map of one re-uploading layer: ``H`` followed by ``R_z(x_i)`` on
every qubit ``i``, the angles held in double precision as in
[`predict_probability`](@ref).
"""
encoding_block(n_qubits::Integer) = chain(
    n_qubits,
    reduce(
        vcat,
        [[put(n_qubits, i => H), put(n_qubits, i => Rz(0.0))] for i in 1:n_qubits],
    ),
)

"""
    encoding_inverse_block(n_qubits) -> ChainBlock

Inverse of [`encoding_block`](@ref): ``R_z(-x_i)`` followed by ``H``, the
qubits in descending order, so that its parameters are the negated angles
in reverse.
"""
encoding_inverse_block(n_qubits::Integer) = chain(
    n_qubits,
    reduce(
        vcat,
        [[put(n_qubits, i => Rz(0.0)), put(n_qubits, i => H)] for i in n_qubits:-1:1],
    ),
)

"""
    CircuitWorkspace(model)

The circuit of `model` built once, with the registers it is evaluated on:
the encoding block and its inverse, one ansatz block per layer carrying the
parameters of `model` ([`load_parameters!`](@ref) after they change), the
state register ``|\\psi\\rangle`` and the adjoint register
``|\\lambda\\rangle`` of the reverse pass.

[`predict_probability!`](@ref) and [`accumulate_gradient!`](@ref) mutate
the workspace, so every task needs its own; [`predict_all`](@ref) and
[`batch_gradient`](@ref) allocate one per task. The gates and their order
are those of [`predict_probability`](@ref), whose scores the workspace
reproduces bit for bit.
"""
struct CircuitWorkspace{R<:AbstractArrayReg,C<:ChainBlock,O<:AbstractBlock}
    n_qubits::Int
    ψ::R
    λ::R
    encoding::C
    encoding_inverse::C
    layers::Vector{C}
    observables::Vector{O}
    mean_z::Vector{Float32}
    angles::Vector{Float64}
    derivatives::Vector{Float32}
end

function CircuitWorkspace(model::VariationalQuantumClassifier)
    n = model.n_qubits
    ψ = zero_state(ComplexF32, n)
    workspace = CircuitWorkspace(
        n,
        ψ,
        copy(ψ),
        encoding_block(n),
        encoding_inverse_block(n),
        [build_layer(n) for _ in 1:model.n_layers],
        [put(n, i => Z) for i in 1:n],
        mean_z_diagonal(n),
        zeros(Float64, n),
        sizehint!(Float32[], length(model.params)),
    )
    return load_parameters!(workspace, model.params)
end

"""
    mean_z_diagonal(n_qubits) -> Vector{Float32}

Diagonal of the observable ``\\bar Z = \\frac{1}{n} \\sum_i Z_i`` in the
computational basis: ``1 - 2\\,\\mathrm{popcount}(k)/n`` on basis state ``k``.
"""
mean_z_diagonal(n_qubits::Integer) =
    Float32[(n_qubits - 2 * count_ones(k)) / n_qubits for k in 0:(2^n_qubits-1)]

"""
    load_parameters!(workspace, params) -> workspace

Dispatches the parameter vector of the classifier, one slice per layer,
into the ansatz blocks of `workspace`.
"""
function load_parameters!(workspace::CircuitWorkspace, params::AbstractVector{<:Real})
    n_layers = length(workspace.layers)
    per_layer = nparameters(first(workspace.layers))
    length(params) == n_layers * per_layer || throw(
        DimensionMismatch(
            "$(length(params)) parameters; the circuit has $n_layers layers of $per_layer.",
        ),
    )
    for (layer_idx, block) in enumerate(workspace.layers)
        dispatch!(block, @view(params[((layer_idx-1)*per_layer+1):(layer_idx*per_layer)]))
    end
    return workspace
end

"""
    prepare_state!(workspace, x) -> register

Runs the circuit on ``|0\\rangle^{\\otimes n}`` for the feature vector `x`
and returns the state register of `workspace`.
"""
function prepare_state!(workspace::CircuitWorkspace, x)
    n = workspace.n_qubits
    length(x) == n || throw(
        DimensionMismatch(
            "feature vector has length $(length(x)); expected n_qubits = $n.",
        ),
    )
    for i in 1:n
        workspace.angles[i] = Float64(x[i])
    end
    dispatch!(workspace.encoding, workspace.angles)
    amplitudes = state(workspace.ψ)
    fill!(amplitudes, 0)
    amplitudes[1] = 1
    for block in workspace.layers
        apply!(workspace.ψ, workspace.encoding)
        apply!(workspace.ψ, block)
    end
    return workspace.ψ
end

"""
    measured_probability(workspace) -> Float32

``(1 - \\langle \\bar Z \\rangle)/2`` of the state register, accumulated
qubit by qubit as in [`predict_probability`](@ref).
"""
function measured_probability(workspace::CircuitWorkspace)
    total_z = 0.0f0
    for observable in workspace.observables
        total_z += real(expect(observable, workspace.ψ))
    end
    return (1.0f0 - total_z / Float32(workspace.n_qubits)) / 2.0f0
end

"""
    predict_probability!(workspace, x) -> Float32

Classifier probability of the feature vector `x` on the circuit of
`workspace`, evaluated in place; equal bit for bit to
[`predict_probability`](@ref) of the model whose parameters the workspace
carries.
"""
function predict_probability!(workspace::CircuitWorkspace, x)
    prepare_state!(workspace, x)
    return measured_probability(workspace)
end

"""
    weighted_bce_derivative(p, y; positive_weight = 1) -> Float32

Derivative of [`weighted_bce`](@ref) with respect to `p`; zero where `p`
lies outside the clamp ``[10^{-7}, 1 - 10^{-7}]``.
"""
function weighted_bce_derivative(p::Real, y::Real; positive_weight::Real = 1)
    p_c = Float32(p)
    (1.0f-7 <= p_c <= 1.0f0 - 1.0f-7) || return 0.0f0
    yf = Float32(y)
    return -(Float32(positive_weight) * yf / p_c - (1.0f0 - yf) / (1.0f0 - p_c))
end

"""
    accumulate_gradient!(gradient, workspace, x, y; positive_weight = 1) -> Float32

Adds to `gradient` the gradient of the weighted cross-entropy of the sample
`(x, y)` with respect to the ansatz parameters, and returns the sample
loss ([`sample_loss`](@ref)).

Adjoint differentiation (Jones & Gacon 2020, arXiv:2009.02823; Yao's
reversible mode, Luo et al. 2020, doi:10.22331/q-2020-10-11-341): after
the forward pass the adjoint register is seeded with
``|\\lambda\\rangle = \\bar Z |\\psi\\rangle``; the circuit is then undone
gate by gate on both registers, and each rotation ``e^{-i\\theta G/2}``
contributes ``\\partial_\\theta \\langle \\bar Z \\rangle =
-\\mathrm{Im}\\,\\langle G \\psi | \\lambda \\rangle`` at the point where it
is undone. The cost is three passes over the circuit and two registers,
whatever the number of parameters. The encoding angles are not
differentiated. The chain rule through ``p = (1 - \\langle \\bar Z
\\rangle)/2`` and the loss is applied to the result
([`weighted_bce_derivative`](@ref)).
"""
function accumulate_gradient!(
    gradient::AbstractVector{Float32},
    workspace::CircuitWorkspace,
    x,
    y::Real;
    positive_weight::Real = 1,
)
    n = workspace.n_qubits
    prepare_state!(workspace, x)
    p = measured_probability(workspace)
    loss = weighted_bce(p, y; positive_weight = positive_weight)
    dloss_dp = weighted_bce_derivative(p, y; positive_weight = positive_weight)
    state(workspace.λ) .= workspace.mean_z .* state(workspace.ψ)
    for i in 1:n
        workspace.angles[i] = -Float64(x[n+1-i])
    end
    dispatch!(workspace.encoding_inverse, workspace.angles)
    empty!(workspace.derivatives)
    registers = (workspace.ψ, workspace.λ)
    for layer in Iterators.reverse(workspace.layers)
        for block in Iterators.reverse(subblocks(layer))
            if nparameters(block) == 0
                inverse = block'
                apply!(workspace.ψ, inverse)
                apply!(workspace.λ, inverse)
            else
                # prepends -Im⟨Gψ|λ⟩/2 of the rotation, then undoes it
                apply_back!(registers, block, workspace.derivatives)
            end
        end
        apply!(workspace.ψ, workspace.encoding_inverse)
        apply!(workspace.λ, workspace.encoding_inverse)
    end
    # ∂ℓ/∂θ = ℓ′(p) · (-1/2) · ∂⟨Z̄⟩/∂θ, with ∂⟨Z̄⟩/∂θ twice the collected value
    gradient .-= dloss_dp .* workspace.derivatives
    return loss
end
