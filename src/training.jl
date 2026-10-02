# src/training.jl — the non-mutating forward pass of the data re-uploading
# circuit (the reference of src/circuit.jl), the class-weighted binary
# cross-entropy, and the batch gradient, serial or over the Julia threads.

"""
    dispatch_params!(model::VariationalQuantumClassifier)

Synchronises the ansatz blocks of `model` with its parameter vector
`model.params`, so that the blocks of a loaded model carry the persisted
values. The forward pass itself reads `model.params` and never mutates
the blocks.
"""
function dispatch_params!(model::VariationalQuantumClassifier)
    idx = 1
    for l in model.ansatz_layers
        np = nparameters(l)
        dispatch!(l, model.params[idx:(idx+np-1)])
        idx += np
    end
end

"""
    predict_probability(model, x) -> Float32

Classifier probability of one feature vector `x` (length `n_qubits`,
angles in ``[0, 2π]``): data re-uploading interleaves, for every layer,
the feature map — ``H`` followed by ``R_z(x_i)`` on qubit ``i`` — with the
ansatz layer, and the probability is ``(1 - \\langle Z \\rangle)/2`` with
``\\langle Z \\rangle`` averaged over the qubits. The register is prepared
from `model.params` alone: the parameter slice of every layer is
dispatched into a rebuilt copy of the ansatz block (Yao's non-mutating
`dispatch` constructs new gate objects), so the function is pure,
differentiable by Zygote with respect to `model.params`, and safe to call
from several threads at once.
"""
function predict_probability(model::VariationalQuantumClassifier, x)
    length(x) == model.n_qubits || throw(
        DimensionMismatch(
            "feature vector has length $(length(x)); expected n_qubits = $(model.n_qubits).",
        ),
    )
    n_qubits = model.n_qubits
    np_layer = div(length(model.params), model.n_layers)
    # ComplexF32 register for type stability with the Float32 parameters
    st = zero_state(ComplexF32, n_qubits)
    for layer_idx in 1:model.n_layers
        for i in 1:n_qubits
            st = apply(st, put(n_qubits, i=>H))
            st = apply(st, put(n_qubits, i=>Rz(Float64(x[i]))))
        end
        p_layer = model.params[((layer_idx-1)*np_layer+1):(layer_idx*np_layer)]
        st = apply(st, dispatch(model.ansatz_layers[layer_idx], p_layer))
    end
    total_z = 0.0f0
    for i in 1:n_qubits
        total_z += real(expect(put(n_qubits, i=>Z), st))
    end
    return (1.0f0 - total_z / Float32(n_qubits)) / 2.0f0
end

"""
    predict(model, x) -> Int

Class decision for a single feature vector `x` at the fixed probability
threshold 0.5: `1` when `predict_probability(model, x) > 0.5`, else `0`.
Run-specific thresholds fitted on the calibration block live in the
training stage, not here.
"""
function predict(model::VariationalQuantumClassifier, x)
    prob = predict_probability(model, x)
    return prob > 0.5f0 ? 1 : 0
end

"""
    accuracy(model, X, y) -> Float64

Fraction of the rows of the feature matrix `X` (samples along the first
dimension) whose `predict` decision equals the corresponding label in `y`.
"""
function accuracy(model::VariationalQuantumClassifier, X, y)
    correct = 0
    for i in 1:size(X, 1)
        if predict(model, @view(X[i, :])) == y[i]
            correct += 1
        end
    end
    return correct / length(y)
end

"""
    weighted_bce(p, y; positive_weight = 1) -> Float32

Binary cross-entropy of the probability `p` against the label `y` (0 or
1), the positive term weighted by `positive_weight`; `p` is clamped to
``[10^{-7}, 1 - 10^{-7}]``.
"""
function weighted_bce(p::Real, y::Real; positive_weight::Real = 1)
    p_c = clamp(Float32(p), 1.0f-7, 1.0f0 - 1.0f-7)
    yf = Float32(y)
    return -(Float32(positive_weight) * yf * log(p_c) + (1.0f0 - yf) * log(1.0f0 - p_c))
end

"""
    sample_loss(model, x, y; positive_weight = 1) -> Float32

Weighted binary cross-entropy ([`weighted_bce`](@ref)) of one feature
vector `x` with label `y`.
"""
function sample_loss(model::VariationalQuantumClassifier, x, y; positive_weight::Real = 1)
    return weighted_bce(predict_probability(model, x), y; positive_weight = positive_weight)
end

"""
    check_batch(model, X_batch, y_batch)

Dimension checks of a batch: `X_batch` holds the samples along its first
dimension and `n_qubits` features along the second, `y_batch` one label
per sample.
"""
function check_batch(model::VariationalQuantumClassifier, X_batch, y_batch)
    size(X_batch, 2) == model.n_qubits || throw(
        DimensionMismatch(
            "feature dimension $(size(X_batch, 2)); expected n_qubits = $(model.n_qubits).",
        ),
    )
    size(X_batch, 1) == length(y_batch) || throw(
        DimensionMismatch("$(size(X_batch, 1)) samples but $(length(y_batch)) labels."),
    )
    return nothing
end

"""
    loss_function(model, X_batch, y_batch; positive_weight = 1) -> Float32

Mean weighted binary cross-entropy over the rows of `X_batch` (samples ×
features), evaluated serially on one automatic-differentiation tape; the
reference of [`batch_gradient`](@ref).
"""
function loss_function(
    model::VariationalQuantumClassifier,
    X_batch,
    y_batch;
    positive_weight::Real = 1,
)
    check_batch(model, X_batch, y_batch)
    l = 0.0f0
    for k in 1:size(X_batch, 1)
        l += sample_loss(
            model,
            @view(X_batch[k, :]),
            y_batch[k];
            positive_weight = positive_weight,
        )
    end
    return l / size(X_batch, 1)
end

"""
    batch_gradient(model, X_batch, y_batch; positive_weight = 1,
                   threaded = Threads.nthreads() > 1, chunk_size = GRADIENT_CHUNK,
                   method = :adjoint, workspaces = nothing)
        -> (loss, gradient)

Mean weighted binary cross-entropy of the batch and its gradient with
respect to `model.params`. The batch is cut into consecutive chunks of
`chunk_size` samples; the chunk sums are stored by chunk index and added
in that order, so the result depends on `chunk_size` but neither on the
thread count nor on the scheduling.

`method = :adjoint` (default): every chunk runs the circuit in place on its
own [`CircuitWorkspace`](@ref) and differentiates by uncomputing it
([`accumulate_gradient!`](@ref)); serial and threaded evaluation give the
same bits. `workspaces`, when given, supplies at least one workspace per
chunk (their parameters are reloaded from `model` on entry); otherwise
they are allocated on the call.

`method = :zygote`: the reference, a Zygote tape over the non-mutating
[`loss_function`](@ref) — one tape over the whole batch when serial
(`threaded = false`, or a batch that fits one chunk), one per chunk on the
Julia threads otherwise, the two differing by the rounding of the
accumulation order. The adjoint gradient agrees with it to single-precision
rounding at a small fraction of the cost.
"""
function batch_gradient(
    model::VariationalQuantumClassifier,
    X_batch,
    y_batch;
    positive_weight::Real = 1,
    threaded::Bool = Threads.nthreads() > 1,
    chunk_size::Integer = GRADIENT_CHUNK,
    method::Symbol = :adjoint,
    workspaces::Union{Nothing,AbstractVector{<:CircuitWorkspace}} = nothing,
)
    check_batch(model, X_batch, y_batch)
    chunk_size >= 1 || throw(ArgumentError("chunk_size = $chunk_size; at least 1."))
    method in (:adjoint, :zygote) ||
        throw(ArgumentError("method = :$method; expected :adjoint or :zygote."))
    n = size(X_batch, 1)
    n >= 1 || throw(ArgumentError("the batch is empty."))
    method == :adjoint && return adjoint_batch_gradient(
        model,
        X_batch,
        y_batch,
        positive_weight,
        threaded,
        Int(chunk_size),
        workspaces,
    )
    if !threaded || n <= chunk_size
        loss_serial, grads_serial = Zygote.withgradient(model) do m
            loss_function(m, X_batch, y_batch; positive_weight = positive_weight)
        end
        return Float32(loss_serial), Vector{Float32}(grads_serial[1].params)
    end
    n_chunks = cld(n, chunk_size)
    # Per-chunk sums (not means), so that the reduction is exact in the
    # chunk sizes; divided by the batch size once at the end.
    losses = Vector{Float32}(undef, n_chunks)
    gradients = Matrix{Float32}(undef, length(model.params), n_chunks)
    Threads.@threads for c in 1:n_chunks
        # Every name assigned in the body is local to the iteration: the
        # body runs as a closure on several threads, and a name that also
        # exists in the enclosing scope would be shared between them.
        local rows = ((c-1)*chunk_size+1):min(n, c*chunk_size)
        local X_chunk = @view(X_batch[rows, :])
        local y_chunk = @view(y_batch[rows])
        local val, grads = Zygote.withgradient(model) do m
            length(rows) * loss_function(m, X_chunk, y_chunk; positive_weight = positive_weight)
        end
        losses[c] = val
        gradients[:, c] = grads[1].params
    end
    return sum(losses) / Float32(n), vec(sum(gradients; dims = 2)) ./ Float32(n)
end

# Chunk sums of the loss and of the adjoint gradient, one workspace per chunk
function adjoint_batch_gradient(
    model,
    X_batch,
    y_batch,
    positive_weight,
    threaded,
    chunk_size,
    workspaces,
)
    n = size(X_batch, 1)
    n_chunks = cld(n, chunk_size)
    pool =
        workspaces === nothing ? [CircuitWorkspace(model) for _ in 1:n_chunks] : workspaces
    length(pool) >= n_chunks || throw(
        ArgumentError("$(length(pool)) workspaces for $n_chunks chunks of the batch."),
    )
    losses = zeros(Float32, n_chunks)
    gradients = zeros(Float32, length(model.params), n_chunks)
    chunk_sum!(c) = begin
        workspace = load_parameters!(pool[c], model.params)
        gradient = @view(gradients[:, c])
        for k in ((c-1)*chunk_size+1):min(n, c*chunk_size)
            losses[c] += accumulate_gradient!(
                gradient,
                workspace,
                @view(X_batch[k, :]),
                y_batch[k];
                positive_weight = positive_weight,
            )
        end
    end
    if threaded && n_chunks > 1
        Threads.@threads for c in 1:n_chunks
            chunk_sum!(c)
        end
    else
        foreach(chunk_sum!, 1:n_chunks)
    end
    return sum(losses) / Float32(n), vec(sum(gradients; dims = 2)) ./ Float32(n)
end

"""
    train_step!(model, opt_state, X_batch, y_batch; positive_weight = 1,
                threaded = Threads.nthreads() > 1, method = :adjoint,
                workspaces = nothing) -> loss

One optimisation step: the batch gradient ([`batch_gradient`](@ref), with
its `method` and `workspaces`) applied to `model.params` in place through
the Flux optimiser state `opt_state`. Returns the batch loss before the
update.
"""
function train_step!(
    model::VariationalQuantumClassifier,
    opt_state,
    X_batch,
    y_batch;
    positive_weight::Real = 1,
    threaded::Bool = Threads.nthreads() > 1,
    method::Symbol = :adjoint,
    workspaces::Union{Nothing,AbstractVector{<:CircuitWorkspace}} = nothing,
)
    val, gradient = batch_gradient(
        model,
        X_batch,
        y_batch;
        positive_weight = positive_weight,
        threaded = threaded,
        method = method,
        workspaces = workspaces,
    )
    Flux.update!(opt_state, model.params, gradient)
    return val
end
