# src/training.jl

"""
    dispatch_params!(model::VariationalQuantumClassifier)

Synchronizes the internal ansatz blocks with the global parameter vector `model.params`.
Required for automatic differentiation via Zygote.
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
    predict_probability(model, x)

Performs a forward pass for a single input vector `x`.
Implements Ensemble Measurement by averaging the expectation value of Z
across all qubits to produce the final classification probability.
"""
function predict_probability(model::VariationalQuantumClassifier, x)
    length(x) == model.n_qubits || throw(
        DimensionMismatch(
            "feature vector has length $(length(x)); expected n_qubits = $(model.n_qubits).",
        ),
    )
    dispatch_params!(model)

    steps = [build_step(model, x, i) for i in 1:model.n_layers]
    c = chain(model.n_qubits, steps...)

    # Use ComplexF32 for type stability with Float32 network parameters
    reg = zero_state(ComplexF32, model.n_qubits) |> c

    total_z = 0.0f0
    for i in 1:model.n_qubits
        total_z += real(expect(put(model.n_qubits, i=>Z), reg))
    end
    avg_z = total_z / Float32(model.n_qubits)

    return (1.0f0 - avg_z) / 2.0f0
end

"""
    predict(model, x) -> Int

Class decision for a single feature vector `x` at the fixed probability
threshold 0.5: `1` when `predict_probability(model, x) > 0.5`, else `0`.
Run-specific thresholds selected from the ROC curve live in the inference
script, not here.
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
    loss_function(model, X_batch, y_batch; positive_weight = 1)

Binary cross-entropy loss of a batch, with the positive-class term weighted
by `positive_weight` (the negative-to-positive count ratio balances the
classes).
Supports automatic differentiation by ensuring all stateful circuit updates
are tracked via the `params` vector.
"""
function loss_function(
    model::VariationalQuantumClassifier,
    X_batch,
    y_batch;
    positive_weight::Real = 1,
)
    size(X_batch, 2) == model.n_qubits || throw(
        DimensionMismatch(
            "feature dimension $(size(X_batch, 2)); expected n_qubits = $(model.n_qubits).",
        ),
    )
    size(X_batch, 1) == length(y_batch) || throw(
        DimensionMismatch("$(size(X_batch, 1)) samples but $(length(y_batch)) labels."),
    )
    l = 0.0f0
    n_qubits = model.n_qubits
    n_layers = model.n_layers
    np_layer = div(length(model.params), n_layers)

    for k in 1:size(X_batch, 1)
        x = @view(X_batch[k, :])
        y = y_batch[k]

        # Use ComplexF32 for memory efficiency and type stability
        st = zero_state(ComplexF32, n_qubits)

        for layer_idx in 1:n_layers
            for i in 1:n_qubits
                st = apply(st, put(n_qubits, i=>H))
                st = apply(st, put(n_qubits, i=>Rz(Float64(x[i]))))
            end

            p_layer = model.params[((layer_idx-1)*np_layer+1):(layer_idx*np_layer)]
            ansatz = dispatch(model.ansatz_layers[layer_idx], p_layer)
            st = apply(st, ansatz)
        end

        total_z = 0.0f0
        for i in 1:n_qubits
            total_z += real(expect(put(n_qubits, i=>Z), st))
        end
        avg_z = total_z / Float32(n_qubits)

        prob = (1.0f0 - avg_z) / 2.0f0
        p_c = clamp(prob, 1.0f-7, 1.0f0 - 1.0f-7)
        l -= (Float32(positive_weight) * y * log(p_c) + (1.0f0 - y) * log(1.0f0 - p_c))
    end

    return l / size(X_batch, 1)
end

"""
    train_step!(model, opt_state, X_batch, y_batch; positive_weight = 1)

Executes a single optimization step (forward + backward pass) using Zygote.
Updates the `model.params` in-place.
"""
function train_step!(
    model::VariationalQuantumClassifier,
    opt_state,
    X_batch,
    y_batch;
    positive_weight::Real = 1,
)
    val, grads = Zygote.withgradient(model) do m
        loss_function(m, X_batch, y_batch; positive_weight = positive_weight)
    end
    Flux.update!(opt_state, model.params, grads[1].params)
    return val
end
