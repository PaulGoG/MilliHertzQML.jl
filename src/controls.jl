# Classical controls of the variational classifier: a logistic regression and
# a one-hidden-layer network on the encoded features the circuit takes,
# trained, persisted and scored through the same stages.

"""
    MODEL_KINDS

The classifiers of `[model] kind`: `"circuit"` (the variational quantum
classifier, [`VariationalQuantumClassifier`](@ref)), `"logistic"`
(logistic regression) and `"perceptron"` (one hidden tanh layer of
`hidden_units` units); the last two are [`ClassicalControl`](@ref)s.
"""
const MODEL_KINDS = ("circuit", "logistic", "perceptron")

"""
    ClassicalControl(n_features, hidden, span, params)
    ClassicalControl(n_features, hidden = 0; span = π, rng = Random.default_rng())

Classical control of the variational classifier on the same encoded
features. With the encoded features ``x \\in [0, \\phi]^{n}`` (the output
of [`encode_features`](@ref), ``\\phi`` = `span` [rad], ``n`` =
`n_features`) and ``u = x / \\phi``:

- `hidden == 0`, logistic regression: ``p = \\sigma(w \\cdot u + b)``,
  ``n + 1`` parameters, `params = [w; b]`;
- `hidden ≥ 1`, one hidden tanh layer of ``m`` = `hidden` units:
  ``p = \\sigma(w_2 \\cdot \\tanh(W_1 u + b_1) + b_2)``, ``m (n + 2) + 1``
  parameters, `params = [vec(W₁); b₁; w₂; b₂]` with ``W_1`` of size
  ``m \\times n`` stored column-major;

with ``\\sigma(z) = 1 / (1 + e^{-z})``. The second constructor draws the
parameters from a zero-mean normal distribution of standard deviation 0.5
on `rng`, as those of the circuit.

# Arguments
- `n_features`: feature dimension, ``\\geq 1``.
- `hidden`: hidden units, ``0`` for the logistic regression.
- `span`: encoding span ``\\phi`` of the features [rad]; positive, finite.
- `params`: parameter vector of length
  [`control_parameter_count`](@ref)`(n_features, hidden)`.

Throws an `ArgumentError` for an invalid `n_features`, `hidden` or `span`
and a `DimensionMismatch` for a parameter vector of the wrong length.
"""
struct ClassicalControl <: AbstractClassifier
    n_features::Int
    hidden::Int
    span::Float32
    params::Vector{Float32}
    function ClassicalControl(
        n_features::Integer,
        hidden::Integer,
        span::Real,
        params::AbstractVector{<:Real},
    )
        n_features >= 1 || throw(
            ArgumentError("n_features = $n_features; at least 1 feature is required."),
        )
        hidden >= 0 ||
            throw(ArgumentError("hidden = $hidden; must be 0 (logistic) or positive."))
        (span > 0 && isfinite(span)) ||
            throw(ArgumentError("span = $span rad; must be positive and finite."))
        n_params = control_parameter_count(n_features, hidden)
        length(params) == n_params || throw(
            DimensionMismatch(
                "parameter vector has length $(length(params)); expected $n_params " *
                "for n_features = $n_features and hidden = $hidden.",
            ),
        )
        return new(n_features, hidden, Float32(span), Vector{Float32}(params))
    end
end

function ClassicalControl(
    n_features::Integer,
    hidden::Integer = 0;
    span::Real = π,
    rng::AbstractRNG = Random.default_rng(),
)
    return ClassicalControl(
        n_features,
        hidden,
        span,
        randn(rng, Float32, control_parameter_count(n_features, hidden)) .* 0.5f0,
    )
end

"""
    control_parameter_count(n_features, hidden) -> Int

Number of parameters of a [`ClassicalControl`](@ref): `n_features + 1`
for the logistic regression (`hidden == 0`), `hidden (n_features + 2) + 1`
for the network of `hidden` tanh units.
"""
control_parameter_count(n_features::Integer, hidden::Integer) =
    hidden == 0 ? n_features + 1 : hidden * (n_features + 2) + 1

# Views of the network parameters into `params`: W₁ (hidden × n_features,
# column-major), b₁, w₂, and the scalar b₂
function network_parameters(model::ClassicalControl)
    n, m, θ = model.n_features, model.hidden, model.params
    W₁ = reshape(@view(θ[1:(m*n)]), m, n)
    b₁ = @view θ[(m*n+1):(m*(n+1))]
    w₂ = @view θ[(m*(n+1)+1):(m*(n+2))]
    return W₁, b₁, w₂, θ[end]
end

# Pre-activation of hidden unit j at the encoded features x
function preactivation(W₁, b₁, x, span::Float32, j::Integer)
    a = b₁[j]
    for i in axes(W₁, 2)
        a += W₁[j, i] * (Float32(x[i]) / span)
    end
    return a
end

# Logit of the logistic regression at the encoded features x
function logistic_logit(θ, x, span::Float32, n::Integer)
    z = θ[n+1]
    for i in 1:n
        z += θ[i] * (Float32(x[i]) / span)
    end
    return z
end

"""
    predict_probability(model::ClassicalControl, x) -> Float32

Probability of the MBHB class of one encoded feature vector `x` (length
`n_features`, values in ``[0, \\phi]`` with ``\\phi`` = `model.span`)
under the logistic regression or the network of `model`
([`ClassicalControl`](@ref)), evaluated in single precision without
allocation. Throws a `DimensionMismatch` for a vector of another length.
"""
function predict_probability(model::ClassicalControl, x)
    length(x) == model.n_features || throw(
        DimensionMismatch(
            "feature vector has length $(length(x)); " *
            "expected n_features = $(model.n_features).",
        ),
    )
    if model.hidden == 0
        z = logistic_logit(model.params, x, model.span, model.n_features)
    else
        W₁, b₁, w₂, b₂ = network_parameters(model)
        z = b₂
        for j in 1:model.hidden
            z += w₂[j] * tanh(preactivation(W₁, b₁, x, model.span, j))
        end
    end
    return 1.0f0 / (1.0f0 + exp(-z))
end

"""
    predict_all(model::ClassicalControl, X; progress = false, threaded = false)
        -> Vector{Float32}

Probability of every row of the encoded feature matrix `X` (samples ×
features), [`predict_probability`](@ref) row by row in one loop;
`threaded` is accepted for the signature common to every classifier and
ignored. With `progress`, a line is printed after every tenth of the rows.
"""
function predict_all(
    model::ClassicalControl,
    X::AbstractMatrix{<:Real};
    progress::Bool = false,
    threaded::Bool = false,
)
    n = size(X, 1)
    probabilities = Vector{Float32}(undef, n)
    tenth = max(1, cld(n, 10))
    for lo in 1:tenth:n
        hi = min(n, lo + tenth - 1)
        for i in lo:hi
            probabilities[i] = predict_probability(model, @view(X[i, :]))
        end
        progress && println("  Progress: $(round(Int, hi / n * 100)) %")
    end
    return probabilities
end

"""
    check_batch(model::ClassicalControl, X_batch, y_batch)

Dimension checks of a batch: `X_batch` holds the samples along its first
dimension and `n_features` features along the second, `y_batch` one label
per sample.
"""
function check_batch(model::ClassicalControl, X_batch, y_batch)
    size(X_batch, 2) == model.n_features || throw(
        DimensionMismatch(
            "feature dimension $(size(X_batch, 2)); " *
            "expected n_features = $(model.n_features).",
        ),
    )
    size(X_batch, 1) == length(y_batch) || throw(
        DimensionMismatch("$(size(X_batch, 1)) samples but $(length(y_batch)) labels."),
    )
    return nothing
end

"""
    batch_gradient(model::ClassicalControl, X_batch, y_batch; positive_weight = 1,
                   kwargs...) -> (loss, gradient)

Mean weighted binary cross-entropy ([`weighted_bce`](@ref)) of the batch
(samples × features) and its gradient with respect to `model.params`, in
the layout of `params`. The gradient is analytic: with
``\\partial \\ell / \\partial z`` = [`weighted_bce_derivative`](@ref)
``(p, y) \\, p (1 - p)`` and ``u = x / \\phi``, the logistic regression has
``\\partial z / \\partial w_i = u_i`` and ``\\partial z / \\partial b = 1``;
the network, with ``h_j = \\tanh a_j`` and
``\\delta_j = w_{2,j} (1 - h_j^2)``, has
``\\partial z / \\partial w_{2,j} = h_j``, ``\\partial z / \\partial b_2 = 1``,
``\\partial z / \\partial W_{1,ji} = \\delta_j u_i`` and
``\\partial z / \\partial b_{1,j} = \\delta_j``. Loss and gradient are summed
over the samples and divided by the batch size.

The further keyword arguments of the circuit's method (`threaded`,
`method`, `workspaces`, …) are accepted and ignored, so that the training
stage calls one signature for every classifier.
"""
function batch_gradient(
    model::ClassicalControl,
    X_batch,
    y_batch;
    positive_weight::Real = 1,
    kwargs...,
)
    check_batch(model, X_batch, y_batch)
    n_samples = size(X_batch, 1)
    n_samples >= 1 || throw(ArgumentError("the batch is empty."))
    n, m, span = model.n_features, model.hidden, model.span
    θ = model.params
    gradient = zeros(Float32, length(θ))
    loss = 0.0f0
    if m == 0
        for k in 1:n_samples
            x = @view X_batch[k, :]
            y = y_batch[k]
            p = 1.0f0 / (1.0f0 + exp(-logistic_logit(θ, x, span, n)))
            loss += weighted_bce(p, y; positive_weight = positive_weight)
            dℓdz =
                weighted_bce_derivative(p, y; positive_weight = positive_weight) *
                p *
                (1.0f0 - p)
            for i in 1:n
                gradient[i] += dℓdz * (Float32(x[i]) / span)
            end
            gradient[n+1] += dℓdz
        end
    else
        W₁, b₁, w₂, b₂ = network_parameters(model)
        ∂W₁ = reshape(@view(gradient[1:(m*n)]), m, n)
        ∂b₁ = @view gradient[(m*n+1):(m*(n+1))]
        ∂w₂ = @view gradient[(m*(n+1)+1):(m*(n+2))]
        h = Vector{Float32}(undef, m)
        for k in 1:n_samples
            x = @view X_batch[k, :]
            y = y_batch[k]
            z = b₂
            for j in 1:m
                h[j] = tanh(preactivation(W₁, b₁, x, span, j))
                z += w₂[j] * h[j]
            end
            p = 1.0f0 / (1.0f0 + exp(-z))
            loss += weighted_bce(p, y; positive_weight = positive_weight)
            dℓdz =
                weighted_bce_derivative(p, y; positive_weight = positive_weight) *
                p *
                (1.0f0 - p)
            for j in 1:m
                δ = w₂[j] * (1.0f0 - h[j]^2)
                ∂w₂[j] += dℓdz * h[j]
                ∂b₁[j] += dℓdz * δ
                for i in 1:n
                    ∂W₁[j, i] += dℓdz * δ * (Float32(x[i]) / span)
                end
            end
            gradient[end] += dℓdz
        end
    end
    return loss / Float32(n_samples), gradient ./ Float32(n_samples)
end

"""
    train_step!(model::ClassicalControl, opt_state, X_batch, y_batch;
                positive_weight = 1, kwargs...) -> loss

One optimisation step: the analytic batch gradient
([`batch_gradient`](@ref)) applied to `model.params` in place through the
Optimisers.jl state `opt_state`. Returns the batch loss before the update.
The further keyword arguments of the circuit's method are accepted and
ignored.
"""
function train_step!(
    model::ClassicalControl,
    opt_state,
    X_batch,
    y_batch;
    positive_weight::Real = 1,
    kwargs...,
)
    loss, gradient =
        batch_gradient(model, X_batch, y_batch; positive_weight = positive_weight)
    Optimisers.update!(opt_state, model.params, gradient)
    return loss
end

"""
    build_classifier(mdl, scaler) -> AbstractClassifier

The classifier of the `[model]` settings `mdl` ([`model_settings`](@ref)):
for `kind = "circuit"` a [`VariationalQuantumClassifier`](@ref) of
`n_qubits` qubits and `n_layers` layers, for `"logistic"` and
`"perceptron"` a [`ClassicalControl`](@ref) without or with
`hidden_units` hidden units, on the encoding span of `scaler`. `n_qubits`
is the feature dimension of every kind. The parameters are drawn from the
default random stream, which the caller seeds. Throws an `ArgumentError`
for a kind outside [`MODEL_KINDS`](@ref).
"""
function build_classifier(mdl::NamedTuple, scaler::FeatureScaler)
    mdl.kind == "circuit" && return VariationalQuantumClassifier(mdl.n_qubits, mdl.n_layers)
    mdl.kind == "logistic" &&
        return ClassicalControl(mdl.n_qubits, 0; span = scaler.phase_span)
    mdl.kind == "perceptron" &&
        return ClassicalControl(mdl.n_qubits, mdl.hidden_units; span = scaler.phase_span)
    throw(
        ArgumentError(
            "[model] kind = \"$(mdl.kind)\"; expected one of " *
            join(("\"$k\"" for k in MODEL_KINDS), ", ") *
            ".",
        ),
    )
end
