# The train-fitted feature scaler onto the phase-encoding interval of the
# circuit.

"""
    FeatureScaler(lower, upper; phase_span = π)

Per-feature affine map from the range `[lower[j], upper[j]]` onto the
phase-encoding interval ``[0, \\phi]`` with ``\\phi`` = `phase_span` [rad];
values outside the range are clamped. The encoding gate ``R_z(x)`` is
``2\\pi``-periodic with ``R_z(2\\pi) = -I``, so a span of ``2\\pi`` prepares
the same state at both clamp ends and folds every feature saturated above
the training range back onto the noise floor; ``\\pi``, the default, is the
widest span whose ends are distinct states. Fitted on the training
partition by [`fit_scaler`](@ref), persisted with the model, and applied
by [`encode_features`](@ref).
"""
struct FeatureScaler
    lower::Vector{Float32}
    upper::Vector{Float32}
    phase_span::Float32
    function FeatureScaler(
        lower::AbstractVector{<:Real},
        upper::AbstractVector{<:Real};
        phase_span::Real = π,
    )
        length(lower) == length(upper) || throw(
            DimensionMismatch(
                "scaler bounds have lengths $(length(lower)) and $(length(upper)).",
            ),
        )
        all(upper .> lower) ||
            throw(ArgumentError("every scaler upper bound must exceed its lower bound."))
        span = Float32(phase_span)
        0 < span <= Float32(2π) ||
            throw(ArgumentError("phase_span = $phase_span rad; must lie in (0, 2π]."))
        return new(Vector{Float32}(lower), Vector{Float32}(upper), span)
    end
end

"""
    fit_scaler(X; quantiles = (0.005, 0.995), phase_span = π) -> FeatureScaler

Scaler whose bounds are the per-column empirical quantiles `quantiles` of the
feature matrix `X` (samples × features), mapping onto ``[0, \\phi]`` with
``\\phi`` = `phase_span` [rad]. Throws an `ArgumentError` for a constant
feature column.
"""
function fit_scaler(
    X::AbstractMatrix{<:Real};
    quantiles::Tuple{Real,Real} = (0.005, 0.995),
    phase_span::Real = π,
)
    0 <= quantiles[1] < quantiles[2] <= 1 ||
        throw(ArgumentError("quantiles = $quantiles; need 0 <= lower < upper <= 1."))
    lower = [quantile(view(X, :, j), quantiles[1]) for j in 1:size(X, 2)]
    upper = [quantile(view(X, :, j), quantiles[2]) for j in 1:size(X, 2)]
    for j in 1:size(X, 2)
        upper[j] > lower[j] || throw(
            ArgumentError(
                "feature column $j is constant between the $(quantiles) quantiles; " *
                "it carries no information and cannot be scaled.",
            ),
        )
    end
    return FeatureScaler(lower, upper; phase_span = phase_span)
end

"""
    encode_features(scaler, X) -> Matrix{Float32}

`X` (samples × features) clamped to the scaler bounds and mapped linearly
onto ``[0, \\phi]``, the scaler's `phase_span`.
"""
function encode_features(scaler::FeatureScaler, X::AbstractMatrix{<:Real})
    size(X, 2) == length(scaler.lower) || throw(
        DimensionMismatch(
            "feature matrix has $(size(X, 2)) columns; the scaler expects $(length(scaler.lower)).",
        ),
    )
    E = Matrix{Float32}(undef, size(X))
    for j in 1:size(X, 2)
        lo, hi = scaler.lower[j], scaler.upper[j]
        for i in 1:size(X, 1)
            E[i, j] = (clamp(Float32(X[i, j]), lo, hi) - lo) / (hi - lo) * scaler.phase_span
        end
    end
    return E
end
