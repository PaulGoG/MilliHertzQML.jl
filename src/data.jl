# src/data.jl — whitened spectral features of a strain window, the
# train-fitted feature scaler, and CSV loading of feature and label tables.

"""
    feature_names(feature_set) -> Vector{Symbol}

Column names of the feature table produced by [`extract_features`](@ref)
for `feature_set` (`:whitened` or `:paper`).
"""
function feature_names(feature_set::Symbol)
    feature_set == :whitened && return [:p_low, :p_high, :spectral_entropy, :log_power_std]
    feature_set == :paper &&
        return [:spectral_entropy, :log_power_mean, :log_power_std, :log_power_max]
    throw(ArgumentError("feature_set = $feature_set; expected :whitened or :paper."))
end

"""
    extract_features(x, sample_rate = 0.2; low_band = (1e-3, 5e-3),
                     high_band = (5e-3, 1e-1), taper = :hann, feature_set = :whitened)

Four-dimensional feature vector of a window `x` sampled at `sample_rate`
[Hz], computed from its tapered periodogram ``P_k``
([`tapered_periodogram`](@ref)). Returns a tuple of `Float32` whose entries
are named by [`feature_names`](@ref).

`feature_set = :whitened` (the default) expects a window of the
**whitened** record ([`whiten_record`](@ref)), whose periodogram has unit
mean for noise and is therefore independent of window length and strain
amplitude:

1. mean whitened power in `low_band` [Hz];
2. mean whitened power in `high_band` [Hz];
3. spectral entropy of the normalized whitened power, divided by
   ``\\ln N_\\mathrm{bins}`` so that it lies in ``[0, 1]``;
4. ``\\log_{10}`` of the standard deviation of the whitened power (0 for
   white noise, whose periodogram is exponentially distributed).

`feature_set = :paper` is the set of Isfan et al. (2025) on the raw window:
the normalized spectral entropy and ``\\log_{10}`` of the mean, standard
deviation, and maximum of the periodogram (the paper uses the raw
moments; the logarithm is a monotone transform that keeps their min–max
scaling well conditioned over the many decades a TDI spectrum spans).

Throws an `ArgumentError` when an analysis band holds no frequency bin.
"""
function extract_features(
    x::AbstractVector{<:Real},
    sample_rate::Real = 0.2;
    low_band::Tuple{Real,Real} = (1e-3, 5e-3),
    high_band::Tuple{Real,Real} = (5e-3, 1e-1),
    taper::Symbol = :hann,
    feature_set::Symbol = :whitened,
)
    sample_rate > 0 || throw(ArgumentError("sample_rate = $sample_rate; must be positive."))
    feature_set in (:whitened, :paper) ||
        throw(ArgumentError("feature_set = $feature_set; expected :whitened or :paper."))
    power = tapered_periodogram(x; taper = taper)
    n_samples = length(x)

    total = sum(power)
    n_bins = length(power) - 1   # the DC bin carries no power
    entropy = 0.0
    if total > 0 && n_bins > 1
        for p in power
            p > 0 || continue
            q = p / total
            entropy -= q * log(q)
        end
        entropy /= log(n_bins)
    end
    positive = @view power[2:end]
    log_power_std = log10(std(positive) + 1e-300)

    if feature_set == :paper
        log_power_mean = log10(mean(positive) + 1e-300)
        log_power_max = log10(maximum(positive) + 1e-300)
        return Float32(entropy),
        Float32(log_power_mean),
        Float32(log_power_std),
        Float32(log_power_max)
    end

    freqs = rfftfreq(n_samples, sample_rate)
    mask_low = (freqs .>= low_band[1]) .& (freqs .<= low_band[2])
    mask_high = (freqs .> high_band[1]) .& (freqs .<= high_band[2])
    (any(mask_low) && any(mask_high)) || throw(
        ArgumentError(
            "window of $n_samples samples at $sample_rate Hz has no frequency bins " *
            "in the $(low_band) Hz or $(high_band) Hz analysis bands; use a longer window.",
        ),
    )
    p_low = mean(@view power[mask_low])
    p_high = mean(@view power[mask_high])
    return Float32(p_low), Float32(p_high), Float32(entropy), Float32(log_power_std)
end

"""
    FeatureScaler(lower, upper)

Per-feature affine map from the range `[lower[j], upper[j]]` onto the
phase-encoding interval ``[0, 2\\pi]``; values outside the range are
clamped. Fitted on the training partition by [`fit_scaler`](@ref),
persisted with the model, and applied by [`encode_features`](@ref).
"""
struct FeatureScaler
    lower::Vector{Float32}
    upper::Vector{Float32}
    function FeatureScaler(lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real})
        length(lower) == length(upper) || throw(
            DimensionMismatch(
                "scaler bounds have lengths $(length(lower)) and $(length(upper)).",
            ),
        )
        all(upper .> lower) ||
            throw(ArgumentError("every scaler upper bound must exceed its lower bound."))
        return new(Vector{Float32}(lower), Vector{Float32}(upper))
    end
end

"""
    fit_scaler(X; quantiles = (0.005, 0.995)) -> FeatureScaler

Scaler whose bounds are the per-column empirical quantiles `quantiles` of the
feature matrix `X` (samples × features). Throws an `ArgumentError` for a
constant feature column.
"""
function fit_scaler(X::AbstractMatrix{<:Real}; quantiles::Tuple{Real,Real} = (0.005, 0.995))
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
    return FeatureScaler(lower, upper)
end

"""
    encode_features(scaler, X) -> Matrix{Float32}

`X` (samples × features) clamped to the scaler bounds and mapped linearly
onto ``[0, 2\\pi]``.
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
            E[i, j] = (clamp(Float32(X[i, j]), lo, hi) - lo) / (hi - lo) * Float32(2π)
        end
    end
    return E
end

"""
    load_features(feature_path) -> Matrix{Float32}

Raw feature matrix (samples × features) from a CSV written by
`scripts/preprocess_ldc.jl`. Encoding is a separate step
([`encode_features`](@ref)) so that the scaler fitted on the training
partition is applied identically at inference.
"""
function load_features(feature_path::AbstractString)
    return Matrix{Float32}(CSV.read(feature_path, DataFrame))
end

"""
    load_data(feature_path, label_path) -> (X, y, df_labels)

Raw feature matrix, integer label vector, and the full label table (which
retains auxiliary columns such as `SNR`) from the CSVs written by
`scripts/preprocess_ldc.jl`.
"""
function load_data(feature_path::AbstractString, label_path::AbstractString)
    X = load_features(feature_path)
    df_labels = CSV.read(label_path, DataFrame)
    y = Int.(df_labels[:, :Label])
    size(X, 1) == length(y) ||
        throw(DimensionMismatch("$(size(X, 1)) feature rows but $(length(y)) labels."))
    return X, y, df_labels
end
