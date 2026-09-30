# Sliding windows over a record: their count, the edge margin, and the
# feature and label table of every window.

"""
    window_count(n_points, window_size, step_size) -> Int

Number of sliding windows of `window_size` samples advancing by `step_size`
over a record of `n_points` samples; throws when the record is shorter
than one window or the geometry is invalid.
"""
function window_count(n_points::Integer, window_size::Integer, step_size::Integer)
    window_size >= 2 || throw(ArgumentError("window_size = $window_size; must be >= 2."))
    1 <= step_size <= window_size ||
        throw(ArgumentError("step_size = $step_size; must lie in [1, $window_size]."))
    n_points >= window_size || throw(
        ArgumentError(
            "record of $n_points samples is shorter than window_size = $window_size.",
        ),
    )
    return div(n_points - window_size, step_size) + 1
end

"""
    edge_margin_windows(settings) -> Int

Number of windows dropped at each end of a record for the `edge_margin`
of the `[preprocessing]` settings, given in window lengths:
`round(edge_margin * window_size / step_size)`.
"""
function edge_margin_windows(settings::NamedTuple)
    return round(Int, settings.edge_margin * settings.window_size / settings.step_size)
end

"""
    window_features(A, fs; window_size, step_size, low_band, high_band,
                    band_edges = [1e-3, 5e-3, 1e-1], feature_set) -> Matrix{Float32}

Feature matrix of the sliding windows of the (whitened) record `A` sampled
at `fs` [Hz]: one row per window of `window_size` samples advancing by
`step_size`, the columns named by [`feature_names`](@ref) and computed by
[`extract_features`](@ref) with the analysis bands `low_band`,
`high_band` [Hz] (`:whitened`) or the `band_edges` [Hz] (`:bands`).
"""
function window_features(
    A::AbstractVector{<:Real},
    fs::Real;
    window_size::Integer,
    step_size::Integer,
    low_band::Tuple{Real,Real},
    high_band::Tuple{Real,Real},
    band_edges::AbstractVector{<:Real} = [1e-3, 5e-3, 1e-1],
    feature_set::Symbol,
)
    n_windows = window_count(length(A), window_size, step_size)
    names = feature_names(feature_set; n_bands = length(band_edges) - 1)
    features = Matrix{Float32}(undef, n_windows, length(names))
    decile = max(1, div(n_windows, 10))
    for i in 1:n_windows
        lo = (i - 1) * step_size + 1
        window = @view A[lo:(lo+window_size-1)]
        features[i, :] .= extract_features(
            window,
            fs;
            low_band = low_band,
            high_band = high_band,
            band_edges = band_edges,
            feature_set = feature_set,
        )
        i % decile == 0 && @info "feature extraction" windows = "$i / $n_windows"
    end
    return features
end

"""
    window_labels(raw_labels, raw_snrs; window_size, step_size) -> (labels, snrs)

Per-window labels of the point-wise `raw_labels` and signal-to-noise
ratios `raw_snrs` under the sliding-window geometry of
[`window_features`](@ref): a window is positive (label 1) when any of its
samples is labelled 1, and carries the largest per-sample SNR inside it.
Returns `labels::Vector{Int}` and `snrs::Vector{Float32}`.
"""
function window_labels(
    raw_labels::AbstractVector{<:Integer},
    raw_snrs::AbstractVector{<:Real};
    window_size::Integer,
    step_size::Integer,
)
    length(raw_labels) == length(raw_snrs) || throw(
        DimensionMismatch(
            "$(length(raw_labels)) labels for $(length(raw_snrs)) SNR values.",
        ),
    )
    n_windows = window_count(length(raw_labels), window_size, step_size)
    labels = zeros(Int, n_windows)
    snrs = zeros(Float32, n_windows)
    for i in 1:n_windows
        lo = (i - 1) * step_size + 1
        hi = lo + window_size - 1
        labels[i] = any(==(1), @view raw_labels[lo:hi]) ? 1 : 0
        snrs[i] = maximum(@view raw_snrs[lo:hi])
    end
    return labels, snrs
end
