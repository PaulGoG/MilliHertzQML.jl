# src/data.jl

"""
    load_data(feature_path, label_path)

Load a feature matrix and label vector from CSV files produced by
`scripts/preprocess_ldc.jl`.

Each feature column is clamped to a fixed scale and mapped linearly to
``[0, 2\\pi]`` for phase encoding. The clamping scales are calibrated to the
unit-variance output of `scripts/generate_data.jl`; physical-amplitude LDC
strain data saturates the clamps and is not currently supported.

Returns `(X, y, df_labels)`, where `df_labels` retains auxiliary columns such
as `SNR` when present.
"""
function load_data(feature_path, label_path)
    X = load_features(feature_path)
    df_labels = CSV.read(label_path, DataFrame)
    y = Int.(df_labels[:, :Label])
    return X, y, df_labels
end

"""
    FEATURE_SCALES

Fixed `(min, max)` clamping scales per feature column, in feature order
(low-band magnitude, high-band magnitude, spectral entropy, log10 PSD
standard deviation). Calibrated to the unit-variance output of the
integrated simulator.
"""
const FEATURE_SCALES = (
    (0.0f0, 50.0f0),
    (0.0f0, 50.0f0),
    (0.0f0, 10.0f0),
    (0.0f0, 7.0f0),
)

"""
    load_features(feature_path)

Load a feature matrix from CSV without labels (blind inference path).
Each column is clamped to `FEATURE_SCALES` and mapped linearly to
``[0, 2\\pi]`` for phase encoding.
"""
function load_features(feature_path)
    df_features = CSV.read(feature_path, DataFrame)
    X = Matrix{Float32}(df_features)
    for col in 1:min(size(X, 2), length(FEATURE_SCALES))
        s_min, s_max = FEATURE_SCALES[col]
        X[:, col] .= clamp.(X[:, col], s_min, s_max)
        X[:, col] .= (X[:, col] .- s_min) ./ (s_max - s_min) .* Float32(2π)
    end
    return X
end

"""
    extract_features(x, sample_rate=0.2)

Calculates the 4-dimensional physical feature vector from a raw time-series strain segment `x`.
This function acts as the bridge between raw LDC telemetry and the quantum classifier.
Uses LISA milliHertz physics bands: Low (1mHz - 5mHz), High (5mHz - 100mHz).
"""
function extract_features(x, sample_rate=0.2)
    spec = abs.(rfft(x))
    n_samples = length(x)
    freqs = rfftfreq(n_samples, sample_rate)

    # Base Band: Low (1mHz - 5mHz), High (5mHz - 100mHz)
    p_low = mean(spec[(freqs .>= 1e-3) .& (freqs .<= 5e-3)])
    p_high = mean(spec[(freqs .> 5e-3) .& (freqs .<= 1e-1)])

    psd = spec.^2 .+ 1e-12
    p_norm = psd ./ sum(psd)
    entropy = -sum(p_norm .* log.(p_norm))

    psd_std = log10(std(psd) + 1e-12)

    return Float32(p_low), Float32(p_high), Float32(entropy), Float32(psd_std)
end