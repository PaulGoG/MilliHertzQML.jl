# src/vqc_scorer.jl — the variational quantum classifier as a window scorer
# of the streaming detector.

"""
    VQCScorer(model, scaler, features)

The classifier as an [`AbstractWindowScorer`](@ref): the features of a
conditioned window under the [`FeatureMap`](@ref) `features`, encoded with
the train-fitted `scaler` ([`encode_features`](@ref)) and scored as the
probability of the MBHB class ([`predict_probability`](@ref)).
"""
struct VQCScorer <: AbstractWindowScorer
    model::VariationalQuantumClassifier
    scaler::FeatureScaler
    features::FeatureMap
end

function window_score(scorer::VQCScorer, window::AbstractVector{<:Real}, sample_rate::Real)
    features = extract_features(scorer.features, window, sample_rate)
    encoded = encode_features(scorer.scaler, reshape(collect(Float32.(features)), 1, :))
    return predict_probability(scorer.model, vec(encoded))
end

score_label(::VQCScorer) = "MBHB probability"

"""
    StreamingDetector(model, scaler, threshold; sample_rate, window_size, step_size,
                      psd = nothing, highpass_cutoff_hz = 5e-4, highpass_order = 8,
                      low_band = (1e-3, 5e-3), high_band = (5e-3, 1e-1),
                      band_edges = [1e-3, 5e-3, 1e-1], feature_set = :whitened,
                      context_windows = 4)

The detector of a trained classifier: a [`VQCScorer`](@ref) of `model`,
`scaler` and the feature map of the band keywords, under the conditioning
keywords of the generic constructor.
"""
function StreamingDetector(
    model::VariationalQuantumClassifier,
    scaler::FeatureScaler,
    threshold::Real;
    low_band::Tuple{Real,Real} = (1e-3, 5e-3),
    high_band::Tuple{Real,Real} = (5e-3, 1e-1),
    band_edges::AbstractVector{<:Real} = [1e-3, 5e-3, 1e-1],
    feature_set::Symbol = :whitened,
    conditioning...,
)
    features = FeatureMap(;
        feature_set = feature_set,
        low_band = low_band,
        high_band = high_band,
        band_edges = band_edges,
    )
    return StreamingDetector(VQCScorer(model, scaler, features), threshold; conditioning...)
end
