# src/estimators.jl — the interface between the conditioning chain of a
# streamed or recorded time series and the method applied to each window:
# estimators and their memory trait, scalar scorers for detection, and the
# spectral feature map of a conditioned window.

"""
    AbstractWindowEstimator

A method applied to one conditioned window of a time series (a detector
score, a parameter estimate, a posterior summary, a forecast). The chain
around it — delivery, gaps, window scheduling, conditioning, calibration,
evaluation — does not depend on the method. Concrete estimators declare
their memory with [`estimator_memory`](@ref).
"""
abstract type AbstractWindowEstimator end

"""
    AbstractWindowScorer <: AbstractWindowEstimator

An estimator whose output is one scalar score per window, larger meaning
more evidence for the event class, bounded by [`score_bounds`](@ref). The
detection chain (threshold, persistence, crediting) acts on it. A scorer
implements [`window_score`](@ref).
"""
abstract type AbstractWindowScorer <: AbstractWindowEstimator end

"""
    EstimatorMemory

Memory trait of an [`AbstractWindowEstimator`](@ref): [`Stateless`](@ref)
or [`Stateful`](@ref).
"""
abstract type EstimatorMemory end

"""
    Stateless()

An estimator whose output for a window depends on that window alone, so
windows may be evaluated in any order, as they complete.
"""
struct Stateless <: EstimatorMemory end

"""
    Stateful()

An estimator that carries state from window to window (a reservoir, a
sequential posterior) and so must see the windows in content order, each
once, with every hole declared.
"""
struct Stateful <: EstimatorMemory end

"""
    estimator_memory(estimator) -> EstimatorMemory

[`Stateless`](@ref) unless an estimator declares otherwise.
"""
estimator_memory(::AbstractWindowEstimator) = Stateless()

"""
    window_score(scorer, window, sample_rate) -> Float32

Score of one conditioned `window` (samples at `sample_rate` [Hz]).
Implemented by every concrete [`AbstractWindowScorer`](@ref).
"""
function window_score end

"""
    score_label(scorer) -> String

Axis label of the score in figures; `"Score"` unless a scorer declares
otherwise.
"""
score_label(::AbstractWindowScorer) = "Score"

"""
    score_bounds(scorer) -> Tuple{Float64,Float64}

Range of the score; `(0.0, 1.0)` unless a scorer declares otherwise.
"""
score_bounds(::AbstractWindowScorer) = (0.0, 1.0)

"""
    FeatureMap(; feature_set = :whitened, low_band = (1e-3, 5e-3),
               high_band = (5e-3, 1e-1), band_edges = [1e-3, 5e-3, 1e-1])

The spectral features of a conditioned window ([`extract_features`](@ref)):
the feature set and its analysis bands [Hz]. `band_edges` is validated
([`check_band_edges`](@ref)) for every set, as the conditioning of a run
records it, and used only by the `:bands` set.
"""
struct FeatureMap
    feature_set::Symbol
    low_band::Tuple{Float64,Float64}
    high_band::Tuple{Float64,Float64}
    band_edges::Vector{Float64}
    function FeatureMap(;
        feature_set::Symbol = :whitened,
        low_band::Tuple{Real,Real} = (1e-3, 5e-3),
        high_band::Tuple{Real,Real} = (5e-3, 1e-1),
        band_edges::AbstractVector{<:Real} = [1e-3, 5e-3, 1e-1],
    )
        feature_set in FEATURE_SETS || throw(
            ArgumentError("feature_set = $feature_set; expected one of $(FEATURE_SETS)."),
        )
        return new(
            feature_set,
            (Float64(low_band[1]), Float64(low_band[2])),
            (Float64(high_band[1]), Float64(high_band[2])),
            check_band_edges(band_edges),
        )
    end
end

"""
    extract_features(map::FeatureMap, window, sample_rate) -> Tuple

Features of `window` under the set and bands of `map`.
"""
function extract_features(
    map::FeatureMap,
    window::AbstractVector{<:Real},
    sample_rate::Real,
)
    return extract_features(
        window,
        sample_rate;
        low_band = map.low_band,
        high_band = map.high_band,
        band_edges = map.band_edges,
        feature_set = map.feature_set,
    )
end
