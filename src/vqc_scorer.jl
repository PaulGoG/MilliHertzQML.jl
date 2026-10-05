# The variational quantum classifier as a window scorer of the streaming
# detector, and the detector of a training run.

"""
    detector_from_run(model_path; threshold_path = joinpath(dirname(model_path), "threshold.toml"),
                      config_path = joinpath(dirname(model_path), "config.toml"),
                      psd_sidecar = "", context_windows = 4) -> StreamingDetector

The [`StreamingDetector`](@ref) of a training run: classifier (circuit or
classical control) and scaler from the artifact, the fitted threshold from `threshold.toml`, and the
conditioning — window geometry, whitening PSD, analysis bands, record
high-pass, feature set — from the sidecar of the feature table the run was
trained on (recorded in its `config.toml` snapshot). For a model trained
on several channels the detector carries one whitening PSD per channel and
the combination of their features, and scores the runs that deliver those
channels ([`ScheduledRecordRun`](@ref)).

`psd_sidecar`, when given, supplies the whitening PSD from a different
feature sidecar while everything else still comes from the training one.
A whitening PSD is a calibration of the record being scored, not a
property of the model: the persisted training PSD describes the noise of
the training record, and where that noise is not stationary between
records — the Galactic foreground is modulated over the year by the
constellation's antenna pattern — whitening a later record with it
mis-scales every band power, the feature scaler clips the result, and the
scores fall. Give the sidecar of the record under analysis whenever
one exists.

`context_windows` must reach past the conditioning kernel. The kernel of
the record high-pass followed by whitening decays as a power law, not
exponentially, when the PSD resolves sharp spectral features; with the
Sangria Welch estimate its envelope is still 7 % of the peak eight window
lengths from the impulse and the streamed scores agree with the batch
pipeline only from about twenty window lengths upward, not monotonically
below that. Measure the agreement against the batch path for the record
at hand rather than assuming a value is large enough.
"""
function detector_from_run(
    model_path::AbstractString;
    threshold_path::AbstractString = joinpath(dirname(model_path), "threshold.toml"),
    config_path::AbstractString = joinpath(dirname(model_path), "config.toml"),
    psd_sidecar::AbstractString = "",
    context_windows::Integer = 4,
)
    isfile(model_path) || throw(ArgumentError("model artifact not found: $model_path"))
    isfile(threshold_path) ||
        throw(ArgumentError("threshold file not found: $threshold_path"))
    isfile(config_path) || throw(ArgumentError("training snapshot not found: $config_path"))
    model, _, scaler = load_model(model_path)
    scaler === nothing &&
        throw(ArgumentError("the model artifact carries no feature scaler."))
    threshold = Float32(TOML.parsefile(threshold_path)["threshold"]["value"])
    snapshot = TOML.parsefile(config_path)
    features_path = resolvepath(
        cfgget(section(snapshot, "training"), "train_features", ""; type = String),
    )
    sidecar_path = replace(features_path, r"\.csv$" => ".toml")
    isfile(sidecar_path) || throw(
        ArgumentError(
            "the feature sidecar $sidecar_path of the training run is required to " *
            "reproduce the conditioning.",
        ),
    )
    features = get(TOML.parsefile(sidecar_path), "features", Dict{String,Any}())
    low = cfgget(features, "low_band_hz", [1e-3, 5e-3]; type = AbstractVector)
    high = cfgget(features, "high_band_hz", [5e-3, 1e-1]; type = AbstractVector)
    edges = cfgget(features, "band_edges_hz", [1e-3, 5e-3, 1e-1]; type = AbstractVector)
    if !isempty(psd_sidecar)
        isfile(psd_sidecar) ||
            throw(ArgumentError("whitening sidecar not found: $psd_sidecar"))
        trained = recorded_channels(features_path; default = "A")
        supplied =
            recorded_channels(replace(psd_sidecar, r"\.toml$" => ".csv"); default = "A")
        trained == supplied || throw(
            ArgumentError(
                "the whitening sidecar $psd_sidecar describes the channels $supplied; " *
                "the model was trained on $trained.",
            ),
        )
        @info "whitening with a sidecar other than the training run's" psd_sidecar =
            psd_sidecar training_sidecar = sidecar_path
    end
    return StreamingDetector(
        model,
        scaler,
        threshold;
        sample_rate = cfgget(features, "sample_rate", 0.2; type = Float64, min = 1e-6),
        window_size = cfgget(features, "window_size", 1000; type = Int, min = 2),
        step_size = cfgget(features, "step_size", 100; type = Int, min = 1),
        psd = whitening_psd_from_sidecar(isempty(psd_sidecar) ? sidecar_path : psd_sidecar),
        highpass_cutoff_hz = cfgget(features, "highpass_cutoff_hz", 5e-4; type = Float64),
        highpass_order = cfgget(features, "highpass_order", 8; type = Int, min = 1),
        low_band = (Float64(low[1]), Float64(low[2])),
        high_band = (Float64(high[1]), Float64(high[2])),
        band_edges = Float64.(edges),
        feature_set = Symbol(cfgget(features, "feature_set", "whitened"; type = String)),
        combination = Symbol(
            cfgget(features, "channel_combination", "mean"; type = String),
        ),
        context_windows = context_windows,
    )
end

"""
    ClassifierScorer(model, scaler, features)

A classifier ([`AbstractClassifier`](@ref)) as an
[`AbstractWindowScorer`](@ref): the features of a conditioned window — a
vector, or a matrix of several channels — under the [`FeatureMap`](@ref)
`features`, encoded with the train-fitted `scaler`
([`encode_features`](@ref)) and scored as the probability of the MBHB
class ([`predict_probability`](@ref)).
"""
struct ClassifierScorer{M<:AbstractClassifier} <: AbstractWindowScorer
    model::M
    scaler::FeatureScaler
    features::FeatureMap
end

"""
    VQCScorer

The scorer of a variational circuit, alias of
`ClassifierScorer{VariationalQuantumClassifier}` ([`ClassifierScorer`](@ref)).
"""
const VQCScorer = ClassifierScorer{VariationalQuantumClassifier}

window_score(scorer::ClassifierScorer, window::AbstractVector{<:Real}, sample_rate::Real) =
    classifier_score(scorer, window, sample_rate)
window_score(scorer::ClassifierScorer, window::AbstractMatrix{<:Real}, sample_rate::Real) =
    classifier_score(scorer, window, sample_rate)

function classifier_score(scorer::ClassifierScorer, window, sample_rate)
    features = extract_features(scorer.features, window, sample_rate)
    encoded = encode_features(scorer.scaler, reshape(collect(Float32.(features)), 1, :))
    return predict_probability(scorer.model, vec(encoded))
end

score_label(::ClassifierScorer) = "MBHB probability"

"""
    StreamingDetector(model, scaler, threshold; sample_rate, window_size, step_size,
                      psd = nothing, highpass_cutoff_hz = 5e-4, highpass_order = 8,
                      low_band = (1e-3, 5e-3), high_band = (5e-3, 1e-1),
                      band_edges = [1e-3, 5e-3, 1e-1], feature_set = :whitened,
                      combination = :mean, context_windows = 4)

The detector of a trained classifier: a [`ClassifierScorer`](@ref) of `model`,
`scaler` and the feature map of the band keywords, under the conditioning
keywords of the generic constructor. For a model trained on several
channels `psd` holds one whitening PSD per channel and `combination` says
how their features are combined.
"""
function StreamingDetector(
    model::AbstractClassifier,
    scaler::FeatureScaler,
    threshold::Real;
    low_band::Tuple{Real,Real} = (1e-3, 5e-3),
    high_band::Tuple{Real,Real} = (5e-3, 1e-1),
    band_edges::AbstractVector{<:Real} = [1e-3, 5e-3, 1e-1],
    feature_set::Symbol = :whitened,
    combination::Symbol = :mean,
    conditioning...,
)
    features = FeatureMap(;
        feature_set = feature_set,
        low_band = low_band,
        high_band = high_band,
        band_edges = band_edges,
        combination = combination,
    )
    return StreamingDetector(
        ClassifierScorer(model, scaler, features),
        threshold;
        conditioning...,
    )
end
