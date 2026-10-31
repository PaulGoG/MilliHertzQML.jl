# The variational quantum classifier as a window scorer of the streaming
# detector, and the detector of a training run.

"""
    detector_from_run(model_path; threshold_path = joinpath(dirname(model_path), "threshold.toml"),
                      config_path = joinpath(dirname(model_path), "config.toml"),
                      psd_sidecar = "", context_windows = 4) -> StreamingDetector

The [`StreamingDetector`](@ref) of a training run: classifier (circuit or
classical control) and scaler from the artifact, the fitted threshold from
`threshold.toml`, and the conditioning — window geometry, whitening PSD,
record high-pass, and the feature map ([`FeatureMap`](@ref) of the
sidecar: feature set, analysis bands, channel combination) — from the
sidecar of the feature table the run was trained on (`[training]
train_features` of its `config.toml` snapshot). The sidecar must record
`sample_rate`, `window_size`, `step_size`, `highpass_cutoff_hz`,
`highpass_order` and the three band keys, as the pre-processor writes
them; a missing key is refused with an `ArgumentError`. For a model
trained on several channels the detector carries one whitening PSD per
channel and the combination of their features, and scores the runs that
deliver those channels ([`ScheduledRecordRun`](@ref)).

`psd_sidecar`, when given, supplies the whitening PSD from a different
feature sidecar while everything else still comes from the training one.
A whitening PSD calibrates the record being scored: the persisted
training PSD describes the noise of the training record, and where that
noise is not stationary between records — the Galactic foreground is
modulated over the year by the constellation's antenna pattern —
whitening a later record with it mis-scales every band power, the feature
scaler clips the result, and the scores fall. The sidecar of the record
under analysis is to be preferred whenever one exists.

`context_windows` must reach past the conditioning kernel. The kernel of
the record high-pass followed by whitening decays as a power law, not
exponentially, when the PSD resolves sharp spectral features; with the
Sangria Welch estimate its envelope is still 7 % of the peak eight window
lengths from the impulse and the streamed scores agree with the batch
pipeline only from about twenty window lengths upward, not monotonically
below that. The value is set by measuring the agreement with the batch
path on the record at hand.
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
    model, _, stored_scaler = load_model(model_path)
    scaler = require_scaler(stored_scaler, model_path)
    threshold = Float32(TOML.parsefile(threshold_path)["threshold"]["value"])
    training = section(TOML.parsefile(config_path), "training")
    haskey(training, "train_features") || throw(
        ArgumentError(
            "the snapshot $config_path records no [training] train_features; the " *
            "detector reproduces the conditioning of a training run.",
        ),
    )
    features_path = resolvepath(cfgget(training, "train_features", ""; type = String))
    training_sidecar = sidecar_path(features_path)
    isfile(training_sidecar) || throw(
        ArgumentError(
            "the feature sidecar $training_sidecar of the training run is required to " *
            "reproduce the conditioning.",
        ),
    )
    conditioning = get(TOML.parsefile(training_sidecar), "features", Dict{String,Any}())
    for key in CONDITIONING_KEYS
        haskey(conditioning, key) ||
            throw(ArgumentError("the sidecar $training_sidecar does not record $key."))
    end
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
            psd_sidecar training_sidecar = training_sidecar
    end
    return StreamingDetector(
        model,
        scaler,
        threshold;
        features = FeatureMap(features_path),
        sample_rate = cfgget(
            conditioning,
            "sample_rate",
            nothing;
            type = Float64,
            min = 1e-6,
        ),
        window_size = cfgget(conditioning, "window_size", nothing; type = Int, min = 2),
        step_size = cfgget(conditioning, "step_size", nothing; type = Int, min = 1),
        psd = whitening_psd_from_sidecar(
            isempty(psd_sidecar) ? training_sidecar : psd_sidecar,
        ),
        highpass_cutoff_hz = cfgget(
            conditioning,
            "highpass_cutoff_hz",
            nothing;
            type = Float64,
            min = 0.0,
        ),
        highpass_order = cfgget(
            conditioning,
            "highpass_order",
            nothing;
            type = Int,
            min = 1,
        ),
        context_windows = context_windows,
    )
end

"""
    CONDITIONING_KEYS

The keys of the `[features]` table of a feature sidecar that
[`detector_from_run`](@ref) reads to reproduce the conditioning of a
training run, besides the band keys of [`FeatureMap`](@ref).
"""
const CONDITIONING_KEYS =
    ("sample_rate", "window_size", "step_size", "highpass_cutoff_hz", "highpass_order")

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
    StreamingDetector(model, scaler, threshold; features, sample_rate, window_size,
                      step_size, conditioning...)

The detector of a trained classifier: a [`ClassifierScorer`](@ref) of
`model`, `scaler` and the [`FeatureMap`](@ref) `features`, under the
conditioning keywords of the generic constructor (`sample_rate`,
`window_size`, `step_size`, `psd`, `highpass_cutoff_hz`,
`highpass_order`, `context_windows`). For a model trained on several
channels `psd` holds one whitening PSD per channel and `features.combination`
says how their features are combined.

# Example

```julia
features = FeatureMap(; low_band = (1e-3, 5e-3), high_band = (5e-3, 1e-1),
                      band_edges = [1e-3, 5e-3, 1e-1])
detector = StreamingDetector(model, scaler, 0.5; features, sample_rate = 0.2,
                             window_size = 1000, step_size = 100)
```
"""
function StreamingDetector(
    model::AbstractClassifier,
    scaler::FeatureScaler,
    threshold::Real;
    features::FeatureMap,
    conditioning...,
)
    return StreamingDetector(
        ClassifierScorer(model, scaler, features),
        threshold;
        conditioning...,
    )
end
