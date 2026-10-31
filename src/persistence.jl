# src/persistence.jl — model artifacts: the classifier, its feature
# scaler and metadata in one JLD2 file.

"""
    save_model(path, model; metadata = Dict{String, Any}(), scaler = nothing)

Persist a classifier of either kind to a JLD2 file. A
`VariationalQuantumClassifier` is stored as `kind = "circuit"`, its
parameter vector plus hyperparameters (`n_qubits`, `n_layers`), the
feature `scaler` fitted on the training partition (bounds and
phase-encoding span; `nothing` when absent), and an arbitrary `metadata`
dictionary (run identifier, seed, configuration snapshot).

The circuit itself is not serialised; `load_model` rebuilds it from the
hyperparameters, keeping artifacts loadable across package versions.
"""
function save_model(
    path::AbstractString,
    model::VariationalQuantumClassifier;
    metadata::AbstractDict = Dict{String,Any}(),
    scaler::Union{Nothing,FeatureScaler} = nothing,
)
    jldsave(
        path;
        kind = "circuit",
        n_qubits = model.n_qubits,
        n_layers = model.n_layers,
        params = model.params,
        scaler_lower = scaler === nothing ? nothing : scaler.lower,
        scaler_upper = scaler === nothing ? nothing : scaler.upper,
        scaler_phase_span = scaler === nothing ? nothing : scaler.phase_span,
        metadata = Dict{String,Any}(metadata),
    )
    return path
end

"""
    save_model(path, model::ClassicalControl; metadata = Dict{String, Any}(),
               scaler = nothing)

Persist a [`ClassicalControl`](@ref) to a JLD2 file: `kind = "control"`,
its feature dimension `n_features`, hidden units `hidden`, the flag
`harmonic` of the first-harmonic model, encoding `span` and parameter
vector `params`, the feature `scaler` as the circuit method stores it, and
the `metadata` dictionary. Returns `path`.
"""
function save_model(
    path::AbstractString,
    model::ClassicalControl;
    metadata::AbstractDict = Dict{String,Any}(),
    scaler::Union{Nothing,FeatureScaler} = nothing,
)
    jldsave(
        path;
        kind = "control",
        n_features = model.n_features,
        hidden = model.hidden,
        span = model.span,
        params = model.params,
        harmonic = model.harmonic,
        scaler_lower = scaler === nothing ? nothing : scaler.lower,
        scaler_upper = scaler === nothing ? nothing : scaler.upper,
        scaler_phase_span = scaler === nothing ? nothing : scaler.phase_span,
        metadata = Dict{String,Any}(metadata),
    )
    return path
end

"""
    load_model(path) -> (model, metadata, scaler)

Load a classifier of either kind persisted by [`save_model`](@ref).
`kind = "circuit"` holds a `VariationalQuantumClassifier`: the circuit is
rebuilt from the stored hyperparameters and the stored parameter vector
dispatched into the ansatz layers; the circuit artifacts of releases up to
3.0.0 carry no `kind` key and load as circuits. `kind = "control"` holds
a [`ClassicalControl`](@ref); the control artifacts of the classical-control
study written before the first-harmonic model carry no `harmonic` key and
load as the model they are, `harmonic = false`. Any other kind is refused
with an `ArgumentError`. Returns the feature scaler as well (`nothing` for
artifacts saved without one, see [`require_scaler`](@ref)); a scaler
without its phase-encoding span (artifacts before release 1.1.0) is
refused with an `ArgumentError`.
"""
function load_model(path::AbstractString)
    d = JLD2.load(path)
    kind = get(d, "kind", "circuit")
    if kind == "control"
        model = ClassicalControl(
            d["n_features"],
            d["hidden"],
            d["span"],
            Vector{Float32}(d["params"]);
            harmonic = get(d, "harmonic", false),
        )
    elseif kind == "circuit"
        model = VariationalQuantumClassifier(d["n_qubits"], d["n_layers"])
        model.params = Vector{Float32}(d["params"])
        dispatch_params!(model)
    else
        throw(
            ArgumentError(
                "model artifact $path holds kind = \"$kind\"; " *
                "expected \"circuit\" or \"control\".",
            ),
        )
    end
    lower = get(d, "scaler_lower", nothing)
    lower === nothing && return model, d["metadata"], nothing
    span = get(d, "scaler_phase_span", nothing)
    span === nothing && throw(
        ArgumentError(
            "the model artifact $path records no phase-encoding span of its scaler, " *
            "as artifacts before release 1.1.0 did; retrain it.",
        ),
    )
    return model, d["metadata"], FeatureScaler(lower, d["scaler_upper"]; phase_span = span)
end

"""
    require_scaler(scaler, path) -> FeatureScaler

The feature scaler `scaler` of the model artifact at `path`, as
[`load_model`](@ref) returns it; an `ArgumentError` for an artifact saved
without one, whose features the stages cannot encode.
"""
function require_scaler(scaler::Union{Nothing,FeatureScaler}, path::AbstractString)
    scaler === nothing && throw(
        ArgumentError(
            "the model artifact $path carries no feature scaler; retrain it with the " *
            "current pipeline.",
        ),
    )
    return scaler
end
