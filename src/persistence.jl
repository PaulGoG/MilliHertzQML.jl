# src/persistence.jl

"""
    save_model(path, model; metadata = Dict{String, Any}(), scaler = nothing)

Persist a classifier of either kind to a JLD2 file. A
`VariationalQuantumClassifier` is stored as its parameter vector plus
hyperparameters (`n_qubits`, `n_layers`), the feature `scaler` fitted on
the training partition (bounds and phase-encoding span; `nothing` when
absent), and an arbitrary `metadata` dictionary (run identifier, seed,
configuration snapshot).

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
its feature dimension `n_features`, hidden units `hidden`, encoding `span`
and parameter vector `params`, the feature `scaler` as the circuit method
stores it, and the `metadata` dictionary. Returns `path`.
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
        scaler_lower = scaler === nothing ? nothing : scaler.lower,
        scaler_upper = scaler === nothing ? nothing : scaler.upper,
        scaler_phase_span = scaler === nothing ? nothing : scaler.phase_span,
        metadata = Dict{String,Any}(metadata),
    )
    return path
end

"""
    load_model(path) -> (model, metadata, scaler)

Load a classifier of either kind persisted by [`save_model`](@ref). An
artifact without a `kind` key, or with `kind = "circuit"`, holds a
`VariationalQuantumClassifier`: the circuit is rebuilt from the stored
hyperparameters and the stored parameter vector dispatched into the ansatz
layers. `kind = "control"` holds a [`ClassicalControl`](@ref); any other
kind is refused with an `ArgumentError`. Returns the feature scaler as
well (`nothing` for artifacts saved without one). An artifact written
before the phase-encoding span was persisted is restored with the full
period ``2\\pi`` its scaler was trained with, and a warning names it: such
a model prepares the same state at both clamp ends and cannot separate a
saturated feature from the noise floor.
"""
function load_model(path::AbstractString)
    d = JLD2.load(path)
    kind = get(d, "kind", "circuit")
    if kind == "control"
        model = ClassicalControl(
            d["n_features"],
            d["hidden"],
            d["span"],
            Vector{Float32}(d["params"]),
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
    scaler = nothing
    if lower !== nothing
        span = get(d, "scaler_phase_span", nothing)
        if span === nothing
            @warn "the model artifact predates the persisted phase-encoding span; its " *
                  "scaler is restored with the full period 2π it was trained with, " *
                  "under which the encoding folds saturated features back onto the " *
                  "noise floor. Retrain to encode on [0, π]." path
            span = Float32(2π)
        end
        scaler = FeatureScaler(lower, d["scaler_upper"]; phase_span = span)
    end
    return model, d["metadata"], scaler
end
