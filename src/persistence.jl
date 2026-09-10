# src/persistence.jl

"""
    save_model(path, model; metadata = Dict{String, Any}(), scaler = nothing)

Persist a `VariationalQuantumClassifier` to a JLD2 file as its parameter
vector plus hyperparameters (`n_qubits`, `n_layers`), the feature `scaler`
fitted on the training partition (bounds only; `nothing` when absent), and
an arbitrary `metadata` dictionary (run identifier, seed, configuration
snapshot).

The circuit itself is not serialized; `load_model` rebuilds it from the
hyperparameters, keeping artifacts robust across package versions.
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
        metadata = Dict{String,Any}(metadata),
    )
    return path
end

"""
    load_model(path) -> (model, metadata, scaler)

Load a `VariationalQuantumClassifier` persisted by [`save_model`](@ref).
Rebuilds the circuit from the stored hyperparameters, dispatches the stored
parameter vector into the ansatz layers, and returns the feature scaler
(`nothing` for artifacts saved without one).
"""
function load_model(path::AbstractString)
    d = JLD2.load(path)
    model = VariationalQuantumClassifier(d["n_qubits"], d["n_layers"])
    model.params = Vector{Float32}(d["params"])
    dispatch_params!(model)
    lower = get(d, "scaler_lower", nothing)
    scaler = lower === nothing ? nothing : FeatureScaler(lower, d["scaler_upper"])
    return model, d["metadata"], scaler
end
