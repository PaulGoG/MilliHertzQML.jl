# src/persistence.jl

"""
    save_model(path, model; metadata = Dict{String, Any}())

Persist a `VariationalQuantumClassifier` to a JLD2 file as its parameter
vector plus hyperparameters (`n_qubits`, `n_layers`), together with an
arbitrary `metadata` dictionary (run identifier, seed, configuration
snapshot, decision threshold).

The circuit itself is not serialized; `load_model` rebuilds it from the
hyperparameters, keeping artifacts robust across package versions.
"""
function save_model(
    path::AbstractString,
    model::VariationalQuantumClassifier;
    metadata::AbstractDict = Dict{String,Any}(),
)
    jldsave(
        path;
        n_qubits = model.n_qubits,
        n_layers = model.n_layers,
        params = model.params,
        metadata = Dict{String,Any}(metadata),
    )
    return path
end

"""
    load_model(path) -> (model, metadata)

Load a `VariationalQuantumClassifier` persisted by [`save_model`](@ref).
Rebuilds the circuit from the stored hyperparameters and dispatches the
stored parameter vector into the ansatz layers.
"""
function load_model(path::AbstractString)
    d = JLD2.load(path)
    model = VariationalQuantumClassifier(d["n_qubits"], d["n_layers"])
    model.params = Vector{Float32}(d["params"])
    dispatch_params!(model)
    return model, d["metadata"]
end
