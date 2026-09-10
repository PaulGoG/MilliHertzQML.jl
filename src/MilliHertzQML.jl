module MilliHertzQML

using CSV: CSV
using DataFrames: DataFrames, DataFrame
using FFTW: rfft, rfftfreq
using Flux: Flux
using Functors: Functors
using JLD2: JLD2, jldsave
using Random: Random, AbstractRNG
using Statistics: mean, std
using Yao:
    Yao,
    AbstractBlock,
    H,
    Ry,
    Rz,
    X,
    Z,
    apply,
    chain,
    control,
    dispatch,
    dispatch!,
    expect,
    nparameters,
    put,
    zero_state
using Zygote: Zygote

export VariationalQuantumClassifier
export train_step!, predict_probability, predict, loss_function, accuracy
export load_data, load_features, extract_features
export save_model, load_model

include("model.jl")
include("training.jl")
include("data.jl")
include("persistence.jl")

end # module
