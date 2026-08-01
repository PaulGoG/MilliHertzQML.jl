module QuantumGW

using Yao
using Flux
using Functors
using Zygote
using CSV
using DataFrames
using MLUtils
using Statistics
using LinearAlgebra
using Random
using FFTW

# Exports
export VariationalQuantumClassifier
export train_step!, predict_probability, predict, loss_function, accuracy
export load_data, extract_features

# Includes
include("model.jl")
include("training.jl")
include("data.jl")

end # module