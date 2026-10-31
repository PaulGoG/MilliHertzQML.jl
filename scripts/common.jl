# scripts/common.jl — activation of the script environment (scripts/Project.toml,
# package consumed by path), the default configuration of the scripts, and
# the score axis of the classifier figures.

include(joinpath(@__DIR__, "activate.jl"))

using MilliHertzQML

# The default configuration of the scripts; a script runs its stage in the
# root of the configuration it was given (`config_argument`).
const DEFAULT_CONFIG = joinpath(dirname(@__DIR__), "configs", "default.toml")

# Score axis of the classifier in the trace, alert and replay figures: the
# figures of StreamingInference.jl label a generic score on the range of the
# data unless told otherwise.
const CLASSIFIER_SCORE_AXIS = (
    score_label = "MBHB probability",
    score_name = "Classifier output",
    score_range = (0.0, 1.0),
)
