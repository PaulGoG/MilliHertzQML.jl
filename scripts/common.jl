# scripts/common.jl — activation of the script environment (scripts/Project.toml,
# package consumed by path). Configuration access, path resolution, and
# provenance live in StreamingInference.jl (config.jl and provenance.jl).

include(joinpath(@__DIR__, "activate.jl"))

using MilliHertzQML

# The default configuration of the scripts, and the configuration a script
# was given (its first positional argument), whose root the stage runs in.
const DEFAULT_CONFIG = joinpath(dirname(@__DIR__), "configs", "default.toml")
config_argument() =
    isempty(ARGS) || startswith(first(ARGS), "-") ? DEFAULT_CONFIG : first(ARGS)

# Score axis of the classifier in the trace, alert and replay figures: the
# figures of StreamingInference.jl label a generic score on the range of the
# data unless told otherwise.
const CLASSIFIER_SCORE_AXIS = (
    score_label = "MBHB probability",
    score_name = "Classifier output",
    score_range = (0.0, 1.0),
)
