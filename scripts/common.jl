# scripts/common.jl — activation of the script environment (scripts/Project.toml,
# package consumed by path). Configuration access, path resolution, and
# provenance live in the package (src/StreamingInference/config.jl and provenance.jl).

include(joinpath(@__DIR__, "activate.jl"))

using MilliHertzQML

# The default configuration of the scripts, and the configuration a script
# was given (its first positional argument), whose root the stage runs in.
const DEFAULT_CONFIG = joinpath(dirname(@__DIR__), "configs", "default.toml")
config_argument() =
    isempty(ARGS) || startswith(first(ARGS), "-") ? DEFAULT_CONFIG : first(ARGS)
