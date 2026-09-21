# scripts/common.jl — activation of the script environment (scripts/Project.toml,
# package consumed by path). Configuration access, path resolution, and
# provenance live in the package (src/config.jl, src/provenance.jl).

include(joinpath(@__DIR__, "activate.jl"))

using MilliHertzQML
