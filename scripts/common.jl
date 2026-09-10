# scripts/common.jl — activation of the script environment (scripts/Project.toml,
# package consumed by path). Configuration access, path resolution, and
# provenance live in the package (src/config.jl, src/provenance.jl).

using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using MilliHertzQML
