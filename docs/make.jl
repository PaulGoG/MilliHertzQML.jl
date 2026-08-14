using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using Documenter
using MilliHertzQML

makedocs(
    sitename = "MilliHertzQML",
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls = false,
        size_threshold_ignore = ["api.md"],
    ),
    modules = [MilliHertzQML],
    pages = [
        "Home" => "index.md",
        "Physics & Data" => "physics.md",
        "Quantum Architecture" => "architecture.md",
        "API Reference" => "api.md",
    ]
)