using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using Documenter
using DocumenterCitations
using MilliHertzQML

bibliography = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

makedocs(
    sitename = "MilliHertzQML",
    remotes = nothing,
    format = Documenter.HTML(prettyurls = false, size_threshold_ignore = ["api.md"]),
    modules = [MilliHertzQML],
    plugins = [bibliography],
    pages = [
        "Home" => "index.md",
        "Physics & Data" => "physics.md",
        "Quantum Architecture" => "architecture.md",
        "Telemetry Coupling" => "telemetry.md",
        "Sangria Benchmark" => "benchmark.md",
        "API Reference" => "api.md",
        "References" => "references.md",
    ],
)
