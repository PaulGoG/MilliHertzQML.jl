include(joinpath(@__DIR__, "activate.jl"))

using Documenter
using DocumenterCitations
using MilliHertzQML

bibliography = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

makedocs(
    sitename = "MilliHertzQML",
    repo = Remotes.GitHub("PaulGoG", "MilliHertzQML.jl"),
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://PaulGoG.github.io/MilliHertzQML.jl",
        size_threshold_ignore = ["api.md"],
    ),
    modules = [MilliHertzQML],
    plugins = [bibliography],
    pages = [
        "Home" => "index.md",
        "Results" => "results.md",
        "Physics & Data" => "physics.md",
        "Quantum Architecture" => "architecture.md",
        "Telemetry Coupling" => "telemetry.md",
        "Sangria Benchmark" => "benchmark.md",
        "API Reference" => "api.md",
        "References" => "references.md",
    ],
)

# Deployment to the gh-pages branch: `main` under dev/, release tags under
# their version and stable/. Outside GitHub Actions this is a no-op.
deploydocs(
    repo = "github.com/PaulGoG/MilliHertzQML.jl.git",
    devbranch = "main",
    push_preview = false,
)
