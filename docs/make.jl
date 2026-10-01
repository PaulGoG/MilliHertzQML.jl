include(joinpath(@__DIR__, "activate.jl"))

using Documenter
using DocumenterCitations
using DocumenterInterLinks
using MilliHertzQML

bibliography = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)
# References to the two layers resolve against their manuals.
links = InterLinks(
    "StreamingInference" => "https://PaulGoG.github.io/StreamingInference.jl/dev/",
    "MilliHertzBase" => "https://PaulGoG.github.io/MilliHertzBase.jl/dev/",
)
fallbacks = ExternalFallbacks(; automatic = true)

makedocs(
    sitename = "MilliHertzQML",
    repo = Remotes.GitHub("PaulGoG", "MilliHertzQML.jl"),
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://PaulGoG.github.io/MilliHertzQML.jl",
        size_threshold_ignore = ["api.md"],
    ),
    modules = [MilliHertzQML],
    plugins = [bibliography, links, fallbacks],
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
