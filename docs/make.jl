using Documenter
using QuantumGW

makedocs(
    sitename = "QuantumGW",
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls = false,
        size_threshold_ignore = ["api.md"],
    ),
    modules = [QuantumGW],
    pages = [
        "Home" => "index.md",
        "Physics & Data" => "physics.md",
        "Quantum Architecture" => "architecture.md",
        "API Reference" => "api.md",
    ]
)