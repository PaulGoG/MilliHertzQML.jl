# TOML configuration: loading, validated key access, path resolution against
# the pipeline root, and the settings of the sections every pipeline shares.

"""
    project_root() -> String

Directory of the package (the pipeline root): the parent of `src/`. Every
relative path of a configuration resolves against it, whichever
environment (`scripts/`, `test/`, `docs/`, `bench/`) is active.
"""
project_root() = pkgdir(@__MODULE__)::String

"""
    resolvepath(p) -> String

`p` resolved against [`project_root`](@ref) unless it is already absolute.
"""
resolvepath(p::AbstractString) = isabspath(p) ? String(p) : joinpath(project_root(), p)

"""
    rootrelative(p) -> String

`p` expressed relative to [`project_root`](@ref) when it lies inside it,
otherwise the absolute path unchanged; used when persisting paths in
provenance snapshots so that run artifacts stay portable across machines.
"""
function rootrelative(p::AbstractString)
    ap = abspath(p)
    root = project_root()
    return startswith(ap, root) ? relpath(ap, root) : ap
end

"""
    provenance_path(p) -> String

`p` as recorded in a provenance snapshot: relative to
[`project_root`](@ref) when it lies inside it, and otherwise its file
name alone. A path outside the package root belongs to the machine that
ran the stage, not to the run, and carries the account name and the
directory layout of that machine into artifacts that are meant to be
published; the file name is what identifies the input. Use
[`rootrelative`](@ref) instead wherever the recorded path is read back
and resolved rather than only reported.
"""
function provenance_path(p::AbstractString)
    rel = rootrelative(p)
    return isabspath(rel) ? basename(rel) : rel
end

"""
    load_config(path) -> Dict{String, Any}

Parse the TOML configuration at `path`, failing fast when the file is
absent.
"""
function load_config(path::AbstractString)
    isfile(path) || throw(ArgumentError("configuration file not found: $path"))
    return TOML.parsefile(path)
end

"""
    cfgget(section, key, default; type = Any, min = nothing, max = nothing, choices = nothing)

Read `key` from a configuration `section`, falling back to `default` when
the key is absent. Validates the value against an expected `type`, optional
inclusive bounds, and an optional set of admissible `choices`, throwing an
`ArgumentError` naming the offending key on any violation. Numeric values
are converted to `type` when the conversion is exact.
"""
function cfgget(
    section::AbstractDict,
    key::AbstractString,
    default;
    type::Type = Any,
    min = nothing,
    max = nothing,
    choices = nothing,
)
    value = get(section, key, default)
    if type !== Any && !(value isa type)
        if value isa Real && type <: Real
            value = convert(type, value)
        else
            throw(
                ArgumentError(
                    "configuration key `$key` has value $(repr(value)); expected type $type.",
                ),
            )
        end
    end
    min !== nothing &&
        value < min &&
        throw(ArgumentError("configuration key `$key` = $value; must be >= $min."))
    max !== nothing &&
        value > max &&
        throw(ArgumentError("configuration key `$key` = $value; must be <= $max."))
    choices !== nothing &&
        !(value in choices) &&
        throw(
            ArgumentError(
                "configuration key `$key` = $(repr(value)); must be one of " *
                join(repr.(choices), " | ") *
                ".",
            ),
        )
    return value
end

"""
    override(cli_value, cfg_value)

Precedence of an explicit command-line value over the configuration:
`cli_value` unless it is `nothing`.
"""
override(cli_value, cfg_value) = cli_value !== nothing ? cli_value : cfg_value

"""
    section(config, name) -> Dict{String, Any}

The table `name` of `config`, or an empty table when absent.
"""
section(config::AbstractDict, name::AbstractString) =
    get(config, name, Dict{String,Any}())::AbstractDict

"""
    analysis_band(section, key, default) -> Tuple{Float64,Float64}

Two-element ascending positive frequency band [Hz] from the configuration.
"""
function analysis_band(section::AbstractDict, key::AbstractString, default)
    band = cfgget(section, key, default; type = AbstractVector)
    (length(band) == 2 && all(x -> x isa Real, band) && 0 < band[1] < band[2]) || throw(
        ArgumentError(
            "configuration key `$key` = $(repr(band)); expected two ascending positive frequencies [Hz].",
        ),
    )
    return (Float64(band[1]), Float64(band[2]))
end

"""
    pipeline_paths(config) -> NamedTuple

Output roots of the pipeline from the `[paths]` section — `inputs`,
`models`, `plots`, `results` — resolved against the package root. Absent
keys fall back to the standard tree (`data/inputs`, `models`,
`data/outputs/plots`, `data/outputs/results`).
"""
function pipeline_paths(config::AbstractDict)
    p = section(config, "paths")
    return (
        inputs = resolvepath(
            cfgget(p, "inputs", joinpath("data", "inputs"); type = String),
        ),
        models = resolvepath(cfgget(p, "models", "models"; type = String)),
        plots = resolvepath(
            cfgget(p, "plots", joinpath("data", "outputs", "plots"); type = String),
        ),
        results = resolvepath(
            cfgget(p, "results", joinpath("data", "outputs", "results"); type = String),
        ),
    )
end

"""
    inference_settings(config) -> NamedTuple

Validated `[inference]` parameters: feature and label tables, block
selection, and the window geometry used when a feature table has no
sidecar.
"""
function inference_settings(config::AbstractDict)
    i = section(config, "inference")
    return (
        features = resolvepath(
            cfgget(i, "features", "data/inputs/inference_features.csv"; type = String),
        ),
        labels = cfgget(i, "labels", "data/inputs/inference_labels.csv"; type = String),
        block = cfgget(
            i,
            "block",
            "all";
            type = String,
            choices = ("all", "validation", "test"),
        ),
        step_size = cfgget(i, "step_size", 100; type = Int, min = 1),
        sample_rate = cfgget(i, "sample_rate", 0.2; type = Float64, min = 1e-6),
    )
end

"""
    feature_geometry(features_path, config) -> NamedTuple

Window geometry (`window_size`, `step_size`, `sample_rate`, and
`first_window`, the record window index of the table's first row, above 1
when an edge margin was dropped) of a feature table, read from the
sidecar `<stem>.toml` written beside it by the pre-processor; falls back
to the `[preprocessing]` section with a warning when the sidecar is
absent.
"""
function feature_geometry(features_path::AbstractString, config::AbstractDict)
    sidecar = replace(features_path, r"\.csv$" => ".toml")
    sec = section(config, "preprocessing")
    if isfile(sidecar)
        sec = get(TOML.parsefile(sidecar), "features", Dict{String,Any}())
    else
        @warn "no feature sidecar at $sidecar; window geometry taken from [preprocessing]."
    end
    return (
        window_size = cfgget(sec, "window_size", 1000; type = Int, min = 2),
        step_size = cfgget(sec, "step_size", 100; type = Int, min = 1),
        sample_rate = cfgget(sec, "sample_rate", 0.2; type = Float64, min = 1e-6),
        first_window = cfgget(sec, "first_window", 1; type = Int, min = 1),
    )
end
