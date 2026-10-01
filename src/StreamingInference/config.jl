# TOML configuration: loading, validated key access, path resolution against
# the pipeline root, and the settings of the sections every pipeline shares.

"""
    PIPELINE_ROOT

The pipeline root set for the current scope by [`with_pipeline_root`](@ref);
`nothing` outside any.
"""
const PIPELINE_ROOT = ScopedValue{Union{Nothing,String}}(nothing)

"""
    with_pipeline_root(f, root)

Run `f()` with `root` as the pipeline root ([`project_root`](@ref)) of
every path resolved inside it, in this task and the tasks it spawns.
Scripts run their stage inside it, with the root of their configuration
([`config_root`](@ref)).
"""
with_pipeline_root(f, root::AbstractString) = with(f, PIPELINE_ROOT => abspath(root))

"""
    project_root() -> String

The pipeline root: the directory every relative path of a configuration
resolves against and whose repository the provenance records. In order:
the root set by [`with_pipeline_root`](@ref); the environment variable
`STREAMINGINFERENCE_ROOT`; the nearest directory above the active
environment whose `Project.toml` declares a package (the `scripts/`,
`test/`, `docs/` and `bench/` environments of a pipeline resolve to its
repository). Throws an `ArgumentError` when none applies, rather than
resolving against the directory of an installed package.
"""
function project_root()
    root = PIPELINE_ROOT[]
    root === nothing || return root
    variable = get(ENV, "STREAMINGINFERENCE_ROOT", "")
    isempty(variable) || return abspath(variable)
    active = Base.active_project()
    found =
        active === nothing ? nothing : ancestor_with_project(dirname(active); named = true)
    found === nothing && throw(
        ArgumentError(
            "no pipeline root: run inside `with_pipeline_root(root) do … end`, set " *
            "STREAMINGINFERENCE_ROOT, or activate an environment of the pipeline.",
        ),
    )
    return found
end

"""
    ancestor_with_project(dir; named = false) -> Union{Nothing,String}

The nearest of `dir` and its ancestors holding a `Project.toml` (with
`named = true`, one that declares a package `name`), or `nothing`.
"""
function ancestor_with_project(dir::AbstractString; named::Bool = false)
    current = abspath(dir)
    while true
        project = joinpath(current, "Project.toml")
        if isfile(project) && (!named || haskey(TOML.parsefile(project), "name"))
            return current
        end
        parent = dirname(current)
        parent == current && return nothing
        current = parent
    end
end

"""
    config_root(path) -> String

The pipeline root of the configuration file at `path`: its `[paths] root`
(relative to the file's directory) when given, otherwise the nearest
directory above the file holding a `Project.toml`, otherwise
[`project_root`](@ref).
"""
function config_root(path::AbstractString)
    file = abspath(path)
    paths = section(load_config(file), "paths")
    if haskey(paths, "root")
        root = cfgget(paths, "root", "."; type = String)
        resolved = normpath(isabspath(root) ? root : joinpath(dirname(file), root))
        # `normpath` keeps the separator after a trailing `..`
        return length(resolved) > 1 && endswith(resolved, Base.Filesystem.path_separator) ?
               resolved[1:(end-1)] : resolved
    end
    found = ancestor_with_project(dirname(file))
    return found === nothing ? project_root() : found
end

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
    # A path inside the root starts with the root and a separator, so that a
    # sibling directory whose name extends the root's is not taken for it
    inside = ap == root || startswith(ap, joinpath(root, ""))
    return inside ? relpath(ap, root) : ap
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
