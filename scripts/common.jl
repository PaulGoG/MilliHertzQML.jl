# scripts/common.jl — shared script preamble: activation of the script environment
# (scripts/Project.toml, package consumed by path), path resolution against the
# project root, and validated configuration access.

using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using TOML
using InteractiveUtils, LinearAlgebra, Distributed

const PROJECT_ROOT = dirname(@__DIR__)

"""
    resolvepath(p)

Resolve `p` against the project root unless it is already absolute.
"""
resolvepath(p::AbstractString) = isabspath(p) ? p : joinpath(PROJECT_ROOT, p)

"""
    rootrelative(p)

Express `p` relative to the project root when it lies inside it; otherwise
return the absolute path unchanged. Used when persisting paths in provenance
snapshots so that run artifacts remain portable across machines.
"""
function rootrelative(p::AbstractString)
    ap = abspath(p)
    return startswith(ap, PROJECT_ROOT) ? relpath(ap, PROJECT_ROOT) : ap
end

"""
    load_config(path) -> Dict{String, Any}

Parse the TOML configuration at `path`, failing fast when the file is absent.
"""
function load_config(path::AbstractString)
    isfile(path) || throw(ArgumentError("configuration file not found: $path"))
    return TOML.parsefile(path)
end

"""
    cfgget(section, key, default; type = Any, min = nothing, max = nothing, choices = nothing)

Read `key` from a configuration `section`, falling back to `default` when the
key is absent. Validates the value against an expected `type`, optional
inclusive bounds, and an optional set of admissible `choices`, throwing an
`ArgumentError` naming the offending key on any violation. Numeric values are
converted to `type` when the conversion is exact.
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

CLI-over-configuration precedence: return `cli_value` unless it is `nothing`.
"""
override(cli_value, cfg_value) = cli_value !== nothing ? cli_value : cfg_value

"""
    pipeline_paths(config) -> NamedTuple

Output roots of the pipeline from the `[paths]` section — `inputs`, `models`,
`plots`, `results` — resolved against the project root. Absent keys fall back
to the standard tree (`data/inputs`, `models`, `data/outputs/plots`,
`data/outputs/results`).
"""
function pipeline_paths(config::AbstractDict)
    p = get(config, "paths", Dict{String,Any}())
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
    hardware_fingerprint() -> Dict{String, Any}

Collect the platform fingerprint recorded in run provenance snapshots:
hostname, OS kernel, CPU model and logical core count, total memory, Julia
version with full `versioninfo()` output, and thread/worker counts (Julia
threads, BLAS threads, `Distributed` workers). Together with the
configuration snapshot this makes every result attributable to
configuration, code version, and hardware. GPU fields are to be appended
once a functional GPU backend is part of the pipeline.
"""
function hardware_fingerprint()
    cpu = Sys.cpu_info()
    return Dict{String,Any}(
        "hostname" => gethostname(),
        "kernel" => string(Sys.KERNEL),
        "julia_version" => string(VERSION),
        "versioninfo" => sprint(InteractiveUtils.versioninfo),
        "cpu_model" => isempty(cpu) ? "unknown" : first(cpu).model,
        "cpu_threads_logical" => Sys.CPU_THREADS,
        "total_memory_gib" => round(Sys.total_memory() / 2^30; digits = 2),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "distributed_workers" => Distributed.nworkers(),
    )
end
