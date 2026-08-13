# scripts/common.jl — shared script preamble: environment activation, path
# resolution against the project root, and validated configuration access.

using Pkg
Pkg.activate(dirname(@__DIR__); io = devnull)
Pkg.instantiate(; io = devnull)

using TOML

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
    cfgget(section, key, default; type = Any, min = nothing, max = nothing)

Read `key` from a configuration `section`, falling back to `default` when the
key is absent. Validates the value against an expected `type` and optional
inclusive bounds, throwing an `ArgumentError` naming the offending key on any
violation. Numeric values are converted to `type` when the conversion is exact.
"""
function cfgget(section::AbstractDict, key::AbstractString, default;
                type::Type = Any, min = nothing, max = nothing)
    value = get(section, key, default)
    if type !== Any && !(value isa type)
        if value isa Real && type <: Real
            value = convert(type, value)
        else
            throw(ArgumentError(
                "configuration key `$key` has value $(repr(value)); expected type $type."))
        end
    end
    min !== nothing && value < min && throw(ArgumentError(
        "configuration key `$key` = $value; must be >= $min."))
    max !== nothing && value > max && throw(ArgumentError(
        "configuration key `$key` = $value; must be <= $max."))
    return value
end

"""
    override(cli_value, cfg_value)

CLI-over-configuration precedence: return `cli_value` unless it is `nothing`.
"""
override(cli_value, cfg_value) = cli_value !== nothing ? cli_value : cfg_value
