# Provenance and I/O safety: stage timing, run identity, hardware and
# platform fingerprint, git and manifest snapshots, overwrite-safe writers,
# and the memory pre-flight of a stage.

"""
    TIMER

Package-wide `TimerOutput` accumulating the wall time and allocations of
every pipeline stage (`@timeit TIMER "stage" ...`); [`report_timing`](@ref)
prints its table.
"""
const TIMER = TimerOutput()

"""
    report_timing(io = stdout)

Print the stage-timing table of [`TIMER`](@ref).
"""
function report_timing(io::IO = stdout)
    println(io)
    print_timer(io, TIMER; sortby = :firstexec)
    println(io)
    return nothing
end

"""
    new_run_id() -> String

Eight-character run identifier drawn from a UUID.
"""
new_run_id() = string(uuid4())[1:8]

"""
    machine_id() -> String

Stable anonymous identifier of the host: the first twelve hexadecimal
characters of the SHA-256 digest of its name. Two runs on the same
machine share it and runs on different machines do not, which is what
provenance needs, while the machine's name — which is personal data in a
published artifact — is not recoverable from it.
"""
machine_id() = bytes2hex(sha256(gethostname()))[1:12]

"""
    sanitized_versioninfo() -> String

`InteractiveUtils.versioninfo()` output with the user's home directory
replaced by `~`. The `Environment:` block echoes every `JULIA_*`
variable, several of which customarily hold paths under the home
directory and with it the account name.
"""
function sanitized_versioninfo()
    text = sprint(InteractiveUtils.versioninfo)
    home = homedir()
    return isempty(home) ? text : replace(text, home => "~")
end

"""
    hardware_fingerprint() -> Dict{String, Any}

Platform fingerprint recorded in run provenance snapshots: an anonymous
machine identifier ([`machine_id`](@ref)), OS kernel, CPU model and
logical core count, total memory, Julia version with the sanitised
`versioninfo()` output ([`sanitized_versioninfo`](@ref)), and
thread/worker counts (Julia threads, BLAS threads, `Distributed`
workers). Together with the configuration snapshot and the git
description this makes every result attributable to configuration, code
version, and hardware. The host's name never enters a snapshot. GPU
fields are to be appended once a functional GPU backend is part of the
pipeline.
"""
function hardware_fingerprint()
    cpu = Sys.cpu_info()
    return Dict{String,Any}(
        "machine_id" => machine_id(),
        "kernel" => string(Sys.KERNEL),
        "julia_version" => string(VERSION),
        "versioninfo" => sanitized_versioninfo(),
        "cpu_model" => isempty(cpu) ? "unknown" : first(cpu).model,
        "cpu_threads_logical" => Sys.CPU_THREADS,
        "total_memory_gib" => round(Sys.total_memory() / 2^30; digits = 2),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "distributed_workers" => Distributed.nworkers(),
    )
end

"""
    git_provenance() -> Dict{String, Any}

Git description of the package tree (`DrWatson.gitdescribe`), whether the
tree is dirty, and the package version; `"unknown"` for the commit and
`true` for the dirty flag when the tree's state cannot be established
(outside a git repository, or without git).
"""
function git_provenance()
    root = project_root()
    # DrWatson warns on a dirty tree at every call; the flag is recorded
    # explicitly below instead.
    commit = try
        with_logger(NullLogger()) do
            something(DrWatson.gitdescribe(root), "unknown")
        end
    catch e
        e isa InterruptException && rethrow()
        "unknown"
    end
    # A tree whose state cannot be established is not recorded as clean.
    dirty = commit == "unknown" ? true : try
        DrWatson.isdirty(root)
    catch e
        e isa InterruptException && rethrow()
        true
    end
    return Dict{String,Any}(
        "git_commit" => commit,
        "git_dirty" => dirty,
        "package_version" => string(pkgversion(@__MODULE__)),
    )
end

"""
    active_manifest_path() -> Union{Nothing, String}

Path of the Manifest resolved for the active project, or `nothing` when the
active project has none on disk. Candidates are searched in the directory
of `Base.active_project()` in the order Julia itself applies: the
version-specific `Manifest-v<major>.<minor>.toml`, `JuliaManifest.toml`,
`Manifest.toml`.
"""
function active_manifest_path()
    project = Base.active_project()
    project === nothing && return nothing
    dir = dirname(project)
    candidates = (
        "Manifest-v$(VERSION.major).$(VERSION.minor).toml",
        "JuliaManifest.toml",
        "Manifest.toml",
    )
    for name in candidates
        path = joinpath(dir, name)
        isfile(path) && return path
    end
    return nothing
end

"""
    manifest_sha256() -> String

SHA-256 digest (hexadecimal) of the active Manifest
([`active_manifest_path`](@ref)), or `"unknown"` when there is none. The
digest identifies the resolved dependency state of a run without carrying
the file into every record.
"""
function manifest_sha256()
    path = active_manifest_path()
    path === nothing && return "unknown"
    return bytes2hex(open(sha256, path))
end

"""
    snapshot_manifest(dir) -> Union{Nothing, String}

Copy the active Manifest ([`active_manifest_path`](@ref)) to
`<dir>/manifest_snapshot.toml`, backing up an existing snapshot first
([`backup_existing!`](@ref)), and return the path written; `nothing`, with
a warning, when the active project has no Manifest. Manifests are not
tracked in the repository, so this copy is what pins the resolved
environment of a run.
"""
function snapshot_manifest(dir::AbstractString)
    source = active_manifest_path()
    if source === nothing
        @warn "no Manifest found for the active project; the resolved environment of this run is not recorded." active_project =
            Base.active_project()
        return nothing
    end
    target = joinpath(dir, "manifest_snapshot.toml")
    mkpath(dir)
    backup_existing!(target)
    cp(source, target)
    return String(target)
end

"""
    provenance() -> Dict{String, Any}

`hardware`, `git`, and `environment` sections of a provenance snapshot, plus
the wall-clock time of writing. `environment` names the active project and
the digest of its Manifest ([`manifest_sha256`](@ref)).
"""
function provenance()
    return Dict{String,Any}(
        "hardware" => hardware_fingerprint(),
        "git" => git_provenance(),
        "environment" => Dict{String,Any}(
            "active_project" =>
                provenance_path(something(Base.active_project(), "unknown")),
            "manifest_sha256" => manifest_sha256(),
        ),
        "written_at" => string(Dates.now()),
    )
end

"""
    content_digest(path) -> String

SHA-256 (hexadecimal) of the content of the file at `path`: the identity
of an input product whatever its location or modification time.
"""
content_digest(path::AbstractString) = bytes2hex(open(sha256, path))

"""
    parameter_digest(parameters) -> String

SHA-256 (hexadecimal) of the key-sorted TOML rendering of a parameter
dictionary: the identity of the parameters of a product, the same in
every process, on every machine and under every Julia version (unlike
`Base.hash`).
"""
function parameter_digest(parameters::AbstractDict)
    io = IOBuffer()
    TOML.print(io, parameters; sorted = true)
    return bytes2hex(sha256(take!(io)))
end

"""
    product_table(kind; channels, parents = Dict{String,Any}(), schema = 1) -> Dict{String,Any}

The `[product]` table of a sidecar: the `kind` of product (`"features"`,
`"labels"`, …), the `channels` it was made from (a channel-set token such
as `"A"`), the `schema` version of its tables, and the content digests
([`content_digest`](@ref)) of its `parents`, so that a product can be
traced to its inputs and refused when they do not match.
"""
function product_table(
    kind::AbstractString;
    channels::AbstractString,
    parents::AbstractDict = Dict{String,Any}(),
    schema::Integer = 1,
)
    schema >= 1 || throw(ArgumentError("schema must be at least 1."))
    return Dict{String,Any}(
        "kind" => String(kind),
        "channels" => String(channels),
        "schema" => Int(schema),
        "parents" => Dict{String,Any}(String(k) => v for (k, v) in parents),
    )
end

"""
    backup_existing!(path) -> Union{Nothing, String}

Move an existing file at `path` to `<stem>_#k<ext>` with the first free
`k ≥ 1` (the `safesave` convention of DrWatson), so that a new write at
`path` never destroys a previous result. Returns the backup path, or
`nothing` when there was nothing to move.
"""
function backup_existing!(path::AbstractString)
    isfile(path) || return nothing
    stem, ext = splitext(path)
    k = 1
    while isfile("$(stem)_#$(k)$(ext)")
        k += 1
    end
    backup = "$(stem)_#$(k)$(ext)"
    mv(path, backup)
    @info "existing file moved to a backup" path = path backup = backup
    return backup
end

"""
    write_toml(path, data; safe = true, tag = true)

Write the dictionary `data` as TOML at `path`, backing up an existing file
first ([`backup_existing!`](@ref)) when `safe`, and merging the
[`provenance`](@ref) sections when `tag`. Returns `path`.
"""
function write_toml(
    path::AbstractString,
    data::AbstractDict;
    safe::Bool = true,
    tag::Bool = true,
)
    payload = Dict{String,Any}(data)
    tag && merge!(payload, provenance())
    safe && backup_existing!(path)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, payload)
    end
    return String(path)
end

"""
    write_csv(path, table; safe = true)

Write `table` with `CSV.write` at `path`, backing up an existing file first
when `safe`. Returns `path`.
"""
function write_csv(path::AbstractString, table; safe::Bool = true)
    safe && backup_existing!(path)
    mkpath(dirname(path))
    CSV.write(path, table)
    return String(path)
end

"""
    resource_settings(config) -> NamedTuple

Memory-safety thresholds of the `[resources]` section in GiB:
`max_memory_gib` (a stage whose estimate exceeds it refuses to start;
default half of the machine's memory) and `warn_memory_gib` (a warning;
default a quarter).
"""
function resource_settings(config::AbstractDict)
    r = section(config, "resources")
    total = Sys.total_memory() / 2^30
    max_gib = cfgget(
        r,
        "max_memory_gib",
        round(total / 2; digits = 2);
        type = Float64,
        min = 1e-3,
    )
    warn_gib = cfgget(
        r,
        "warn_memory_gib",
        round(total / 4; digits = 2);
        type = Float64,
        min = 0.0,
    )
    warn_gib <= max_gib || throw(
        ArgumentError("warn_memory_gib = $warn_gib exceeds max_memory_gib = $max_gib."),
    )
    return (max_memory_gib = max_gib, warn_memory_gib = warn_gib, total_memory_gib = total)
end

"""
    record_memory_estimate_gib(n_samples; copies = 6) -> Float64

Pre-flight estimate of the memory of processing a record of `n_samples`
double-precision samples through the high-pass, whitening, and windowing
chain, which holds about `copies` record-length arrays (time series, its
transform, and intermediates).
"""
function record_memory_estimate_gib(n_samples::Integer; copies::Integer = 6)
    n_samples >= 0 || throw(ArgumentError("n_samples must be non-negative."))
    return copies * n_samples * 8 / 2^30
end

"""
    check_memory(estimate_gib, resources; stage)

Enforce the `[resources]` thresholds on a pre-flight estimate: throw an
`ArgumentError` naming `stage` above `max_memory_gib`, warn above
`warn_memory_gib`, otherwise log the estimate. Returns `estimate_gib`.
"""
function check_memory(estimate_gib::Real, resources::NamedTuple; stage::AbstractString)
    if estimate_gib > resources.max_memory_gib
        throw(
            ArgumentError(
                "$stage needs an estimated $(round(estimate_gib; digits = 2)) GiB, above " *
                "[resources] max_memory_gib = $(resources.max_memory_gib) GiB; reduce the " *
                "configuration or raise the threshold.",
            ),
        )
    elseif estimate_gib > resources.warn_memory_gib
        @warn "$stage memory estimate above the warning threshold" estimate_gib =
            round(estimate_gib; digits = 2) warn_memory_gib = resources.warn_memory_gib
    else
        @info "$stage memory estimate" estimate_gib = round(estimate_gib; digits = 3)
    end
    return estimate_gib
end
