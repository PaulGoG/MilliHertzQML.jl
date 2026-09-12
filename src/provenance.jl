# src/provenance.jl — run identifiers and directories, hardware and git
# provenance of every artifact, overwrite-safe writing, memory-safety
# thresholds with pre-flight estimates, and the stage timer.

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
logical core count, total memory, Julia version with the sanitized
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
tree is dirty, and the package version; `"unknown"` values outside a git
repository.
"""
function git_provenance()
    root = project_root()
    # DrWatson warns on a dirty tree at every call; the flag is recorded
    # explicitly below instead.
    commit = try
        with_logger(NullLogger()) do
            something(DrWatson.gitdescribe(root), "unknown")
        end
    catch
        "unknown"
    end
    dirty = try
        DrWatson.isdirty(root)
    catch
        false
    end
    return Dict{String,Any}(
        "git_commit" => commit,
        "git_dirty" => dirty,
        "package_version" => string(pkgversion(MilliHertzQML)),
    )
end

"""
    provenance() -> Dict{String, Any}

`hardware` and `git` sections of a provenance snapshot, plus the wall-clock
time of writing.
"""
function provenance()
    return Dict{String,Any}(
        "hardware" => hardware_fingerprint(),
        "git" => git_provenance(),
        "written_at" => string(Dates.now()),
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
    training_memory_estimate_gib(n_qubits, n_layers, batch_size) -> Float64

Pre-flight estimate of the memory of one training step: the statevector of
``2^{n}`` complex single-precision amplitudes is copied at every gate
application under the automatic-differentiation tape, ``n_\\mathrm{qubits}
(2 + 2) + n_\\mathrm{qubits}`` gates per layer (feature map, rotations,
CNOT ring), for every sample of the batch, plus the same again for the
adjoint pass.
"""
function training_memory_estimate_gib(
    n_qubits::Integer,
    n_layers::Integer,
    batch_size::Integer,
)
    (n_qubits >= 1 && n_layers >= 1 && batch_size >= 1) ||
        throw(ArgumentError("n_qubits, n_layers, and batch_size must be positive."))
    statevector_bytes = 2.0^n_qubits * 8
    gates_per_layer = 5 * n_qubits
    return 2 * batch_size * n_layers * gates_per_layer * statevector_bytes / 2^30
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
