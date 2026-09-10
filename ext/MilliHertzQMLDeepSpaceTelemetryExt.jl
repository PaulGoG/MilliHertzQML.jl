# ext/MilliHertzQMLDeepSpaceTelemetryExt.jl — adapter of a DeepSpaceTelemetry
# run directory to the consumer interface of src/telemetry.jl, through the
# producer's own API (configuration snapshot, batch metadata, segment
# loader), so that a change of the run-directory contract surfaces as a
# version bump or an API error rather than as silent format drift. The
# consumer only reads.
module MilliHertzQMLDeepSpaceTelemetryExt

using DeepSpaceTelemetry: DeepSpaceTelemetry
using DeepSpaceTelemetry.TelemetryCore: TelemetryCore
using CSV: CSV
using DataFrames: DataFrame, nrow
using Dates: DateTime
using MilliHertzQML
using MilliHertzQML:
    AbstractTelemetryRun,
    RunGeometry,
    BatchRecord,
    ArrivalEvent,
    parse_batch_name,
    batch_rows,
    row_time,
    event_symbol
import MilliHertzQML:
    open_telemetry_run, run_geometry, list_batches, read_batch, arrival_events, run_state

"""
    DeepSpaceTelemetryRun

A DeepSpaceTelemetry run directory opened read-only: its path, the
[`RunGeometry`](@ref) taken from the run's configuration snapshot, and the
snapshot itself.
"""
struct DeepSpaceTelemetryRun <: AbstractTelemetryRun
    run_dir::String
    geometry::RunGeometry
    config::Dict{String,Any}
end

"""
    producer_version(config) -> String

The producer's package version recorded under `[provenance.platform]` of a
configuration snapshot, or `"unknown"`.
"""
function producer_version(config::AbstractDict)
    provenance = get(config, "provenance", Dict{String,Any}())
    platform =
        provenance isa AbstractDict ? get(provenance, "platform", Dict{String,Any}()) :
        Dict{String,Any}()
    version =
        platform isa AbstractDict ? get(platform, "package_version", "unknown") : "unknown"
    return String(string(version))
end

function open_telemetry_run(
    run_dir::AbstractString;
    producer_compat::AbstractString = "1.0",
)
    isdir(run_dir) || throw(ArgumentError("run directory not found: $run_dir"))
    any(
        isfile(joinpath(run_dir, s)) for s in ("RUN_ACTIVE", "RUN_COMPLETE", "RUN_ABORTED")
    ) || throw(ArgumentError("$run_dir carries no lifecycle sentinel; not a producer run."))
    isfile(joinpath(run_dir, "config_snapshot.toml")) ||
        throw(ArgumentError("$run_dir holds no config_snapshot.toml."))
    config = TelemetryCore.load_run_config(String(run_dir))
    physics = get(config, "physics", Dict{String,Any}())
    simulation = get(config, "simulation", Dict{String,Any}())
    for (sec, key) in (
        (physics, "sample_rate"),
        (physics, "segment_duration_sec"),
        (physics, "batch_size"),
        (simulation, "start_sim_time"),
    )
        haskey(sec, key) ||
            throw(ArgumentError("the configuration snapshot of $run_dir lacks $key."))
    end
    version = producer_version(config)
    if version == "unknown"
        @warn "the run records no producer version; compatibility not checked" run_dir
    else
        VersionNumber(version) >= VersionNumber(producer_compat) || throw(
            ArgumentError(
                "producer version $version of $run_dir is below the accepted " *
                "producer_compat = $producer_compat.",
            ),
        )
    end
    # The producer records the ingested external payload under [provenance];
    # windows beyond its rows would score the zero-padded stream.
    provenance = get(config, "provenance", Dict{String,Any}())
    payload_rows =
        provenance isa AbstractDict ? get(provenance, "external_data_rows", 0) : 0
    payload_rows = payload_rows isa Real ? Int(payload_rows) : 0
    geometry = RunGeometry(
        physics["sample_rate"],
        physics["segment_duration_sec"],
        physics["batch_size"],
        DateTime(String(simulation["start_sim_time"])),
        version;
        payload_rows = max(payload_rows, 0),
    )
    return DeepSpaceTelemetryRun(String(run_dir), geometry, config)
end

run_geometry(run::DeepSpaceTelemetryRun) = run.geometry

function run_state(run::DeepSpaceTelemetryRun)
    isfile(joinpath(run.run_dir, "RUN_COMPLETE")) && return :complete
    isfile(joinpath(run.run_dir, "RUN_ABORTED")) && return :aborted
    return :active
end

function list_batches(run::DeepSpaceTelemetryRun)
    records = BatchRecord[]
    P = run.geometry.points_per_batch
    for (sub, state) in (("ground", :ground), ("lost", :lost))
        dir = joinpath(run.run_dir, sub)
        isdir(dir) || continue
        for name in readdir(dir)
            batch_dir = joinpath(dir, name)
            isdir(batch_dir) || continue
            k, live = try
                parse_batch_name(name)
            catch
                continue      # a foreign directory
            end
            meta = TelemetryCore.read_batch_metadata(batch_dir)
            rows = batch_rows(k, P)
            epoch =
                haskey(meta, "content_epoch") ?
                something(
                    tryparse(DateTime, String(meta["content_epoch"])),
                    row_time(run.geometry, first(rows)),
                ) : row_time(run.geometry, first(rows))
            batch_state =
                state == :ground && isfile(joinpath(batch_dir, "PRUNED")) ? :pruned : state
            push!(records, BatchRecord(name, k, live, rows, epoch, batch_state))
        end
    end
    sort!(records; by = r -> r.index)
    return records
end

function read_batch(run::DeepSpaceTelemetryRun, name::AbstractString)
    batch_dir = joinpath(run.run_dir, "ground", name)
    isdir(batch_dir) ||
        throw(ArgumentError("batch $name is not on the ground of $(run.run_dir)."))
    isfile(joinpath(batch_dir, "PRUNED")) &&
        throw(ArgumentError("the payload of batch $name has been pruned."))
    files = filter(f -> occursin(r"^seg_\d+\.csv$", f), readdir(batch_dir))
    isempty(files) && throw(ArgumentError("batch $name holds no segment files."))
    sort!(files; by = f -> parse(Int, match(r"^seg_(\d+)\.csv$", f)[1]))
    meta = TelemetryCore.read_batch_metadata(batch_dir)
    if haskey(meta, "segment_count")
        Int(meta["segment_count"]) == length(files) || throw(
            ArgumentError(
                "batch $name declares $(meta["segment_count"]) segments but holds $(length(files)).",
            ),
        )
    end
    samples = Float32[]
    for f in files
        segment = TelemetryCore.load_segment(joinpath(batch_dir, f))
        append!(samples, segment.data)
    end
    return samples
end

function arrival_events(run::DeepSpaceTelemetryRun)
    path = joinpath(run.run_dir, "events_rx.csv")
    isfile(path) || return ArrivalEvent[]
    table = CSV.read(path, DataFrame)
    nrow(table) == 0 && return ArrivalEvent[]
    for column in ("SimTime", "Batch", "Event", "Attempt")
        column in names(table) || throw(
            ArgumentError("events_rx.csv of $(run.run_dir) lacks the column $column."),
        )
    end
    events = ArrivalEvent[]
    for row in eachrow(table)
        sim_time = row.SimTime isa DateTime ? row.SimTime : DateTime(String(row.SimTime))
        attempt =
            row.Attempt isa Integer ? Int(row.Attempt) :
            something(tryparse(Int, string(row.Attempt)), 0)
        push!(
            events,
            ArrivalEvent(
                sim_time,
                String(row.Batch),
                event_symbol(String(row.Event)),
                attempt,
            ),
        )
    end
    sort!(events; by = e -> e.sim_time, alg = Base.Sort.MergeSort)   # stable
    return events
end

end # module
