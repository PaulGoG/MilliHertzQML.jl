# ext/MilliHertzQMLDeepSpaceTelemetryExt.jl — adapter of a DeepSpaceTelemetry
# run directory to the consumer interface of src/StreamingInference/telemetry.jl, through the
# producer's own API (configuration snapshot, batch metadata, segment
# loader), so that a change of the run-directory contract surfaces as a
# version bump or an API error rather than as silent format drift. The
# consumer only reads.
module MilliHertzQMLDeepSpaceTelemetryExt

using DeepSpaceTelemetry: DeepSpaceTelemetry
using DeepSpaceTelemetry.TelemetryCore: TelemetryCore
using CSV: CSV
using DataFrames: DataFrame, nrow
using Dates: DateTime, Millisecond
using MilliHertzQML
using MilliHertzQML:
    AbstractTelemetryRun,
    RunGeometry,
    BatchRecord,
    ArrivalEvent,
    parse_batch_name,
    batch_rows,
    time_row,
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

"""
    payload_origin(config, start_sim_time) -> DateTime

Mission time of payload row 1: `[provenance] payload_origin` where the
producer records it (from 2.1.0), otherwise `start_sim_time` less
`[simulation] initial_downtime_days`, where earlier producers anchored the
instrument.
"""
function payload_origin(config::AbstractDict, start_sim_time::DateTime)
    provenance = get(config, "provenance", Dict{String,Any}())
    if provenance isa AbstractDict && haskey(provenance, "payload_origin")
        origin = tryparse(DateTime, String(string(provenance["payload_origin"])))
        origin === nothing && throw(
            ArgumentError(
                "payload_origin = $(repr(provenance["payload_origin"])) is not an ISO-8601 datetime.",
            ),
        )
        return origin
    end
    simulation = get(config, "simulation", Dict{String,Any}())
    downtime = simulation isa AbstractDict ? get(simulation, "initial_downtime_days", 0) : 0
    downtime isa Real ||
        throw(ArgumentError("initial_downtime_days = $(repr(downtime)) is not numeric."))
    return start_sim_time - Millisecond(round(Int, downtime * 86_400_000))
end

"""
    check_payload_alignment(run_dir, config, version)

Refuses an external-payload run of DeepSpaceTelemetry 2.0.1 or earlier
that holds a scheduled generation gap or an emitter restart. Those
producers read the payload sequentially: after a `SCHEDULED` gap the
content epoch jumps to the gap end while the payload resumes where it
stopped, and a restarted emitter reads it again from row 1, so every later
batch carries rows other than those of its content epoch. Fixed in
DeepSpaceTelemetry 2.1.0; a run of unknown version is treated as affected.
"""
function check_payload_alignment(run_dir::AbstractString, config::AbstractDict, version)
    physics = get(config, "physics", Dict{String,Any}())
    get(physics, "data_source", "synthetic") == "external" || return nothing
    version != "unknown" && VersionNumber(version) >= v"2.1.0" && return nothing
    first_affected = nothing
    tx = joinpath(run_dir, "events_tx.csv")
    if isfile(tx)
        table = CSV.read(tx, DataFrame; types = String)
        for row in eachrow(table)
            row.Batch in ("SCHEDULED", "STREAM") && row.Event == "gap_start" || continue
            first_affected = something(first_affected, row.SimTime)
            break
        end
    end
    components = joinpath(run_dir, "component_events.csv")
    if isfile(components)
        table = CSV.read(components, DataFrame; types = String)
        for row in eachrow(table)
            row.Component == "emitter" && row.Event == "restart" || continue
            first_affected =
                first_affected === nothing ? row.SimTime : min(first_affected, row.SimTime)
            break
        end
    end
    first_affected === nothing && return nothing
    throw(
        ArgumentError(
            "$run_dir is an external-payload run of DeepSpaceTelemetry $version with a " *
            "generation gap or an emitter restart (first at $first_affected); producers " *
            "up to 2.0.1 misplace the payload from that instant on. Regenerate the run " *
            "with DeepSpaceTelemetry 2.1.0 or later.",
        ),
    )
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
    check_payload_alignment(run_dir, config, version)
    # Rows are counted from the mission epoch, as the payload export writes
    # them; a producer that anchors payload row 1 elsewhere (an initial
    # downtime) would shift every row against the event tables.
    start = DateTime(String(simulation["start_sim_time"]))
    origin = payload_origin(config, start)
    origin == start || throw(
        ArgumentError(
            "payload row 1 of $run_dir lies at $origin, not at the mission epoch $start " *
            "(initial downtime); the consumer counts payload rows from the mission epoch.",
        ),
    )
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
        start,
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
            catch e
                e isa ArgumentError || rethrow()
                continue      # a foreign directory
            end
            meta = TelemetryCore.read_batch_metadata(batch_dir)
            # The batch's first row is the payload row the producer stamps on
            # it (from 2.1.0), or else the row of its content epoch.
            # Reconstructing it from the index is only correct while the
            # producer stores every batch it produces; when its recorder
            # overflows it discards production and keeps numbering what it
            # stores, so the index drifts behind the content.
            stamped =
                haskey(meta, "content_epoch") ?
                tryparse(DateTime, String(meta["content_epoch"])) : nothing
            rows = batch_rows(k, P)
            if haskey(meta, "payload_row")
                row0 = Int(meta["payload_row"])
                stamped === nothing ||
                    time_row(run.geometry, stamped) == row0 ||
                    throw(
                        ArgumentError(
                            "batch $name carries payload_row $row0 but content epoch " *
                            "$stamped, which is row $(time_row(run.geometry, stamped)).",
                        ),
                    )
                row0 >= 1 || throw(ArgumentError("batch $name carries payload_row $row0."))
                rows = row0:(row0+P-1)
            elseif stamped !== nothing
                row0 = time_row(run.geometry, stamped)
                if row0 >= 1
                    rows = row0:(row0+P-1)
                else
                    @warn "a batch is stamped before the mission epoch; falling back " *
                          "to its index for the payload rows." batch = name content_epoch =
                        stamped
                end
            end
            epoch = stamped === nothing ? row_time(run.geometry, first(rows)) : stamped
            batch_state =
                state == :ground && isfile(joinpath(batch_dir, "PRUNED")) ? :pruned : state
            push!(records, BatchRecord(name, k, live, rows, epoch, batch_state))
        end
    end
    sort!(records; by = r -> r.index)
    # A stored index that has fallen behind the content means the producer
    # discarded production: the record then has permanent gaps that no arrival
    # will fill, which the coverage handles; the drift is reported.
    drifted = count(r -> first(r.rows) != first(batch_rows(r.index, P)), records)
    drifted == 0 || @warn "the producer discarded production: the stored batch index " *
          "no longer tracks the payload rows, so the record carries permanent gaps." batches_affected =
        drifted total_batches = length(records)
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
