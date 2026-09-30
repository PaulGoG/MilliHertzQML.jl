# Payload-export stage of the telemetry
# coupling: the A channel of an HDF5 TDI product written as the
# single-column amplitude CSV that the telemetry producer ingests gaplessly,
# beside a scenario fragment holding the external-data physics table, the
# mission start time, the coalescence markers of an event catalogue, and the
# label spans from which the consumer rebuilds the point-wise labels.

"""
    samples_per_batch(sample_rate, segment_duration_sec, batch_size) -> Int

Payload rows consumed by one batch of the telemetry producer:
`sample_rate × segment_duration_sec` rows per segment, `batch_size`
segments per batch. The rows per segment must be a whole number, since the
producer cuts the record gaplessly into equal segments; `ArgumentError`
otherwise.
"""
function samples_per_batch(
    sample_rate::Real,
    segment_duration_sec::Real,
    batch_size::Integer,
)
    sample_rate > 0 || throw(ArgumentError("sample_rate = $sample_rate; must be positive."))
    segment_duration_sec > 0 || throw(
        ArgumentError("segment_duration_sec = $segment_duration_sec; must be positive."),
    )
    batch_size >= 1 || throw(ArgumentError("batch_size = $batch_size; must be at least 1."))
    per_segment = sample_rate * segment_duration_sec
    n_segment = round(Int, per_segment)
    (n_segment >= 1 && isapprox(per_segment, n_segment; atol = 1e-6)) || throw(
        ArgumentError(
            "segment_duration_sec = $segment_duration_sec at $sample_rate Hz holds " *
            "$per_segment samples; a segment must hold a whole number of samples.",
        ),
    )
    return n_segment * batch_size
end

"""
    catalog_events(table) -> NamedTuple

Event identifiers, coalescence times [s] in the record's time coordinate,
and label spans of an event catalogue table: the simulator catalogue (columns
`event_id`, `t_c_sec`) or the LDC event table (`event`, `merger_time_s`).
`spans` holds `(label_start_index, label_end_index)` pairs when both
columns are present and is `nothing` otherwise. `ArgumentError` for a table
of neither schema.
"""
function catalog_events(table::DataFrame)
    cols = DataFrames.names(table)
    ids, times = if "event_id" in cols && "t_c_sec" in cols
        table.event_id, table.t_c_sec
    elseif "event" in cols && "merger_time_s" in cols
        table.event, table.merger_time_s
    else
        throw(
            ArgumentError(
                "the event catalog needs the columns `event_id` and `t_c_sec` " *
                "(simulator catalog) or `event` and `merger_time_s` (LDC event table); " *
                "found: " *
                join(cols, ", ") *
                ".",
            ),
        )
    end
    all(x -> x isa Real, times) ||
        throw(ArgumentError("the coalescence times of the catalog must be numeric."))
    spans = if "label_start_index" in cols && "label_end_index" in cols
        [(Int(s), Int(e)) for (s, e) in zip(table.label_start_index, table.label_end_index)]
    else
        nothing
    end
    return (ids = [string(id) for id in ids], times = Float64.(times), spans = spans)
end

"""
    export_telemetry_payload(config; h5_file = nothing, tdi_group = nothing,
                             catalog = nothing, output_prefix = nothing) -> NamedTuple

Payload-export stage of the telemetry coupling. The Michelson variables of
the HDF5 TDI product `h5_file` (default `[preprocessing] h5_file`) at
`tdi_group` (default `[preprocessing] tdi_group`; [`read_tdi`](@ref)) are
combined into the A channel ([`tdi_to_aet`](@ref)), which is written in
single precision as the one-column (`Amplitude`) CSV
`<inputs>/<output_prefix>_payload.csv` that the telemetry producer ingests
gaplessly. Beside it, `<output_prefix>_scenario.toml` holds the scenario
fragment of the producer's external-data mode: `[physics]` with
`data_source = "external"`, the payload path relative to the package root,
the sampling frequency of the file, and the `segment_duration_sec` and
`batch_size` of [`telemetry_settings`](@ref); `[simulation]` with
`start_sim_time`; `[[events.markers]]` with one `{time, label = "mbhb_<id>"}`
entry per event of `catalog` — the simulator catalogue (`event_id`,
`t_c_sec`) or the LDC event table (`event`, `merger_time_s`;
[`catalog_events`](@ref)) — at `start_sim_time` plus the coalescence time
measured from the first record sample, rounded to the millisecond;
`[[labels]]` with the `label_start_index`/`label_end_index` span of every
event when the catalogue carries them (a span with start > end is empty), so
that the consumer can rebuild the point-wise labels; and `[payload]` with
the source, group, row count, catalog, and batch geometry. The hardware and
git provenance of [`write_toml`](@ref) is merged in. Without a catalogue the
marker array is empty and no label array is written.

Payload row ``r`` corresponds to `start_sim_time` + ``(r - 1)/f_s`` and
batch ``k`` to the rows ``[(k - 1)P + 1, kP]`` with ``P`` =
[`samples_per_batch`](@ref). The record must hold at least one full batch,
every coalescence must lie inside it, and every non-empty label span inside
`1:n_rows`; `ArgumentError` otherwise. Existing outputs are backed up
([`backup_existing!`](@ref)); the memory of the record arrays is checked
against `[resources]` ([`check_memory`](@ref)). Timed under `TIMER` as
`"payload export"`.

Returns `(; payload_path, scenario_path, n_rows, sample_rate,
start_sim_time, markers)` with `markers` a vector of
`(label, time::DateTime)` tuples.
"""
function export_telemetry_payload(
    config::AbstractDict;
    h5_file::Union{Nothing,AbstractString} = nothing,
    tdi_group::Union{Nothing,AbstractString} = nothing,
    catalog::Union{Nothing,AbstractString} = nothing,
    output_prefix::Union{Nothing,AbstractString} = nothing,
)
    @timeit TIMER "payload export" begin
        settings = telemetry_settings(config)
        pre = preprocessing_settings(config)
        source = resolvepath(override(h5_file, pre.h5_file))
        group = String(override(tdi_group, pre.tdi_group))
        prefix = String(override(output_prefix, settings.output_prefix))
        catalog_path = catalog === nothing ? nothing : resolvepath(catalog)
        catalog_path === nothing ||
            isfile(catalog_path) ||
            throw(ArgumentError("event catalog not found: $catalog_path"))

        # A channel of the record and the batch geometry it must satisfy
        tdi = read_tdi(source; group = group)
        n_rows = length(tdi.t)
        sample_rate = 1 / tdi.dt
        check_memory(
            record_memory_estimate_gib(n_rows),
            resource_settings(config);
            stage = "payload export",
        )
        A, _, _ = tdi_to_aet(tdi.X, tdi.Y, tdi.Z)
        rows_per_batch = samples_per_batch(
            sample_rate,
            settings.segment_duration_sec,
            settings.batch_size,
        )
        n_rows >= rows_per_batch || throw(
            ArgumentError(
                "the record holds $n_rows samples at $sample_rate Hz, fewer than the " *
                "$rows_per_batch of one batch ($(settings.batch_size) segments of " *
                "$(settings.segment_duration_sec) s).",
            ),
        )
        @info "telemetry payload" source group n_rows sample_rate_hz = sample_rate rows_per_batch n_batches =
            div(n_rows, rows_per_batch)

        # Coalescence markers and label spans from the event catalogue
        markers = NamedTuple{(:label, :time),Tuple{String,Dates.DateTime}}[]
        label_entries = nothing
        if catalog_path !== nothing
            events = catalog_events(CSV.read(catalog_path, DataFrame))
            t_end = (n_rows - 1) / sample_rate
            for (id, t_c) in zip(events.ids, events.times)
                offset = t_c - tdi.t[1]
                0 <= offset <= t_end || throw(
                    ArgumentError(
                        "event $id coalesces $offset s after the first sample, outside " *
                        "the record of $t_end s; the catalog does not belong to $source.",
                    ),
                )
                push!(
                    markers,
                    (
                        label = "mbhb_$id",
                        time = settings.start_sim_time +
                               Dates.Millisecond(round(Int, 1000 * offset)),
                    ),
                )
            end
            if events.spans !== nothing
                label_entries = Dict{String,Any}[]
                for (id, (lo, hi)) in zip(events.ids, events.spans)
                    (lo > hi || (1 <= lo && hi <= n_rows)) || throw(
                        ArgumentError(
                            "label span $lo:$hi of event $id lies outside 1:$n_rows.",
                        ),
                    )
                    push!(
                        label_entries,
                        Dict{String,Any}(
                            "label" => "mbhb_$id",
                            "label_start_index" => lo,
                            "label_end_index" => hi,
                        ),
                    )
                end
            end
            @info "event catalog" catalog_path n_events = length(markers) label_spans =
                events.spans !== nothing
        end

        # Persist: single-column payload, scenario fragment with provenance
        out_dir = pipeline_paths(config).inputs
        payload_path = joinpath(out_dir, "$(prefix)_payload.csv")
        scenario_path = joinpath(out_dir, "$(prefix)_scenario.toml")
        for out in (payload_path, scenario_path), inp in (source, catalog_path)
            inp !== nothing &&
                abspath(out) == abspath(inp) &&
                throw(
                    ArgumentError(
                        "output $out coincides with an input; choose another prefix.",
                    ),
                )
        end
        write_csv(payload_path, DataFrame(Amplitude = Float32.(A)))
        scenario = Dict{String,Any}(
            "physics" => Dict{String,Any}(
                "data_source" => "external",
                "external_data_path" => rootrelative(payload_path),
                "sample_rate" => sample_rate,
                "segment_duration_sec" => settings.segment_duration_sec,
                "batch_size" => settings.batch_size,
            ),
            "simulation" =>
                Dict{String,Any}("start_sim_time" => string(settings.start_sim_time)),
            "events" => Dict{String,Any}(
                "markers" => [
                    Dict{String,Any}("time" => string(m.time), "label" => m.label) for m in markers
                ],
            ),
            "payload" => Dict{String,Any}(
                "source" => provenance_path(source),
                "tdi_group" => group,
                "n_rows" => n_rows,
                "catalog" =>
                    catalog_path === nothing ? "" : provenance_path(catalog_path),
                "samples_per_batch" => rows_per_batch,
                "n_batches" => div(n_rows, rows_per_batch),
            ),
        )
        label_entries === nothing || (scenario["labels"] = label_entries)
        write_toml(scenario_path, scenario; tag = true)
        @info "payload written" payload_path scenario_path n_rows n_markers =
            length(markers)

        (
            payload_path = payload_path,
            scenario_path = scenario_path,
            n_rows = n_rows,
            sample_rate = sample_rate,
            start_sim_time = settings.start_sim_time,
            markers = markers,
        )
    end
end
