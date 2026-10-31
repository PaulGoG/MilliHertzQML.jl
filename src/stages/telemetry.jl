# src/stages/telemetry.jl — telemetry replay stage: a producer run directory
# replayed, or followed live, with the detector of a trained classifier;
# the scored-window table, the alert-latency table and the snapshot. The
# script scripts/infer_telemetry.jl adds only the command line and the
# alert figure.

"""
    replay_classifier(config; model, run_id = new_run_id(), run_dir = nothing,
                      events = nothing, tdi_file = nothing, tdi_group = nothing,
                      tdi_first_row = 1, live = false, link_volume_channels = 1,
                      on_window = nothing) -> NamedTuple

Telemetry stage driven by the `[telemetry]`, `[preprocessing]` and
`[paths]` sections of `config`: the producer run directory `run_dir`
(default `[telemetry] run_dir`) is scored as it arrives by the detector of
the training run whose artifact is `model` ([`detector_from_run`](@ref)),
replayed from its arrival feed ([`replay_run`](@ref)) or, with `live` or
`[telemetry] mode = "live"`, followed until the run ends
([`follow_run`](@ref)). The consumer only reads the run directory.

The channels are those the model was trained on ([`run_channels`](@ref)).
A run that carries them as payload columns is served them; a model of
several channels whose run carried the A channel alone, or a replay given
`tdi_file`, is served the channels of every delivered batch from the TDI
record the payload was exported from ([`ScheduledRecordRun`](@ref)),
`tdi_group` of that file (default: the component groups of
`[preprocessing]`, [`record_groups`](@ref)) from its row `tdi_first_row`
on (a mission over a later stretch of the record). Under `[telemetry]
psd_mode = "trailing"` the whitening PSD is a trailing Welch estimate of
the delivered record ([`TrailingWelch`](@ref)) and `psd_sidecar` is
ignored with a warning; otherwise `psd_sidecar` (default: the training
run's) whitens the record.

With an event table `events` (default `[telemetry] events_csv`; the
merger times, label spans and signal onsets of the record), alerts are
credited from the onset of the detector's channel set
([`mode_events`](@ref)) under the persistence, crediting and confirmation
rules of `[telemetry]` ([`alert_latency_table`](@ref)).
`link_volume_channels`, the number of channels the production rate of the
mission was scaled for, is stated by the caller and recorded.
`on_window`, when given, is called with every committed window record.

Artifacts in `<results>/run_<run_id>`: `telemetry_windows.csv`
([`windows_table`](@ref)), with events `alert_latency.csv`, and the
snapshot `config_telemetry.toml` (section `telemetry`: the run, the
producer version, the channels and payload columns, the record, the
detector threshold, the replay and alert settings, the window and alarm
counts; section `producer`: the producer's run descriptor
[`run_descriptor`](@ref); plus provenance). The results directory must
lie outside the run directory.

Returns `(run_id, results_dir, plot_dir, windows, latencies, events,
spans, crediting, threshold, geometry, channels, snapshot)`: `latencies`,
`events` (the mode's event table) and `spans` (the credited spans in
mission time, [`credited_span_times`](@ref)) are `nothing` without an
event table; `geometry` is the [`RunGeometry`](@ref) of the run and
`snapshot` the dictionary written to `config_telemetry.toml`.
"""
function replay_classifier(
    config::AbstractDict;
    model::AbstractString,
    run_id::AbstractString = new_run_id(),
    run_dir::Union{Nothing,AbstractString} = nothing,
    events::Union{Nothing,AbstractString} = nothing,
    tdi_file::Union{Nothing,AbstractString} = nothing,
    tdi_group::Union{Nothing,AbstractString} = nothing,
    tdi_first_row::Integer = 1,
    live::Bool = false,
    link_volume_channels::Integer = 1,
    on_window = nothing,
)
    return @timeit TIMER "telemetry replay" begin
        isempty(run_id) && throw(ArgumentError("run_id must not be empty."))
        link_volume_channels >= 1 || throw(
            ArgumentError(
                "link_volume_channels = $link_volume_channels; must be at least 1.",
            ),
        )
        settings = telemetry_settings(config)
        paths = pipeline_paths(config)
        run_path = override(run_dir, settings.run_dir)
        isempty(run_path) && throw(
            ArgumentError("no run directory: give `run_dir` or set [telemetry] run_dir."),
        )
        run_path = resolvepath(run_path)
        model_path = resolvepath(model)
        results_dir = joinpath(paths.results, "run_$run_id")
        plot_dir = joinpath(paths.plots, "run_$run_id")
        startswith(abspath(results_dir), abspath(run_path)) &&
            throw(ArgumentError("the output directory must lie outside the run directory."))
        mkpath(results_dir)
        mkpath(plot_dir)

        # The channels the model reads: a producer run may carry them as
        # payload columns of one mission; a mission that carried the A
        # channel alone, or a replay given the record explicitly, takes the
        # channels of every delivered batch from the record the payload was
        # exported from
        channels = run_channels(dirname(model_path); default = "A")
        wanted = [String(c) for c in channel_names(channels)]
        carried = telemetry_run_columns(run_path)
        from_record =
            channels != "A" && (tdi_file !== nothing || !issubset(wanted, carried))
        # The A channel is the one column of a one-column run, else the column A
        columns =
            length(carried) == 1 ? nothing : (channels == "A" || from_record ? "A" : wanted)
        run = open_telemetry_run(
            run_path;
            producer_compat = settings.producer_compat,
            columns = columns,
        )
        geometry = run_geometry(run)
        psd_sidecar = settings.psd_sidecar
        if settings.psd_mode == "trailing"
            # Causal whitening, with a PSD estimated only from data already
            # delivered to the ground station: the estimate follows the
            # delivered record, and a sidecar PSD of the whole record would
            # make it non-causal.
            isempty(psd_sidecar) ||
                @warn "psd_sidecar is ignored under psd_mode = \"trailing\"" psd_sidecar
            psd_sidecar = ""
        end
        detector = detector_from_run(
            model_path;
            context_windows = settings.context_windows,
            psd_sidecar = psd_sidecar,
        )
        record_path = ""
        if from_record
            tdi_file === nothing && throw(
                ArgumentError(
                    "the model was trained on the channels $channels and the run " *
                    "carries the payload columns $(join(carried, ", ")); give " *
                    "`tdi_file`, the record the payload of the mission was exported from.",
                ),
            )
            record_path = resolvepath(tdi_file)
            record, record_rate = channel_record(
                record_path;
                group = tdi_group === nothing ?
                        record_groups(preprocessing_settings(config)) :
                        String(tdi_group),
                channels = channels,
            )
            isapprox(record_rate, geometry.sample_rate; rtol = 1e-9) || throw(
                ArgumentError(
                    "$record_path samples at $record_rate Hz, the mission at " *
                    "$(geometry.sample_rate) Hz.",
                ),
            )
            1 <= tdi_first_row <= size(record, 1) || throw(
                ArgumentError(
                    "tdi_first_row = $tdi_first_row; the record holds " *
                    "$(size(record, 1)) rows.",
                ),
            )
            tdi_first_row > 1 && (record = record[tdi_first_row:end, :])
            run = ScheduledRecordRun(run, record)
        end
        trailing = nothing
        if settings.psd_mode == "trailing"
            rows_per_day = 86400 * geometry.sample_rate
            # The high-pass rings at the ends of every delivered run; the
            # transient is trimmed over a few cutoff periods
            edge_rows =
                detector.highpass_cutoff_hz > 0 ?
                round(
                    Int,
                    settings.psd_edge_periods * geometry.sample_rate /
                    detector.highpass_cutoff_hz,
                ) : 0
            trailing = TrailingWelch(
                round(Int, settings.psd_trailing_days * rows_per_day),
                round(Int, settings.psd_refresh_days * rows_per_day),
                settings.psd_segment_length;
                edge_rows = edge_rows,
                # Smoothed as the batch estimate that whitened the model's features
                smoothing_dex = preprocessing_settings(config).psd_smoothing_dex,
            )
        end
        @info "telemetry run opened" run_dir = run_path state = run_state(run) sample_rate =
            geometry.sample_rate points_per_batch = geometry.points_per_batch producer =
            geometry.package_version
        following = live || settings.mode == "live"
        n_alarms = Ref(0)
        on_record =
            record -> begin
                record.decision == 1 && (n_alarms[] += 1)
                record.window % 100 == 0 &&
                    @info "scored" window = record.window score =
                        round(record.score; digits = 3) complete_at = record.complete_at alarms =
                        n_alarms[]
                on_window === nothing || on_window(record)
            end
        windows = if following
            follow_run(
                run,
                detector;
                poll_interval_sec = settings.poll_interval_sec,
                min_coverage = settings.min_coverage,
                tdi_gap_dilation_sec = settings.tdi_gap_dilation_sec,
                trailing_psd = trailing,
                on_window = on_record,
            )
        else
            replay_run(
                run,
                detector;
                min_coverage = settings.min_coverage,
                tdi_gap_dilation_sec = settings.tdi_gap_dilation_sec,
                trailing_psd = trailing,
                on_window = on_record,
            )
        end
        write_csv(joinpath(results_dir, "telemetry_windows.csv"), windows)
        @info "windows scored" n_windows = nrow(windows) alarms = n_alarms[]

        events_path = override(events, settings.events_csv)
        crediting = Symbol(settings.alert_crediting)
        latencies = nothing
        event_table = nothing
        spans = nothing
        if !isempty(events_path)
            # Alerts are credited from the onset of the channels the detector reads
            event_table =
                mode_events(CSV.read(resolvepath(events_path), DataFrame), channels)
            latencies = alert_latency_table(
                windows,
                event_table,
                geometry;
                processing_latency_hours = settings.processing_latency_hours,
                persistence = settings.alert_persistence,
                crediting = crediting,
                confirmation_persistence = settings.alert_confirmation_persistence,
            )
            write_csv(joinpath(results_dir, "alert_latency.csv"), latencies)
            spans = credited_span_times(event_table, geometry; crediting = crediting)
            for row in eachrow(latencies)
                if row.detected
                    @info "event detected" label = row.label latency_data_h =
                        round(row.latency_data_h; digits = 2) latency_total_h =
                        round(row.latency_total_h; digits = 2)
                else
                    @info "event missed" label = row.label
                end
                if settings.alert_confirmation_persistence > 0 && row.confirmed
                    @info "alert confirmed" label = row.label confirmation_latency_data_h =
                        round(row.confirmation_latency_data_h; digits = 2)
                end
            end
        end
        snapshot = Dict{String,Any}(
            "telemetry" => Dict{String,Any}(
                "run_dir" => provenance_path(run_path),
                "producer_version" => geometry.package_version,
                "run_state" => String(run_state(run)),
                "mode" => following ? "live" : "replay",
                "model" => rootrelative(model_path),
                "channels" => channels,
                "payload_columns" => carried,
                "served_columns" =>
                    columns === nothing ? carried :
                    columns isa AbstractString ? [columns] : columns,
                # The volume the producer's link was sized for is stated by
                # the caller
                "link_volume_channels" => link_volume_channels,
                "tdi_file" => isempty(record_path) ? "" : provenance_path(record_path),
                "tdi_first_row" => tdi_first_row,
                "threshold" => Float64(detector.threshold),
                "min_coverage" => settings.min_coverage,
                "tdi_gap_dilation_sec" => settings.tdi_gap_dilation_sec,
                "context_windows" => settings.context_windows,
                "psd_mode" => settings.psd_mode,
                "psd_sidecar" => isempty(psd_sidecar) ? "" : rootrelative(psd_sidecar),
                "psd_trailing_days" => settings.psd_trailing_days,
                "psd_refresh_days" => settings.psd_refresh_days,
                "psd_segment_length" => settings.psd_segment_length,
                "processing_latency_hours" => settings.processing_latency_hours,
                "alert_persistence" => settings.alert_persistence,
                "alert_confirmation_persistence" =>
                    settings.alert_confirmation_persistence,
                "alert_crediting" => settings.alert_crediting,
                "events_csv" =>
                    isempty(events_path) ? "" : rootrelative(resolvepath(events_path)),
                "run_id" => String(run_id),
                "n_windows" => nrow(windows),
                "n_alarms" => n_alarms[],
            ),
            # The producer's own identity of the run (descriptor.toml)
            "producer" => run_descriptor(run),
        )
        write_toml(joinpath(results_dir, "config_telemetry.toml"), snapshot)
        @info "telemetry artifacts written" results_dir

        (
            run_id = String(run_id),
            results_dir = results_dir,
            plot_dir = plot_dir,
            windows = windows,
            latencies = latencies,
            events = event_table,
            spans = spans,
            crediting = crediting,
            threshold = detector.threshold,
            geometry = geometry,
            channels = channels,
            snapshot = snapshot,
        )
    end
end
