# Integration test of the telemetry coupling through the producer: a minimal
# DeepSpaceTelemetry mission ingests an exported payload at 0.2 Hz into a
# temporary run root, and the consumer replays the run directory through
# the package extension. Included by runtests.jl.

@testset "Telemetry coupling (DeepSpaceTelemetry run)" begin
    TelemetryCore = DeepSpaceTelemetry.TelemetryCore
    Supervisor = DeepSpaceTelemetry.Supervisor
    rng = StableRNG(41)
    fs = 0.2
    n_rows = 17_280                       # one mission day at 0.2 Hz
    payload = synthesize_noise(rng, n_rows, fs; f_min = 1e-5)
    burst = 9001:10000
    payload[burst] .+= 3e-19 .* sin.(2π * 5e-3 .* (0:999) ./ fs)
    epoch = Dates.DateTime(2035, 1, 1, 6)
    previous_root = TelemetryCore.DATA_ROOT[]
    mktempdir() do dir
        payload_path = joinpath(dir, "payload.csv")
        CSV.write(payload_path, DataFrame(Amplitude = Float32.(payload)))
        scenario = Dict{String,Any}(
            "simulation" => Dict{String,Any}(
                "speed_up" => 3600.0,
                "start_sim_time" => Dates.format(epoch, "yyyy-mm-ddTHH:MM:SS"),
                "mission_wall_seconds" => 26.0,   # one day plus margin at 3600×
                "initial_downtime_days" => 0.0,
                "rng_seed" => 7,
            ),
            "storage" => Dict{String,Any}("max_storage_gb" => 2.0),
            "supervision" => Dict{String,Any}("on_component_failure" => "abort"),
            "telemetry" => Dict{String,Any}(
                "session_start" => "00:00:00",
                "session_duration_hours" => 24.0,
                "bandwidth_profile" => "flat",
                "max_batches_per_hour" => 200.0,
            ),
            "physics" => Dict{String,Any}(
                "data_source" => "external",
                "external_data_path" => payload_path,
                "sample_rate" => fs,
                "segment_duration_sec" => 50.0,
                "batch_size" => 10,
            ),
            "packet_loss" => Dict{String,Any}("enabled" => false),
            "dashboard" => Dict{String,Any}(
                "open_live_viewer" => false,
                "open_receiver_log" => false,
                "open_emitter_log" => false,
            ),
            "post_processing" => Dict{String,Any}(
                "generate_mask_timeline" => false,
                "expand_to_pointwise_masks" => false,
                "alert_latency" => false,
                "delivery_delay" => false,
                "hdf5_export" => false,
            ),
        )
        TelemetryCore.DATA_ROOT[] = joinpath(dir, "producer")
        run_dir = try
            Supervisor.run_mission(scenario; run_id = "coupling_test", orig_stdout = devnull)
        finally
            TelemetryCore.DATA_ROOT[] = previous_root
        end
        @test isfile(joinpath(run_dir, "RUN_COMPLETE"))
        @test isfile(joinpath(run_dir, "config_snapshot.toml"))

        run = open_telemetry_run(run_dir; producer_compat = "1.0")
        geometry = run_geometry(run)
        @test geometry.sample_rate == fs && geometry.points_per_batch == 100
        @test geometry.start_sim_time == epoch
        @test run_state(run) == :complete
        @test VersionNumber(geometry.package_version) >= v"1.0"
        batches = list_batches(run)
        @test !isempty(batches) && issorted([b.index for b in batches])
        delivered = filter(b -> b.state == :ground, batches)
        @test !isempty(delivered)
        first_batch = delivered[1]
        @test read_batch(run, first_batch.name) == Float32.(payload[first_batch.rows])
        events = arrival_events(run)
        @test !isempty(events) && issorted([e.sim_time for e in events])
        @test all(e -> e.event in (:ingested, :retry, :lost, :pruned, :other), events)
        @test count(e -> e.event == :ingested, events) == length(delivered)
        @test_throws ArgumentError open_telemetry_run(run_dir; producer_compat = "99.0")
        @test_throws ArgumentError open_telemetry_run(joinpath(dir, "absent"))

        model = VariationalQuantumClassifier(4, 2; rng = rng)
        scaler = FeatureScaler([0.0, 0.0, 0.0, -1.0], [3.0, 3.0, 1.0, 1.0])
        detector = StreamingDetector(
            model,
            scaler,
            0.5;
            sample_rate = fs,
            window_size = 1000,
            step_size = 100,
            psd = lisa_noise_psd,
            context_windows = 2,
        )
        windows = replay_run(run, detector)
        @test nrow(windows) >= 1
        @test all(windows.coverage .== 1.0)
        @test all(windows.complete_at .>= windows.content_end)
        @test issorted(windows.complete_at)
        # The score of a window equals a direct evaluation on the stretch that
        # was on the ground when it completed (record-context whitening)
        w = windows[1, :]
        arrived = Coverage()
        for e in events
            e.event == :ingested && e.sim_time <= w.complete_at || continue
            k, _ = parse_batch_name(e.batch)
            add!(arrived, batch_rows(k, geometry.points_per_batch))
        end
        span = covered_stretch(arrived, w.row_start:w.row_end)
        lo = max(first(span), w.row_start - 2000)
        hi = min(last(span), w.row_end + 2000)
        stretch = Float32.(payload[lo:hi])
        @test isapprox(
            windows.score[1],
            score_window(detector, stretch, w.row_start - lo + 1);
            atol = 1e-6,
        )
        @test replay_run(run, detector).score == windows.score
        events_table = DataFrame(
            event = [1],
            merger_time_s = [9500 / fs],
            label_start_index = [first(burst)],
            label_end_index = [last(burst)],
            signal_start_index = [first(burst)],
        )
        latency = alert_latency_table(windows, events_table, geometry)
        @test nrow(latency) == 1 && latency.merger_time_s[1] == 9500 / fs
        @test latency.t_merger[1] == epoch + Dates.Second(9500 * 5)

        # A scheduled generation gap: every delivered batch holds the payload
        # rows of its content epoch, and the gap's rows are never delivered.
        gapped = deepcopy(scenario)
        gapped["disruption"] = Dict{String,Any}(
            "events" => [
                Dict{String,Any}(
                    "type" => "antenna_repointing",
                    "label" => "repointing",
                    "start_day" => 0.1,
                    "duration_hours" => 1.0,
                ),
            ],
        )
        TelemetryCore.DATA_ROOT[] = joinpath(dir, "producer")
        gap_dir = try
            Supervisor.run_mission(gapped; run_id = "coupling_gap", orig_stdout = devnull)
        finally
            TelemetryCore.DATA_ROOT[] = previous_root
        end
        gap_run = open_telemetry_run(gap_dir)
        # Batches past the payload end carry the producer's zero padding.
        gap_delivered =
            filter(b -> b.state == :ground && last(b.rows) <= n_rows, list_batches(gap_run))
        @test !isempty(gap_delivered)
        @test all(
            b -> read_batch(gap_run, b.name) == Float32.(payload[b.rows]),
            gap_delivered,
        )
        gap_rows =
            time_row(geometry, epoch+Dates.Minute(144)):(time_row(
                geometry,
                epoch+Dates.Minute(204),
            )-1)
        @test all(b -> isempty(intersect(b.rows, gap_rows)), gap_delivered)
        @test any(b -> first(b.rows) > last(gap_rows), gap_delivered)
    end
end

@testset "Payload alignment of producer runs" begin
    fs = 0.2
    epoch = Dates.DateTime(2035, 1, 1)
    function fake_run(
        dir;
        version = "2.0.1",
        source = "external",
        downtime = 0.0,
        origin = nothing,
        tx = String[],
        components = String[],
    )
        mkpath(dir)
        touch(joinpath(dir, "RUN_COMPLETE"))
        provenance =
            Dict{String,Any}("platform" => Dict{String,Any}("package_version" => version))
        origin === nothing || (provenance["payload_origin"] = origin)
        open(joinpath(dir, "config_snapshot.toml"), "w") do io
            TOML.print(
                io,
                Dict{String,Any}(
                    "physics" => Dict{String,Any}(
                        "sample_rate" => fs,
                        "segment_duration_sec" => 50.0,
                        "batch_size" => 10,
                        "data_source" => source,
                    ),
                    "simulation" => Dict{String,Any}(
                        "start_sim_time" => string(epoch),
                        "initial_downtime_days" => downtime,
                    ),
                    "provenance" => provenance,
                ),
            )
        end
        isempty(tx) || write(
            joinpath(dir, "events_tx.csv"),
            join(["SimTime,Batch,Event"; tx], "\n") * "\n",
        )
        isempty(components) || write(
            joinpath(dir, "component_events.csv"),
            join(["SimTime,Component,Event"; components], "\n") * "\n",
        )
        return dir
    end
    gap = [
        "2035-01-01T02:00:00.0,SCHEDULED,gap_start",
        "2035-01-01T03:00:00.0,SCHEDULED,gap_end",
    ]
    restart = ["2035-01-01T04:00:00.0,emitter,restart"]
    mktempdir() do dir
        # Producers up to 2.0.1 misplace an external payload after a scheduled
        # gap or a restart; such runs are refused, others open.
        @test_throws ArgumentError open_telemetry_run(
            fake_run(joinpath(dir, "a"); tx = gap),
        )
        @test_throws ArgumentError open_telemetry_run(
            fake_run(joinpath(dir, "b"); components = restart),
        )
        @test_throws ArgumentError open_telemetry_run(
            fake_run(joinpath(dir, "c"); version = "unknown", tx = gap),
        )
        @test open_telemetry_run(
            fake_run(joinpath(dir, "d"); version = "2.1.1", tx = gap),
        ) isa AbstractTelemetryRun
        @test open_telemetry_run(
            fake_run(joinpath(dir, "e"); source = "synthetic", tx = gap),
        ) isa AbstractTelemetryRun
        @test open_telemetry_run(
            fake_run(joinpath(dir, "f"); tx = ["2035-01-01T02:00:00.0,RECORDER,gap_start"]),
        ) isa AbstractTelemetryRun
        # Payload row 1 must lie at the mission epoch.
        @test_throws ArgumentError open_telemetry_run(
            fake_run(joinpath(dir, "g"); downtime = 0.5),
        )
        @test_throws ArgumentError open_telemetry_run(
            fake_run(joinpath(dir, "h"); version = "2.1.1", origin = "2034-12-31T12:00:00"),
        )
        # A stamped payload row sets the rows of a batch and must agree with
        # its content epoch.
        run_dir = fake_run(joinpath(dir, "i"); version = "2.1.1", origin = string(epoch))
        batch = joinpath(run_dir, "ground", "LIVE_batch_7")
        mkpath(batch)
        write(
            joinpath(batch, "metadata.json"),
            """{"segment_count":10,"content_epoch":"2035-01-01T00:50:00","payload_row":601,"batch_id":7}""",
        )
        run = open_telemetry_run(run_dir)
        @test only(list_batches(run)).rows == 601:700
        write(
            joinpath(batch, "metadata.json"),
            """{"segment_count":10,"content_epoch":"2035-01-01T00:50:00","payload_row":501,"batch_id":7}""",
        )
        @test_throws ArgumentError list_batches(run)
    end
end
