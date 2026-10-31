# Integration test of the classifier on a producer run: a minimal
# DeepSpaceTelemetry mission on the producer's tick clock ingests an
# exported payload at 0.2 Hz into a temporary run root, and the detector of
# a classifier replays the run directory through StreamingInference's
# extension, directly and through the replay stage. The reader itself is
# tested in StreamingInference.jl and MilliHertzBase.jl. Included by
# runtests.jl.

@testset "Telemetry coupling (DeepSpaceTelemetry run)" begin
    TelemetryCore = DeepSpaceTelemetry.TelemetryCore
    Supervisor = DeepSpaceTelemetry.Supervisor
    rng = StableRNG(41)
    fs = 0.2
    n_rows = 17_280                       # one mission day at 0.2 Hz
    payload = synthesize_noise(rng, n_rows, fs; f_min = 1e-5, psd = lisa_noise_psd)
    burst = 9001:10000
    payload[burst] .+= 3e-19 .* sin.(2π * 5e-3 .* (0:999) ./ fs)
    epoch = Dates.DateTime(2035, 1, 1, 6)
    # A scenario run on the tick clock: the same span, deterministic, seconds
    function tick_mission(dir, scenario, run_id)
        path = joinpath(dir, "$run_id.toml")
        open(io -> TOML.print(io, scenario), path, "w")
        cfg = TelemetryCore.tick_configuration(TelemetryCore.load_config(path))
        return Supervisor.run_mission(
            cfg;
            run_id = run_id,
            orig_stdout = devnull,
            data_root = joinpath(dir, "producer"),
        )
    end
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
                "alert_latency" => false,
                "delivery_delay" => false,
                "hdf5_export" => false,
            ),
        )
        run_dir = tick_mission(dir, scenario, "coupling_test")
        run = open_telemetry_run(run_dir)
        geometry = run_geometry(run)
        events = arrival_events(run)

        model = VariationalQuantumClassifier(4, 2; rng = rng)
        scaler = FeatureScaler([0.0, 0.0, 0.0, -1.0], [3.0, 3.0, 1.0, 1.0])
        detector = StreamingDetector(
            model,
            scaler,
            0.5;
            features = FeatureMap(; BANDS...),
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

        # The replay stage on the same run: the detector of a training run
        # directory, the tables and the snapshot it writes
        features_path = joinpath(dir, "train_features.csv")
        CSV.write(
            features_path,
            DataFrame(rand(rng, Float32, 4, 4), feature_names(:whitened)),
        )
        open(joinpath(dir, "train_features.toml"), "w") do io
            TOML.print(
                io,
                Dict(
                    "features" => merge(
                        Dict{String,Any}(
                            "window_size" => 1000,
                            "step_size" => 100,
                            "sample_rate" => fs,
                            "psd" => "model",
                            "observation_years" => 1.0,
                            "feature_set" => "whitened",
                            "channels" => "A",
                        ),
                        SIDECAR_CONDITIONING,
                    ),
                ),
            )
        end
        model_dir = joinpath(dir, "models", "run_trained")
        mkpath(model_dir)
        model_path =
            save_model(joinpath(model_dir, "gw_model.jld2"), model; scaler = scaler)
        open(joinpath(model_dir, "threshold.toml"), "w") do io
            TOML.print(io, Dict("threshold" => Dict("value" => 0.5)))
        end
        open(joinpath(model_dir, "config.toml"), "w") do io
            TOML.print(io, Dict("training" => Dict("train_features" => features_path)))
        end
        events_path = joinpath(dir, "events.csv")
        CSV.write(events_path, events_table)
        config = Dict{String,Any}(
            "paths" => Dict{String,Any}(
                "inputs" => dir,
                "models" => joinpath(dir, "models"),
                "plots" => joinpath(dir, "plots"),
                "results" => joinpath(dir, "results"),
            ),
            "telemetry" => Dict{String,Any}("context_windows" => 2),
        )
        replay = replay_classifier(
            config;
            model = model_path,
            run_id = "stage",
            run_dir = run_dir,
            events = events_path,
        )
        stage_detector = detector_from_run(model_path; context_windows = 2)
        @test replay.windows.score == replay_run(
            run,
            stage_detector;
            min_coverage = 1.0,
            tdi_gap_dilation_sec = 100.0,
        ).score
        @test isequal(
            replay.latencies,
            alert_latency_table(
                replay.windows,
                events_table,
                geometry;
                processing_latency_hours = 1.0,
                persistence = 3,
                crediting = :signal,
                confirmation_persistence = 0,
            ),
        )
        @test replay.spans ==
              [(row_time(geometry, first(burst)), row_time(geometry, last(burst)))]
        results_dir = joinpath(dir, "results", "run_stage")
        @test replay.results_dir == results_dir
        for file in ("telemetry_windows.csv", "alert_latency.csv", "config_telemetry.toml")
            @test isfile(joinpath(results_dir, file))
        end
        snapshot = TOML.parsefile(joinpath(results_dir, "config_telemetry.toml"))
        @test snapshot["telemetry"]["n_windows"] == nrow(replay.windows)
        @test snapshot["telemetry"]["n_alarms"] == count(==(1), replay.windows.decision)
        @test snapshot["telemetry"]["channels"] == "A" &&
              snapshot["telemetry"]["mode"] == "replay"
        @test snapshot["producer"] == run_descriptor(run)
        # The outputs stay outside the run directory
        @test_throws ArgumentError replay_classifier(
            Dict{String,Any}(
                "paths" => Dict{String,Any}("results" => joinpath(run_dir, "results")),
            );
            model = model_path,
            run_id = "inside",
            run_dir = run_dir,
        )
    end
end
