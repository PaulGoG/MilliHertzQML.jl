# Unit tests of the consumer side of the telemetry coupling (src/telemetry.jl)
# on an in-memory run; included by runtests.jl.

@testset "Telemetry coupling (core)" begin
    epoch = Dates.DateTime(2035, 1, 1)
    geometry = RunGeometry(0.2, 50.0, 10, epoch, "1.0.0")
    @test geometry.points_per_batch == 100
    @test_throws ArgumentError RunGeometry(0.2, 50.0, 0, epoch)
    @test RunGeometry(0.3, 50.0, 10, epoch).points_per_batch == 150
    @test_throws ArgumentError RunGeometry(0.2, 7.0, 3, epoch)     # 4.2 samples
    @test parse_batch_name("LIVE_batch_12") == (12, true)
    @test parse_batch_name("ARCH_batch_3") == (3, false)
    @test_throws ArgumentError parse_batch_name("batch_3")
    @test batch_rows(1, 100) == 1:100 && batch_rows(7, 100) == 601:700
    @test row_time(geometry, 1) == epoch
    @test row_time(geometry, 101) == epoch + Dates.Second(500)
    @test event_symbol("Ingested") == :ingested && event_symbol("weird") == :other

    # Coverage algebra is order-agnostic; erosion and holes
    c = Coverage()
    add!(c, 201:300)
    add!(c, 1:100)
    add!(c, 101:200)
    @test c.intervals == [1:300]
    @test covered_fraction(c, 1:300) == 1.0
    @test isapprox(covered_fraction(c, 251:350), 0.5)
    @test holes(c, 250:350) == [301:350]
    @test covered_stretch(c, 50:250) == 1:300 && isempty(covered_stretch(c, 250:350))
    remove!(c, 150:160)
    @test c.intervals == [1:149, 161:300]
    @test holes(c, 1:300) == [150:160]
    @test isempty(covered_stretch(c, 100:200))
    @test covered_fraction(Coverage(), 1:10) == 0.0 && holes(Coverage(), 1:10) == [1:10]

    # Window scheduler: window m covers [1 + (m − 1) S, (m − 1) S + W]
    s = WindowScheduler(1000, 100)
    @test window_rows(s, 1) == 1:1000 && window_rows(s, 11) == 1001:2000
    @test windows_touching(s, 1:100) == 1:1
    @test windows_touching(s, 1001:1100) == 2:11
    @test windows_touching(s, 2001:2100) == 12:21
    cov = Coverage()
    ready = Int[]
    for k in 1:12
        add!(cov, batch_rows(k, 100))
        append!(ready, newly_evaluable!(s, cov, batch_rows(k, 100)))
    end
    @test ready == collect(1:3)          # windows 1–3 complete after 1200 rows
    @test isempty(newly_evaluable!(s, cov, 1:1200))   # nothing emitted twice
    @test_throws ArgumentError WindowScheduler(1000, 2000)
    @test_throws ArgumentError WindowScheduler(1000, 100; min_coverage = 0.0)
    partial = WindowScheduler(1000, 100; min_coverage = 0.5)
    cov2 = Coverage()
    add!(cov2, 1:500)
    @test newly_evaluable!(partial, cov2, 1:500) == [1]

    # Streaming detector on an in-memory run: a burst is scored where it lies
    rng = StableRNG(31)
    fs = 0.2
    n_rows = 6000
    payload = synthesize_noise(rng, n_rows, fs; f_min = 1e-5)
    burst = 3001:4000
    payload[burst] .+= 3e-19 .* sin.(2π * 5e-3 .* (0:999) ./ fs)
    n_batches = div(n_rows, 100)
    events = ArrivalEvent[]
    # LIFO-like delivery: batches arrive out of order in pairs
    order = collect(1:n_batches)
    for k in 1:2:(n_batches-1)
        order[k], order[k+1] = order[k+1], order[k]
    end
    for (i, k) in enumerate(order)
        push!(
            events,
            ArrivalEvent(epoch + Dates.Second(600 * i), "LIVE_batch_$k", :ingested, 0),
        )
    end
    run = MemoryTelemetryRun(geometry, payload, events)
    @test length(list_batches(run)) == n_batches
    @test read_batch(run, "LIVE_batch_2") == Float32.(payload[101:200])
    @test run_state(run) == :complete
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
    @test_throws ArgumentError StreamingDetector(
        model,
        scaler,
        0.5;
        sample_rate = fs,
        window_size = 1000,
        step_size = 2000,
    )
    windows = replay_run(run, detector)
    @test nrow(windows) == div(n_rows - 1000, 100) + 1
    @test windows.window == 1:nrow(windows)
    @test all(windows.coverage .== 1.0)
    @test all(0 .<= windows.score .<= 1)
    @test all(windows.decision .== Int.(windows.score .>= 0.5f0))
    @test issorted(windows.complete_at)
    # A window completes when its last batch arrives: window 1 needs batches
    # 1–10, and under the pairwise swap batch 9 is the tenth event
    @test windows.complete_at[1] == epoch + Dates.Second(600 * 10)
    @test windows.row_start[1] == 1 && windows.row_end[end] == n_rows
    @test all(windows.inference_wall_ms .>= 0)
    # The scored value equals a direct evaluation on the same delivered
    # stretch: window 11 (rows 1001:2000) completes with batch 19 as the
    # twentieth event, when rows 1:2000 are on the ground
    direct = score_window(detector, payload[1:2000], 1001)
    @test isapprox(windows.score[11], direct; atol = 1e-6)
    # Lost batches are removed with erosion and their windows never complete
    lossy = ArrivalEvent[]
    for k in 1:n_batches
        push!(
            lossy,
            ArrivalEvent(
                epoch + Dates.Second(600 * k),
                "LIVE_batch_$k",
                k == 30 ? :lost : :ingested,
                0,
            ),
        )
    end
    run_lossy = MemoryTelemetryRun(geometry, payload, lossy; lost = ["LIVE_batch_30"])
    lossy_windows = replay_run(run_lossy, detector; tdi_gap_dilation_sec = 100.0)
    # Window m covers rows [1 + 100 (m − 1), 100 (m − 1) + 1000]; batch 30 is rows
    # 2901:3000, eroded to 2881:3020, so windows 21–31 never complete — including
    # window 31, whose first rows arrive only after the loss
    @test all(w -> w <= 20 || w >= 32, lossy_windows.window)
    @test nrow(lossy_windows) == nrow(windows) - 11
    # Live mode on a completed run drains the feed once and stops
    followed = follow_run(run, detector; poll_interval_sec = 0.01, max_wall_sec = 30)
    @test nrow(followed) == nrow(windows) && followed.score == windows.score

    # Alert latency: one event inside the burst, one outside every alarm
    forced = copy(windows)
    forced.decision .= 0
    forced.decision[25:28] .= 1                     # windows covering rows 2401:3700
    events_table = DataFrame(
        event = [1, 2],
        merger_time_s = [3500 / fs, 5500 / fs],
        label_start_index = [3001, 5401],
        label_end_index = [4000, 5600],
    )
    latency =
        alert_latency_table(forced, events_table, geometry; processing_latency_hours = 1.0)
    @test nrow(latency) == 2
    @test latency.detected == [true, false]
    @test latency.alarm_window[1] == 25
    @test latency.t_alarm[1] == forced.complete_at[25]
    @test isapprox(
        latency.latency_data_h[1],
        Dates.value(forced.complete_at[25] - (epoch + Dates.Second(3500 * 5))) / 3.6e6,
    )
    @test latency.latency_total_h[1] == latency.latency_data_h[1] + 1.0
    @test ismissing(latency.t_alarm[2])
    @test latency.false_alarm_episodes[1] == 0
    # Window 45 (rows 4401:5400) ends before the second span, window 46
    # (4501:5500) overlaps it; window 5 (401:1400) lies outside every span
    forced.decision[45:46] .= 1
    forced.decision[5] = 1
    latency2 = alert_latency_table(forced, events_table, geometry)
    @test latency2.detected == [true, true]
    @test latency2.alarm_window[2] == 46
    @test latency2.false_alarm_episodes[1] == 2
    @test latency2.false_alarms_per_30d[1] > 0
    @test_throws ArgumentError alert_latency_table(forced, DataFrame(x = [1]), geometry)

    # Detector reconstruction from a training run directory
    mktempdir() do dir
        features = joinpath(dir, "feat_features.csv")
        CSV.write(
            features,
            DataFrame(
                p_low = [1.0, 1.1],
                p_high = [1.0, 0.9],
                spectral_entropy = [0.9, 0.8],
                log_power_std = [0.0, 0.1],
            ),
        )
        open(joinpath(dir, "feat_features.toml"), "w") do io
            TOML.print(
                io,
                Dict(
                    "features" => Dict(
                        "window_size" => 1000,
                        "step_size" => 100,
                        "sample_rate" => 0.2,
                        "psd" => "model",
                        "observation_years" => 1.0,
                        "highpass_cutoff_hz" => 5e-4,
                        "highpass_order" => 8,
                        "low_band_hz" => [1e-3, 5e-3],
                        "high_band_hz" => [5e-3, 1e-1],
                        "feature_set" => "whitened",
                    ),
                ),
            )
        end
        run_dir = joinpath(dir, "run_x")
        mkpath(run_dir)
        model_path = joinpath(run_dir, "gw_model.jld2")
        save_model(model_path, model; scaler = scaler)
        open(joinpath(run_dir, "threshold.toml"), "w") do io
            TOML.print(io, Dict("threshold" => Dict("value" => 0.42)))
        end
        open(joinpath(run_dir, "config.toml"), "w") do io
            TOML.print(io, Dict("training" => Dict("train_features" => features)))
        end
        d = detector_from_run(model_path; context_windows = 3)
        @test d.threshold == 0.42f0 && d.window_size == 1000 && d.context_windows == 3
        @test d.psd !== nothing && d.feature_set == :whitened
        @test isapprox(d.psd(1e-3), lisa_noise_psd(1e-3))
        # A whitening sidecar other than the training run's is honoured —
        # here one that whitens not at all, so the override is unambiguous —
        # while the rest of the conditioning still comes from the training
        # sidecar; a missing one fails fast rather than falling back silently
        other_sidecar = joinpath(dir, "other_features.toml")
        open(other_sidecar, "w") do io
            TOML.print(
                io,
                Dict(
                    "features" => Dict(
                        "window_size" => 1000,
                        "step_size" => 100,
                        "sample_rate" => 0.2,
                        "psd" => "none",
                        "highpass_cutoff_hz" => 5e-4,
                        "highpass_order" => 8,
                        "feature_set" => "whitened",
                    ),
                ),
            )
        end
        d_other =
            detector_from_run(model_path; psd_sidecar = other_sidecar, context_windows = 3)
        @test d_other.psd === nothing
        @test d_other.threshold == d.threshold &&
              d_other.window_size == d.window_size &&
              d_other.feature_set == d.feature_set
        @test_throws ArgumentError detector_from_run(
            model_path;
            psd_sidecar = joinpath(dir, "absent.toml"),
        )
        @test_throws ArgumentError detector_from_run(joinpath(dir, "absent.jld2"))
        @test whitening_psd_from_sidecar(joinpath(dir, "feat_features.toml"))(1e-3) ==
              lisa_noise_psd(1e-3)
    end
end
