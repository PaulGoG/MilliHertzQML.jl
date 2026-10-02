# The classifier as the window scorer of the streaming detector, on an
# in-memory run, and the detector of a training run directory; included by
# runtests.jl. The consumer side of the telemetry coupling is tested in
# StreamingInference.jl and MilliHertzBase.jl.

@testset "Telemetry coupling (classifier)" begin
    epoch = Dates.DateTime(2035, 1, 1)
    geometry = RunGeometry(0.2, 50.0, 10, epoch, "1.0.0")

    # Streaming detector on an in-memory run: a burst is scored where it lies
    rng = StableRNG(31)
    fs = 0.2
    n_rows = 6000
    payload = synthesize_noise(rng, n_rows, fs; f_min = 1e-5, psd = lisa_noise_psd)
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

    # The scorer of the detector is the classifier: a stateless scorer of the
    # MBHB probability, evaluated on the conditioned window
    @test detector.scorer isa VQCScorer && estimator_memory(detector.scorer) == Stateless()
    @test score_label(detector.scorer) == "MBHB probability"
    @test score_bounds(detector.scorer) == (0.0, 1.0)
    stretch = payload[1:5000]
    @test score_window(detector, stretch, 2001) ==
          window_score(detector.scorer, condition_window(detector, stretch, 2001), fs)
    @test all(0 .<= windows.score .<= 1)
    @test all(windows.decision .== Int.(windows.score .>= 0.5f0))
    # The scored value equals a direct evaluation on the same delivered
    # stretch: window 11 (rows 1001:2000) is scored once its stretch 1:4000
    # is on the ground, the window sitting at offset 1001 inside it. The
    # batches reach the consumer in single precision, so the comparison is
    # made against the same rounding.
    direct = score_window(detector, Float32.(payload[1:4000]), 1001)
    @test isapprox(windows.score[11], direct; atol = 1e-6)

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
        @test d.psd !== nothing && d.scorer.features.feature_set == :whitened
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
              d_other.scorer.features.feature_set == d.scorer.features.feature_set
        @test_throws ArgumentError detector_from_run(
            model_path;
            psd_sidecar = joinpath(dir, "absent.toml"),
        )
        @test_throws ArgumentError detector_from_run(joinpath(dir, "absent.jld2"))
    end
end
