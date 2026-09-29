# Unit tests of the consumer side of the telemetry coupling (src/telemetry.jl)
# on an in-memory run; included by runtests.jl.

# An in-memory run whose producer discarded production: after `gap_after`
# batches a hole of `gap_rows` payload rows opens, and every later batch
# holds the rows its content epoch says, so its rows no longer follow its
# index. Exercises the consumer's handling of index drift.
struct DriftedTelemetryRun <: MilliHertzQML.AbstractTelemetryRun
    geometry::RunGeometry
    payload::Vector{Float32}
    events::Vector{ArrivalEvent}
    gap_after::Int
    gap_rows::Int
end

function drifted_rows(run::DriftedTelemetryRun, k::Integer)
    P = run.geometry.points_per_batch
    shift = k > run.gap_after ? run.gap_rows : 0
    return ((k-1)*P+1+shift):(k*P+shift)
end

MilliHertzQML.run_geometry(run::DriftedTelemetryRun) = run.geometry

function MilliHertzQML.list_batches(run::DriftedTelemetryRun)
    n = div(length(run.payload) - run.gap_rows, run.geometry.points_per_batch)
    return [
        BatchRecord(
            "LIVE_batch_$k",
            k,
            true,
            drifted_rows(run, k),
            row_time(run.geometry, first(drifted_rows(run, k))),
            :ground,
        ) for k in 1:n
    ]
end

function MilliHertzQML.read_batch(run::DriftedTelemetryRun, name::AbstractString)
    return run.payload[drifted_rows(run, parse_batch_name(name)[1])]
end

MilliHertzQML.arrival_events(run::DriftedTelemetryRun) = run.events
MilliHertzQML.run_state(::DriftedTelemetryRun) = :complete

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
    @test time_row(geometry, epoch) == 1
    @test time_row(geometry, epoch + Dates.Second(500)) == 101
    @test time_row(geometry, row_time(geometry, 54_321)) == 54_321
    @test time_row(geometry, epoch - Dates.Second(500)) == -99
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
    # `min_coverage` bounds the conditioning stretch, never the window: a
    # window with half its own rows missing is not scored across the hole
    partial = WindowScheduler(1000, 100; min_coverage = 0.5, context_rows = 1000)
    cov2 = Coverage()
    add!(cov2, 1:500)
    @test isempty(newly_evaluable!(partial, cov2, 1:500))
    add!(cov2, 501:1000)
    @test newly_evaluable!(partial, cov2, 501:1000) == [1]   # stretch 1:2000 half there

    # With a conditioning stretch a window waits for the data after it,
    # and an arriving batch can release a window that lies earlier
    sched = WindowScheduler(10, 5; context_rows = 10, payload_rows = 60)
    @test conditioning_rows(sched, 1) == 1:20
    @test conditioning_rows(sched, 3) == 1:30
    @test conditioning_rows(sched, 6) == 16:45
    cov = Coverage()
    add!(cov, 1:20)
    @test newly_evaluable!(sched, cov, 1:20) == [1]
    add!(cov, 21:30)
    @test newly_evaluable!(sched, cov, 21:30) == [2, 3]
    # The record's end is not waited for beyond the payload
    sched_end = WindowScheduler(10, 5; context_rows = 10, payload_rows = 30)
    @test conditioning_rows(sched_end, 5) == 11:30
    cov_end = Coverage()
    add!(cov_end, 1:30)
    @test 5 in newly_evaluable!(sched_end, cov_end, 1:30)
    # Without a stretch the scheduler behaves as before
    plain = WindowScheduler(10, 5)
    @test conditioning_rows(plain, 3) == window_rows(plain, 3)

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
    @test all(windows.psd_row .== 0)          # the detector's static PSD throughout
    # A window is scored when its conditioning stretch is delivered, not when
    # its own rows are: with two window lengths of context, window 1 waits for
    # rows 1:3000, batches 1–30, the last of which is the thirtieth event
    @test windows.complete_at[1] == epoch + Dates.Second(600 * 30)
    @test windows.row_start[1] == 1 && windows.row_end[end] == n_rows
    @test all(windows.inference_wall_ms .>= 0)
    # The scored value equals a direct evaluation on the same delivered
    # stretch: window 11 (rows 1001:2000) is scored once its stretch 1:4000
    # is on the ground, the window sitting at offset 1001 inside it. The
    # batches reach the consumer in single precision, so the comparison is
    # made against the same rounding.
    direct = score_window(detector, Float32.(payload[1:4000]), 1001)
    @test isapprox(windows.score[11], direct; atol = 1e-6)

    # Causal whitening: every window is whitened by the Welch estimate
    # of the delivered record behind its conditioning stretch, and the table
    # records the last row of that record
    @test_throws ArgumentError TrailingWelch(500, 100, 1000)
    @test_throws ArgumentError TrailingWelch(3000, 0, 1000)
    @test_throws ArgumentError TrailingWelch(3000, 100, 1)
    trailing = replay_run(run, detector; trailing_psd = TrailingWelch(3000, 500, 1000))
    @test nrow(trailing) == nrow(windows) && trailing.window == windows.window
    @test all(trailing.psd_row .>= 1000)
    @test all(0 .<= trailing.score .<= 1)
    @test !all(isapprox.(trailing.score, windows.score; atol = 1e-4))
    for m in (11, 31)
        hi = trailing.psd_row[m]
        lo = max(1, hi - 3000 + 1)
        record = highpass_record(
            Float64.(Float32.(payload[lo:hi])),
            fs;
            cutoff = detector.highpass_cutoff_hz,
            order = detector.highpass_order,
        )
        freqs, table = welch_psd(record, fs; segment_length = 1000, average = :median)
        w_lo = 1 + 100 * (m - 1)
        s_lo, s_hi = max(1, w_lo - 2000), min(n_rows, w_lo + 999 + 2000)
        expected = score_window(
            detector,
            Float32.(payload[s_lo:s_hi]),
            w_lo - s_lo + 1;
            psd = interpolated_psd(freqs, table),
        )
        @test isapprox(trailing.score[m], expected; atol = 1e-6)
    end
    # The estimate is reused while the delivered record has advanced by less
    # than the refresh, redone otherwise
    @test length(unique(trailing.psd_row)) < nrow(trailing)
    @test all(diff(sort(unique(trailing.psd_row))) .>= 500)
    # Replays are deterministic
    @test replay_run(run, detector; trailing_psd = TrailingWelch(3000, 500, 1000)).score ==
          trailing.score
    # Without one segment of record on the ground the static PSD is used
    static = replay_run(run, detector; trailing_psd = TrailingWelch(7000, 500, 7000))
    @test all(static.psd_row .== 0) && static.score == windows.score
    # A smoothed trailing estimate changes the whitening and not the schedule
    smoothed = replay_run(
        run,
        detector;
        trailing_psd = TrailingWelch(3000, 500, 1000; smoothing_dex = 0.01),
    )
    @test smoothed.psd_row == trailing.psd_row && smoothed.score != trailing.score
    @test_throws ArgumentError TrailingWelch(3000, 500, 1000; smoothing_dex = -0.01)
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
    # Without a conditioning stretch a hole blocks exactly the windows it
    # touches: window m covers rows [1 + 100 (m − 1), 100 (m − 1) + 1000], batch
    # 30 is rows 2901:3000, eroded to 2881:3020, so windows 21–31 never complete
    # — including window 31, whose first rows arrive only after the loss
    bare = StreamingDetector(
        model,
        scaler,
        0.5;
        sample_rate = fs,
        window_size = 1000,
        step_size = 100,
        psd = lisa_noise_psd,
        context_windows = 0,
    )
    lossy_windows = replay_run(run_lossy, bare; tdi_gap_dilation_sec = 100.0)
    @test all(w -> w <= 20 || w >= 32, lossy_windows.window)
    @test nrow(lossy_windows) == nrow(windows) - 11
    # The conditioning stretch widens that reach: with two window lengths of
    # context every window of this 6000-row record needs rows within 2000 of the
    # hole, so none of them can be conditioned at all. The number of windows a
    # permanent hole excludes grows with the context the whitening requires.
    @test nrow(replay_run(run_lossy, detector; tdi_gap_dilation_sec = 100.0)) == 0
    # Live mode on a completed run drains the feed once and stops
    followed = follow_run(run, detector; poll_interval_sec = 0.01, max_wall_sec = 30)
    @test nrow(followed) == nrow(windows) && followed.score == windows.score

    # Alert latency: one event inside the burst, one outside every alarm
    forced = copy(windows)
    # One arrival per window, so that the completing window of a run is
    # unambiguous
    forced.complete_at .= epoch .+ Dates.Second.(600 .* (1:nrow(forced)))
    forced.decision .= 0
    forced.decision[25:28] .= 1                     # windows covering rows 2401:3700
    events_table = DataFrame(
        event = [1, 2],
        merger_time_s = [3500 / fs, 5500 / fs],
        label_start_index = [3001, 5401],
        label_end_index = [4000, 5600],
        signal_start_index = [3001, 5401],
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
    @test all(latency2.alert_persistence .== 1)
    @test_throws ArgumentError alert_latency_table(forced, DataFrame(x = [1]), geometry)
    @test_throws ArgumentError alert_latency_table(
        forced,
        events_table,
        geometry;
        persistence = 0,
    )
    # A persistence criterion: the alert is raised by the arrival completing
    # `persistence` consecutive alarmed windows, and isolated alarms are
    # neither alerts nor false-alarm episodes. Windows complete in index
    # order here, so the run 25–28 is complete at window 27 under three and
    # at window 26 under two; the pair 45–46 needs two, the singleton 5 none.
    three = alert_latency_table(forced, events_table, geometry; persistence = 3)
    @test three.detected == [true, false]
    @test three.alarm_window[1] == 27 && three.t_alarm[1] == forced.complete_at[27]
    @test three.false_alarm_episodes[1] == 0
    @test all(three.alert_persistence .== 3)
    two = alert_latency_table(forced, events_table, geometry; persistence = 2)
    @test two.detected == [true, true]
    @test two.alarm_window == [26, 46]
    @test two.false_alarm_episodes[1] == 0
    @test !any(two.shared_alert) && !any(three.shared_alert)
    # Two events whose spans overlap can share one alert; the table says so
    twin = DataFrame(
        event = [1, 2],
        merger_time_s = [3500 / fs, 3600 / fs],
        label_start_index = [3001, 3201],
        label_end_index = [4000, 4100],
        signal_start_index = [3001, 3201],
    )
    shared = alert_latency_table(forced, twin, geometry; persistence = 3)
    @test shared.alarm_window == [27, 27] && all(shared.shared_alert)
    # Out-of-order arrival: the run is complete only when its last-arriving
    # window lands, whichever index that is
    shuffled = copy(forced)
    shuffled.complete_at[26] = maximum(forced.complete_at) + Dates.Hour(1)
    late = alert_latency_table(shuffled, events_table, geometry; persistence = 3)
    @test late.alarm_window[1] == 26 && late.t_alarm[1] == shuffled.complete_at[26]
    # Crediting from the signal onset: the run 25–28 (rows 2401:3700) lies
    # in the first label span but before an onset at row 3801, so it is a
    # false-alarm episode, not an early detection; crediting the whole
    # label span counts it as the event's alert
    late_onset = copy(events_table)
    late_onset.signal_start_index = [3801, 5401]
    signal = alert_latency_table(forced, late_onset, geometry; persistence = 3)
    @test signal.detected == [false, false]
    @test signal.false_alarm_episodes[1] == 1
    @test all(signal.alert_crediting .== "signal")
    label = alert_latency_table(
        forced,
        late_onset,
        geometry;
        persistence = 3,
        crediting = :label,
    )
    @test label.detected == [true, false] && label.alarm_window[1] == 27
    @test label.false_alarm_episodes[1] == 0
    @test all(label.alert_crediting .== "label")
    # A table with spans but no onsets is refused under signal crediting
    no_onset = DataFrames.select(events_table, DataFrames.Not(:signal_start_index))
    @test_throws ArgumentError alert_latency_table(forced, no_onset, geometry)
    @test alert_latency_table(forced, no_onset, geometry; crediting = :label).detected ==
          [true, true]
    # Onsets outside their span, and unknown criteria, are refused
    outside = copy(events_table)
    outside.signal_start_index = [2000, 5401]
    @test_throws ArgumentError alert_latency_table(forced, outside, geometry)
    @test_throws ArgumentError alert_latency_table(
        forced,
        events_table,
        geometry;
        crediting = :merger,
    )

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

    @testset "Index drift" begin
        # After batch 20 the producer discarded 37 rows of production; the
        # hole is never delivered, the later batches hold the rows of their
        # content epoch, and the consumer scores from those rows.
        rng = StableRNG(37)
        fs = 0.2
        gap_after, gap_rows = 20, 37
        n_batches = 60
        payload =
            Float32.(synthesize_noise(rng, n_batches * 100 + gap_rows, fs; f_min = 1e-5))
        drift_geometry =
            RunGeometry(fs, 50.0, 10, epoch, "2.0.0"; payload_rows = length(payload))
        events = [
            ArrivalEvent(epoch + Dates.Second(600 * k), "LIVE_batch_$k", :ingested, 0)
            for k in 1:n_batches
        ]
        run = DriftedTelemetryRun(drift_geometry, payload, events, gap_after, gap_rows)
        records = list_batches(run)
        @test records[gap_after].rows == 1901:2000
        @test records[gap_after+1].rows == (2001+gap_rows):(2100+gap_rows)
        @test read_batch(run, "LIVE_batch_21") == payload[(2001+gap_rows):(2100+gap_rows)]
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
            context_windows = 1,
        )
        windows = replay_run(run, detector)
        hole = 2001:(2000+gap_rows)
        # Windows are scored on both sides of the hole and none whose
        # conditioning stretch touches it
        before = filter(r -> r.row_end + 1000 < first(hole), eachrow(windows))
        after = filter(r -> r.row_start - 1000 > last(hole), eachrow(windows))
        @test !isempty(before) && !isempty(after)
        @test length(before) + length(after) == nrow(windows)
        @test all(windows.coverage .== 1.0)
        # A window after the hole equals a direct evaluation on the rows its
        # batches actually hold
        w = after[1]
        stretch = payload[(w.row_start-1000):(w.row_end+1000)]
        @test isapprox(w.score, score_window(detector, stretch, 1001); atol = 1e-6)
        @test replay_run(run, detector).score == windows.score
        # Under the trailing estimate, the windows just after the hole have
        # fewer contiguous rows behind them than one segment and keep the
        # previous estimate of the delivered record instead of the static PSD
        # Under the trailing estimate the runs on both sides of the hole are
        # pooled, so the windows after the hole carry an estimate made
        # behind them, never the static PSD
        trailing = replay_run(run, detector; trailing_psd = TrailingWelch(3000, 500, 1024))
        rows_after = trailing.row_start .- 1000 .> last(hole)
        estimated = trailing.psd_row[rows_after]
        @test any(rows_after) && all(estimated .> last(hole))
        @test all(estimated .<= trailing.row_end[rows_after] .+ 1000)
        # With an edge trim of 300 rows the run before the hole (2000 rows)
        # no longer holds a segment beyond the trim, the run after it does
        # once 1624 rows are delivered, and the estimate is kept meanwhile
        trimmed = replay_run(
            run,
            detector;
            trailing_psd = TrailingWelch(3000, 500, 1024; edge_rows = 300),
        )
        @test trimmed.psd_row[rows_after][end] > last(hole)
        @test all(r -> r == 0 || r > last(hole), trimmed.psd_row[rows_after])
        @test_throws ArgumentError TrailingWelch(3000, 500, 1024; edge_rows = -1)
    end
end
