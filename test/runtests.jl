# The suite runs in its own environment (test/Project.toml, with the package
# consumed by path through [sources]) and loads MilliHertzQML as a real
# package, so static QA resolves the package identity and `Pkg.test` agrees
# with a direct `julia test/runtests.jl` invocation.
using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using Test
using Statistics, Random, TOML
using FFTW: rfft, rfftfreq
using CSV, DataFrames
using StableRNGs
using Aqua, JET, ExplicitImports
using MilliHertzQML
using Yao, Flux, Zygote

const PROJECT_ROOT = dirname(@__DIR__)

@testset "Static QA (Aqua)" begin
    # Scripts, benchmarks, and tests carry their own environments, so the
    # package dependency graph is exactly what src/ loads and the stale-deps
    # check runs unexempted. The persistent-tasks check precompiles a wrapper
    # package against the live registry; it is gated off CI, where it fails
    # for environmental reasons (see the sibling package's record), and runs
    # locally.
    Aqua.test_all(MilliHertzQML; persistent_tasks = get(ENV, "CI", "") != "true")
end

@testset "Static QA (ExplicitImports)" begin
    @test ExplicitImports.check_no_stale_explicit_imports(MilliHertzQML) === nothing
    @test ExplicitImports.check_no_implicit_imports(MilliHertzQML) === nothing
end

@testset "Static QA (JET)" begin
    # Reports are restricted to this package; dependencies are analyzed but
    # not reported against.
    JET.test_package(MilliHertzQML; target_modules = (MilliHertzQML,))
end

@testset "MilliHertzQML Tests (Multi-Qubit VQC)" begin
    rng = StableRNG(1234)

    @testset "Model Initialization" begin
        n_qubits = 4
        n_layers = 2
        model = VariationalQuantumClassifier(n_qubits, n_layers; rng = rng)
        @test length(model.params) == n_layers * 8 # 2 params per qubit (Ry, Rz) per layer
        @test model.n_qubits == 4
        @test length(model.ansatz_layers) == n_layers
        # Seeded construction is reproducible
        @test VariationalQuantumClassifier(4, 2; rng = StableRNG(7)).params ==
              VariationalQuantumClassifier(4, 2; rng = StableRNG(7)).params
    end

    @testset "Forward Pass Logic" begin
        model = VariationalQuantumClassifier(4, 2; rng = rng)
        x = rand(rng, Float32, 4)
        p = predict_probability(model, x)
        @test 0.0 <= p <= 1.0
    end

    @testset "Training & Gradients" begin
        model = VariationalQuantumClassifier(4, 2; rng = rng)
        X_batch = rand(rng, Float32, 4, 4)
        y_batch = [0, 1, 0, 1]

        opt_state = Flux.setup(Adam(0.1), model.params)
        l_init = loss_function(model, X_batch, y_batch)

        grads = Zygote.gradient(model) do m
            loss_function(m, X_batch, y_batch)
        end
        @test any(abs.(grads[1].params) .> 0.0)

        for _ in 1:20
            train_step!(model, opt_state, X_batch, y_batch)
        end

        l_final = loss_function(model, X_batch, y_batch)
        @test l_final < l_init
    end

    @testset "Feature Extraction" begin
        # Pure 2 mHz sine wave (milliHertz regime)
        fs = 0.2
        t = range(0, 5000, length = 1000)
        signal = sin.(2π * 0.002 * t)

        p_low, p_high, ent, std_psd = extract_features(signal, fs)

        @test p_low > p_high # 2 mHz lies in the low band (1-5 mHz)
        @test ent > 0.0
        @test std_psd isa Float32
    end

    @testset "Data Loading" begin
        mktempdir() do dir
            feat_path = joinpath(dir, "test_feats.csv")
            lab_path = joinpath(dir, "test_labs.csv")
            df_f = DataFrame(
                p_low = [0.5, 3.0, 1.0],
                p_high = [1.0, 0.9, 1.1],
                spectral_entropy = [0.95, 0.6, 0.9],
                log_power_std = [0.0, 0.8, 0.1],
            )
            CSV.write(feat_path, df_f)
            CSV.write(lab_path, DataFrame(Label = [0, 1, 0], SNR = [0.0, 12.0, 0.0]))

            X, y, df_l = load_data(feat_path, lab_path)
            @test X == Matrix{Float32}(df_f)
            @test y == [0, 1, 0]
            @test df_l.SNR == [0.0, 12.0, 0.0]
            # The label-free path yields the identical raw matrix
            @test load_features(feat_path) == X

            CSV.write(lab_path, DataFrame(Label = [0, 1]))
            @test_throws DimensionMismatch load_data(feat_path, lab_path)
        end
    end

    @testset "Input Validation" begin
        @test_throws ArgumentError VariationalQuantumClassifier(1, 2)
        @test_throws ArgumentError VariationalQuantumClassifier(4, 0)

        model = VariationalQuantumClassifier(4, 2; rng = rng)
        @test_throws DimensionMismatch predict_probability(model, rand(rng, Float32, 3))
        @test_throws DimensionMismatch loss_function(
            model,
            rand(rng, Float32, 4, 3),
            [0, 1, 0, 1],
        )
        @test_throws DimensionMismatch loss_function(
            model,
            rand(rng, Float32, 4, 4),
            [0, 1],
        )

        # 16 samples at 0.2 Hz resolve no bins inside the 1-5 mHz band
        @test_throws ArgumentError extract_features(randn(rng, 16), 0.2)
    end

    @testset "Feature Edge Cases" begin
        # A constant (zero) signal yields finite features
        feats = extract_features(zeros(Float64, 1000), 0.2)
        @test all(isfinite, feats)
    end

    @testset "Model Persistence" begin
        mktempdir() do dir
            model = VariationalQuantumClassifier(4, 3; rng = rng)
            x = rand(rng, Float32, 4)
            p_ref = predict_probability(model, x)
            scaler = FeatureScaler([0.0, 0.0, 0.0, -1.0], [3.0, 3.0, 1.0, 1.0])

            path = joinpath(dir, "model.jld2")
            meta_in = Dict("run_id" => "test", "seed" => 1234)
            save_model(path, model; metadata = meta_in, scaler = scaler)

            loaded, meta_out, scaler_out = load_model(path)
            @test loaded.n_qubits == model.n_qubits
            @test loaded.n_layers == model.n_layers
            @test loaded.params == model.params
            @test meta_out["run_id"] == "test"
            @test meta_out["seed"] == 1234
            @test scaler_out.lower == scaler.lower && scaler_out.upper == scaler.upper
            @test isapprox(predict_probability(loaded, x), p_ref; atol = 1e-6)

            # Artifacts without a scaler load with `nothing`
            bare = joinpath(dir, "bare.jld2")
            save_model(bare, model)
            @test load_model(bare)[3] === nothing
        end
    end
end

@testset "Noise model (Robson, Cornish & Liu 2019)" begin
    # Structural properties of the sensitivity curve
    @test instrument_psd(0.0) == Inf
    @test confusion_psd(-1.0) == Inf
    @test lisa_noise_psd(1e-3) > 0
    # The confusion foreground dominates the instrument term near 1 mHz for the
    # one-year fit and is negligible above 10 mHz
    @test confusion_psd(1e-3; observation_years = 1.0) > instrument_psd(1e-3)
    @test confusion_psd(1e-2; observation_years = 1.0) < 1e-3 * instrument_psd(1e-2)
    # More resolved and subtracted binaries with longer observation
    @test confusion_psd(1e-3; observation_years = 4.0) <
          confusion_psd(1e-3; observation_years = 0.5)
    # The sensitivity has its minimum in the milliHertz band
    @test lisa_noise_psd(1e-2) < lisa_noise_psd(3e-4)
    @test lisa_noise_psd(1e-2) < lisa_noise_psd(1e-1)
    # Hand-evaluated reference values of the published formulas
    # Reference values evaluated independently from the published formulas
    @test isapprox(instrument_psd(1e-3), 1.634101e-38; rtol = 1e-5)
    @test isapprox(confusion_psd(1e-3; observation_years = 1.0), 1.663516e-37; rtol = 1e-5)
    @test isapprox(instrument_psd(1e-2), 1.443169e-40; rtol = 1e-5)
    @test isapprox(confusion_psd(3e-3; observation_years = 1.0), 5.591648e-40; rtol = 1e-5)
    @test_throws ArgumentError confusion_psd(1e-3; observation_years = 3.0)
end

@testset "Noise synthesis calibration" begin
    rng = StableRNG(2026)
    fs = 0.2
    n = 2^15
    x = synthesize_noise(rng, n, fs)
    @test length(x) == n
    @test eltype(x) == Float64
    @test isapprox(mean(x), 0.0; atol = 3 * std(x) / sqrt(n))
    # Whitening the record turns the noise into unit-variance white noise;
    # 655 bins in 1-5 mHz give a 4 % standard error on the mean power
    w = whiten_record(x, fs)
    @test isapprox(var(w), 1.0; atol = 0.05)
    power = tapered_periodogram(w; taper = :none)
    freqs = rfftfreq(n, fs)
    inband = (freqs .>= 1e-3) .& (freqs .<= 5e-3)
    @test isapprox(mean(power[inband]), 1.0; atol = 0.15)
    @test power[1] == 0
    # Seeded synthesis is reproducible
    @test synthesize_noise(StableRNG(5), 64, fs) == synthesize_noise(StableRNG(5), 64, fs)
    # Bins below the synthesis floor carry no power
    floored = synthesize_noise(StableRNG(5), 4096, fs; f_min = 1e-3)
    spectrum = abs.(rfft(floored))
    low_bins = rfftfreq(4096, fs) .< 1e-3
    @test maximum(spectrum[low_bins]) < 1e-10 * maximum(spectrum)
    @test_throws ArgumentError synthesize_noise(rng, 64, fs; f_min = -1.0)
    @test_throws ArgumentError synthesize_noise(rng, 1, fs)
    @test_throws ArgumentError synthesize_noise(rng, 64, 0.0)
end

@testset "Tapered periodogram" begin
    rng = StableRNG(99)
    white = randn(rng, 8192)
    for taper in (:hann, :none)
        p = tapered_periodogram(white; taper = taper)
        @test length(p) == 4097
        @test p[1] == 0
        @test isapprox(mean(p[2:end]), 1.0; atol = 0.05)
    end
    @test_throws ArgumentError tapered_periodogram(white; taper = :tukey)
    @test_throws ArgumentError tapered_periodogram([1.0])
end

@testset "Matched-filter SNR" begin
    fs = 0.2
    n = 4096
    T = n / fs
    t = (0:(n-1)) ./ fs
    # An on-bin sinusoid of amplitude A has ρ = A sqrt(T / S_n(f0)) exactly
    k = 60
    f0 = k * fs / n
    A = 1e-20
    h = A .* cos.(2π * f0 .* t)
    ρ_expected = A * sqrt(T / lisa_noise_psd(f0))
    @test isapprox(matched_filter_snr(h, fs), ρ_expected; rtol = 1e-6)
    # Linear in amplitude and rescalable to a target
    @test isapprox(matched_filter_snr(3h, fs), 3ρ_expected; rtol = 1e-6)
    @test isapprox(matched_filter_snr(scale_to_snr(h, fs, 12.0), fs), 12.0; rtol = 1e-6)
    @test_throws ArgumentError scale_to_snr(zeros(n), fs, 10.0)
    @test_throws ArgumentError scale_to_snr(h, fs, 0.0)
end

@testset "Signal placement" begin
    strain = zeros(10)
    signal = [1.0, 2.0, 3.0, 4.0]
    # Anchor sample 3 of the signal on sample 2 of the record: sample 1 is dropped
    covered = place_signal!(strain, signal, 2, 3)
    @test covered == 1:3
    @test strain[1:3] == [2.0, 3.0, 4.0]
    @test all(iszero, strain[4:end])
    # Truncation at the end of the record
    strain2 = zeros(10)
    @test place_signal!(strain2, signal, 10, 1) == 10:10
    @test strain2[10] == 1.0
    # No overlap
    strain3 = zeros(10)
    @test isempty(place_signal!(strain3, signal, 20, 1))
    @test_throws BoundsError place_signal!(strain3, signal, 5, 9)
end

@testset "Whitened features" begin
    rng = StableRNG(11)
    fs = 0.2
    # Unit-mean band powers for noise, independent of the window length
    record = highpass_record(synthesize_noise(rng, 40000, fs), fs; cutoff = 5e-4)
    long = whiten_record(record, fs)
    p_low_long, p_high_long, ent_long, lstd_long = extract_features(long, fs)
    @test isapprox(p_low_long, 1.0; atol = 0.15)
    @test isapprox(p_high_long, 1.0; atol = 0.15)
    @test 0.85 < ent_long <= 1.0
    @test abs(lstd_long) < 0.15
    # Windows cut from the whitened record
    p_low_short, p_high_short, ent_short, _ = extract_features(view(long, 1:4000), fs)
    @test isapprox(p_low_short, 1.0; atol = 0.35)
    @test isapprox(p_high_short, 1.0; atol = 0.15)
    @test 0.8 < ent_short <= 1.0
    p_low_win, _, _, _ = extract_features(view(long, 10001:11000), fs)
    @test isapprox(p_low_win, 1.0; atol = 0.7)
    # A strong in-band sinusoid raises the low-band power and lowers the entropy
    t = (0:39999) ./ fs
    loud = long .+ cos.(2π * 2e-3 .* t)
    p_low_loud, p_high_loud, ent_loud, _ = extract_features(loud, fs)
    @test p_low_loud > 5 * p_low_long
    @test isapprox(p_high_loud, p_high_long; atol = 0.3)
    @test ent_loud < ent_long
    @test_throws ArgumentError extract_features(long, 0.0)
end

@testset "IMRPhenomA waveform" begin
    fs = 0.2
    p6 = phenoma_parameters(1e6, 1.0)
    @test p6.η ≈ 0.25
    @test p6.f_merg < p6.f_ring < p6.f_cut
    # Transition frequencies scale inversely with the total mass
    @test phenoma_parameters(2e6, 1.0).f_merg ≈ p6.f_merg / 2
    @test isapprox(p6.f_merg, 8.10e-3; rtol = 1e-2)   # 0.1254 / (π M) for η = 1/4
    # Amplitude continuity at the transitions and zero beyond the cutoff
    @test isapprox(phenoma_amplitude(p6.f_merg, p6), 1.0; atol = 1e-12)
    ε = 1e-9
    @test isapprox(
        phenoma_amplitude(p6.f_ring - ε, p6),
        phenoma_amplitude(p6.f_ring + ε, p6);
        rtol = 1e-6,
    )
    @test phenoma_amplitude(p6.f_cut, p6) == 0
    @test phenoma_amplitude(0.0, p6) == 0

    h, k_m, p = phenoma_waveform(fs, 2 * 86400; total_mass = 1e6, mass_ratio = 1.0)
    n = length(h)
    @test n == 34560
    @test maximum(abs, h) == 1
    # The amplitude peak lands at the target merger time
    T = n / fs
    t_pad = clamp(max(0.05 * T, 20 / (π * p.sigma)), 0.0, 0.5 * T)
    # (the amplitude peak precedes the arrival of the ringdown frequency by
    # some ten total masses; a merger above the Nyquist taper adds a few samples)
    @test abs(k_m - (round(Int, (T - t_pad) * fs) + 1)) / fs <= 20 * p.M_sec + 10 / fs
    # Forward chirp: the zero-crossing frequency rises towards the merger
    function zc_frequency(seg)
        return count(j -> sign(seg[j]) != sign(seg[j-1]), 2:length(seg)) / 2 /
               (length(seg) / fs)
    end
    f_early = zc_frequency(view(h, div(k_m, 4):div(k_m, 2)))
    f_late = zc_frequency(view(h, (k_m-400):k_m))
    @test f_late > 3 * f_early
    # Inspiral spectral slope −7/6 on the realized spectrum
    Hs = abs.(rfft(h))
    fr = rfftfreq(n, fs)
    f_lo = 2 * phenoma_start_frequency(p, (k_m - 1) / fs)
    f_hi = p.f_merg / 2
    band = (fr .>= f_lo) .& (fr .<= f_hi)
    X = log.(fr[band])
    Y = log.(Hs[band])
    @test isapprox(cov(X, Y) / var(X), -7 / 6; atol = 0.15)
    # No power above the Nyquist taper
    @test maximum(Hs[fr .> 0.9*fs/2]) < 1e-3 * maximum(Hs)
    # The segment starts quietly (roll-on and ramp)
    @test maximum(abs, view(h, 1:100)) < 0.05
    # Heavy and light binaries generate and place their peak correctly
    for (M, q) in ((1e7, 4.0), (1e5, 1.0))
        hh, kk, pp = phenoma_waveform(fs, 2 * 86400; total_mass = M, mass_ratio = q)
        Tn = length(hh) / fs
        tp = clamp(max(0.05 * Tn, 20 / (π * pp.sigma)), 0.0, 0.5 * Tn)
        @test abs(kk - (round(Int, (Tn - tp) * fs) + 1)) / fs <= 20 * pp.M_sec + 10 / fs
    end
    @test_throws ArgumentError phenoma_parameters(0.0, 1.0)
    @test_throws ArgumentError phenoma_parameters(1e6, 0.5)
    @test_throws ArgumentError phenoma_waveform(
        fs,
        1000.0;
        total_mass = 1e6,
        mass_ratio = 1.0,
        nyquist_taper = 1.5,
    )
end

@testset "Detectable span" begin
    fs = 0.2
    n = 20000
    t = (0:(n-1)) ./ fs
    placed = zeros(n)
    covered = 8001:9000
    placed[covered] .= cos.(2π * 3e-3 .* t[covered])
    # Scale so that a window holding the whole burst has SNR 20
    ρ_full = matched_filter_snr(view(placed, 8001:9000), fs)
    placed .*= 20 / ρ_full
    span = detectable_span(placed, covered, fs, 1000, 5.0; step = 10)
    @test span !== nothing
    @test first(span) < first(covered) && last(span) > last(covered)
    @test first(span) >= first(covered) - 999 && last(span) <= last(covered) + 999
    # A higher threshold narrows the span, an unreachable one empties it
    narrow = detectable_span(placed, covered, fs, 1000, 15.0; step = 10)
    @test narrow !== nothing && length(narrow) < length(span)
    @test detectable_span(placed, covered, fs, 1000, 1e6; step = 10) === nothing
    @test detectable_span(placed, 5:4, fs, 1000, 5.0) === nothing
    @test_throws ArgumentError detectable_span(placed, covered, fs, 1, 5.0)
    @test_throws ArgumentError detectable_span(placed, covered, fs, 1000, 0.0)
end

@testset "Record high-pass" begin
    fs = 0.2
    n = 20000
    t = (0:(n-1)) ./ fs
    low = cos.(2π * 1e-4 .* t)
    inband = cos.(2π * 5e-3 .* t)
    y = highpass_record(low .+ inband, fs; cutoff = 5e-4, order = 8)
    # The 0.1 mHz component is suppressed by more than 1e5 in power, the 5 mHz
    # component is preserved to better than 1 %
    Xy = abs.(rfft(y))
    Xl = abs.(rfft(low .+ inband))
    k_low = round(Int, 1e-4 * n / fs) + 1
    k_in = round(Int, 5e-3 * n / fs) + 1
    @test (Xy[k_low] / Xl[k_low])^2 < 1e-5
    @test isapprox(Xy[k_in] / Xl[k_in], 1.0; atol = 1e-2)
    @test highpass_record(inband, fs; cutoff = 0.0) == inband
    @test_throws ArgumentError highpass_record(inband, fs; cutoff = -1.0)
    @test_throws ArgumentError highpass_record(inband, fs; cutoff = 1e-3, order = 0)
end

@testset "Feature scaler" begin
    rng = StableRNG(3)
    X = randn(rng, 500, 4) .* [1.0 10.0 0.1 100.0] .+ [0.0 5.0 0.5 -50.0]
    scaler = fit_scaler(X; quantiles = (0.01, 0.99))
    E = encode_features(scaler, X)
    @test size(E) == size(X)
    @test eltype(E) == Float32
    @test all(0 .<= E .<= Float32(2π))
    @test minimum(E) == 0.0f0 && isapprox(maximum(E), 2π; atol = 1e-5)
    # The bounds map to the interval ends; outside values clamp
    @test encode_features(scaler, reshape(scaler.lower, 1, :)) == zeros(Float32, 1, 4)
    @test all(
        isapprox.(encode_features(scaler, reshape(scaler.upper, 1, :)), 2π; atol = 1e-5),
    )
    @test all(encode_features(scaler, fill(1e9, 1, 4)) .≈ Float32(2π))
    @test_throws ArgumentError fit_scaler(hcat(X, ones(500)))
    @test_throws ArgumentError fit_scaler(X; quantiles = (0.9, 0.1))
    @test_throws DimensionMismatch encode_features(scaler, X[:, 1:3])
    @test_throws ArgumentError FeatureScaler([0.0, 1.0], [1.0, 1.0])
end

@testset "Pipeline smoke test" begin
    # The four scripts run as child processes on a three-day configuration
    # whose [paths] section points into a temporary directory; nothing is
    # written into the project tree. Each script activates the scripts
    # environment itself.
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    scripts = joinpath(PROJECT_ROOT, "scripts")
    mktempdir() do dir
        cfg = TOML.parsefile(joinpath(PROJECT_ROOT, "config.toml"))
        inputs = joinpath(dir, "inputs")
        cfg["paths"] = Dict(
            "inputs" => inputs,
            "models" => joinpath(dir, "models"),
            "plots" => joinpath(dir, "plots"),
            "results" => joinpath(dir, "results"),
        )
        h5 = joinpath(inputs, "smoke_telemetry.h5")
        raw_labels = replace(h5, ".h5" => "_labels.csv")
        merge!(
            cfg["generation"],
            Dict(
                "days" => 3.0,
                "n_mbhb" => 2,
                "n_gbs" => 5,
                "n_emris" => 1,
                "output" => h5,
            ),
        )
        cfg["preprocessing"]["output_prefix"] = "smoke"
        feats = joinpath(inputs, "smoke_features.csv")
        labs = joinpath(inputs, "smoke_labels.csv")
        merge!(
            cfg["training"],
            Dict(
                "train_features" => feats,
                "train_labels" => labs,
                "epochs" => 2,
                "batch_size" => 32,
                "patience" => 2,
            ),
        )
        merge!(cfg["inference"], Dict("features" => feats, "labels" => labs))
        cfg_path = joinpath(dir, "config.toml")
        open(cfg_path, "w") do io
            TOML.print(io, cfg)
        end
        log = joinpath(dir, "pipeline.log")

        function stage(script, args...)
            cmd = `$julia --startup-file=no $(joinpath(scripts, script)) --config $cfg_path $(collect(args))`
            ok = success(pipeline(cmd; stdout = log, stderr = log, append = true))
            ok || println(read(log, String))
            @test ok
            return ok
        end

        fs = cfg["generation"]["fs"]
        n_total = round(Int, 3.0 * 24 * 3600 * fs)
        n_windows =
            div(
                n_total - cfg["preprocessing"]["window_size"],
                cfg["preprocessing"]["step_size"],
            ) + 1

        stage("generate_data.jl", "--run-id", "smoke") || return
        @test isfile(h5)
        @test nrow(CSV.read(raw_labels, DataFrame)) == n_total
        events = CSV.read(replace(h5, ".h5" => "_events.csv"), DataFrame)
        @test nrow(events) == 2
        @test all(8.0 .<= events.snr .<= 50.0)
        @test all(events.start_index .<= events.merger_index .<= events.end_index)
        @test all(1e5 .<= events.total_mass_msun .<= 1e7)
        @test all(events.label_start_index .>= events.start_index .- 999)
        @test all(events.label_end_index .<= events.end_index .+ 999)

        stage("preprocess_ldc.jl", "--h5-file", h5, "--label-file", raw_labels) || return
        @test nrow(CSV.read(feats, DataFrame)) == n_windows
        @test nrow(CSV.read(labs, DataFrame)) == n_windows

        stage("train.jl", "--run-id", "smoke") || return
        model_path = joinpath(dir, "models", "run_smoke", "gw_model.jld2")
        @test isfile(model_path)
        @test isfile(joinpath(dir, "models", "run_smoke", "config.toml"))

        stage("infer.jl", "--run-id", "smoke") || return
        probs = CSV.read(
            joinpath(dir, "results", "run_smoke", "inference_probabilities.csv"),
            DataFrame,
        )
        @test nrow(probs) == n_windows
        @test all(0 .<= probs.Probability .<= 1)
        @test isfile(joinpath(dir, "models", "run_smoke", "threshold.toml"))

        stage(
            "infer.jl",
            "--model",
            model_path,
            "--run-id",
            "smoke_blind",
            "--labels",
            "",
        ) || return
        blind =
            TOML.parsefile(joinpath(dir, "results", "run_smoke_blind", "config_infer.toml"))
        @test blind["inference"]["blind"] == true
        blind_probs = CSV.read(
            joinpath(dir, "results", "run_smoke_blind", "inference_probabilities.csv"),
            DataFrame,
        )
        @test nrow(blind_probs) == n_windows

        # The project tree received nothing
        @test !isdir(joinpath(PROJECT_ROOT, "models", "run_smoke"))
        @test !isfile(joinpath(PROJECT_ROOT, "data", "inputs", "smoke_features.csv"))
    end
end
