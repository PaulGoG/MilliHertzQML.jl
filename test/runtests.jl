# The suite runs in its own environment (test/Project.toml, with the package
# consumed by path through [sources]) and loads MilliHertzQML as a real
# package, so static QA resolves the package identity and `Pkg.test` agrees
# with a direct `julia test/runtests.jl` invocation.
include(joinpath(@__DIR__, "activate.jl"))

using Test
using Statistics, Random, TOML, Dates
using FFTW: rfft, rfftfreq
using CSV, DataFrames
using StableRNGs
using Aqua, JET, ExplicitImports
using MilliHertzQML
using Yao, Flux, Zygote
using CairoMakie: CairoMakie
using DeepSpaceTelemetry: DeepSpaceTelemetry
using CurvatureDistinguishability: CurvatureDistinguishability

const PROJECT_ROOT = dirname(@__DIR__)
# The pipeline root of the whole run, child processes included: a sandboxed
# test environment (Pkg.test) lies outside the repository
ENV["STREAMINGINFERENCE_ROOT"] = PROJECT_ROOT

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
    # Reports are restricted to this package and its two layers; dependencies
    # are analysed but not reported against.
    JET.test_package(
        MilliHertzQML;
        target_modules = (
            MilliHertzQML,
            MilliHertzQML.StreamingInference,
            MilliHertzQML.MilliHertzBase,
        ),
    )
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

        # The batch loss is the mean of the sample losses; the weighted
        # cross-entropy is the plain one at unit weight
        @test weighted_bce(0.5, 1) ≈ log(2) atol = 1e-6
        @test weighted_bce(0.5, 0) ≈ log(2) atol = 1e-6
        @test weighted_bce(0.5, 1; positive_weight = 3) ≈ 3 * log(2) atol = 1e-6
        @test weighted_bce(1.0, 1) < 1e-6 && weighted_bce(0.0, 1) > 10
        @test sum(sample_loss(model, X_batch[k, :], y_batch[k]) for k in 1:4) / 4 ≈
              loss_function(model, X_batch, y_batch)

        # Threaded batch gradient: same loss, the gradient of the serial tape
        # up to accumulation rounding, deterministic across calls
        X_big = rand(rng, Float32, 24, 4)
        y_big = rand(rng, 0:1, 24)
        l_serial, g_serial =
            batch_gradient(model, X_big, y_big; positive_weight = 2.0, threaded = false)
        l_threads, g_threads =
            batch_gradient(model, X_big, y_big; positive_weight = 2.0, threaded = true)
        @test l_serial ≈ loss_function(model, X_big, y_big; positive_weight = 2.0)
        @test l_threads ≈ l_serial rtol = 1e-5
        @test length(g_threads) == length(model.params)
        @test isapprox(g_threads, g_serial; rtol = 1e-4, atol = 1e-6)
        @test g_threads ==
              batch_gradient(model, X_big, y_big; positive_weight = 2.0, threaded = true)[2]
        @test g_serial == Zygote.gradient(
            m -> loss_function(m, X_big, y_big; positive_weight = 2.0),
            model,
        )[1].params
        @test_throws ArgumentError batch_gradient(model, zeros(Float32, 0, 4), Int[])
        # One training step from the same state along either path lands on
        # the same parameters to floating-point tolerance. The R_z rotations
        # of the last layer commute with the Z measurement (pulled back
        # through the CNOT ring, every measured Z_k is a product of Z's), so
        # their gradient vanishes and Adam's normalisation turns rounding
        # noise into an arbitrary step: they are pinned at zero gradient and
        # excluded from the comparison.
        m_serial = VariationalQuantumClassifier(4, 2; rng = StableRNG(5))
        m_threads = VariationalQuantumClassifier(4, 2; rng = StableRNG(5))
        g_ref = batch_gradient(m_serial, X_big, y_big; threaded = false)[2]
        @test all(abs.(g_ref[(end-3):end]) .< 1e-6)
        @test count(abs.(g_ref) .> 1e-5) >= 8
        live = abs.(g_ref) .> 1e-5
        train_step!(
            m_serial,
            Flux.setup(Adam(0.1), m_serial.params),
            X_big,
            y_big;
            threaded = false,
        )
        train_step!(
            m_threads,
            Flux.setup(Adam(0.1), m_threads.params),
            X_big,
            y_big;
            threaded = true,
        )
        @test isapprox(
            m_serial.params[live],
            m_threads.params[live];
            rtol = 1e-4,
            atol = 1e-6,
        )

        # Threaded forward passes reproduce the serial ones exactly
        p_serial = MilliHertzQML.predict_all(model, X_big; threaded = false)
        p_threads = MilliHertzQML.predict_all(model, X_big; threaded = true)
        @test p_threads == p_serial
        @test p_serial == [predict_probability(model, X_big[k, :]) for k in 1:24]
        loss_val, acc_val = MilliHertzQML.epoch_validation(
            model,
            X_big,
            y_big;
            positive_weight = 2.0,
            threaded = true,
        )
        @test loss_val ≈ loss_function(model, X_big, y_big; positive_weight = 2.0) rtol =
            1e-5
        @test acc_val ≈ accuracy(model, X_big, y_big)
        @test_throws DimensionMismatch MilliHertzQML.epoch_validation(
            model,
            X_big,
            y_big[1:5];
            positive_weight = 1.0,
        )
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
            @test scaler_out.phase_span == scaler.phase_span == Float32(π)
            @test isapprox(predict_probability(loaded, x), p_ref; atol = 1e-6)
            # The span is persisted with the bounds
            wide = FeatureScaler(scaler.lower, scaler.upper; phase_span = 2π)
            save_model(joinpath(dir, "wide.jld2"), model; scaler = wide)
            @test load_model(joinpath(dir, "wide.jld2"))[3].phase_span == Float32(2π)
            # An artifact written before the span was persisted restores the
            # full period its scaler was trained with, and says so
            legacy = joinpath(dir, "legacy.jld2")
            MilliHertzQML.JLD2.jldsave(
                legacy;
                n_qubits = model.n_qubits,
                n_layers = model.n_layers,
                params = model.params,
                scaler_lower = scaler.lower,
                scaler_upper = scaler.upper,
                metadata = Dict{String,Any}("run_id" => "old"),
            )
            _, meta_legacy, scaler_legacy =
                @test_logs (:warn, r"predates the persisted phase-encoding span") load_model(
                    legacy,
                )
            @test meta_legacy["run_id"] == "old"
            @test scaler_legacy.phase_span == Float32(2π)

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
    # Far above the knee the direct product would be 0 × Inf; the log-space
    # evaluation vanishes and the sensitivity stays finite up to 10 Hz
    @test confusion_psd(10.0; observation_years = 1.0) == 0.0
    @test all(isfinite, lisa_noise_psd.((0.05, 0.5, 2.0, 10.0)))
    # Agreement with the direct product where both are finite
    p = MilliHertzQML.MilliHertzBase.confusion_fit(1.0)
    f = 2e-3
    direct =
        MilliHertzQML.MilliHertzBase.CONFUSION_AMPLITUDE *
        f^(-7 / 3) *
        exp(-f^p.α + p.β * f * sin(p.κ * f)) *
        (1 + tanh(p.γ * (p.f_k - f)))
    @test isapprox(confusion_psd(f), direct; rtol = 1e-10)
end

@testset "Noise synthesis calibration" begin
    rng = StableRNG(2026)
    fs = 0.2
    n = 2^15
    x = synthesize_noise(rng, n, fs; psd = lisa_noise_psd)
    @test length(x) == n
    @test eltype(x) == Float64
    @test isapprox(mean(x), 0.0; atol = 3 * std(x) / sqrt(n))
    # Whitening the record turns the noise into unit-variance white noise;
    # 655 bins in 1-5 mHz give a 4 % standard error on the mean power
    w = whiten_record(x, fs; psd = lisa_noise_psd)
    @test isapprox(var(w), 1.0; atol = 0.05)
    power = tapered_periodogram(w; taper = :none)
    freqs = rfftfreq(n, fs)
    inband = (freqs .>= 1e-3) .& (freqs .<= 5e-3)
    @test isapprox(mean(power[inband]), 1.0; atol = 0.15)
    @test power[1] == 0
    # Seeded synthesis is reproducible
    @test synthesize_noise(StableRNG(5), 64, fs; psd = lisa_noise_psd) ==
          synthesize_noise(StableRNG(5), 64, fs; psd = lisa_noise_psd)
    # Bins below the synthesis floor carry no power
    floored = synthesize_noise(StableRNG(5), 4096, fs; f_min = 1e-3, psd = lisa_noise_psd)
    spectrum = abs.(rfft(floored))
    low_bins = rfftfreq(4096, fs) .< 1e-3
    @test maximum(spectrum[low_bins]) < 1e-10 * maximum(spectrum)
    @test_throws ArgumentError synthesize_noise(
        rng,
        64,
        fs;
        f_min = -1.0,
        psd = lisa_noise_psd,
    )
    @test_throws ArgumentError synthesize_noise(rng, 1, fs; psd = lisa_noise_psd)
    @test_throws ArgumentError synthesize_noise(rng, 64, 0.0; psd = lisa_noise_psd)
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
    @test isapprox(matched_filter_snr(h, fs; psd = lisa_noise_psd), ρ_expected; rtol = 1e-6)
    # Linear in amplitude and rescalable to a target
    @test isapprox(
        matched_filter_snr(3h, fs; psd = lisa_noise_psd),
        3ρ_expected;
        rtol = 1e-6,
    )
    @test isapprox(
        matched_filter_snr(
            scale_to_snr(h, fs, 12.0; psd = lisa_noise_psd),
            fs;
            psd = lisa_noise_psd,
        ),
        12.0;
        rtol = 1e-6,
    )
    @test_throws ArgumentError scale_to_snr(zeros(n), fs, 10.0; psd = lisa_noise_psd)
    @test_throws ArgumentError scale_to_snr(h, fs, 0.0; psd = lisa_noise_psd)
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
    record = highpass_record(
        synthesize_noise(rng, 40000, fs; psd = lisa_noise_psd),
        fs;
        cutoff = 5e-4,
    )
    long = whiten_record(record, fs; psd = lisa_noise_psd)
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
    strong = long .+ cos.(2π * 2e-3 .* t)
    p_low_strong, p_high_strong, ent_strong, _ = extract_features(strong, fs)
    @test p_low_strong > 5 * p_low_long
    @test isapprox(p_high_strong, p_high_long; atol = 0.3)
    @test ent_strong < ent_long
    @test_throws ArgumentError extract_features(long, 0.0)

    # The bands set with the default edges reproduces the whitened set exactly
    bands =
        extract_features(long, fs; feature_set = :bands, band_edges = [1e-3, 5e-3, 1e-1])
    @test collect(bands) == collect(extract_features(long, fs))
    @test feature_names(:bands; n_bands = 2) ==
          [:p_band_1, :p_band_2, :spectral_entropy, :log_power_std]
    @test length(feature_names(:bands; n_bands = 5)) == 7
    @test feature_names(:whitened; n_bands = 5) == feature_names(:whitened)
    # Finer bands: unit mean on white noise; a 1.5 mHz tone lands in the
    # (1, 2] mHz band and leaves the others alone
    edges = [5e-4, 1e-3, 2e-3, 4e-3, 1e-2, 4e-2]
    fine = extract_features(long, fs; feature_set = :bands, band_edges = edges)
    @test length(fine) == 7
    @test all(isapprox.(fine[1:5], 1.0; atol = 0.35))
    tone = long .+ cos.(2π * 1.5e-3 .* t)
    fine_tone = extract_features(tone, fs; feature_set = :bands, band_edges = edges)
    @test fine_tone[2] > 5 * fine[2]
    @test isapprox(fine_tone[4], fine[4]; atol = 0.3) && fine_tone[6] < fine[6]
    @test_throws ArgumentError extract_features(long, fs; feature_set = :bands)
    @test_throws ArgumentError extract_features(
        long,
        fs;
        feature_set = :bands,
        band_edges = [5e-3, 1e-3],
    )
    @test_throws ArgumentError extract_features(
        view(long, 1:1000),
        fs;
        feature_set = :bands,
        band_edges = [1e-5, 5e-5, 1e-1],
    )
    @test_throws ArgumentError feature_names(:bands; n_bands = 0)
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
    # Inspiral spectral slope −7/6 on the realised spectrum
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
    ρ_full = matched_filter_snr(view(placed, 8001:9000), fs; psd = lisa_noise_psd)
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
    @test scaler.phase_span == Float32(π)
    E = encode_features(scaler, X)
    @test size(E) == size(X)
    @test eltype(E) == Float32
    @test all(0 .<= E .<= Float32(π))
    @test minimum(E) == 0.0f0 && isapprox(maximum(E), π; atol = 1e-5)
    # The bounds map to the interval ends; outside values clamp
    @test encode_features(scaler, reshape(scaler.lower, 1, :)) == zeros(Float32, 1, 4)
    @test all(
        isapprox.(encode_features(scaler, reshape(scaler.upper, 1, :)), π; atol = 1e-5),
    )
    @test all(encode_features(scaler, fill(1e9, 1, 4)) .≈ Float32(π))
    # The full period is admissible for artifacts trained on it, and folds
    # the two clamp ends onto one state: R_z(2π) = −I is a global phase, so
    # a saturated feature scores exactly as one at the floor
    legacy = fit_scaler(X; quantiles = (0.01, 0.99), phase_span = 2π)
    @test legacy.phase_span == Float32(2π)
    @test isapprox(maximum(encode_features(legacy, X)), 2π; atol = 1e-5)
    model = VariationalQuantumClassifier(4, 2; rng = rng)
    x_floor = fill(0.0f0, 4)
    x_wide = vec(encode_features(legacy, fill(1e9, 1, 4)))
    x_half = vec(encode_features(scaler, fill(1e9, 1, 4)))
    @test isapprox(
        predict_probability(model, x_wide),
        predict_probability(model, x_floor);
        atol = 1e-5,
    )
    @test abs(predict_probability(model, x_half) - predict_probability(model, x_floor)) >
          1e-3
    @test_throws ArgumentError FeatureScaler([0.0], [1.0]; phase_span = 3π)
    @test_throws ArgumentError FeatureScaler([0.0], [1.0]; phase_span = 0.0)
    @test_throws ArgumentError fit_scaler(hcat(X, ones(500)))
    @test_throws ArgumentError fit_scaler(X; quantiles = (0.9, 0.1))
    @test_throws DimensionMismatch encode_features(scaler, X[:, 1:3])
    @test_throws ArgumentError FeatureScaler([0.0, 1.0], [1.0, 1.0])
end

@testset "Threshold file migration" begin
    info = Dict{String,Any}(
        "value" => 0.5,
        "validation_recall" => 1.0,
        "validation_windows" => 10,
        "fit_fpr" => 0.0,
        "validation_fpr" => 0.1,
    )
    @test_logs (:info, r"predates the configurable fitting block") MilliHertzQML.migrate_threshold_info!(
        info,
    )
    @test info["fit_recall"] == 1.0 && info["fit_windows"] == 10
    @test !haskey(info, "validation_recall") && !haskey(info, "validation_windows")
    # A key already present under its current name is kept as it is
    @test info["fit_fpr"] == 0.0 && info["validation_fpr"] == 0.1
    @test_logs MilliHertzQML.migrate_threshold_info!(Dict{String,Any}("value" => 0.5))
    mktempdir() do dir
        open(joinpath(dir, "threshold.toml"), "w") do io
            TOML.print(
                io,
                Dict(
                    "threshold" =>
                        Dict("value" => 0.42, "validation_false_alarms_per_30d" => 1.5),
                ),
            )
        end
        thr, old = MilliHertzQML.load_threshold(dir)
        @test thr == 0.42f0 && old["fit_false_alarms_per_30d"] == 1.5
    end
end

@testset "Evaluation protocol" begin
    # Chronological split: 70/15/15 of 100 windows with a five-window buffer
    blocks = chronological_split(100; buffer = 5)
    @test blocks.train == 1:70
    @test blocks.validation == 76:90
    @test blocks.test == 96:100
    @test chronological_split(20).test == 18:20
    @test_throws ArgumentError chronological_split(10; buffer = 10)
    @test_throws ArgumentError chronological_split(100; train_fraction = 0.9)
    @test_throws ArgumentError chronological_split(100; train_fraction = 0.0)
    @test_throws ArgumentError chronological_split(100; buffer = -1)

    # Calibration block: the validation block alone, or validation and test
    # pooled across the buffer so that the range stays contiguous in time
    @test threshold_rows(blocks, "validation") == 76:90
    @test threshold_rows(blocks, "held_out") == 76:100
    @test length(threshold_rows(blocks, "held_out")) ==
          length(blocks.validation) + length(blocks.test) + 5
    @test_throws ArgumentError threshold_rows(blocks, "test")

    # ROC: one misordered positive among six windows
    y6 = [1, 1, 0, 1, 0, 0]
    s6 = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1]
    fpr, tpr, thr = roc_curve(y6, s6)
    @test thr == [Inf, 0.9, 0.8, 0.7, 0.4, 0.3, 0.1]
    @test fpr ≈ [0, 0, 0, 1, 1, 2, 3] ./ 3
    @test tpr ≈ [0, 1, 2, 2, 3, 3, 3] ./ 3
    @test roc_auc(fpr, tpr) ≈ 8 / 9
    @test roc_auc(roc_curve([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])[1:2]...) == 1.0
    @test roc_auc(roc_curve([1, 0, 1, 0], fill(0.5, 4))[1:2]...) ≈ 0.5
    @test isnan(roc_auc(roc_curve([1, 1], [0.2, 0.3])[1:2]...))
    @test_throws DimensionMismatch roc_curve([1, 0], [0.5])

    # Contiguous runs
    @test contiguous_runs([false, true, true, false, true]) == [2:3, 5:5]
    @test isempty(contiguous_runs(falses(3)))
    @test contiguous_runs(trues(4)) == [1:4]

    # Event metrics: two events (3:5, 9:10), one detected; alarms at 4 and at
    # 7:8 (one false-alarm episode); half-day windows, hence six days
    labels = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    decisions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    m = event_metrics(decisions, labels; step_size = 43200, sample_rate = 1.0)
    @test m.precision ≈ 1 / 3
    @test m.recall ≈ 1 / 5
    @test m.f1 ≈ 1 / 4
    @test m.balanced_accuracy ≈ 16 / 35
    @test m.n_events == 2 && m.n_detected == 1 && m.event_recall == 0.5
    @test m.n_false_alarm_episodes == 1
    @test m.observation_days ≈ 6.0
    @test m.false_alarms_per_30d ≈ 5.0
    # A permanent alarm detects every event and counts one false-alarm
    # episode in every unlabelled stretch (before, between, after)
    m_all = event_metrics(ones(Int, 12), labels; step_size = 43200, sample_rate = 1.0)
    @test m_all.n_detected == 2 && m_all.n_false_alarm_episodes == 3
    @test m_all.recall == 1.0
    m_none = event_metrics(zeros(Int, 12), labels; step_size = 1, sample_rate = 1.0)
    @test isnan(m_none.precision) && m_none.recall == 0.0
    @test_throws DimensionMismatch event_metrics(
        [1, 0],
        [1];
        step_size = 1,
        sample_rate = 1.0,
    )
    @test_throws ArgumentError event_metrics([1], [1]; step_size = 0, sample_rate = 1.0)

    # Threshold selection on a validation block of twenty half-day windows
    # (ten days): one event at 8:10 (scores 0.90, 0.95, 0.85), two spurious
    # noise scores 0.60 (window 3) and 0.70 (window 15), the rest below 0.25
    scores = [
        0.10,
        0.12,
        0.60,
        0.11,
        0.13,
        0.14,
        0.15,
        0.90,
        0.95,
        0.85,
        0.16,
        0.17,
        0.18,
        0.19,
        0.70,
        0.20,
        0.21,
        0.22,
        0.23,
        0.24,
    ]
    yv = zeros(Int, 20)
    yv[8:10] .= 1
    geometry = (step_size = 43200, sample_rate = 1.0)
    # The sweep: ascending candidates, monotone window rates, one event
    sweep = threshold_sweep(yv, scores; geometry...)
    @test issorted(sweep.threshold) && allunique(sweep.threshold)
    @test issorted(sweep.fpr; rev = true) && issorted(sweep.recall; rev = true)
    @test all(==(1), sweep.n_events)
    @test first(sweep.threshold) == 0.10 && first(sweep.recall) == 1.0
    @test first(sweep.fpr) == 1.0 && first(sweep.n_false_alarm_episodes) == 2
    @test last(sweep.threshold) == 0.95 && last(sweep.recall) ≈ 1 / 3
    @test last(sweep.fpr) == 0.0 && last(sweep.event_recall) == 1.0
    @test all(sweep.false_alarms_per_30d .≈ sweep.n_false_alarm_episodes .* 3.0)
    @test_throws ArgumentError threshold_sweep(Int[], Float64[]; geometry...)
    @test_throws ArgumentError threshold_sweep(yv, scores; n_candidates = 1, geometry...)
    @test_throws DimensionMismatch threshold_sweep(yv, scores[1:19]; geometry...)
    # far, one episode per 30 d: no episode is admissible on ten days, so the
    # lowest threshold clearing both spurious scores is chosen
    t_far, info = select_threshold(
        yv,
        scores;
        criterion = "far",
        target_far_per_30d = 1.0,
        geometry...,
    )
    @test 0.70 < t_far <= 0.85
    @test info["criterion"] == "far"
    @test info["fit_recall"] == 1.0
    @test info["fit_fpr"] == 0.0
    @test info["fit_false_alarms_per_30d"] == 0.0
    # far, three episodes per 30 d: one episode (window 15) is admitted once
    # the duty-cycle guard allows one of seventeen negatives
    t_far3, info3 = select_threshold(
        yv,
        scores;
        criterion = "far",
        target_far_per_30d = 3.0,
        target_fpr = 0.1,
        geometry...,
    )
    @test 0.60 < t_far3 <= 0.70
    @test info3["fit_false_alarms_per_30d"] ≈ 3.0
    @test info3["fit_fpr"] ≈ 1 / 17
    # The duty-cycle guard alone (every episode rate admissible) stops at
    # the first false positive
    t_guard, _ = select_threshold(
        yv,
        scores;
        criterion = "far",
        target_far_per_30d = 100.0,
        target_fpr = 0.05,
        geometry...,
    )
    @test 0.70 < t_guard <= 0.85
    # Descending scan: with the event at the start of the block, a permanent
    # alarm is one long episode (3 per 30 d) and admissible by rate alone;
    # the operating point must nevertheless stay on the branch of short
    # episodes, above the second spurious score
    y_head = zeros(Int, 20)
    y_head[1:3] .= 1
    s_head = fill(0.2, 20)
    s_head[1:3] .= [0.90, 0.95, 0.85]
    s_head[8] = 0.60
    s_head[15] = 0.70
    t_head, info_head = select_threshold(
        y_head,
        s_head;
        criterion = "far",
        target_far_per_30d = 3.0,
        target_fpr = 1.0,
        geometry...,
    )
    @test 0.60 < t_head <= 0.70
    @test info_head["fit_false_alarms_per_30d"] ≈ 3.0
    @test event_metrics(ones(Int, 20), y_head; geometry...).false_alarms_per_30d ≈ 3.0
    # fpr: 5 % of 17 negatives admits none, 10 % admits one
    t_fpr, _ =
        select_threshold(yv, scores; criterion = "fpr", target_fpr = 0.05, geometry...)
    @test 0.70 < t_fpr <= 0.85
    t_fpr10, _ =
        select_threshold(yv, scores; criterion = "fpr", target_fpr = 0.10, geometry...)
    @test 0.60 < t_fpr10 <= 0.70
    # youden: TPR - FPR peaks at the lowest event score
    t_youden, info_y = select_threshold(yv, scores; criterion = "youden", geometry...)
    @test t_youden == 0.85
    @test info_y["criterion"] == "youden"
    # youden without positives warns and falls back to fpr
    y0 = zeros(Int, 20)
    t_fb, info_fb = @test_logs (:warn, r"Youden") match_mode = :any select_threshold(
        y0,
        scores;
        criterion = "youden",
        target_fpr = 0.05,
        geometry...,
    )
    @test info_fb["criterion"] == "fpr" && info_fb["requested_criterion"] == "youden"
    @test 0.90 < t_fb <= 0.95
    # far without positives and no admissible episode disables the alarm
    t_inf, _ =
        @test_logs (:warn, r"alarms are disabled") match_mode = :any select_threshold(
            y0,
            scores;
            criterion = "far",
            target_far_per_30d = 0.0,
            geometry...,
        )
    @test t_inf == Inf
    @test_throws ArgumentError select_threshold(
        yv,
        scores;
        criterion = "bogus",
        geometry...,
    )
    @test_throws ArgumentError select_threshold(Int[], Float64[]; geometry...)
    @test_throws DimensionMismatch select_threshold(yv, scores[1:19]; geometry...)

    # Class-weighted loss: weight 1 is the plain BCE; a larger weight raises
    # the loss of every imperfectly scored positive
    rng = StableRNG(11)
    model = VariationalQuantumClassifier(4, 2; rng = rng)
    Xw = rand(rng, Float32, 6, 4) .* Float32(2π)
    yw = [1, 0, 1, 0, 0, 0]
    @test loss_function(model, Xw, yw; positive_weight = 1) == loss_function(model, Xw, yw)
    @test loss_function(model, Xw, yw; positive_weight = 3) > loss_function(model, Xw, yw)
    opt_state = Flux.setup(Adam(0.01), model.params)
    @test isfinite(train_step!(model, opt_state, Xw, yw; positive_weight = 2.0))
end

@testset "LDC noise model and readers" begin
    # Doctest of the ldc package: SciRDv1 X-channel PSD at five frequencies
    f5 = 10.0 .^ range(-5, 0; length = 5)
    x_ref = [7.13597299e-40, 2.76990908e-42, 9.52379492e-43, 1.92645601e-40, 1.15359813e-36]
    @test all(
        isapprox(ldc_tdi_psd(f; channel = :X, model = "SciRDv1"), r; rtol = 1e-8) for
        (f, r) in zip(f5, x_ref)
    )
    # A = E for equal arms; T is quieter than X in band; TDI 2 rescales by 4 sin²(2x)
    @test ldc_tdi_psd(2e-3; channel = :A) == ldc_tdi_psd(2e-3; channel = :E)
    @test ldc_tdi_psd(2e-3; channel = :T) < ldc_tdi_psd(2e-3; channel = :X)
    x =
        2π * 2e-3 * MilliHertzQML.MilliHertzBase.L_ARM /
        MilliHertzQML.MilliHertzBase.C_LIGHT
    @test isapprox(
        ldc_tdi_psd(2e-3; channel = :A, tdi2 = true),
        4 * sin(2x)^2 * ldc_tdi_psd(2e-3; channel = :A);
        rtol = 1e-12,
    )
    @test ldc_tdi_psd(0.0) == Inf
    @test ldc_tdi_psd(1e-3; observation_years = 1.0) > ldc_tdi_psd(1e-3)
    @test ldc_confusion_psd(1e-3; channel = :A) ==
          1.5 * ldc_confusion_psd(1e-3; channel = :X)
    @test ldc_confusion_psd(1e-3; observation_years = 4.0) <
          ldc_confusion_psd(1e-3; observation_years = 0.5)
    @test_throws ArgumentError ldc_tdi_psd(1e-3; model = "unknown")
    @test_throws ArgumentError ldc_tdi_psd(1e-3; channel = :B)
    @test_throws ArgumentError ldc_confusion_psd(1e-3; observation_years = 20.0)

    # A/E/T is an orthonormal combination
    rng = StableRNG(5)
    X, Y, Z = randn(rng, 100), randn(rng, 100), randn(rng, 100)
    A, E, T = tdi_to_aet(X, Y, Z)
    @test isapprox(
        sum(A .^ 2 .+ E .^ 2 .+ T .^ 2),
        sum(X .^ 2 .+ Y .^ 2 .+ Z .^ 2);
        rtol = 1e-12,
    )
    @test A == (Z .- X) ./ sqrt(2)
    @test_throws DimensionMismatch tdi_to_aet(X, Y, Z[1:99])

    # Compound and group HDF5 layouts read identically; catalogues become tables
    mktempdir() do dir
        n = 64
        t = collect(0.0:5.0:(5.0*(n-1)))
        rows = [(t = t[i], X = 1.0 * i, Y = 2.0 * i, Z = 3.0 * i) for i in 1:n]
        cat = [(Mass1 = 1e6, Mass2 = 5e5, CoalescenceTime = 100.0)]
        compound = joinpath(dir, "ldc.h5")
        MilliHertzQML.MilliHertzBase.h5open(compound, "w") do f
            f["obs/tdi"] = rows
            MilliHertzQML.MilliHertzBase.attributes(f["obs/tdi"])["dt"] = 5.0
            f["sky/mbhb/cat"] = cat
        end
        grouped = joinpath(dir, "sim.h5")
        MilliHertzQML.MilliHertzBase.h5open(grouped, "w") do f
            f["obs/tdi/t"] = t
            f["obs/tdi/X"] = [1.0 * i for i in 1:n]
            f["obs/tdi/Z"] = [3.0 * i for i in 1:n]
        end
        a = read_tdi(compound)
        b = read_tdi(grouped)
        @test a.t == b.t == t && a.X == b.X && a.Z == b.Z && a.dt == b.dt == 5.0
        @test a.Y == 2.0 .* (1:n) && b.Y == zeros(n)
        table = read_catalog(compound)
        @test nrow(table) == 1 && table.CoalescenceTime[1] == 100.0
        @test_throws ArgumentError read_tdi(compound; group = "missing")
        @test_throws ArgumentError read_tdi(joinpath(dir, "absent.h5"))
    end

    # Welch estimate: white noise of variance σ² has one-sided PSD 2σ²/fs
    fs = 0.2
    σ = 3.0
    white = σ .* randn(rng, 200_000)
    fw, sw = welch_psd(white, fs; segment_length = 1024)
    @test length(fw) == length(sw) == 512 && fw[1] == fs / 1024
    @test isapprox(median(sw), 2σ^2 / fs; rtol = 0.03)
    fw2, sw2 = welch_psd(white, fs; segment_length = 1024, average = :mean)
    @test isapprox(mean(sw2), 2σ^2 / fs; rtol = 0.03)
    # Coloured noise synthesised from the model PSD is recovered in band
    colored =
        synthesize_noise(StableRNG(6), 400_000, fs; f_min = 1e-5, psd = lisa_noise_psd)
    fc, sc = welch_psd(colored, fs; segment_length = 8192)
    inband = (fc .>= 1e-3) .& (fc .<= 5e-2)
    @test isapprox(median(sc[inband] ./ lisa_noise_psd.(fc[inband])), 1.0; rtol = 0.1)
    @test_throws ArgumentError welch_psd(white, fs; segment_length = 1)
    @test_throws ArgumentError welch_psd(white, fs; segment_length = 1024, overlap = 1.0)
    @test_throws ArgumentError welch_psd(white, fs; segment_length = 1024, average = :max)
    # Pooling the segments of several records: two copies of one record give
    # the record's own estimate, a record shorter than a segment is skipped,
    # and no record holding a segment is an error
    fp, sp = welch_psd([white, white], fs; segment_length = 1024)
    @test fp == fw && sp == sw
    @test welch_psd([white, white[1:100]], fs; segment_length = 1024)[2] == sw
    @test_throws ArgumentError welch_psd([white[1:100]], fs; segment_length = 1024)

    # Log-frequency smoothing: a power law is left as it is away from the
    # ends of the table, a one-bin line is diluted by the ≈ 600 bins of a
    # 0.01-dex kernel at 10 mHz, and zero width is the identity
    fg = collect(rfftfreq(65536, fs)[2:end])
    power_law = 1e-40 .* (fg ./ 1e-2) .^ -2
    smoothed = smooth_psd(fg, power_law, 0.01)
    interior = (fg .>= 1e-4) .& (fg .<= 5e-2)
    @test maximum(abs.(smoothed[interior] ./ power_law[interior] .- 1)) < 1e-2
    k0 = searchsortedfirst(fg, 1e-2)
    line = copy(power_law)
    line[k0] *= 100
    diluted = smooth_psd(fg, line, 0.01)
    @test diluted[k0] / power_law[k0] < 1.1
    @test smooth_psd(fg, line, 0.0) == line
    @test_throws ArgumentError smooth_psd(fg, line, -0.01)
    @test_throws DimensionMismatch smooth_psd(fg, line[1:10], 0.01)
    @test_throws ArgumentError smooth_psd(reverse(fg), line, 0.01)
    @test_throws ArgumentError smooth_psd(fg, -line, 0.01)

    # Log-log interpolation: exact at the knots, geometric in between, flat outside
    S = interpolated_psd([1e-3, 1e-2, 1e-1], [1.0, 100.0, 1.0])
    @test S(1e-3) == 1.0 && S(1e-2) == 100.0
    @test isapprox(S(sqrt(1e-3 * 1e-2)), 10.0; rtol = 1e-12)
    @test S(1e-4) == 1.0 && S(1.0) == 1.0 && S(0.0) == Inf
    @test_throws ArgumentError interpolated_psd([1e-2, 1e-3], [1.0, 2.0])
    @test_throws ArgumentError interpolated_psd([1e-3, 1e-2], [1.0, 0.0])
    @test_throws DimensionMismatch interpolated_psd([1e-3, 1e-2], [1.0])

    # Windowed SNR of a placed sinusoid burst peaks on the burst, and the
    # labelling helpers locate it
    n = 20_000
    sig = zeros(n)
    burst = 8001:9000
    sig[burst] .= 1e-20 .* sin.(2π * 5e-3 .* (0:999) ./ fs)
    starts, ρ = windowed_snr(sig, fs; window_size = 1000, step = 100, psd = lisa_noise_psd)
    @test length(starts) == length(ρ) == div(n - 1000, 100) + 1
    @test starts[argmax(ρ)] == 8001
    @test ρ[argmax(ρ)] > 5 && all(ρ[starts .> 9000] .== 0)
    peaks = snr_peaks(starts, ρ; threshold = 5.0, min_separation = 5000)
    @test peaks == [argmax(ρ)]
    # A small local maximum shortly before a large one is an inspiral
    # fluctuation, not a merger; two comparable peaks stay distinct
    series = zeros(50)
    series[10] = 6.0
    series[20] = 600.0
    series[30] = 500.0
    grid = 1:100:5000
    @test snr_peaks(grid, series; threshold = 5.0, min_separation = 500) == [10, 20, 30]
    @test snr_peaks(
        grid,
        series;
        threshold = 5.0,
        min_separation = 500,
        precursor_window = 1500,
    ) == [20, 30]
    @test_throws ArgumentError snr_peaks(
        grid,
        series;
        threshold = 5.0,
        min_separation = 0,
        precursor_ratio = 2.0,
    )
    spans = detectable_spans(starts, ρ, 1000; threshold = 5.0)
    @test length(spans) == 1 && first(spans[1]) <= 8001 && last(spans[1]) >= 9000
    @test fixed_spans([8500], fs, n; before = 100.0, after = 50.0) == [8480:8510]
    @test fixed_spans([5], fs, n; before = 100.0, after = 0.0) == [1:5]
    labels = span_labels(n, [10:20, 15:30])
    @test count(==(1), labels) == 21 && labels[9] == 0 && labels[31] == 0
    @test_throws ArgumentError span_labels(10, [5:12])
    @test_throws ArgumentError fixed_spans([0], fs, n; before = 1.0, after = 1.0)
    # Signal onsets: the first window from the lower bound that reaches the
    # threshold, the merger sample when none does before it
    @test signal_onsets(starts, ρ, [9000], [1]; threshold = 5.0) == [first(spans[1])]
    @test signal_onsets(starts, ρ, [9000], [first(spans[1]) + 1]; threshold = 5.0)[1] >
          first(spans[1])
    @test signal_onsets(starts, ρ, [2000], [1]; threshold = 5.0) == [2000]
    @test_throws ArgumentError signal_onsets(starts, ρ, [100], [200]; threshold = 5.0)
    @test_throws ArgumentError signal_onsets(starts, ρ, [100], [1]; threshold = 0.0)
    @test_throws DimensionMismatch signal_onsets(
        starts,
        ρ,
        [100, 200],
        [1];
        threshold = 5.0,
    )
    # The generator's onset of one injection: a window reaching the threshold
    # between the label start and the coalescence, its predecessor below it
    onset_settings = (
        label_span = "fixed",
        label_window_size = 1000,
        label_step = 100,
        label_snr_threshold = 5.0,
    )
    k_on = MilliHertzQML.MilliHertzBase.signal_onset(
        onset_settings,
        (sig,),
        burst,
        8900,
        7001,
        9100,
        fs,
        lisa_noise_psd,
    )
    @test 7001 <= k_on <= 8900
    @test matched_filter_snr(view(sig, k_on:(k_on+999)), fs; psd = lisa_noise_psd) >= 5
    @test matched_filter_snr(view(sig, (k_on-100):(k_on+899)), fs; psd = lisa_noise_psd) < 5
    @test MilliHertzQML.MilliHertzBase.signal_onset(
        onset_settings,
        (zeros(n),),
        burst,
        8900,
        7001,
        9100,
        fs,
        lisa_noise_psd,
    ) == 8900
    @test MilliHertzQML.MilliHertzBase.signal_onset(
        merge(onset_settings, (label_span = "detectable",)),
        (sig,),
        burst,
        8900,
        7500,
        9100,
        fs,
        lisa_noise_psd,
    ) == 7500
    @test_throws ArgumentError windowed_snr(
        sig,
        fs;
        window_size = 1,
        step = 1,
        psd = lisa_noise_psd,
    )
    @test isempty(snr_peaks(starts, ρ; threshold = 1e9, min_separation = 0))

    # Paper feature set on a raw window: four finite values, entropy in [0, 1]
    paper = extract_features(colored[1:1000], fs; feature_set = :paper)
    @test length(paper) == 4 && all(isfinite, paper) && 0 <= paper[1] <= 1
    @test feature_names(:paper) ==
          [:spectral_entropy, :log_power_mean, :log_power_std, :log_power_max]
    @test feature_names(:whitened) == [:p_low, :p_high, :spectral_entropy, :log_power_std]
    @test_throws ArgumentError feature_names(:other)
    @test_throws ArgumentError extract_features(colored[1:1000], fs; feature_set = :other)
end

@testset "Configuration and provenance" begin
    @test isfile(joinpath(project_root(), "Project.toml"))
    # The pipeline root: the scope, then the environment variable, then the
    # active environment; a configuration's root from its [paths] root or the
    # nearest Project.toml above it
    @test project_root() == PROJECT_ROOT
    mktempdir() do dir
        @test with_pipeline_root(project_root, dir) == abspath(dir)
        @test with_pipeline_root(() -> resolvepath("data"), dir) ==
              joinpath(abspath(dir), "data")
        config = joinpath(dir, "configs", "run.toml")
        mkpath(dirname(config))
        write(config, "[paths]\nroot = \"..\"\n")
        @test config_root(config) == normpath(abspath(dir))
        write(config, "[paths]\ninputs = \"in\"\n")
        @test config_root(config) == project_root()
        touch(joinpath(dir, "Project.toml"))
        @test config_root(config) == abspath(dir)
    end
    @test config_root(joinpath(PROJECT_ROOT, "configs", "experiments", "q8_b6.toml")) ==
          PROJECT_ROOT
    if startswith(Base.active_project(), PROJECT_ROOT)
        @test withenv(project_root, "STREAMINGINFERENCE_ROOT" => nothing) == PROJECT_ROOT
    end
    @test resolvepath("data") == joinpath(project_root(), "data")
    @test resolvepath("/abs/x") == "/abs/x"
    @test rootrelative(joinpath(project_root(), "data", "x.csv")) ==
          joinpath("data", "x.csv")
    @test rootrelative("/elsewhere/x.csv") == "/elsewhere/x.csv"
    # A sibling directory whose name extends the root's lies outside it
    @test rootrelative(project_root() * "x/data.csv") == project_root() * "x/data.csv"
    @test rootrelative(project_root()) == "."
    # Provenance paths: relative inside the root, bare file name outside,
    # so that no snapshot carries the account name of the running machine
    @test provenance_path(joinpath(PROJECT_ROOT, "data", "x.csv")) ==
          joinpath("data", "x.csv")
    @test provenance_path(joinpath(homedir(), "elsewhere", "product.h5")) == "product.h5"
    @test !occursin(homedir(), provenance_path(joinpath(homedir(), "p.h5")))
    # The fingerprint identifies the host without naming it
    fp = hardware_fingerprint()
    @test !haskey(fp, "hostname")
    @test length(fp["machine_id"]) == 12 &&
          all(c -> c in "0123456789abcdef", fp["machine_id"])
    @test fp["machine_id"] == MilliHertzQML.StreamingInference.machine_id()
    @test !occursin(gethostname(), fp["machine_id"])
    @test !occursin(homedir(), fp["versioninfo"])
    @test_throws ArgumentError load_config(joinpath(project_root(), "absent.toml"))
    sec = Dict{String,Any}("a" => 3, "b" => 2.5, "c" => "x", "d" => [1, 2])
    @test cfgget(sec, "a", 0; type = Float64) === 3.0
    @test cfgget(sec, "missing", 7; type = Int) == 7
    @test_throws ArgumentError cfgget(sec, "c", "y"; type = Int)
    @test_throws ArgumentError cfgget(sec, "a", 0; type = Int, min = 4)
    @test_throws ArgumentError cfgget(sec, "b", 0.0; type = Float64, max = 2.0)
    @test_throws ArgumentError cfgget(sec, "c", "x"; choices = ("y", "z"))
    @test override(nothing, 1) == 1 && override(2, 1) == 2
    @test analysis_band(Dict{String,Any}("band" => [1e-3, 5e-3]), "band", nothing) ==
          (1e-3, 5e-3)
    @test_throws ArgumentError analysis_band(
        Dict{String,Any}("band" => [5e-3, 1e-3]),
        "band",
        nothing,
    )
    # An empty configuration yields the documented, validated defaults
    empty = Dict{String,Any}()
    @test pipeline_paths(empty).inputs == joinpath(project_root(), "data", "inputs")
    @test generation_settings(empty).snr_max == 50.0
    @test preprocessing_settings(empty).psd == "model"
    @test preprocessing_settings(empty).feature_set == :whitened
    @test preprocessing_settings(empty).band_edges_hz == [1e-3, 5e-3, 1e-1]
    @test preprocessing_settings(empty).edge_margin == 0.0
    @test_throws ArgumentError preprocessing_settings(
        Dict{String,Any}("preprocessing" => Dict{String,Any}("edge_margin" => -1.0)),
    )
    @test preprocessing_settings(empty).psd_smoothing_dex == 0.0
    @test_throws ArgumentError preprocessing_settings(
        Dict{String,Any}("preprocessing" => Dict{String,Any}("psd_smoothing_dex" => -0.01)),
    )
    @test feature_geometry(joinpath(tempdir(), "absent_features.csv"), empty).first_window ==
          1
    @test preprocessing_settings(
        Dict{String,Any}(
            "preprocessing" => Dict{String,Any}(
                "feature_set" => "bands",
                "band_edges_hz" => [5e-4, 2e-3, 8e-3],
            ),
        ),
    ).band_edges_hz == [5e-4, 2e-3, 8e-3]
    @test_throws ArgumentError preprocessing_settings(
        Dict{String,Any}(
            "preprocessing" => Dict{String,Any}("band_edges_hz" => [5e-3, 1e-3]),
        ),
    )
    @test model_settings(empty).n_qubits == 4
    @test training_settings(empty).threshold_criterion == "far"
    @test training_settings(empty).threshold_block == "validation"
    @test training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("threshold_block" => "held_out")),
    ).threshold_block == "held_out"
    @test_throws ArgumentError training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("threshold_block" => "test")),
    )
    @test training_settings(empty).phase_span == 1.0
    @test training_settings(empty).min_fit_episodes == 5
    @test training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("phase_span" => 2.0)),
    ).phase_span == 2.0
    @test_throws ArgumentError training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("phase_span" => 2.5)),
    )
    @test_throws ArgumentError training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("min_fit_episodes" => -1)),
    )
    @test telemetry_settings(empty).alert_persistence == 3
    @test telemetry_settings(empty).alert_crediting == "signal"
    @test_throws ArgumentError telemetry_settings(
        Dict{String,Any}("telemetry" => Dict{String,Any}("alert_crediting" => "merger")),
    )
    @test telemetry_settings(empty).psd_mode == "sidecar"
    @test telemetry_settings(empty).psd_segment_length == 65536
    @test telemetry_settings(
        Dict{String,Any}("telemetry" => Dict{String,Any}("psd_mode" => "trailing")),
    ).psd_mode == "trailing"
    @test_throws ArgumentError telemetry_settings(
        Dict{String,Any}("telemetry" => Dict{String,Any}("psd_mode" => "full_record")),
    )
    @test_throws ArgumentError telemetry_settings(
        Dict{String,Any}("telemetry" => Dict{String,Any}("alert_persistence" => 0)),
    )
    @test inference_settings(empty).block == "all"
    @test ldc_settings(empty).label_before_sec == 4 * 86400.0
    @test_throws ArgumentError training_settings(
        Dict{String,Any}("training" => Dict{String,Any}("train_fraction" => 0.9)),
    )
    @test_throws ArgumentError preprocessing_settings(
        Dict{String,Any}("preprocessing" => Dict{String,Any}("step_size" => 2000)),
    )
    @test_throws ArgumentError generation_settings(
        Dict{String,Any}("generation" => Dict{String,Any}("observation_years" => 3.0)),
    )
    # Resources: machine-derived defaults, ordered thresholds, monotone estimates
    r = resource_settings(empty)
    @test r.warn_memory_gib <= r.max_memory_gib <= r.total_memory_gib
    r16 = resource_settings(
        Dict{String,Any}(
            "resources" =>
                Dict{String,Any}("max_memory_gib" => 16.0, "warn_memory_gib" => 8.0),
        ),
    )
    @test r16.max_memory_gib == 16.0
    @test_throws ArgumentError resource_settings(
        Dict{String,Any}(
            "resources" =>
                Dict{String,Any}("max_memory_gib" => 1.0, "warn_memory_gib" => 2.0),
        ),
    )
    # One more qubit doubles the statevector and adds a fifth of the gates
    @test training_memory_estimate_gib(5, 4, 64) ==
          2.5 * training_memory_estimate_gib(4, 4, 64)
    @test training_memory_estimate_gib(4, 4, 128) ==
          2 * training_memory_estimate_gib(4, 4, 64)
    @test record_memory_estimate_gib(2^30 ÷ 8) == 6.0
    @test check_memory(0.1, r16; stage = "test") == 0.1
    @test_logs (:warn, r"warning threshold") match_mode = :any check_memory(
        9.0,
        r16;
        stage = "test",
    )
    @test_throws ArgumentError check_memory(17.0, r16; stage = "test")
    # Provenance: run identifiers, fingerprint, git, overwrite-safe writing
    @test length(new_run_id()) == 8 && new_run_id() != new_run_id()
    hw = hardware_fingerprint()
    @test hw["julia_version"] == string(VERSION)
    @test hw["cpu_threads_logical"] == Sys.CPU_THREADS
    g = git_provenance()
    @test haskey(g, "git_commit") &&
          g["package_version"] == string(pkgversion(MilliHertzQML))
    mktempdir() do dir
        path = joinpath(dir, "snap.toml")
        write_toml(path, Dict("stage" => Dict("a" => 1)))
        snap = TOML.parsefile(path)
        @test snap["stage"]["a"] == 1
        @test haskey(snap, "hardware") && haskey(snap, "git") && haskey(snap, "written_at")
        write_toml(path, Dict("stage" => Dict("a" => 2)))
        @test TOML.parsefile(path)["stage"]["a"] == 2
        @test TOML.parsefile(joinpath(dir, "snap_#1.toml"))["stage"]["a"] == 1
        @test backup_existing!(joinpath(dir, "absent.toml")) === nothing
        csv = joinpath(dir, "t.csv")
        write_csv(csv, DataFrame(x = [1, 2]))
        write_csv(csv, DataFrame(x = [3]))
        @test nrow(CSV.read(csv, DataFrame)) == 1 && isfile(joinpath(dir, "t_#1.csv"))
        plain = joinpath(dir, "plain.toml")
        write_toml(plain, Dict("k" => "v"); tag = false)
        @test !haskey(TOML.parsefile(plain), "git")
    end
    @test occursin("Time", sprint(report_timing))

    # Resolved environment: manifests are not tracked, so the digest in every
    # tagged record and the copy in the run directory carry it instead.
    manifest = active_manifest_path()
    @test manifest !== nothing && isfile(manifest)
    @test dirname(manifest) == dirname(Base.active_project())
    digest = manifest_sha256()
    @test occursin(r"^[0-9a-f]{64}$", digest)
    env = provenance()["environment"]
    @test Set(keys(provenance())) == Set(["hardware", "git", "environment", "written_at"])
    @test env["manifest_sha256"] == digest
    @test !isabspath(env["active_project"])
    mktempdir() do dir
        target = snapshot_manifest(joinpath(dir, "run"))
        @test target == joinpath(dir, "run", "manifest_snapshot.toml")
        @test read(target) == read(manifest)
        snapshot_manifest(joinpath(dir, "run"))
        @test isfile(joinpath(dir, "run", "manifest_snapshot_#1.toml"))
        @test read(target) == read(manifest)
    end
end

@testset "Product identity" begin
    p = Dict{String,Any}("b" => 2, "a" => [1.0, 2.0])
    q = Dict{String,Any}("a" => [1.0, 2.0], "b" => 2)
    @test parameter_digest(p) == parameter_digest(q)
    @test length(parameter_digest(p)) == 64 &&
          all(in("0123456789abcdef"), parameter_digest(p))
    @test parameter_digest(merge(p, Dict{String,Any}("b" => 3))) != parameter_digest(p)
    table = product_table("features"; channels = "A", parents = Dict("source" => "x"))
    @test table["kind"] == "features" && table["channels"] == "A" && table["schema"] == 1
    @test table["parents"] == Dict{String,Any}("source" => "x")
    @test_throws ArgumentError product_table("features"; channels = "A", schema = 0)
    mktempdir() do dir
        fs = 0.2
        n = 12_000
        h5 = joinpath(dir, "record.h5")
        HDF5 = MilliHertzQML.MilliHertzBase.HDF5
        function write_record(seed)
            noise =
                synthesize_noise(StableRNG(seed), n, fs; f_min = 1e-5, psd = lisa_noise_psd)
            HDF5.h5open(h5, "w") do file
                tdi = HDF5.create_group(HDF5.create_group(file, "obs"), "tdi")
                tdi["t"] = collect((0:(n-1)) ./ fs)
                tdi["X"] = zeros(n)
                tdi["Y"] = zeros(n)
                tdi["Z"] = sqrt(2.0) .* noise
            end
        end
        write_record(1)
        digest = content_digest(h5)
        @test length(digest) == 64 && content_digest(h5) == digest
        config = Dict{String,Any}(
            "paths" => Dict{String,Any}("inputs" => joinpath(dir, "inputs")),
            "preprocessing" => Dict{String,Any}(
                "h5_file" => h5,
                "tdi_group" => "obs/tdi",
                "psd" => "none",
                "output_prefix" => "identity",
                "edge_margin" => 1.0,
            ),
        )
        product = preprocess_record(config)
        @test !product.skipped
        sidecar = TOML.parsefile(product.sidecar_path)
        @test sidecar["product"]["kind"] == "features" &&
              sidecar["product"]["channels"] == "A"
        @test sidecar["product"]["parents"] == Dict{String,Any}("source" => digest)
        # An input touched but unchanged keeps the identity of the product;
        # an input with other content makes a new one
        touch(h5)
        @test preprocess_record(config).skipped
        write_record(2)
        @test content_digest(h5) != digest
        @test !preprocess_record(config).skipped
    end
end

@testset "Figures (CairoMakie extension)" begin
    # Plain-decimal labels of a sparse logarithmic axis: below unity the
    # mantissa follows the leading zeros (0.2, not "2.1").
    ext = Base.get_extension(MilliHertzQML, :StreamingInferenceCairoMakieExt)
    SI = MilliHertzQML.StreamingInference
    @test all(
        m -> Base.get_extension(MilliHertzQML, m) !== nothing,
        (
            :StreamingInferenceCairoMakieExt,
            :MilliHertzBaseCairoMakieExt,
            :MilliHertzQMLCairoMakieExt,
        ),
    )
    @test SI.decade_label.([-2, -1, 0, 1, 2]) == ["0.01", "0.1", "1", "10", "100"]
    @test SI.decade_label(-1, 2) == "0.2"
    @test SI.decade_label(-2, 5) == "0.05"
    @test SI.decade_label(1, 2) == "20"
    @test ext.count_ticks(134) == [0, 50, 100] && ext.count_ticks(23) == [0, 5, 10, 15, 20]
    @test ext.count_ticks(1) == [0, 1] && ext.count_ticks(0) == [0]
    @test_throws ArgumentError ext.count_ticks(-1)
    @test SI.dense_log_ticks(0.35, 37.0) ==
          ([0.5, 1.0, 2.0, 5.0, 10.0, 20.0], ["0.5", "1", "2", "5", "10", "20"])
    @test SI.dense_log_ticks(1e-3, 1e3) == SI.log_ticks(1e-3, 1e3)
    # Alert labels: a right-hand label that would cover the next marker moves
    # to the left, and so does one that would leave the axis
    @test ext.alert_label_placement([0.5, 0.56], [0.5, 0.52], ["−27.8 h", "−21.1 h"]) ==
          [:above_left, :above_right]
    @test ext.alert_label_placement([0.95], [0.5], ["15.1 h"]) == [:above_left]
    @test_throws DimensionMismatch ext.alert_label_placement([0.1], [0.1, 0.2], ["a"])
    @test_throws ArgumentError SI.decade_label(0, 10)
    values, ticklabels = SI.log_ticks(0.15, 3.0)
    @test values == [0.2, 0.5, 1.0, 2.0]
    @test ticklabels == ["0.2", "0.5", "1", "2"]
    values, ticklabels = SI.log_ticks(0.01, 100.0)
    @test ticklabels == ["0.01", "0.1", "1", "10", "100"]

    rng = StableRNG(21)
    history = (
        epochs = collect(1:14),
        train_loss = 1.0 .- 0.03 .* (1:14),
        val_loss = 1.05 .- 0.02 .* (1:14),
        val_acc = 0.6 .+ 0.02 .* (1:14),
    )
    n = 2000
    days = collect(range(0, 10; length = n))
    labels = zeros(Int, n)
    labels[400:500] .= 1
    labels[1200:1350] .= 1
    probs = clamp.(0.3 .+ 0.08 .* randn(rng, n) .+ 0.3 .* labels, 0, 1)
    snrs = 8 .+ 40 .* rand(rng, n) .* labels
    decisions = Int.(probs .>= 0.55)
    fpr, tpr, _ = roc_curve(labels, probs)
    @test figure_training_history(history) isa CairoMakie.Figure
    @test figure_mission_trace(days, probs, 0.55; labels = labels) isa CairoMakie.Figure
    @test figure_mission_trace(days, probs, 0.55) isa CairoMakie.Figure
    @test figure_roc(fpr, tpr, roc_auc(fpr, tpr)) isa CairoMakie.Figure
    sweep = threshold_sweep(labels, probs; step_size = 432, sample_rate = 1.0)
    @test figure_threshold_sweep(sweep, 0.55; target_far_per_30d = 3.0) isa
          CairoMakie.Figure
    @test figure_threshold_sweep(
        sweep,
        0.55;
        target_far_per_30d = 3.0,
        operating_point = (n_detected = 2, n_events = 2, false_alarms_per_30d = 1.5),
    ) isa CairoMakie.Figure
    @test_throws ArgumentError figure_threshold_sweep(
        sweep,
        0.55;
        operating_point = (n_detected = 2,),
    )
    @test figure_threshold_sweep(sweep, Inf) isa CairoMakie.Figure
    # A block without negatives has no false alarm at any threshold
    clean = threshold_sweep(ones(Int, n), probs; step_size = 432, sample_rate = 1.0)
    @test all(==(0), clean.false_alarms_per_30d) && all(isnan, clean.fpr)
    @test figure_threshold_sweep(clean, 0.55; target_far_per_30d = 3.0) isa
          CairoMakie.Figure
    @test_throws ArgumentError figure_threshold_sweep(DataFrame(), 0.5)
    @test_throws ArgumentError figure_threshold_sweep(DataFrame(threshold = [Inf]), 0.5)
    @test figure_sensitivity(snrs, labels, decisions) isa CairoMakie.Figure
    @test figure_sensitivity(snrs, zeros(Int, n), decisions) === nothing
    @test figure_score_distribution(probs, 0.55; labels = labels) isa CairoMakie.Figure
    @test figure_score_distribution(probs, 0.55) isa CairoMakie.Figure
    strain = synthesize_noise(rng, 4000, 0.2; f_min = 1e-5, psd = lisa_noise_psd)
    t_days = ((0:3999) ./ 0.2) ./ 86400
    lab = zeros(Int, 4000)
    lab[1500:1800] .= 1
    @test figure_telemetry_trace(t_days, strain, lab) isa CairoMakie.Figure
    @test figure_telemetry_trace(
        t_days,
        strain,
        lab;
        whitened = whiten_record(strain, 0.2; psd = lisa_noise_psd),
    ) isa CairoMakie.Figure
    # Alert figure: of two alerts close in mission time and in latency, the
    # lower one is labelled beneath its marker; labels give the data latency
    # t_alarm − t_merger with a typographic minus.
    epoch = DateTime(2035, 1, 1)
    content_end = [epoch + Hour(3i) for i in 1:200]
    alert_windows = DataFrame(
        content_end = content_end,
        complete_at = content_end .+ Hour(30),
        score = rand(StableRNG(7), 200),
        decision = rand(StableRNG(8), 0:1, 200),
    )
    alert_rows = DataFrame(
        detected = [true, true, true, false],
        t_alarm = Union{Missing,DateTime}[
            epoch+Hour(75),
            epoch+Hour(78),
            epoch+Hour(410),
            missing,
        ],
        t_merger = [epoch + Hour(h) for h in (100, 106, 400, 500)],
    )
    alert_figure = figure_telemetry_alerts(
        alert_windows,
        0.5;
        epoch = epoch,
        label_spans = [(epoch + Hour(90), epoch + Hour(100))],
        latencies = alert_rows,
    )
    @test alert_figure isa CairoMakie.Figure
    ax_latency = only(
        filter(
            a -> a isa CairoMakie.Axis && a.ylabel[] == "Latency [h]",
            alert_figure.content,
        ),
    )
    alert_labels = Dict(
        first(vcat(p.text[])) => p.align[] for
        p in ax_latency.scene.plots if p isa CairoMakie.Text
    )
    @test alert_labels == Dict(
        "−25.0 h" => (:left, :bottom),
        "−28.0 h" => (:left, :top),
        "10.0 h" => (:left, :bottom),
    )
    y_limits = ax_latency.limits[][2]
    @test y_limits[1] < -28 && y_limits[2] > 30
    @test_throws ArgumentError figure_telemetry_alerts(
        alert_windows[1:0, :],
        0.5;
        epoch = epoch,
    )
    @test_throws DimensionMismatch figure_mission_trace(days[1:10], probs, 0.5)
    @test_throws ArgumentError figure_training_history((
        epochs = Int[],
        train_loss = Float32[],
        val_loss = Float32[],
        val_acc = Float32[],
    ))
    # The base layout: a 900 × 600 pt single panel, 350 pt per further main
    # panel, 180 pt per auxiliary strip
    @test_throws ArgumentError figure_theme(; size = (0, 600))
    @test_throws ArgumentError figure_theme(; fontsize = 0)
    theme = figure_theme()
    @test theme.size[] == (900, 600) && theme.fontsize[] == 26
    @test figure_size() == (900, 600)
    @test figure_size(2) == (900, 950) && figure_size(1, 1) == (900, 780)
    @test figure_size(2, 2) == (900, 1310)
    @test_throws ArgumentError figure_size(0)
    @test_throws ArgumentError figure_size(1, -1)
    @test keys(FIGURE_STROKES) == keys(FIGURE_COLORS)
    mktempdir() do dir
        stem = joinpath(dir, "roc_curve")
        written = save_figure(figure_roc(fpr, tpr, 0.9), stem; run_id = "unit")
        @test written == ["$stem.pdf", "$stem.png"]
        @test all(isfile, written) && filesize("$stem.pdf") > 1000
        side = TOML.parsefile("$stem.toml")
        @test side["figure"]["run_id"] == "unit" && haskey(side, "git")
        # The sidecar records the canvas actually exported
        # (the ROC axis is square, on a canvas of the base height)
        @test side["figure"]["size_pt"][2] == 600 && !haskey(side["figure"], "width_mm")
        save_figure(figure_mission_trace(days, probs, 0.55), joinpath(dir, "single"))
        @test TOML.parsefile(joinpath(dir, "single.toml"))["figure"]["size_pt"] ==
              [900, 600]
        save_figure(
            figure_seed_spread([1, 2], [0.5, 0.6], [1.0, 2.0], [2, 2], 2),
            joinpath(dir, "wide"),
        )
        @test TOML.parsefile(joinpath(dir, "wide.toml"))["figure"]["size_pt"] == [900, 950]
        gap = figure_gap_study(
            ["Reference", "Loss 0.3 %", "Outage 24 h"],
            ["reference", "loss", "outage"],
            [1.0, 0.44, 0.99],
            [2, 2, 2],
            2,
            [18.9, 104.2, NaN];
            reference_far = 18.9,
        )
        save_figure(gap, joinpath(dir, "gap"))
        @test TOML.parsefile(joinpath(dir, "gap.toml"))["figure"]["size_pt"] == [1200, 600]
        @test_throws DimensionMismatch figure_gap_study(
            ["a"],
            ["f"],
            [1.0, 0.5],
            [2],
            2,
            [1.0],
        )
        @test_throws ArgumentError figure_gap_study(["a"], ["f"], [1.0], [3], 2, [1.0])
        # Grid over seeds: a missing run (NaN) is left out, a run that
        # missed a blind event is drawn open, and the rates of a positive
        # grid go on a logarithmic axis.
        grid = figure_grid_seeds(
            ["q8", "q6", "q4"],
            [4 10 NaN; 8 7 9; 9 10 12],
            [1.57 2.06 NaN; 5.5 9.8 4.0; 2.6 1.98 3.3],
            [5 5 0; 5 5 5; 5 4 5],
            5;
            seeds = [9999, 1009, 2027],
            selected = "q8",
            target_far = 3.0,
        )
        save_figure(grid, joinpath(dir, "grid"))
        @test TOML.parsefile(joinpath(dir, "grid.toml"))["figure"]["size_pt"] == [1200, 600]
        @test_throws DimensionMismatch figure_grid_seeds(
            ["a"],
            ones(1, 2),
            ones(1, 3),
            ones(Int, 1, 2),
            5,
        )
        @test_throws ArgumentError figure_grid_seeds(
            ["a"],
            ones(1, 1),
            ones(1, 1),
            fill(6, 1, 1),
            5,
        )
        @test_throws ArgumentError figure_grid_seeds(
            ["a"],
            ones(1, 1),
            ones(1, 1),
            ones(Int, 1, 1),
            5;
            selected = "b",
        )
        save_figure(figure_roc(fpr, tpr, 0.9), stem; run_id = "unit")
        @test isfile(joinpath(dir, "roc_curve_#1.pdf"))
        @test_throws ArgumentError save_figure(
            figure_roc(fpr, tpr, 0.9),
            stem;
            formats = ("bmp",),
        )
    end
end

@testset "Animations (CairoMakie extension)" begin
    history = (
        epochs = collect(1:5),
        train_loss = [1.0, 0.8, 0.7, 0.65, 0.62],
        val_loss = [1.1, 0.9, 0.85, 0.88, 0.87],
        val_acc = [0.5, 0.6, 0.68, 0.66, 0.7],
    )
    # Arrival out of mission order: two windows reach the ground swapped
    n = 12
    content_end = [DateTime(2035, 1, 1) + Dates.Minute(10 * k) for k in 1:n]
    arrival = content_end .+ Dates.Hour(3)
    arrival[3], arrival[4] = arrival[4], arrival[3]
    windows = DataFrame(
        window = 1:n,
        content_end = content_end,
        complete_at = arrival,
        coverage = fill(1.0, n),
        score = range(0.1, 0.9; length = n),
        decision = [k > 9 ? 1 : 0 for k in 1:n],
    )
    theme = animation_theme()
    @test all(isinteger, theme.size[])
    mktempdir() do dir
        path = animate_training_history(
            history,
            joinpath(dir, "training_history.gif");
            framerate = 4,
            hold_frames = 0,
        )
        @test isfile(path) && filesize(path) > 1000
        stem = joinpath(dir, "mission_replay")
        written = save_animation(stem; run_id = "unit") do target
            animate_mission_replay(
                windows,
                0.8,
                target;
                n_frames = 3,
                framerate = 4,
                hold_frames = 0,
                label_spans = [(content_end[2], content_end[5])],
            )
        end
        @test written == "$stem.gif"
        @test isfile(written) && filesize(written) > 1000
        side = TOML.parsefile("$stem.toml")
        @test side["animation"]["run_id"] == "unit" && haskey(side, "git")
        # The sidecar records the frame size of the written GIF: the canvas
        # at the raster scale
        @test side["animation"]["frame_px"] == 2 .* collect(figure_size(2, 2))
        @test side["animation"]["px_per_unit"] == 2
        # An animation is written as GIF, and a fractional raster scale renders
        # frames the encoder does not reproduce
        @test_throws ArgumentError animate_training_history(
            history,
            joinpath(dir, "training_history.mp4"),
        )
        @test_throws ArgumentError animate_mission_replay(
            windows,
            0.8,
            joinpath(dir, "replay.gif");
            px_per_unit = 2.5,
        )
        # Qualified: Yao and DataFrames both export `select`.
        @test_throws ArgumentError animate_mission_replay(
            DataFrames.select(windows, DataFrames.Not(:score)),
            0.8,
            joinpath(dir, "replay.gif"),
        )
        @test_throws ArgumentError animate_training_history(
            (epochs = [1], train_loss = [1.0], val_loss = [1.0], val_acc = [0.5]),
            joinpath(dir, "replay.gif"),
        )
    end
end

include("export_payload_tests.jl")
include("telemetry_tests.jl")
include("telemetry_integration_tests.jl")
include("labeling_tests.jl")

# Validation anchors on the LDC Sangria training product. They run only when
# MILLIHERTZQML_LDC_DIR names a directory holding LDC2_sangria_training_v2.h5
# (a 3 GB download from Zenodo record 7132178).
const SANGRIA_FILE =
    joinpath(get(ENV, "MILLIHERTZQML_LDC_DIR", ""), "LDC2_sangria_training_v2.h5")
if isfile(SANGRIA_FILE)
    @testset "Sangria anchors" begin
        catalog = read_catalog(SANGRIA_FILE)
        @test nrow(catalog) == 15
        @test isapprox(catalog.CoalescenceTime[5], 11526944.9; atol = 1.0)

        # A reference evaluation quotes an optimal A-channel
        # SNR of 1885.7 for catalogue row 4 (0-based) against the SciRDv1 noise
        # model. Its neighbour (row 3, merging 3.1 d earlier) is excluded by a
        # segment starting 3 d before the merger; the segment ends are tapered.
        truth = read_tdi(SANGRIA_FILE; group = "sky/mbhb/tdi")
        fs = 1 / truth.dt
        A, _, _ = tdi_to_aet(truth.X, truth.Y, truth.Z)
        tc = catalog.CoalescenceTime[5]
        ic = round(Int, (tc - truth.t[1]) * fs) + 1
        seg = A[(ic-round(Int, 3*86400*fs)):(ic+round(Int, 7200*fs))]
        m = 100
        for k in 1:m
            w = 0.5 * (1 - cos(π * (k - 1) / m))
            seg[k] *= w
            seg[end-k+1] *= w
        end
        ρ = matched_filter_snr(
            seg,
            fs;
            psd = f -> ldc_tdi_psd(f; channel = :A, model = "SciRDv1"),
        )
        @test isapprox(ρ, 1885.7; rtol = 0.03)

        # Noise-only null test: the observed record minus every truth stream
        # is instrument noise, whose A-channel PSD follows the "sangria" model
        # to within the simulator's filters; whitened by that model and
        # high-passed, its windows have unit band powers.
        obs = read_tdi(SANGRIA_FILE; group = "obs/tdi")
        residual, _, _ = tdi_to_aet(obs.X, obs.Y, obs.Z)
        residual .-= A
        for source in ("dgb", "igb", "vgb")
            sky = read_tdi(SANGRIA_FILE; group = "sky/$source/tdi")
            residual .-= tdi_to_aet(sky.X, sky.Y, sky.Z)[1]
        end
        psd_sangria = f -> ldc_tdi_psd(f; channel = :A, model = "sangria")
        fw, sw = welch_psd(residual, fs; segment_length = 65536)
        band = (fw .>= 3e-4) .& (fw .<= 5e-2)
        ratio = sw[band] ./ psd_sangria.(fw[band])
        @test 0.8 < median(ratio) < 1.25
        white = whiten_record(
            highpass_record(residual, fs; cutoff = 5e-4, order = 8),
            fs;
            psd = psd_sangria,
        )
        # The equal-arm analytic PSD vanishes at the TDI null f = c/(2L) ≈ 60 mHz,
        # where the data keep a finite floor, so the analytic whitening is only
        # checked below the null; the Welch estimate whitens the full band.
        starts = round.(Int, range(1, length(white) - 1000; length = 400))
        feats = [
            extract_features(view(white, s:(s+999)), fs; high_band = (5e-3, 4e-2)) for
            s in starts
        ]
        @test isapprox(mean(first.(feats)), 1.0; atol = 0.25)
        @test isapprox(mean(getindex.(feats, 2)), 1.0; atol = 0.25)
        white_welch = whiten_record(
            highpass_record(residual, fs; cutoff = 5e-4, order = 8),
            fs;
            psd = interpolated_psd(fw, sw),
        )
        feats_welch = [extract_features(view(white_welch, s:(s+999)), fs) for s in starts]
        @test isapprox(mean(first.(feats_welch)), 1.0; atol = 0.15)
        @test isapprox(mean(getindex.(feats_welch, 2)), 1.0; atol = 0.15)
    end
else
    @info "Sangria anchors skipped: set MILLIHERTZQML_LDC_DIR to the directory holding LDC2_sangria_training_v2.h5"
end

include("response_tests.jl")

@testset "Pipeline smoke test" begin
    # The four scripts run as child processes on a three-day configuration
    # whose [paths] section points into a temporary directory; nothing is
    # written into the project tree. Each script activates the scripts
    # environment itself.
    # The command of this process, so that the stages run by the scripts
    # inherit its code-coverage setting
    julia = Base.julia_cmd()
    scripts = joinpath(PROJECT_ROOT, "scripts")
    mktempdir() do dir
        cfg = TOML.parsefile(joinpath(PROJECT_ROOT, "configs", "default.toml"))
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
        cfg["preprocessing"]["edge_margin"] = 1.0
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
            cmd = `$julia --startup-file=no $(joinpath(scripts, script)) $cfg_path $(collect(args))`
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
        margin = round(
            Int,
            cfg["preprocessing"]["edge_margin"] * cfg["preprocessing"]["window_size"] /
            cfg["preprocessing"]["step_size"],
        )
        n_kept = n_windows - 2 * margin

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
        # Detectable spans start at the signal onset itself
        @test all(
            r ->
                r.label_start_index > r.label_end_index ||
                r.signal_start_index == r.label_start_index,
            eachrow(events),
        )

        stage("preprocess_ldc.jl", "--h5-file", h5, "--label-file", raw_labels) || return
        @test nrow(CSV.read(feats, DataFrame)) == n_kept
        @test nrow(CSV.read(labs, DataFrame)) == n_kept
        sidecar = TOML.parsefile(replace(feats, ".csv" => ".toml"))["features"]
        @test sidecar["window_size"] == cfg["preprocessing"]["window_size"]
        @test sidecar["step_size"] == cfg["preprocessing"]["step_size"]
        @test sidecar["sample_rate"] == fs
        @test sidecar["n_windows"] == n_kept
        @test sidecar["first_window"] == margin + 1 &&
              sidecar["edge_margin_windows"] == margin
        @test sidecar["n_windows_record"] == n_windows && margin == 10
        # The analytic whitening PSD is rebuilt from the sidecar alone
        @test sidecar["psd"] == "model" && haskey(sidecar, "observation_years")
        @test whitening_psd_from_sidecar(replace(feats, ".csv" => ".toml"))(2e-3) ==
              lisa_noise_psd(2e-3; observation_years = sidecar["observation_years"])

        stage("train.jl", "--run-id", "smoke") || return
        run_dir = joinpath(dir, "models", "run_smoke")
        model_path = joinpath(run_dir, "gw_model.jld2")
        @test isfile(model_path)
        @test isfile(joinpath(run_dir, "config.toml"))
        @test isfile(joinpath(run_dir, "manifest_snapshot.toml"))
        # Chronological blocks with a one-window buffer
        blocks = TOML.parsefile(joinpath(run_dir, "split.toml"))["split"]
        buffer = cld(cfg["preprocessing"]["window_size"], cfg["preprocessing"]["step_size"])
        @test blocks["n_windows"] == n_kept
        @test blocks["buffer_windows"] == buffer
        @test blocks["train"] == [1, floor(Int, 0.7 * n_kept)]
        @test blocks["validation"][1] == blocks["train"][2] + buffer + 1
        @test blocks["test"][1] == blocks["validation"][2] + buffer + 1
        @test blocks["test"][2] == n_kept
        n_test = blocks["test"][2] - blocks["test"][1] + 1
        # Threshold fitted on the calibration block; metrics of both blocks
        thr = TOML.parsefile(joinpath(run_dir, "threshold.toml"))["threshold"]
        @test haskey(thr, "value") && haskey(thr, "criterion") && haskey(thr, "auc")
        @test thr["criterion"] in ("far", "fpr")
        @test thr["block"] == "validation"
        @test thr["fit_windows"] == blocks["validation"][2] - blocks["validation"][1] + 1
        @test haskey(thr, "fit_false_alarm_episodes")
        metrics = TOML.parsefile(joinpath(run_dir, "metrics.toml"))
        @test haskey(metrics, "validation") && haskey(metrics, "test")
        @test isfinite(metrics["test"]["false_alarms_per_30d"])
        @test metrics["test"]["n_windows"] == n_test
        @test metrics["validation"]["threshold"] == thr["value"]
        # The operating characteristic of the calibration block and its figure
        sweep = CSV.read(joinpath(run_dir, "threshold_sweep.csv"), DataFrame)
        @test issorted(sweep.threshold) && "false_alarms_per_30d" in names(sweep)
        @test allequal(sweep.n_events)
        @test first(sweep.n_events) >= metrics["validation"]["n_events"]
        @test isfile(joinpath(dir, "plots", "run_smoke", "threshold_sweep.pdf"))

        stage("infer.jl", "--run-id", "smoke") || return
        probs = CSV.read(
            joinpath(dir, "results", "run_smoke", "inference_probabilities.csv"),
            DataFrame,
        )
        @test nrow(probs) == n_kept
        @test all(0 .<= probs.Probability .<= 1)
        @test probs.Window == 1:n_kept
        infer_metrics =
            TOML.parsefile(joinpath(dir, "results", "run_smoke", "metrics.toml"))["metrics"]
        @test infer_metrics["n_windows"] == n_kept
        @test infer_metrics["threshold"] == thr["value"]
        @test haskey(infer_metrics, "fpr")
        @test isfile(joinpath(dir, "results", "run_smoke", "threshold_sweep.csv"))
        @test isfile(joinpath(dir, "plots", "run_smoke", "threshold_sweep.pdf"))

        # The test block alone, through the run's split.toml
        stage(
            "infer.jl",
            "--model",
            model_path,
            "--run-id",
            "smoke_test",
            "--block",
            "test",
        ) || return
        block_probs = CSV.read(
            joinpath(dir, "results", "run_smoke_test", "inference_probabilities.csv"),
            DataFrame,
        )
        @test nrow(block_probs) == n_test
        @test first(block_probs.Window) == blocks["test"][1]
        block_metrics =
            TOML.parsefile(joinpath(dir, "results", "run_smoke_test", "metrics.toml"))["metrics"]
        @test block_metrics["block"] == "test"
        @test block_metrics["n_windows"] == n_test

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
        @test nrow(blind_probs) == n_kept
        @test !isfile(joinpath(dir, "results", "run_smoke_blind", "threshold_sweep.csv"))

        # The project tree received nothing
        @test !isdir(joinpath(PROJECT_ROOT, "models", "run_smoke"))
        @test !isfile(joinpath(PROJECT_ROOT, "data", "inputs", "smoke_features.csv"))
    end
end
