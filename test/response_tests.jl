# test/response_tests.jl — constellation response of the simulator through
# the CurvatureDistinguishability extension: the normalisation of the
# antenna patterns taken from that package, the polarisation and inclination
# conventions of the projection, and an end-to-end two-channel record.
@testset "Constellation response (CurvatureDistinguishability extension)" begin
    ext = Base.get_extension(MilliHertzQML, :MilliHertzQMLCurvatureDistinguishabilityExt)
    @test ext !== nothing
    response = lisa_response()
    @test channel_count(response) == 2
    @test channel_count(SkyAveragedResponse()) == 1
    lisa_settings = generation_settings(
        Dict{String,Any}("generation" => Dict{String,Any}("response" => "lisa")),
    )
    @test detector_response(lisa_settings) isa ext.LisaResponse
    @test detector_response(generation_settings(Dict{String,Any}())) isa SkyAveragedResponse
    @test_throws ArgumentError generation_settings(
        Dict{String,Any}("generation" => Dict{String,Any}("response" => "tdi")),
    )
    @test_throws ArgumentError generation_settings(
        Dict{String,Any}(
            "generation" => Dict{String,Any}(
                "mbhb_distance_min_gpc" => 10.0,
                "mbhb_distance_max_gpc" => 1.0,
            ),
        ),
    )
    @test_throws ArgumentError generation_settings(
        Dict{String,Any}("generation" => Dict{String,Any}("label_channel" => "T")),
    )
    year = 3.15576e7

    @testset "Sky and polarization average of the antenna patterns" begin
        # Robson et al. (2019): the long-wavelength response of one 60°
        # Michelson averages to 3/10; the extension's A and E carry that
        # normalisation, which channel_noise_psd assumes.
        rng = StableRNG(11)
        n = 8000
        sum_A = 0.0
        sum_E = 0.0
        for _ in 1:n
            source = draw_extrinsic(rng)
            frame = source_frame(
                response,
                source.longitude,
                source.latitude,
                source.polarization,
            )
            F_plus_A, F_cross_A, F_plus_E, F_cross_E, _, _ =
                ext.channel_patterns(rand(rng) * year, frame, response.orbit_phase)
            sum_A += F_plus_A^2 + F_cross_A^2
            sum_E += F_plus_E^2 + F_cross_E^2
        end
        @test isapprox(sum_A / n, 0.3; rtol = 0.03)
        @test isapprox(sum_E / n, 0.3; rtol = 0.03)
        @test channel_noise_psd(response, lisa_noise_psd)(1e-3) ≈
              sky_averaged_response(1e-3) * lisa_noise_psd(1e-3)
        @test sky_averaged_response(1e-4) ≈ 0.3 rtol = 1e-4
        @test sky_averaged_response(MilliHertzQML.MilliHertzBase.F_STAR) ≈ 0.3 / 1.6
    end

    @testset "Doppler phase" begin
        polar = source_frame(response, 0.3, π / 2, 0.1)
        _, _, _, _, cos_alpha, sin_alpha = ext.channel_patterns(1e6, polar, 0.0)
        @test abs(ext.doppler_phase(1e-3, cos_alpha, sin_alpha, polar)) < 1e-9
        ecliptic = source_frame(response, 0.0, 0.0, 0.0)
        # a source on the ecliptic along x: 2π f R at α = 0, R the orbital radius in light-seconds
        @test ext.doppler_phase(1e-3, 1.0, 0.0, ecliptic) ≈ 2π * 1e-3 * 499.00478383615643 rtol =
            1e-9
        @test ext.doppler_phase(1e-3, 0.0, 1.0, ecliptic) ≈ 0.0 atol = 1e-12
    end

    @testset "Face-on source at the average pattern has the sensitivity SNR" begin
        fs = 0.2
        spectrum = phenoma_spectrum(fs, 0.25 * 86400; total_mass = 1e6, mass_ratio = 1.5)
        scale = phenoma_physical_amplitude(spectrum.p; distance_sec = 5 * GIGAPARSEC_SEC)
        H = scale .* spectrum.H
        ρ_reference =
            matched_filter_snr(phenoma_series(H, spectrum.n, fs), fs; psd = lisa_noise_psd)
        @test ρ_reference > 10
        # anchor of the physical scale: an equal-mass 10^6 M⊙ source-frame
        # binary at z = 1 (2 × 10^6 M⊙ detector frame, 6.8 Gpc) reaches a
        # face-on SNR of a few thousand against the sensitivity, the scale of
        # the LISA horizon plots
        anchor = phenoma_spectrum(fs, 2 * 86400; total_mass = 2e6, mass_ratio = 1.0)
        anchor_scale =
            phenoma_physical_amplitude(anchor.p; distance_sec = 6.8 * GIGAPARSEC_SEC)
        ρ_anchor = matched_filter_snr(
            phenoma_series(anchor_scale .* anchor.H, anchor.n, fs),
            fs;
            psd = lisa_noise_psd,
        )
        @test 3e3 < ρ_anchor < 1.2e4
        # the transform pair is normalised: the series of a spectrum returns it
        h_anchor = phenoma_series(anchor_scale .* anchor.H, anchor.n, fs)
        H_back = rfft(vcat(h_anchor, zeros(anchor.n))) ./ fs
        k = findfirst(>=(2e-3), anchor.freqs)
        @test isapprox(abs(H_back[k]), anchor_scale * abs(anchor.H[k]); rtol = 0.05)
        @test_throws ArgumentError phenoma_series(H, spectrum.n, 0.0)
        @test_throws DimensionMismatch phenoma_series(H[1:(end-1)], spectrum.n, fs)
        psd_channel = channel_noise_psd(response, lisa_noise_psd)
        delays = [phenoma_arrival_delay(f, spectrum.p) for f in spectrum.freqs]
        @test all(delays .>= 0)
        @test delays[end] == 0
        @test issorted(
            delays[2:findfirst(>=(spectrum.p.f_merg), spectrum.freqs)];
            rev = true,
        )
        rng = StableRNG(3)
        n_draw = 600
        face_on = 0.0
        edge_on = 0.0
        for _ in 1:n_draw
            source = draw_extrinsic(rng)
            frame = source_frame(
                response,
                source.longitude,
                source.latitude,
                source.polarization,
            )
            t_c = rand(rng) * year
            H_A, _ = project_spectrum(response, frame, spectrum.freqs, H, delays, t_c, 0.0)
            face_on +=
                matched_filter_snr(
                    phenoma_series(H_A, spectrum.n, fs),
                    fs;
                    psd = psd_channel,
                )^2
            H_A, _ =
                project_spectrum(response, frame, spectrum.freqs, H, delays, t_c, π / 2)
            edge_on +=
                matched_filter_snr(
                    phenoma_series(H_A, spectrum.n, fs),
                    fs;
                    psd = psd_channel,
                )^2
        end
        # ⟨F⁺² + F×²⟩ = 3/10 cancels the 3/10 of the channel PSD
        @test isapprox(sqrt(face_on / n_draw), ρ_reference; rtol = 0.06)
        # edge-on: a₊ = 1/2, a× = 0, ⟨F⁺²⟩ = 3/20 → one eighth of the face-on power
        @test isapprox(edge_on / face_on, 1 / 8; rtol = 0.12)
        # the projection is linear and the E channel differs from A
        frame = source_frame(response, 1.0, 0.4, 0.2)
        H_A, H_E = project_spectrum(response, frame, spectrum.freqs, H, delays, 1e7, 0.7)
        H_A2, _ =
            project_spectrum(response, frame, spectrum.freqs, 2 .* H, delays, 1e7, 0.7)
        @test H_A2 ≈ 2 .* H_A
        @test !(H_A ≈ H_E)
        @test_throws DimensionMismatch project_spectrum(
            response,
            frame,
            spectrum.freqs[1:(end-1)],
            H,
            delays,
            1e7,
            0.7,
        )
    end

    @testset "Time-domain projection" begin
        n = 20000
        t = collect(range(0.0, year; length = n))
        f = 3e-3
        frame = source_frame(response, 1.0, 0.05, 0.2)
        h_A, h_E = project_series(response, frame, t, ones(n), 2π * f .* t, fill(f, n), 0.7)
        @test all(isfinite, h_A) && all(isfinite, h_E)
        @test !(h_A ≈ h_E)
        # the antenna pattern of an ecliptic source turns with the constellation:
        # the monthly RMS of the channel varies over the year
        months = [sqrt(mean(abs2, h_A[k:(k+n÷12-1)])) for k in 1:(n÷12):(n-n÷12+1)]
        @test maximum(months) / minimum(months) > 1.1
        # amplitude and inclination enter linearly: edge-on plus only, halved
        h_edge, _ =
            project_series(response, frame, t, fill(2.0, n), 2π * f .* t, fill(f, n), π / 2)
        h_face, _ =
            project_series(response, frame, t, ones(n), 2π * f .* t, fill(f, n), 0.0)
        @test !(h_edge ≈ h_face)
        @test_throws DimensionMismatch project_series(
            response,
            frame,
            t,
            ones(n - 1),
            2π * f .* t,
            fill(f, n),
            0.7,
        )
        # detectable span over two channels equals the single-channel span when
        # the second channel is silent
        placed = zeros(20000)
        placed[8001:12000] .= 1e-20 .* sin.(2π * 5e-3 .* (1:4000) ./ 0.2)
        span_one = detectable_span(placed, 8001:12000, 0.2, 1000, 5.0; step = 10)
        span_two =
            detectable_span((placed, zeros(20000)), 8001:12000, 0.2, 1000, 5.0; step = 10)
        @test span_one == span_two
        @test_throws DimensionMismatch detectable_span(
            (placed, zeros(19999)),
            8001:12000,
            0.2,
            1000,
            5.0,
        )
    end

    @testset "Two-channel record" begin
        mktempdir() do dir
            config = load_config(joinpath(PROJECT_ROOT, "configs", "default.toml"))
            config["generation"] = merge(
                config["generation"],
                Dict{String,Any}(
                    "days" => 4.0,
                    "n_mbhb" => 2,
                    "n_gbs" => 3,
                    "n_emris" => 1,
                    "seed" => 7,
                    "response" => "lisa",
                    "mbhb_duration_days" => 0.5,
                    "output" => joinpath(dir, "lisa.h5"),
                ),
            )
            result = generate_telemetry(config; run_id = "response_test")
            @test keys(result.channels) == (:A, :E)
            @test result.strain === result.channels.A
            tdi = read_tdi(result.h5_file)
            A, E, T = tdi_to_aet(tdi.X, tdi.Y, tdi.Z)
            @test A ≈ result.channels.A rtol = 1e-12
            @test E ≈ result.channels.E rtol = 1e-12
            @test maximum(abs, T) < 1e-14 * maximum(abs, A)
            catalog = result.catalog
            @test nrow(catalog) == 2
            for column in (
                "distance_gpc",
                "ecliptic_longitude",
                "ecliptic_latitude",
                "inclination",
                "polarization",
                "snr_a",
                "snr_e",
            )
                @test column in names(catalog)
            end
            @test all(1.0 .<= catalog.distance_gpc .<= 50.0)
            @test all(catalog.snr .== catalog.snr_a)        # label_channel = "A"
            @test all(catalog.snr_a .> 0) && all(catalog.snr_e .> 0)
            @test all(catalog.label_start_index .>= 1)
            @test all(
                r ->
                    r.label_start_index > r.label_end_index ||
                    r.signal_start_index == r.label_start_index,
                eachrow(catalog),
            )
            snapshot = TOML.parsefile(result.snapshot_file)["generation"]
            @test snapshot["response"] == "lisa"
            @test snapshot["channels"] == ["A", "E"]
            @test snapshot["label_channel"] == "A"
            # the labels lie inside the qualifying windows, which overhang the
            # covered span by less than one window at either end
            positives = findall(==(1), result.labels)
            @test !isempty(positives)
            window = config["generation"]["label_window_size"]
            @test all(
                any(
                    r.start_index - window < k < r.end_index + window for
                    r in eachrow(catalog)
                ) for k in positives
            )
            @test all(
                r.label_start_index > r.start_index - window for r in eachrow(catalog)
            )
            @test all(r.label_end_index < r.end_index + window for r in eachrow(catalog))
            # a seed reproduces the record
            again = generate_telemetry(
                config;
                run_id = "response_test_again",
                output = joinpath(dir, "lisa_again.h5"),
            )
            @test again.channels.A == result.channels.A
            @test again.channels.E == result.channels.E
            # the network label channel carries the quadrature SNR
            config["generation"]["label_channel"] = "network"
            config["generation"]["output"] = joinpath(dir, "network.h5")
            network = generate_telemetry(config; run_id = "response_network")
            @test all(
                network.catalog.snr .≈ hypot.(network.catalog.snr_a, network.catalog.snr_e),
            )

            # a noise-only record has the Michelson-channel PSD
            config["generation"] = merge(
                config["generation"],
                Dict{String,Any}(
                    "n_mbhb" => 0,
                    "n_gbs" => 0,
                    "n_emris" => 0,
                    "days" => 8.0,
                    "output" => joinpath(dir, "noise.h5"),
                ),
            )
            noise = generate_telemetry(config; run_id = "response_noise")
            @test nrow(noise.catalog) == 0
            freqs, table =
                welch_psd(noise.channels.A, 0.2; segment_length = 4096, average = :mean)
            psd_channel = channel_noise_psd(response, lisa_noise_psd)
            band = (freqs .>= 1e-3) .& (freqs .<= 2e-2)
            @test isapprox(mean(table[band] ./ psd_channel.(freqs[band])), 1.0; atol = 0.15)
            @test !isapprox(
                mean(table[band] ./ lisa_noise_psd.(freqs[band])),
                1.0;
                atol = 0.15,
            )
            # the pre-processor's "channel" whitening is that PSD
            pre = merge(config["preprocessing"], Dict{String,Any}("psd" => "channel"))
            settings = preprocessing_settings(Dict{String,Any}("preprocessing" => pre))
            psd_w, description, table_w = whitening_psd(settings, noise.channels.A, 0.2)
            @test psd_w(3e-3) ≈ psd_channel(3e-3)
            @test occursin("Michelson-channel", description) && table_w === nothing
        end
    end
end
