# test/export_payload_tests.jl — payload-export stage: the A-channel payload
# CSV, the scenario fragment with its markers and label spans, the settings
# reader, and the batch-geometry guard. Included from runtests.jl, which
# loads Test, TOML, CSV, DataFrames, Dates, StableRNGs, and MilliHertzQML.

@testset "Telemetry payload export" begin
    @testset "Settings and batch geometry" begin
        defaults = telemetry_settings(Dict{String,Any}())
        @test defaults.segment_duration_sec == 50.0
        @test defaults.batch_size == 10
        @test defaults.start_sim_time == DateTime(2035, 1, 1)
        @test defaults.output_prefix == "telemetry"
        bad(key, value) = Dict{String,Any}("telemetry" => Dict{String,Any}(key => value))
        @test_throws ArgumentError telemetry_settings(
            bad("start_sim_time", "mission start"),
        )
        @test_throws ArgumentError telemetry_settings(bad("batch_size", 0))
        @test_throws ArgumentError telemetry_settings(bad("segment_duration_sec", 0))
        @test samples_per_batch(0.2, 50, 10) == 100
        @test samples_per_batch(4.0, 0.5, 4) == 8
        # A segment must hold a whole number of samples
        @test_throws ArgumentError samples_per_batch(0.2, 52, 10)
        @test_throws ArgumentError samples_per_batch(0.2, 1, 10)
    end

    @testset "Catalog schemas" begin
        sim = catalog_events(DataFrame(event_id = [3, 4], t_c_sec = [10.0, 20.0]))
        @test sim.ids == ["3", "4"] && sim.times == [10.0, 20.0] && sim.spans === nothing
        ldc = catalog_events(
            DataFrame(
                event = [1],
                merger_time_s = [12.5],
                label_start_index = [2],
                label_end_index = [9],
            ),
        )
        @test ldc.ids == ["1"] && ldc.times == [12.5] && ldc.spans == [(2, 9)]
        @test_throws ArgumentError catalog_events(DataFrame(name = ["a"], time = [1.0]))
    end

    mktempdir() do dir
        fs = 0.2
        n = 3000
        noise = synthesize_noise(StableRNG(2035), n, fs)
        t = collect((0:(n-1)) ./ fs)
        X = zeros(n)
        Z = sqrt(2.0) .* noise
        h5 = joinpath(dir, "record.h5")
        # HDF5 is a package dependency, not a test dependency; the simulator's
        # group layout is written through the module.
        MilliHertzQML.HDF5.h5open(h5, "w") do file
            g_obs = MilliHertzQML.HDF5.create_group(file, "obs")
            g_tdi = MilliHertzQML.HDF5.create_group(g_obs, "tdi")
            g_tdi["t"] = t
            g_tdi["X"] = X
            g_tdi["Z"] = Z
        end
        catalog_path = joinpath(dir, "record_events.csv")
        CSV.write(
            catalog_path,
            DataFrame(
                event_id = [1, 2],
                t_c_sec = [1500.0, 9005.0],
                merger_index = [301, 1802],
                snr = [12.0, 30.0],
                label_start_index = [250, 1700],
                label_end_index = [320, 1850],
            ),
        )
        start = DateTime(2035, 1, 1)
        config = Dict{String,Any}(
            "paths" => Dict{String,Any}("inputs" => joinpath(dir, "inputs")),
            "preprocessing" =>
                Dict{String,Any}("h5_file" => h5, "tdi_group" => "obs/tdi"),
            "telemetry" => Dict{String,Any}(
                "segment_duration_sec" => 50,
                "batch_size" => 10,
                "start_sim_time" => string(start),
                "output_prefix" => "unit",
            ),
        )

        @testset "Payload and scenario" begin
            result = export_telemetry_payload(config; catalog = catalog_path)
            @test result.n_rows == n
            @test result.sample_rate == fs
            @test result.start_sim_time == start
            @test result.payload_path == joinpath(dir, "inputs", "unit_payload.csv")
            @test result.scenario_path == joinpath(dir, "inputs", "unit_scenario.toml")

            payload = CSV.read(result.payload_path, DataFrame; types = Float32)
            @test names(payload) == ["Amplitude"]
            @test nrow(payload) == n
            @test eltype(payload.Amplitude) == Float32
            @test payload.Amplitude == Float32.((Z .- X) ./ sqrt(2))
            @test isapprox(payload.Amplitude, Float32.(noise); rtol = 1e-6)

            scenario = TOML.parsefile(result.scenario_path)
            for key in
                ("physics", "simulation", "events", "labels", "payload", "git", "hardware")
                @test haskey(scenario, key)
            end
            physics = scenario["physics"]
            @test physics["data_source"] == "external"
            @test resolvepath(physics["external_data_path"]) == result.payload_path
            @test physics["sample_rate"] == fs
            @test physics["segment_duration_sec"] == 50.0
            @test physics["batch_size"] == 10
            @test scenario["simulation"]["start_sim_time"] == string(start)
            @test DateTime(scenario["simulation"]["start_sim_time"]) == start

            markers = scenario["events"]["markers"]
            @test length(markers) == 2 == length(result.markers)
            @test [m["label"] for m in markers] == ["mbhb_1", "mbhb_2"]
            @test [DateTime(m["time"]) for m in markers] == [start + Millisecond(1_500_000), start + Millisecond(9_005_000)]
            @test [m.time for m in result.markers] == [DateTime(m["time"]) for m in markers]

            labels = scenario["labels"]
            @test [l["label"] for l in labels] == ["mbhb_1", "mbhb_2"]
            @test [l["label_start_index"] for l in labels] == [250, 1700]
            @test [l["label_end_index"] for l in labels] == [320, 1850]

            payload_meta = scenario["payload"]
            @test payload_meta["n_rows"] == n
            @test payload_meta["tdi_group"] == "obs/tdi"
            # Inputs outside the package root are recorded by file name, the
            # directory belonging to the machine that ran the stage rather
            # than to the run (`provenance_path`)
            @test payload_meta["source"] == basename(h5)
            @test payload_meta["catalog"] == basename(catalog_path)
            @test !occursin(homedir(), payload_meta["source"])
            @test payload_meta["samples_per_batch"] == 100
            @test payload_meta["n_batches"] == 30
        end

        @testset "LDC event table and millisecond markers" begin
            ldc_catalog = joinpath(dir, "ldc_events.csv")
            CSV.write(
                ldc_catalog,
                DataFrame(
                    event = [1, 2],
                    merger_index = [4, 1201],
                    merger_time_s = [12.5, 6000.25],
                    merger_window_snr = [9.0, 11.0],
                ),
            )
            result = export_telemetry_payload(
                config;
                catalog = ldc_catalog,
                output_prefix = "unit_ldc",
            )
            scenario = TOML.parsefile(result.scenario_path)
            @test [DateTime(m["time"]) for m in scenario["events"]["markers"]] == [start + Millisecond(12_500), start + Millisecond(6_000_250)]
            @test !haskey(scenario, "labels")   # no span columns in this table
        end

        @testset "Without a catalog" begin
            result = export_telemetry_payload(config; output_prefix = "unit_bare")
            @test isempty(result.markers)
            scenario = TOML.parsefile(result.scenario_path)
            @test isempty(scenario["events"]["markers"])
            @test !haskey(scenario, "labels")
            @test scenario["payload"]["catalog"] == ""
        end

        @testset "Fail-fast guards" begin
            # The record must hold one full batch: 10 segments of 50 s at
            # 0.2 Hz need 100 rows; 1000 segments need 10000
            short = deepcopy(config)
            short["telemetry"]["batch_size"] = 1000
            @test_throws ArgumentError export_telemetry_payload(
                short;
                catalog = catalog_path,
            )
            fractional = deepcopy(config)
            fractional["telemetry"]["segment_duration_sec"] = 52
            @test_throws ArgumentError export_telemetry_payload(fractional)
            # A coalescence outside the record means the catalogue is not this record's
            foreign = joinpath(dir, "foreign_events.csv")
            CSV.write(foreign, DataFrame(event_id = [1], t_c_sec = [1e6]))
            @test_throws ArgumentError export_telemetry_payload(config; catalog = foreign)
            # A non-empty label span outside the record
            outside = joinpath(dir, "outside_events.csv")
            CSV.write(
                outside,
                DataFrame(
                    event_id = [1],
                    t_c_sec = [100.0],
                    label_start_index = [10],
                    label_end_index = [5000],
                ),
            )
            @test_throws ArgumentError export_telemetry_payload(config; catalog = outside)
            @test_throws ArgumentError export_telemetry_payload(
                config;
                catalog = joinpath(dir, "absent.csv"),
            )
            @test_throws ArgumentError export_telemetry_payload(
                config;
                h5_file = joinpath(dir, "absent.h5"),
            )
        end
    end
end
