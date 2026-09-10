ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "common.jl"))

using Random, Statistics, Plots, ArgParse, HDF5, CSV, DataFrames, TOML, UUIDs
using MilliHertzQML

# Publication-ready plotting setup
default(
    dpi = 600,
    frame = :box,
    fontfamily = "Computer Modern",
    grid = true,
    gridalpha = 0.2,
    minorgrid = false,
    margin = 5Plots.mm,
)

function parse_commandline()
    s = ArgParseSettings(
        description = "Continuous LISA telemetry simulator (milliHertz band)",
    )
    @add_arg_table s begin
        "--config"
        help = "Path to the configuration file"
        default = joinpath(dirname(@__DIR__), "config.toml")
        "--days"
        help = "Number of days of continuous telemetry to simulate"
        arg_type = Float64
        default = nothing
        "--fs"
        help = "Sampling frequency in Hz (LISA L1 cadence is 0.2 Hz)"
        arg_type = Float64
        default = nothing
        "--n-mbhb"
        help = "Number of massive black hole binary events to inject"
        arg_type = Int
        default = nothing
        "--n-gbs"
        help = "Number of resolvable galactic binaries above the confusion foreground"
        arg_type = Int
        default = nothing
        "--n-emris"
        help = "Number of extreme mass ratio inspirals"
        arg_type = Int
        default = nothing
        "--output"
        help = "Path to the output HDF5 file"
        default = nothing
        "--run-id"
        help = "Optional custom run ID for the simulation output plots"
        default = ""
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    # 1. Load and validate the TOML configuration
    config_file = load_config(parsed_args["config"])
    gen_cfg = get(config_file, "generation", Dict{String,Any}())

    # 2. Harmonize CLI with TOML defaults (CLI takes precedence)
    days = override(
        parsed_args["days"],
        cfgget(gen_cfg, "days", 30.0; type = Float64, min = 0.0),
    )
    fs = override(parsed_args["fs"], cfgget(gen_cfg, "fs", 0.2; type = Float64, min = 1e-6))
    n_mbhb =
        override(parsed_args["n-mbhb"], cfgget(gen_cfg, "n_mbhb", 5; type = Int, min = 0))
    n_gbs =
        override(parsed_args["n-gbs"], cfgget(gen_cfg, "n_gbs", 50; type = Int, min = 0))
    n_emris =
        override(parsed_args["n-emris"], cfgget(gen_cfg, "n_emris", 5; type = Int, min = 0))
    out_file = resolvepath(
        override(
            parsed_args["output"],
            cfgget(gen_cfg, "output", "data/inputs/simulated_telemetry.h5"; type = String),
        ),
    )
    run_id = isempty(parsed_args["run-id"]) ? string(uuid4())[1:8] : parsed_args["run-id"]
    seed = cfgget(gen_cfg, "seed", 42; type = Int)
    observation_years = cfgget(
        gen_cfg,
        "observation_years",
        1.0;
        type = Float64,
        choices = (0.5, 1.0, 2.0, 4.0),
    )
    snr_min = cfgget(gen_cfg, "snr_min", 8.0; type = Float64, min = 1e-3)
    snr_max = cfgget(gen_cfg, "snr_max", 50.0; type = Float64, min = snr_min)
    gb_snr_min = cfgget(gen_cfg, "gb_snr_min", 1.0; type = Float64, min = 1e-3)
    gb_snr_max = cfgget(gen_cfg, "gb_snr_max", 10.0; type = Float64, min = gb_snr_min)
    emri_snr_min = cfgget(gen_cfg, "emri_snr_min", 2.0; type = Float64, min = 1e-3)
    emri_snr_max = cfgget(gen_cfg, "emri_snr_max", 10.0; type = Float64, min = emri_snr_min)
    label_before_sec =
        cfgget(gen_cfg, "label_before_sec", 43200.0; type = Float64, min = 0.0)
    label_after_sec = cfgget(gen_cfg, "label_after_sec", 3600.0; type = Float64, min = 0.0)
    mbhb_duration_days =
        cfgget(gen_cfg, "mbhb_duration_days", 2.0; type = Float64, min = 1e-3)
    noise_f_min = cfgget(gen_cfg, "noise_f_min_hz", 1e-5; type = Float64, min = 0.0)
    mass_min = cfgget(gen_cfg, "mbhb_total_mass_min", 1e5; type = Float64, min = 1.0)
    mass_max = cfgget(gen_cfg, "mbhb_total_mass_max", 1e7; type = Float64, min = mass_min)
    mass_ratio_max = cfgget(gen_cfg, "mbhb_mass_ratio_max", 10.0; type = Float64, min = 1.0)
    nyquist_taper =
        cfgget(gen_cfg, "nyquist_taper", 0.9; type = Float64, min = 0.3, max = 1.0)
    label_span = cfgget(
        gen_cfg,
        "label_span",
        "detectable";
        type = String,
        choices = ("detectable", "injection", "fixed"),
    )
    label_snr_threshold =
        cfgget(gen_cfg, "label_snr_threshold", 5.0; type = Float64, min = 1e-3)
    label_window_size = cfgget(gen_cfg, "label_window_size", 1000; type = Int, min = 2)
    label_step = cfgget(gen_cfg, "label_step", 10; type = Int, min = 1)

    rng = Xoshiro(seed)
    psd = f -> lisa_noise_psd(f; observation_years = observation_years)

    n_total = round(Int, days * 24 * 3600 * fs)
    n_total >= 2 ||
        throw(ArgumentError("days = $days at fs = $fs Hz yields $n_total samples."))
    t_arr = (0:(n_total-1)) ./ fs
    T_record = n_total / fs

    println(
        "================================================================================",
    )
    println("  GENERATING CONTINUOUS LISA TELEMETRY [ID: $run_id]")
    println(
        "================================================================================",
    )
    println("  Duration : $days days ($n_total samples at $fs Hz)")
    println("  Noise    : Robson–Cornish–Liu 2019, confusion fit $observation_years yr")
    println("  MBHB SNR : [$snr_min, $snr_max] (matched filter)")
    println("  MBHB mass: [$mass_min, $mass_max] M⊙, q ≤ $mass_ratio_max; IMRPhenomA")

    # 1. Instrument plus confusion noise at physical strain amplitude
    println("\n[1/4] Synthesizing calibrated Gaussian noise...")
    strain = synthesize_noise(rng, n_total, fs; psd = psd, f_min = noise_f_min)

    # 2. Resolvable sources above the confusion fit, scaled to a matched-filter
    #    SNR over the simulated record
    println("[2/4] Injecting resolvable sources ($n_gbs GBs, $n_emris EMRIs)...")
    for _ in 1:n_gbs
        f = rand(rng, 0.0001:0.00001:0.01)   # 0.1 mHz to 10 mHz
        φ = rand(rng) * 2π
        gb = cos.(2π * f .* t_arr .+ φ)
        ρ = gb_snr_min + rand(rng) * (gb_snr_max - gb_snr_min)
        strain .+= scale_to_snr(gb, fs, ρ; psd = psd)
    end
    for _ in 1:n_emris
        f0 = rand(rng, 0.001:0.0005:0.005)
        dfdt = rand(rng, 1e-9:1e-10:1e-8)
        sig = zeros(n_total)
        for harmonic in (1, 2, 3)
            f_t = harmonic .* (f0 .+ dfdt .* t_arr)
            phase = 2π .* cumsum(f_t) ./ fs
            sig .+= (1.0 / harmonic) .* cos.(phase)
        end
        ρ = emri_snr_min + rand(rng) * (emri_snr_max - emri_snr_min)
        strain .+= scale_to_snr(sig, fs, ρ; psd = psd)
    end

    # 3. MBHB injections aligned on the coalescence sample, with labels and an
    #    event catalog
    println("[3/4] Injecting $n_mbhb massive black hole binaries...")
    labels = zeros(Int32, n_total)
    snrs = zeros(Float32, n_total)
    catalog = DataFrame(
        event_id = Int[],
        t_c_sec = Float64[],
        merger_index = Int[],
        snr = Float64[],
        total_mass_msun = Float64[],
        mass_ratio = Float64[],
        eta = Float64[],
        f_merger_hz = Float64[],
        f_ring_hz = Float64[],
        f_cut_hz = Float64[],
        start_index = Int[],
        end_index = Int[],
        label_start_index = Int[],
        label_end_index = Int[],
    )
    duration_sec = min(mbhb_duration_days * 86400, 0.4 * T_record)
    for i in 1:n_mbhb
        k_c = rand(rng, round(Int, 0.1*n_total):round(Int, 0.9*n_total))
        k_c = clamp(k_c, 1, n_total)
        t_c = (k_c - 1) / fs
        total_mass =
            10.0^(log10(mass_min) + rand(rng) * (log10(mass_max) - log10(mass_min)))
        mass_ratio = 1 + rand(rng) * (mass_ratio_max - 1)
        signal, merger_index, pars = phenoma_waveform(
            fs,
            duration_sec;
            total_mass = total_mass,
            mass_ratio = mass_ratio,
            nyquist_taper = nyquist_taper,
        )

        # Truncate to the record before scaling, so the recorded SNR is that
        # of the injected samples.
        placed = zeros(n_total)
        covered = place_signal!(placed, signal, k_c, merger_index)
        isempty(covered) && continue
        ρ_target = snr_min + rand(rng) * (snr_max - snr_min)
        segment = scale_to_snr(view(placed, covered), fs, ρ_target; psd = psd)
        strain[covered] .+= segment

        if label_span == "detectable"
            placed[covered] .= segment
            span = detectable_span(
                placed,
                covered,
                fs,
                label_window_size,
                label_snr_threshold;
                step = label_step,
                psd = psd,
            )
            span === nothing && (span = (k_c+1):k_c)   # no window reaches the threshold
            lbl_start, lbl_end = first(span), last(span)
        elseif label_span == "injection"
            lbl_start, lbl_end = first(covered), last(covered)
        else
            lbl_start = clamp(k_c - round(Int, label_before_sec * fs), 1, n_total)
            lbl_end = clamp(k_c + round(Int, label_after_sec * fs), 1, n_total)
        end
        if lbl_start <= lbl_end
            labels[lbl_start:lbl_end] .= 1
            snrs[lbl_start:lbl_end] .= Float32(ρ_target)
        end
        push!(
            catalog,
            (
                i,
                t_c,
                k_c,
                ρ_target,
                total_mass,
                mass_ratio,
                pars.η,
                pars.f_merg,
                pars.f_ring,
                pars.f_cut,
                first(covered),
                last(covered),
                lbl_start,
                lbl_end,
            ),
        )
    end

    # 4. Persist: HDF5 strain (X ≡ 0, Z = √2 A so that the pre-processor
    #    round-trips), point-wise labels, event catalog, provenance snapshot
    mkpath(dirname(out_file))
    println("\n[4/4] Saving continuous telemetry to HDF5 ($out_file)...")
    h5open(out_file, "w") do f
        g_obs = create_group(f, "obs")
        g_tdi = create_group(g_obs, "tdi")
        g_tdi["t"] = collect(t_arr)
        g_tdi["X"] = zeros(n_total)
        g_tdi["Z"] = strain .* sqrt(2.0)
    end

    label_file = replace(out_file, ".h5" => "_labels.csv")
    CSV.write(label_file, DataFrame(Label = labels, SNR = snrs))
    println("  - Labels saved to: $label_file")
    catalog_file = replace(out_file, ".h5" => "_events.csv")
    CSV.write(catalog_file, catalog)
    println("  - Event catalog saved to: $catalog_file ($(nrow(catalog)) events)")

    snapshot = Dict(
        "generation" => Dict(
            "days" => days,
            "fs" => fs,
            "n_mbhb" => n_mbhb,
            "n_gbs" => n_gbs,
            "n_emris" => n_emris,
            "output" => rootrelative(out_file),
            "seed" => seed,
            "run_id" => run_id,
            "observation_years" => observation_years,
            "snr_min" => snr_min,
            "snr_max" => snr_max,
            "gb_snr_min" => gb_snr_min,
            "gb_snr_max" => gb_snr_max,
            "emri_snr_min" => emri_snr_min,
            "emri_snr_max" => emri_snr_max,
            "label_before_sec" => label_before_sec,
            "label_after_sec" => label_after_sec,
            "mbhb_duration_days" => mbhb_duration_days,
            "noise_f_min_hz" => noise_f_min,
            "mbhb_total_mass_min" => mass_min,
            "mbhb_total_mass_max" => mass_max,
            "mbhb_mass_ratio_max" => mass_ratio_max,
            "nyquist_taper" => nyquist_taper,
            "label_span" => label_span,
            "label_snr_threshold" => label_snr_threshold,
            "label_window_size" => label_window_size,
            "label_step" => label_step,
            "n_events_injected" => nrow(catalog),
        ),
        "hardware" => hardware_fingerprint(),
    )
    open(replace(out_file, ".h5" => "_generation.toml"), "w") do io
        TOML.print(io, snapshot)
    end

    println("Generating trace plot...")
    ds = max(1, round(Int, n_total / 5000))
    t_days = t_arr ./ (24 * 3600)

    plot_dir = joinpath(pipeline_paths(config_file).plots, "run_$run_id")
    mkpath(plot_dir)

    p = plot(
        t_days[1:ds:end],
        strain[1:ds:end],
        xlabel = "Mission time [days]",
        ylabel = "Strain",
        lw = 0.5,
        color = :gray,
        label = "Data",
    )
    plot!(
        p,
        t_days[1:ds:end],
        labels[1:ds:end] .* maximum(abs, strain) / 2,
        st = :step,
        color = :red,
        alpha = 0.5,
        fill = (0, 0.5, :red),
        label = "MBHB label span",
    )
    savefig(joinpath(plot_dir, "simulated_continuous_trace.png"))

    println("\n[SUCCESS] Continuous pipeline dataset generated. Run ID: $run_id")
end

main()
