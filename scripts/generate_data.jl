ENV["GKSwstype"] = "100"
using Pkg
Pkg.activate("QuantumGW", io=devnull)
push!(LOAD_PATH, "QuantumGW/src")

using Random, Statistics, FFTW, Plots, ArgParse, HDF5, CSV, DataFrames, TOML, UUIDs

# Publication-ready plotting setup
default(dpi=600, frame=:box, fontfamily="Computer Modern", grid=true, gridalpha=0.2, minorgrid=false, margin=5Plots.mm)

function parse_commandline()
    s = ArgParseSettings(description = "Continuous LISA Telemetry Simulator (MilliHertz Regime)")
    @add_arg_table s begin
        "--config"
            help = "Path to the configuration file"
            default = "QuantumGW/config.toml"
        "--days"
            help = "Number of days of continuous telemetry to simulate"
            arg_type = Float64
            default = nothing
        "--fs"
            help = "Sampling frequency in Hz (LISA nominal is ~0.2 Hz)"
            arg_type = Float64
            default = nothing
        "--n-mbhb"
            help = "Number of Massive Black Hole Binary events to inject"
            arg_type = Int
            default = nothing
        "--n-gbs"
            help = "Number of continuous Galactic Binaries (Background source confusion)"
            arg_type = Int
            default = nothing
        "--n-emris"
            help = "Number of Extreme Mass Ratio Inspirals"
            arg_type = Int
            default = nothing
        "--output"
            help = "Path to output HDF5 file"
            default = nothing
        "--run-id"
            help = "Optional custom Run ID for the simulation output plots"
            default = ""
    end
    return parse_args(s)
end

# Physics Constants
const L_ARM = 2.5e9            # LISA arm length (m)
const C_LIGHT = 2.99792458e8   # Speed of light (m/s)
const F_STAR = C_LIGHT / (2π * L_ARM) # Transfer frequency (~19 mHz)

# --- LISA Noise PSD (Robson, Cornish, Liu 2019) ---
function lisa_noise_psd(f)
    if f <= 0.0; return 1e-30; end
    p_oms = (1.5e-11)^2 * (1 + (2e-3/f)^4)
    p_acc = (3e-15)^2 * (1 + (0.4e-3/f)^2) * (1 + (f/8e-3)^4)
    s_inst = (p_oms / L_ARM^2) + (2 * p_acc / ( (2π*f)^4 * L_ARM^2 )) * (1 + cos(f/F_STAR)^2)

    A, fk, B, C, D = 1.8e-44, 1.0e-4, 292.0, 10.0^(-3.5), 10.0^(-4.5)
    s_gal = A * f^(-7/3) * exp(-(f/fk)^B) * (1 + tanh((C-f)/D))

    return s_inst + s_gal
end

function generate_lisa_noise(fs, n_samples)
    freqs = rfftfreq(n_samples, fs)
    psd = lisa_noise_psd.(freqs)
    white_f = randn(ComplexF32, length(freqs)) .* sqrt.(psd ./ 2.0)
    white_f[1] = 0 # DC
    noise = irfft(white_f, n_samples)
    return Float32.(noise ./ std(noise))
end

function generate_mbhb(fs, t_c, duration_secs)
    # Generate the signal for `duration_secs` ending at t_c
    n_samples = Int(round(duration_secs * fs))
    t = range(t_c - duration_secs, t_c, length=n_samples)

    # milliHertz physics scaling
    chirp_scale = rand(0.5:0.1:2.0)
    tau = max.(t_c .- t, 0.1)
    phase = -2.0 .* chirp_scale .* tau.^(5/8)
    amp_insp = tau.^(-0.25)

    f_ring = rand(0.005:0.001:0.05) # 5 mHz to 50 mHz
    tau_ring = rand(100.0:10.0:500.0) # Seconds

    merger_idx = length(t)
    amp_at_merger = amp_insp[merger_idx]

    # Create ringdown part
    n_ring = Int(round(tau_ring * 5 * fs)) # 5 time constants
    t_ring = range(t_c, t_c + tau_ring*5, length=n_ring)
    amp_ring = amp_at_merger .* exp.(-(t_ring .- t_c) ./ tau_ring)

    sig = vcat((amp_insp .* cos.(phase)), (amp_ring .* cos.(2π * f_ring .* (t_ring .- t_c) .+ phase[end])))
    time_vec = vcat(t, t_ring)

    return time_vec, Float32.(sig)
end

function main()
    parsed_args = parse_commandline()

    # 1. Load TOML
    config_file = isfile(parsed_args["config"]) ? TOML.parsefile(parsed_args["config"]) : Dict{String, Any}()
    gen_cfg = get(config_file, "generation", Dict{String, Any}())

    # 2. Harmonize CLI with TOML Defaults
    days = parsed_args["days"] !== nothing ? parsed_args["days"] : get(gen_cfg, "days", 30.0)
    fs = parsed_args["fs"] !== nothing ? parsed_args["fs"] : get(gen_cfg, "fs", 0.2)
    n_mbhb = parsed_args["n-mbhb"] !== nothing ? parsed_args["n-mbhb"] : get(gen_cfg, "n_mbhb", 5)
    n_gbs = parsed_args["n-gbs"] !== nothing ? parsed_args["n-gbs"] : get(gen_cfg, "n_gbs", 50)
    n_emris = parsed_args["n-emris"] !== nothing ? parsed_args["n-emris"] : get(gen_cfg, "n_emris", 5)
    out_file = parsed_args["output"] !== nothing ? parsed_args["output"] : get(gen_cfg, "output", "QuantumGW/data/inputs/simulated_telemetry.h5")
    run_id = isempty(parsed_args["run-id"]) ? string(uuid4())[1:8] : parsed_args["run-id"]

    n_total = Int(round(days * 24 * 3600 * fs))
    t_arr = range(0, days * 24 * 3600, length=n_total)

    println("================================================================================")
    println("  🌌 GENERATING CONTINUOUS LISA TELEMETRY [ID: $run_id]")
    println("================================================================================")
    println("  Duration : $days Days")
    println("  Samples  : $n_total")
    println("  Sampling : $fs Hz")

    # 1. Generate Base Noise
    println("\n[1/4] Generating LISA Instrumental + Confusion Noise...")
    strain = generate_lisa_noise(fs, n_total)

    # 2. Add Background Forest
    println("[2/4] Injecting Background Source Confusion ($n_gbs GBs & $n_emris EMRIs)...")
    for _ in 1:n_gbs
        f = rand(0.0001:0.00001:0.01) # 0.1mHz to 10mHz
        phase = rand(0:0.1:2π)
        gb = Float32.(cos.(2π * f .* t_arr .+ phase))
        snr = rand(0.5:0.1:2.0)
        strain .+= gb .* (snr / std(gb))
    end

    for _ in 1:n_emris
        f0 = rand(0.001:0.0005:0.005)
        dfdt = rand(1e-9:1e-10:1e-8)
        sig = zeros(Float32, n_total)
        for harmonic in [1, 2, 3]
            f_t = harmonic .* (f0 .+ dfdt .* t_arr)
            phase = 2π .* cumsum(f_t) ./ fs
            sig .+= (1.0f0/harmonic) .* cos.(phase)
        end
        snr = rand(1.0:0.1:3.0)
        strain .+= sig .* (snr / std(sig))
    end

    # 3. Inject MBHBs and Create Labels
    println("[3/4] Injecting $n_mbhb Massive Black Hole Binaries...")
    labels = zeros(Int32, n_total)
    snrs = zeros(Float32, n_total)

    for i in 1:n_mbhb
        t_c = rand(t_arr[Int(round(n_total*0.1))] : t_arr[Int(round(n_total*0.9))])
        t_vec, sig = generate_mbhb(fs, t_c, 2 * 24 * 3600)

        snr = rand(3.0:0.5:8.0) # Realistic low SNR
        sig_scaled = sig .* (snr / std(sig))

        start_idx = searchsortedfirst(t_arr, t_vec[1])
        end_idx = start_idx + length(sig) - 1

        if end_idx <= n_total && start_idx > 0
            strain[start_idx:end_idx] .+= sig_scaled

            label_start_t = t_c - (12 * 3600)
            label_end_t = t_c + 3600

            lbl_start_idx = searchsortedfirst(t_arr, label_start_t)
            lbl_end_idx = searchsortedfirst(t_arr, label_end_t)
            labels[lbl_start_idx:lbl_end_idx] .= 1
            snrs[lbl_start_idx:lbl_end_idx] .= snr
        end
    end

    # 4. Save to HDF5 + CSV
    mkpath(dirname(out_file))
    println("\n[4/4] Saving continuous telemetry to HDF5 ($out_file)...")

    z_chan = strain .* Float32(sqrt(2.0))
    x_chan = zeros(Float32, n_total)

    h5open(out_file, "w") do f
        g_obs = create_group(f, "obs")
        g_tdi = create_group(g_obs, "tdi")
        g_tdi["t"] = collect(t_arr)
        g_tdi["X"] = x_chan
        g_tdi["Z"] = z_chan
    end

    label_file = replace(out_file, ".h5" => "_labels.csv")
    CSV.write(label_file, DataFrame(Label=labels, SNR=snrs))
    println("  - Labels saved to: $label_file")

    println("Generating trace plot...")
    ds = max(1, Int(round(n_total / 5000)))
    t_days = t_arr ./ (24*3600)

    plot_dir = "QuantumGW/data/outputs/plots/run_$run_id"
    mkpath(plot_dir)

    p = plot(t_days[1:ds:end], strain[1:ds:end], title="Simulated Continuous Telemetry",
             xlabel="Mission Time [Days]", ylabel="Strain", lw=0.5, color=:gray, label="Data")
    plot!(p, t_days[1:ds:end], labels[1:ds:end] .* maximum(strain)/2, st=:step, color=:red, alpha=0.5, fill=(0, 0.5, :red), label="MBHB Events")
    savefig(joinpath(plot_dir, "simulated_continuous_trace.png"))

    println("\n[SUCCESS] Continuous pipeline dataset generated. Run ID: $run_id")
end

main()
