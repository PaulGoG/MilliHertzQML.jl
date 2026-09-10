include(joinpath(@__DIR__, "common.jl"))

using HDF5, CSV, DataFrames, Statistics, MilliHertzQML, ArgParse, TOML

function parse_commandline()
    s = ArgParseSettings(
        description = "Pre-process raw HDF5 telemetry into whitened window features",
    )
    @add_arg_table s begin
        "--config"
        help = "Path to the configuration file"
        default = joinpath(dirname(@__DIR__), "config.toml")
        "--h5-file"
        help = "Path to the raw HDF5 telemetry file"
        default = nothing
        "--label-file"
        help = "Path to the associated point-wise label CSV (optional, used for training data)"
        default = ""
        "--output-prefix"
        help = "Prefix of the output CSV files (e.g. 'telemetry_train')"
        default = nothing
        "--window-size"
        help = "Length of the sliding window in samples"
        arg_type = Int
        default = nothing
        "--step-size"
        help = "Step of the sliding window in samples"
        arg_type = Int
        default = nothing
        "--sample-rate"
        help = "Sampling frequency of the telemetry in Hz"
        arg_type = Float64
        default = nothing
    end
    return parse_args(s)
end

"""
    analysis_band(section, key, default) -> Tuple{Float64,Float64}

Two-element ascending frequency band [Hz] from the configuration.
"""
function analysis_band(section, key, default)
    band = cfgget(section, key, default; type = AbstractVector)
    (length(band) == 2 && all(x -> x isa Real, band) && 0 < band[1] < band[2]) || throw(
        ArgumentError(
            "configuration key `$key` = $(repr(band)); expected two ascending positive frequencies [Hz].",
        ),
    )
    return (Float64(band[1]), Float64(band[2]))
end

function main()
    parsed_args = parse_commandline()

    # 1. Load and validate the TOML configuration
    config_file = load_config(parsed_args["config"])
    pre_cfg = get(config_file, "preprocessing", Dict{String,Any}())

    # 2. Harmonize CLI with TOML defaults (CLI takes precedence)
    h5_path = resolvepath(
        override(
            parsed_args["h5-file"],
            cfgget(pre_cfg, "h5_file", "data/inputs/simulated_telemetry.h5"; type = String),
        ),
    )
    window_size = override(
        parsed_args["window-size"],
        cfgget(pre_cfg, "window_size", 1000; type = Int, min = 2),
    )
    step_size = override(
        parsed_args["step-size"],
        cfgget(pre_cfg, "step_size", 100; type = Int, min = 1),
    )
    fs = override(
        parsed_args["sample-rate"],
        cfgget(pre_cfg, "sample_rate", 0.2; type = Float64, min = 1e-6),
    )
    output_prefix = override(
        parsed_args["output-prefix"],
        cfgget(pre_cfg, "output_prefix", "telemetry"; type = String),
    )
    observation_years = cfgget(
        pre_cfg,
        "observation_years",
        1.0;
        type = Float64,
        choices = (0.5, 1.0, 2.0, 4.0),
    )
    highpass_cutoff = cfgget(pre_cfg, "highpass_cutoff_hz", 5e-4; type = Float64, min = 0.0)
    highpass_order = cfgget(pre_cfg, "highpass_order", 8; type = Int, min = 1)
    low_band = analysis_band(pre_cfg, "low_band_hz", [1e-3, 5e-3])
    high_band = analysis_band(pre_cfg, "high_band_hz", [5e-3, 1e-1])
    step_size <= window_size ||
        throw(ArgumentError("step_size = $step_size exceeds window_size = $window_size."))
    psd = f -> lisa_noise_psd(f; observation_years = observation_years)

    println(
        "================================================================================",
    )
    println("  TELEMETRY PRE-PROCESSOR (HDF5 -> whitened window features)")
    println(
        "================================================================================",
    )
    println("File: $h5_path")
    println("Window: $window_size | Step: $step_size | FS: $(fs) Hz")
    println("Whitening PSD: Robson–Cornish–Liu 2019, confusion fit $observation_years yr")
    println("Record high-pass: $(highpass_cutoff) Hz, order $highpass_order")

    # 1. Read the TDI channels
    println("\n[1/4] Reading TDI channels from HDF5...")
    local x_obs, z_obs
    h5open(h5_path, "r") do f
        # Simple arrays (simulator) or a compound dataset (LDC)
        tdi = f["obs"]["tdi"]
        if typeof(tdi) <: HDF5.Group
            x_obs = Float64.(read(tdi["X"]))
            z_obs = Float64.(read(tdi["Z"]))
        else
            d = read(tdi)
            x_obs = Float64[row.X for row in d]
            z_obs = Float64[row.Z for row in d]
        end
    end

    n_points = length(x_obs)
    # 2. Orthogonal A channel
    println("[2/4] Computing the orthogonal A channel...")
    A_obs = (z_obs .- x_obs) ./ sqrt(2.0)
    # Zero-phase high-pass below the analysis bands: the steep low-frequency
    # noise would otherwise leak into every window through the taper.
    A_obs = highpass_record(A_obs, fs; cutoff = highpass_cutoff, order = highpass_order)
    # Whiten the whole record by the model PSD: noise becomes unit-variance
    # white, so the tapered periodogram of every window is unbiased.
    A_obs = whiten_record(A_obs, fs; psd = psd)

    # 3. Point-wise labels (optional)
    has_labels = !isempty(parsed_args["label-file"])
    local raw_labels
    local raw_snrs
    if has_labels
        println("[*] Loading point-wise labels...")
        label_df = CSV.read(resolvepath(parsed_args["label-file"]), DataFrame)
        nrow(label_df) == n_points || throw(
            DimensionMismatch("$(nrow(label_df)) labels for $n_points telemetry samples."),
        )
        raw_labels = label_df[:, :Label]
        raw_snrs = "SNR" in names(label_df) ? label_df[:, :SNR] : zeros(Float32, n_points)
    end

    # 4. Sliding-window feature extraction
    n_points >= window_size ||
        error("Telemetry too short: $n_points samples < window_size = $window_size.")
    n_windows = div(n_points - window_size, step_size) + 1
    println("[3/4] Extracting whitened features over $n_windows windows...")

    features = zeros(Float32, n_windows, 4)
    window_labels = zeros(Int, n_windows)
    window_snrs = zeros(Float32, n_windows)

    for i in 1:n_windows
        start_idx = (i - 1) * step_size + 1
        end_idx = start_idx + window_size - 1
        window = @view A_obs[start_idx:end_idx]

        p_low, p_high, entropy, log_power_std =
            extract_features(window, fs; low_band = low_band, high_band = high_band)
        features[i, :] .= (p_low, p_high, entropy, log_power_std)

        # A window is positive when any of its samples is labeled; it carries
        # the largest per-sample SNR inside it.
        if has_labels
            window_labels[i] = any(==(1), @view raw_labels[start_idx:end_idx]) ? 1 : 0
            window_snrs[i] = maximum(@view raw_snrs[start_idx:end_idx])
        end

        if i % max(1, div(n_windows, 10)) == 0
            println("  Progress: $(round(Int, i / n_windows * 100))%")
        end
    end

    # 5. Persist
    println("\n[4/4] Saving processed data to CSV...")
    out_dir = pipeline_paths(config_file).inputs
    mkpath(out_dir)

    feat_path = joinpath(out_dir, "$(output_prefix)_features.csv")
    lab_path = joinpath(out_dir, "$(output_prefix)_labels.csv")
    # Never overwrite an input: an output prefix equal to the stem of the
    # telemetry file would land the window labels on the point-wise labels.
    inputs = filter(
        !isempty,
        [h5_path, has_labels ? resolvepath(parsed_args["label-file"]) : ""],
    )
    for out in (feat_path, lab_path), inp in inputs
        abspath(out) == abspath(inp) && throw(
            ArgumentError(
                "output $out coincides with input $inp; choose another output prefix.",
            ),
        )
    end
    CSV.write(
        feat_path,
        DataFrame(features, [:p_low, :p_high, :spectral_entropy, :log_power_std]),
    )
    println("  - Features saved to: $feat_path")

    if has_labels
        CSV.write(lab_path, DataFrame(Label = window_labels, SNR = window_snrs))
        println("  - Labels saved to: $lab_path")
    end

    println("\n[SUCCESS] Pre-processing complete.")
end

main()
