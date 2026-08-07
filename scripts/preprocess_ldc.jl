using Pkg
Pkg.activate(dirname(@__DIR__); io = devnull)
Pkg.instantiate(; io = devnull)

using HDF5, CSV, DataFrames, Statistics, MilliHertzQML, ArgParse, TOML

const PROJECT_ROOT = dirname(@__DIR__)
resolvepath(p) = isabspath(p) ? p : joinpath(PROJECT_ROOT, p)

function parse_commandline()
    s = ArgParseSettings(description = "Pre-process raw HDF5 telemetry into QNN features")
    @add_arg_table s begin
        "--config"
            help = "Path to the configuration file"
            default = joinpath(dirname(@__DIR__), "config.toml")
        "--h5-file"
            help = "Path to the raw HDF5 telemetry file"
            default = nothing
        "--label-file"
            help = "Path to the associated labels CSV (optional, used for training data)"
            default = ""
        "--output-prefix"
            help = "Prefix for the output CSV files (e.g., 'telemetry_train')"
            default = nothing
        "--window-size"
            help = "Size of the sliding window in samples"
            arg_type = Int
            default = nothing
        "--step-size"
            help = "Step size for the sliding window in samples"
            arg_type = Int
            default = nothing
        "--sample-rate"
            help = "Sampling rate of the data (usually 0.2 Hz)"
            arg_type = Float64
            default = nothing
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    # 1. Load TOML
    config_file = isfile(parsed_args["config"]) ? TOML.parsefile(parsed_args["config"]) : Dict{String, Any}()
    pre_cfg = get(config_file, "preprocessing", Dict{String, Any}())

    # 2. Harmonize CLI with TOML Defaults
    h5_path = resolvepath(parsed_args["h5-file"] !== nothing ? parsed_args["h5-file"] : get(pre_cfg, "h5_file", "data/inputs/simulated_telemetry.h5"))
    window_size = parsed_args["window-size"] !== nothing ? parsed_args["window-size"] : get(pre_cfg, "window_size", 1000)
    step_size = parsed_args["step-size"] !== nothing ? parsed_args["step-size"] : get(pre_cfg, "step_size", 100)
    fs = parsed_args["sample-rate"] !== nothing ? parsed_args["sample-rate"] : get(pre_cfg, "sample_rate", 0.2)
    output_prefix = parsed_args["output-prefix"] !== nothing ? parsed_args["output-prefix"] : get(pre_cfg, "output_prefix", "telemetry")

    println("================================================================================")
    println("  TELEMETRY PRE-PROCESSOR (HDF5 -> QNN Features)")
    println("================================================================================")
    println("File: $h5_path")
    println("Window: $window_size | Step: $step_size | FS: $(fs) Hz")

    # 1. Read HDF5 Data
    println("\n[1/4] Reading TDI channels from HDF5...")
    local x_obs, z_obs
    h5open(h5_path, "r") do f
        # Handle both real compound arrays and simple simulated arrays
        tdi = f["obs"]["tdi"]
        if typeof(tdi) <: HDF5.Group
            x_obs = Float32.(read(tdi["X"]))
            z_obs = Float32.(read(tdi["Z"]))
        else
            d = read(tdi)
            x_obs = Float32[row.X for row in d]
            z_obs = Float32[row.Z for row in d]
        end
    end

    n_points = length(x_obs)
    # 2. Compute Optimal A Channel
    println("[2/4] Computing orthogonal A channel (strain)...")
    A_obs = (z_obs .- x_obs) ./ Float32(sqrt(2.0))

    # 3. Read Labels (if provided)
    has_labels = !isempty(parsed_args["label-file"])
    local raw_labels
    local raw_snrs
    if has_labels
        println("[*] Loading raw point-wise labels...")
        label_df = CSV.read(resolvepath(parsed_args["label-file"]), DataFrame)
        raw_labels = label_df[:, :Label]
        raw_snrs = "SNR" in names(label_df) ? label_df[:, :SNR] : zeros(Float32, n_points)
    end

    # 4. Sliding Window Feature Extraction
    n_points >= window_size || error(
        "Telemetry too short: $n_points samples < window_size = $window_size.")
    n_windows = div(n_points - window_size, step_size) + 1
    println("[3/4] Extracting quantum features via sliding window (N = $n_windows)...")

    features = zeros(Float32, n_windows, 4)
    window_labels = zeros(Int, n_windows)
    window_snrs = zeros(Float32, n_windows)

    for i in 1:n_windows
        start_idx = (i - 1) * step_size + 1
        end_idx = start_idx + window_size - 1

        window = A_obs[start_idx:end_idx]

        # Use the unified extract_features from MilliHertzQML
        p_low, p_high, entropy, psd_std = extract_features(window, fs)

        features[i, :] .= [p_low, p_high, entropy, psd_std]

        # Label is 1 if any point in the window is an MBHB
        if has_labels
            window_labels[i] = any(raw_labels[start_idx:end_idx] .== 1) ? 1 : 0
            # Assign the max SNR present in the window to the whole window
            window_snrs[i] = maximum(raw_snrs[start_idx:end_idx])
        end

        if i % max(1, div(n_windows, 10)) == 0
            println("  Progress: $(round(Int, i/n_windows*100))%")
        end
    end

    # 5. Save Results
    println("\n[4/4] Saving processed data to CSV...")
    out_dir = joinpath(PROJECT_ROOT, "data", "inputs")
    mkpath(out_dir)

    prefix = parsed_args["output-prefix"] !== nothing ? parsed_args["output-prefix"] : output_prefix
    feat_path = joinpath(out_dir, "$(prefix)_features.csv")
    CSV.write(feat_path, DataFrame(features, [:PLow, :PHigh, :Entropy, :Log_PSD_Std]))
    println("  - Features saved to: $feat_path")

    if has_labels
        lab_path = joinpath(out_dir, "$(prefix)_labels.csv")
        CSV.write(lab_path, DataFrame(Label=window_labels, SNR=window_snrs))
        println("  - Labels saved to: $lab_path")
    end

    println("\n[SUCCESS] Pre-processing complete. Ready for Quantum Training.")
end

main()
