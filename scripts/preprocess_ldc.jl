include(joinpath(@__DIR__, "common.jl"))

using CSV, DataFrames, Statistics, MilliHertzQML, ArgParse, TOML

function parse_commandline()
    s = ArgParseSettings(
        description = "Pre-process an HDF5 TDI product into per-window features",
    )
    @add_arg_table s begin
        "--config"
        help = "Path to the configuration file"
        default = joinpath(dirname(@__DIR__), "config.toml")
        "--h5-file"
        help = "Path to the HDF5 TDI product (simulator output or LDC file)"
        default = nothing
        "--tdi-group"
        help = "HDF5 group or compound dataset holding t, X, Y, Z (default from [preprocessing] tdi_group)"
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
        help = "Expected sampling frequency in Hz; must agree with the file"
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

"""
    whitening_psd(mode, pre_cfg, A, fs) -> (psd, description, table)

Callable one-sided PSD used to whiten the record for `mode`: `"model"`
(Robson–Cornish–Liu strain sensitivity, for simulator products), `"ldc"`
(analytic TDI PSD of the `ldc` package in fractional-frequency units, for
LDC products), `"welch"` (median-averaged estimate from the record itself),
or `"none"` (no whitening; `psd` is `nothing`). `table` is the estimated
PSD for `"welch"` (persisted beside the features) and `nothing` otherwise.
"""
function whitening_psd(mode::AbstractString, pre_cfg::AbstractDict, A, fs)
    if mode == "model"
        years = cfgget(
            pre_cfg,
            "observation_years",
            1.0;
            type = Float64,
            choices = (0.5, 1.0, 2.0, 4.0),
        )
        return f -> lisa_noise_psd(f; observation_years = years),
        "Robson–Cornish–Liu 2019 strain sensitivity, confusion fit $years yr",
        nothing
    elseif mode == "ldc"
        model = cfgget(pre_cfg, "ldc_model", "sangria"; type = String)
        tdi2 = cfgget(pre_cfg, "ldc_tdi2", false; type = Bool)
        years = cfgget(pre_cfg, "ldc_observation_years", 0.0; type = Float64, min = 0.0)
        psd =
            f -> ldc_tdi_psd(
                f;
                channel = :A,
                model = model,
                tdi2 = tdi2,
                observation_years = years,
            )
        psd(1e-3)   # validates the model name before the record is processed
        return psd,
        "LDC analytic A-channel PSD, model $model, TDI $(tdi2 ? 2 : 1.5), confusion $years yr",
        nothing
    elseif mode == "welch"
        segment = cfgget(pre_cfg, "welch_segment_length", 65536; type = Int, min = 2)
        segment <= length(A) || throw(
            ArgumentError(
                "welch_segment_length = $segment exceeds the record length $(length(A)).",
            ),
        )
        freqs, table = welch_psd(A, fs; segment_length = segment, average = :median)
        return interpolated_psd(freqs, table),
        "median Welch estimate of the record, segment $segment samples",
        DataFrame(frequency_hz = freqs, psd = table)
    elseif mode == "none"
        return nothing, "none", nothing
    end
    throw(ArgumentError("psd = $(repr(mode)); expected model, ldc, welch, or none."))
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
    tdi_group = override(
        parsed_args["tdi-group"],
        cfgget(pre_cfg, "tdi_group", "obs/tdi"; type = String),
    )
    window_size = override(
        parsed_args["window-size"],
        cfgget(pre_cfg, "window_size", 1000; type = Int, min = 2),
    )
    step_size = override(
        parsed_args["step-size"],
        cfgget(pre_cfg, "step_size", 100; type = Int, min = 1),
    )
    expected_fs = override(
        parsed_args["sample-rate"],
        cfgget(pre_cfg, "sample_rate", nothing; type = Union{Nothing,Float64}),
    )
    output_prefix = override(
        parsed_args["output-prefix"],
        cfgget(pre_cfg, "output_prefix", "telemetry"; type = String),
    )
    psd_mode = cfgget(
        pre_cfg,
        "psd",
        "model";
        type = String,
        choices = ("model", "ldc", "welch", "none"),
    )
    feature_set = Symbol(
        cfgget(
            pre_cfg,
            "feature_set",
            "whitened";
            type = String,
            choices = ("whitened", "paper"),
        ),
    )
    highpass_cutoff = cfgget(pre_cfg, "highpass_cutoff_hz", 5e-4; type = Float64, min = 0.0)
    highpass_order = cfgget(pre_cfg, "highpass_order", 8; type = Int, min = 1)
    low_band = analysis_band(pre_cfg, "low_band_hz", [1e-3, 5e-3])
    high_band = analysis_band(pre_cfg, "high_band_hz", [5e-3, 1e-1])
    step_size <= window_size ||
        throw(ArgumentError("step_size = $step_size exceeds window_size = $window_size."))

    println(
        "================================================================================",
    )
    println("  TELEMETRY PRE-PROCESSOR (HDF5 -> per-window features)")
    println(
        "================================================================================",
    )
    println("File: $h5_path ($tdi_group)")

    # 3. Read the TDI channels; the file's sampling step is authoritative
    println("\n[1/4] Reading TDI channels from HDF5...")
    tdi = read_tdi(h5_path; group = tdi_group)
    fs = 1 / tdi.dt
    if expected_fs !== nothing && !isapprox(expected_fs, fs; rtol = 1e-9)
        throw(
            ArgumentError(
                "sample_rate = $expected_fs Hz disagrees with the file's $(fs) Hz (dt = $(tdi.dt) s).",
            ),
        )
    end
    n_points = length(tdi.t)
    println("Samples: $n_points | Window: $window_size | Step: $step_size | FS: $(fs) Hz")

    # 4. Orthogonal A channel, record high-pass, whitening
    println("[2/4] Computing the orthogonal A channel...")
    A_obs, _, _ = tdi_to_aet(tdi.X, tdi.Y, tdi.Z)
    if highpass_cutoff > 0
        # Zero-phase high-pass below the analysis bands: the steep low-frequency
        # noise would otherwise leak into every window through the taper.
        A_obs = highpass_record(A_obs, fs; cutoff = highpass_cutoff, order = highpass_order)
    end
    psd, psd_description, psd_table = whitening_psd(psd_mode, pre_cfg, A_obs, fs)
    println("Record high-pass: $(highpass_cutoff) Hz, order $highpass_order")
    println("Whitening PSD: $psd_description")
    println("Feature set: $feature_set")
    if psd !== nothing
        # Whiten the whole record: noise becomes unit-variance white, so the
        # tapered periodogram of every window is unbiased.
        A_obs = whiten_record(A_obs, fs; psd = psd)
    end

    # 5. Point-wise labels (optional)
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

    # 6. Sliding-window feature extraction
    n_points >= window_size ||
        error("Telemetry too short: $n_points samples < window_size = $window_size.")
    n_windows = div(n_points - window_size, step_size) + 1
    println("[3/4] Extracting features over $n_windows windows...")

    features = zeros(Float32, n_windows, 4)
    window_labels = zeros(Int, n_windows)
    window_snrs = zeros(Float32, n_windows)

    for i in 1:n_windows
        start_idx = (i - 1) * step_size + 1
        end_idx = start_idx + window_size - 1
        window = @view A_obs[start_idx:end_idx]

        features[i, :] .= extract_features(
            window,
            fs;
            low_band = low_band,
            high_band = high_band,
            feature_set = feature_set,
        )

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

    # 7. Persist
    println("\n[4/4] Saving processed data to CSV...")
    out_dir = pipeline_paths(config_file).inputs
    mkpath(out_dir)

    feat_path = joinpath(out_dir, "$(output_prefix)_features.csv")
    lab_path = joinpath(out_dir, "$(output_prefix)_labels.csv")
    psd_path = joinpath(out_dir, "$(output_prefix)_psd.csv")
    # Never overwrite an input: an output prefix equal to the stem of the
    # telemetry file would land the window labels on the point-wise labels.
    inputs = filter(
        !isempty,
        [h5_path, has_labels ? resolvepath(parsed_args["label-file"]) : ""],
    )
    for out in (feat_path, lab_path, psd_path), inp in inputs
        abspath(out) == abspath(inp) && throw(
            ArgumentError(
                "output $out coincides with input $inp; choose another output prefix.",
            ),
        )
    end
    CSV.write(feat_path, DataFrame(features, feature_names(feature_set)))
    println("  - Features saved to: $feat_path")
    if psd_table !== nothing
        CSV.write(psd_path, psd_table)
        println("  - Whitening PSD saved to: $psd_path")
    end
    sidecar = replace(feat_path, r"\.csv$" => ".toml")
    open(sidecar, "w") do io
        TOML.print(
            io,
            Dict(
                "features" => Dict(
                    "source" => rootrelative(h5_path),
                    "tdi_group" => tdi_group,
                    "window_size" => window_size,
                    "step_size" => step_size,
                    "sample_rate" => fs,
                    "feature_set" => String(feature_set),
                    "feature_names" => String.(feature_names(feature_set)),
                    "psd" => psd_mode,
                    "psd_description" => psd_description,
                    "low_band_hz" => collect(low_band),
                    "high_band_hz" => collect(high_band),
                    "highpass_cutoff_hz" => highpass_cutoff,
                    "highpass_order" => highpass_order,
                    "n_windows" => n_windows,
                ),
            ),
        )
    end
    println("  - Geometry sidecar saved to: $sidecar")

    if has_labels
        CSV.write(lab_path, DataFrame(Label = window_labels, SNR = window_snrs))
        println("  - Labels saved to: $lab_path")
    end

    println("\n[SUCCESS] Pre-processing complete.")
end

main()
