include(joinpath(@__DIR__, "common.jl"))

using CSV, DataFrames, Statistics, MilliHertzQML, ArgParse, TOML, Printf

function parse_commandline()
    s = ArgParseSettings(
        description = "Point-wise MBHB labels of an LDC product from its truth stream",
    )
    @add_arg_table s begin
        "--config"
        help = "Path to the configuration file"
        default = joinpath(dirname(@__DIR__), "config.toml")
        "--h5-file"
        help = "LDC training product holding the truth stream and the source catalog"
        default = nothing
        "--truth-csv"
        help = "CSV with columns t, X, Y, Z of the signal-only TDI (blind-set truth); used instead of --h5-file"
        default = nothing
        "--output-prefix"
        help = "Prefix of the output CSV files (default from [ldc] output_prefix)"
        default = nothing
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    # 1. Load and validate the TOML configuration
    config_file = load_config(parsed_args["config"])
    ldc_cfg = get(config_file, "ldc", Dict{String,Any}())
    truth_group = cfgget(ldc_cfg, "truth_group", "sky/mbhb/tdi"; type = String)
    catalog_group = cfgget(ldc_cfg, "catalog_group", "sky/mbhb/cat"; type = String)
    psd_model = cfgget(ldc_cfg, "psd_model", "sangria"; type = String)
    tdi2 = cfgget(ldc_cfg, "tdi2", false; type = Bool)
    observation_years = cfgget(ldc_cfg, "observation_years", 0.0; type = Float64, min = 0.0)
    label_span = cfgget(
        ldc_cfg,
        "label_span",
        "fixed";
        type = String,
        choices = ("fixed", "detectable"),
    )
    before = cfgget(ldc_cfg, "label_before_sec", 4 * 86400.0; type = Float64, min = 0.0)
    after = cfgget(ldc_cfg, "label_after_sec", 27 * 60.0; type = Float64, min = 0.0)
    threshold = cfgget(ldc_cfg, "label_snr_threshold", 5.0; type = Float64, min = 1e-6)
    window_size = cfgget(ldc_cfg, "label_window_size", 1000; type = Int, min = 2)
    step = cfgget(ldc_cfg, "label_step", 10; type = Int, min = 1)
    min_separation_sec =
        cfgget(ldc_cfg, "peak_min_separation_sec", 86400.0; type = Float64, min = 0.0)
    merger_threshold =
        cfgget(ldc_cfg, "merger_snr_threshold", 8.0; type = Float64, min = 1e-6)
    precursor_ratio =
        cfgget(ldc_cfg, "precursor_ratio", 0.1; type = Float64, min = 0.0, max = 1.0)
    output_prefix = override(
        parsed_args["output-prefix"],
        cfgget(ldc_cfg, "output_prefix", "ldc"; type = String),
    )
    h5_path = parsed_args["h5-file"]
    csv_path = parsed_args["truth-csv"]
    if h5_path === nothing && csv_path === nothing
        h5_path = cfgget(ldc_cfg, "h5_file", ""; type = String)
        isempty(h5_path) && throw(
            ArgumentError(
                "no truth source: pass --h5-file or --truth-csv, or set [ldc] h5_file.",
            ),
        )
    end
    (h5_path === nothing || csv_path === nothing) ||
        throw(ArgumentError("--h5-file and --truth-csv are mutually exclusive."))

    println(
        "================================================================================",
    )
    println("  LDC TRUTH-STREAM LABELS")
    println(
        "================================================================================",
    )

    # 2. Truth stream and catalog
    println("[1/4] Reading the signal-only TDI...")
    local truth, catalog
    if h5_path !== nothing
        h5_path = resolvepath(h5_path)
        truth = read_tdi(h5_path; group = truth_group)
        catalog = read_catalog(h5_path; group = catalog_group)
        source = h5_path
    else
        csv_path = resolvepath(csv_path)
        df = CSV.read(csv_path, DataFrame)
        for c in (:t, :X, :Y, :Z)
            c in propertynames(df) || throw(ArgumentError("$csv_path lacks the column $c."))
        end
        dt = df.t[2] - df.t[1]
        truth = (
            t = Float64.(df.t),
            X = Float64.(df.X),
            Y = Float64.(df.Y),
            Z = Float64.(df.Z),
            dt = dt,
        )
        catalog = nothing
        source = csv_path
    end
    n = length(truth.t)
    fs = 1 / truth.dt
    t0 = truth.t[1]
    A, _, _ = tdi_to_aet(truth.X, truth.Y, truth.Z)
    println("Samples: $n | FS: $fs Hz | t0: $t0 s | max |A|: $(maximum(abs, A))")

    # 3. Per-window matched-filter SNR against the analytic TDI PSD
    println("[2/4] Windowed matched-filter SNR (window $window_size, step $step)...")
    psd =
        f -> ldc_tdi_psd(
            f;
            channel = :A,
            model = psd_model,
            tdi2 = tdi2,
            observation_years = observation_years,
        )
    starts, snr = windowed_snr(A, fs; window_size = window_size, step = step, psd = psd)

    # 4. Merger samples: from the catalog when available, else SNR peaks
    println("[3/4] Locating mergers...")
    local merger_indices
    if catalog !== nothing
        "CoalescenceTime" in names(catalog) || throw(
            ArgumentError("the catalog $catalog_group lacks the column CoalescenceTime."),
        )
        merger_indices =
            [clamp(round(Int, (tc - t0) * fs) + 1, 1, n) for tc in catalog.CoalescenceTime]
    else
        peaks = snr_peaks(
            starts,
            snr;
            threshold = merger_threshold,
            min_separation = round(Int, min_separation_sec * fs),
            precursor_window = round(Int, before * fs),
            precursor_ratio = precursor_ratio,
        )
        merger_indices = Int[]
        for p in peaks
            s = starts[p]
            push!(merger_indices, s - 1 + argmax(abs.(view(A, s:(s+window_size-1)))))
        end
    end
    order = sortperm(merger_indices)
    merger_indices = merger_indices[order]
    catalog !== nothing && (catalog = catalog[order, :])
    isempty(merger_indices) && @warn "no merger found; the label column is all zeros."

    # Label spans: the paper's fixed window around each merger, or the union
    # of windows in which the source is detectable (as the simulator labels)
    spans = if label_span == "fixed"
        fixed_spans(merger_indices, fs, n; before = before, after = after)
    else
        detectable_spans(starts, snr, window_size; threshold = threshold)
    end
    labels = span_labels(n, spans)

    # Per-sample SNR: the peak windowed SNR of the event the sample belongs to
    snr_column = zeros(Float32, n)
    peak_snr = Float64[]
    for span in spans
        first_w = max(1, cld(first(span) - window_size + 1 - 1, step) + 1)
        last_w = min(length(starts), div(last(span) - 1, step) + 1)
        ρ = first_w <= last_w ? maximum(view(snr, first_w:last_w)) : 0.0
        push!(peak_snr, ρ)
        snr_column[span] .= max.(snr_column[span], Float32(ρ))
    end

    # 5. Persist
    println("[4/4] Saving labels and the event table...")
    out_dir = pipeline_paths(config_file).inputs
    mkpath(out_dir)
    label_path = joinpath(out_dir, "$(output_prefix)_labels.csv")
    events_path = joinpath(out_dir, "$(output_prefix)_events.csv")
    for out in (label_path, events_path)
        abspath(out) == abspath(source) && throw(
            ArgumentError("output $out coincides with the input; choose another prefix."),
        )
    end
    CSV.write(label_path, DataFrame(Label = labels, SNR = snr_column))

    events = DataFrame(
        event = 1:length(merger_indices),
        merger_index = merger_indices,
        merger_time_s = t0 .+ (merger_indices .- 1) ./ fs,
        merger_window_snr = [
            snr[clamp(div(i - 1, step) + 1, 1, length(snr))] for i in merger_indices
        ],
    )
    if catalog !== nothing
        for c in ("CoalescenceTime", "Mass1", "Mass2", "Redshift", "Distance")
            c in names(catalog) && (events[!, c] = catalog[!, c])
        end
    end
    if label_span == "fixed"
        events.label_start_index = first.(spans)
        events.label_end_index = last.(spans)
        events.label_peak_snr = peak_snr
    end
    CSV.write(events_path, events)
    spans_df = DataFrame(
        span = 1:length(spans),
        start_index = first.(spans),
        end_index = last.(spans),
        peak_snr = peak_snr,
    )
    CSV.write(joinpath(out_dir, "$(output_prefix)_spans.csv"), spans_df)
    open(joinpath(out_dir, "$(output_prefix)_labels.toml"), "w") do io
        TOML.print(
            io,
            Dict(
                "labels" => Dict(
                    "source" => rootrelative(source),
                    "truth_group" => truth_group,
                    "psd_model" => psd_model,
                    "tdi2" => tdi2,
                    "observation_years" => observation_years,
                    "label_span" => label_span,
                    "label_before_sec" => before,
                    "label_after_sec" => after,
                    "label_snr_threshold" => threshold,
                    "merger_snr_threshold" => merger_threshold,
                    "precursor_ratio" => precursor_ratio,
                    "peak_min_separation_sec" => min_separation_sec,
                    "label_window_size" => window_size,
                    "label_step" => step,
                    "n_samples" => n,
                    "n_events" => length(merger_indices),
                    "n_label_runs" => length(contiguous_runs(labels .== 1)),
                    "positive_fraction" => count(==(1), labels) / n,
                ),
                "hardware" => hardware_fingerprint(),
            ),
        )
    end
    @printf(
        "  %d mergers, %d label runs, %.2f %% positive samples\n",
        length(merger_indices),
        length(contiguous_runs(labels .== 1)),
        100 * count(==(1), labels) / n
    )
    println("  - Labels saved to: $label_path")
    println("  - Events saved to: $events_path")
    println("\n[SUCCESS] Labeling complete.")
end

main()
