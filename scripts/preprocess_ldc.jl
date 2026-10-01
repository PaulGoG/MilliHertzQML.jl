# scripts/preprocess_ldc.jl — dispatcher of the pre-processing stage
# (MilliHertzQML.preprocess_record): an HDF5 TDI product to per-window
# features. Every parameter comes from the configuration; the command line
# names the configuration file and the external inputs only.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        description = "Pre-process an HDF5 TDI product into per-window features",
    )
    @add_arg_table! s begin
        "config"
        help = "Path to the configuration file"
        required = false
        default = DEFAULT_CONFIG
        "--h5-file"
        help = "Path to the HDF5 TDI product (simulator output or LDC file); default from [preprocessing] h5_file"
        default = nothing
        "--tdi-group"
        help = "HDF5 group or compound dataset holding t, X, Y, Z; default from [preprocessing] tdi_group"
        default = nothing
        "--label-file"
        help = "Path to the associated point-wise label CSV (optional, used for training data)"
        default = ""
        "--output-prefix"
        help = "Prefix of the output CSV files; default from [preprocessing] output_prefix"
        default = nothing
        "--force"
        help = "Recompute the product even when one with the same parameters exists"
        action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    product = preprocess_record(
        config;
        h5_file = args["h5-file"],
        tdi_group = args["tdi-group"],
        label_file = args["label-file"],
        output_prefix = args["output-prefix"],
        force = args["force"],
    )
    g = product.geometry
    println(product.skipped ? "Pre-processed product reused." : "Pre-processing complete.")
    println(
        "  $(product.n_windows) windows of $(g.window_size) samples, step $(g.step_size), " *
        "at $(g.sample_rate) Hz",
    )
    println("  Features: $(product.features_path)")
    println("  Sidecar:  $(product.sidecar_path)")
    product.labels_path !== nothing && println("  Labels:   $(product.labels_path)")
    product.psd_path !== nothing && println("  PSD:      $(product.psd_path)")
    report_timing()
    return nothing
end

with_pipeline_root(main, config_root(config_argument()))
