# scripts/label_ldc.jl — dispatcher of the labeling stage
# (MilliHertzQML.label_truth_stream): point-wise MBHB labels of an LDC
# product from its truth stream. Every parameter comes from the
# configuration; the command line names the configuration file and the
# external inputs only.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse
using DataFrames: nrow

function parse_commandline()
    s = ArgParseSettings(
        description = "Point-wise MBHB labels of an LDC product from its truth stream",
    )
    @add_arg_table! s begin
        "config"
        help = "Path to the configuration file"
        required = false
        default = joinpath(project_root(), "configs", "default.toml")
        "--h5-file"
        help = "LDC training product holding the truth stream and the source catalog; default from [ldc] h5_file"
        default = nothing
        "--truth-csv"
        help = "CSV with columns t, X, Y, Z of the signal-only TDI (blind-set truth); used instead of --h5-file"
        default = nothing
        "--output-prefix"
        help = "Prefix of the output CSV files; default from [ldc] output_prefix"
        default = nothing
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    product = label_truth_stream(
        config;
        h5_file = args["h5-file"],
        truth_csv = args["truth-csv"],
        output_prefix = args["output-prefix"],
    )
    println("Labeling complete.")
    println(
        "  $(nrow(product.events)) mergers, $(product.n_label_runs) label runs, " *
        "$(round(100 * product.positive_fraction; digits = 2)) % positive samples",
    )
    println("  Labels: $(product.label_path)")
    println("  Events: $(product.events_path)")
    println("  Spans:  $(product.spans_path)")
    report_timing()
    return nothing
end

main()
