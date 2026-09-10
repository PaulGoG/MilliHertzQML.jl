# scripts/export_telemetry_payload.jl — dispatcher of the payload-export
# stage (`export_telemetry_payload`): the A channel of an HDF5 TDI product
# as the amplitude CSV and scenario fragment ingested by the telemetry
# producer. Every parameter comes from the configuration; the command line
# names the configuration file and the external inputs only.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args

"""
    parse_commandline() -> Dict{String, Any}

Command line of the payload-export stage: an optional positional `config`
(default `config.toml` at the package root), `--h5-file`, `--tdi-group`,
`--catalog`, and `--output-prefix`.
"""
function parse_commandline()
    s = ArgParseSettings(;
        description = "Export the A channel of an HDF5 TDI product as a telemetry payload.",
    )
    @add_arg_table! s begin
        "config"
        help = "TOML configuration file"
        required = false
        default = joinpath(project_root(), "config.toml")
        "--h5-file"
        help = "HDF5 TDI product (simulator output or LDC file); default from [preprocessing] h5_file"
        default = nothing
        "--tdi-group"
        help = "HDF5 group or compound dataset holding t, X, Y, Z; default from [preprocessing] tdi_group"
        default = nothing
        "--catalog"
        help = "event catalog CSV (simulator catalog or LDC event table) providing the markers and label spans"
        default = nothing
        "--output-prefix"
        help = "prefix of the payload CSV and scenario TOML; default from [telemetry] output_prefix"
        default = nothing
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    result = export_telemetry_payload(
        config;
        h5_file = args["h5-file"],
        tdi_group = args["tdi-group"],
        catalog = args["catalog"],
        output_prefix = args["output-prefix"],
    )
    println("Payload export complete.")
    println(
        "  $(result.n_rows) rows at $(result.sample_rate) Hz from $(result.start_sim_time)",
    )
    println("  Payload:  $(result.payload_path)")
    println("  Scenario: $(result.scenario_path)")
    println("  Markers:  $(length(result.markers))")
    report_timing()
    return nothing
end

main()
