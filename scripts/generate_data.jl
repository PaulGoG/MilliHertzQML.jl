# scripts/generate_data.jl — command-line dispatcher of the telemetry
# generation stage (`generate_telemetry`). The TOML configuration is the
# single source of every physical and numerical parameter; the command line
# adds only the run identifier and an output-path override. The trace figure
# comes from the CairoMakie extension.

include(joinpath(@__DIR__, "common.jl"))

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using CairoMakie: CairoMakie

"""
    parse_commandline() -> Dict{String, Any}

Command line of the generation stage: an optional positional `config`
(default `config.toml` at the package root), `--run-id`, and `--output`.
"""
function parse_commandline()
    s = ArgParseSettings(;
        description = "Continuous LISA telemetry simulator (millihertz band).",
    )
    @add_arg_table! s begin
        "config"
        help = "TOML configuration file"
        required = false
        default = joinpath(project_root(), "config.toml")
        "--run-id"
        help = "run identifier of the products and figures (default: a fresh identifier)"
        default = nothing
        "--output"
        help = "output HDF5 path, overriding [generation] output"
        default = nothing
    end
    return parse_args(s)
end

"""
    trace_figure(stem, config, result)

Trace of the simulated strain against mission time with the labeled spans
and, in a second panel, the record high-passed and whitened by the model
sensitivity as the pre-processor sees it; exported at `stem` as PDF and
PNG with a provenance sidecar.
"""
function trace_figure(stem::AbstractString, config::AbstractDict, result::NamedTuple)
    gen = generation_settings(config)
    pre = preprocessing_settings(config)
    fs = result.fs
    t_days = ((0:(result.n_total-1)) ./ fs) ./ 86400
    whitened = whiten_record(
        highpass_record(
            result.strain,
            fs;
            cutoff = pre.highpass_cutoff_hz,
            order = pre.highpass_order,
        ),
        fs;
        psd = f -> lisa_noise_psd(f; observation_years = gen.observation_years),
    )
    figure = figure_telemetry_trace(
        t_days,
        result.strain,
        Int.(result.labels);
        whitened = whitened,
    )
    return save_figure(figure, stem; run_id = result.run_id)
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    run_id = something(args["run-id"], new_run_id())
    result = generate_telemetry(config; run_id = run_id, output = args["output"])

    plot_dir = joinpath(pipeline_paths(config).plots, "run_$(result.run_id)")
    written = trace_figure(joinpath(plot_dir, "simulated_continuous_trace"), config, result)
    @info "trace figure saved" files = written

    report_timing()
    return nothing
end

main()
