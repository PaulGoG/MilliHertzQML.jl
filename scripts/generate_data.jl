# scripts/generate_data.jl — command-line dispatcher of the telemetry
# generation stage (`generate_telemetry`). The TOML configuration is the
# single source of every physical and numerical parameter; the command line
# adds only the run identifier and an output-path override.

ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "common.jl"))

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using Plots: Plots, default, plot, plot!, savefig

default(
    dpi = 600,
    frame = :box,
    fontfamily = "Computer Modern",
    grid = true,
    gridalpha = 0.2,
    minorgrid = false,
    margin = 5Plots.mm,
)

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
    trace_figure(path, n_total, fs, strain, labels)

Down-sampled trace of the simulated strain against mission time [days],
with the positive-label spans shaded, saved at `path`.
"""
function trace_figure(
    path::AbstractString,
    n_total::Integer,
    fs::Real,
    strain::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
)
    length(strain) == n_total == length(labels) ||
        throw(DimensionMismatch("strain and labels must hold n_total = $n_total samples."))
    ds = max(1, round(Int, n_total / 5000))
    t_days = ((0:(n_total-1)) ./ fs) ./ (24 * 3600)
    p = plot(
        t_days[1:ds:end],
        strain[1:ds:end];
        xlabel = "Mission time [days]",
        ylabel = "Strain",
        lw = 0.5,
        color = :gray,
        label = "Data",
    )
    plot!(
        p,
        t_days[1:ds:end],
        labels[1:ds:end] .* maximum(abs, strain) / 2;
        st = :step,
        color = :red,
        alpha = 0.5,
        fill = (0, 0.5, :red),
        label = "MBHB label span",
    )
    mkpath(dirname(path))
    savefig(p, path)
    return path
end

function main()
    args = parse_commandline()
    config = load_config(args["config"])
    run_id = something(args["run-id"], new_run_id())
    result = generate_telemetry(config; run_id = run_id, output = args["output"])

    plot_dir = joinpath(pipeline_paths(config).plots, "run_$(result.run_id)")
    figure = trace_figure(
        joinpath(plot_dir, "simulated_continuous_trace.png"),
        result.n_total,
        result.fs,
        result.strain,
        result.labels,
    )
    @info "trace figure saved" path = figure

    report_timing()
    return nothing
end

main()
