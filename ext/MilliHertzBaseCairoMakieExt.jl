# ext/MilliHertzBaseCairoMakieExt.jl — figures of the gravitational-wave
# layer, loaded together with CairoMakie: classifier output and simulated
# strain against mission time. Theme, legend and labelled bands come from
# the figures of the domain-general layer.
module MilliHertzBaseCairoMakieExt

using CairoMakie.Makie: Figure, Axis, with_theme, lines!, hlines!
using CairoMakie.Makie: linkxaxes!, hidexdecorations!, rowgap!, xlims!, ylims!
using CairoMakie.Makie: LaTeXStrings
using MilliHertzQML.StreamingInference: FIGURE_COLORS, figure_size, figure_theme
using MilliHertzQML.StreamingInference: decimation, top_legend!, label_bands!
import MilliHertzQML.MilliHertzBase: figure_mission_trace, figure_telemetry_trace

function figure_mission_trace(
    days::AbstractVector{<:Real},
    probabilities::AbstractVector{<:Real},
    threshold::Real;
    labels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    max_points::Integer = 5000,
)
    n = length(probabilities)
    n == length(days) ||
        throw(DimensionMismatch("$(length(days)) times for $n probabilities."))
    n >= 1 || throw(ArgumentError("the trace is empty."))
    idx = decimation(n, max_points)
    return with_theme(figure_theme(; size = figure_size(1))) do
        figure = Figure()
        axis =
            Axis(figure[1, 1]; xlabel = "Mission time [days]", ylabel = "MBHB probability")
        labels === nothing || label_bands!(axis, days, labels)
        lines!(
            axis,
            days[idx],
            probabilities[idx];
            color = FIGURE_COLORS.data,
            linewidth = 2,
            label = "Classifier output",
        )
        hlines!(
            axis,
            [threshold];
            color = FIGURE_COLORS.threshold,
            linestyle = :dash,
            linewidth = 1.5,
            label = "Threshold $(round(threshold; digits = 3))",
        )
        xlims!(axis, days[1], days[end] == days[1] ? days[1] + 1 : days[end])
        ylims!(axis, 0, 1)
        top_legend!(figure, axis; nbanks = 2)
        figure
    end
end

function figure_telemetry_trace(
    t_days::AbstractVector{<:Real},
    strain::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer};
    max_points::Integer = 5000,
    whitened::Union{Nothing,AbstractVector{<:Real}} = nothing,
)
    n = length(strain)
    n == length(t_days) == length(labels) ||
        throw(DimensionMismatch("t_days, strain, and labels differ in length."))
    n >= 2 || throw(ArgumentError("the record must hold at least 2 samples."))
    whitened === nothing ||
        length(whitened) == n ||
        throw(DimensionMismatch("the whitened record differs in length."))
    idx = decimation(n, max_points)
    amplitude = maximum(abs, strain)
    exponent = amplitude > 0 ? floor(Int, log10(amplitude)) : 0
    exponent in (0, 1) && (exponent = 0)  # 10⁰ and 10¹ multipliers fold into the tick values
    scale = 10.0^exponent
    two_panels = whitened !== nothing
    return with_theme(
        figure_theme(; size = two_panels ? figure_size(2) : figure_size(1)),
    ) do
        figure = Figure()
        ax_strain = Axis(
            figure[1, 1];
            ylabel = exponent == 0 ? "Strain" :
                     LaTeXStrings.latexstring("\\mathrm{Strain}\\ [10^{$exponent}]"),
        )
        label_bands!(ax_strain, t_days, labels)
        lines!(
            ax_strain,
            t_days[idx],
            strain[idx] ./ scale;
            color = FIGURE_COLORS.data,
            linewidth = 2,
            label = "Simulated record",
        )
        xlims!(ax_strain, t_days[1], t_days[end])
        if two_panels
            ax_white =
                Axis(figure[2, 1]; xlabel = "Mission time [days]", ylabel = "Whitened")
            label_bands!(ax_white, t_days, labels)
            lines!(
                ax_white,
                t_days[idx],
                whitened[idx];
                color = FIGURE_COLORS.data,
                linewidth = 2,
            )
            linkxaxes!(ax_strain, ax_white)
            hidexdecorations!(ax_strain; grid = false, ticks = false)
            rowgap!(figure.layout, 10)
        else
            ax_strain.xlabel = "Mission time [days]"
        end
        top_legend!(figure, ax_strain)
        figure
    end
end

end # module
