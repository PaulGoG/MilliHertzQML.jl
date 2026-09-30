# ext/MilliHertzQMLCurvatureDistinguishabilityExt.jl — constellation response
# of the simulator on CurvatureDistinguishability.jl: the antenna patterns of
# the noise-orthogonal A and E channels on the analytic LISA orbits, the
# orbital Doppler phase and the finite-arm transfer roll-off, applied to the
# waveforms this package generates. Only the response comes from that package; the
# inspiral–merger–ringdown amplitude and phase stay in src/waveforms.jl and
# the channel noise follows the sensitivity model of src/simulation.jl.
module MilliHertzQMLCurvatureDistinguishabilityExt

using CurvatureDistinguishability: waveform_params
using CurvatureDistinguishability.Detector: channel_patterns, doppler_phase
using CurvatureDistinguishability.Physics: transfer_frequency
using MilliHertzQML: AbstractDetectorResponse
using MilliHertzQML.MilliHertzBase: L_ARM
import MilliHertzQML:
    channel_count, lisa_response, source_frame, project_spectrum, project_series

"""
    LisaResponse

Constellation response on the analytic equal-arm orbits of Rubbo, Cornish
& Poujade (2004) as implemented by CurvatureDistinguishability: arm length
[m], and the orbital and cartwheel phases of the constellation at the
record start [rad]. Built by [`MilliHertzQML.lisa_response`](@ref).
"""
struct LisaResponse <: AbstractDetectorResponse
    arm_length::Float64
    orbit_phase::Float64
    constellation_phase::Float64
end

function lisa_response(; orbit_phase::Real = 0.0, constellation_phase::Real = 0.0)
    return LisaResponse(L_ARM, Float64(orbit_phase), Float64(constellation_phase))
end

channel_count(::LisaResponse) = 2

function source_frame(
    response::LisaResponse,
    longitude::Real,
    latitude::Real,
    polarization::Real,
)
    parameters = waveform_params(;
        ecliptic_longitude = longitude,
        ecliptic_latitude = latitude,
        polarization = polarization,
        orbit_phase = response.orbit_phase,
        constellation_phase = response.constellation_phase,
        arm_length = response.arm_length,
    )
    return parameters.geometry
end

# Plus and cross amplitude factors of the dominant harmonic for the
# inclination ι between the line of sight and the orbital angular momentum.
@inline inclination_factors(inclination) = ((1 + cos(inclination)^2) / 2, cos(inclination))

# Finite-arm transfer roll-off of Robson, Cornish & Liu (2019) at frequency f.
@inline transfer_rolloff(f, f_star) = 1 / sqrt(1 + 0.6 * (f / f_star)^2)

function project_spectrum(
    response::LisaResponse,
    frame,
    freqs::AbstractVector{<:Real},
    H::AbstractVector{<:Complex},
    delays::AbstractVector{<:Real},
    t_c::Real,
    inclination::Real,
)
    length(freqs) == length(H) == length(delays) || throw(
        DimensionMismatch(
            "freqs, H, and delays must have equal length; got $(length(freqs)), " *
            "$(length(H)), $(length(delays)).",
        ),
    )
    a_plus, a_cross = inclination_factors(inclination)
    f_star = transfer_frequency(response.arm_length)
    H_A = zeros(ComplexF64, length(H))
    H_E = zeros(ComplexF64, length(H))
    @inbounds for k in eachindex(H)
        h = H[k]
        h == 0 && continue
        f = freqs[k]
        F_plus_A, F_cross_A, F_plus_E, F_cross_E, cos_alpha, sin_alpha =
            channel_patterns(t_c - delays[k], frame, response.orbit_phase)
        factor =
            transfer_rolloff(f, f_star) *
            cis(doppler_phase(f, cos_alpha, sin_alpha, frame)) *
            h
        H_A[k] = (a_plus * F_plus_A - im * a_cross * F_cross_A) * factor
        H_E[k] = (a_plus * F_plus_E - im * a_cross * F_cross_E) * factor
    end
    return H_A, H_E
end

function project_series(
    response::LisaResponse,
    frame,
    t::AbstractVector{<:Real},
    amplitude::AbstractVector{<:Real},
    phase::AbstractVector{<:Real},
    frequency::AbstractVector{<:Real},
    inclination::Real,
)
    n = length(t)
    (length(amplitude) == n && length(phase) == n && length(frequency) == n) || throw(
        DimensionMismatch(
            "t, amplitude, phase, and frequency must have equal length; got $n, " *
            "$(length(amplitude)), $(length(phase)), $(length(frequency)).",
        ),
    )
    a_plus, a_cross = inclination_factors(inclination)
    f_star = transfer_frequency(response.arm_length)
    h_A = Vector{Float64}(undef, n)
    h_E = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        F_plus_A, F_cross_A, F_plus_E, F_cross_E, cos_alpha, sin_alpha =
            channel_patterns(t[j], frame, response.orbit_phase)
        Φ = phase[j] + doppler_phase(frequency[j], cos_alpha, sin_alpha, frame)
        a = amplitude[j] * transfer_rolloff(frequency[j], f_star)
        h_plus = a * a_plus * cos(Φ)
        h_cross = a * a_cross * sin(Φ)
        h_A[j] = F_plus_A * h_plus + F_cross_A * h_cross
        h_E[j] = F_plus_E * h_plus + F_cross_E * h_cross
    end
    return h_A, h_E
end

end # module
