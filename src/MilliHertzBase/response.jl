# Detector response of the simulator: the sky-averaged
# single strain that the sensitivity model refers to, and the interface of
# the constellation response (A and E channels on the LISA orbits) that the
# CurvatureDistinguishability extension implements.

"""
    AbstractDetectorResponse

How a simulated source reaches the recorded channels. The package provides
[`SkyAveragedResponse`](@ref); the CurvatureDistinguishability extension
adds the constellation response built by [`lisa_response`](@ref).
"""
abstract type AbstractDetectorResponse end

"""
    SkyAveragedResponse()

One strain channel referred to the sky- and polarisation-averaged
sensitivity ``S_n(f)`` of [`lisa_noise_psd`](@ref): no antenna pattern, no
orbital modulation, no transfer roll-off. Sources are placed at a
matched-filter SNR rather than at a distance.
"""
struct SkyAveragedResponse <: AbstractDetectorResponse end

"""
    channel_count(response) -> Int

Number of recorded channels: 1 for [`SkyAveragedResponse`](@ref), 2 (A
and E) for the constellation response.
"""
channel_count(::SkyAveragedResponse) = 1

"""
    GIGAPARSEC_SEC

One gigaparsec in light-seconds.
"""
const GIGAPARSEC_SEC = 3.0856775814913673e25 / C_LIGHT

"""
    sky_averaged_response(f)

``R(f) = \\frac{3}{10} \\left[1 + \\frac{6}{10} (f/f_*)^2\\right]^{-1}``
(Robson, Cornish & Liu 2019, Eq. 9): the sky- and polarisation-averaged
squared antenna pattern of one 60° Michelson channel, ``3/10``, times the
finite-arm transfer roll-off. The channel noise PSD is ``R(f) S_n(f)`` for
the sensitivity ``S_n`` of [`lisa_noise_psd`](@ref), so that a face-on
source at a sky position of average pattern has the sensitivity-curve SNR
in one channel.
"""
sky_averaged_response(f::Real) = 0.3 / (1 + 0.6 * (f / F_STAR)^2)

"""
    channel_noise_psd(response, psd) -> callable

One-sided noise PSD of each recorded channel for the sensitivity `psd(f)`
[Hz⁻¹]: the sensitivity itself for [`SkyAveragedResponse`](@ref), and
``R(f)\\,\\mathrm{psd}(f)`` ([`sky_averaged_response`](@ref)) for every
constellation response, whose channel antenna patterns average to 3/10.
"""
channel_noise_psd(::SkyAveragedResponse, psd) = psd

channel_noise_psd(::AbstractDetectorResponse, psd) = f -> sky_averaged_response(f) * psd(f)

"""
    draw_extrinsic(rng) -> NamedTuple

Isotropic extrinsic parameters of one source, in draw order: ecliptic
`longitude` uniform on ``[0, 2\\pi)``, `latitude` with a uniform sine on
``[-1, 1]``, `inclination` with a uniform cosine on ``[-1, 1]``, and
`polarization` uniform on ``[0, \\pi)`` [rad].
"""
function draw_extrinsic(rng::AbstractRNG)
    longitude = 2π * rand(rng)
    latitude = asin(2 * rand(rng) - 1)
    inclination = acos(2 * rand(rng) - 1)
    polarization = π * rand(rng)
    return (
        longitude = longitude,
        latitude = latitude,
        inclination = inclination,
        polarization = polarization,
    )
end

response_unavailable(response) = throw(
    ArgumentError(
        "no channel projection is implemented for $(typeof(response)); the constellation " *
        "response needs the CurvatureDistinguishability extension (add the package to the " *
        "environment and load it with `using CurvatureDistinguishability`).",
    ),
)

"""
    lisa_response(; orbit_phase = 0.0, constellation_phase = 0.0) -> AbstractDetectorResponse

Constellation response on the analytic LISA orbits — antenna patterns of
the noise-orthogonal A and E channels, orbital Doppler phase, finite-arm
transfer roll-off — with the orbital and cartwheel phases of the
constellation at the record start [rad]. Defined by the
CurvatureDistinguishability extension; [`detector_response`](@ref) names
the package when the extension is absent.
"""
function lisa_response end

"""
    source_frame(response, longitude, latitude, polarization)

Response constants of one source at ecliptic `longitude` and `latitude`
with polarisation angle `polarization` [rad], consumed by
[`project_spectrum`](@ref) and [`project_series`](@ref). Implemented per
response type by the extension.
"""
source_frame(
    response::AbstractDetectorResponse,
    longitude::Real,
    latitude::Real,
    polarization::Real,
) = response_unavailable(response)

"""
    project_spectrum(response, frame, freqs, H, delays, t_c, inclination) -> (H_A, H_E)

Channel spectra of the barycentric spectrum `H` on the grid `freqs` [Hz]
of a source in `frame` ([`source_frame`](@ref)) with orbital
`inclination` [rad]: frequency `freqs[k]` arrives `delays[k]` seconds
before the coalescence at mission time `t_c` [s] and meets the antenna
patterns, Doppler phase and transfer roll-off of that instant, with
``\\tilde h_+ = \\tfrac{1}{2}(1 + \\cos^2\\iota) H`` and
``\\tilde h_\\times = -i \\cos\\iota\\, H`` (the ``e^{-2\\pi i f t}`` transform
of a chirp whose phase increases with time). Implemented per response type
by the extension.
"""
project_spectrum(
    response::AbstractDetectorResponse,
    frame,
    freqs::AbstractVector{<:Real},
    H::AbstractVector{<:Complex},
    delays::AbstractVector{<:Real},
    t_c::Real,
    inclination::Real,
) = response_unavailable(response)

"""
    project_series(response, frame, t, amplitude, phase, frequency, inclination) -> (h_A, h_E)

Channel series of a quasi-monochromatic source in `frame`
([`source_frame`](@ref)): at mission time `t[j]` [s] the source has
amplitude `amplitude[j]`, phase `phase[j]` [rad] and instantaneous
frequency `frequency[j]` [Hz], and the channels receive
``h_+ = a\\,\\tfrac{1}{2}(1 + \\cos^2\\iota) \\cos(\\Phi + \\Delta_D)`` and
``h_\\times = a \\cos\\iota \\sin(\\Phi + \\Delta_D)`` through their antenna
patterns, with the orbital Doppler phase ``\\Delta_D`` and the transfer
roll-off at `frequency[j]`. Implemented per response type by the
extension.
"""
project_series(
    response::AbstractDetectorResponse,
    frame,
    t::AbstractVector{<:Real},
    amplitude::AbstractVector{<:Real},
    phase::AbstractVector{<:Real},
    frequency::AbstractVector{<:Real},
    inclination::Real,
) = response_unavailable(response)

"""
    detector_response(settings) -> AbstractDetectorResponse

The response of the `[generation]` settings ([`generation_settings`](@ref)):
[`SkyAveragedResponse`](@ref) for `response = "sky_averaged"`, and for
`"lisa"` the constellation response of [`lisa_response`](@ref) with the
settings' `orbit_phase` and `constellation_phase`, taken from the
CurvatureDistinguishability extension; an `ArgumentError` names the
package when the extension is not loaded.
"""
function detector_response(settings::NamedTuple)
    settings.response == "sky_averaged" && return SkyAveragedResponse()
    ext = Base.get_extension(
        Base.moduleroot(@__MODULE__),
        :MilliHertzQMLCurvatureDistinguishabilityExt,
    )
    ext === nothing && throw(
        ArgumentError(
            "response = \"lisa\" needs the CurvatureDistinguishability extension: add the " *
            "package to the environment and load it (`using CurvatureDistinguishability`) " *
            "before generating.",
        ),
    )
    return ext.lisa_response(;
        orbit_phase = settings.orbit_phase,
        constellation_phase = settings.constellation_phase,
    )
end
