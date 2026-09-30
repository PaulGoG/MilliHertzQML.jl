# Non-spinning phenomenological inspiral–merger–ringdown
# waveform (IMRPhenomA; Ajith et al., Phys. Rev. D 77, 104017, 2008) generated
# in the frequency domain on the sampling grid of the injection segment.

"""
    T_SUN

Geometrised solar mass ``G M_\\odot / c^3`` [s].
"""
const T_SUN = 4.925490947e-6

"""
    PHENOMA_COEFFICIENTS

Coefficients ``(a, b, c)`` of the IMRPhenomA fits (Ajith et al. 2008,
Table I; the values of the LALSuite reference implementation). A
transition frequency is ``f_j = (a_j \\eta^2 + b_j \\eta + c_j) / (\\pi M)``
and a phase coefficient ``\\psi_k = (a_k \\eta^2 + b_k \\eta + c_k) / \\eta``,
with ``M`` the total mass in seconds and ``\\eta`` the symmetric mass ratio.
"""
const PHENOMA_COEFFICIENTS = (
    f_merg = (2.9740e-1, 4.4810e-2, 9.5560e-2),
    f_ring = (5.9411e-1, 8.9794e-2, 1.9111e-1),
    sigma = (5.0801e-1, 7.7515e-2, 2.2369e-2),
    f_cut = (8.4845e-1, 1.2848e-1, 2.7299e-1),
    psi0 = (1.7516e-1, 7.9483e-2, -7.2390e-2),
    psi2 = (-5.1571e1, -1.7595e1, 1.3253e1),
    psi3 = (6.5866e2, 1.7803e2, -1.5972e2),
    psi4 = (-3.9031e3, -7.7493e2, 8.8195e2),
    psi6 = (-2.4874e4, -1.4892e3, 4.4588e3),
    psi7 = (2.5196e4, 3.3970e2, -3.9573e3),
)

phenoma_fit(coefficients, η) = coefficients[1] * η^2 + coefficients[2] * η + coefficients[3]

"""
    phenoma_parameters(total_mass, mass_ratio) -> NamedTuple

Transition frequencies ``f_\\mathrm{merg}``, ``f_\\mathrm{ring}``,
``f_\\mathrm{cut}`` and Lorentzian width ``\\sigma`` [Hz], the phase
coefficients ``\\psi_k`` (``k = 0, 2, 3, 4, 6, 7``), the total mass in
seconds, and ``\\eta``, for a binary of detector-frame `total_mass`
[M⊙] and `mass_ratio` ``q = m_1/m_2 \\ge 1``.
"""
function phenoma_parameters(total_mass::Real, mass_ratio::Real)
    total_mass > 0 || throw(ArgumentError("total_mass = $total_mass; must be positive."))
    mass_ratio >= 1 || throw(ArgumentError("mass_ratio = $mass_ratio; must be at least 1."))
    M = total_mass * T_SUN
    η = mass_ratio / (1 + mass_ratio)^2
    c = PHENOMA_COEFFICIENTS
    return (
        M_sec = M,
        η = η,
        f_merg = phenoma_fit(c.f_merg, η) / (π * M),
        f_ring = phenoma_fit(c.f_ring, η) / (π * M),
        sigma = phenoma_fit(c.sigma, η) / (π * M),
        f_cut = phenoma_fit(c.f_cut, η) / (π * M),
        psi = (
            phenoma_fit(c.psi0, η) / η,
            phenoma_fit(c.psi2, η) / η,
            phenoma_fit(c.psi3, η) / η,
            phenoma_fit(c.psi4, η) / η,
            phenoma_fit(c.psi6, η) / η,
            phenoma_fit(c.psi7, η) / η,
        ),
    )
end

"""
    phenoma_amplitude(f, p)

Effective amplitude of IMRPhenomA at frequency `f` [Hz] for the parameters
`p` of [`phenoma_parameters`](@ref), up to the overall constant fixed
later by the matched-filter SNR: ``(f/f_\\mathrm{merg})^{-7/6}`` in the
inspiral, ``(f/f_\\mathrm{merg})^{-2/3}`` between ``f_\\mathrm{merg}`` and
``f_\\mathrm{ring}``, and a Lorentzian of width ``\\sigma`` centred on
``f_\\mathrm{ring}`` (scaled for continuity) up to ``f_\\mathrm{cut}``; zero
outside ``(0, f_\\mathrm{cut})``.
"""
function phenoma_amplitude(f::Real, p)
    (f <= 0 || f >= p.f_cut) && return 0.0
    if f < p.f_merg
        return (f / p.f_merg)^(-7 / 6)
    elseif f < p.f_ring
        return (f / p.f_merg)^(-2 / 3)
    else
        lorentzian = (p.sigma / (2π)) / ((f - p.f_ring)^2 + p.sigma^2 / 4)
        w = (π * p.sigma / 2) * (p.f_ring / p.f_merg)^(-2 / 3)
        return w * lorentzian
    end
end

"""
    phenoma_phase(f, p)

Frequency-domain phase polynomial of IMRPhenomA at `f` [Hz], without the
time-shift and constant terms:
``\\Psi(f) = \\sum_{k \\in \\{0,2,3,4,6,7\\}} \\psi_k (\\pi M f)^{(k-5)/3}``.
"""
function phenoma_phase(f::Real, p)
    x = π * p.M_sec * f
    exponents = (-5 / 3, -1.0, -2 / 3, -1 / 3, 1 / 3, 2 / 3)
    Ψ = 0.0
    for (ψ, e) in zip(p.psi, exponents)
        Ψ += ψ * x^e
    end
    return Ψ
end

"""
    phenoma_group_delay(f, p)

``g(f) = \\Psi'(f) / (2\\pi)`` [s] of [`phenoma_phase`](@ref). With the
spectrum ``\\tilde h(f) = A(f) \\exp[i(\\Psi(f) - 2\\pi f t_0)]`` and the
transform convention of `irfft`, the component at frequency `f` arrives
at ``t(f) = t_0 - g(f)``: ``g`` decreases with ``f`` through the inspiral,
so lower frequencies arrive earlier (a forward chirp), and the
merger–ringdown arrives at ``t_0 - g(f_\\mathrm{ring})``.
"""
function phenoma_group_delay(f::Real, p)
    x = π * p.M_sec * f
    exponents = (-5 / 3, -1.0, -2 / 3, -1 / 3, 1 / 3, 2 / 3)
    d = 0.0
    for (ψ, e) in zip(p.psi, exponents)
        d += ψ * e * x^(e - 1)
    end
    return d * p.M_sec / 2
end

"""
    phenoma_start_frequency(p, delay; f_upper = p.f_merg)

Frequency [Hz] whose group delay equals `delay` [s], i.e. the component
arriving `delay` seconds before the merger–ringdown when
[`phenoma_group_delay`](@ref) is referenced to ``g(f_\\mathrm{ring})``;
found by bisection on ``(0, f_\\mathrm{upper}]`` where ``g`` is monotonic.
Returns `f_upper` when even that frequency arrives earlier than `delay`
before the merger (the segment is longer than the inspiral above it).
"""
function phenoma_start_frequency(p, delay::Real; f_upper::Real = p.f_merg)
    g_ref = phenoma_group_delay(p.f_ring, p)
    target = delay + g_ref
    lo = 1e-7
    hi = f_upper
    phenoma_group_delay(hi, p) >= target && return hi
    phenoma_group_delay(lo, p) <= target && return lo
    for _ in 1:200
        mid = sqrt(lo * hi)
        if phenoma_group_delay(mid, p) > target
            lo = mid
        else
            hi = mid
        end
        hi / lo - 1 < 1e-9 && break
    end
    return sqrt(lo * hi)
end

"""
    cosine_rolloff(f, f_start, f_stop)

Smooth weight rising from 0 at `f_start` to 1 at `f_stop` (half-cosine);
0 below, 1 above. With `f_start > f_stop` the weight falls instead.
"""
function cosine_rolloff(f::Real, f_start::Real, f_stop::Real)
    if f_start <= f_stop
        f <= f_start && return 0.0
        f >= f_stop && return 1.0
        return 0.5 * (1 - cos(π * (f - f_start) / (f_stop - f_start)))
    else
        f >= f_start && return 0.0
        f <= f_stop && return 1.0
        return 0.5 * (1 - cos(π * (f_start - f) / (f_start - f_stop)))
    end
end

"""
    phenoma_spectrum(fs, duration_sec; total_mass, mass_ratio,
                     nyquist_taper = 0.9, rolloff_width = 0.1) -> NamedTuple

Frequency-domain IMRPhenomA strain of a non-spinning binary of
detector-frame `total_mass` [M⊙] and `mass_ratio` ``q \\ge 1`` on the
`rfft` grid of twice the segment of `duration_sec` [s] sampled at `fs`
[Hz], with the coalescence near the end of the segment:

- the spectrum is ``H(f) = A(f) \\exp[i(\\Psi(f) - 2\\pi f t_0)]`` with the
  unit inspiral normalisation of [`phenoma_amplitude`](@ref) and ``t_0``
  chosen so that the ringdown frequency arrives at the target merger time
  ([`phenoma_group_delay`](@ref)); inspiral content arriving before the
  segment start is rolled on with a half-cosine of relative width
  `rolloff_width` above [`phenoma_start_frequency`](@ref), so that the
  signal starts inside the segment instead of wrapping around;
- the spectrum is tapered to zero with a half-cosine ending at
  `nyquist_taper` times the Nyquist frequency (starting 0.1 Nyquist
  below it), so the sampled waveform cannot alias whatever the mass.

Returns `(; freqs, H, n, t_merger, t0, p)`: the grid frequencies [Hz], the
spectrum, the segment length in samples, the local merger time and time
shift [s], and the parameters `p` of [`phenoma_parameters`](@ref). The
spectrum carries no physical scale: [`phenoma_physical_amplitude`](@ref)
supplies it, and [`phenoma_series`](@ref) inverts it.
"""
function phenoma_spectrum(
    fs::Real,
    duration_sec::Real;
    total_mass::Real,
    mass_ratio::Real,
    nyquist_taper::Real = 0.9,
    rolloff_width::Real = 0.1,
)
    fs > 0 || throw(ArgumentError("fs = $fs; the sampling frequency must be positive."))
    duration_sec > 0 ||
        throw(ArgumentError("duration_sec = $duration_sec; must be positive."))
    0.2 < nyquist_taper <= 1 ||
        throw(ArgumentError("nyquist_taper = $nyquist_taper; must lie in (0.2, 1]."))
    0 < rolloff_width <= 1 ||
        throw(ArgumentError("rolloff_width = $rolloff_width; must lie in (0, 1]."))
    p = phenoma_parameters(total_mass, mass_ratio)
    n = max(4, round(Int, duration_sec * fs))
    n_gen = 2n
    T_seg = n / fs

    # Coalescence inside the segment, leaving room for the ringdown; the
    # time shift is referenced to the arrival of the ringdown frequency.
    t_pad = clamp(max(0.05 * T_seg, 20 / (π * p.sigma)), 0.0, 0.5 * T_seg)
    t_merger = T_seg - t_pad
    t0 = t_merger + phenoma_group_delay(p.f_ring, p)
    # Inspiral content arriving before the segment start is rolled off
    f_start = phenoma_start_frequency(p, t_merger)

    f_ny = fs / 2
    f_taper_stop = nyquist_taper * f_ny
    f_taper_start = max(f_taper_stop - 0.1 * f_ny, 0.0)

    freqs = rfftfreq(n_gen, fs)
    H = zeros(ComplexF64, length(freqs))
    for k in 2:length(freqs)
        f = freqs[k]
        a = phenoma_amplitude(f, p)
        a == 0 && continue
        a *= cosine_rolloff(f, f_start, f_start * (1 + rolloff_width))
        a *= cosine_rolloff(f, f_taper_stop, f_taper_start)
        a == 0 && continue
        H[k] = a * cis(phenoma_phase(f, p) - 2π * f * t0)
    end
    return (freqs = freqs, H = H, n = n, t_merger = t_merger, t0 = t0, p = p)
end

"""
    inverse_segment(H, n) -> Vector{Float64}

Inverse transform of a spectrum on the `rfft` grid of `2n` samples, cut to
its first `n` samples, with the first 5 % ramped by a half-Hann window to
remove residual ringing of the roll-on; in the unnormalised FFT
convention, so without physical scale.
"""
function inverse_segment(H::AbstractVector{<:Complex}, n::Integer)
    n >= 2 || throw(ArgumentError("n = $n; the segment must hold at least 2 samples."))
    length(H) == n + 1 || throw(
        DimensionMismatch(
            "the spectrum holds $(length(H)) bins; the rfft grid of 2n = $(2n) samples has $(n + 1).",
        ),
    )
    h_full = irfft(H, 2n)
    h = h_full[1:n]
    n_ramp = max(2, round(Int, 0.05 * n))
    for j in 1:n_ramp
        h[j] *= 0.5 * (1 - cos(π * (j - 1) / n_ramp))
    end
    return h
end

"""
    phenoma_series(H, n, fs) -> Vector{Float64}

Time-domain segment of a spectrum built by [`phenoma_spectrum`](@ref) (or
projected from one) on the `rfft` grid of `2n` samples at sampling
frequency `fs` [Hz]: the inverse transform of [`inverse_segment`](@ref)
scaled by `fs`, so that a spectrum ``\\tilde h(f)`` in seconds gives the
strain ``h(t) = \\int \\tilde h(f) e^{2\\pi i f t}\\, \\mathrm{d}f`` that
[`matched_filter_snr`](@ref) transforms back to ``\\tilde h``.
"""
function phenoma_series(H::AbstractVector{<:Complex}, n::Integer, fs::Real)
    fs > 0 || throw(ArgumentError("fs = $fs; the sampling frequency must be positive."))
    return inverse_segment(H, n) .* fs
end

"""
    phenoma_physical_amplitude(p; distance_sec)

Newtonian normalisation of the IMRPhenomA spectrum for a source at
luminosity distance `distance_sec` [light-seconds]:
``\\sqrt{5/24}\\, \\pi^{-2/3}\\, \\mathcal{M}^{5/6} f_\\mathrm{merg}^{-7/6} / d_L``
[s], with the chirp mass ``\\mathcal{M} = M \\eta^{3/5}`` in seconds
(Ajith et al. 2008, Eq. 4.17). Multiplying the unit spectrum of
[`phenoma_spectrum`](@ref), whose inspiral is ``(f/f_\\mathrm{merg})^{-7/6}``,
gives ``|H(f)| = \\sqrt{5/24}\\, \\pi^{-2/3} \\mathcal{M}^{5/6} f^{-7/6} / d_L``
in the inspiral: the plus-polarisation amplitude of a face-on source,
before the inclination factors ``(1 + \\cos^2\\iota)/2`` and ``\\cos\\iota``
and the antenna patterns.
"""
function phenoma_physical_amplitude(p; distance_sec::Real)
    distance_sec > 0 ||
        throw(ArgumentError("distance_sec = $distance_sec; must be positive."))
    chirp_mass = p.M_sec * p.η^(3 / 5)
    return sqrt(5 / 24) * π^(-2 / 3) * chirp_mass^(5 / 6) * p.f_merg^(-7 / 6) / distance_sec
end

"""
    phenoma_arrival_delay(f, p)

Time [s] before the arrival of the merger–ringdown at which frequency `f`
[Hz] arrives: ``g(f) - g(f_\\mathrm{ring})`` of [`phenoma_group_delay`](@ref)
below the ringdown frequency, clamped at zero, and zero at and above it,
where the derivative of the fitted phase is no longer an arrival time
(and at ``f \\le 0``, which carries no signal). A detector response evaluated at
this instant, and at the coalescence through the merger–ringdown, is the
leading order of the Fourier-domain response of Marsat & Baker (2018).
"""
function phenoma_arrival_delay(f::Real, p)
    (f <= 0 || f >= p.f_ring) && return 0.0
    return max(phenoma_group_delay(f, p) - phenoma_group_delay(p.f_ring, p), 0.0)
end

"""
    phenoma_waveform(fs, duration_sec; total_mass, mass_ratio,
                     nyquist_taper = 0.9, rolloff_width = 0.1) -> (h, merger_index, p)

Unit-peak time-domain IMRPhenomA strain of a non-spinning binary of
detector-frame `total_mass` [M⊙] and `mass_ratio` ``q \\ge 1``, sampled at
`fs` [Hz] over `duration_sec`, with coalescence near the end of the
segment: the spectrum of [`phenoma_spectrum`](@ref) inverted by
[`inverse_segment`](@ref) and scaled to unit peak amplitude.

Returns the strain, the sample of peak amplitude (`merger_index`), and the
parameters `p` of [`phenoma_parameters`](@ref). The overall amplitude
carries no physical meaning; injections are scaled to a matched-filter
SNR by [`scale_to_snr`](@ref).
"""
function phenoma_waveform(
    fs::Real,
    duration_sec::Real;
    total_mass::Real,
    mass_ratio::Real,
    nyquist_taper::Real = 0.9,
    rolloff_width::Real = 0.1,
)
    spectrum = phenoma_spectrum(
        fs,
        duration_sec;
        total_mass = total_mass,
        mass_ratio = mass_ratio,
        nyquist_taper = nyquist_taper,
        rolloff_width = rolloff_width,
    )
    h = inverse_segment(spectrum.H, spectrum.n)
    peak = maximum(abs, h)
    peak > 0 || throw(
        ArgumentError(
            "the waveform vanishes on the sampling grid (total_mass = $total_mass M⊙, " *
            "fs = $fs Hz): every frequency lies outside the representable band.",
        ),
    )
    h ./= peak
    merger_index = argmax(abs.(h))
    return h, merger_index, spectrum.p
end
