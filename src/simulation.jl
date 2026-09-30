# src/simulation.jl — noise model of Robson, Cornish & Liu (2019), calibrated
# Gaussian-noise synthesis, matched-filter signal-to-noise ratio, and PSD
# whitening. Frequencies in Hz, strain PSDs in Hz⁻¹.

"""
    L_ARM

LISA arm length [m].
"""
const L_ARM = 2.5e9

"""
    C_LIGHT

Speed of light [m s⁻¹].
"""
const C_LIGHT = 2.99792458e8

"""
    F_STAR

Transfer frequency ``f_* = c / (2\\pi L)`` [Hz] of the LISA arm.
"""
const F_STAR = C_LIGHT / (2π * L_ARM)

"""
    CONFUSION_AMPLITUDE

Amplitude ``A`` [Hz⁻¹] of the galactic confusion fit of Robson, Cornish & Liu
(2019), Eq. 14.
"""
const CONFUSION_AMPLITUDE = 9e-45

"""
    CONFUSION_FIT_PARAMETERS

Fit parameters ``(\\alpha, \\beta, \\kappa, \\gamma, f_k)`` of the galactic
confusion foreground (Robson, Cornish & Liu 2019, Table 1), keyed by the
observation time in years over which resolvable binaries are assumed
subtracted. ``f_k`` in Hz.
"""
const CONFUSION_FIT_PARAMETERS =
    Dict{Float64,NamedTuple{(:α, :β, :κ, :γ, :f_k),NTuple{5,Float64}}}(
        0.5 => (α = 0.133, β = 243.0, κ = 482.0, γ = 917.0, f_k = 0.00258),
        1.0 => (α = 0.171, β = 292.0, κ = 1020.0, γ = 1680.0, f_k = 0.00215),
        2.0 => (α = 0.165, β = 299.0, κ = 611.0, γ = 1340.0, f_k = 0.00173),
        4.0 => (α = 0.138, β = -221.0, κ = 521.0, γ = 1680.0, f_k = 0.00113),
    )

"""
    confusion_fit(observation_years) -> NamedTuple

Fit parameters of [`confusion_psd`](@ref) for `observation_years`, which
must be one of the tabulated values (0.5, 1, 2, 4).
"""
function confusion_fit(observation_years::Real)
    key = Float64(observation_years)
    haskey(CONFUSION_FIT_PARAMETERS, key) || throw(
        ArgumentError(
            "observation_years = $observation_years; the confusion fit is tabulated " *
            "for $(sort(collect(keys(CONFUSION_FIT_PARAMETERS)))) years only.",
        ),
    )
    return CONFUSION_FIT_PARAMETERS[key]
end

"""
    instrument_psd(f)

Sky- and polarisation-averaged instrument-noise sensitivity of LISA
(Robson, Cornish & Liu 2019, Eq. 1) at frequency `f` [Hz], in Hz⁻¹:

```math
S_\\mathrm{inst}(f) = \\frac{10}{3 L^2} \\left[ P_\\mathrm{OMS}(f)
  + 2 \\left(1 + \\cos^2 \\frac{f}{f_*}\\right) \\frac{P_\\mathrm{acc}(f)}{(2\\pi f)^4} \\right]
  \\left[ 1 + \\frac{6}{10} \\left(\\frac{f}{f_*}\\right)^2 \\right]
```

with the optical-metrology and acceleration terms of Eqs. 10–11. Returns
`Inf` for `f ≤ 0`, so that whitening sets the DC bin to zero.
"""
function instrument_psd(f::Real)
    f > 0 || return Inf
    p_oms = (1.5e-11)^2 * (1 + (2e-3 / f)^4)
    p_acc = (3e-15)^2 * (1 + (0.4e-3 / f)^2) * (1 + (f / 8e-3)^4)
    x = f / F_STAR
    return 10 / (3 * L_ARM^2) *
           (p_oms + 2 * (1 + cos(x)^2) * p_acc / (2π * f)^4) *
           (1 + 0.6 * x^2)
end

"""
    confusion_psd(f; observation_years = 1.0)

Galactic confusion foreground (Robson, Cornish & Liu 2019, Eq. 14) at
frequency `f` [Hz], in Hz⁻¹:

```math
S_c(f) = A f^{-7/3} \\exp\\left(-f^{\\alpha} + \\beta f \\sin(\\kappa f)\\right)
  \\left[ 1 + \\tanh\\left(\\gamma (f_k - f)\\right) \\right]
```

with the parameters of [`confusion_fit`](@ref) for `observation_years`.
Returns `Inf` for `f ≤ 0`.

The product is evaluated in log space with
``\\log(1 + \\tanh x) = \\log 2 - \\mathrm{softplus}(-2x)``: far above the
knee ``f_k`` the cutoff factor underflows to zero while the exponential can
overflow, and the direct product would give `NaN` where the foreground
vanishes.
"""
function confusion_psd(f::Real; observation_years::Real = 1.0)
    p = confusion_fit(observation_years)
    f > 0 || return Inf
    y = -2 * p.γ * (p.f_k - f)
    softplus = max(y, zero(y)) + log1p(exp(-abs(y)))
    log_s =
        log(CONFUSION_AMPLITUDE) - (7 / 3) * log(f) - f^p.α +
        p.β * f * sin(p.κ * f) +
        log(2) - softplus
    return exp(log_s)
end

"""
    lisa_noise_psd(f; observation_years = 1.0)

Total sky-averaged strain-noise sensitivity ``S_n(f)`` [Hz⁻¹] of LISA:
[`instrument_psd`](@ref) plus [`confusion_psd`](@ref) for the confusion fit
of `observation_years`. This is the quantity the simulator synthesises
noise against, the matched filter integrates over, and the feature
extractor whitens by.
"""
function lisa_noise_psd(f::Real; observation_years::Real = 1.0)
    return instrument_psd(f) + confusion_psd(f; observation_years = observation_years)
end

"""
    synthesize_noise(rng, n, fs; psd, f_min = 0.0) -> Vector{Float64}

`n` samples of zero-mean stationary Gaussian noise at sampling frequency
`fs` [Hz] whose one-sided power spectral density is `psd(f)` [Hz⁻¹]. The
spectral coefficients are drawn as complex normals scaled to
``E|X_k|^2 = S(f_k) f_s n / 2`` (the unnormalised `rfft` convention), the DC
bin is zeroed, and the Nyquist bin of an even `n` is forced real, so the
inverse transform is a real series with the correct absolute amplitude.
Bins below `f_min` [Hz] are left empty: a sensitivity model fitted over a
band (such as [`lisa_noise_psd`](@ref)) would, extrapolated towards zero
frequency, put a drift many orders of magnitude above the in-band level
into the record.
"""
function synthesize_noise(rng::AbstractRNG, n::Integer, fs::Real; psd, f_min::Real = 0.0)
    n >= 2 || throw(ArgumentError("n = $n; at least 2 samples are required."))
    fs > 0 || throw(ArgumentError("fs = $fs; the sampling frequency must be positive."))
    f_min >= 0 || throw(ArgumentError("f_min = $f_min; must be non-negative."))
    freqs = rfftfreq(n, fs)
    z = randn(rng, ComplexF64, length(freqs))
    z[1] = 0
    iseven(n) && (z[end] = sqrt(2) * real(z[end]))
    for k in 2:length(freqs)
        z[k] *= freqs[k] < f_min ? 0.0 : sqrt(psd(freqs[k]) * fs * n / 2)
    end
    return irfft(z, n)
end

"""
    matched_filter_snr(h, fs; psd)

Optimal matched-filter signal-to-noise ratio of the strain series `h`
sampled at `fs` [Hz] against the one-sided noise PSD `psd(f)`:

```math
\\rho^2 = 4 \\int_0^\\infty \\frac{|\\tilde h(f)|^2}{S_n(f)} \\, \\mathrm{d}f
       \\approx 4 \\Delta f \\sum_{k \\ge 1} \\frac{|\\tilde h(f_k)|^2}{S_n(f_k)},
```

with ``\\tilde h(f_k) = \\Delta t \\sum_j h_j e^{-2\\pi i f_k t_j}`` and
``\\Delta f = f_s / n``. The DC bin is excluded.
"""
function matched_filter_snr(h::AbstractVector{<:Real}, fs::Real; psd)
    length(h) >= 2 || throw(ArgumentError("the series must hold at least 2 samples."))
    fs > 0 || throw(ArgumentError("fs = $fs; the sampling frequency must be positive."))
    n = length(h)
    H = rfft(h) ./ fs
    freqs = rfftfreq(n, fs)
    ρ² = 0.0
    for k in 2:length(freqs)
        ρ² += abs2(H[k]) / psd(freqs[k])
    end
    return sqrt(4 * (fs / n) * ρ²)
end

"""
    scale_to_snr(h, fs, ρ_target; psd)

`h` rescaled so that its [`matched_filter_snr`](@ref) equals `ρ_target`.
"""
function scale_to_snr(h::AbstractVector{<:Real}, fs::Real, ρ_target::Real; psd)
    ρ_target > 0 || throw(ArgumentError("ρ_target = $ρ_target; must be positive."))
    ρ = matched_filter_snr(h, fs; psd = psd)
    ρ > 0 ||
        throw(ArgumentError("the series has zero matched-filter SNR; it cannot be scaled."))
    return h .* (ρ_target / ρ)
end

"""
    whiten_record(x, fs; psd) -> Vector{Float64}

Frequency-domain whitening of the whole record `x` sampled at `fs` [Hz] by
the one-sided noise PSD `psd(f)`: ``X_k \\to X_k \\sqrt{2 / (f_s S_n(f_k))}``
with the DC bin zeroed, so that noise following `psd` becomes white with
unit variance (one-sided PSD ``2/f_s``). Applied once per record, before
windowing: the tapered periodogram of a window of the white series is
unbiased irrespective of the PSD slope ([`tapered_periodogram`](@ref)),
whereas the periodogram of a separately whitened window is the PSD smoothed
by the taper's main lobe. The filter is circular; on real records the first
and last windows are edge-affected.
"""
function whiten_record(x::AbstractVector{<:Real}, fs::Real; psd)
    n = length(x)
    n >= 2 || throw(ArgumentError("the record must hold at least 2 samples."))
    fs > 0 || throw(ArgumentError("fs = $fs; the sampling frequency must be positive."))
    X = rfft(x)
    freqs = rfftfreq(n, fs)
    X[1] = 0
    for k in 2:length(freqs)
        X[k] *= sqrt(2 / (fs * psd(freqs[k])))
    end
    return irfft(X, n)
end

"""
    tapered_periodogram(x; taper = :hann) -> Vector{Float64}

One-sided periodogram of the window `x` after multiplication by the taper
``v``: ``P_k = |\\mathrm{FFT}(x v)_k|^2 / (n \\overline{v^2})`` with
``\\overline{v^2}`` the mean square of the taper, so that
``E P_k = \\sigma^2`` for white noise of variance ``\\sigma^2`` (unit mean on
the output of [`whiten_record`](@ref)). The DC bin is set to zero. `taper`
is `:hann` (default) or `:none`.
"""
function tapered_periodogram(x::AbstractVector{<:Real}; taper::Symbol = :hann)
    n = length(x)
    n >= 2 || throw(ArgumentError("the window must hold at least 2 samples."))
    if taper === :hann
        v = [0.5 * (1 - cos(2π * (j - 1) / n)) for j in 1:n]
        X = rfft(x .* v)
        norm = n * mean(abs2, v)
    elseif taper === :none
        X = rfft(x)
        norm = Float64(n)
    else
        throw(ArgumentError("taper = $(repr(taper)); expected :hann or :none."))
    end
    power = abs2.(X) ./ norm
    power[1] = 0
    return power
end

"""
    highpass_record(x, fs; cutoff, order = 8) -> Vector{Float64}

Zero-phase high-pass filtering of the whole record `x` sampled at `fs` [Hz]
in the frequency domain, with the Butterworth magnitude response
``|H(f)| = [1 + (f_c / f)^{2p}]^{-1/2}`` of cutoff `cutoff` ``= f_c`` [Hz]
and order `order` ``= p`` (``H(0) = 0``). `cutoff = 0` returns a copy of
`x`. The milliHertz noise rises steeply towards low frequencies
(acceleration noise ``\\propto f^{-6}`` below 0.4 mHz), so a record must be
high-passed below the analysis bands before it is cut into windows;
otherwise the sub-window drift leaks into every band through any taper.
"""
function highpass_record(
    x::AbstractVector{<:Real},
    fs::Real;
    cutoff::Real,
    order::Integer = 8,
)
    n = length(x)
    n >= 2 || throw(ArgumentError("the record must hold at least 2 samples."))
    fs > 0 || throw(ArgumentError("fs = $fs; the sampling frequency must be positive."))
    cutoff >= 0 || throw(ArgumentError("cutoff = $cutoff; must be non-negative."))
    order >= 1 || throw(ArgumentError("order = $order; must be at least 1."))
    cutoff == 0 && return Vector{Float64}(x)
    X = rfft(x)
    freqs = rfftfreq(n, fs)
    X[1] = 0
    for k in 2:length(freqs)
        X[k] /= sqrt(1 + (cutoff / freqs[k])^(2 * order))
    end
    return irfft(X, n)
end

"""
    place_signal!(strain, signal, anchor_index, signal_anchor) -> UnitRange{Int}

Add `signal` into `strain` so that `signal[signal_anchor]` lands on
`strain[anchor_index]`, dropping the parts of `signal` that fall outside the
record. Returns the range of `strain` indices that received the signal
(empty when nothing overlaps).
"""
function place_signal!(
    strain::AbstractVector{<:Real},
    signal::AbstractVector{<:Real},
    anchor_index::Integer,
    signal_anchor::Integer,
)
    1 <= signal_anchor <= length(signal) || throw(BoundsError(signal, signal_anchor))
    offset = anchor_index - signal_anchor   # strain index = signal index + offset
    first_sig = max(1, 1 - offset)
    last_sig = min(length(signal), length(strain) - offset)
    first_sig <= last_sig || return (anchor_index+1):anchor_index   # empty
    for j in first_sig:last_sig
        strain[j+offset] += signal[j]
    end
    return (first_sig+offset):(last_sig+offset)
end

"""
    detectable_span(placed, covered, fs, window_size, threshold;
                    step = 10, psd = lisa_noise_psd) -> Union{Nothing,UnitRange{Int}}

Range of samples of the record-length signal `placed` (zero outside
`covered`, the range it occupies) that belong to at least one window of
`window_size` samples whose matched-filter SNR against `psd` reaches
`threshold`. Window starts are scanned with stride `step` from
`window_size - 1` samples before `covered` to its end. The per-window SNR
of a chirp rises towards the merger and falls after the ringdown, so the
qualifying windows form one interval; `nothing` when no window qualifies.
This is the span over which an optimal filter operating on single windows
sees the injection, and the default positive-label span of the simulator.
"""
function detectable_span(
    placed::AbstractVector{<:Real},
    covered::AbstractUnitRange{<:Integer},
    fs::Real,
    window_size::Integer,
    threshold::Real;
    step::Integer = 10,
    psd = lisa_noise_psd,
)
    window_size >= 2 ||
        throw(ArgumentError("window_size = $window_size; must be at least 2."))
    step >= 1 || throw(ArgumentError("step = $step; must be at least 1."))
    threshold > 0 || throw(ArgumentError("threshold = $threshold; must be positive."))
    isempty(covered) && return nothing
    n = length(placed)
    lo = max(1, first(covered) - window_size + 1)
    hi = min(n - window_size + 1, last(covered))
    lo <= hi || return nothing
    first_positive = typemax(Int)
    last_positive = 0
    for s in lo:step:hi
        ρ = matched_filter_snr(view(placed, s:(s+window_size-1)), fs; psd = psd)
        if ρ >= threshold
            first_positive = min(first_positive, s)
            last_positive = max(last_positive, s + window_size - 1)
        end
    end
    last_positive == 0 && return nothing
    return first_positive:last_positive
end

"""
    detectable_span(channels::Tuple, covered, fs, window_size, threshold;
                    step = 10, psd = lisa_noise_psd)

The detectable span of a source recorded in several channels, each a
record-length series of the tuple `channels` occupying `covered`: a window
qualifies when the quadrature sum of its per-channel matched-filter SNRs
against the channel PSD `psd` reaches `threshold`. With one channel this
is the single-series method.
"""
function detectable_span(
    channels::Tuple{Vararg{AbstractVector{<:Real}}},
    covered::AbstractUnitRange{<:Integer},
    fs::Real,
    window_size::Integer,
    threshold::Real;
    step::Integer = 10,
    psd = lisa_noise_psd,
)
    isempty(channels) && throw(ArgumentError("at least one channel is required."))
    length(channels) == 1 && return detectable_span(
        channels[1],
        covered,
        fs,
        window_size,
        threshold;
        step = step,
        psd = psd,
    )
    window_size >= 2 ||
        throw(ArgumentError("window_size = $window_size; must be at least 2."))
    step >= 1 || throw(ArgumentError("step = $step; must be at least 1."))
    threshold > 0 || throw(ArgumentError("threshold = $threshold; must be positive."))
    n = length(channels[1])
    all(length(c) == n for c in channels) ||
        throw(DimensionMismatch("the channels must have equal length."))
    isempty(covered) && return nothing
    lo = max(1, first(covered) - window_size + 1)
    hi = min(n - window_size + 1, last(covered))
    lo <= hi || return nothing
    first_positive = typemax(Int)
    last_positive = 0
    for s in lo:step:hi
        ρ² = 0.0
        for c in channels
            ρ² += matched_filter_snr(view(c, s:(s+window_size-1)), fs; psd = psd)^2
        end
        if ρ² >= threshold^2
            first_positive = min(first_positive, s)
            last_positive = max(last_positive, s + window_size - 1)
        end
    end
    last_positive == 0 && return nothing
    return first_positive:last_positive
end
