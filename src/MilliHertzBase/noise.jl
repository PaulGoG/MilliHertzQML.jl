# LISA noise model of Robson, Cornish & Liu (2019) and the detectable span
# of an injected signal.

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
