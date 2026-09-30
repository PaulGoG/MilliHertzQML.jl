# Signal processing of sampled records: Gaussian noise of a given PSD, the
# matched-filter SNR, whitening, the tapered periodogram, the zero-phase
# high-pass, signal placement, and Welch estimation, smoothing and
# interpolation of a PSD.

"""
    synthesize_noise(rng, n, fs; psd, f_min = 0.0) -> Vector{Float64}

`n` samples of zero-mean stationary Gaussian noise at sampling frequency
`fs` [Hz] whose one-sided power spectral density is `psd(f)` [Hz⁻¹]. The
spectral coefficients are drawn as complex normals scaled to
``E|X_k|^2 = S(f_k) f_s n / 2`` (the unnormalised `rfft` convention), the DC
bin is zeroed, and the Nyquist bin of an even `n` is forced real, so the
inverse transform is a real series with the correct absolute amplitude.
Bins below `f_min` [Hz] are left empty: a sensitivity model fitted over a
band would, extrapolated towards zero frequency, put a drift many orders of
magnitude above the in-band level into the record.
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
    welch_psd(x, fs; segment_length, overlap = 0.5, taper = :hann, average = :median)
        -> (freqs, psd)
    welch_psd(records, fs; segment_length, overlap = 0.5, taper = :hann, average = :median)
        -> (freqs, psd)

One-sided PSD estimate [Hz⁻¹] of the record `x` sampled at `fs` [Hz] from
tapered periodograms ([`tapered_periodogram`](@ref)) of segments of
`segment_length` samples overlapping by the fraction `overlap`. `average`
is `:mean` (Welch) or `:median` (robust to transients; the median of the
exponentially distributed periodogram is corrected by ``1/\\ln 2``). The DC
bin is dropped, so `freqs` starts at ``f_s / \\texttt{segment\\_length}``.

Given a vector of `records` — the delivered runs of a record with holes —
the segments of every record long enough to hold one are pooled into a
single estimate, so that no segment spans a hole; a record shorter than a
segment is skipped, and `ArgumentError` is thrown when none holds one.
"""
function welch_psd(
    x::AbstractVector{<:Real},
    fs::Real;
    segment_length::Integer,
    overlap::Real = 0.5,
    taper::Symbol = :hann,
    average::Symbol = :median,
)
    2 <= segment_length <= length(x) || throw(
        ArgumentError(
            "segment_length = $segment_length; must lie in [2, $(length(x))] for this record.",
        ),
    )
    return welch_psd([x], fs; segment_length, overlap, taper, average)
end

function welch_psd(
    records::AbstractVector{<:AbstractVector{<:Real}},
    fs::Real;
    segment_length::Integer,
    overlap::Real = 0.5,
    taper::Symbol = :hann,
    average::Symbol = :median,
)
    fs > 0 || throw(ArgumentError("fs = $fs; the sampling frequency must be positive."))
    segment_length >= 2 ||
        throw(ArgumentError("segment_length = $segment_length; must be at least 2."))
    0 <= overlap < 1 || throw(ArgumentError("overlap = $overlap; must lie in [0, 1)."))
    average in (:mean, :median) ||
        throw(ArgumentError("average = $average; expected :mean or :median."))
    long = filter(x -> length(x) >= segment_length, records)
    isempty(long) &&
        throw(ArgumentError("no record holds a segment of $segment_length samples."))
    hop = max(1, round(Int, segment_length * (1 - overlap)))
    n_freqs = div(segment_length, 2) + 1
    columns = Vector{Vector{Float64}}()
    for x in long
        n_segments = div(length(x) - segment_length, hop) + 1
        for s in 1:n_segments
            lo = (s - 1) * hop + 1
            push!(
                columns,
                tapered_periodogram(view(x, lo:(lo+segment_length-1)); taper = taper),
            )
        end
    end
    P = reduce(hcat, columns)
    psd = Vector{Float64}(undef, n_freqs - 1)
    for k in 2:n_freqs
        row = view(P, k, :)
        psd[k-1] = average == :mean ? mean(row) : median(row) / log(2)
    end
    psd .*= 2 / fs
    freqs = rfftfreq(segment_length, fs)[2:end]
    return collect(freqs), psd
end

"""
    smooth_psd(freqs, psd, sigma_dex) -> Vector{Float64}

The one-sided PSD `psd` tabulated at the strictly increasing positive
frequencies `freqs`, smoothed in log-frequency: ``\\log_{10} S`` is
averaged with Gaussian weights of standard deviation `sigma_dex` in
``\\log_{10} f``, each bin weighted by the log-frequency interval it spans
(``\\propto 1/f`` on a uniform frequency grid), evaluated on a uniform
log-frequency grid of step `sigma_dex / 5` with the kernel truncated at
four standard deviations, and read back at `freqs` by linear
interpolation. Where the table is coarser than the kernel — its lowest
bins — the grid interpolates the table instead. `sigma_dex = 0` returns
the table unchanged.

A Welch estimate carries line-to-line scatter and resolves sharp spectral
features, the TDI transfer notches among them; the inverse square root of
such a spectrum has a long, ringing impulse response, so whitening by it
spreads every sample over many window lengths. Smoothing on scales
narrower than any analysis band removes that fine structure and shortens
the kernel accordingly.
"""
function smooth_psd(
    freqs::AbstractVector{<:Real},
    psd::AbstractVector{<:Real},
    sigma_dex::Real,
)
    n = length(freqs)
    n == length(psd) ||
        throw(DimensionMismatch("$n frequencies for $(length(psd)) PSD values."))
    n >= 2 || throw(ArgumentError("the table holds fewer than two frequencies."))
    sigma_dex >= 0 || throw(ArgumentError("sigma_dex = $sigma_dex; must be non-negative."))
    (first(freqs) > 0 && all(>(0), diff(freqs))) ||
        throw(ArgumentError("the frequencies must be positive and strictly increasing."))
    all(>(0), psd) || throw(ArgumentError("the PSD must be positive at every frequency."))
    iszero(sigma_dex) && return Vector{Float64}(psd)
    f = Vector{Float64}(freqs)
    lf = log10.(f)
    ls = log10.(Vector{Float64}(psd))
    n_grid = max(2, ceil(Int, (lf[end] - lf[1]) / (sigma_dex / 5)) + 1)
    grid = collect(range(lf[1], lf[end]; length = n_grid))
    reach = 4 * sigma_dex
    smooth = similar(grid)
    lo, hi = 1, 0          # the bins within `reach` of the current grid point
    for (i, g) in enumerate(grid)
        while lo <= n && lf[lo] < g - reach
            lo += 1
        end
        while hi < n && lf[hi+1] <= g + reach
            hi += 1
        end
        if lo <= hi
            acc = 0.0
            weight = 0.0
            for j in lo:hi
                w = exp(-0.5 * ((lf[j] - g) / sigma_dex)^2) / f[j]
                acc += w * ls[j]
                weight += w
            end
            smooth[i] = acc / weight
        else
            smooth[i] = linear_interpolation(lf, ls, g)
        end
    end
    return [exp10(linear_interpolation(grid, smooth, x)) for x in lf]
end

"""
    linear_interpolation(x, y, xq) -> Float64

Value at `xq` of the piecewise-linear interpolant of `y` over the strictly
increasing abscissae `x`, constant beyond the ends.
"""
function linear_interpolation(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    xq::Real,
)
    xq <= first(x) && return Float64(first(y))
    xq >= last(x) && return Float64(last(y))
    i = searchsortedlast(x, xq)
    t = (xq - x[i]) / (x[i+1] - x[i])
    return (1 - t) * y[i] + t * y[i+1]
end

"""
    interpolated_psd(freqs, psd) -> Function

Callable ``f \\mapsto S(f)`` interpolating the tabulated one-sided PSD
`psd` at the strictly increasing positive frequencies `freqs` linearly in
``\\log f``–``\\log S``, constant outside the tabulated range, and `Inf` for
``f \\le 0`` (so whitening sets the DC bin to zero).
"""
function interpolated_psd(freqs::AbstractVector{<:Real}, psd::AbstractVector{<:Real})
    length(freqs) == length(psd) || throw(
        DimensionMismatch("$(length(freqs)) frequencies for $(length(psd)) PSD values."),
    )
    length(freqs) >= 2 ||
        throw(ArgumentError("at least two tabulated points are required."))
    (all(>(0), freqs) && issorted(freqs; lt = <=)) ||
        throw(ArgumentError("frequencies must be positive and strictly increasing."))
    all(p -> isfinite(p) && p > 0, psd) ||
        throw(ArgumentError("PSD values must be positive and finite."))
    log_f = log.(Float64.(freqs))
    log_s = log.(Float64.(psd))
    s_knots = Float64.(psd)
    return function (f::Real)
        f > 0 || return Inf
        lf = log(f)
        lf <= log_f[1] && return s_knots[1]
        lf >= log_f[end] && return s_knots[end]
        k = searchsortedlast(log_f, lf)
        lf == log_f[k] && return s_knots[k]
        w = (lf - log_f[k]) / (log_f[k+1] - log_f[k])
        return exp(log_s[k] + w * (log_s[k+1] - log_s[k]))
    end
end
