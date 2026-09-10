# src/ldc.jl — LISA Data Challenge products: the analytic TDI noise model of
# the `ldc` package (equal arms), readers of the compound HDF5 TDI datasets
# and catalogs, the A/E/T combination, Welch PSD estimation of a record, and
# event labeling from a truth stream. TDI variables are dimensionless
# fractional-frequency quantities; PSDs in Hz⁻¹.

"""
    LDC_NOISE_LEVELS

Single-link noise levels of the analytic models of the `ldc` package
(`ldc.lisa.noise.AnalyticNoise`): optical-metrology displacement PSD `oms`
[m² Hz⁻¹] and test-mass acceleration PSD `acc` [m² s⁻⁴ Hz⁻¹], keyed by model
name.
"""
const LDC_NOISE_LEVELS = Dict{String,NamedTuple{(:oms, :acc),NTuple{2,Float64}}}(
    "Proposal" => (oms = (10e-12)^2, acc = (3e-15)^2),
    "SciRDv1" => (oms = (15e-12)^2, acc = (3e-15)^2),
    "MRDv1" => (oms = (10e-12)^2, acc = (2.4e-15)^2),
    "MRD_MFR" => (oms = (13.5e-12)^2, acc = (2.7e-15)^2),
    "sangria" => (oms = (7.9e-12)^2, acc = (2.4e-15)^2),
    "spritz" => (oms = (7.9e-12)^2, acc = (2.4e-15)^2),
)

"""
    LDC_CONFUSION_FIT

Galactic-confusion fit of the `ldc` package (`GalNoise`, six links, SNR 7
removal threshold): amplitude, exponent, transition width, and the
observation-time scalings of the roll-off and knee frequencies.
"""
const LDC_CONFUSION_FIT = (
    amplitude = 1.28265531e-44,
    α = 1.62966700,
    f_r2 = 4.81078093e-4,
    a_f1 = -2.23499956e-1,
    b_f1 = -2.70408439,
    a_fk = -3.60976122e-1,
    b_fk = -2.37822436,
)

"""
    ldc_noise_levels(model) -> NamedTuple

Noise levels of [`LDC_NOISE_LEVELS`](@ref) for `model`; `ArgumentError` for
an unknown name.
"""
function ldc_noise_levels(model::AbstractString)
    haskey(LDC_NOISE_LEVELS, model) || throw(
        ArgumentError(
            "unknown LDC noise model $(repr(model)); expected one of " *
            join(sort(collect(keys(LDC_NOISE_LEVELS))), ", ") *
            ".",
        ),
    )
    return LDC_NOISE_LEVELS[model]
end

"""
    ldc_confusion_psd(f; channel = :A, observation_years = 1.0, tdi2 = false)

Galactic-confusion contribution [Hz⁻¹] to the TDI noise PSD of `channel`
(`:X`, `:XY`, `:A`, `:E`) after `observation_years` of foreground
subtraction, as fitted in the `ldc` package (`Noise.wd_confusion` with
`GalNoise`): the sensitivity-level shape

```math
S_\\mathrm{gal}(f) = A\\, e^{-(f/f_1)^{\\alpha}} f^{-7/3}\\,
  \\tfrac{1}{2}\\left[1 + \\tanh\\frac{f_k - f}{f_{r2}}\\right]
```

with ``f_1`` and ``f_k`` power laws of the observation time, multiplied by
the TDI response ``4 x^2 \\sin^2 x`` (``x = 2\\pi f L / c``), by
``4 \\sin^2 2x`` for second-generation TDI, and by 3/2 for `:A`/`:E`
(−1/2 for `:XY`). Valid for 0.25 ≤ `observation_years` ≤ 10. Returns 0 for
`f ≤ 0`.
"""
function ldc_confusion_psd(
    f::Real;
    channel::Symbol = :A,
    observation_years::Real = 1.0,
    tdi2::Bool = false,
)
    0.25 <= observation_years <= 10 || throw(
        ArgumentError(
            "observation_years = $observation_years; the confusion fit is valid " *
            "between 0.25 and 10 years.",
        ),
    )
    f > 0 || return 0.0
    p = LDC_CONFUSION_FIT
    lt = log10(observation_years)
    f_1 = 10.0^(p.a_f1 * lt + p.b_f1)
    f_k = 10.0^(p.a_fk * lt + p.b_fk)
    shape =
        p.amplitude *
        exp(-(f / f_1)^p.α) *
        f^(-7 / 3) *
        0.5 *
        (1 + tanh((f_k - f) / p.f_r2))
    x = 2π * f * L_ARM / C_LIGHT
    s = 4 * x^2 * sin(x)^2 * shape
    tdi2 && (s *= 4 * sin(2x)^2)
    channel in (:A, :E) && return 1.5 * s
    channel == :XY && return -0.5 * s
    channel == :X && return s
    throw(
        ArgumentError("channel = $channel; the confusion term is defined for X, XY, A, E."),
    )
end

"""
    ldc_tdi_psd(f; channel = :A, model = "sangria", tdi2 = false, observation_years = 0.0)

One-sided noise PSD [Hz⁻¹] of the TDI variable `channel` (`:X`, `:XY`,
`:A`, `:E`, `:T`) in fractional-frequency units, for equal arms of length
``L`` = [`L_ARM`](@ref), as in `AnalyticNoise.psd` of the `ldc` package.
With ``x = 2\\pi f L / c``, the single-link test-mass and optical-metrology
terms in fractional frequency

```math
S_\\mathrm{pm} = S_\\mathrm{acc}(f)\\,(2\\pi f)^{-4} (2\\pi f / c)^2, \\qquad
S_\\mathrm{op} = S_\\mathrm{oms}(f)\\,(2\\pi f / c)^2,
```

with ``S_\\mathrm{acc} = A_\\mathrm{acc} [1 + (0.4\\,\\mathrm{mHz}/f)^2][1 + (f/8\\,\\mathrm{mHz})^4]``
and ``S_\\mathrm{oms} = A_\\mathrm{oms} [1 + (2\\,\\mathrm{mHz}/f)^4]`` (no
relaxation term for `"spritz"`), enter the first-generation TDI PSDs

```math
\\begin{aligned}
S_X &= 16 \\sin^2 x\\,[2 (1 + \\cos^2 x) S_\\mathrm{pm} + S_\\mathrm{op}],\\\\
S_A = S_E &= 8 \\sin^2 x\\,[2 S_\\mathrm{pm} (3 + 2\\cos x + \\cos 2x) + S_\\mathrm{op} (2 + \\cos x)],\\\\
S_T &= 16 S_\\mathrm{op} (1 - \\cos x) \\sin^2 x + 128 S_\\mathrm{pm} \\sin^2 x \\sin^4 (x/2),
\\end{aligned}
```

and ``S_{XY} = -4 \\sin 2x \\sin x\\,(S_\\mathrm{op} + 4 S_\\mathrm{pm})``.
`tdi2` multiplies by ``4 \\sin^2 2x`` (second-generation TDI); a positive
`observation_years` adds [`ldc_confusion_psd`](@ref). The levels
``A_\\mathrm{acc}, A_\\mathrm{oms}`` come from [`LDC_NOISE_LEVELS`](@ref).
Returns `Inf` for `f ≤ 0`. Reproduces the package doctest of the SciRDv1
X-channel PSD to eight digits.
"""
function ldc_tdi_psd(
    f::Real;
    channel::Symbol = :A,
    model::AbstractString = "sangria",
    tdi2::Bool = false,
    observation_years::Real = 0.0,
)
    levels = ldc_noise_levels(model)
    channel in (:X, :XY, :A, :E, :T) ||
        throw(ArgumentError("channel = $channel; expected one of X, XY, A, E, T."))
    f > 0 || return Inf
    s_acc = levels.acc * (1 + (0.4e-3 / f)^2) * (1 + (f / 8e-3)^4)
    s_pm = s_acc * (2π * f)^(-4) * (2π * f / C_LIGHT)^2
    relaxation = model == "spritz" ? 1.0 : 1 + (2e-3 / f)^4
    s_op = levels.oms * relaxation * (2π * f / C_LIGHT)^2
    x = 2π * f * L_ARM / C_LIGHT
    s = if channel == :X
        16 * sin(x)^2 * (2 * (1 + cos(x)^2) * s_pm + s_op)
    elseif channel == :XY
        -4 * sin(2x) * sin(x) * (s_op + 4 * s_pm)
    elseif channel == :T
        16 * s_op * (1 - cos(x)) * sin(x)^2 + 128 * s_pm * sin(x)^2 * sin(x / 2)^4
    else
        8 * sin(x)^2 * (2 * s_pm * (3 + 2cos(x) + cos(2x)) + s_op * (2 + cos(x)))
    end
    tdi2 && (s *= 4 * sin(2x)^2)
    if observation_years > 0
        s += ldc_confusion_psd(
            f;
            channel = channel,
            observation_years = observation_years,
            tdi2 = tdi2,
        )
    end
    return s
end

"""
    tdi_to_aet(X, Y, Z) -> (A, E, T)

Orthogonal TDI combinations ``A = (Z - X)/\\sqrt{2}``,
``E = (X - 2Y + Z)/\\sqrt{6}``, ``T = (X + Y + Z)/\\sqrt{3}`` of the
first-generation Michelson variables (the LDC convention).
"""
function tdi_to_aet(
    X::AbstractVector{<:Real},
    Y::AbstractVector{<:Real},
    Z::AbstractVector{<:Real},
)
    length(X) == length(Y) == length(Z) || throw(
        DimensionMismatch("X, Y, Z have lengths $(length(X)), $(length(Y)), $(length(Z))."),
    )
    A = (Z .- X) ./ sqrt(2)
    E = (X .- 2 .* Y .+ Z) ./ sqrt(6)
    T = (X .+ Y .+ Z) ./ sqrt(3)
    return A, E, T
end

"""
    hdf5_dataset(parent, name) -> HDF5.Dataset

The dataset `name` of the HDF5 file or group `parent`; `ArgumentError` when
the object is absent or is not a dataset.
"""
function hdf5_dataset(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString)
    haskey(parent, name) ||
        throw(ArgumentError("$(HDF5.filename(parent)) holds no object at $(repr(name))."))
    obj = parent[name]
    obj isa HDF5.Dataset ||
        throw(ArgumentError("$(repr(name)) in $(HDF5.filename(parent)) is not a dataset."))
    return obj
end

"""
    read_tdi(path; group = "obs/tdi") -> NamedTuple

Time vector and Michelson variables `(t, X, Y, Z, dt)` of an HDF5 TDI
product. `group` is either a compound dataset with fields `t`, `X`, `Y`, `Z`
(the LDC layout, e.g. `obs/tdi`, `sky/mbhb/tdi`) or a group of plain
datasets `t`, `X`, `Z` and optionally `Y` (the simulator's layout; a missing
`Y` reads as zeros). `dt` comes from the dataset attribute of that name when
present, otherwise from the first two time samples.
"""
function read_tdi(path::AbstractString; group::AbstractString = "obs/tdi")
    isfile(path) || throw(ArgumentError("HDF5 file not found: $path"))
    return h5open(path, "r") do file
        haskey(file, group) ||
            throw(ArgumentError("$path holds no object at $(repr(group))."))
        obj = file[group]
        obj isa HDF5.Group ||
            obj isa HDF5.Dataset ||
            throw(
                ArgumentError("$(repr(group)) of $path is neither a group nor a dataset."),
            )
        local t, X, Y, Z
        if obj isa HDF5.Group
            for name in ("t", "X", "Z")
                haskey(obj, name) || throw(
                    ArgumentError("group $(repr(group)) of $path lacks the dataset $name."),
                )
            end
            t = Float64.(vec(read(hdf5_dataset(obj, "t"))))
            X = Float64.(vec(read(hdf5_dataset(obj, "X"))))
            Z = Float64.(vec(read(hdf5_dataset(obj, "Z"))))
            Y =
                haskey(obj, "Y") ? Float64.(vec(read(hdf5_dataset(obj, "Y")))) :
                zeros(length(t))
        else
            rows = vec(read(obj))
            isempty(rows) &&
                throw(ArgumentError("dataset $(repr(group)) of $path is empty."))
            for name in (:t, :X, :Y, :Z)
                hasproperty(first(rows), name) || throw(
                    ArgumentError(
                        "compound dataset $(repr(group)) of $path lacks the field $name.",
                    ),
                )
            end
            t = Float64[r.t for r in rows]
            X = Float64[r.X for r in rows]
            Y = Float64[r.Y for r in rows]
            Z = Float64[r.Z for r in rows]
        end
        length(t) == length(X) == length(Y) == length(Z) ||
            throw(DimensionMismatch("the TDI datasets of $path differ in length."))
        dt = if haskey(attributes(obj), "dt")
            Float64(read(attributes(obj)["dt"]))
        elseif length(t) >= 2
            t[2] - t[1]
        else
            throw(ArgumentError("cannot determine the sampling step of $path."))
        end
        dt > 0 || throw(ArgumentError("non-positive sampling step $dt in $path."))
        (t = t, X = X, Y = Y, Z = Z, dt = dt)
    end
end

"""
    read_catalog(path; group = "sky/mbhb/cat") -> DataFrame

Source catalog of an LDC product: the compound dataset at `group` as a
table, one column per field.
"""
function read_catalog(path::AbstractString; group::AbstractString = "sky/mbhb/cat")
    isfile(path) || throw(ArgumentError("HDF5 file not found: $path"))
    return h5open(path, "r") do file
        haskey(file, group) ||
            throw(ArgumentError("$path holds no object at $(repr(group))."))
        rows = vec(read(hdf5_dataset(file, group)))
        DataFrame(rows)
    end
end

"""
    welch_psd(x, fs; segment_length, overlap = 0.5, taper = :hann, average = :median)
        -> (freqs, psd)

One-sided PSD estimate [Hz⁻¹] of the record `x` sampled at `fs` [Hz] from
tapered periodograms ([`tapered_periodogram`](@ref)) of segments of
`segment_length` samples overlapping by the fraction `overlap`. `average`
is `:mean` (Welch) or `:median` (robust to transients; the median of the
exponentially distributed periodogram is corrected by ``1/\\ln 2``). The DC
bin is dropped, so `freqs` starts at ``f_s / \\texttt{segment\\_length}``.
"""
function welch_psd(
    x::AbstractVector{<:Real},
    fs::Real;
    segment_length::Integer,
    overlap::Real = 0.5,
    taper::Symbol = :hann,
    average::Symbol = :median,
)
    fs > 0 || throw(ArgumentError("fs = $fs; the sampling frequency must be positive."))
    2 <= segment_length <= length(x) || throw(
        ArgumentError(
            "segment_length = $segment_length; must lie in [2, $(length(x))] for this record.",
        ),
    )
    0 <= overlap < 1 || throw(ArgumentError("overlap = $overlap; must lie in [0, 1)."))
    average in (:mean, :median) ||
        throw(ArgumentError("average = $average; expected :mean or :median."))
    hop = max(1, round(Int, segment_length * (1 - overlap)))
    n_segments = div(length(x) - segment_length, hop) + 1
    n_freqs = div(segment_length, 2) + 1
    P = Matrix{Float64}(undef, n_freqs, n_segments)
    for s in 1:n_segments
        lo = (s - 1) * hop + 1
        P[:, s] = tapered_periodogram(view(x, lo:(lo+segment_length-1)); taper = taper)
    end
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
    interpolated_psd(freqs, psd) -> Function

Callable ``f \\mapsto S(f)`` interpolating the tabulated one-sided PSD
`psd` at the strictly increasing positive frequencies `freqs` linearly in
``\\log f``–``\\log S``, constant outside the tabulated range, and `Inf` for
``f \\le 0`` (so whitening annihilates the DC bin).
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

"""
    windowed_snr(x, fs; window_size, step, psd) -> (starts, snr)

Matched-filter SNR ([`matched_filter_snr`](@ref)) of every window of
`window_size` samples of the signal `x` at the window starts
`1:step:(n - window_size + 1)`, against the one-sided PSD `psd(f)`.
"""
function windowed_snr(
    x::AbstractVector{<:Real},
    fs::Real;
    window_size::Integer,
    step::Integer,
    psd,
)
    window_size >= 2 ||
        throw(ArgumentError("window_size = $window_size; must be at least 2."))
    step >= 1 || throw(ArgumentError("step = $step; must be at least 1."))
    length(x) >= window_size || throw(
        ArgumentError("the signal holds $(length(x)) samples, fewer than window_size."),
    )
    starts = 1:step:(length(x)-window_size+1)
    snr = Vector{Float64}(undef, length(starts))
    for (i, s) in enumerate(starts)
        snr[i] = matched_filter_snr(view(x, s:(s+window_size-1)), fs; psd = psd)
    end
    return starts, snr
end

"""
    snr_peaks(starts, snr; threshold, min_separation, precursor_window = 0,
              precursor_ratio = 0.1) -> Vector{Int}

Indices into `snr` of its local maxima at or above `threshold`, greedily
accepted in decreasing order so that accepted peaks are at least
`min_separation` samples apart (window starts `starts` are in samples).
A peak that precedes a larger accepted peak by at most `precursor_window`
samples while staying below `precursor_ratio` times its height is a
fluctuation of that source's inspiral, not a merger, and is dropped. Used
to locate mergers in a truth stream without a catalog.
"""
function snr_peaks(
    starts::AbstractRange{<:Integer},
    snr::AbstractVector{<:Real};
    threshold::Real,
    min_separation::Integer,
    precursor_window::Integer = 0,
    precursor_ratio::Real = 0.1,
)
    length(starts) == length(snr) || throw(
        DimensionMismatch("$(length(starts)) window starts for $(length(snr)) values."),
    )
    threshold > 0 || throw(ArgumentError("threshold = $threshold; must be positive."))
    min_separation >= 0 ||
        throw(ArgumentError("min_separation = $min_separation; must be non-negative."))
    precursor_window >= 0 ||
        throw(ArgumentError("precursor_window = $precursor_window; must be non-negative."))
    0 <= precursor_ratio <= 1 ||
        throw(ArgumentError("precursor_ratio = $precursor_ratio; must lie in [0, 1]."))
    n = length(snr)
    candidates = Int[]
    for i in 1:n
        snr[i] >= threshold || continue
        (i == 1 || snr[i] >= snr[i-1]) &&
            (i == n || snr[i] > snr[i+1]) &&
            push!(candidates, i)
    end
    sort!(candidates; by = i -> snr[i], rev = true)
    accepted = Int[]
    for i in candidates
        all(j -> abs(starts[i] - starts[j]) >= min_separation, accepted) &&
            push!(accepted, i)
    end
    precursor(i) = any(
        j ->
            0 < starts[j] - starts[i] <= precursor_window &&
            snr[i] < precursor_ratio * snr[j],
        accepted,
    )
    filter!(i -> !precursor(i), accepted)
    return sort!(accepted)
end

"""
    detectable_spans(starts, snr, window_size; threshold) -> Vector{UnitRange{Int}}

Maximal sample ranges covered by windows (of `window_size` samples starting
at `starts`) whose SNR reaches `threshold`; consecutive qualifying windows
merge into one span. The multi-event analogue of
[`detectable_span`](@ref).
"""
function detectable_spans(
    starts::AbstractRange{<:Integer},
    snr::AbstractVector{<:Real},
    window_size::Integer;
    threshold::Real,
)
    length(starts) == length(snr) || throw(
        DimensionMismatch("$(length(starts)) window starts for $(length(snr)) values."),
    )
    threshold > 0 || throw(ArgumentError("threshold = $threshold; must be positive."))
    spans = UnitRange{Int}[]
    for run in contiguous_runs(snr .>= threshold)
        lo = starts[first(run)]
        hi = starts[last(run)] + window_size - 1
        if !isempty(spans) && lo <= last(spans[end]) + 1
            spans[end] = first(spans[end]):hi
        else
            push!(spans, lo:hi)
        end
    end
    return spans
end

"""
    fixed_spans(merger_indices, fs, n; before, after) -> Vector{UnitRange{Int}}

Sample ranges `[i - before·fs, i + after·fs]` around the merger sample
indices, clipped to `1:n` (`before`, `after` in seconds): the label window
of Isfan et al. (2025) is `before = 4 d`, `after = 27 min`.
"""
function fixed_spans(
    merger_indices::AbstractVector{<:Integer},
    fs::Real,
    n::Integer;
    before::Real,
    after::Real,
)
    (before >= 0 && after >= 0) ||
        throw(ArgumentError("before and after must be non-negative."))
    fs > 0 || throw(ArgumentError("fs = $fs; the sampling frequency must be positive."))
    spans = UnitRange{Int}[]
    for i in merger_indices
        1 <= i <= n || throw(ArgumentError("merger index $i lies outside 1:$n."))
        push!(spans, max(1, i-round(Int, before*fs)):min(n, i+round(Int, after*fs)))
    end
    return spans
end

"""
    span_labels(n, spans) -> Vector{Int}

Point-wise labels of length `n`: 1 inside any of the sample ranges `spans`,
0 elsewhere.
"""
function span_labels(n::Integer, spans::AbstractVector{<:AbstractUnitRange{<:Integer}})
    labels = zeros(Int, n)
    for s in spans
        isempty(s) && continue
        (first(s) >= 1 && last(s) <= n) ||
            throw(ArgumentError("span $s lies outside 1:$n."))
        labels[s] .= 1
    end
    return labels
end
