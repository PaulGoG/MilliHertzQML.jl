# Telemetry generation stage: calibrated LISA
# noise (instrument plus confusion), resolvable galactic binaries and EMRIs
# scaled to a matched-filter SNR over the record, MBHB injections aligned on
# the coalescence sample with point-wise labels and an event catalogue, and
# the persisted products (HDF5 record, label CSV, catalogue CSV, snapshot
# TOML). The random-draw order of every injector is fixed so that a seed
# reproduces a record bit for bit.

"""
    inject_galactic_binaries!(rng, strain, t, fs, n, snr_range, psd) -> strain

Add `n` monochromatic galactic binaries to `strain` in place. Each source
draws a frequency on the grid 0.1 mHz – 10 mHz (step 10 μHz), a uniform
phase, and a matched-filter SNR over the record uniform in `snr_range`
(inclusive bounds, ``\\rho_\\min \\le \\rho_\\max``); the sinusoid is
scaled with [`scale_to_snr`](@ref) against the one-sided PSD `psd(f)`.
`t` [s] is the sample-time vector of the record at sampling frequency `fs`
[Hz]. Draw order per source: frequency, phase, SNR.
"""
function inject_galactic_binaries!(
    rng::AbstractRNG,
    strain::AbstractVector{Float64},
    t::AbstractVector{<:Real},
    fs::Real,
    n::Integer,
    snr_range::Tuple{<:Real,<:Real},
    psd,
)
    length(t) == length(strain) || throw(
        DimensionMismatch("t has $(length(t)) samples but strain has $(length(strain))."),
    )
    n >= 0 || throw(ArgumentError("n = $n; the number of sources must be non-negative."))
    ρ_min, ρ_max = snr_range
    0 < ρ_min <= ρ_max ||
        throw(ArgumentError("snr_range = $snr_range; expected 0 < ρ_min <= ρ_max."))
    for _ in 1:n
        f = rand(rng, 0.0001:0.00001:0.01)   # 0.1 mHz to 10 mHz
        φ = rand(rng) * 2π
        gb = cos.(2π * f .* t .+ φ)
        ρ = ρ_min + rand(rng) * (ρ_max - ρ_min)
        strain .+= scale_to_snr(gb, fs, ρ; psd = psd)
    end
    return strain
end

"""
    inject_emris!(rng, strain, t, fs, n, snr_range, psd) -> strain

Add `n` extreme-mass-ratio inspirals to `strain` in place. Each source is a
linearly chirping fundamental — start frequency on the grid 1 mHz – 5 mHz
(step 0.5 mHz), drift on the grid 1 – 10 nHz s⁻¹ (step 0.1 nHz s⁻¹) — with
its second and third harmonics at amplitudes 1/2 and 1/3, phased by the
cumulative sum of the instantaneous frequency, and scaled with
[`scale_to_snr`](@ref) to a matched-filter SNR over the record uniform in
`snr_range` against the one-sided PSD `psd(f)`. `t` [s] is the sample-time
vector of the record at sampling frequency `fs` [Hz]. Draw order per
source: start frequency, drift, SNR.
"""
function inject_emris!(
    rng::AbstractRNG,
    strain::AbstractVector{Float64},
    t::AbstractVector{<:Real},
    fs::Real,
    n::Integer,
    snr_range::Tuple{<:Real,<:Real},
    psd,
)
    length(t) == length(strain) || throw(
        DimensionMismatch("t has $(length(t)) samples but strain has $(length(strain))."),
    )
    n >= 0 || throw(ArgumentError("n = $n; the number of sources must be non-negative."))
    ρ_min, ρ_max = snr_range
    0 < ρ_min <= ρ_max ||
        throw(ArgumentError("snr_range = $snr_range; expected 0 < ρ_min <= ρ_max."))
    n_total = length(strain)
    for _ in 1:n
        f0 = rand(rng, 0.001:0.0005:0.005)
        dfdt = rand(rng, 1e-9:1e-10:1e-8)
        sig = zeros(n_total)
        for harmonic in (1, 2, 3)
            f_t = harmonic .* (f0 .+ dfdt .* t)
            phase = 2π .* cumsum(f_t) ./ fs
            sig .+= (1.0 / harmonic) .* cos.(phase)
        end
        ρ = ρ_min + rand(rng) * (ρ_max - ρ_min)
        strain .+= scale_to_snr(sig, fs, ρ; psd = psd)
    end
    return strain
end

"""
    event_catalog() -> DataFrame

Empty MBHB event catalogue with the column schema filled by
[`inject_mbhb!`](@ref): event identifier, coalescence time [s] and sample,
matched-filter SNR of the injected samples, detector-frame total mass
[M⊙], mass ratio, symmetric mass ratio, IMRPhenomA merger, ringdown, and
cut-off frequencies [Hz], injected sample span, positive-label span, and
the signal onset from which alerts are credited ([`signal_onset`](@ref)).
"""
function event_catalog()
    return DataFrame(
        event_id = Int[],
        t_c_sec = Float64[],
        merger_index = Int[],
        snr = Float64[],
        total_mass_msun = Float64[],
        mass_ratio = Float64[],
        eta = Float64[],
        f_merger_hz = Float64[],
        f_ring_hz = Float64[],
        f_cut_hz = Float64[],
        start_index = Int[],
        end_index = Int[],
        label_start_index = Int[],
        label_end_index = Int[],
        signal_start_index = Int[],
    )
end

"""
    inject_mbhb!(rng, strain, labels, snrs, catalog, event_id, fs, duration_sec, settings, psd) -> Bool

Inject one massive-black-hole binary into `strain` in place, mark its
positive-label span in `labels` (`1`) and `snrs` (the injected SNR), and
append its row to `catalog` (schema of [`event_catalog`](@ref)). The
coalescence sample is uniform over the central 80 % of the record, the
detector-frame total mass log-uniform in
`[mbhb_total_mass_min, mbhb_total_mass_max]`, and the mass ratio uniform in
`[1, mbhb_mass_ratio_max]`; the IMRPhenomA waveform of `duration_sec` [s]
([`phenoma_waveform`](@ref)) is placed with [`place_signal!`](@ref),
truncated to the record, and the covered samples are scaled with
[`scale_to_snr`](@ref) to a matched-filter SNR uniform in
`[snr_min, snr_max]`. The label span follows `label_span`: `"detectable"`
takes the [`detectable_span`](@ref) of the injected samples at
`label_snr_threshold` over windows of `label_window_size` samples with
stride `label_step` (an empty span when no window reaches the threshold);
`"injection"` takes the covered samples; `"fixed"` takes
`label_before_sec` before and `label_after_sec` after the coalescence.
`settings` is the NamedTuple of [`generation_settings`](@ref) carrying these
keys and `nyquist_taper`; `psd(f)` is the one-sided noise PSD [Hz⁻¹].

Returns `true` when the event was injected and catalogued, `false` when the
placed waveform covers no sample of the record (no SNR is drawn then). Draw
order per event: coalescence sample, total mass, mass ratio, SNR.
"""
function inject_mbhb!(
    rng::AbstractRNG,
    strain::AbstractVector{Float64},
    labels::AbstractVector{Int32},
    snrs::AbstractVector{Float32},
    catalog::DataFrame,
    event_id::Integer,
    fs::Real,
    duration_sec::Real,
    settings::NamedTuple,
    psd,
)
    n_total = length(strain)
    (length(labels) == n_total && length(snrs) == n_total) || throw(
        DimensionMismatch(
            "strain, labels, and snrs must have equal length; got " *
            "$(n_total), $(length(labels)), $(length(snrs)).",
        ),
    )
    n_total >= 2 || throw(ArgumentError("the record must hold at least 2 samples."))
    duration_sec > 0 ||
        throw(ArgumentError("duration_sec = $duration_sec; must be positive."))

    k_c = rand(rng, round(Int, 0.1*n_total):round(Int, 0.9*n_total))
    k_c = clamp(k_c, 1, n_total)
    t_c = (k_c - 1) / fs
    log_mass_min = log10(settings.mbhb_total_mass_min)
    log_mass_max = log10(settings.mbhb_total_mass_max)
    total_mass = 10.0^(log_mass_min + rand(rng) * (log_mass_max - log_mass_min))
    mass_ratio = 1 + rand(rng) * (settings.mbhb_mass_ratio_max - 1)
    signal, merger_index, pars = phenoma_waveform(
        fs,
        duration_sec;
        total_mass = total_mass,
        mass_ratio = mass_ratio,
        nyquist_taper = settings.nyquist_taper,
    )

    # Truncate to the record before scaling, so the recorded SNR is that of
    # the injected samples.
    placed = zeros(n_total)
    covered = place_signal!(placed, signal, k_c, merger_index)
    isempty(covered) && return false
    ρ_target = settings.snr_min + rand(rng) * (settings.snr_max - settings.snr_min)
    segment = scale_to_snr(view(placed, covered), fs, ρ_target; psd = psd)
    strain[covered] .+= segment

    placed[covered] .= segment
    lbl_start, lbl_end = label_bounds(settings, (placed,), covered, k_c, fs, psd)
    onset = signal_onset(settings, (placed,), covered, k_c, lbl_start, lbl_end, fs, psd)
    if lbl_start <= lbl_end
        labels[lbl_start:lbl_end] .= 1
        snrs[lbl_start:lbl_end] .= Float32(ρ_target)
    end
    push!(
        catalog,
        (
            event_id,
            t_c,
            k_c,
            ρ_target,
            total_mass,
            mass_ratio,
            pars.η,
            pars.f_merg,
            pars.f_ring,
            pars.f_cut,
            first(covered),
            last(covered),
            lbl_start,
            lbl_end,
            onset,
        ),
    )
    return true
end

"""
    label_bounds(settings, series, covered, k_c, fs, psd) -> (first, last)

Positive-label span of an injection recorded in the record-length
`series` (a tuple of channels, zero outside `covered`) with coalescence
sample `k_c`, by the criterion `settings.label_span`: `"detectable"` takes
the [`detectable_span`](@ref) at `label_snr_threshold` over windows of
`label_window_size` samples with stride `label_step` against the channel
PSD `psd` (an empty span `(k_c + 1, k_c)` when no window reaches the
threshold); `"injection"` takes the covered samples; `"fixed"` takes
`label_before_sec` before and `label_after_sec` after the coalescence.
"""
function label_bounds(
    settings::NamedTuple,
    series::Tuple{Vararg{AbstractVector{<:Real}}},
    covered::AbstractUnitRange{<:Integer},
    k_c::Integer,
    fs::Real,
    psd,
)
    n_total = length(series[1])
    if settings.label_span == "detectable"
        span = detectable_span(
            series,
            covered,
            fs,
            settings.label_window_size,
            settings.label_snr_threshold;
            step = settings.label_step,
            psd = psd,
        )
        span === nothing && (span = (k_c+1):k_c)   # no window reaches the threshold
        return first(span), last(span)
    elseif settings.label_span == "injection"
        return first(covered), last(covered)
    elseif settings.label_span == "fixed"
        return clamp(k_c - round(Int, settings.label_before_sec * fs), 1, n_total),
        clamp(k_c + round(Int, settings.label_after_sec * fs), 1, n_total)
    end
    throw(ArgumentError("label_span = $(repr(settings.label_span)); unknown criterion."))
end

"""
    signal_onset(settings, series, covered, k_c, label_start, label_end, fs, psd) -> Int

Signal onset of an injection, the start of the span from which alerts are
credited: the first sample of the earliest window of `label_window_size`
samples (stride `label_step`) between the label start and the coalescence
sample `k_c` whose matched-filter SNR of `series` (the injection alone,
zero outside `covered`) against `psd` reaches `label_snr_threshold`
([`detectable_span`](@ref)), or `k_c` when none does. With
`label_span = "detectable"` the label span already starts there, and its
start is returned.
"""
function signal_onset(
    settings::NamedTuple,
    series::Tuple{Vararg{AbstractVector{<:Real}}},
    covered::AbstractUnitRange{<:Integer},
    k_c::Integer,
    label_start::Integer,
    label_end::Integer,
    fs::Real,
    psd,
)
    if settings.label_span == "detectable"
        return label_start <= label_end ? Int(label_start) : Int(k_c)
    end
    lo = max(first(covered), label_start)
    hi = min(last(covered), k_c)
    lo <= hi || return Int(k_c)
    span = detectable_span(
        series,
        lo:hi,
        fs,
        settings.label_window_size,
        settings.label_snr_threshold;
        step = settings.label_step,
        psd = psd,
    )
    span === nothing && return Int(k_c)
    return clamp(first(span), Int(label_start), Int(k_c))
end

"""
    channel_catalog() -> DataFrame

Empty MBHB event catalogue of the constellation response: the columns of
[`event_catalog`](@ref) — `snr` holding the SNR of the channel or network
selected by `label_channel` — followed by the luminosity distance [Gpc],
the ecliptic longitude and latitude, the inclination and polarisation
[rad], and the per-channel matched-filter SNRs `snr_a` and `snr_e`.
"""
function channel_catalog()
    catalog = event_catalog()
    catalog.distance_gpc = Float64[]
    catalog.ecliptic_longitude = Float64[]
    catalog.ecliptic_latitude = Float64[]
    catalog.inclination = Float64[]
    catalog.polarization = Float64[]
    catalog.snr_a = Float64[]
    catalog.snr_e = Float64[]
    return catalog
end

"""
    add_at_network_snr!(channels, series, fs, ρ_target, psd) -> scale

Add the two-channel `series` to `channels` scaled so that the quadrature
sum of its per-channel matched-filter SNRs against the channel PSD `psd`
equals `ρ_target`; returns the scale applied.
"""
function add_at_network_snr!(
    channels::NTuple{2,AbstractVector{Float64}},
    series::NTuple{2,AbstractVector{<:Real}},
    fs::Real,
    ρ_target::Real,
    psd,
)
    ρ = hypot(
        matched_filter_snr(series[1], fs; psd = psd),
        matched_filter_snr(series[2], fs; psd = psd),
    )
    ρ > 0 || throw(
        ArgumentError("the projected source has zero network SNR; it cannot be scaled."),
    )
    scale = ρ_target / ρ
    channels[1] .+= scale .* series[1]
    channels[2] .+= scale .* series[2]
    return scale
end

"""
    inject_galactic_binaries!(rng, channels, t, fs, n, snr_range, psd, response) -> channels

Constellation-response method: each source draws its frequency, phase and
network SNR as the single-strain method does, then the extrinsic
parameters of [`draw_extrinsic`](@ref); the sinusoid is projected on the
A and E channels by [`project_series`](@ref) and scaled with
[`add_at_network_snr!`](@ref) against the channel PSD `psd(f)`.
"""
function inject_galactic_binaries!(
    rng::AbstractRNG,
    channels::NTuple{2,AbstractVector{Float64}},
    t::AbstractVector{<:Real},
    fs::Real,
    n::Integer,
    snr_range::Tuple{<:Real,<:Real},
    psd,
    response::AbstractDetectorResponse,
)
    n_total = length(t)
    all(length(c) == n_total for c in channels) || throw(
        DimensionMismatch(
            "t has $n_total samples but the channels have $(length.(channels)).",
        ),
    )
    n >= 0 || throw(ArgumentError("n = $n; the number of sources must be non-negative."))
    ρ_min, ρ_max = snr_range
    0 < ρ_min <= ρ_max ||
        throw(ArgumentError("snr_range = $snr_range; expected 0 < ρ_min <= ρ_max."))
    for _ in 1:n
        f = rand(rng, 0.0001:0.00001:0.01)   # 0.1 mHz to 10 mHz
        φ = rand(rng) * 2π
        ρ = ρ_min + rand(rng) * (ρ_max - ρ_min)
        source = draw_extrinsic(rng)
        frame =
            source_frame(response, source.longitude, source.latitude, source.polarization)
        series = project_series(
            response,
            frame,
            t,
            ones(n_total),
            2π * f .* t .+ φ,
            fill(f, n_total),
            source.inclination,
        )
        add_at_network_snr!(channels, series, fs, ρ, psd)
    end
    return channels
end

"""
    inject_emris!(rng, channels, t, fs, n, snr_range, psd, response) -> channels

Constellation-response method: each source draws its start frequency,
drift and network SNR as the single-strain method does, then the extrinsic
parameters of [`draw_extrinsic`](@ref); the three harmonics are projected
on the A and E channels by [`project_series`](@ref), each at its own
instantaneous frequency, and scaled together with
[`add_at_network_snr!`](@ref) against the channel PSD `psd(f)`.
"""
function inject_emris!(
    rng::AbstractRNG,
    channels::NTuple{2,AbstractVector{Float64}},
    t::AbstractVector{<:Real},
    fs::Real,
    n::Integer,
    snr_range::Tuple{<:Real,<:Real},
    psd,
    response::AbstractDetectorResponse,
)
    n_total = length(t)
    all(length(c) == n_total for c in channels) || throw(
        DimensionMismatch(
            "t has $n_total samples but the channels have $(length.(channels)).",
        ),
    )
    n >= 0 || throw(ArgumentError("n = $n; the number of sources must be non-negative."))
    ρ_min, ρ_max = snr_range
    0 < ρ_min <= ρ_max ||
        throw(ArgumentError("snr_range = $snr_range; expected 0 < ρ_min <= ρ_max."))
    for _ in 1:n
        f0 = rand(rng, 0.001:0.0005:0.005)
        dfdt = rand(rng, 1e-9:1e-10:1e-8)
        ρ = ρ_min + rand(rng) * (ρ_max - ρ_min)
        source = draw_extrinsic(rng)
        frame =
            source_frame(response, source.longitude, source.latitude, source.polarization)
        f_t = f0 .+ dfdt .* t
        phase = 2π .* cumsum(f_t) ./ fs
        h_A = zeros(n_total)
        h_E = zeros(n_total)
        for harmonic in (1, 2, 3)
            a, e = project_series(
                response,
                frame,
                t,
                fill(1.0 / harmonic, n_total),
                harmonic .* phase,
                harmonic .* f_t,
                source.inclination,
            )
            h_A .+= a
            h_E .+= e
        end
        add_at_network_snr!(channels, (h_A, h_E), fs, ρ, psd)
    end
    return channels
end

"""
    inject_mbhb!(rng, channels, labels, snrs, catalog, event_id, fs, duration_sec,
                 settings, psd, response) -> Bool

Constellation-response method of [`inject_mbhb!`](@ref): the coalescence
sample, total mass and mass ratio are drawn as in the single-strain
method, then the luminosity distance log-uniformly in
`[mbhb_distance_min_gpc, mbhb_distance_max_gpc]` and the extrinsic
parameters of [`draw_extrinsic`](@ref). The IMRPhenomA spectrum
([`phenoma_spectrum`](@ref)) at the physical amplitude of that distance
([`phenoma_physical_amplitude`](@ref)) is projected on the A and E channels
by [`project_spectrum`](@ref), each frequency at its arrival time before
the coalescence ([`phenoma_arrival_delay`](@ref)), inverted by
[`phenoma_series`](@ref), anchored on the coalescence sample by the peak
of the two channels' quadrature sum, and added to the record. The
matched-filter SNRs of the injected samples against the channel PSD `psd`
are recorded per channel; `snr` and the label span follow the channel or
network selected by `settings.label_channel` ([`label_bounds`](@ref)).
`catalog` carries the schema of [`channel_catalog`](@ref). Draw order per
event: coalescence sample, total mass, mass ratio, distance, longitude,
latitude, inclination, polarisation.
"""
function inject_mbhb!(
    rng::AbstractRNG,
    channels::NTuple{2,AbstractVector{Float64}},
    labels::AbstractVector{Int32},
    snrs::AbstractVector{Float32},
    catalog::DataFrame,
    event_id::Integer,
    fs::Real,
    duration_sec::Real,
    settings::NamedTuple,
    psd,
    response::AbstractDetectorResponse,
)
    n_total = length(channels[1])
    (
        length(channels[2]) == n_total &&
        length(labels) == n_total &&
        length(snrs) == n_total
    ) || throw(
        DimensionMismatch(
            "the channels, labels, and snrs must have equal length; got " *
            "$(length.(channels)), $(length(labels)), $(length(snrs)).",
        ),
    )
    n_total >= 2 || throw(ArgumentError("the record must hold at least 2 samples."))
    duration_sec > 0 ||
        throw(ArgumentError("duration_sec = $duration_sec; must be positive."))

    k_c = rand(rng, round(Int, 0.1*n_total):round(Int, 0.9*n_total))
    k_c = clamp(k_c, 1, n_total)
    t_c = (k_c - 1) / fs
    log_mass_min = log10(settings.mbhb_total_mass_min)
    log_mass_max = log10(settings.mbhb_total_mass_max)
    total_mass = 10.0^(log_mass_min + rand(rng) * (log_mass_max - log_mass_min))
    mass_ratio = 1 + rand(rng) * (settings.mbhb_mass_ratio_max - 1)
    log_d_min = log10(settings.mbhb_distance_min_gpc)
    log_d_max = log10(settings.mbhb_distance_max_gpc)
    distance_gpc = 10.0^(log_d_min + rand(rng) * (log_d_max - log_d_min))
    source = draw_extrinsic(rng)

    spectrum = phenoma_spectrum(
        fs,
        duration_sec;
        total_mass = total_mass,
        mass_ratio = mass_ratio,
        nyquist_taper = settings.nyquist_taper,
    )
    scale =
        phenoma_physical_amplitude(spectrum.p; distance_sec = distance_gpc * GIGAPARSEC_SEC)
    delays = [phenoma_arrival_delay(f, spectrum.p) for f in spectrum.freqs]
    frame = source_frame(response, source.longitude, source.latitude, source.polarization)
    H_A, H_E = project_spectrum(
        response,
        frame,
        spectrum.freqs,
        scale .* spectrum.H,
        delays,
        t_c,
        source.inclination,
    )
    h_A = phenoma_series(H_A, spectrum.n, fs)
    h_E = phenoma_series(H_E, spectrum.n, fs)
    merger_index = argmax(h_A .^ 2 .+ h_E .^ 2)

    placed_A = zeros(n_total)
    placed_E = zeros(n_total)
    covered = place_signal!(placed_A, h_A, k_c, merger_index)
    place_signal!(placed_E, h_E, k_c, merger_index)
    isempty(covered) && return false
    channels[1][covered] .+= @view placed_A[covered]
    channels[2][covered] .+= @view placed_E[covered]
    ρ_A = matched_filter_snr(view(placed_A, covered), fs; psd = psd)
    ρ_E = matched_filter_snr(view(placed_E, covered), fs; psd = psd)
    if settings.label_channel == "A"
        ρ_label, series = ρ_A, (placed_A,)
    elseif settings.label_channel == "E"
        ρ_label, series = ρ_E, (placed_E,)
    else
        ρ_label, series = hypot(ρ_A, ρ_E), (placed_A, placed_E)
    end
    lbl_start, lbl_end = label_bounds(settings, series, covered, k_c, fs, psd)
    onset = signal_onset(settings, series, covered, k_c, lbl_start, lbl_end, fs, psd)
    if lbl_start <= lbl_end
        labels[lbl_start:lbl_end] .= 1
        snrs[lbl_start:lbl_end] .= Float32(ρ_label)
    end
    push!(
        catalog,
        (
            event_id,
            t_c,
            k_c,
            ρ_label,
            total_mass,
            mass_ratio,
            spectrum.p.η,
            spectrum.p.f_merg,
            spectrum.p.f_ring,
            spectrum.p.f_cut,
            first(covered),
            last(covered),
            lbl_start,
            lbl_end,
            onset,
            distance_gpc,
            source.longitude,
            source.latitude,
            source.inclination,
            source.polarization,
            ρ_A,
            ρ_E,
        ),
    )
    return true
end

"""
    simulate_record(rng, settings, t, psd, response) -> NamedTuple

Noise and sources of one record on the sample times `t` [s] for the
sensitivity `psd(f)`: with [`SkyAveragedResponse`](@ref) one strain
against the sensitivity itself;
with a constellation response two channels, A and E, each of independent
noise against [`channel_noise_psd`](@ref), the background sources and the
MBHB injections projected by the response. Returns `(; channels, labels,
snrs, catalog)` with `channels` a NamedTuple of record-length vectors.
"""
function simulate_record(
    rng::AbstractRNG,
    g::NamedTuple,
    t::AbstractVector{<:Real},
    psd,
    ::SkyAveragedResponse,
)
    n_total = length(t)
    T_record = n_total / g.fs
    # Instrument plus confusion noise at physical strain amplitude.
    strain = @timeit TIMER "noise" synthesize_noise(
        rng,
        n_total,
        g.fs;
        psd = psd,
        f_min = g.noise_f_min_hz,
    )
    # Resolvable sources above the confusion fit, scaled to a
    # matched-filter SNR over the simulated record.
    @info "injecting resolvable background sources" n_gbs = g.n_gbs n_emris = g.n_emris
    @timeit TIMER "galactic binaries" inject_galactic_binaries!(
        rng,
        strain,
        t,
        g.fs,
        g.n_gbs,
        (g.gb_snr_min, g.gb_snr_max),
        psd,
    )
    @timeit TIMER "EMRIs" inject_emris!(
        rng,
        strain,
        t,
        g.fs,
        g.n_emris,
        (g.emri_snr_min, g.emri_snr_max),
        psd,
    )
    # MBHB injections aligned on the coalescence sample, with labels and
    # an event catalogue.
    @info "injecting massive black hole binaries" n_mbhb = g.n_mbhb label_span =
        g.label_span
    labels = zeros(Int32, n_total)
    snrs = zeros(Float32, n_total)
    catalog = event_catalog()
    duration_sec = min(g.mbhb_duration_days * 86400, 0.4 * T_record)
    @timeit TIMER "MBHBs" for i in 1:g.n_mbhb
        inject_mbhb!(rng, strain, labels, snrs, catalog, i, g.fs, duration_sec, g, psd)
    end
    return (channels = (A = strain,), labels = labels, snrs = snrs, catalog = catalog)
end

function simulate_record(
    rng::AbstractRNG,
    g::NamedTuple,
    t::AbstractVector{<:Real},
    psd,
    response::AbstractDetectorResponse,
)
    n_total = length(t)
    T_record = n_total / g.fs
    psd_channel = channel_noise_psd(response, psd)
    # Independent noise per channel at the Michelson-channel level.
    A = @timeit TIMER "noise" synthesize_noise(
        rng,
        n_total,
        g.fs;
        psd = psd_channel,
        f_min = g.noise_f_min_hz,
    )
    E = @timeit TIMER "noise" synthesize_noise(
        rng,
        n_total,
        g.fs;
        psd = psd_channel,
        f_min = g.noise_f_min_hz,
    )
    channels = (A, E)
    @info "injecting resolvable background sources through the response" n_gbs = g.n_gbs n_emris =
        g.n_emris
    @timeit TIMER "galactic binaries" inject_galactic_binaries!(
        rng,
        channels,
        t,
        g.fs,
        g.n_gbs,
        (g.gb_snr_min, g.gb_snr_max),
        psd_channel,
        response,
    )
    @timeit TIMER "EMRIs" inject_emris!(
        rng,
        channels,
        t,
        g.fs,
        g.n_emris,
        (g.emri_snr_min, g.emri_snr_max),
        psd_channel,
        response,
    )
    @info "injecting massive black hole binaries at physical amplitude" n_mbhb = g.n_mbhb distance_range_gpc =
        (g.mbhb_distance_min_gpc, g.mbhb_distance_max_gpc) label_span = g.label_span label_channel =
        g.label_channel
    labels = zeros(Int32, n_total)
    snrs = zeros(Float32, n_total)
    catalog = channel_catalog()
    duration_sec = min(g.mbhb_duration_days * 86400, 0.4 * T_record)
    @timeit TIMER "MBHBs" for i in 1:g.n_mbhb
        inject_mbhb!(
            rng,
            channels,
            labels,
            snrs,
            catalog,
            i,
            g.fs,
            duration_sec,
            g,
            psd_channel,
            response,
        )
    end
    return (channels = (A = A, E = E), labels = labels, snrs = snrs, catalog = catalog)
end

"""
    write_record(h5_file, t, channels)

HDF5 record of the simulated channels under `obs/tdi`, laid out so that
the pre-processor's TDI recombination round-trips them: one channel `A`
is written as `X ≡ 0`, `Z = √2 A`; two channels `A`, `E` as
``X = -A/\\sqrt{2} + E/\\sqrt{6}``, ``Y = -2E/\\sqrt{6}``,
``Z = A/\\sqrt{2} + E/\\sqrt{6}``, whose null combination `T` vanishes
identically.
"""
function write_record(
    h5_file::AbstractString,
    t::AbstractVector{<:Real},
    channels::NamedTuple,
)
    HDF5.h5open(h5_file, "w") do file
        g_obs = HDF5.create_group(file, "obs")
        g_tdi = HDF5.create_group(g_obs, "tdi")
        g_tdi["t"] = collect(t)
        if length(channels) == 1
            g_tdi["X"] = zeros(length(t))
            g_tdi["Z"] = channels.A .* sqrt(2.0)
        else
            A, E = channels.A, channels.E
            g_tdi["X"] = -A ./ sqrt(2.0) .+ E ./ sqrt(6.0)
            g_tdi["Y"] = -2 .* E ./ sqrt(6.0)
            g_tdi["Z"] = A ./ sqrt(2.0) .+ E ./ sqrt(6.0)
        end
    end
    return h5_file
end

"""
    generate_telemetry(config; run_id = new_run_id(), output = nothing) -> NamedTuple

Generation stage of the pipeline. Every parameter comes from the
`[generation]` section of `config` ([`generation_settings`](@ref)); `output`
overrides the configured HDF5 path. The stage synthesises Gaussian noise of
the Robson–Cornish–Liu (2019) instrument-plus-confusion PSD
([`synthesize_noise`](@ref)), adds the resolvable background
([`inject_galactic_binaries!`](@ref), [`inject_emris!`](@ref)), injects
the MBHB events with their labels and catalogue ([`inject_mbhb!`](@ref)) —
through the detector response of `[generation] response`
([`simulate_record`](@ref)) — and persists the products beside `output`:

- `<stem>.h5` — group `obs/tdi` with datasets `t` [s] and the TDI
  combinations of [`write_record`](@ref): `X ≡ 0`, `Z = √2 h` for the
  sky-averaged strain `h`, or `X`, `Y`, `Z` recombining to the `A` and `E`
  channels of the constellation response (`response = "lisa"`, through the
  CurvatureDistinguishability extension);
- `<stem>_labels.csv` — point-wise `Label` (0/1) and `SNR` columns;
- `<stem>_events.csv` — the event catalogue ([`event_catalog`](@ref));
- `<stem>_generation.toml` — the `[generation]` snapshot with `run_id` and
  `n_events_injected`, plus the hardware and git provenance added by
  [`write_toml`](@ref).

Existing files are backed up DrWatson-style ([`backup_existing!`](@ref)).
Before allocation the memory demand of eight record-length arrays is
checked against the `[resources]` thresholds ([`check_memory`](@ref)).
The whole stage is timed under `TIMER` as `"generation"`.

Returns `(; run_id, h5_file, label_file, catalog_file, snapshot_file,
catalog, n_total, fs, strain, channels, labels)`, with `channels` the
NamedTuple of simulated channels (`A`, and `E` under the constellation
response), `strain::Vector{Float64}` the `A` channel the pipeline
analyses, and `labels::Vector{Int32}` the point-wise labels.
"""
function generate_telemetry(
    config::AbstractDict;
    run_id::AbstractString = new_run_id(),
    output::Union{Nothing,AbstractString} = nothing,
)
    @timeit TIMER "generation" begin
        isempty(run_id) && throw(ArgumentError("run_id must be a non-empty string."))
        g = generation_settings(config)
        response = detector_response(g)
        n_channels = channel_count(response)
        h5_file = output === nothing ? g.output : resolvepath(output)
        endswith(h5_file, ".h5") || throw(
            ArgumentError("output = $(repr(h5_file)); the HDF5 path must end in `.h5`."),
        )
        stem = first(splitext(h5_file))
        label_file = stem * "_labels.csv"
        catalog_file = stem * "_events.csv"
        snapshot_file = stem * "_generation.toml"

        n_total = round(Int, g.days * 24 * 3600 * g.fs)
        n_total >= 2 || throw(
            ArgumentError("days = $(g.days) at fs = $(g.fs) Hz yields $n_total samples."),
        )
        check_memory(
            record_memory_estimate_gib(n_total; copies = n_channels == 1 ? 8 : 14),
            resource_settings(config);
            stage = "generation",
        )
        t = (0:(n_total-1)) ./ g.fs
        rng = Xoshiro(g.seed)
        psd = f -> lisa_noise_psd(f; observation_years = g.observation_years)

        @info "generating continuous LISA telemetry" run_id n_total days = g.days fs = g.fs response =
            g.response channels = n_channels
        @info "noise model: Robson–Cornish–Liu (2019)" seed = g.seed confusion_fit_years =
            g.observation_years
        if n_channels == 1
            @info "MBHB population (IMRPhenomA)" snr_range = (g.snr_min, g.snr_max) total_mass_range =
                (g.mbhb_total_mass_min, g.mbhb_total_mass_max) mass_ratio_max =
                g.mbhb_mass_ratio_max
        else
            @info "MBHB population (IMRPhenomA through the constellation response)" distance_range_gpc =
                (g.mbhb_distance_min_gpc, g.mbhb_distance_max_gpc) total_mass_range =
                (g.mbhb_total_mass_min, g.mbhb_total_mass_max) mass_ratio_max =
                g.mbhb_mass_ratio_max
        end

        record = simulate_record(rng, g, t, psd, response)
        channels, labels, snrs, catalog =
            record.channels, record.labels, record.snrs, record.catalog

        # Persist: HDF5 channels, point-wise labels, event catalogue,
        # provenance snapshot.
        @timeit TIMER "persist" begin
            backup_existing!(h5_file)
            mkpath(dirname(h5_file))
            write_record(h5_file, t, channels)
            write_csv(label_file, DataFrame(Label = labels, SNR = snrs))
            write_csv(catalog_file, catalog)
            snapshot = Dict{String,Any}(String(k) => getfield(g, k) for k in keys(g))
            snapshot["output"] = provenance_path(h5_file)
            snapshot["run_id"] = run_id
            snapshot["n_events_injected"] = nrow(catalog)
            snapshot["channels"] = collect(String.(keys(channels)))
            write_toml(snapshot_file, Dict{String,Any}("generation" => snapshot))
        end
        @info "telemetry generated" run_id h5_file label_file catalog_file snapshot_file n_events_injected =
            nrow(catalog)

        (
            run_id = String(run_id),
            h5_file = h5_file,
            label_file = label_file,
            catalog_file = catalog_file,
            snapshot_file = snapshot_file,
            catalog = catalog,
            n_total = n_total,
            fs = g.fs,
            strain = channels.A,
            channels = channels,
            labels = labels,
        )
    end
end
