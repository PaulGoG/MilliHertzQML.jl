# src/stages/generation.jl — telemetry generation stage: calibrated LISA
# noise (instrument plus confusion), resolvable galactic binaries and EMRIs
# scaled to a matched-filter SNR over the record, MBHB injections aligned on
# the coalescence sample with point-wise labels and an event catalog, and
# the persisted products (HDF5 record, label CSV, catalog CSV, snapshot
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

Empty MBHB event catalog with the column schema filled by
[`inject_mbhb!`](@ref): event identifier, coalescence time [s] and sample,
matched-filter SNR of the injected samples, detector-frame total mass
[M⊙], mass ratio, symmetric mass ratio, IMRPhenomA merger, ringdown, and
cut-off frequencies [Hz], injected sample span, and positive-label span.
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

    if settings.label_span == "detectable"
        placed[covered] .= segment
        span = detectable_span(
            placed,
            covered,
            fs,
            settings.label_window_size,
            settings.label_snr_threshold;
            step = settings.label_step,
            psd = psd,
        )
        span === nothing && (span = (k_c+1):k_c)   # no window reaches the threshold
        lbl_start, lbl_end = first(span), last(span)
    elseif settings.label_span == "injection"
        lbl_start, lbl_end = first(covered), last(covered)
    elseif settings.label_span == "fixed"
        lbl_start = clamp(k_c - round(Int, settings.label_before_sec * fs), 1, n_total)
        lbl_end = clamp(k_c + round(Int, settings.label_after_sec * fs), 1, n_total)
    else
        throw(
            ArgumentError("label_span = $(repr(settings.label_span)); unknown criterion."),
        )
    end
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
        ),
    )
    return true
end

"""
    generate_telemetry(config; run_id = new_run_id(), output = nothing) -> NamedTuple

Generation stage of the pipeline. Every parameter comes from the
`[generation]` section of `config` ([`generation_settings`](@ref)); `output`
overrides the configured HDF5 path. The stage synthesizes Gaussian noise of
the Robson–Cornish–Liu (2019) instrument-plus-confusion PSD
([`synthesize_noise`](@ref)), adds the resolvable background
([`inject_galactic_binaries!`](@ref), [`inject_emris!`](@ref)), injects
the MBHB events with their labels and catalog ([`inject_mbhb!`](@ref)),
and persists the products beside `output`:

- `<stem>.h5` — group `obs/tdi` with datasets `t` [s], `X` (zero), and
  `Z = √2 h` so that the pre-processor's A-channel recombination
  round-trips the simulated strain `h`;
- `<stem>_labels.csv` — point-wise `Label` (0/1) and `SNR` columns;
- `<stem>_events.csv` — the event catalog ([`event_catalog`](@ref));
- `<stem>_generation.toml` — the `[generation]` snapshot with `run_id` and
  `n_events_injected`, plus the hardware and git provenance added by
  [`write_toml`](@ref).

Existing files are backed up DrWatson-style ([`backup_existing!`](@ref)).
Before allocation the memory demand of eight record-length arrays is
checked against the `[resources]` thresholds ([`check_memory`](@ref)).
The whole stage is timed under `TIMER` as `"generation"`.

Returns `(; run_id, h5_file, label_file, catalog_file, snapshot_file,
catalog, n_total, fs, strain, labels)`, with `strain::Vector{Float64}` the
simulated strain and `labels::Vector{Int32}` the point-wise labels.
"""
function generate_telemetry(
    config::AbstractDict;
    run_id::AbstractString = new_run_id(),
    output::Union{Nothing,AbstractString} = nothing,
)
    @timeit TIMER "generation" begin
        isempty(run_id) && throw(ArgumentError("run_id must be a non-empty string."))
        g = generation_settings(config)
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
            record_memory_estimate_gib(n_total; copies = 8),
            resource_settings(config);
            stage = "generation",
        )
        t = (0:(n_total-1)) ./ g.fs
        T_record = n_total / g.fs
        rng = Xoshiro(g.seed)
        psd = f -> lisa_noise_psd(f; observation_years = g.observation_years)

        @info "generating continuous LISA telemetry" run_id n_total days = g.days fs = g.fs
        @info "noise model: Robson–Cornish–Liu (2019)" seed = g.seed confusion_fit_years =
            g.observation_years
        @info "MBHB population (IMRPhenomA)" snr_range = (g.snr_min, g.snr_max) total_mass_range =
            (g.mbhb_total_mass_min, g.mbhb_total_mass_max) mass_ratio_max =
            g.mbhb_mass_ratio_max

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
        # an event catalog.
        @info "injecting massive black hole binaries" n_mbhb = g.n_mbhb label_span =
            g.label_span
        labels = zeros(Int32, n_total)
        snrs = zeros(Float32, n_total)
        catalog = event_catalog()
        duration_sec = min(g.mbhb_duration_days * 86400, 0.4 * T_record)
        @timeit TIMER "MBHBs" for i in 1:g.n_mbhb
            inject_mbhb!(rng, strain, labels, snrs, catalog, i, g.fs, duration_sec, g, psd)
        end

        # Persist: HDF5 strain (X ≡ 0, Z = √2 h so that the pre-processor
        # round-trips), point-wise labels, event catalog, provenance snapshot.
        @timeit TIMER "persist" begin
            backup_existing!(h5_file)
            mkpath(dirname(h5_file))
            HDF5.h5open(h5_file, "w") do file
                g_obs = HDF5.create_group(file, "obs")
                g_tdi = HDF5.create_group(g_obs, "tdi")
                g_tdi["t"] = collect(t)
                g_tdi["X"] = zeros(n_total)
                g_tdi["Z"] = strain .* sqrt(2.0)
            end
            write_csv(label_file, DataFrame(Label = labels, SNR = snrs))
            write_csv(catalog_file, catalog)
            snapshot = Dict{String,Any}(String(k) => getfield(g, k) for k in keys(g))
            snapshot["output"] = provenance_path(h5_file)
            snapshot["run_id"] = run_id
            snapshot["n_events_injected"] = nrow(catalog)
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
            strain = strain,
            labels = labels,
        )
    end
end
