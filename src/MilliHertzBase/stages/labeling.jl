# Labelling stage: point-wise MBHB labels of an LDC
# product from its signal-only truth stream. The A channel of the truth
# stream is scanned with the windowed matched-filter SNR against the
# analytic TDI PSD; mergers come from the source catalogue when the product
# carries one and from SNR peaks otherwise; the positive span of every
# merger is either the fixed window of Isfan et al. (2025) or the union of
# windows in which the source is detectable; with fixed spans the event
# table also records the signal onset from which alerts are credited.

"""
    read_truth_csv(path) -> NamedTuple

Signal-only TDI `(t, X, Y, Z, dt)` from a CSV with columns `t`, `X`, `Y`,
`Z` (the blind-set truth of an LDC release); the sampling step is the
difference of the first two time samples.
"""
function read_truth_csv(path::AbstractString)
    isfile(path) || throw(ArgumentError("truth CSV not found: $path"))
    table = CSV.read(path, DataFrame)
    for c in (:t, :X, :Y, :Z)
        c in propertynames(table) || throw(ArgumentError("$path lacks the column $c."))
    end
    nrow(table) >= 2 || throw(ArgumentError("$path holds fewer than two samples."))
    dt = Float64(table.t[2] - table.t[1])
    dt > 0 || throw(ArgumentError("non-positive sampling step $dt in $path."))
    return (
        t = Float64.(table.t),
        X = Float64.(table.X),
        Y = Float64.(table.Y),
        Z = Float64.(table.Z),
        dt = dt,
    )
end

"""
    merger_indices_from_catalog(catalog, t0, fs, n) -> Vector{Int}

Sample indices of the coalescence times of `catalog` (column
`CoalescenceTime` [s]) in a record of `n` samples starting at `t0` [s] and
sampled at `fs` [Hz], clipped to `1:n`.
"""
function merger_indices_from_catalog(catalog::DataFrame, t0::Real, fs::Real, n::Integer)
    "CoalescenceTime" in DataFrames.names(catalog) ||
        throw(ArgumentError("the catalog lacks the column CoalescenceTime."))
    return [clamp(round(Int, (tc - t0) * fs) + 1, 1, n) for tc in catalog.CoalescenceTime]
end

"""
    merger_indices_from_peaks(A, starts, snr, settings, fs) -> Vector{Int}

Sample indices of the mergers of the truth-stream channel `A` located
without a catalogue: the SNR peaks ([`snr_peaks`](@ref)) of the windowed
scan (`starts`, `snr`) at or above `settings.merger_snr_threshold`,
separated by at least `settings.peak_min_separation_sec`, with inspiral
precursors within `settings.label_before_sec` below
`settings.precursor_ratio` of a larger peak discarded; the merger is the
sample of largest `|A|` inside each peak window.
"""
function merger_indices_from_peaks(
    A::AbstractVector{<:Real},
    starts::AbstractRange{<:Integer},
    snr::AbstractVector{<:Real},
    settings::NamedTuple,
    fs::Real,
)
    peaks = snr_peaks(
        starts,
        snr;
        threshold = settings.merger_snr_threshold,
        min_separation = round(Int, settings.peak_min_separation_sec * fs),
        precursor_window = round(Int, settings.label_before_sec * fs),
        precursor_ratio = settings.precursor_ratio,
    )
    window_size = settings.label_window_size
    return [
        starts[p] - 1 + argmax(abs.(view(A, starts[p]:(starts[p]+window_size-1)))) for
        p in peaks
    ]
end

"""
    span_peak_snr(spans, starts, snr; window_size, step) -> Vector{Float64}

Largest windowed SNR among the windows (of `window_size` samples at the
starts `starts`, stride `step`) overlapping each sample range of `spans`;
0 for a span no window overlaps.
"""
function span_peak_snr(
    spans::AbstractVector{<:AbstractUnitRange{<:Integer}},
    starts::AbstractRange{<:Integer},
    snr::AbstractVector{<:Real};
    window_size::Integer,
    step::Integer,
)
    peak = Vector{Float64}(undef, length(spans))
    for (k, span) in enumerate(spans)
        first_w = max(1, cld(first(span) - window_size, step) + 1)
        last_w = min(length(starts), div(last(span) - 1, step) + 1)
        peak[k] = first_w <= last_w ? maximum(view(snr, first_w:last_w)) : 0.0
    end
    return peak
end

"""
    label_truth_stream(config; h5_file = nothing, truth_csv = nothing,
                       output_prefix = nothing) -> NamedTuple

Labelling stage: the signal-only TDI of an LDC product — the compound
dataset `truth_group` of `h5_file` with the catalogue `catalog_group`
([`read_tdi`](@ref), [`read_catalog`](@ref)), or the columns `t, X, Y, Z`
of `truth_csv` ([`read_truth_csv`](@ref)) — is combined into the A channel
([`tdi_to_aet`](@ref)) and scanned with the windowed matched-filter SNR
([`windowed_snr`](@ref)) against the analytic TDI PSD
([`ldc_tdi_psd`](@ref)) of `psd_model`. Mergers are the catalogue's
coalescence times when a catalogue is available and SNR peaks otherwise
([`merger_indices_from_peaks`](@ref)); their positive spans are the fixed
window `[-label_before_sec, +label_after_sec]` ([`fixed_spans`](@ref)) or
the union of windows reaching `label_snr_threshold`
([`detectable_spans`](@ref)) according to `label_span`. Every parameter
comes from the `[ldc]`, `[paths]`, and `[resources]` sections of `config`
([`ldc_settings`](@ref)); `h5_file` and `truth_csv` are mutually exclusive
and replace the configured `h5_file`.

Writes, under the `inputs` root, `<output_prefix>_labels.csv` (`Label`,
`SNR`: the peak windowed SNR of the event a sample belongs to),
`<output_prefix>_events.csv` (one row per merger with its sample index,
time, window SNR, catalogue parameters, and — for fixed spans — the label
range, its peak SNR, and the signal onset `signal_start_index`, the first
window inside the span and past the preceding event's span that reaches
`label_snr_threshold` ([`signal_onsets`](@ref)), from which alerts are
credited), `<output_prefix>_spans.csv`, and the snapshot
`<output_prefix>_labels.toml` with the labelling parameters and the
provenance sections of [`write_toml`](@ref). Existing files are backed up
first.

Returns `(label_path, events_path, spans_path, snapshot_path, events,
n_label_runs, positive_fraction)` with `events` the event table.
"""
function label_truth_stream(
    config::AbstractDict;
    h5_file::Union{Nothing,AbstractString} = nothing,
    truth_csv::Union{Nothing,AbstractString} = nothing,
    output_prefix::Union{Nothing,AbstractString} = nothing,
)
    @timeit TIMER "labeling" begin
        settings = ldc_settings(config)
        resources = resource_settings(config)
        prefix = String(override(output_prefix, settings.output_prefix))
        (h5_file === nothing || truth_csv === nothing) ||
            throw(ArgumentError("h5_file and truth_csv are mutually exclusive."))
        if h5_file === nothing && truth_csv === nothing
            isempty(settings.h5_file) && throw(
                ArgumentError(
                    "no truth source: pass h5_file or truth_csv, or set [ldc] h5_file.",
                ),
            )
            h5_file = settings.h5_file
        end

        # Truth stream and catalogue
        local truth, catalog, source
        if h5_file !== nothing
            source = resolvepath(h5_file)
            truth = read_tdi(source; group = settings.truth_group)
            catalog = read_catalog(source; group = settings.catalog_group)
        elseif truth_csv !== nothing
            source = resolvepath(truth_csv)
            truth = read_truth_csv(source)
            catalog = nothing
        else
            throw(ArgumentError("no truth source: pass h5_file or truth_csv."))
        end
        n = length(truth.t)
        fs = 1 / truth.dt
        t0 = truth.t[1]
        check_memory(record_memory_estimate_gib(n), resources; stage = "labeling")
        A, _, _ = tdi_to_aet(truth.X, truth.Y, truth.Z)
        @info "truth stream" source n_samples = n sample_rate_hz = fs t0 max_abs_A =
            maximum(abs, A)

        # Windowed matched-filter SNR against the analytic TDI PSD
        window_size = settings.label_window_size
        step = settings.label_step
        psd_model = settings.psd_model
        tdi2 = settings.tdi2
        observation_years = settings.observation_years
        psd =
            f -> ldc_tdi_psd(
                f;
                channel = :A,
                model = psd_model,
                tdi2 = tdi2,
                observation_years = observation_years,
            )
        @info "windowed matched-filter SNR" window_size step psd_model tdi2 observation_years
        starts, snr = windowed_snr(A, fs; window_size = window_size, step = step, psd = psd)

        # Mergers: from the catalogue when available, else from SNR peaks
        merger_indices = if catalog !== nothing
            merger_indices_from_catalog(catalog, t0, fs, n)
        else
            merger_indices_from_peaks(A, starts, snr, settings, fs)
        end
        order = sortperm(merger_indices)
        merger_indices = merger_indices[order]
        catalog !== nothing && (catalog = catalog[order, :])
        isempty(merger_indices) && @warn "no merger found; the label column is all zeros."

        spans = if settings.label_span == "fixed"
            fixed_spans(
                merger_indices,
                fs,
                n;
                before = settings.label_before_sec,
                after = settings.label_after_sec,
            )
        else
            detectable_spans(starts, snr, window_size; threshold = settings.label_snr_threshold)
        end
        labels = span_labels(n, spans)
        peak_snr = span_peak_snr(spans, starts, snr; window_size = window_size, step = step)
        snr_column = zeros(Float32, n)
        for (span, ρ) in zip(spans, peak_snr)
            snr_column[span] .= max.(snr_column[span], Float32(ρ))
        end
        n_label_runs = length(contiguous_runs(labels .== 1))
        positive_fraction = count(==(1), labels) / n

        # Persist
        out_dir = pipeline_paths(config).inputs
        label_path = joinpath(out_dir, "$(prefix)_labels.csv")
        events_path = joinpath(out_dir, "$(prefix)_events.csv")
        spans_path = joinpath(out_dir, "$(prefix)_spans.csv")
        snapshot_path = joinpath(out_dir, "$(prefix)_labels.toml")
        for out in (label_path, events_path, spans_path, snapshot_path)
            abspath(out) == abspath(source) && throw(
                ArgumentError(
                    "output $out coincides with the input; choose another prefix.",
                ),
            )
        end
        write_csv(label_path, DataFrame(Label = labels, SNR = snr_column))

        events = DataFrame(
            event = 1:length(merger_indices),
            merger_index = merger_indices,
            merger_time_s = t0 .+ (merger_indices .- 1) ./ fs,
            merger_window_snr = [
                snr[clamp(div(i - 1, step) + 1, 1, length(snr))] for i in merger_indices
            ],
        )
        if catalog !== nothing
            for c in ("CoalescenceTime", "Mass1", "Mass2", "Redshift", "Distance")
                c in DataFrames.names(catalog) && (events[!, c] = catalog[!, c])
            end
        end
        if settings.label_span == "fixed"
            events.label_start_index = first.(spans)
            events.label_end_index = last.(spans)
            events.label_peak_snr = peak_snr
            # The onset search starts at the span start, or past the preceding
            # event's span when that ends later: the 96-h spans of mergers a
            # day apart overlap, and the earlier source must not supply the
            # onset of the later one.
            lower = [
                min(m, max(first(spans[i]), i == 1 ? 1 : last(spans[i-1]) + 1)) for
                (i, m) in enumerate(merger_indices)
            ]
            events.signal_start_index = signal_onsets(
                starts,
                snr,
                merger_indices,
                lower;
                threshold = settings.label_snr_threshold,
            )
        end
        write_csv(events_path, events)
        write_csv(
            spans_path,
            DataFrame(
                span = 1:length(spans),
                start_index = first.(spans),
                end_index = last.(spans),
                peak_snr = peak_snr,
            ),
        )
        write_toml(
            snapshot_path,
            Dict{String,Any}(
                "labels" => Dict{String,Any}(
                    "source" => provenance_path(source),
                    "truth_group" => settings.truth_group,
                    "psd_model" => psd_model,
                    "tdi2" => tdi2,
                    "observation_years" => observation_years,
                    "label_span" => settings.label_span,
                    "label_before_sec" => settings.label_before_sec,
                    "label_after_sec" => settings.label_after_sec,
                    "label_snr_threshold" => settings.label_snr_threshold,
                    "signal_onset" =>
                        settings.label_span == "fixed" ?
                        "first window reaching label_snr_threshold inside the span and past the preceding span" :
                        "",
                    "merger_snr_threshold" => settings.merger_snr_threshold,
                    "precursor_ratio" => settings.precursor_ratio,
                    "peak_min_separation_sec" => settings.peak_min_separation_sec,
                    "label_window_size" => window_size,
                    "label_step" => step,
                    "n_samples" => n,
                    "n_events" => length(merger_indices),
                    "n_label_runs" => n_label_runs,
                    "positive_fraction" => positive_fraction,
                ),
            );
            tag = true,
        )
        @info "labels written" label_path n_events = length(merger_indices) n_label_runs positive_fraction =
            round(positive_fraction; digits = 4)

        (
            label_path = label_path,
            events_path = events_path,
            spans_path = spans_path,
            snapshot_path = snapshot_path,
            events = events,
            n_label_runs = n_label_runs,
            positive_fraction = positive_fraction,
        )
    end
end
