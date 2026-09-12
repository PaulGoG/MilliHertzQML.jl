# src/telemetry.jl — consumer side of the telemetry coupling: the run
# interface a producer adapter implements, batch-to-row geometry, the
# coverage set of delivered samples, the window scheduler, the streaming
# detector (record-context whitening of a complete window), the replay of
# an arrival feed into scored windows, and the alert-latency table. The
# DeepSpaceTelemetry adapter lives in the package extension
# MilliHertzQMLDeepSpaceTelemetryExt; nothing here writes into a run
# directory.

"""
    RunGeometry

Sampling and batching geometry of a telemetry run: `sample_rate` [Hz],
`segment_duration_sec`, `batch_size` (segments per batch),
`points_per_batch`, the mission epoch `start_sim_time`, the producer's
`package_version`, and `payload_rows`, the number of rows of the external
payload the producer ingested (0 when unknown); the producer zero-pads its
stream beyond the payload, and windows past that bound are not scored.
"""
struct RunGeometry
    sample_rate::Float64
    segment_duration_sec::Float64
    batch_size::Int
    points_per_batch::Int
    start_sim_time::Dates.DateTime
    package_version::String
    payload_rows::Int
    function RunGeometry(
        sample_rate::Real,
        segment_duration_sec::Real,
        batch_size::Integer,
        start_sim_time::Dates.DateTime,
        package_version::AbstractString = "unknown";
        payload_rows::Integer = 0,
    )
        payload_rows >= 0 || throw(ArgumentError("payload_rows must be non-negative."))
        sample_rate > 0 ||
            throw(ArgumentError("sample_rate = $sample_rate; must be positive."))
        segment_duration_sec > 0 ||
            throw(ArgumentError("segment_duration_sec must be positive."))
        batch_size >= 1 ||
            throw(ArgumentError("batch_size = $batch_size; must be at least 1."))
        points = sample_rate * segment_duration_sec * batch_size
        isinteger(points) || throw(
            ArgumentError(
                "sample_rate × segment_duration_sec × batch_size = $points is not an integer.",
            ),
        )
        return new(
            Float64(sample_rate),
            Float64(segment_duration_sec),
            Int(batch_size),
            Int(points),
            start_sim_time,
            String(package_version),
            Int(payload_rows),
        )
    end
end

"""
    BatchRecord

One telemetry batch as seen by the consumer: `name` (`LIVE_batch_<k>` or
`ARCH_batch_<k>`), global 1-based `index`, `live` flag, `rows` covered in
the payload, `content_epoch` of its first sample, and `state`
(`:ground`, `:lost`, or `:pruned`).
"""
struct BatchRecord
    name::String
    index::Int
    live::Bool
    rows::UnitRange{Int}
    content_epoch::Dates.DateTime
    state::Symbol
end

"""
    ArrivalEvent

One row of the producer's arrival feed: the mission `sim_time`, the `batch`
name, the `event` (`:ingested`, `:retry`, `:lost`, `:pruned`, or `:other`
for values unknown to this version), and the `attempt` counter.
"""
struct ArrivalEvent
    sim_time::Dates.DateTime
    batch::String
    event::Symbol
    attempt::Int
end

"""
    parse_batch_name(name) -> (index, live)

Global batch index and live flag of a batch directory name
`LIVE_batch_<k>` or `ARCH_batch_<k>`; `ArgumentError` for any other form.
"""
function parse_batch_name(name::AbstractString)
    m = match(r"^(LIVE|ARCH)_batch_(\d+)$", name)
    m === nothing && throw(
        ArgumentError("batch name $(repr(name)) is not LIVE_batch_<k> or ARCH_batch_<k>."),
    )
    kind = m[1]
    digits = m[2]
    (kind === nothing || digits === nothing) &&
        throw(ArgumentError("batch name $(repr(name)) has no index."))
    return parse(Int, digits), kind == "LIVE"
end

"""
    batch_rows(k, points_per_batch) -> UnitRange{Int}

Payload rows `[(k − 1) P + 1, k P]` of the global batch index `k` with `P`
samples per batch (row-index exact in the producer's external mode).
"""
function batch_rows(k::Integer, points_per_batch::Integer)
    (k >= 1 && points_per_batch >= 1) ||
        throw(ArgumentError("batch index and points per batch must be positive."))
    return ((k-1)*points_per_batch+1):(k*points_per_batch)
end

"""
    row_time(geometry, row) -> DateTime

Mission time of payload row `row`: `start_sim_time + (row − 1) / sample_rate`.
"""
function row_time(geometry::RunGeometry, row::Integer)
    row >= 1 || throw(ArgumentError("row = $row; rows are 1-based."))
    return geometry.start_sim_time +
           Dates.Millisecond(round(Int, 1000 * (row - 1) / geometry.sample_rate))
end

"""
    event_symbol(value) -> Symbol

`:ingested`, `:retry`, `:lost`, or `:pruned` for the producer's event
values, `:other` for anything else (tolerated, ignored).
"""
function event_symbol(value::AbstractString)
    v = lowercase(strip(value))
    v == "ingested" && return :ingested
    v == "retry" && return :retry
    v == "lost" && return :lost
    v == "pruned" && return :pruned
    return :other
end

# --- Run interface -----------------------------------------------------

"""
    AbstractTelemetryRun

A producer run as seen by the consumer. An adapter implements
[`run_geometry`](@ref), [`list_batches`](@ref), [`read_batch`](@ref),
[`arrival_events`](@ref), and [`run_state`](@ref); the consumer never writes
into the run.
"""
abstract type AbstractTelemetryRun end

"""
    run_geometry(run) -> RunGeometry

Sampling and batching geometry of `run`.
"""
function run_geometry end

"""
    list_batches(run) -> Vector{BatchRecord}

Every batch known to the run (delivered, lost, or pruned), sorted by index.
"""
function list_batches end

"""
    read_batch(run, name) -> Vector{Float32}

Payload samples of the delivered batch `name`, its segments concatenated in
ascending segment id.
"""
function read_batch end

"""
    arrival_events(run) -> Vector{ArrivalEvent}

The arrival feed in mission-time order.
"""
function arrival_events end

"""
    run_state(run) -> Symbol

`:active`, `:complete`, or `:aborted` from the run's lifecycle sentinels.
"""
function run_state end

"""
    MemoryTelemetryRun(geometry, payload, events; lost = String[])

In-memory run used by the tests: the whole `payload`, an arrival feed
`events`, and the names of batches declared lost. Batch `k` holds payload
rows [`batch_rows`](@ref)`(k, P)`; every batch of the payload exists.
"""
struct MemoryTelemetryRun <: AbstractTelemetryRun
    geometry::RunGeometry
    payload::Vector{Float32}
    events::Vector{ArrivalEvent}
    lost::Set{String}
    function MemoryTelemetryRun(
        geometry::RunGeometry,
        payload::AbstractVector{<:Real},
        events::AbstractVector{ArrivalEvent};
        lost = String[],
    )
        length(payload) >= geometry.points_per_batch ||
            throw(ArgumentError("the payload holds fewer samples than one batch."))
        bounded = RunGeometry(
            geometry.sample_rate,
            geometry.segment_duration_sec,
            geometry.batch_size,
            geometry.start_sim_time,
            geometry.package_version;
            payload_rows = length(payload),
        )
        return new(bounded, Vector{Float32}(payload), collect(events), Set(String.(lost)))
    end
end

run_geometry(run::MemoryTelemetryRun) = run.geometry

function list_batches(run::MemoryTelemetryRun)
    P = run.geometry.points_per_batch
    n = div(length(run.payload), P)
    return [
        BatchRecord(
            "LIVE_batch_$k",
            k,
            true,
            batch_rows(k, P),
            row_time(run.geometry, (k - 1) * P + 1),
            "LIVE_batch_$k" in run.lost ? :lost : :ground,
        ) for k in 1:n
    ]
end

function read_batch(run::MemoryTelemetryRun, name::AbstractString)
    k, _ = parse_batch_name(name)
    rows = batch_rows(k, run.geometry.points_per_batch)
    last(rows) <= length(run.payload) ||
        throw(ArgumentError("batch $name lies beyond the payload."))
    return run.payload[rows]
end

arrival_events(run::MemoryTelemetryRun) = run.events

run_state(::MemoryTelemetryRun) = :complete

# --- Coverage ----------------------------------------------------------

"""
    Coverage()

Set of delivered payload rows as sorted, disjoint `UnitRange{Int}`
intervals, updated by [`add!`](@ref) and [`remove!`](@ref); the order of
arrivals is irrelevant to the resulting set.
"""
mutable struct Coverage
    intervals::Vector{UnitRange{Int}}
    Coverage() = new(UnitRange{Int}[])
end

"""
    add!(coverage, rows) -> Coverage

Mark `rows` as delivered, merging adjacent and overlapping intervals.
"""
function add!(coverage::Coverage, rows::AbstractUnitRange{<:Integer})
    isempty(rows) && return coverage
    lo, hi = Int(first(rows)), Int(last(rows))
    kept = UnitRange{Int}[]
    for r in coverage.intervals
        if last(r) < lo - 1 || first(r) > hi + 1
            push!(kept, r)
        else
            lo = min(lo, first(r))
            hi = max(hi, last(r))
        end
    end
    push!(kept, lo:hi)
    sort!(kept; by = first)
    coverage.intervals = kept
    return coverage
end

"""
    remove!(coverage, rows) -> Coverage

Mark `rows` as no longer available (a pruned payload).
"""
function remove!(coverage::Coverage, rows::AbstractUnitRange{<:Integer})
    isempty(rows) && return coverage
    lo, hi = Int(first(rows)), Int(last(rows))
    kept = UnitRange{Int}[]
    for r in coverage.intervals
        if last(r) < lo || first(r) > hi
            push!(kept, r)
        else
            first(r) < lo && push!(kept, first(r):(lo-1))
            last(r) > hi && push!(kept, (hi+1):last(r))
        end
    end
    coverage.intervals = kept
    return coverage
end

"""
    covered_fraction(coverage, rows) -> Float64

Fraction of `rows` that are delivered.
"""
function covered_fraction(coverage::Coverage, rows::AbstractUnitRange{<:Integer})
    isempty(rows) && return 1.0
    covered = 0
    for r in coverage.intervals
        lo = max(first(r), first(rows))
        hi = min(last(r), last(rows))
        hi >= lo && (covered += hi - lo + 1)
    end
    return covered / length(rows)
end

"""
    holes(coverage, rows) -> Vector{UnitRange{Int}}

Sub-intervals of `rows` that are not delivered, in order.
"""
function holes(coverage::Coverage, rows::AbstractUnitRange{<:Integer})
    isempty(rows) && return UnitRange{Int}[]
    gaps = UnitRange{Int}[]
    cursor = Int(first(rows))
    for r in coverage.intervals
        last(r) < cursor && continue
        first(r) > last(rows) && break
        first(r) > cursor && push!(gaps, cursor:(first(r)-1))
        cursor = max(cursor, last(r) + 1)
    end
    cursor <= last(rows) && push!(gaps, cursor:Int(last(rows)))
    return gaps
end

"""
    covered_stretch(coverage, rows) -> UnitRange{Int}

The delivered interval containing every row of `rows`; empty when the rows
are not fully delivered.
"""
function covered_stretch(coverage::Coverage, rows::AbstractUnitRange{<:Integer})
    for r in coverage.intervals
        first(r) <= first(rows) && last(rows) <= last(r) && return r
    end
    return 1:0
end

# --- Window scheduling -------------------------------------------------

"""
    WindowScheduler(window_size, step_size; min_coverage = 1.0)

Bookkeeping of the sliding windows (window `m` covers rows
`[1 + (m − 1) S, (m − 1) S + W]`) that have become evaluable: a window is
evaluable once at least `min_coverage` of its rows are delivered, and each
window is emitted once.
"""
mutable struct WindowScheduler
    window_size::Int
    step_size::Int
    min_coverage::Float64
    emitted::Set{Int}
    function WindowScheduler(
        window_size::Integer,
        step_size::Integer;
        min_coverage::Real = 1.0,
    )
        window_size >= 2 || throw(ArgumentError("window_size must be at least 2."))
        1 <= step_size <= window_size ||
            throw(ArgumentError("step_size must lie in [1, window_size]."))
        0 < min_coverage <= 1 || throw(ArgumentError("min_coverage must lie in (0, 1]."))
        return new(Int(window_size), Int(step_size), Float64(min_coverage), Set{Int}())
    end
end

"""
    window_rows(scheduler, m) -> UnitRange{Int}

Payload rows of window `m`.
"""
function window_rows(scheduler::WindowScheduler, m::Integer)
    m >= 1 || throw(ArgumentError("window index $m; windows are 1-based."))
    lo = 1 + (m - 1) * scheduler.step_size
    return lo:(lo+scheduler.window_size-1)
end

"""
    windows_touching(scheduler, rows) -> UnitRange{Int}

Indices of the windows that intersect `rows`.
"""
function windows_touching(scheduler::WindowScheduler, rows::AbstractUnitRange{<:Integer})
    isempty(rows) && return 1:0
    W, S = scheduler.window_size, scheduler.step_size
    # The first window whose last row reaches first(rows): (m − 1) S + W ≥ first(rows)
    first_m = max(1, cld(Int(first(rows)) - W, S) + 1)
    # The last window whose first row is at most last(rows): 1 + (m − 1) S ≤ last(rows)
    last_m = div(Int(last(rows)) - 1, S) + 1
    return first_m:last_m
end

"""
    newly_evaluable!(scheduler, coverage, rows) -> Vector{Int}

Windows touching `rows` that are now evaluable under `coverage` and have
not been emitted before; they are recorded as emitted.
"""
function newly_evaluable!(
    scheduler::WindowScheduler,
    coverage::Coverage,
    rows::AbstractUnitRange{<:Integer},
)
    ready = Int[]
    for m in windows_touching(scheduler, rows)
        m in scheduler.emitted && continue
        if covered_fraction(coverage, window_rows(scheduler, m)) >= scheduler.min_coverage
            push!(scheduler.emitted, m)
            push!(ready, m)
        end
    end
    return ready
end

# --- Streaming detector ------------------------------------------------

"""
    StreamingDetector(model, scaler, threshold; sample_rate, window_size, step_size,
                      psd = nothing, highpass_cutoff_hz = 5e-4, highpass_order = 8,
                      low_band = (1e-3, 5e-3), high_band = (5e-3, 1e-1),
                      band_edges = [1e-3, 5e-3, 1e-1], feature_set = :whitened,
                      context_windows = 4)

Scoring of complete windows out of a partially delivered record with the
same conditioning as the batch pipeline. A window is cut from the delivered
stretch around it — up to `context_windows` window lengths on each side —
which is high-passed and, when `psd` is given, whitened as a whole
([`highpass_record`](@ref), [`whiten_record`](@ref)); the window's features
([`extract_features`](@ref)) are then encoded with the persisted `scaler`
and scored with `model`. Isolated windows without context are scored on
their own samples (edge-affected, as the first windows of any record).
"""
struct StreamingDetector
    model::VariationalQuantumClassifier
    scaler::FeatureScaler
    threshold::Float32
    sample_rate::Float64
    window_size::Int
    step_size::Int
    psd::Any
    highpass_cutoff_hz::Float64
    highpass_order::Int
    low_band::Tuple{Float64,Float64}
    high_band::Tuple{Float64,Float64}
    band_edges::Vector{Float64}
    feature_set::Symbol
    context_windows::Int
    function StreamingDetector(
        model::VariationalQuantumClassifier,
        scaler::FeatureScaler,
        threshold::Real;
        sample_rate::Real,
        window_size::Integer,
        step_size::Integer,
        psd = nothing,
        highpass_cutoff_hz::Real = 5e-4,
        highpass_order::Integer = 8,
        low_band::Tuple{Real,Real} = (1e-3, 5e-3),
        high_band::Tuple{Real,Real} = (5e-3, 1e-1),
        band_edges::AbstractVector{<:Real} = [1e-3, 5e-3, 1e-1],
        feature_set::Symbol = :whitened,
        context_windows::Integer = 4,
    )
        sample_rate > 0 || throw(ArgumentError("sample_rate must be positive."))
        window_size >= 2 || throw(ArgumentError("window_size must be at least 2."))
        1 <= step_size <= window_size ||
            throw(ArgumentError("step_size must lie in [1, window_size]."))
        context_windows >= 0 ||
            throw(ArgumentError("context_windows must be non-negative."))
        highpass_cutoff_hz >= 0 ||
            throw(ArgumentError("highpass_cutoff_hz must be non-negative."))
        highpass_order >= 1 || throw(ArgumentError("highpass_order must be at least 1."))
        feature_set in FEATURE_SETS || throw(
            ArgumentError("feature_set = $feature_set; expected one of $(FEATURE_SETS)."),
        )
        return new(
            model,
            scaler,
            Float32(threshold),
            Float64(sample_rate),
            Int(window_size),
            Int(step_size),
            psd,
            Float64(highpass_cutoff_hz),
            Int(highpass_order),
            (Float64(low_band[1]), Float64(low_band[2])),
            (Float64(high_band[1]), Float64(high_band[2])),
            check_band_edges(band_edges),
            feature_set,
            Int(context_windows),
        )
    end
end

"""
    score_window(detector, stretch, offset) -> Float32

Classifier probability of the window starting at `offset` (1-based) of the
contiguous delivered `stretch`, conditioned as described for
[`StreamingDetector`](@ref).
"""
function score_window(
    detector::StreamingDetector,
    stretch::AbstractVector{<:Real},
    offset::Integer,
)
    W = detector.window_size
    1 <= offset && offset + W - 1 <= length(stretch) ||
        throw(ArgumentError("window at offset $offset does not fit the stretch."))
    record = Float64.(stretch)
    if detector.highpass_cutoff_hz > 0
        record = highpass_record(
            record,
            detector.sample_rate;
            cutoff = detector.highpass_cutoff_hz,
            order = detector.highpass_order,
        )
    end
    detector.psd === nothing ||
        (record = whiten_record(record, detector.sample_rate; psd = detector.psd))
    window = view(record, offset:(offset+W-1))
    features = extract_features(
        window,
        detector.sample_rate;
        low_band = detector.low_band,
        high_band = detector.high_band,
        band_edges = detector.band_edges,
        feature_set = detector.feature_set,
    )
    encoded = encode_features(detector.scaler, reshape(collect(Float32.(features)), 1, :))
    return predict_probability(detector.model, vec(encoded))
end

"""
    whitening_psd_from_sidecar(sidecar_path) -> Union{Nothing, Function}

The whitening PSD recorded in a feature sidecar written by the
pre-processor: the Robson–Cornish–Liu model, the LDC analytic model, the
persisted Welch table (`<stem>_psd.csv` beside the features), or `nothing`
for `psd = "none"`.
"""
function whitening_psd_from_sidecar(sidecar_path::AbstractString)
    isfile(sidecar_path) || throw(ArgumentError("feature sidecar not found: $sidecar_path"))
    features = get(TOML.parsefile(sidecar_path), "features", Dict{String,Any}())
    mode = cfgget(features, "psd", "model"; type = String)
    if mode == "model"
        years = cfgget(features, "observation_years", 1.0; type = Float64)
        return f -> lisa_noise_psd(f; observation_years = years)
    elseif mode == "ldc"
        model = cfgget(features, "ldc_model", "sangria"; type = String)
        tdi2 = cfgget(features, "ldc_tdi2", false; type = Bool)
        years = cfgget(features, "ldc_observation_years", 0.0; type = Float64)
        return f -> ldc_tdi_psd(
            f;
            channel = :A,
            model = model,
            tdi2 = tdi2,
            observation_years = years,
        )
    elseif mode == "welch"
        stem = replace(sidecar_path, r"_features\.toml$" => "")
        table_path = stem * "_psd.csv"
        isfile(table_path) || throw(
            ArgumentError("Welch PSD table not found beside the sidecar: $table_path"),
        )
        table = CSV.read(table_path, DataFrame)
        return interpolated_psd(
            Vector{Float64}(table.frequency_hz),
            Vector{Float64}(table.psd),
        )
    elseif mode == "none"
        return nothing
    end
    throw(
        ArgumentError("sidecar psd = $(repr(mode)); expected model, ldc, welch, or none."),
    )
end

# --- Replay ------------------------------------------------------------

"""
    WindowRecord

One scored window of a replay: `window` index, payload `row_start` and
`row_end`, their mission times `content_start` and `content_end`, the
arrival time `complete_at` of the `completing_batch`, the window's
`coverage`, the classifier `score`, the `decision`, and the inference wall
time `inference_wall_ms`.
"""
struct WindowRecord
    window::Int
    row_start::Int
    row_end::Int
    content_start::Dates.DateTime
    content_end::Dates.DateTime
    complete_at::Dates.DateTime
    completing_batch::String
    coverage::Float64
    score::Float32
    decision::Int
    inference_wall_ms::Float64
end

"""
    ReplayState(run, detector; min_coverage = 1.0, tdi_gap_dilation_sec = 0.0)

Consumer state of a replay: the batches of the run, the [`Coverage`](@ref)
of delivered rows, the [`WindowScheduler`](@ref), the delivered payload
per batch, the scored windows, and the number of arrival events consumed.
Fed one event at a time by [`process_event!`](@ref).
"""
mutable struct ReplayState
    geometry::RunGeometry
    detector::StreamingDetector
    batches::Dict{String,BatchRecord}
    coverage::Coverage
    scheduler::WindowScheduler
    payload::Dict{Int,Vector{Float32}}
    windows::Vector{WindowRecord}
    excluded::Vector{UnitRange{Int}}
    erosion::Int
    consumed::Int
    function ReplayState(
        run::AbstractTelemetryRun,
        detector::StreamingDetector;
        min_coverage::Real = 1.0,
        tdi_gap_dilation_sec::Real = 0.0,
    )
        geometry = run_geometry(run)
        isapprox(geometry.sample_rate, detector.sample_rate; rtol = 1e-9) || throw(
            ArgumentError(
                "the run samples at $(geometry.sample_rate) Hz but the detector expects " *
                "$(detector.sample_rate) Hz.",
            ),
        )
        tdi_gap_dilation_sec >= 0 ||
            throw(ArgumentError("tdi_gap_dilation_sec must be non-negative."))
        return new(
            geometry,
            detector,
            Dict(b.name => b for b in list_batches(run)),
            Coverage(),
            WindowScheduler(
                detector.window_size,
                detector.step_size;
                min_coverage = min_coverage,
            ),
            Dict{Int,Vector{Float32}}(),
            WindowRecord[],
            UnitRange{Int}[],
            round(Int, tdi_gap_dilation_sec * geometry.sample_rate),
            0,
        )
    end
end

"""
    delivered_stretch(state, window) -> (samples, offset)

The delivered samples around `window` — the covered interval containing it,
cut to `context_windows` window lengths on each side — and the 1-based
offset of the window inside them; `(nothing, 0)` when the window is not
fully delivered.
"""
function delivered_stretch(state::ReplayState, window::UnitRange{Int})
    span = covered_stretch(state.coverage, window)
    isempty(span) && return nothing, 0
    context = state.detector.context_windows * state.detector.window_size
    lo = max(first(span), first(window) - context)
    hi = min(last(span), last(window) + context)
    P = state.geometry.points_per_batch
    samples = Vector{Float32}(undef, hi - lo + 1)
    for k in (div(lo-1, P)+1):(div(hi-1, P)+1)
        data = state.payload[k]
        r = batch_rows(k, P)
        a = max(lo, first(r))
        b = min(hi, last(r))
        samples[(a-lo+1):(b-lo+1)] = view(data, (a-first(r)+1):(b-first(r)+1))
    end
    return samples, first(window) - lo + 1
end

"""
    process_event!(state, run, event; on_window = nothing) -> Vector{WindowRecord}

Consume one arrival event: an `ingested` batch is read, added to the
coverage, and every window that thereby becomes evaluable is scored at the
event's mission time; a `lost` or `pruned` batch is removed from the
coverage with the configured erosion on each side. Other events are
ignored. Returns the windows scored by this event; `on_window(record)` is
called for each.
"""
function process_event!(
    state::ReplayState,
    run::AbstractTelemetryRun,
    event::ArrivalEvent;
    on_window = nothing,
)
    state.consumed += 1
    scored = WindowRecord[]
    haskey(state.batches, event.batch) || return scored
    batch = state.batches[event.batch]
    if event.event == :ingested
        state.payload[batch.index] = read_batch(run, batch.name)
        add!(state.coverage, batch.rows)
        # Permanent holes (lost or pruned batches with their erosion) stay
        # excluded whatever arrives later around them.
        for hole in state.excluded
            remove!(state.coverage, hole)
        end
        for m in newly_evaluable!(state.scheduler, state.coverage, batch.rows)
            window = window_rows(state.scheduler, m)
            # Beyond the ingested payload the producer streams zeros
            0 < state.geometry.payload_rows < last(window) && continue
            stretch, offset = delivered_stretch(state, window)
            stretch === nothing && continue
            t0 = time()
            score = score_window(state.detector, stretch, offset)
            record = WindowRecord(
                m,
                first(window),
                last(window),
                row_time(state.geometry, first(window)),
                row_time(state.geometry, last(window)),
                event.sim_time,
                batch.name,
                covered_fraction(state.coverage, window),
                score,
                score >= state.detector.threshold ? 1 : 0,
                1000 * (time() - t0),
            )
            push!(state.windows, record)
            push!(scored, record)
            on_window === nothing || on_window(record)
        end
    elseif event.event in (:lost, :pruned)
        lo = max(1, first(batch.rows) - state.erosion)
        hi = last(batch.rows) + state.erosion
        push!(state.excluded, lo:hi)
        remove!(state.coverage, lo:hi)
        delete!(state.payload, batch.index)
    end
    return scored
end

"""
    windows_table(state) -> DataFrame

The scored windows of a replay as a table, one row per
[`WindowRecord`](@ref).
"""
function windows_table(state::ReplayState)
    w = state.windows
    return DataFrame(
        window = [r.window for r in w],
        row_start = [r.row_start for r in w],
        row_end = [r.row_end for r in w],
        content_start = [r.content_start for r in w],
        content_end = [r.content_end for r in w],
        complete_at = [r.complete_at for r in w],
        completing_batch = [r.completing_batch for r in w],
        coverage = [r.coverage for r in w],
        score = [r.score for r in w],
        decision = [r.decision for r in w],
        inference_wall_ms = [r.inference_wall_ms for r in w],
    )
end

"""
    replay_run(run, detector; min_coverage = 1.0, tdi_gap_dilation_sec = 0.0,
               on_window = nothing) -> DataFrame

Replay the arrival feed of `run` in mission-time order through
[`process_event!`](@ref) and return the [`windows_table`](@ref): one row
per scored window with `window`, `row_start`, `row_end`, `content_start`,
`content_end`, `complete_at`, `completing_batch`, `coverage`, `score`,
`decision`, `inference_wall_ms`.
"""
function replay_run(
    run::AbstractTelemetryRun,
    detector::StreamingDetector;
    min_coverage::Real = 1.0,
    tdi_gap_dilation_sec::Real = 0.0,
    on_window = nothing,
)
    return @timeit TIMER "telemetry replay" begin
        state = ReplayState(
            run,
            detector;
            min_coverage = min_coverage,
            tdi_gap_dilation_sec = tdi_gap_dilation_sec,
        )
        for event in arrival_events(run)
            process_event!(state, run, event; on_window = on_window)
        end
        windows_table(state)
    end
end

"""
    follow_run(run, detector; poll_interval_sec = 1.0, min_coverage = 1.0,
               tdi_gap_dilation_sec = 0.0, on_window = nothing, max_wall_sec = Inf)
        -> DataFrame

Live mode: poll the arrival feed of `run` every `poll_interval_sec`,
consuming events beyond those already processed, until the run reaches a
terminal state (`run_state` ≠ `:active`) and the feed is drained, or
`max_wall_sec` of wall time elapse. The consumer never writes into the run.
Returns the [`windows_table`](@ref).
"""
function follow_run(
    run::AbstractTelemetryRun,
    detector::StreamingDetector;
    poll_interval_sec::Real = 1.0,
    min_coverage::Real = 1.0,
    tdi_gap_dilation_sec::Real = 0.0,
    on_window = nothing,
    max_wall_sec::Real = Inf,
)
    poll_interval_sec > 0 || throw(ArgumentError("poll_interval_sec must be positive."))
    state = ReplayState(
        run,
        detector;
        min_coverage = min_coverage,
        tdi_gap_dilation_sec = tdi_gap_dilation_sec,
    )
    start = time()
    while true
        events = arrival_events(run)
        fresh = length(events) > state.consumed
        if fresh
            # New batches may have appeared since the batch list was taken
            for b in list_batches(run)
                state.batches[b.name] = b
            end
            for event in events[(state.consumed+1):end]
                process_event!(state, run, event; on_window = on_window)
            end
        end
        terminal = run_state(run) != :active
        (terminal && !fresh) && break
        time() - start > max_wall_sec && break
        sleep(poll_interval_sec)
    end
    return windows_table(state)
end

"""
    detector_from_run(model_path; threshold_path = joinpath(dirname(model_path), "threshold.toml"),
                      config_path = joinpath(dirname(model_path), "config.toml"),
                      psd_sidecar = "", context_windows = 4) -> StreamingDetector

The [`StreamingDetector`](@ref) of a training run: model and scaler from
the artifact, the fitted threshold from `threshold.toml`, and the
conditioning — window geometry, whitening PSD, analysis bands, record
high-pass, feature set — from the sidecar of the feature table the run was
trained on (recorded in its `config.toml` snapshot).

`psd_sidecar`, when given, supplies the whitening PSD from a different
feature sidecar while everything else still comes from the training one.
A whitening PSD is a calibration of the record being scored, not a
property of the model: the persisted training PSD describes the noise of
the training record, and where that noise is not stationary between
records — the Galactic foreground is modulated over the year by the
constellation's antenna pattern — whitening a later record with it
mis-scales every band power, the feature scaler clips the result, and the
scores collapse. Give the sidecar of the record under analysis whenever
one exists.

`context_windows` must reach past the conditioning kernel. The kernel of
the record high-pass followed by whitening decays as a power law, not
exponentially, when the PSD resolves sharp spectral features; with the
Sangria Welch estimate its envelope is still 7 % of the peak eight window
lengths from the impulse and the streamed scores agree with the batch
pipeline only from about twenty window lengths upward, not monotonically
below that. Measure the agreement against the batch path for the record
at hand rather than assuming a value is large enough.
"""
function detector_from_run(
    model_path::AbstractString;
    threshold_path::AbstractString = joinpath(dirname(model_path), "threshold.toml"),
    config_path::AbstractString = joinpath(dirname(model_path), "config.toml"),
    psd_sidecar::AbstractString = "",
    context_windows::Integer = 4,
)
    isfile(model_path) || throw(ArgumentError("model artifact not found: $model_path"))
    isfile(threshold_path) ||
        throw(ArgumentError("threshold file not found: $threshold_path"))
    isfile(config_path) || throw(ArgumentError("training snapshot not found: $config_path"))
    model, _, scaler = load_model(model_path)
    scaler === nothing &&
        throw(ArgumentError("the model artifact carries no feature scaler."))
    threshold = Float32(TOML.parsefile(threshold_path)["threshold"]["value"])
    snapshot = TOML.parsefile(config_path)
    features_path = resolvepath(
        cfgget(section(snapshot, "training"), "train_features", ""; type = String),
    )
    sidecar_path = replace(features_path, r"\.csv$" => ".toml")
    isfile(sidecar_path) || throw(
        ArgumentError(
            "the feature sidecar $sidecar_path of the training run is required to " *
            "reproduce the conditioning.",
        ),
    )
    features = get(TOML.parsefile(sidecar_path), "features", Dict{String,Any}())
    low = cfgget(features, "low_band_hz", [1e-3, 5e-3]; type = AbstractVector)
    high = cfgget(features, "high_band_hz", [5e-3, 1e-1]; type = AbstractVector)
    edges = cfgget(features, "band_edges_hz", [1e-3, 5e-3, 1e-1]; type = AbstractVector)
    if !isempty(psd_sidecar)
        isfile(psd_sidecar) ||
            throw(ArgumentError("whitening sidecar not found: $psd_sidecar"))
        @info "whitening with a sidecar other than the training run's" psd_sidecar =
            psd_sidecar training_sidecar = sidecar_path
    end
    return StreamingDetector(
        model,
        scaler,
        threshold;
        sample_rate = cfgget(features, "sample_rate", 0.2; type = Float64, min = 1e-6),
        window_size = cfgget(features, "window_size", 1000; type = Int, min = 2),
        step_size = cfgget(features, "step_size", 100; type = Int, min = 1),
        psd = whitening_psd_from_sidecar(isempty(psd_sidecar) ? sidecar_path : psd_sidecar),
        highpass_cutoff_hz = cfgget(features, "highpass_cutoff_hz", 5e-4; type = Float64),
        highpass_order = cfgget(features, "highpass_order", 8; type = Int, min = 1),
        low_band = (Float64(low[1]), Float64(low[2])),
        high_band = (Float64(high[1]), Float64(high[2])),
        band_edges = Float64.(edges),
        feature_set = Symbol(cfgget(features, "feature_set", "whitened"; type = String)),
        context_windows = context_windows,
    )
end

"""
    open_telemetry_run(run_dir; producer_compat = "1.0") -> AbstractTelemetryRun

Open a DeepSpaceTelemetry run directory through the producer's own API.
Implemented by the package extension that loads with DeepSpaceTelemetry;
`producer_compat` is the accepted lower bound of the producer version
recorded in the run's configuration snapshot.
"""
function open_telemetry_run end

# --- Alert latency -----------------------------------------------------

"""
    event_merger_times(events) -> Vector{Float64}

Merger times [s after the mission epoch] of an event table: the column
`merger_time_s` (LDC event tables) or `t_c_sec` (the simulator catalog).
"""
function event_merger_times(events::DataFrame)
    "merger_time_s" in names(events) && return Float64.(events.merger_time_s)
    "t_c_sec" in names(events) && return Float64.(events.t_c_sec)
    throw(
        ArgumentError(
            "the event table lacks a merger-time column (merger_time_s or t_c_sec).",
        ),
    )
end

"""
    alert_latency_table(windows, events, geometry; processing_latency_hours = 1.0)
        -> DataFrame

Per event of `events` (columns `merger_time_s` [s after the mission epoch]
and, when present, `label_start_index`/`label_end_index` payload rows; a
`label` or `event` column names it): the first alarmed window of `windows`
(the table of [`replay_run`](@ref)) overlapping the event's label span
(rows, or the window containing the merger when no span is given),
`t_alarm` = its `complete_at`, `latency_data_h = t_alarm − t_merger`
(negative when the alarm precedes the merger, i.e. the inspiral was
detected inside the label span before the coalescence),
`latency_total_h = latency_data_h + processing_latency_hours`, the
inference wall time of that window, and `detected`. Windows alarmed outside
every label span count as false alarms, reported as
`false_alarms_per_30d` in every row.
"""
function alert_latency_table(
    windows::DataFrame,
    events::DataFrame,
    geometry::RunGeometry;
    processing_latency_hours::Real = 1.0,
)
    processing_latency_hours >= 0 ||
        throw(ArgumentError("processing_latency_hours must be non-negative."))
    times = event_merger_times(events)
    n_events = nrow(events)
    has_span = "label_start_index" in names(events) && "label_end_index" in names(events)
    spans = UnitRange{Int}[]
    for i in 1:n_events
        if has_span
            push!(spans, Int(events.label_start_index[i]):Int(events.label_end_index[i]))
        else
            row = round(Int, times[i] * geometry.sample_rate) + 1
            push!(spans, row:row)
        end
    end
    labels = if "label" in names(events)
        String.(events.label)
    elseif "event" in names(events)
        ["event_$(e)" for e in events.event]
    else
        ["event_$i" for i in 1:n_events]
    end
    alarmed = nrow(windows) == 0 ? Int[] : findall(==(1), windows.decision)
    in_span = falses(length(alarmed))
    out = DataFrame(
        label = String[],
        merger_time_s = Float64[],
        t_merger = Dates.DateTime[],
        detected = Bool[],
        alarm_window = Union{Missing,Int}[],
        t_alarm = Union{Missing,Dates.DateTime}[],
        latency_data_h = Union{Missing,Float64}[],
        latency_total_h = Union{Missing,Float64}[],
        inference_wall_ms = Union{Missing,Float64}[],
    )
    for i in 1:n_events
        span = spans[i]
        t_merger = geometry.start_sim_time + Dates.Millisecond(round(Int, 1000 * times[i]))
        first_alarm = nothing
        for (j, w) in enumerate(alarmed)
            r = Int(windows.row_start[w]):Int(windows.row_end[w])
            overlaps = first(r) <= last(span) && first(span) <= last(r)
            overlaps || continue
            in_span[j] = true
            if first_alarm === nothing ||
               windows.complete_at[w] < windows.complete_at[first_alarm]
                first_alarm = w
            end
        end
        if first_alarm === nothing
            push!(
                out,
                (
                    labels[i],
                    times[i],
                    t_merger,
                    false,
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                ),
            )
        else
            t_alarm = windows.complete_at[first_alarm]
            latency_h = Dates.value(t_alarm - t_merger) / 3.6e6
            push!(
                out,
                (
                    labels[i],
                    times[i],
                    t_merger,
                    true,
                    Int(windows.window[first_alarm]),
                    t_alarm,
                    latency_h,
                    latency_h + processing_latency_hours,
                    Float64(windows.inference_wall_ms[first_alarm]),
                ),
            )
        end
    end
    n_false = 0
    if !isempty(alarmed)
        outside = alarmed[.!in_span]
        # Contiguous alarmed windows outside every span form one episode
        sorted = sort(Int.(windows.window[outside]))
        n_false =
            isempty(sorted) ? 0 :
            1 + count(k -> sorted[k] != sorted[k-1] + 1, 2:length(sorted))
    end
    observation_days =
        nrow(windows) == 0 ? 0.0 :
        nrow(windows) * geometry.sample_rate^-1 * detector_step(windows) / 86400
    out.false_alarm_episodes = fill(n_false, n_events)
    out.false_alarms_per_30d =
        fill(observation_days == 0 ? NaN : n_false / observation_days * 30, n_events)
    return out
end

"""
    detector_step(windows) -> Int

Window step [samples] implied by a replay table: the smallest positive
difference between distinct `row_start` values (windows complete out of
order, so the table is not sorted by window), or the window length when a
single window was scored.
"""
function detector_step(windows::DataFrame)
    nrow(windows) == 0 && return 0
    starts = sort(unique(Int.(windows.row_start)))
    length(starts) >= 2 || return Int(windows.row_end[1] - windows.row_start[1] + 1)
    return minimum(diff(starts))
end
