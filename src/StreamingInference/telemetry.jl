# Streamed consumption of a delivered record: the run interface of a
# producer, the coverage of delivered rows, window scheduling, the streaming
# detector and its conditioning, the causal whitening estimate, the replay
# and live-follow engine, and the alert tables of a replay.

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
    time_row(geometry, t) -> Int

Payload row whose mission time is `t`, the inverse of [`row_time`](@ref)
rounded to the nearest row. Rows are 1-based and a time before
`start_sim_time` yields a row below 1, which callers must reject rather
than clamp: it means the batch does not belong to the payload the run was
started from.
"""
function time_row(geometry::RunGeometry, t::Dates.DateTime)
    ms = Dates.value(Dates.Millisecond(t - geometry.start_sim_time))
    return round(Int, ms * geometry.sample_rate / 1000) + 1
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
    WindowScheduler(window_size, step_size; min_coverage = 1.0, context_rows = 0, payload_rows = 0)

Bookkeeping of the sliding windows (window `m` covers rows
`[1 + (m − 1) S, (m − 1) S + W]`) that have become evaluable: a window is
evaluable once its own rows are all delivered and at least `min_coverage`
of its conditioning stretch is, and each window is emitted once. A hole
inside a window is never scored across; `min_coverage` below one only
admits a window whose context is incomplete.

`context_rows` is the conditioning stretch the detector needs on each
side of a window: a window becomes evaluable only once that stretch, cut
to the record by `payload_rows` (0 meaning unknown, so no cut at the
end), is covered. The whitening is zero-phase and its kernel therefore
two-sided, so a window scored before the data after it has arrived is not
conditioned as the batch pipeline conditions it; matching that
conditioning requires a lag of `context_rows` samples behind the delivery
front.
"""
mutable struct WindowScheduler
    window_size::Int
    step_size::Int
    min_coverage::Float64
    context_rows::Int
    payload_rows::Int
    emitted::Set{Int}
    function WindowScheduler(
        window_size::Integer,
        step_size::Integer;
        min_coverage::Real = 1.0,
        context_rows::Integer = 0,
        payload_rows::Integer = 0,
    )
        window_size >= 2 || throw(ArgumentError("window_size must be at least 2."))
        1 <= step_size <= window_size ||
            throw(ArgumentError("step_size must lie in [1, window_size]."))
        0 < min_coverage <= 1 || throw(ArgumentError("min_coverage must lie in (0, 1]."))
        context_rows >= 0 || throw(ArgumentError("context_rows must be non-negative."))
        payload_rows >= 0 || throw(ArgumentError("payload_rows must be non-negative."))
        return new(
            Int(window_size),
            Int(step_size),
            Float64(min_coverage),
            Int(context_rows),
            Int(payload_rows),
            Set{Int}(),
        )
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
    conditioning_rows(scheduler, m) -> UnitRange{Int}

Payload rows that must be delivered before window `m` can be scored: its
own rows widened by `context_rows` on each side and cut to the record
(`payload_rows`, when known). At the ends of a record the stretch cannot
be centred on the window and the score is edge-affected, exactly as the
first and last windows of a batch-processed record are.
"""
function conditioning_rows(scheduler::WindowScheduler, m::Integer)
    w = window_rows(scheduler, m)
    lo = max(1, first(w) - scheduler.context_rows)
    hi = last(w) + scheduler.context_rows
    scheduler.payload_rows > 0 && (hi = min(hi, scheduler.payload_rows))
    return lo:max(hi, last(w))
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

Windows whose conditioning stretch ([`conditioning_rows`](@ref))
intersects `rows` and is now covered under `coverage`, and that have not
been emitted before; they are recorded as emitted. An arriving batch can
therefore make a window evaluable that lies up to `context_rows` earlier
in the record, which is why the search range is widened.
"""
function newly_evaluable!(
    scheduler::WindowScheduler,
    coverage::Coverage,
    rows::AbstractUnitRange{<:Integer},
)
    ready = Int[]
    widened = (first(rows)-scheduler.context_rows):(last(rows)+scheduler.context_rows)
    for m in windows_touching(scheduler, widened)
        m in scheduler.emitted && continue
        # The window's own rows must all be on the ground; `min_coverage`
        # bounds the delivered fraction of the stretch around them.
        if !isempty(covered_stretch(coverage, window_rows(scheduler, m))) &&
           covered_fraction(coverage, conditioning_rows(scheduler, m)) >=
           scheduler.min_coverage
            push!(scheduler.emitted, m)
            push!(ready, m)
        end
    end
    return ready
end

# --- Streaming detector ------------------------------------------------

"""
    StreamingDetector(scorer, threshold; sample_rate, window_size, step_size,
                      psd = nothing, highpass_cutoff_hz = 5e-4, highpass_order = 8,
                      context_windows = 4)

Scoring of complete windows out of a partially delivered record with the
same conditioning as the batch pipeline. A window is cut from the delivered
stretch around it — up to `context_windows` window lengths on each side —
which is high-passed and, when `psd` is given, whitened as a whole
([`condition_window`](@ref)); the conditioned window is scored by `scorer`
([`window_score`](@ref)) and alarms at `threshold`. Isolated windows
without context are scored on their own samples (edge-affected, as the
first windows of any record). The conditioning belongs to the detector,
the method to the scorer, so any [`AbstractWindowScorer`](@ref) runs
through the same replay.
"""
struct StreamingDetector{S<:AbstractWindowScorer}
    scorer::S
    threshold::Float32
    sample_rate::Float64
    window_size::Int
    step_size::Int
    psd::Any
    highpass_cutoff_hz::Float64
    highpass_order::Int
    context_windows::Int
    function StreamingDetector(
        scorer::S,
        threshold::Real;
        sample_rate::Real,
        window_size::Integer,
        step_size::Integer,
        psd = nothing,
        highpass_cutoff_hz::Real = 5e-4,
        highpass_order::Integer = 8,
        context_windows::Integer = 4,
    ) where {S<:AbstractWindowScorer}
        sample_rate > 0 || throw(ArgumentError("sample_rate must be positive."))
        window_size >= 2 || throw(ArgumentError("window_size must be at least 2."))
        1 <= step_size <= window_size ||
            throw(ArgumentError("step_size must lie in [1, window_size]."))
        context_windows >= 0 ||
            throw(ArgumentError("context_windows must be non-negative."))
        highpass_cutoff_hz >= 0 ||
            throw(ArgumentError("highpass_cutoff_hz must be non-negative."))
        highpass_order >= 1 || throw(ArgumentError("highpass_order must be at least 1."))
        return new{S}(
            scorer,
            Float32(threshold),
            Float64(sample_rate),
            Int(window_size),
            Int(step_size),
            psd,
            Float64(highpass_cutoff_hz),
            Int(highpass_order),
            Int(context_windows),
        )
    end
end

"""
    condition_window(detector, stretch, offset; psd = detector.psd) -> AbstractVector{Float64}

The window starting at `offset` (1-based) of the contiguous delivered
`stretch`, conditioned as the batch pipeline conditions a record: the
stretch high-passed ([`highpass_record`](@ref)) and, unless `psd` is
`nothing`, whitened by `psd` ([`whiten_record`](@ref)) as a whole, then
the window cut from it. `psd` replaces the detector's whitening PSD for
this window — in a replay under [`TrailingWelch`](@ref), the causal PSD
estimate, made only from data already delivered to the ground station.
"""
function condition_window(
    detector::StreamingDetector,
    stretch::AbstractVector{<:Real},
    offset::Integer;
    psd = detector.psd,
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
    psd === nothing || (record = whiten_record(record, detector.sample_rate; psd = psd))
    return view(record, offset:(offset+W-1))
end

"""
    score_window(detector, stretch, offset; psd = detector.psd) -> Float32

Score of the window starting at `offset` of the contiguous delivered
`stretch`: the window conditioned ([`condition_window`](@ref)) and scored
by the detector's scorer ([`window_score`](@ref)).
"""
function score_window(
    detector::StreamingDetector,
    stretch::AbstractVector{<:Real},
    offset::Integer;
    psd = detector.psd,
)
    window = condition_window(detector, stretch, offset; psd = psd)
    return window_score(detector.scorer, window, detector.sample_rate)
end

# --- Replay ------------------------------------------------------------

"""
    WindowRecord

One scored window of a replay: `window` index, payload `row_start` and
`row_end`, their mission times `content_start` and `content_end`, the
arrival time `complete_at` of the `completing_batch`, the window's
`coverage`, the classifier `score`, the `decision`, the inference wall
time `inference_wall_ms`, and `psd_row`, the last delivered row behind
the causal PSD estimate ([`TrailingWelch`](@ref)) the window
was whitened with, 0 when the detector's static PSD was used, and
`release_at`, the arrival time at which the window was scored: its
`complete_at` for a [`Stateless`](@ref) scorer, and for a
[`Stateful`](@ref) one the arrival that released it in content order
([`OrderedCommit`](@ref)). The twelve-argument constructor sets
`release_at = complete_at`.
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
    psd_row::Int
    release_at::Dates.DateTime
end

function WindowRecord(
    window::Integer,
    row_start::Integer,
    row_end::Integer,
    content_start::Dates.DateTime,
    content_end::Dates.DateTime,
    complete_at::Dates.DateTime,
    completing_batch::AbstractString,
    coverage::Real,
    score::Real,
    decision::Integer,
    inference_wall_ms::Real,
    psd_row::Integer,
)
    return WindowRecord(
        window,
        row_start,
        row_end,
        content_start,
        content_end,
        complete_at,
        completing_batch,
        coverage,
        score,
        decision,
        inference_wall_ms,
        psd_row,
        complete_at,
    )
end

"""
    PendingWindow

A window conditioned when it became evaluable and held by an
[`OrderedCommit`](@ref) until its release: its payload `rows`, the
conditioned `samples`, `complete_at` and `completing_batch`, its
`coverage` and `psd_row` (as in [`WindowRecord`](@ref)).
"""
struct PendingWindow
    rows::UnitRange{Int}
    samples::Vector{Float64}
    complete_at::Dates.DateTime
    completing_batch::String
    coverage::Float64
    psd_row::Int
end

"""
    OrderedCommit(last_window; order_horizon = nothing, late_policy = :skip)

Release of the windows of a replay to a [`Stateful`](@ref) estimator in
content order. A window is conditioned when it becomes evaluable, exactly
as for a [`Stateless`](@ref) one, and held ([`PendingWindow`](@ref));
windows are released by strictly increasing index, each once, and a run
of windows that can never be scored is declared by one
[`GapEvent`](@ref) instead, which the estimator receives
([`estimator_gap!`](@ref)) before the window after it. `last_window` is
the last window of the record, 0 when its length is unknown.

A window can never be scored when its own rows, or more of its
conditioning stretch than the coverage bound admits, fell in a lost or
pruned batch (`:lost`). Without further bound a missing window holds back
every later one until the end of the record, where the windows still
missing are declared `:undelivered`. `order_horizon`, a time period,
bounds that wait: when a later window has been held for longer than it,
the windows missing before it are declared `:horizon`. A window that
becomes evaluable after its gap was declared is late: skipped and counted
(`late_policy = :skip`) or refused (`:error`).
"""
mutable struct OrderedCommit
    next::Int
    last_window::Int
    pending::Dict{Int,PendingWindow}
    gaps::Vector{GapEvent}
    late::Int
    now::Dates.DateTime
    order_horizon::Union{Nothing,Dates.Millisecond}
    late_policy::Symbol
    function OrderedCommit(
        last_window::Integer;
        order_horizon::Union{Nothing,Dates.Period} = nothing,
        late_policy::Symbol = :skip,
    )
        last_window >= 0 || throw(ArgumentError("last_window must be non-negative."))
        late_policy in (:skip, :error) ||
            throw(ArgumentError("late_policy = $late_policy; expected :skip or :error."))
        horizon = order_horizon === nothing ? nothing : Dates.Millisecond(order_horizon)
        horizon === nothing ||
            horizon > Dates.Millisecond(0) ||
            throw(ArgumentError("order_horizon must be positive."))
        return new(
            1,
            Int(last_window),
            Dict{Int,PendingWindow}(),
            GapEvent[],
            0,
            Dates.DateTime(0),
            horizon,
            late_policy,
        )
    end
end

"""
    TrailingWelch(span_rows, refresh_rows, segment_length; edge_rows = 0, smoothing_dex = 0)

Causal whitening of a replay. Each window is whitened by the median
Welch estimate ([`welch_psd`](@ref), segments of `segment_length` samples)
of the delivered record behind it: every delivered run inside the last
`span_rows` rows before the end of the window's conditioning stretch, each
run high-passed as the detector high-passes every stretch — the order in
which the batch pre-processor estimates its `"welch"` PSD — and the
segments of all runs pooled, so that no segment spans a delivery hole.
The high-pass rings at both ends of a run (the ends of the record in the
batch pipeline, whose edge windows are dropped), so `edge_rows` rows are
trimmed from each end of a high-passed run before it is segmented, and a
run must hold a segment beyond that trim. A positive `smoothing_dex`
smooths the estimate in log-frequency ([`smooth_psd`](@ref)), as
`[preprocessing] psd_smoothing_dex` smooths the batch estimate the model's
features were whitened with. Nothing that has not reached the ground
enters the estimate, unlike a
PSD of the whole record, which at every window of a streamed mission
contains data still to be delivered. The estimate is redone once the end
of that record has moved by `refresh_rows` rows since the previous one.
While no delivered run behind a window holds a segment — at the start of
a mission — the window is whitened by the previous estimate, and by the
detector's own static PSD only before any estimate exists.
"""
struct TrailingWelch
    span_rows::Int
    refresh_rows::Int
    segment_length::Int
    edge_rows::Int
    smoothing_dex::Float64
    function TrailingWelch(
        span_rows::Integer,
        refresh_rows::Integer,
        segment_length::Integer;
        edge_rows::Integer = 0,
        smoothing_dex::Real = 0.0,
    )
        segment_length >= 2 || throw(ArgumentError("segment_length must be at least 2."))
        span_rows >= segment_length || throw(
            ArgumentError(
                "span_rows = $span_rows; must hold at least one segment of $segment_length rows.",
            ),
        )
        refresh_rows >= 1 || throw(ArgumentError("refresh_rows must be at least 1."))
        edge_rows >= 0 || throw(ArgumentError("edge_rows must be non-negative."))
        smoothing_dex >= 0 || throw(ArgumentError("smoothing_dex must be non-negative."))
        return new(
            Int(span_rows),
            Int(refresh_rows),
            Int(segment_length),
            Int(edge_rows),
            Float64(smoothing_dex),
        )
    end
end

"""
    ReplayState(run, detector; min_coverage = 1.0, tdi_gap_dilation_sec = 0.0,
                trailing_psd = nothing, order_horizon = nothing, late_policy = :skip)

Consumer state of a replay: the batches of the run, the [`Coverage`](@ref)
of delivered rows, the [`WindowScheduler`](@ref), the delivered payload
per batch, the scored windows, the number of arrival events consumed,
under `trailing_psd::`[`TrailingWelch`](@ref) the current causal PSD
estimate with the last delivered row it was made on, and, when the
detector's scorer is [`Stateful`](@ref), the [`OrderedCommit`](@ref) that
releases its windows in content order (`order_horizon`, `late_policy`);
the scorer is reset ([`reset_estimator!`](@ref)) on construction. Fed one
event at a time by [`process_event!`](@ref) and closed by
[`finalize_replay!`](@ref).
"""
mutable struct ReplayState
    geometry::RunGeometry
    detector::StreamingDetector
    batches::Dict{String,BatchRecord}
    coverage::Coverage
    scheduler::WindowScheduler
    # Delivered payload keyed by the first row a batch holds, with those
    # first rows kept sorted: a producer that discarded production keeps
    # numbering the batches it stores, so a batch's rows follow its content
    # epoch and not its index.
    payload::Dict{Int,Vector{Float32}}
    starts::Vector{Int}
    windows::Vector{WindowRecord}
    excluded::Vector{UnitRange{Int}}
    erosion::Int
    consumed::Int
    trailing::Union{Nothing,TrailingWelch}
    trailing_psd::Any
    trailing_row::Int
    commit::Union{Nothing,OrderedCommit}
    function ReplayState(
        run::AbstractTelemetryRun,
        detector::StreamingDetector;
        min_coverage::Real = 1.0,
        tdi_gap_dilation_sec::Real = 0.0,
        trailing_psd::Union{Nothing,TrailingWelch} = nothing,
        order_horizon::Union{Nothing,Dates.Period} = nothing,
        late_policy::Symbol = :skip,
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
        W, S = detector.window_size, detector.step_size
        last_window = geometry.payload_rows >= W ? div(geometry.payload_rows - W, S) + 1 : 0
        commit =
            estimator_memory(detector.scorer) isa Stateful ?
            OrderedCommit(
                last_window;
                order_horizon = order_horizon,
                late_policy = late_policy,
            ) : nothing
        commit === nothing || reset_estimator!(detector.scorer)
        return new(
            geometry,
            detector,
            Dict(b.name => b for b in list_batches(run)),
            Coverage(),
            WindowScheduler(
                detector.window_size,
                detector.step_size;
                min_coverage = min_coverage,
                context_rows = detector.context_windows * detector.window_size,
                payload_rows = max(0, geometry.payload_rows),
            ),
            Dict{Int,Vector{Float32}}(),
            Int[],
            WindowRecord[],
            UnitRange{Int}[],
            round(Int, tdi_gap_dilation_sec * geometry.sample_rate),
            0,
            trailing_psd,
            nothing,
            0,
            commit,
        )
    end
end

"""
    delivered_rows(state, rows) -> Vector{Float32}

The payload of the delivered rows `rows`, assembled from the batches on the
ground by the rows they hold (`ArgumentError` when a row is not covered).
"""
function delivered_rows(state::ReplayState, rows::UnitRange{Int})
    lo, hi = first(rows), last(rows)
    samples = Vector{Float32}(undef, hi - lo + 1)
    covered = falses(hi - lo + 1)
    i = max(1, searchsortedlast(state.starts, lo))
    while i <= length(state.starts) && state.starts[i] <= hi
        s = state.starts[i]
        data = state.payload[s]
        a = max(lo, s)
        b = min(hi, s + length(data) - 1)
        if a <= b
            samples[(a-lo+1):(b-lo+1)] = view(data, (a-s+1):(b-s+1))
            covered[(a-lo+1):(b-lo+1)] .= true
        end
        i += 1
    end
    all(covered) || throw(
        ArgumentError(
            "rows $rows are not all on the ground ($(count(covered)) delivered).",
        ),
    )
    return samples
end

"""
    whitening_psd!(state, window) -> (psd, psd_row)

Whitening PSD of `window` in a replay: the detector's static PSD with
`psd_row = 0`, or, under [`TrailingWelch`](@ref), the causal PSD
estimate over the delivered record behind the window's conditioning
stretch and the last row it was made on — reused while that end has moved
by less than `refresh_rows`, redone otherwise, and replaced by the static
PSD while fewer than one segment of rows is on the ground.
"""
function whitening_psd!(state::ReplayState, window::UnitRange{Int})
    tw = state.trailing
    tw === nothing && return state.detector.psd, 0
    span = covered_stretch(state.coverage, window)
    isempty(span) && return state.detector.psd, 0
    d = state.detector
    hi = min(last(span), last(window) + d.context_windows * d.window_size)
    lo = hi - tw.span_rows + 1
    # Every delivered run inside the span that holds at least one segment
    # contributes its segments: a hole bounds the runs, never a segment.
    # While no run holds a segment, the previous estimate of the delivered
    # record is kept rather than the detector's static PSD, which belongs to
    # another record.
    # The high-pass rings at both ends of a run, so `edge_rows` are trimmed
    # from each end of a high-passed run before it is segmented, and a run
    # must hold a segment beyond that trim.
    edge = d.highpass_cutoff_hz > 0 ? tw.edge_rows : 0
    runs = UnitRange{Int}[]
    for r in state.coverage.intervals
        a, b = max(first(r), lo), min(last(r), hi)
        b - a + 1 >= tw.segment_length + 2 * edge && push!(runs, a:b)
    end
    if isempty(runs)
        state.trailing_psd === nothing && return d.psd, 0
        return state.trailing_psd, state.trailing_row
    end
    if state.trailing_psd !== nothing && abs(hi - state.trailing_row) < tw.refresh_rows
        return state.trailing_psd, state.trailing_row
    end
    records = map(runs) do r
        record = Float64.(delivered_rows(state, r))
        d.highpass_cutoff_hz > 0 || return record
        filtered = highpass_record(
            record,
            d.sample_rate;
            cutoff = d.highpass_cutoff_hz,
            order = d.highpass_order,
        )
        filtered[(edge+1):(end-edge)]
    end
    freqs, table = welch_psd(
        records,
        d.sample_rate;
        segment_length = tw.segment_length,
        average = :median,
    )
    tw.smoothing_dex > 0 && (table = smooth_psd(freqs, table, tw.smoothing_dex))
    state.trailing_psd = interpolated_psd(freqs, table)
    state.trailing_row = hi
    return state.trailing_psd, hi
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
    return delivered_rows(state, lo:hi), first(window) - lo + 1
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
        start = first(batch.rows)
        haskey(state.payload, start) ||
            insert!(state.starts, searchsortedfirst(state.starts, start), start)
        state.payload[start] = read_batch(run, batch.name)
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
            psd, psd_row = whitening_psd!(state, window)
            if state.commit !== nothing
                hold_window!(state, m, window, stretch, offset, psd, psd_row, event, batch)
                continue
            end
            t0 = time()
            score = score_window(state.detector, stretch, offset; psd = psd)
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
                psd_row,
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
        start = first(batch.rows)
        if haskey(state.payload, start)
            delete!(state.payload, start)
            deleteat!(state.starts, searchsortedfirst(state.starts, start))
        end
    end
    state.commit === nothing || release_windows!(state, event.sim_time, scored, on_window)
    return scored
end

"""
    hold_window!(state, m, window, stretch, offset, psd, psd_row, event, batch)

Condition the evaluable window `m` of a replay with a [`Stateful`](@ref)
scorer and hold it for its ordered release; a window whose gap was already
declared is late ([`OrderedCommit`](@ref)).
"""
function hold_window!(
    state::ReplayState,
    m::Integer,
    window::UnitRange{Int},
    stretch::AbstractVector{<:Real},
    offset::Integer,
    psd,
    psd_row::Integer,
    event::ArrivalEvent,
    batch::BatchRecord,
)
    commit = state.commit
    if m < commit.next
        commit.late_policy === :error && throw(
            ArgumentError(
                "window $m became evaluable at $(event.sim_time), after the gap " *
                "that covers it was declared.",
            ),
        )
        commit.late += 1
        return nothing
    end
    samples = collect(condition_window(state.detector, stretch, offset; psd = psd))
    commit.pending[m] = PendingWindow(
        window,
        samples,
        event.sim_time,
        batch.name,
        covered_fraction(state.coverage, window),
        psd_row,
    )
    return nothing
end

"""
    window_lost(state, m) -> Bool

Whether window `m` can never become evaluable: its own rows, or more of
its conditioning stretch than the scheduler's `min_coverage` admits, lie
in the permanent holes of lost or pruned batches.
"""
function window_lost(state::ReplayState, m::Integer)
    isempty(state.excluded) && return false
    window = window_rows(state.scheduler, m)
    any(hole -> !isempty(intersect(hole, window)), state.excluded) && return true
    stretch = conditioning_rows(state.scheduler, m)
    lost = Coverage()
    for hole in state.excluded
        part = intersect(hole, stretch)
        isempty(part) || add!(lost, part)
    end
    n_lost = round(Int, covered_fraction(lost, stretch) * length(stretch))
    return (length(stretch) - n_lost) / length(stretch) < state.scheduler.min_coverage
end

"""
    declare_gap!(state, gap::GapEvent)

Record `gap`, pass it to the stateful scorer ([`estimator_gap!`](@ref)),
and move the release past it.
"""
function declare_gap!(state::ReplayState, gap::GapEvent)
    push!(state.commit.gaps, gap)
    estimator_gap!(state.detector.scorer, gap)
    state.commit.next = gap.last_window + 1
    return nothing
end

"""
    release_windows!(state, t, scored, on_window)

Release the held windows of a stateful replay in content order at mission
time `t`: score every window from the next index on that is held,
declare a [`GapEvent`](@ref) over every run of windows that can never be
scored or has outlived the order horizon, and stop at the first window
that may still arrive ([`OrderedCommit`](@ref)).
"""
function release_windows!(state::ReplayState, t::Dates.DateTime, scored, on_window)
    commit = state.commit
    commit.now = max(commit.now, t)
    detector = state.detector
    while true
        m = commit.next
        commit.last_window > 0 && m > commit.last_window && break
        if haskey(commit.pending, m)
            held = pop!(commit.pending, m)
            t0 = time()
            score = window_score(detector.scorer, held.samples, detector.sample_rate)
            record = WindowRecord(
                m,
                first(held.rows),
                last(held.rows),
                row_time(state.geometry, first(held.rows)),
                row_time(state.geometry, last(held.rows)),
                held.complete_at,
                held.completing_batch,
                held.coverage,
                score,
                score >= detector.threshold ? 1 : 0,
                1000 * (time() - t0),
                held.psd_row,
                commit.now,
            )
            push!(state.windows, record)
            push!(scored, record)
            on_window === nothing || on_window(record)
            commit.next += 1
        elseif window_lost(state, m)
            stop = m
            while !haskey(commit.pending, stop + 1) &&
                  (commit.last_window == 0 || stop + 1 <= commit.last_window) &&
                  window_lost(state, stop + 1)
                stop += 1
            end
            declare_gap!(state, GapEvent(m, stop, :lost, commit.now))
        elseif commit.order_horizon !== nothing &&
               !isempty(commit.pending) &&
               commit.now - minimum(p.complete_at for p in values(commit.pending)) >
               commit.order_horizon
            declare_gap!(
                state,
                GapEvent(m, minimum(keys(commit.pending)) - 1, :horizon, commit.now),
            )
        else
            break
        end
    end
    return nothing
end

"""
    finalize_replay!(state; on_window = nothing)

Close a replay at the end of its arrival feed: for a [`Stateful`](@ref)
scorer, release every held window in content order and declare the windows
still missing, up to the last window of the record when its length is
known, as `:undelivered` gaps. A no-op for a [`Stateless`](@ref) scorer.
Returns the windows released.
"""
function finalize_replay!(state::ReplayState; on_window = nothing)
    released = WindowRecord[]
    commit = state.commit
    commit === nothing && return released
    while true
        release_windows!(state, commit.now, released, on_window)
        m = commit.next
        commit.last_window > 0 && m > commit.last_window && break
        if isempty(commit.pending)
            commit.last_window > 0 && declare_gap!(
                state,
                GapEvent(m, commit.last_window, :undelivered, commit.now),
            )
            break
        end
        declare_gap!(
            state,
            GapEvent(m, minimum(keys(commit.pending)) - 1, :undelivered, commit.now),
        )
    end
    return released
end

"""
    windows_table(state) -> DataFrame

The scored windows of a replay as a table, one row per
[`WindowRecord`](@ref), in the order they were scored; the column
`release_at` is added for a [`Stateful`](@ref) scorer, whose windows are
scored in content order.
"""
function windows_table(state::ReplayState)
    w = state.windows
    table = DataFrame(
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
        psd_row = [r.psd_row for r in w],
    )
    state.commit === nothing || (table.release_at = [r.release_at for r in w])
    return table
end

"""
    gaps_table(state) -> DataFrame

The [`GapEvent`](@ref)s of a stateful replay, one row each:
`first_window`, `last_window`, `cause`, `declared_at`; empty for a
[`Stateless`](@ref) scorer.
"""
function gaps_table(state::ReplayState)
    gaps = state.commit === nothing ? GapEvent[] : state.commit.gaps
    return DataFrame(
        first_window = [g.first_window for g in gaps],
        last_window = [g.last_window for g in gaps],
        cause = [String(g.cause) for g in gaps],
        declared_at = [g.declared_at for g in gaps],
    )
end

"""
    replay_run(run, detector; min_coverage = 1.0, tdi_gap_dilation_sec = 0.0,
               trailing_psd = nothing, order_horizon = nothing, late_policy = :skip,
               on_window = nothing) -> DataFrame

Replay the arrival feed of `run` in mission-time order through
[`process_event!`](@ref) and return the [`windows_table`](@ref): one row
per scored window with `window`, `row_start`, `row_end`, `content_start`,
`content_end`, `complete_at`, `completing_batch`, `coverage`, `score`,
`decision`, `inference_wall_ms`, `psd_row` (and `release_at` for a
[`Stateful`](@ref) scorer, whose windows are released in content order
under `order_horizon` and `late_policy`; [`OrderedCommit`](@ref)).
`trailing_psd`, a [`TrailingWelch`](@ref), whitens every window by a
causal PSD estimate instead of the detector's static PSD. The gaps of a
stateful replay are read from [`replay_state`](@ref).
"""
function replay_run(run::AbstractTelemetryRun, detector::StreamingDetector; kwargs...)
    return windows_table(replay_state(run, detector; kwargs...))
end

"""
    replay_state(run, detector; min_coverage = 1.0, tdi_gap_dilation_sec = 0.0,
                 trailing_psd = nothing, order_horizon = nothing, late_policy = :skip,
                 on_window = nothing) -> ReplayState

The [`ReplayState`](@ref) after the whole arrival feed of `run`, finalised
([`finalize_replay!`](@ref)): [`windows_table`](@ref) and
[`gaps_table`](@ref) read from it.
"""
function replay_state(
    run::AbstractTelemetryRun,
    detector::StreamingDetector;
    min_coverage::Real = 1.0,
    tdi_gap_dilation_sec::Real = 0.0,
    trailing_psd::Union{Nothing,TrailingWelch} = nothing,
    order_horizon::Union{Nothing,Dates.Period} = nothing,
    late_policy::Symbol = :skip,
    on_window = nothing,
)
    return @timeit TIMER "telemetry replay" begin
        state = ReplayState(
            run,
            detector;
            min_coverage = min_coverage,
            tdi_gap_dilation_sec = tdi_gap_dilation_sec,
            trailing_psd = trailing_psd,
            order_horizon = order_horizon,
            late_policy = late_policy,
        )
        for event in arrival_events(run)
            process_event!(state, run, event; on_window = on_window)
        end
        finalize_replay!(state; on_window = on_window)
        state
    end
end

"""
    follow_run(run, detector; poll_interval_sec = 1.0, min_coverage = 1.0,
               tdi_gap_dilation_sec = 0.0, trailing_psd = nothing,
               order_horizon = nothing, late_policy = :skip,
               on_window = nothing, max_wall_sec = Inf) -> DataFrame

Live mode: poll the arrival feed of `run` every `poll_interval_sec`,
consuming events beyond those already processed, until the run reaches a
terminal state (`run_state` ≠ `:active`) and the feed is drained, or
`max_wall_sec` of wall time elapse. The consumer never writes into the run.
A stateful scorer is served in content order as in [`replay_run`](@ref);
the replay is finalised ([`finalize_replay!`](@ref)) when the loop ends.
Returns the [`windows_table`](@ref).
"""
function follow_run(
    run::AbstractTelemetryRun,
    detector::StreamingDetector;
    poll_interval_sec::Real = 1.0,
    min_coverage::Real = 1.0,
    tdi_gap_dilation_sec::Real = 0.0,
    trailing_psd::Union{Nothing,TrailingWelch} = nothing,
    order_horizon::Union{Nothing,Dates.Period} = nothing,
    late_policy::Symbol = :skip,
    on_window = nothing,
    max_wall_sec::Real = Inf,
)
    poll_interval_sec > 0 || throw(ArgumentError("poll_interval_sec must be positive."))
    state = ReplayState(
        run,
        detector;
        min_coverage = min_coverage,
        tdi_gap_dilation_sec = tdi_gap_dilation_sec,
        trailing_psd = trailing_psd,
        order_horizon = order_horizon,
        late_policy = late_policy,
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
    finalize_replay!(state; on_window = on_window)
    return windows_table(state)
end

# --- Alert latency -----------------------------------------------------

"""
    event_merger_times(events) -> Vector{Float64}

Merger times [s after the mission epoch] of an event table: the column
`merger_time_s` (LDC event tables) or `t_c_sec` (the simulator catalogue).
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
    alert_latency_table(windows, events, geometry; processing_latency_hours = 1.0,
                        persistence = 1, crediting = :signal) -> DataFrame

Per event of `events` (columns `merger_time_s` [s after the mission epoch]
and, when present, `label_start_index`/`label_end_index` payload rows; a
`label` or `event` column names it): the earliest alert of `windows` (the
table of [`replay_run`](@ref)) touching the event's credited span (rows, or
the window containing the merger when no span is given).

`crediting` fixes where the credited span starts. With `:signal` it starts
at the signal onset `signal_start_index` of the event table (the first
window in which the source reaches the labelling SNR threshold) and ends
with the label span: an alarm earlier in the label span cannot be caused by the source and
counts as a false alarm. With `:label` the whole label span is credited,
the convention of the training labels (96 h before to 27 min after the
merger for the Isfan et al. (2025) spans), which credits noise alarms
preceding the signal as early detections. A table with spans but without
`signal_start_index` is refused under `:signal`.

An alert is a run of `persistence` consecutive alarmed windows —
consecutive in window index, whatever order they reached the ground — at
least one of which overlaps the span. It is raised by the arrival that
completes the run: `t_alarm` is the latest `complete_at` of its windows
and `alarm_window` the window of that arrival. With `persistence = 1`
every alarmed window is an alert on its own. `latency_data_h = t_alarm −
t_merger` (negative when the alert precedes the merger),
`latency_total_h = latency_data_h + processing_latency_hours`, the
inference wall time of the completing window, `detected`, and
`shared_alert`, true when the same alert is also another event's (label
spans that overlap). Runs of
alarmed windows outside every label span that reach `persistence` are
false-alarm episodes, reported as `false_alarms_per_30d` in every row;
shorter runs raise no alert and are not counted. `alert_persistence`
and `alert_crediting` record the criteria.
"""
function alert_latency_table(
    windows::DataFrame,
    events::DataFrame,
    geometry::RunGeometry;
    processing_latency_hours::Real = 1.0,
    persistence::Integer = 1,
    crediting::Symbol = :signal,
)
    processing_latency_hours >= 0 ||
        throw(ArgumentError("processing_latency_hours must be non-negative."))
    persistence >= 1 || throw(ArgumentError("persistence must be at least 1."))
    crediting in (:signal, :label) ||
        throw(ArgumentError("crediting = :$crediting; one of :signal, :label."))
    times = event_merger_times(events)
    n_events = nrow(events)
    starts = credited_span_starts(events, crediting)
    has_span = starts !== nothing
    spans = UnitRange{Int}[]
    for i in 1:n_events
        if has_span
            push!(spans, Int(starts[i]):Int(events.label_end_index[i]))
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
    # Alarmed windows in window order, cut into maximal runs of consecutive
    # indices; `runs` holds position ranges into `alarmed`.
    alarmed = nrow(windows) == 0 ? Int[] : findall(==(1), windows.decision)
    alarmed = alarmed[sortperm(Int.(windows.window[alarmed]))]
    runs = alarm_runs(Int.(windows.window[alarmed]))
    overlaps(j, span) = begin
        r = Int(windows.row_start[alarmed[j]]):Int(windows.row_end[alarmed[j]])
        first(r) <= last(span) && first(span) <= last(r)
    end
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
        hits = Set(j for j in eachindex(alarmed) if overlaps(j, span))
        for j in hits
            in_span[j] = true
        end
        best_t = nothing
        best_w = 0
        for r in runs
            length(r) >= persistence || continue
            for a in first(r):(last(r)-persistence+1)
                sub = a:(a+persistence-1)
                any(in(hits), sub) || continue
                # The arrival completing the run; among windows completed by
                # the same arrival, the latest in mission time
                ts = [windows.complete_at[alarmed[j]] for j in sub]
                t = maximum(ts)
                k = sub[findlast(==(t), ts)]
                if best_t === nothing || t < best_t
                    best_t = t
                    best_w = alarmed[k]
                end
            end
        end
        if best_t === nothing
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
            latency_h = Dates.value(best_t - t_merger) / 3.6e6
            push!(
                out,
                (
                    labels[i],
                    times[i],
                    t_merger,
                    true,
                    Int(windows.window[best_w]),
                    best_t,
                    latency_h,
                    latency_h + processing_latency_hours,
                    Float64(windows.inference_wall_ms[best_w]),
                ),
            )
        end
    end
    # One alert can touch two events whose label spans overlap; the table
    # says so rather than counting each as detected on its own.
    raised = [ismissing(w) ? 0 : Int(w) for w in out.alarm_window]
    out.shared_alert = [w != 0 && count(==(w), raised) > 1 for w in raised]
    # Contiguous alarmed windows outside every span form one episode; only
    # those long enough to raise an alert are counted.
    outside = Int.(windows.window[alarmed[.!in_span]])
    n_false = count(r -> length(r) >= persistence, alarm_runs(outside))
    observation_days =
        nrow(windows) == 0 ? 0.0 :
        nrow(windows) * geometry.sample_rate^-1 * detector_step(windows) / 86400
    out.false_alarm_episodes = fill(n_false, n_events)
    out.false_alarms_per_30d =
        fill(observation_days == 0 ? NaN : n_false / observation_days * 30, n_events)
    out.alert_persistence = fill(Int(persistence), n_events)
    out.alert_crediting = fill(String(crediting), n_events)
    return out
end

"""
    credited_span_label(crediting) -> String

Legend name of the spans alerts are credited to under `crediting`.
"""
credited_span_label(crediting::Symbol) =
    crediting === :signal ? "Signal span" : "Labelled span"

"""
    credited_span_starts(events, crediting) -> Union{Nothing,Vector{Int}}

First payload rows of the credited spans of an event table: the column
`signal_start_index` under `crediting = :signal`, `label_start_index` under
`:label`; `nothing` when the table carries no label spans. Refuses a table
with spans but without the signal onset under `:signal`, and onsets outside
their label span.
"""
function credited_span_starts(events::DataFrame, crediting::Symbol)
    cols = names(events)
    ("label_start_index" in cols && "label_end_index" in cols) || return nothing
    crediting === :label && return Int.(events.label_start_index)
    "signal_start_index" in cols || throw(
        ArgumentError(
            "the event table has label spans but no signal_start_index column; " *
            "regenerate it with the labelling or generation stage, or credit the " *
            "whole label span with crediting = :label.",
        ),
    )
    starts = Int.(events.signal_start_index)
    for (i, (s, lo, hi)) in
        enumerate(zip(starts, events.label_start_index, events.label_end_index))
        lo <= s <= hi || throw(
            ArgumentError(
                "signal_start_index $s of event $i lies outside its label span $lo:$hi.",
            ),
        )
    end
    return starts
end

"""
    alarm_runs(indices) -> Vector{UnitRange{Int}}

Maximal runs of consecutive integers of the sorted vector `indices`, as
ranges of positions into it.
"""
function alarm_runs(indices::AbstractVector{<:Integer})
    runs = UnitRange{Int}[]
    isempty(indices) && return runs
    start = 1
    for j in 2:length(indices)
        if indices[j] != indices[j-1] + 1
            push!(runs, start:(j-1))
            start = j
        end
    end
    push!(runs, start:length(indices))
    return runs
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
