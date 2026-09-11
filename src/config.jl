# src/config.jl — TOML configuration: loading, validated key access, path
# resolution against the package root, and the typed settings of every
# pipeline section. The TOML file is the single source of truth of every
# tunable; the scripts add only a run identifier, a test-mode switch, and
# the location of external inputs.

"""
    project_root() -> String

Directory of the package (the pipeline root): the parent of `src/`. Every
relative path of a configuration resolves against it, whichever
environment (`scripts/`, `test/`, `docs/`, `bench/`) is active.
"""
project_root() = dirname(@__DIR__)

"""
    resolvepath(p) -> String

`p` resolved against [`project_root`](@ref) unless it is already absolute.
"""
resolvepath(p::AbstractString) = isabspath(p) ? String(p) : joinpath(project_root(), p)

"""
    rootrelative(p) -> String

`p` expressed relative to [`project_root`](@ref) when it lies inside it,
otherwise the absolute path unchanged; used when persisting paths in
provenance snapshots so that run artifacts stay portable across machines.
"""
function rootrelative(p::AbstractString)
    ap = abspath(p)
    root = project_root()
    return startswith(ap, root) ? relpath(ap, root) : ap
end

"""
    load_config(path) -> Dict{String, Any}

Parse the TOML configuration at `path`, failing fast when the file is
absent.
"""
function load_config(path::AbstractString)
    isfile(path) || throw(ArgumentError("configuration file not found: $path"))
    return TOML.parsefile(path)
end

"""
    cfgget(section, key, default; type = Any, min = nothing, max = nothing, choices = nothing)

Read `key` from a configuration `section`, falling back to `default` when
the key is absent. Validates the value against an expected `type`, optional
inclusive bounds, and an optional set of admissible `choices`, throwing an
`ArgumentError` naming the offending key on any violation. Numeric values
are converted to `type` when the conversion is exact.
"""
function cfgget(
    section::AbstractDict,
    key::AbstractString,
    default;
    type::Type = Any,
    min = nothing,
    max = nothing,
    choices = nothing,
)
    value = get(section, key, default)
    if type !== Any && !(value isa type)
        if value isa Real && type <: Real
            value = convert(type, value)
        else
            throw(
                ArgumentError(
                    "configuration key `$key` has value $(repr(value)); expected type $type.",
                ),
            )
        end
    end
    min !== nothing &&
        value < min &&
        throw(ArgumentError("configuration key `$key` = $value; must be >= $min."))
    max !== nothing &&
        value > max &&
        throw(ArgumentError("configuration key `$key` = $value; must be <= $max."))
    choices !== nothing &&
        !(value in choices) &&
        throw(
            ArgumentError(
                "configuration key `$key` = $(repr(value)); must be one of " *
                join(repr.(choices), " | ") *
                ".",
            ),
        )
    return value
end

"""
    override(cli_value, cfg_value)

Precedence of an explicit command-line value over the configuration:
`cli_value` unless it is `nothing`.
"""
override(cli_value, cfg_value) = cli_value !== nothing ? cli_value : cfg_value

"""
    section(config, name) -> Dict{String, Any}

The table `name` of `config`, or an empty table when absent.
"""
section(config::AbstractDict, name::AbstractString) =
    get(config, name, Dict{String,Any}())::AbstractDict

"""
    analysis_band(section, key, default) -> Tuple{Float64,Float64}

Two-element ascending positive frequency band [Hz] from the configuration.
"""
function analysis_band(section::AbstractDict, key::AbstractString, default)
    band = cfgget(section, key, default; type = AbstractVector)
    (length(band) == 2 && all(x -> x isa Real, band) && 0 < band[1] < band[2]) || throw(
        ArgumentError(
            "configuration key `$key` = $(repr(band)); expected two ascending positive frequencies [Hz].",
        ),
    )
    return (Float64(band[1]), Float64(band[2]))
end

"""
    pipeline_paths(config) -> NamedTuple

Output roots of the pipeline from the `[paths]` section — `inputs`,
`models`, `plots`, `results` — resolved against the package root. Absent
keys fall back to the standard tree (`data/inputs`, `models`,
`data/outputs/plots`, `data/outputs/results`).
"""
function pipeline_paths(config::AbstractDict)
    p = section(config, "paths")
    return (
        inputs = resolvepath(
            cfgget(p, "inputs", joinpath("data", "inputs"); type = String),
        ),
        models = resolvepath(cfgget(p, "models", "models"; type = String)),
        plots = resolvepath(
            cfgget(p, "plots", joinpath("data", "outputs", "plots"); type = String),
        ),
        results = resolvepath(
            cfgget(p, "results", joinpath("data", "outputs", "results"); type = String),
        ),
    )
end

"""
    generation_settings(config) -> NamedTuple

Validated `[generation]` parameters of the telemetry simulator.
"""
function generation_settings(config::AbstractDict)
    g = section(config, "generation")
    snr_min = cfgget(g, "snr_min", 8.0; type = Float64, min = 1e-3)
    gb_snr_min = cfgget(g, "gb_snr_min", 1.0; type = Float64, min = 1e-3)
    emri_snr_min = cfgget(g, "emri_snr_min", 2.0; type = Float64, min = 1e-3)
    mass_min = cfgget(g, "mbhb_total_mass_min", 1e5; type = Float64, min = 1.0)
    return (
        days = cfgget(g, "days", 30.0; type = Float64, min = 1e-6),
        fs = cfgget(g, "fs", 0.2; type = Float64, min = 1e-6),
        n_mbhb = cfgget(g, "n_mbhb", 5; type = Int, min = 0),
        n_gbs = cfgget(g, "n_gbs", 50; type = Int, min = 0),
        n_emris = cfgget(g, "n_emris", 5; type = Int, min = 0),
        output = resolvepath(
            cfgget(g, "output", "data/inputs/simulated_telemetry.h5"; type = String),
        ),
        seed = cfgget(g, "seed", 42; type = Int),
        observation_years = cfgget(
            g,
            "observation_years",
            1.0;
            type = Float64,
            choices = (0.5, 1.0, 2.0, 4.0),
        ),
        snr_min = snr_min,
        snr_max = cfgget(g, "snr_max", 50.0; type = Float64, min = snr_min),
        gb_snr_min = gb_snr_min,
        gb_snr_max = cfgget(g, "gb_snr_max", 10.0; type = Float64, min = gb_snr_min),
        emri_snr_min = emri_snr_min,
        emri_snr_max = cfgget(g, "emri_snr_max", 10.0; type = Float64, min = emri_snr_min),
        label_before_sec = cfgget(
            g,
            "label_before_sec",
            43200.0;
            type = Float64,
            min = 0.0,
        ),
        label_after_sec = cfgget(g, "label_after_sec", 3600.0; type = Float64, min = 0.0),
        mbhb_duration_days = cfgget(
            g,
            "mbhb_duration_days",
            2.0;
            type = Float64,
            min = 1e-3,
        ),
        noise_f_min_hz = cfgget(g, "noise_f_min_hz", 1e-5; type = Float64, min = 0.0),
        mbhb_total_mass_min = mass_min,
        mbhb_total_mass_max = cfgget(
            g,
            "mbhb_total_mass_max",
            1e7;
            type = Float64,
            min = mass_min,
        ),
        mbhb_mass_ratio_max = cfgget(
            g,
            "mbhb_mass_ratio_max",
            10.0;
            type = Float64,
            min = 1.0,
        ),
        nyquist_taper = cfgget(
            g,
            "nyquist_taper",
            0.9;
            type = Float64,
            min = 0.3,
            max = 1.0,
        ),
        label_span = cfgget(
            g,
            "label_span",
            "detectable";
            type = String,
            choices = ("detectable", "injection", "fixed"),
        ),
        label_snr_threshold = cfgget(
            g,
            "label_snr_threshold",
            5.0;
            type = Float64,
            min = 1e-3,
        ),
        label_window_size = cfgget(g, "label_window_size", 1000; type = Int, min = 2),
        label_step = cfgget(g, "label_step", 10; type = Int, min = 1),
    )
end

"""
    preprocessing_settings(config) -> NamedTuple

Validated `[preprocessing]` parameters: input product and TDI group, window
geometry, whitening PSD selection, analysis bands, record high-pass, and
feature set.
"""
function preprocessing_settings(config::AbstractDict)
    p = section(config, "preprocessing")
    window_size = cfgget(p, "window_size", 1000; type = Int, min = 2)
    step_size = cfgget(p, "step_size", 100; type = Int, min = 1)
    step_size <= window_size ||
        throw(ArgumentError("step_size = $step_size exceeds window_size = $window_size."))
    return (
        h5_file = resolvepath(
            cfgget(p, "h5_file", "data/inputs/simulated_telemetry.h5"; type = String),
        ),
        tdi_group = cfgget(p, "tdi_group", "obs/tdi"; type = String),
        window_size = window_size,
        step_size = step_size,
        sample_rate = cfgget(p, "sample_rate", nothing; type = Union{Nothing,Float64}),
        psd = cfgget(
            p,
            "psd",
            "model";
            type = String,
            choices = ("model", "ldc", "welch", "none"),
        ),
        observation_years = cfgget(
            p,
            "observation_years",
            1.0;
            type = Float64,
            choices = (0.5, 1.0, 2.0, 4.0),
        ),
        ldc_model = cfgget(p, "ldc_model", "sangria"; type = String),
        ldc_tdi2 = cfgget(p, "ldc_tdi2", false; type = Bool),
        ldc_observation_years = cfgget(
            p,
            "ldc_observation_years",
            0.0;
            type = Float64,
            min = 0.0,
        ),
        welch_segment_length = cfgget(
            p,
            "welch_segment_length",
            65536;
            type = Int,
            min = 2,
        ),
        feature_set = Symbol(
            cfgget(
                p,
                "feature_set",
                "whitened";
                type = String,
                choices = ("whitened", "paper"),
            ),
        ),
        highpass_cutoff_hz = cfgget(
            p,
            "highpass_cutoff_hz",
            5e-4;
            type = Float64,
            min = 0.0,
        ),
        highpass_order = cfgget(p, "highpass_order", 8; type = Int, min = 1),
        low_band_hz = analysis_band(p, "low_band_hz", [1e-3, 5e-3]),
        high_band_hz = analysis_band(p, "high_band_hz", [5e-3, 1e-1]),
        output_prefix = cfgget(p, "output_prefix", "telemetry"; type = String),
    )
end

"""
    model_settings(config) -> NamedTuple

Validated `[model]` parameters of the classifier.
"""
function model_settings(config::AbstractDict)
    m = section(config, "model")
    return (
        n_qubits = cfgget(m, "n_qubits", 4; type = Int, min = 2, max = 24),
        n_layers = cfgget(m, "n_layers", 4; type = Int, min = 1),
    )
end

"""
    training_settings(config) -> NamedTuple

Validated `[training]` parameters: inputs, optimizer, chronological blocks,
class weighting, threshold criterion, scaler quantiles, test-mode caps,
threading.
"""
function training_settings(config::AbstractDict)
    t = section(config, "training")
    train_fraction =
        cfgget(t, "train_fraction", 0.7; type = Float64, min = 0.05, max = 0.95)
    validation_fraction =
        cfgget(t, "validation_fraction", 0.15; type = Float64, min = 0.01, max = 0.5)
    train_fraction + validation_fraction < 1 || throw(
        ArgumentError(
            "train_fraction + validation_fraction = $(train_fraction + validation_fraction); " *
            "must leave a test block.",
        ),
    )
    quantiles = cfgget(t, "scaler_quantiles", [0.005, 0.995]; type = AbstractVector)
    (
        length(quantiles) == 2 &&
        all(q -> q isa Real, quantiles) &&
        0 <= quantiles[1] < quantiles[2] <= 1
    ) || throw(
        ArgumentError(
            "configuration key `scaler_quantiles` = $(repr(quantiles)); " *
            "expected two ascending values in [0, 1].",
        ),
    )
    return (
        train_features = resolvepath(
            cfgget(t, "train_features", "data/inputs/train_features.csv"; type = String),
        ),
        train_labels = resolvepath(
            cfgget(t, "train_labels", "data/inputs/train_labels.csv"; type = String),
        ),
        epochs = cfgget(t, "epochs", 100; type = Int, min = 1),
        batch_size = cfgget(t, "batch_size", 32; type = Int, min = 1),
        learning_rate = cfgget(t, "learning_rate", 0.01; type = Float64, min = 1e-8),
        lr_decay = cfgget(t, "lr_decay", 0.95; type = Float64, min = 1e-3, max = 1.0),
        patience = cfgget(t, "patience", 12; type = Int, min = 1),
        train_fraction = train_fraction,
        validation_fraction = validation_fraction,
        class_weight = cfgget(
            t,
            "class_weight",
            "balanced";
            type = String,
            choices = ("balanced", "none"),
        ),
        threshold_criterion = cfgget(
            t,
            "threshold_criterion",
            "far";
            type = String,
            choices = ("far", "fpr", "youden"),
        ),
        target_far_per_30d = cfgget(
            t,
            "target_far_per_30d",
            3.0;
            type = Float64,
            min = 0.0,
        ),
        target_fpr = cfgget(t, "target_fpr", 0.05; type = Float64, min = 0.0, max = 1.0),
        scaler_quantiles = (Float64(quantiles[1]), Float64(quantiles[2])),
        test_mode_samples = cfgget(t, "test_mode_samples", 5000; type = Int, min = 1),
        test_mode_epochs = cfgget(t, "test_mode_epochs", 20; type = Int, min = 1),
        threaded = cfgget(t, "threaded", true; type = Bool),
        seed = cfgget(t, "seed", 42; type = Int),
    )
end

"""
    inference_settings(config) -> NamedTuple

Validated `[inference]` parameters: feature and label tables, block
selection, and the window geometry used when a feature table has no
sidecar.
"""
function inference_settings(config::AbstractDict)
    i = section(config, "inference")
    return (
        features = resolvepath(
            cfgget(i, "features", "data/inputs/inference_features.csv"; type = String),
        ),
        labels = cfgget(i, "labels", "data/inputs/inference_labels.csv"; type = String),
        block = cfgget(
            i,
            "block",
            "all";
            type = String,
            choices = ("all", "validation", "test"),
        ),
        step_size = cfgget(i, "step_size", 100; type = Int, min = 1),
        sample_rate = cfgget(i, "sample_rate", 0.2; type = Float64, min = 1e-6),
    )
end

"""
    ldc_settings(config) -> NamedTuple

Validated `[ldc]` parameters of the truth-stream labeling.
"""
function ldc_settings(config::AbstractDict)
    l = section(config, "ldc")
    h5_file = cfgget(l, "h5_file", ""; type = String)
    return (
        h5_file = isempty(h5_file) ? "" : resolvepath(h5_file),
        truth_group = cfgget(l, "truth_group", "sky/mbhb/tdi"; type = String),
        catalog_group = cfgget(l, "catalog_group", "sky/mbhb/cat"; type = String),
        psd_model = cfgget(l, "psd_model", "sangria"; type = String),
        tdi2 = cfgget(l, "tdi2", false; type = Bool),
        observation_years = cfgget(l, "observation_years", 0.0; type = Float64, min = 0.0),
        label_span = cfgget(
            l,
            "label_span",
            "fixed";
            type = String,
            choices = ("fixed", "detectable"),
        ),
        label_before_sec = cfgget(
            l,
            "label_before_sec",
            4 * 86400.0;
            type = Float64,
            min = 0.0,
        ),
        label_after_sec = cfgget(
            l,
            "label_after_sec",
            27 * 60.0;
            type = Float64,
            min = 0.0,
        ),
        label_snr_threshold = cfgget(
            l,
            "label_snr_threshold",
            5.0;
            type = Float64,
            min = 1e-6,
        ),
        label_window_size = cfgget(l, "label_window_size", 1000; type = Int, min = 2),
        label_step = cfgget(l, "label_step", 10; type = Int, min = 1),
        peak_min_separation_sec = cfgget(
            l,
            "peak_min_separation_sec",
            86400.0;
            type = Float64,
            min = 0.0,
        ),
        merger_snr_threshold = cfgget(
            l,
            "merger_snr_threshold",
            8.0;
            type = Float64,
            min = 1e-6,
        ),
        precursor_ratio = cfgget(
            l,
            "precursor_ratio",
            0.1;
            type = Float64,
            min = 0.0,
            max = 1.0,
        ),
        output_prefix = cfgget(l, "output_prefix", "ldc"; type = String),
    )
end

"""
    telemetry_settings(config) -> NamedTuple

Validated `[telemetry]` parameters of the coupling to a DeepSpaceTelemetry
run: the exported scenario geometry (`segment_duration_sec`, `batch_size`,
`start_sim_time`, `output_prefix`) and the consumer's replay settings
(`run_dir`, `mode`, `min_coverage`, `tdi_gap_dilation_sec`,
`context_windows`, `poll_interval_sec`, `producer_compat`,
`processing_latency_hours`, `events_csv`).
"""
function telemetry_settings(config::AbstractDict)
    t = section(config, "telemetry")
    start = cfgget(t, "start_sim_time", "2035-01-01T00:00:00"; type = String)
    start_sim_time = tryparse(Dates.DateTime, start)
    start_sim_time === nothing && throw(
        ArgumentError(
            "configuration key `start_sim_time` = $(repr(start)); expected an ISO-8601 datetime.",
        ),
    )
    run_dir = cfgget(t, "run_dir", ""; type = String)
    events_csv = cfgget(t, "events_csv", ""; type = String)
    return (
        segment_duration_sec = cfgget(
            t,
            "segment_duration_sec",
            50.0;
            type = Float64,
            min = 1e-6,
        ),
        batch_size = cfgget(t, "batch_size", 10; type = Int, min = 1),
        start_sim_time = start_sim_time,
        output_prefix = cfgget(t, "output_prefix", "telemetry"; type = String),
        run_dir = isempty(run_dir) ? "" : resolvepath(run_dir),
        mode = cfgget(t, "mode", "replay"; type = String, choices = ("replay", "live")),
        min_coverage = cfgget(
            t,
            "min_coverage",
            1.0;
            type = Float64,
            min = 1e-6,
            max = 1.0,
        ),
        tdi_gap_dilation_sec = cfgget(
            t,
            "tdi_gap_dilation_sec",
            100.0;
            type = Float64,
            min = 0.0,
        ),
        context_windows = cfgget(t, "context_windows", 4; type = Int, min = 0),
        poll_interval_sec = cfgget(t, "poll_interval_sec", 1.0; type = Float64, min = 1e-3),
        producer_compat = cfgget(t, "producer_compat", "1.0"; type = String),
        processing_latency_hours = cfgget(
            t,
            "processing_latency_hours",
            1.0;
            type = Float64,
            min = 0.0,
        ),
        events_csv = isempty(events_csv) ? "" : resolvepath(events_csv),
    )
end

"""
    feature_geometry(features_path, config) -> NamedTuple

Window geometry (`window_size`, `step_size`, `sample_rate`) of a feature
table, read from the sidecar `<stem>.toml` written beside it by the
pre-processor; falls back to the `[preprocessing]` section with a warning
when the sidecar is absent.
"""
function feature_geometry(features_path::AbstractString, config::AbstractDict)
    sidecar = replace(features_path, r"\.csv$" => ".toml")
    sec = section(config, "preprocessing")
    if isfile(sidecar)
        sec = get(TOML.parsefile(sidecar), "features", Dict{String,Any}())
    else
        @warn "no feature sidecar at $sidecar; window geometry taken from [preprocessing]."
    end
    return (
        window_size = cfgget(sec, "window_size", 1000; type = Int, min = 2),
        step_size = cfgget(sec, "step_size", 100; type = Int, min = 1),
        sample_rate = cfgget(sec, "sample_rate", 0.2; type = Float64, min = 1e-6),
    )
end
