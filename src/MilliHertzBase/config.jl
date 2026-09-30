# Settings of the gravitational-wave sections: synthetic telemetry
# generation, pre-processing of TDI products, truth labelling of LDC
# products, and the telemetry payload export.

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
    distance_min = cfgget(g, "mbhb_distance_min_gpc", 1.0; type = Float64, min = 1e-6)
    return (
        days = cfgget(g, "days", 30.0; type = Float64, min = 1e-6),
        fs = cfgget(g, "fs", 0.2; type = Float64, min = 1e-6),
        n_mbhb = cfgget(g, "n_mbhb", 5; type = Int, min = 0),
        n_gbs = cfgget(g, "n_gbs", 50; type = Int, min = 0),
        n_emris = cfgget(g, "n_emris", 5; type = Int, min = 0),
        output = resolvepath(
            cfgget(
                g,
                "output",
                "data/inputs/simulated_telemetry_complex.h5";
                type = String,
            ),
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
        response = cfgget(
            g,
            "response",
            "sky_averaged";
            type = String,
            choices = ("sky_averaged", "lisa"),
        ),
        mbhb_distance_min_gpc = distance_min,
        mbhb_distance_max_gpc = cfgget(
            g,
            "mbhb_distance_max_gpc",
            50.0;
            type = Float64,
            min = distance_min,
        ),
        label_channel = cfgget(
            g,
            "label_channel",
            "A";
            type = String,
            choices = ("A", "E", "network"),
        ),
        orbit_phase = cfgget(g, "orbit_phase", 0.0; type = Float64),
        constellation_phase = cfgget(g, "constellation_phase", 0.0; type = Float64),
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
            cfgget(
                p,
                "h5_file",
                "data/inputs/simulated_telemetry_complex.h5";
                type = String,
            ),
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
            choices = ("model", "channel", "ldc", "welch", "none"),
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
        psd_smoothing_dex = cfgget(p, "psd_smoothing_dex", 0.0; type = Float64, min = 0.0),
        feature_set = Symbol(
            cfgget(
                p,
                "feature_set",
                "whitened";
                type = String,
                choices = ("whitened", "paper", "bands"),
            ),
        ),
        band_edges_hz = check_band_edges(
            cfgget(p, "band_edges_hz", [1e-3, 5e-3, 1e-1]; type = AbstractVector),
        ),
        highpass_cutoff_hz = cfgget(
            p,
            "highpass_cutoff_hz",
            5e-4;
            type = Float64,
            min = 0.0,
        ),
        highpass_order = cfgget(p, "highpass_order", 8; type = Int, min = 1),
        edge_margin = cfgget(p, "edge_margin", 0.0; type = Float64, min = 0.0),
        low_band_hz = analysis_band(p, "low_band_hz", [1e-3, 5e-3]),
        high_band_hz = analysis_band(p, "high_band_hz", [5e-3, 1e-1]),
        output_prefix = cfgget(p, "output_prefix", "telemetry"; type = String),
    )
end

"""
    ldc_settings(config) -> NamedTuple

Validated `[ldc]` parameters of the truth-stream labelling.
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
`context_windows`, `psd_sidecar`, `psd_mode`, `psd_trailing_days`,
`psd_refresh_days`, `psd_segment_length`, `psd_edge_periods`, `poll_interval_sec`,
`producer_compat`, `processing_latency_hours`, `alert_persistence`,
`alert_crediting`, `events_csv`). `phase_span` of `[training]` is in units of ``\\pi``.
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
    psd_sidecar = cfgget(t, "psd_sidecar", ""; type = String)
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
        psd_sidecar = isempty(psd_sidecar) ? "" : resolvepath(psd_sidecar),
        poll_interval_sec = cfgget(t, "poll_interval_sec", 1.0; type = Float64, min = 1e-3),
        producer_compat = cfgget(t, "producer_compat", "1.0"; type = String),
        processing_latency_hours = cfgget(
            t,
            "processing_latency_hours",
            1.0;
            type = Float64,
            min = 0.0,
        ),
        alert_persistence = cfgget(t, "alert_persistence", 3; type = Int, min = 1),
        alert_crediting = cfgget(
            t,
            "alert_crediting",
            "signal";
            type = String,
            choices = ("signal", "label"),
        ),
        psd_mode = cfgget(
            t,
            "psd_mode",
            "sidecar";
            type = String,
            choices = ("sidecar", "trailing"),
        ),
        psd_trailing_days = cfgget(
            t,
            "psd_trailing_days",
            30.0;
            type = Float64,
            min = 1e-6,
        ),
        psd_refresh_days = cfgget(t, "psd_refresh_days", 1.0; type = Float64, min = 1e-6),
        psd_segment_length = cfgget(t, "psd_segment_length", 65536; type = Int, min = 2),
        psd_edge_periods = cfgget(t, "psd_edge_periods", 3.0; type = Float64, min = 0.0),
        events_csv = isempty(events_csv) ? "" : resolvepath(events_csv),
    )
end
