module MilliHertzQML

using StreamingInference:
    StreamingInference,
    AbstractTelemetryRun,
    AbstractWindowEstimator,
    AbstractWindowScorer,
    active_manifest_path,
    add!,
    alert_latency_table,
    analysis_band,
    animate_mission_replay,
    animation_theme,
    arrival_events,
    ArrivalEvent,
    backup_existing!,
    batch_rows,
    BatchRecord,
    cfgget,
    check_memory,
    chronological_split,
    condition_window,
    conditioning_rows,
    config_root,
    content_digest,
    contiguous_runs,
    Coverage,
    covered_fraction,
    covered_stretch,
    estimator_gap!,
    estimator_memory,
    EstimatorMemory,
    event_merger_times,
    event_metrics,
    event_symbol,
    extract_features,
    feature_geometry,
    feature_names,
    FeatureMap,
    FIGURE_COLORS,
    figure_roc,
    figure_score_distribution,
    figure_sensitivity,
    FIGURE_SIZE,
    figure_size,
    FIGURE_STROKES,
    figure_telemetry_alerts,
    figure_theme,
    figure_threshold_sweep,
    finalize_replay!,
    fixed_spans,
    follow_run,
    GapEvent,
    gaps_table,
    git_provenance,
    hardware_fingerprint,
    highpass_record,
    holes,
    inference_settings,
    interpolated_psd,
    layer_provenance,
    list_batches,
    load_config,
    load_data,
    load_features,
    manifest_sha256,
    matched_filter_snr,
    network_periodogram,
    MemoryTelemetryRun,
    new_run_id,
    newly_evaluable!,
    OrderedCommit,
    override,
    PANEL_HEIGHT,
    parameter_digest,
    parse_batch_name,
    PendingWindow,
    pipeline_paths,
    place_signal!,
    process_event!,
    product_table,
    project_root,
    provenance,
    provenance_path,
    read_batch,
    record_memory_estimate_gib,
    remove!,
    replay_run,
    replay_state,
    ReplayState,
    report_timing,
    reset_estimator!,
    resolvepath,
    resource_settings,
    roc_auc,
    roc_curve,
    rootrelative,
    row_time,
    run_geometry,
    run_state,
    RunGeometry,
    save_animation,
    save_figure,
    scale_to_snr,
    score_bounds,
    score_window,
    scored_at,
    section,
    select_threshold,
    smooth_psd,
    snapshot_manifest,
    span_labels,
    Stateful,
    Stateless,
    STRIP_HEIGHT,
    synthesize_noise,
    tapered_periodogram,
    threshold_rows,
    threshold_sweep,
    time_row,
    TIMER,
    TrailingWelch,
    welch_psd,
    whiten_record,
    window_features,
    window_labels,
    window_rows,
    WindowRecord,
    windows_table,
    windows_touching,
    WindowScheduler,
    with_pipeline_root,
    write_csv,
    write_toml
using MilliHertzBase:
    MilliHertzBase,
    AbstractDetectorResponse,
    CHANNEL_MODES,
    catalog_events,
    channel_catalog,
    channel_count,
    channel_names,
    channel_noise_psd,
    channel_suffix,
    confusion_psd,
    detectable_span,
    detectable_spans,
    detector_response,
    draw_extrinsic,
    export_telemetry_payload,
    figure_mission_trace,
    figure_telemetry_trace,
    generate_telemetry,
    generation_settings,
    GIGAPARSEC_SEC,
    instrument_psd,
    inverse_segment,
    label_truth_stream,
    ldc_confusion_psd,
    ldc_settings,
    ldc_tdi_psd,
    lisa_noise_psd,
    lisa_response,
    open_telemetry_run,
    phenoma_amplitude,
    phenoma_arrival_delay,
    phenoma_group_delay,
    phenoma_parameters,
    phenoma_phase,
    phenoma_physical_amplitude,
    phenoma_series,
    phenoma_spectrum,
    phenoma_start_frequency,
    phenoma_waveform,
    preprocess_record,
    preprocessing_settings,
    project_series,
    project_spectrum,
    read_catalog,
    read_tdi,
    samples_per_batch,
    signal_onsets,
    sky_averaged_response,
    SkyAveragedResponse,
    snr_peaks,
    source_frame,
    tdi_settings,
    tdi_to_aet,
    telemetry_settings,
    whitening_psd,
    whitening_psd_from_sidecar,
    windowed_snr
import StreamingInference: StreamingDetector, window_score, score_label
using CSV: CSV
using DataFrames: DataFrames, DataFrame
using Dates: Dates
using Distributed: Distributed
using DrWatson: DrWatson
using Functors: Functors
using JLD2: JLD2, jldsave
using LinearAlgebra: LinearAlgebra
using Optimisers: Optimisers
using PrecompileTools: @compile_workload, @setup_workload
using Random: Random, AbstractRNG, Xoshiro, randperm
using Statistics: quantile
using TOML: TOML
using TimerOutputs: @timeit
using Yao:
    Yao,
    AbstractArrayReg,
    AbstractBlock,
    ChainBlock,
    H,
    Ry,
    Rz,
    X,
    Z,
    apply,
    apply!,
    chain,
    control,
    dispatch,
    dispatch!,
    expect,
    nparameters,
    put,
    setiparams!,
    state,
    subblocks,
    zero_state
using Yao.AD: apply_back!
using Zygote: Zygote

export VariationalQuantumClassifier
export train_step!, predict_probability, predict, loss_function, accuracy
export weighted_bce, sample_loss, batch_gradient
export CircuitWorkspace, load_parameters!, predict_probability!, accumulate_gradient!
export gradient_tasks
export load_data, load_features, extract_features, feature_names
export FeatureScaler, fit_scaler, encode_features
export save_model, load_model
export chronological_split, roc_curve, roc_auc, contiguous_runs, event_metrics
export select_threshold, threshold_sweep, threshold_rows
export lisa_noise_psd, instrument_psd, confusion_psd, synthesize_noise
export matched_filter_snr, scale_to_snr, highpass_record, whiten_record
export tapered_periodogram, place_signal!, detectable_span
export phenoma_parameters, phenoma_amplitude, phenoma_phase, phenoma_group_delay
export phenoma_start_frequency,
    phenoma_waveform, phenoma_spectrum, phenoma_series, inverse_segment
export phenoma_physical_amplitude, phenoma_arrival_delay
export AbstractDetectorResponse, SkyAveragedResponse, channel_count, GIGAPARSEC_SEC
export sky_averaged_response, channel_noise_psd, draw_extrinsic, lisa_response
export source_frame, project_spectrum, project_series, detector_response
export ldc_tdi_psd, ldc_confusion_psd, tdi_to_aet, read_tdi, read_catalog
export welch_psd, smooth_psd, interpolated_psd, windowed_snr, snr_peaks
export detectable_spans, fixed_spans, signal_onsets, span_labels
export project_root, resolvepath, rootrelative, provenance_path, load_config
export cfgget, override, section
export analysis_band, pipeline_paths, feature_geometry
export generation_settings, preprocessing_settings, model_settings, training_settings
export inference_settings, ldc_settings, resource_settings, telemetry_settings
export tdi_settings, channel_names, channel_suffix, CHANNEL_MODES, network_periodogram
export TIMER,
    report_timing,
    new_run_id,
    hardware_fingerprint,
    git_provenance,
    layer_provenance,
    provenance,
    active_manifest_path,
    manifest_sha256,
    snapshot_manifest
export backup_existing!, write_toml, write_csv
export training_memory_estimate_gib, record_memory_estimate_gib, check_memory
export generate_telemetry, preprocess_record, label_truth_stream, channel_catalog
export whitening_psd, window_features, window_labels
export train_classifier, evaluate_classifier
export FIGURE_SIZE, PANEL_HEIGHT, STRIP_HEIGHT, figure_size, FIGURE_COLORS, FIGURE_STROKES
export figure_theme, save_figure
export figure_training_history, figure_mission_trace, figure_roc, figure_sensitivity
export figure_threshold_sweep
export figure_score_distribution, figure_telemetry_trace, figure_telemetry_alerts
export figure_loss_survival, figure_seed_spread, figure_gap_study, figure_grid_seeds
export animation_theme, save_animation
export animate_training_history, animate_mission_replay
export RunGeometry, BatchRecord, ArrivalEvent, WindowRecord, parse_batch_name, batch_rows
export row_time,
    time_row, event_symbol, AbstractTelemetryRun, run_geometry, list_batches, read_batch
export arrival_events, run_state, MemoryTelemetryRun, Coverage, add!, remove!
export covered_fraction, holes, covered_stretch, WindowScheduler, window_rows
export conditioning_rows
export windows_touching, newly_evaluable!, StreamingDetector, score_window, TrailingWelch
export AbstractWindowEstimator, AbstractWindowScorer, EstimatorMemory, Stateless, Stateful
export estimator_memory, window_score, score_label, score_bounds, FeatureMap
export condition_window, VQCScorer
export reset_estimator!, GapEvent, estimator_gap!, PendingWindow, OrderedCommit
export finalize_replay!, gaps_table, replay_state
export content_digest, parameter_digest, product_table, scored_at
export with_pipeline_root, config_root
export whitening_psd_from_sidecar, ReplayState, process_event!, windows_table, replay_run
export follow_run, detector_from_run, open_telemetry_run, alert_latency_table
export event_merger_times
export export_telemetry_payload, samples_per_batch, catalog_events

include("settings.jl")
include("model.jl")
include("circuit.jl")
include("training.jl")
include("scaler.jl")
include("persistence.jl")
include("visualization.jl")
include("vqc_scorer.jl")
include("stages/training.jl")
include("stages/inference.jl")

# Precompilation of the paths every script enters: circuit construction,
# feature scaling, the forward pass (non-mutating and in place) and the
# adjoint batch gradient. The adjoint kernel is called directly, because
# `batch_gradient` also holds the Zygote reference path, and a Zygote
# pullback generated while precompiling fails at run time (a BoundsError in
# its gradient accumulation); the tape is compiled where it is used. The
# feature matrix is a literal, so the workload touches neither the
# filesystem nor a random stream beyond the seeded parameter initialisation.
@setup_workload begin
    features = Float32[
        0.10 0.90
        0.35 0.70
        0.60 0.45
        0.85 0.20
        0.20 0.65
        0.55 0.30
        0.75 0.95
        0.40 0.05
    ]
    labels = [0, 1, 0, 1, 0, 0, 1, 0]
    @compile_workload begin
        model = VariationalQuantumClassifier(2, 1; rng = Xoshiro(0))
        scaler = fit_scaler(features)
        encoded = encode_features(scaler, features)
        predict_probability(model, @view(encoded[1, :]))
        predict(model, @view(encoded[1, :]))
        predict_all(model, encoded; threaded = false)
        adjoint_batch_gradient(model, encoded, labels, 1.5, false, GRADIENT_CHUNK, nothing)
    end
end

end # module
