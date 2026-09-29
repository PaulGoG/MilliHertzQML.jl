module MilliHertzQML

using CSV: CSV
using DataFrames: DataFrames, DataFrame, nrow
using Dates: Dates
using Distributed: Distributed
using DrWatson: DrWatson
using FFTW: irfft, rfft, rfftfreq
using Flux: Flux
using Functors: Functors
using HDF5: HDF5, attributes, h5open
using InteractiveUtils: InteractiveUtils
using JLD2: JLD2, jldsave
using LinearAlgebra: LinearAlgebra
using Logging: NullLogger, with_logger
using PrecompileTools: @compile_workload, @setup_workload
using Random: Random, AbstractRNG, Xoshiro
using SHA: sha256
using Statistics: mean, median, quantile, std
using TOML: TOML
using TimerOutputs: TimerOutput, @timeit, print_timer
using UUIDs: uuid4
using Yao:
    Yao,
    AbstractBlock,
    H,
    Ry,
    Rz,
    X,
    Z,
    apply,
    chain,
    control,
    dispatch,
    dispatch!,
    expect,
    nparameters,
    put,
    zero_state
using Zygote: Zygote

export VariationalQuantumClassifier
export train_step!, predict_probability, predict, loss_function, accuracy
export weighted_bce, sample_loss, batch_gradient
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
export TIMER,
    report_timing,
    new_run_id,
    hardware_fingerprint,
    git_provenance,
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
export whitening_psd_from_sidecar, ReplayState, process_event!, windows_table, replay_run
export follow_run, detector_from_run, open_telemetry_run, alert_latency_table
export event_merger_times
export export_telemetry_payload, samples_per_batch, catalog_events

include("config.jl")
include("provenance.jl")
include("simulation.jl")
include("response.jl")
include("waveforms.jl")
include("model.jl")
include("training.jl")
include("data.jl")
include("persistence.jl")
include("evaluation.jl")
include("ldc.jl")
include("visualization.jl")
include("telemetry.jl")
include("stages/generation.jl")
include("stages/preprocessing.jl")
include("stages/labeling.jl")
include("stages/export_payload.jl")
include("stages/training.jl")
include("stages/inference.jl")

# Precompilation of the inference path — circuit construction, feature
# scaling, and the forward pass — which every script and every inference run
# enters first. The gradient path is left out: its Zygote tape dominates the
# precompilation time and is compiled once per training run in any case. The
# feature matrix is a literal, so the workload touches neither the filesystem
# nor a random stream beyond the seeded parameter initialisation.
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
    @compile_workload begin
        model = VariationalQuantumClassifier(2, 1; rng = Xoshiro(0))
        scaler = fit_scaler(features)
        encoded = encode_features(scaler, features)
        predict_probability(model, @view(encoded[1, :]))
        predict(model, @view(encoded[1, :]))
        predict_all(model, encoded; threaded = false)
    end
end

end # module
