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
using Random: Random, AbstractRNG, Xoshiro
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
export phenoma_start_frequency, phenoma_waveform
export ldc_tdi_psd, ldc_confusion_psd, tdi_to_aet, read_tdi, read_catalog
export welch_psd, interpolated_psd, windowed_snr, snr_peaks
export detectable_spans, fixed_spans, span_labels
export project_root, resolvepath, rootrelative, load_config, cfgget, override, section
export analysis_band, pipeline_paths, feature_geometry
export generation_settings, preprocessing_settings, model_settings, training_settings
export inference_settings, ldc_settings, resource_settings, telemetry_settings
export TIMER, report_timing, new_run_id, hardware_fingerprint, git_provenance, provenance
export backup_existing!, write_toml, write_csv
export training_memory_estimate_gib, record_memory_estimate_gib, check_memory
export generate_telemetry, preprocess_record, label_truth_stream
export whitening_psd, window_features, window_labels
export train_classifier, evaluate_classifier
export FIGURE_WIDTH_MM, FIGURE_COLORS, figure_theme, save_figure
export figure_training_history, figure_mission_trace, figure_roc, figure_sensitivity
export figure_threshold_sweep
export figure_score_distribution, figure_telemetry_trace, figure_telemetry_alerts
export RunGeometry, BatchRecord, ArrivalEvent, WindowRecord, parse_batch_name, batch_rows
export row_time, event_symbol, AbstractTelemetryRun, run_geometry, list_batches, read_batch
export arrival_events, run_state, MemoryTelemetryRun, Coverage, add!, remove!
export covered_fraction, holes, covered_stretch, WindowScheduler, window_rows
export windows_touching, newly_evaluable!, StreamingDetector, score_window
export whitening_psd_from_sidecar, ReplayState, process_event!, windows_table, replay_run
export follow_run, detector_from_run, open_telemetry_run, alert_latency_table
export event_merger_times
export export_telemetry_payload, samples_per_batch, catalog_events

include("config.jl")
include("provenance.jl")
include("simulation.jl")
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

end # module
