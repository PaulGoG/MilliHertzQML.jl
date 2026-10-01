"""
    StreamingInference

Domain-general layer of the pipeline: configuration and provenance,
signal processing of sampled records, spectral features, the estimator
interface between the conditioning chain and the method applied to each
window, evaluation and decision thresholds, and the streamed consumption
of a delivered record (run interface, coverage, window scheduling, causal
whitening, replay, alert tables). Nothing in it depends on a physical
domain or on a particular estimator.
"""
module StreamingInference

using CSV: CSV
using DataFrames: DataFrames, DataFrame, nrow
using Dates: Dates
using Distributed: Distributed
using DrWatson: DrWatson
using FFTW: irfft, rfft, rfftfreq
using InteractiveUtils: InteractiveUtils
using LinearAlgebra: LinearAlgebra
using Logging: NullLogger, with_logger
using Random: Random, AbstractRNG
using SHA: sha256
using Statistics: mean, median, quantile, std
using TOML: TOML
using TimerOutputs: TimerOutput, @timeit, print_timer
using UUIDs: uuid4

export load_data, load_features, extract_features, feature_names, chronological_split
export roc_curve, roc_auc, contiguous_runs, event_metrics, select_threshold
export threshold_sweep, threshold_rows, synthesize_noise, matched_filter_snr, scale_to_snr
export highpass_record, whiten_record, tapered_periodogram, place_signal!, welch_psd
export smooth_psd, interpolated_psd, fixed_spans, span_labels, project_root, resolvepath
export rootrelative, provenance_path, load_config, cfgget, override, section
export analysis_band, pipeline_paths, feature_geometry, inference_settings
export resource_settings, TIMER, report_timing, new_run_id, hardware_fingerprint
export git_provenance, provenance, active_manifest_path, manifest_sha256
export snapshot_manifest, backup_existing!, write_toml, write_csv
export record_memory_estimate_gib, check_memory, window_features, window_labels
export FIGURE_SIZE, PANEL_HEIGHT, STRIP_HEIGHT, figure_size, FIGURE_COLORS, FIGURE_STROKES
export figure_theme, save_figure, figure_roc, figure_sensitivity, figure_threshold_sweep
export figure_score_distribution, figure_telemetry_alerts, animation_theme, save_animation
export animate_mission_replay, RunGeometry, BatchRecord, ArrivalEvent, WindowRecord
export parse_batch_name, batch_rows, row_time, time_row, event_symbol
export AbstractTelemetryRun, run_geometry, list_batches, read_batch, arrival_events
export run_state, MemoryTelemetryRun, Coverage, add!, remove!, covered_fraction, holes
export covered_stretch, WindowScheduler, window_rows, conditioning_rows, windows_touching
export newly_evaluable!, StreamingDetector, score_window, TrailingWelch
export AbstractWindowEstimator, AbstractWindowScorer, EstimatorMemory, Stateless, Stateful
export estimator_memory, window_score, score_label, score_bounds, FeatureMap
export condition_window, ReplayState, process_event!, windows_table, replay_run
export follow_run, alert_latency_table, event_merger_times
export reset_estimator!, GapEvent, estimator_gap!, PendingWindow, OrderedCommit
export finalize_replay!, gaps_table, replay_state

include("config.jl")
include("provenance.jl")
include("dsp.jl")
include("features.jl")
include("spans.jl")
include("evaluation.jl")
include("windows.jl")
include("estimators.jl")
include("visualization.jl")
include("telemetry.jl")

end # module
