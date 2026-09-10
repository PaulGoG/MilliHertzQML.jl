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
export load_data, load_features, extract_features, feature_names
export FeatureScaler, fit_scaler, encode_features
export save_model, load_model
export chronological_split, roc_curve, roc_auc, contiguous_runs, event_metrics
export select_threshold
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
export inference_settings, ldc_settings, resource_settings
export TIMER, report_timing, new_run_id, hardware_fingerprint, git_provenance, provenance
export backup_existing!, write_toml, write_csv
export training_memory_estimate_gib, record_memory_estimate_gib, check_memory
export generate_telemetry, preprocess_record, label_truth_stream
export whitening_psd, window_features, window_labels
export train_classifier, evaluate_classifier

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
include("stages/generation.jl")
include("stages/preprocessing.jl")
include("stages/labeling.jl")
include("stages/training.jl")
include("stages/inference.jl")

end # module
