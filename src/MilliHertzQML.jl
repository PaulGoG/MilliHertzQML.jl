module MilliHertzQML

using CSV: CSV
using DataFrames: DataFrames, DataFrame
using FFTW: irfft, rfft, rfftfreq
using Flux: Flux
using Functors: Functors
using HDF5: HDF5, attributes, h5open
using JLD2: JLD2, jldsave
using Random: Random, AbstractRNG
using Statistics: mean, median, quantile, std
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

include("simulation.jl")
include("waveforms.jl")
include("model.jl")
include("training.jl")
include("data.jl")
include("persistence.jl")
include("evaluation.jl")
include("ldc.jl")

end # module
