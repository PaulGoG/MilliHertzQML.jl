"""
    MilliHertzBase

Gravitational-wave layer of the pipeline, on the domain-general
[`StreamingInference`](@ref) layer: the LISA noise model and detector
response, IMRPhenomA waveforms, LDC products and TDI channels, the
whitening PSD of a TDI record, the generation, pre-processing, labelling
and payload-export stages, and the coupling to the DeepSpaceTelemetry
producer.
"""
module MilliHertzBase

using ..StreamingInference
using ..StreamingInference: check_band_edges, edge_margin_windows, window_count
using CSV: CSV
using DataFrames: DataFrames, DataFrame, nrow
using Dates: Dates
using DrWatson: DrWatson
using FFTW: irfft, rfftfreq
using HDF5: HDF5, attributes, h5open
using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG, Xoshiro
using TOML: TOML
using TimerOutputs: @timeit

export lisa_noise_psd, instrument_psd, confusion_psd, detectable_span, phenoma_parameters
export phenoma_amplitude, phenoma_phase, phenoma_group_delay, phenoma_start_frequency
export phenoma_waveform, phenoma_spectrum, phenoma_series, inverse_segment
export phenoma_physical_amplitude, phenoma_arrival_delay, AbstractDetectorResponse
export SkyAveragedResponse, channel_count, GIGAPARSEC_SEC, sky_averaged_response
export channel_noise_psd, draw_extrinsic, lisa_response, detector_response, ldc_tdi_psd
export ldc_confusion_psd, tdi_to_aet, read_tdi, read_catalog, windowed_snr, snr_peaks
export detectable_spans, signal_onsets, generation_settings, preprocessing_settings
export ldc_settings, telemetry_settings, generate_telemetry, preprocess_record
export label_truth_stream, channel_catalog, whitening_psd, figure_mission_trace
export figure_telemetry_trace, whitening_psd_from_sidecar, open_telemetry_run
export export_telemetry_payload, samples_per_batch, source_frame, project_spectrum
export project_series, catalog_events

include("config.jl")
include("noise.jl")
include("response.jl")
include("waveforms.jl")
include("ldc.jl")
include("whitening.jl")
include("telemetry.jl")
include("visualization.jl")
include("stages/generation.jl")
include("stages/preprocessing.jl")
include("stages/labeling.jl")
include("stages/export_payload.jl")

end # module
