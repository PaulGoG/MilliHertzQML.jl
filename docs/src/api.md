# API reference

The package is the classifier layer of three. The domain-general layer,
[StreamingInference.jl](https://github.com/PaulGoG/StreamingInference.jl)
([manual](https://PaulGoG.github.io/StreamingInference.jl/dev/)), holds
configuration and provenance, signal processing, features, the estimator
interface, evaluation and the streamed replay; the gravitational-wave
layer, [MilliHertzBase.jl](https://github.com/PaulGoG/MilliHertzBase.jl)
([manual](https://PaulGoG.github.io/MilliHertzBase.jl/dev/)), holds the
noise model, waveforms, detector response, LISA Data Challenge (LDC)
products, the generation,
pre-processing, labelling and payload-export stages, and the coupling to
the DeepSpaceTelemetry producer. `MilliHertzQML` re-exports every name
the two layers export. The names they declare `public` without exporting
them, such as `credited_span_times` and `dense_log_ticks` of
StreamingInference.jl or `record_groups` and `catalog_events` of
MilliHertzBase.jl, are reached through the modules
`MilliHertzQML.StreamingInference` and `MilliHertzQML.MilliHertzBase`.

## Classifier (`MilliHertzQML`)

The variational quantum classifier, its training and persistence, the
feature scaler, the classifier as a window scorer, the training,
inference, calibration and telemetry-replay stages, and the figures of
its studies.

```@autodocs
Modules = [MilliHertzQML]
```
