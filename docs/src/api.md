# API reference

The package is organised in three layers. `MilliHertzQML` re-exports the
public names of all three, so `using MilliHertzQML` gives the whole
interface.

## Domain-general layer (`MilliHertzQML.StreamingInference`)

Configuration and provenance, signal processing of sampled records,
spectral features, the estimator interface between the conditioning chain
and the method applied to each window, evaluation and decision thresholds,
the streamed consumption of a delivered record, and the figure interface of
evaluation, scores and replays.

```@autodocs
Modules = [MilliHertzQML.StreamingInference]
```

## Gravitational-wave layer (`MilliHertzQML.MilliHertzBase`)

The LISA noise model and detector response, IMRPhenomA waveforms, LDC
products and TDI channels, the whitening PSD of a TDI record, the
generation, pre-processing, labelling and payload-export stages, and the
coupling to the DeepSpaceTelemetry producer.

```@autodocs
Modules = [MilliHertzQML.MilliHertzBase]
```

## Classifier (`MilliHertzQML`)

The variational quantum classifier, its training and persistence, the
feature scaler, the classifier as a window scorer, the training and
inference stages, and the figures of its studies.

```@autodocs
Modules = [MilliHertzQML]
```
