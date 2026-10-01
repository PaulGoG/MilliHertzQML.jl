# API reference

The package is the classifier layer of three. The domain-general layer,
[StreamingInference.jl](https://github.com/PaulGoG/StreamingInference.jl)
([manual](https://PaulGoG.github.io/StreamingInference.jl/dev/)), holds
configuration and provenance, signal processing, features, the estimator
interface, evaluation and the streamed replay; the gravitational-wave
layer, [MilliHertzBase.jl](https://github.com/PaulGoG/MilliHertzBase.jl)
([manual](https://PaulGoG.github.io/MilliHertzBase.jl/dev/)), holds the
noise model, waveforms, detector response, LDC products, the generation,
pre-processing, labelling and payload-export stages, and the coupling to
the DeepSpaceTelemetry producer. `MilliHertzQML` re-exports the public
names of both, so `using MilliHertzQML` gives the whole interface; the two
modules are reachable as `MilliHertzQML.StreamingInference` and
`MilliHertzQML.MilliHertzBase`.

## Classifier (`MilliHertzQML`)

The variational quantum classifier, its training and persistence, the
feature scaler, the classifier as a window scorer, the training and
inference stages, and the figures of its studies.

```@autodocs
Modules = [MilliHertzQML]
```
