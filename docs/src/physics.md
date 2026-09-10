# Physics & Data Simulation

## Observational Context

LISA observes the milliHertz band (approximately ``10^{-4}`` to ``10^{-1}`` Hz). The pipeline operates at the cadence of the LISA Data Challenge level-1 products, 0.2 Hz (``\Delta t = 5`` s), on signals lasting hours to weeks. Telemetry arrives as time-delay interferometry (TDI) channels ``X, Y, Z``; the pre-processor forms the orthogonal channel ``A = (Z - X)/\sqrt{2}`` and operates on it exclusively.

## Noise Model (`src/simulation.jl`)

The strain-noise sensitivity is the sky- and polarization-averaged model of Robson, Cornish & Liu, *Class. Quantum Grav.* **36** 105011 (2019):

```math
S_n(f) = \frac{10}{3 L^2} \left[ P_\mathrm{OMS}(f) + 2 \left(1 + \cos^2 \frac{f}{f_*}\right) \frac{P_\mathrm{acc}(f)}{(2\pi f)^4} \right] \left[ 1 + \frac{6}{10} \left(\frac{f}{f_*}\right)^2 \right] + S_c(f),
```

with ``L = 2.5 \times 10^9`` m, ``f_* = c/(2\pi L)``, the optical-metrology term ``P_\mathrm{OMS} = (1.5 \times 10^{-11}\,\mathrm{m})^2 [1 + (2\,\mathrm{mHz}/f)^4]`` Hz⁻¹, the acceleration term ``P_\mathrm{acc} = (3 \times 10^{-15}\,\mathrm{m\,s^{-2}})^2 [1 + (0.4\,\mathrm{mHz}/f)^2][1 + (f/8\,\mathrm{mHz})^4]`` Hz⁻¹, and the galactic confusion foreground

```math
S_c(f) = A f^{-7/3} e^{-f^\alpha + \beta f \sin(\kappa f)} \left[ 1 + \tanh\left(\gamma (f_k - f)\right) \right], \qquad A = 9 \times 10^{-45}\,\mathrm{Hz^{-1}},
```

whose parameters depend on the observation time over which resolvable binaries are assumed subtracted (Table 1 of the reference; 0.5, 1, 2, or 4 years, selected by `observation_years`). `lisa_noise_psd` returns ``S_n(f)`` in Hz⁻¹; `instrument_psd` and `confusion_psd` expose the two parts. The same function serves the simulator, the matched filter, and the whitening of features, so the three are consistent by construction.

## Simulated Telemetry (`scripts/generate_data.jl`)

The simulator produces a continuous strain record at physical amplitude:

- **Gaussian noise** synthesized in the frequency domain against ``S_n(f)`` (`synthesize_noise`): the spectral coefficients are drawn as complex normals scaled to ``E|X_k|^2 = S_n(f_k) f_s n / 2`` in the unnormalized FFT convention, so the inverse transform has the correct one-sided PSD and absolute amplitude. The confusion foreground is therefore part of the Gaussian noise. Bins below `noise_f_min_hz` (default ``10^{-5}`` Hz, the low end of the band over which the sensitivity model is defined) are not synthesized: the model's extrapolation towards zero frequency would put a drift many orders of magnitude above the in-band level into the record.
- **Resolvable galactic binaries**: monochromatic sinusoids with random frequencies in 0.1–10 mHz; **EMRIs**: three-harmonic chirps with a constant frequency drift. Each is scaled to a matched-filter signal-to-noise ratio drawn from the configured range (`gb_snr_min/max`, `emri_snr_min/max`), evaluated over the simulated record.
- **MBHB injections**: phenomenological inspiral–merger–ringdown waveforms (Newtonian inspiral scalings ``\Phi \propto \tau^{5/8}``, ``A \propto \tau^{-1/4}`` with ``\tau = t_c - t``, followed by an exponentially damped sinusoid). Each waveform is anchored so that its coalescence sample lands on the drawn merger sample ``k_c`` (`place_signal!`), truncated to the record, and then scaled to a matched-filter SNR ``\rho`` drawn from `[snr_min, snr_max]` (default ``[8, 50]``):

```math
\rho^2 = 4 \int_0^\infty \frac{|\tilde h(f)|^2}{S_n(f)} \, \mathrm{d}f \approx 4 \Delta f \sum_{k \ge 1} \frac{|\tilde h(f_k)|^2}{S_n(f_k)}.
```

Point-wise labels mark the configurable span ``[t_c - \texttt{label\_before\_sec},\, t_c + \texttt{label\_after\_sec}]``; the per-sample `SNR` column carries the event's ``\rho``. Output is an HDF5 file (`obs/tdi/{t, X, Z}`, with ``X \equiv 0`` and ``Z = \sqrt{2}\,A`` so the pre-processor round-trips), a point-wise label CSV, an event catalog CSV (merger sample and time, ``\rho``, waveform parameters, covered and labeled sample ranges), and a generation snapshot with the seed and the hardware fingerprint.

## Feature Extraction (`src/data.jl`)

The record is first high-passed in the frequency domain with a zero-phase Butterworth magnitude response (`highpass_record`; default cutoff 0.5 mHz, order 8), because the acceleration noise rises as ``f^{-6}`` below 0.4 mHz and the resulting sub-window drift would leak into every analysis band. The record is then whitened in the frequency domain by the model PSD (`whiten_record`): ``X_k \to X_k \sqrt{2 / (f_s S_n(f_k))}``, so that noise following ``S_n`` becomes white with unit variance. A sliding window (default 1000 samples, step 100) marches over the whitened ``A`` channel; each window is Hann-tapered and its periodogram normalized by the taper's mean square (`tapered_periodogram`), ``P_k = |\mathrm{FFT}(x v)_k|^2 / (n \overline{v^2})``, which has unit mean for noise. Whitening the record rather than each window keeps the tapered periodogram unbiased irrespective of the PSD slope: the periodogram of a tapered window is the PSD smoothed by the taper's main lobe, which is exact only on a flat spectrum. Single-window whitening remains possible for streaming use, where no record context exists. Four features follow from ``P_k``:

1. mean whitened power in the low band (default 1–5 mHz);
2. mean whitened power in the high band (default 5–100 mHz);
3. spectral entropy of the normalized whitened power, divided by ``\ln N_\mathrm{bins}`` (in ``[0, 1]``);
4. ``\log_{10}`` of the standard deviation of the whitened power.

The band edges and the confusion fit used for whitening are configuration keys of `[preprocessing]`. Because the features are PSD-normalized, they are independent of the window length and of the absolute strain amplitude; the same code path serves the simulator and physical-strain LDC data, provided the whitening PSD describes the channel's noise.

Features are mapped onto the phase-encoding interval ``[0, 2\pi]`` by a `FeatureScaler` whose bounds are per-feature quantiles of the **training partition** (`fit_scaler`, `[training] scaler_quantiles`). The scaler is persisted inside the model artifact and applied unchanged at inference (`encode_features`), so no statistic of the evaluated data enters the encoding.

## Known Physical Deficiencies

Retained so that the documentation reflects the code as it stands; remediation is planned.

1. **Waveform model.** The MBHB approximant is phenomenological: the inspiral instantaneous frequency crosses the Nyquist frequency (0.1 Hz) in the final tens of seconds before merger, aliasing the loudest samples, and the ringdown frequency and damping time are drawn independently of the chirp scaling with no mass consistency. A closed-form inspiral–merger–ringdown model with a Nyquist taper replaces it in the next physics change set.
2. **Label span.** The default label span (12 h before to 1 h after coalescence) is a heuristic; the waveform-based default (the interval over which the injected signal is detectable) lands with the waveform model.
3. **Whitening PSD for LDC data.** The whitening uses the analytic sensitivity model; for LDC TDI channels the channel noise PSD must be estimated from the data or taken from the LDC noise model. This is part of the Sangria benchmark stage.
