# Physics & Data Simulation

## Observational Context

LISA observes the milliHertz band (approximately ``10^{-4}`` to ``10^{-1}`` Hz) with slow sampling (0.2 Hz in this pipeline) and signal durations of hours to weeks. Telemetry arrives as time-delay interferometry (TDI) channels ``X, Y, Z``. The pre-processor forms a single strain-like channel ``A = (Z - X)/\sqrt{2}`` and operates on it exclusively.

## Simulated Telemetry (`scripts/generate_data.jl`)

The simulator produces a continuous strain array containing:

- **Instrument noise**, colored in the frequency domain by a simplified analytic PSD following the structure of Robson, Cornish & Liu (2019): an optical-metrology term, an acceleration-noise term, and a galactic-confusion term. The realization is normalized to unit variance, so only the spectral shape enters; absolute strain calibration is not represented.
- **Galactic binaries**: monochromatic sinusoids with random frequencies in 0.1–10 mHz and random amplitudes.
- **EMRIs**: three-harmonic chirps with a constant frequency-derivative drift.
- **MBHB injections**: phenomenological inspiral–merger–ringdown waveforms. The inspiral uses the Newtonian scalings ``\Phi(\tau) \propto \tau^{5/8}`` and ``A(\tau) \propto \tau^{-1/4}`` with ``\tau = t_c - t``; the ringdown is an exponentially damped sinusoid matched in amplitude and phase at merger.

Each injected event carries an `SNR` value used to scale the waveform amplitude, and point-wise labels mark a window around the merger time ``t_c`` (currently ``t_c - 12\,\mathrm{h}`` to ``t_c + 1\,\mathrm{h}``). Output is an HDF5 file (`obs/tdi/{t, X, Z}`, with ``X \equiv 0`` and ``Z = \sqrt{2}\,A`` so the pre-processor round-trips) plus a label CSV.

## Feature Extraction (`src/data.jl`)

A sliding window (default 1000 samples, step 100) marches over the ``A`` channel. Per window, four features are computed from the magnitude of the real FFT:

1. mean spectral magnitude in the low band (1–5 mHz);
2. mean spectral magnitude in the high band (5–100 mHz);
3. spectral entropy of the normalized squared spectrum (natural logarithm, unnormalized by bin count);
4. ``\log_{10}`` of the standard deviation of the squared spectrum.

`load_data` clamps each feature to fixed scales and maps it linearly to ``[0, 2\pi]`` for phase encoding.

## Known Physical Deficiencies

These are properties of the current implementation, retained here so that the documentation reflects the code as it stands. Remediation is planned.

1. **Galactic confusion spectrum.** The confusion term uses `exp(-(f/f_k)^B)` with ``f_k = 0.1`` mHz and ``B = 292``, which vanishes for ``f \gtrsim 0.2`` mHz — removing the confusion foreground precisely in the 0.5–3 mHz band where it should dominate. The instrument terms also omit the ``10/3`` prefactor and the ``1 + \tfrac{6}{10}(f/f_*)^2`` factor of the reference PSD.
2. **Injection amplitude scaling.** The recorded `SNR` is a time-domain amplitude ratio against the unit-variance instrument noise before the background forest is added. It is not the matched-filter signal-to-noise ratio ``\rho``, and detection-efficiency curves parameterized by it are not comparable to the gravitational-wave literature.
3. **Chirp aliasing.** The inspiral instantaneous frequency crosses the Nyquist frequency (0.1 Hz) in the final tens of seconds before merger; the loudest waveform samples are aliased. The ringdown frequency is drawn independently of the chirp scaling, with no mass consistency between inspiral and ringdown.
4. **Injection/label alignment.** The waveform duration is fixed at two days. If ``t_c`` lies less than two days after the start of the record, the injection is shifted to the array start while labels remain centred on ``t_c``, misaligning signal and labels; short simulations may receive no injections at all.
5. **Label coverage.** Labels span ``t_c - 12`` h to ``t_c + 1`` h, but the injected waveform spans two days plus ringdown; signal-bearing windows outside the label span are labeled as noise.
6. **Feature calibration.** Band features are mean FFT magnitudes without PSD normalization (window-length dependent); the entropy is not normalized by ``\ln N``; the fixed clamping scales are tuned to the unit-variance simulator and collapse to constants on physical-amplitude LDC strain data.
