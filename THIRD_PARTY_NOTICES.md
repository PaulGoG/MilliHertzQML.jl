# Third-party notices

## LISA Data Challenge software (`ldc` package)

`src/ldc.jl` contains a Julia port of the equal-arm analytic TDI noise model
of the LISA Data Challenge toolbox, `lisa-data-challenge` 1.2.0
(<https://lisa.pages.in2p3.fr/LDC>): the single-link noise levels of
`ldc.lisa.noise.AnalyticNoise` (`LDC_NOISE_LEVELS`), its equal-arm
first-generation TDI power spectral densities with the second-generation
factor (`ldc_tdi_psd`), and the Galactic-confusion fit of `GalNoise` and
`Noise.wd_confusion` (`LDC_CONFUSION_FIT`, `ldc_confusion_psd`). The
toolbox is distributed under the following licence.

```
MIT License

Copyright (c) 2019 LISA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
