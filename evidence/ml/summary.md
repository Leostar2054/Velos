# Baseline comparison — test split, n = 30
Sparse scan: 64 frames · dense scan: 121 frames (47 % fewer frames).

| Method | NRMSE (complex) ↓ | PSNR amp (dB) ↑ | SSIM amp ↑ | PSNR phase (dB) ↑ | SSIM phase ↑ | MAE phase (rad) ↓ |
|---|---|---|---|---|---|---|
| Physics (sparse scan) | 0.120 ± 0.072 | 8.2 ± 4.9 | 0.140 ± 0.072 | 32.7 ± 7.0 | 0.765 ± 0.162 | 0.060 ± 0.043 |
| Physics (sparse, more iters) | 0.090 ± 0.062 | 10.5 ± 5.9 | 0.213 ± 0.110 | 36.9 ± 8.7 | 0.885 ± 0.113 | 0.036 ± 0.029 |
| Physics (sparse) + ML | 0.095 ± 0.055 | 14.6 ± 5.2 | 0.504 ± 0.113 | 33.9 ± 6.7 | 0.784 ± 0.123 | 0.050 ± 0.029 |
| Physics (sparse, more iters) + ML | 0.083 ± 0.047 | 15.7 ± 5.2 | 0.560 ± 0.120 | 35.1 ± 6.6 | 0.796 ± 0.123 | 0.045 ± 0.025 |
| Physics (dense scan) | 0.014 ± 0.007 | 26.7 ± 5.2 | 0.781 ± 0.068 | 51.2 ± 5.5 | 0.976 ± 0.026 | 0.008 ± 0.004 |

Metrics are computed inside the evaluation mask (illuminated by both scans) after removing the global complex scalar and linear phase ramp, which are unobservable in ptychography.
