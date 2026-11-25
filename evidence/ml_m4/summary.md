#### hold-out region (right 35 % of the trained field, unseen LED draws) — dark-field fraction 30% (n = 6 cases)

| Method | Phase PSNR (dB) ↑ | NRMSE ↓ | Phase SSIM ↑ | Amp PSNR (dB) ↑ | Data err. used ↓ | Prediction err. (LEDs not used) ↓ |
|---|---|---|---|---|---|---|
| Physics (EPRY, reduced LED set) | 25.17 | 0.389 | 0.600 | 18.74 | 0.1785 | 0.3680 |
| Physics + 5 passes | 25.30 (+0.13) | 0.390 | 0.600 | 18.74 | 0.1760 | 0.3674 (-0 %) |
| Physics + 10 passes | 25.37 (+0.20) | 0.392 | 0.599 | 18.72 | 0.1742 | 0.3674 (-0 %) |
| M3 model v2 (no FPM training) | 23.56 (-1.61) | 0.413 | 0.582 | 18.94 | 0.1871 | 0.3812 (+4 %) |
| **FPM model v1** (fine-tuned) | 29.32 (+4.15) | 0.212 | 0.699 | 23.22 | 0.2202 | 0.4305 (+17 %) |
| Hybrid: FPM model + 5 passes | 26.26 (+1.09) | 0.337 | 0.624 | 19.93 | 0.1770 | 0.3680 (+0 %) |
| Hybrid: FPM model + 10 passes | 26.16 (+0.99) | 0.345 | 0.618 | 19.71 | 0.1737 | 0.3672 (-0 %) |

#### hold-out region (right 35 % of the trained field, unseen LED draws) — dark-field fraction 50% (n = 6 cases)

| Method | Phase PSNR (dB) ↑ | NRMSE ↓ | Phase SSIM ↑ | Amp PSNR (dB) ↑ | Data err. used ↓ | Prediction err. (LEDs not used) ↓ |
|---|---|---|---|---|---|---|
| Physics (EPRY, reduced LED set) | 28.14 | 0.276 | 0.668 | 22.09 | 0.1766 | 0.3437 |
| Physics + 5 passes | 28.41 (+0.26) | 0.270 | 0.671 | 22.27 | 0.1741 | 0.3430 (-0 %) |
| Physics + 10 passes | 28.56 (+0.42) | 0.267 | 0.672 | 22.36 | 0.1722 | 0.3425 (-0 %) |
| M3 model v2 (no FPM training) | 24.57 (-3.57) | 0.331 | 0.617 | 20.78 | 0.1854 | 0.3608 (+5 %) |
| **FPM model v1** (fine-tuned) | 30.37 (+2.23) | 0.196 | 0.717 | 24.10 | 0.1960 | 0.3903 (+14 %) |
| Hybrid: FPM model + 5 passes | 29.18 (+1.04) | 0.235 | 0.690 | 22.95 | 0.1750 | 0.3455 (+1 %) |
| Hybrid: FPM model + 10 passes | 29.24 (+1.10) | 0.234 | 0.689 | 22.98 | 0.1724 | 0.3449 (+0 %) |

#### side regions of the sensor (never used for training) — dark-field fraction 30% (n = 4 cases)

| Method | Phase PSNR (dB) ↑ | NRMSE ↓ | Phase SSIM ↑ | Amp PSNR (dB) ↑ | Data err. used ↓ | Prediction err. (LEDs not used) ↓ |
|---|---|---|---|---|---|---|
| Physics (EPRY, reduced LED set) | 17.47 | 0.673 | 0.288 | 14.39 | 0.2414 | 0.4619 |
| Physics + 5 passes | 17.22 (-0.25) | 0.686 | 0.273 | 14.07 | 0.2397 | 0.4636 (+0 %) |
| Physics + 10 passes | 16.94 (-0.53) | 0.700 | 0.255 | 13.78 | 0.2388 | 0.4653 (+1 %) |
| M3 model v2 (no FPM training) | 17.88 (+0.41) | 0.651 | 0.313 | 15.94 | 0.2522 | 0.4772 (+3 %) |
| **FPM model v1** (fine-tuned) | 21.52 (+4.05) | 0.440 | 0.340 | 20.24 | 0.3965 | 0.7864 (+70 %) |
| Hybrid: FPM model + 5 passes | 20.30 (+2.83) | 0.510 | 0.371 | 18.13 | 0.2488 | 0.4872 (+5 %) |
| Hybrid: FPM model + 10 passes | 19.87 (+2.40) | 0.533 | 0.348 | 17.47 | 0.2419 | 0.4824 (+4 %) |

#### side regions of the sensor (never used for training) — dark-field fraction 50% (n = 4 cases)

| Method | Phase PSNR (dB) ↑ | NRMSE ↓ | Phase SSIM ↑ | Amp PSNR (dB) ↑ | Data err. used ↓ | Prediction err. (LEDs not used) ↓ |
|---|---|---|---|---|---|---|
| Physics (EPRY, reduced LED set) | 20.65 | 0.522 | 0.410 | 18.43 | 0.2353 | 0.4459 |
| Physics + 5 passes | 20.72 (+0.07) | 0.523 | 0.410 | 18.42 | 0.2321 | 0.4461 (+0 %) |
| Physics + 10 passes | 20.65 (-0.01) | 0.529 | 0.401 | 18.27 | 0.2296 | 0.4466 (+0 %) |
| M3 model v2 (no FPM training) | 19.66 (-1.00) | 0.540 | 0.371 | 18.73 | 0.2468 | 0.4755 (+7 %) |
| **FPM model v1** (fine-tuned) | 22.15 (+1.49) | 0.406 | 0.379 | 21.49 | 0.3080 | 0.7029 (+58 %) |
| Hybrid: FPM model + 5 passes | 22.12 (+1.47) | 0.418 | 0.428 | 20.81 | 0.2397 | 0.4648 (+4 %) |
| Hybrid: FPM model + 10 passes | 21.98 (+1.33) | 0.426 | 0.418 | 20.52 | 0.2340 | 0.4649 (+4 %) |

#### Per scan (mean of the test LED draws)

| Scan | Physics (EPRY, reduced LED set) | Physics + 5 passes | Physics + 10 passes | M3 model v2 (no FPM training) | FPM model v1 (fine-tuned) | Hybrid: FPM model + 5 passes | Hybrid: FPM model + 10 passes |
|---|---|---|---|---|---|---|---|
| blood_222619 | 32.09 / 0.311 | 32.38 / 0.310 | 32.56 / 0.308 | 27.86 / 0.327 | 33.65 / 0.318 | 32.65 / 0.307 | 32.67 / 0.306 |
| leaf_214040 | 15.50 / 0.441 | 15.55 / 0.441 | 15.58 / 0.443 | 15.90 / 0.452 | 22.42 / 0.597 | 17.52 / 0.454 | 17.51 / 0.452 |
| blood_223858 | 32.02 / 0.312 | 32.30 / 0.311 | 32.48 / 0.309 | 27.89 / 0.328 | 33.61 / 0.319 | 32.60 / 0.308 | 32.62 / 0.307 |
| leaf_215808 | 16.43 / 0.437 | 16.53 / 0.438 | 16.59 / 0.440 | 16.89 / 0.449 | 22.24 / 0.581 | 18.35 / 0.447 | 18.31 / 0.446 |
| blood_225108 | 31.96 / 0.316 | 32.19 / 0.315 | 32.30 / 0.313 | 28.05 / 0.332 | 33.53 / 0.323 | 32.58 / 0.312 | 32.52 / 0.311 |
| blood_230302_2s | 31.93 / 0.318 | 32.17 / 0.317 | 32.30 / 0.315 | 27.81 / 0.337 | 33.62 / 0.325 | 32.62 / 0.313 | 32.56 / 0.313 |
| blood_222619_L512 | 26.85 / 0.439 | 26.69 / 0.439 | 26.31 / 0.440 | 25.38 / 0.472 | 28.37 / 0.574 | 29.40 / 0.428 | 29.11 / 0.431 |
| blood_222619_R512 | 22.74 / 0.494 | 22.62 / 0.493 | 22.38 / 0.492 | 22.69 / 0.533 | 25.59 / 0.593 | 24.65 / 0.494 | 24.10 / 0.493 |
| leaf_214040_L512 | 13.12 / 0.441 | 13.09 / 0.443 | 13.06 / 0.446 | 13.28 / 0.449 | 16.60 / 0.927 | 15.37 / 0.495 | 15.25 / 0.490 |
| leaf_214040_R512 | 13.54 / 0.442 | 13.49 / 0.444 | 13.42 / 0.446 | 13.73 / 0.451 | 16.78 / 0.885 | 15.42 / 0.486 | 15.25 / 0.481 |

(phase PSNR dB / prediction error on the LEDs not used)

#### Repeatability of the reference (full-LED reconstructions of repeated scans, hold-out region)

| Pair | Phase PSNR (dB) | NRMSE | Phase SSIM |
|---|---|---|---|
| blood_222619 vs blood_223858 | 41.19 | 0.051 | 0.979 |
| blood_222619 vs blood_225108 | 33.84 | 0.126 | 0.922 |
| leaf_214040 vs leaf_215808 | 20.13 | 0.425 | 0.204 |

