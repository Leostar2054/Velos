# M3 evaluation — summary tables

## Synthetic test set (n = 30, ground truth; registered, ramp and scalar removed)

| Method | NRMSE ↓ | Phase PSNR (dB) ↑ | Phase SSIM ↑ | Amp PSNR (dB) ↑ | Amp SSIM ↑ | Data err. used ↓ | Prediction err. (unmeasured frames) ↓ |
|---|---|---|---|---|---|---|---|
| Physics, reduced scan (ePIE 300 it) | 0.128 ± 0.099 | 34.8 ± 10.3 | 0.747 | 10.6 | 0.182 | 0.0002 | 0.0135 |
| Physics + 300 further it | 0.116 ± 0.091 | 36.0 ± 10.5 | 0.772 | 11.3 | 0.197 | 0.0005 | 0.0130 |
| M2 model | 0.107 ± 0.060 | 32.6 ± 5.6 | 0.740 | 15.3 | 0.317 | 0.0041 | 0.0128 |
| v2, synthetic only (stage 1) | 0.068 ± 0.052 | 38.6 ± 8.7 | 0.887 | 18.5 | 0.446 | 0.0032 | 0.0115 |
| **v2 model (final)** | 0.068 ± 0.050 | 38.5 ± 8.6 | 0.887 | 17.8 | 0.428 | 0.0030 | 0.0115 |
| Hybrid v2 + 10 it | 0.061 ± 0.052 | 40.5 ± 8.9 | 0.901 | 17.7 | 0.322 | 0.0016 | 0.0105 |
| **Hybrid v2 + 30 it** | 0.059 ± 0.054 | 41.3 ± 9.2 | 0.907 | 18.0 | 0.321 | 0.0016 | 0.0103 |
| Hybrid v2 + 100 it | 0.058 ± 0.055 | 42.1 ± 9.4 | 0.918 | 18.1 | 0.328 | 0.0017 | 0.0101 |
| Dense scan, 121 frames (reference) | 0.009 ± 0.003 | 55.4 ± 4.9 | 0.996 | 29.4 | 0.583 | – | – |

### By number of frames (phase PSNR dB / NRMSE)

| Method | 36 frames (n=8) | 49 frames (n=10) | 64 frames (n=12) |
|---|---|---|---|
| Physics, reduced scan (ePIE 300 it) | 25.6 / 0.246 | 34.2 / 0.107 | 41.3 / 0.066 |
| Physics + 300 further it | 26.5 / 0.227 | 35.2 / 0.098 | 43.0 / 0.057 |
| M2 model | 28.3 / 0.174 | 34.1 / 0.083 | 34.2 / 0.084 |
| v2, synthetic only (stage 1) | 31.7 / 0.119 | 38.2 / 0.056 | 43.4 / 0.044 |
| **v2 model (final)** | 31.8 / 0.118 | 38.1 / 0.057 | 43.3 / 0.045 |
| Hybrid v2 + 10 it | 33.3 / 0.108 | 41.1 / 0.044 | 44.9 / 0.043 |
| **Hybrid v2 + 30 it** | 33.6 / 0.108 | 42.0 / 0.042 | 45.9 / 0.041 |
| Hybrid v2 + 100 it | 34.0 / 0.107 | 43.0 / 0.039 | 46.6 / 0.040 |
| Dense scan, 121 frames (reference) | 56.3 / 0.009 | 55.6 / 0.008 | 54.5 / 0.010 |

## Experimental data (reference = reconstruction from all frames; registered)

| Case | Method | NRMSE ↓ | Phase PSNR ↑ | Phase SSIM ↑ | Data err. used ↓ | Prediction err. (frames not used) ↓ |
|---|---|---|---|---|---|---|
| real_clam/sub_00 | Physics, reduced scan (ePIE 300 it) | 0.428 | 24.6 | 0.376 | 0.0292 | 0.1129 |
| real_clam/sub_00 | M2 model | 0.475 | 23.4 | 0.238 | 0.0684 | 0.1143 |
| real_clam/sub_00 | v2, synthetic only (stage 1) | 0.425 | 24.7 | 0.309 | 0.0591 | 0.0980 |
| real_clam/sub_00 | **v2 model (final)** | 0.344 | 26.5 | 0.445 | 0.0426 | 0.0843 |
| real_clam/sub_00 | Hybrid v2 + 10 it | 0.369 | 26.0 | 0.433 | 0.0318 | 0.0915 |
| real_clam/sub_00 | **Hybrid v2 + 30 it** | 0.389 | 25.5 | 0.421 | 0.0307 | 0.0981 |
| real_clam/sub_00 | Hybrid v2 + 100 it | 0.408 | 25.0 | 0.412 | 0.0339 | 0.1064 |
| real_clam/sub_00 | All frames, 2nd run (repeatability) | 0.052 | 43.3 | 0.992 | 0.0227 | 0.0220 |
| real_clam/sub_01 | Physics, reduced scan (ePIE 300 it) | 0.304 | 27.1 | 0.602 | 0.0252 | 0.0634 |
| real_clam/sub_01 | M2 model | 0.484 | 22.4 | 0.240 | 0.0782 | 0.0939 |
| real_clam/sub_01 | v2, synthetic only (stage 1) | 0.476 | 21.9 | 0.223 | 0.0770 | 0.0867 |
| real_clam/sub_01 | **v2 model (final)** | 0.257 | 28.4 | 0.626 | 0.0379 | 0.0527 |
| real_clam/sub_01 | Hybrid v2 + 10 it | 0.236 | 29.4 | 0.724 | 0.0262 | 0.0530 |
| real_clam/sub_01 | **Hybrid v2 + 30 it** | 0.213 | 30.4 | 0.804 | 0.0316 | 0.0535 |
| real_clam/sub_01 | Hybrid v2 + 100 it | 0.202 | 30.4 | 0.829 | 0.0434 | 0.0586 |
| real_clam/sub_01 | All frames, 2nd run (repeatability) | 0.055 | 42.7 | 0.991 | 0.0212 | 0.0232 |
| real_plant_cal/sub_00 | Physics, reduced scan (ePIE 300 it) | 0.453 | 25.0 | 0.587 | 0.0215 | 0.0841 |
| real_plant_cal/sub_00 | M2 model | 0.510 | 24.6 | 0.430 | 0.0447 | 0.0996 |
| real_plant_cal/sub_00 | v2, synthetic only (stage 1) | 0.510 | 23.3 | 0.409 | 0.0556 | 0.0981 |
| real_plant_cal/sub_00 | **v2 model (final)** | 0.421 | 26.5 | 0.623 | 0.0319 | 0.0643 |
| real_plant_cal/sub_00 | Hybrid v2 + 10 it | 0.421 | 26.3 | 0.626 | 0.0260 | 0.0679 |
| real_plant_cal/sub_00 | **Hybrid v2 + 30 it** | 0.423 | 26.0 | 0.621 | 0.0262 | 0.0729 |
| real_plant_cal/sub_00 | Hybrid v2 + 100 it | 0.424 | 25.4 | 0.608 | 0.0310 | 0.0813 |
| real_plant_cal/sub_00 | All frames, 2nd run (repeatability) | 0.188 | 31.6 | 0.848 | 0.0401 | 0.0421 |
| real_plant_cal/sub_01 | Physics, reduced scan (ePIE 300 it) | 0.407 | 27.3 | 0.666 | 0.0336 | 0.0670 |
| real_plant_cal/sub_01 | M2 model | 0.512 | 25.3 | 0.440 | 0.0583 | 0.0834 |
| real_plant_cal/sub_01 | v2, synthetic only (stage 1) | 0.484 | 24.0 | 0.449 | 0.0684 | 0.0847 |
| real_plant_cal/sub_01 | **v2 model (final)** | 0.388 | 28.1 | 0.681 | 0.0432 | 0.0583 |
| real_plant_cal/sub_01 | Hybrid v2 + 10 it | 0.383 | 28.6 | 0.718 | 0.0368 | 0.0604 |
| real_plant_cal/sub_01 | **Hybrid v2 + 30 it** | 0.380 | 28.3 | 0.688 | 0.0392 | 0.0646 |
| real_plant_cal/sub_01 | Hybrid v2 + 100 it | 0.366 | 28.1 | 0.722 | 0.0522 | 0.0762 |
| real_plant_cal/sub_01 | All frames, 2nd run (repeatability) | 0.191 | 31.8 | 0.847 | 0.0442 | 0.0404 |
| real_plant_cal/sub_02 | Physics, reduced scan (ePIE 300 it) | 0.360 | 27.5 | 0.692 | 0.0370 | 0.0743 |
| real_plant_cal/sub_02 | M2 model | 0.473 | 25.8 | 0.519 | 0.0565 | 0.0906 |
| real_plant_cal/sub_02 | v2, synthetic only (stage 1) | 0.444 | 24.0 | 0.482 | 0.0661 | 0.0901 |
| real_plant_cal/sub_02 | **v2 model (final)** | 0.364 | 27.8 | 0.672 | 0.0427 | 0.0644 |
| real_plant_cal/sub_02 | Hybrid v2 + 10 it | 0.355 | 28.4 | 0.714 | 0.0388 | 0.0660 |
| real_plant_cal/sub_02 | **Hybrid v2 + 30 it** | 0.348 | 28.3 | 0.714 | 0.0433 | 0.0700 |
| real_plant_cal/sub_02 | Hybrid v2 + 100 it | 0.335 | 27.9 | 0.665 | 0.0670 | 0.0886 |
| real_plant_cal/sub_02 | All frames, 2nd run (repeatability) | 0.174 | 32.2 | 0.871 | 0.0415 | 0.0419 |
| real_plant/sub_01 | Physics, reduced scan (ePIE 300 it) | 0.543 | 22.1 | 0.194 | 0.0633 | 0.1535 |
| real_plant/sub_01 | M2 model | 0.585 | 22.3 | 0.159 | 0.0767 | 0.1531 |
| real_plant/sub_01 | v2, synthetic only (stage 1) | 0.582 | 21.9 | 0.204 | 0.0800 | 0.1381 |
| real_plant/sub_01 | **v2 model (final)** | 0.492 | 23.4 | 0.262 | 0.0874 | 0.1110 |
| real_plant/sub_01 | Hybrid v2 + 10 it | 0.500 | 23.1 | 0.254 | 0.0723 | 0.1213 |
| real_plant/sub_01 | **Hybrid v2 + 30 it** | 0.508 | 22.7 | 0.250 | 0.0749 | 0.1330 |
| real_plant/sub_01 | Hybrid v2 + 100 it | 0.513 | 22.4 | 0.231 | 0.0916 | 0.1513 |
| real_plant/sub_02 | Physics, reduced scan (ePIE 300 it) | 0.479 | 23.0 | 0.274 | 0.0669 | 0.1481 |
| real_plant/sub_02 | M2 model | 0.502 | 23.1 | 0.248 | 0.0845 | 0.1592 |
| real_plant/sub_02 | v2, synthetic only (stage 1) | 0.487 | 23.2 | 0.298 | 0.0779 | 0.1425 |
| real_plant/sub_02 | **v2 model (final)** | 0.459 | 23.6 | 0.284 | 0.0838 | 0.1240 |
| real_plant/sub_02 | Hybrid v2 + 10 it | 0.453 | 23.5 | 0.319 | 0.0728 | 0.1292 |
| real_plant/sub_02 | **Hybrid v2 + 30 it** | 0.455 | 23.3 | 0.312 | 0.0784 | 0.1364 |
| real_plant/sub_02 | Hybrid v2 + 100 it | 0.453 | 23.2 | 0.313 | 0.1015 | 0.1516 |

