## 1. Health and installed model packages
```
{
    "status": "ok",
    "version": "0.4.0",
    "queue": {
        "queued": 0,
        "running": 0
    },
    "models": {
        "installed": [
            "velos-refine-fpm-v1",
            "velos-refine-v2"
        ],
        "default": "velos-refine-v2",
        "default_by_modality": {
            "fpm": "velos-refine-fpm-v1",
            "ptychography": "velos-refine-v2"
        }
    }
}
{"id": "velos-refine-fpm-v1", "name": "velos-refine-fpm", "version": "1.0.0", "modality": "fpm", "default": true, "normalisation": "median_ramp", "tile": 256}
{"id": "velos-refine-v2", "name": "velos-refine", "version": "2.0.0", "modality": "ptychography", "default": true, "normalisation": "median_ramp", "tile": 256}
```

## 2. Upload the FPM dataset
```
{
 "id": "056ff76ebec94e41",
 "name": "leaf_214040_holdout_c160.npz",
 "size_bytes": 8070901,
 "summary": {
  "modality": "fpm",
  "n_patterns": 225,
  "frame_shape": [
   1080,
   1920
  ],
  "stored_region": [
   460,
   1056,
   160,
   160
  ],
  "led_color": "GREEN",
  "camera": "Thorlabs CS2100M-USB",
  "exposure_us": [
   1000000.0
  ]
 }
}
```

## 3. Reference: all LEDs, physics only (EPRY, 24 passes)
```
POST /api/v1/jobs {"dataset_id":"056ff76ebec94e41","engine":"fpm","iters":24}
job 348a4d4abca745d6: queued
t=   1s  status=queued  queued
t=  11s  status=running  EPRY pass 24/26 · residual 0.2174
t=  17s  status=succeeded  completed
{
 "leds_used": 222,
 "grid_px": 480,
 "pixel_um": 0.42,
 "synthetic_NA": 0.5770741116658369,
 "resolution_um": {
  "raw": 6.33,
  "synthetic": 1.0969128352902926
 },
 "final_residual": [
  0.21677231553190532
 ],
 "data_error_normalised": 0.20498268967654454,
 "elapsed_s": 18.19
}
```

## 4. Reduced LED set (bright field + 0.3 of the dark-field LEDs) → FPM model → hybrid (10 passes)
```
POST /api/v1/jobs {"dataset_id":"056ff76ebec94e41","engine":"fpm","iters":24,"sparse_fraction":0.3,"refine_ml":true,"hybrid_iters":10}
{
    "id": "9e3bd431b1ea4243",
    "dataset_id": "056ff76ebec94e41",
    "owner": "demo",
    "status": "queued",
    "params": {
        "dataset_id": "056ff76ebec94e41",
        "engine": "fpm",
        "iters": 24,
        "seed": 0,
        "crop": null,
        "sparse_every": 1,
        "engine_params": {},
        "sparse_fraction": 0.3,
        "refine_ml": true,
        "model": null,
        "hybrid_iters": 10,
        "fpm": {},
        "probe_from_job": null
    },
    "progress": 0.0,
    "iteration": 0,
    "message": "queued",
    "error": "",
    "created_at": "2026-11-25T09:14:43.570618Z",
    "started_at": null,
    "finished_at": null,
    "links": {
        "self": "/api/v1/jobs/9e3bd431b1ea4243"
    }
}
t=   1s  status=queued  queued
t=  11s  status=running  EPRY pass 26/26 · residual 0.2373
t=  20s  status=succeeded  completed
```

## 5. Result — physics, AI model and hybrid, each scored against the measured frames
```
{
 "dataset": {
  "name": "leaf_214040",
  "n_patterns": null,
  "stored_region": [
   460,
   1056,
   160,
   160
  ]
 },
 "reduced_leds": {
  "fraction_darkfield_kept": 0.3,
  "leds_used": 74,
  "leds_available": 222
 },
 "data_error_used_leds": {
  "physics": 0.22324794820379382,
  "ml": 0.23109141807076863,
  "hybrid": 0.20181500526045948
 },
 "prediction_error_unused_leds": {
  "physics": 0.5125494811256928,
  "ml": 0.5648030895904439,
  "hybrid": 0.50390305178884
 },
 "model": {
  "model": "velos-refine-fpm-v1",
  "model_version": "1.0.0",
  "runtime": "onnxruntime (CPUExecutionProvider)",
  "elapsed_s": 3.814
 },
 "hybrid": {
  "passes": 10,
  "elapsed_s": 1.47
 },
 "consistency": "ok",
 "elapsed_s": 11.16
}
```

## 6. Artifacts
```
comparison_ml.png    HTTP 200   2485497 bytes
preview.png          HTTP 200   1411791 bytes
refined_preview.png  HTTP 200   1105914 bytes
hybrid_preview.png   HTTP 200   1179901 bytes
convergence.png      HTTP 200     28722 bytes
result.npz           HTTP 200   3457038 bytes
refined.npz          HTTP 200   3336472 bytes
hybrid.npz           HTTP 200   1675525 bytes
summary.json         HTTP 200      5135 bytes
```

reference_job=348a4d4abca745d6 job=9e3bd431b1ea4243 dataset=056ff76ebec94e41
