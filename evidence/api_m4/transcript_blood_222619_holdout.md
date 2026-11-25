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
 "id": "5d624c5419944c49",
 "name": "blood_222619_holdout_c160.npz",
 "size_bytes": 7187787,
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
POST /api/v1/jobs {"dataset_id":"5d624c5419944c49","engine":"fpm","iters":24}
job db12298ebedb48fd: queued
t=   1s  status=queued  queued
t=  11s  status=running  EPRY pass 24/26 · residual 0.1823
t=  19s  status=succeeded  completed
{
 "leds_used": 218,
 "grid_px": 480,
 "pixel_um": 0.42,
 "synthetic_NA": 0.5532251003225204,
 "resolution_um": {
  "raw": 6.33,
  "synthetic": 1.1441997111681526
 },
 "final_residual": [
  0.18099427085708902
 ],
 "data_error_normalised": 0.17535644990110658,
 "elapsed_s": 19.42
}
```

## 4. Reduced LED set (bright field + 0.3 of the dark-field LEDs) → FPM model → hybrid (10 passes)
```
POST /api/v1/jobs {"dataset_id":"5d624c5419944c49","engine":"fpm","iters":24,"sparse_fraction":0.3,"refine_ml":true,"hybrid_iters":10}
{
    "id": "e7931e856c314387",
    "dataset_id": "5d624c5419944c49",
    "owner": "demo",
    "status": "queued",
    "params": {
        "dataset_id": "5d624c5419944c49",
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
    "created_at": "2026-11-25T09:13:59.650359Z",
    "started_at": null,
    "finished_at": null,
    "links": {
        "self": "/api/v1/jobs/e7931e856c314387"
    }
}
t=   1s  status=queued  queued
t=  11s  status=running  EPRY pass 26/26 · residual 0.1657
t=  21s  status=succeeded  completed
```

## 5. Result — physics, AI model and hybrid, each scored against the measured frames
```
{
 "dataset": {
  "name": "blood_222619",
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
  "leds_used": 72,
  "leds_available": 218
 },
 "data_error_used_leds": {
  "physics": 0.15501613609277137,
  "ml": 0.15350565931181107,
  "hybrid": 0.15123754792385383
 },
 "prediction_error_unused_leds": {
  "physics": 0.4046190012364367,
  "ml": 0.3607489863069923,
  "hybrid": 0.3838381704817367
 },
 "model": {
  "model": "velos-refine-fpm-v1",
  "model_version": "1.0.0",
  "runtime": "onnxruntime (CPUExecutionProvider)",
  "elapsed_s": 3.874
 },
 "hybrid": {
  "passes": 10,
  "elapsed_s": 1.41
 },
 "consistency": "ok",
 "elapsed_s": 11.1
}
```

## 6. Artifacts
```
comparison_ml.png    HTTP 200   2081018 bytes
preview.png          HTTP 200   1016661 bytes
refined_preview.png  HTTP 200    840089 bytes
hybrid_preview.png   HTTP 200    947451 bytes
convergence.png      HTTP 200     28854 bytes
result.npz           HTTP 200   3463337 bytes
refined.npz          HTTP 200   3355232 bytes
hybrid.npz           HTTP 200   1723400 bytes
summary.json         HTTP 200      5380 bytes
```

reference_job=db12298ebedb48fd job=e7931e856c314387 dataset=5d624c5419944c49
