## 1. Health — API version and installed model packages
```
{
    "status": "ok",
    "version": "0.3.0",
    "queue": {
        "queued": 0,
        "running": 0
    },
    "models": {
        "installed": [
            "velos-refine-v2"
        ],
        "default": "velos-refine-v2"
    }
}
```

## 2. Model card of the default package (GET /api/v1/models)
```
{
 "id": "velos-refine-v2",
 "name": "velos-refine",
 "version": "2.0.0",
 "created": "2026-11-11",
 "default": true,
 "normalisation": "median_ramp",
 "tile": 256,
 "tile_overlap": 64,
 "architecture": {
  "type": "RefineUNet (residual U-Net, GroupNorm, SiLU)",
  "base_channels": 32,
  "depth": 4,
  "parameters": 7766018
 },
 "files": [
  "README.md",
  "model.onnx",
  "model.ts",
  "model_card.json"
 ]
}
verification: {"onnxruntime": 1.7642974853515625e-05, "torchscript": 0.0} passed: True
intended use: Research and development use inside the Velos/SPARK platform: optional refinement of ePIE reconstructions of reduced visible-light ptychography scans (about 35-80 % of a dense raster), followed by the hybrid physics polish. The physics result is always kept and returned alongside.
```

## 3. Upload the dataset
```
{
    "id": "18cb4c438f504ebc",
    "name": "plant_dense_c128.npz",
    "filename": "plant_dense_c128.npz",
    "size_bytes": 11783169,
    "owner": "velos-lab",
    "summary": {
        "name": "plant_dense_c128.npz",
        "n_patterns": 200,
        "frame_shape": [
            128,
            128
        ],
        "wavelength_nm": 675.0,
        "pixel_pitch_um": 29.6,
        "distance_mm": 29.5,
        "sample_pixel_um": 5.25562183277027,
        "object_shape": [
            275,
            273
        ],
        "median_scan_step_px": 9.219544457292887,
        "has_init_probe": true,
        "has_ground_truth": false,
        "dps_max": 37902.69921875,
        "dps_mean": 2680.02197265625
    },
    "created_at": "2026-11-11T09:38:44.043945Z"
}
```

## 3b. Probe calibration: full scan (all frames), physics only
```
calibration job dbe35cc3cb074658 queued
calibration job dbe35cc3cb074658: succeeded after ~66s
{
 "engine": "epie",
 "iters": 200,
 "elapsed_s": 65.04,
 "object_shape": [
  275,
  273
 ],
 "probe_shape": [
  128,
  128
 ],
 "sample_pixel_um": 5.25562183277027,
 "final_data_error": 417878912.4710813,
 "initial_data_error": 1882867598.0886772,
 "data_error_normalised": 0.04169586425830696
}
```

## 4. Submit: reduced scan (0.35 of the frames) + ML refinement + hybrid (10 it)
```
POST /api/v1/jobs {"dataset_id":"18cb4c438f504ebc","engine":"epie","iters":200,"sparse_fraction":0.35,"refine_ml":true,"hybrid_iters":10,"probe_from_job":"dbe35cc3cb074658"}
{
    "id": "bdfc6f37f74b4af3",
    "dataset_id": "18cb4c438f504ebc",
    "owner": "velos-lab",
    "status": "queued",
    "params": {
        "dataset_id": "18cb4c438f504ebc",
        "engine": "epie",
        "iters": 200,
        "seed": 0,
        "crop": null,
        "sparse_every": 1,
        "engine_params": {},
        "sparse_fraction": 0.35,
        "refine_ml": true,
        "model": null,
        "hybrid_iters": 10,
        "probe_from_job": "dbe35cc3cb074658"
    },
    "progress": 0.0,
    "iteration": 0,
    "message": "queued",
    "error": "",
    "created_at": "2026-11-11T09:39:52.028154Z",
    "started_at": null,
    "finished_at": null,
    "links": {
        "self": "/api/v1/jobs/bdfc6f37f74b4af3"
    }
}
```

## 5. Status polling
```
t=   1s  status=running  running
t=   6s  status=running  iteration 44/200 · err 1.156e+08
t=  11s  status=running  iteration 88/200 · err 1.074e+08
t=  16s  status=running  iteration 136/200 · err 1.015e+08
t=  21s  status=running  iteration 184/200 · err 9.924e+07
t=  26s  status=running  ML refinement
t=  27s  status=succeeded  completed
```

## 6. Result — physics, ML and hybrid, scored against the measured data
```
{
 "dataset": {
  "name": "plant_dense_c128.npz_rand0.35",
  "n_patterns": 70,
  "frame_shape": [
   128,
   128
  ],
  "wavelength_nm": 675.0,
  "sample_pixel_um": 5.25562183277027,
  "object_shape": [
   275,
   273
  ]
 },
 "reconstruction": {
  "engine": "epie",
  "iters": 200,
  "elapsed_s": 22.96,
  "object_shape": [
   275,
   273
  ],
  "probe_shape": [
   128,
   128
  ],
  "sample_pixel_um": 5.25562183277027,
  "final_data_error": 98076082.2419991,
  "initial_data_error": 940684693.2169268,
  "data_error_normalised": 0.02994221932054683,
  "probe_from_job": "dbe35cc3cb074658"
 },
 "ml_refinement": {
  "applied": true,
  "model": "velos-refine-v2",
  "model_name": "velos-refine",
  "model_version": "2.0.0",
  "runtime": "onnxruntime (CPUExecutionProvider)",
  "elapsed_s": 0.538,
  "normalisation": {
   "scale": 0.7751736560623894,
   "phase0": 0.23099067834739942,
   "kx": -0.0010661113176053305,
   "ky": -0.0006877533277761074,
   "ramp": true
  },
  "data_error": {
   "physics": 0.02994221932054683,
   "ml": 0.036664236574257554,
   "hybrid": 0.03129630224698003
  },
  "hybrid": {
   "iters": 10,
   "elapsed_s": 1.22,
   "files": {
    "amplitude": "hybrid_amplitude.png",
    "phase": "hybrid_phase.png",
    "preview": "hybrid_preview.png",
    "hybrid_npz": "hybrid.npz"
   }
  },
  "data_error_ratio": 1.045223198452227,
  "consistency": "ok"
 }
}
```

## 7. Artifacts
```
comparison.png       HTTP 200    699642 bytes
preview.png          HTTP 200    325415 bytes
refined_preview.png  HTTP 200    297143 bytes
hybrid_preview.png   HTTP 200    314050 bytes
probe.png            HTTP 200    143407 bytes
convergence.png      HTTP 200     27824 bytes
refined.npz          HTTP 200    851988 bytes
hybrid.npz           HTTP 200    674288 bytes
result.npz           HTTP 200    677080 bytes
summary.json         HTTP 200      2141 bytes
```

job_id=bdfc6f37f74b4af3 dataset_id=18cb4c438f504ebc
