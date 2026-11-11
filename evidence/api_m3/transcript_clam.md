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
    "id": "c68ff5581c8f49a3",
    "name": "clam_dense_c128.npz",
    "filename": "clam_dense_c128.npz",
    "size_bytes": 23842949,
    "owner": "velos-lab",
    "summary": {
        "name": "clam_dense_c128.npz",
        "n_patterns": 400,
        "frame_shape": [
            128,
            128
        ],
        "wavelength_nm": 635.0000000000001,
        "pixel_pitch_um": 29.6,
        "distance_mm": 60.0,
        "sample_pixel_um": 10.055954391891893,
        "object_shape": [
            417,
            416
        ],
        "median_scan_step_px": 12.041594578792296,
        "has_init_probe": true,
        "has_ground_truth": false,
        "dps_max": 65382.609375,
        "dps_mean": 598.0736083984375
    },
    "created_at": "2026-11-11T09:37:44.549497Z"
}
```

## 4. Submit: reduced scan (0.35 of the frames) + ML refinement + hybrid (10 it)
```
POST /api/v1/jobs {"dataset_id":"c68ff5581c8f49a3","engine":"epie","iters":200,"sparse_fraction":0.35,"refine_ml":true,"hybrid_iters":10}
{
    "id": "efc7671e94ab41b5",
    "dataset_id": "c68ff5581c8f49a3",
    "owner": "velos-lab",
    "status": "queued",
    "params": {
        "dataset_id": "c68ff5581c8f49a3",
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
        "probe_from_job": null
    },
    "progress": 0.0,
    "iteration": 0,
    "message": "queued",
    "error": "",
    "created_at": "2026-11-11T09:37:44.888400Z",
    "started_at": null,
    "finished_at": null,
    "links": {
        "self": "/api/v1/jobs/efc7671e94ab41b5"
    }
}
```

## 5. Status polling
```
t=   1s  status=queued  queued
t=   6s  status=running  iteration 20/200 · err 4.838e+07
t=  11s  status=running  iteration 44/200 · err 3.949e+07
t=  16s  status=running  iteration 68/200 · err 3.589e+07
t=  21s  status=running  iteration 88/200 · err 3.401e+07
t=  26s  status=running  iteration 112/200 · err 3.251e+07
t=  31s  status=running  iteration 132/200 · err 3.142e+07
t=  36s  status=running  iteration 156/200 · err 3.082e+07
t=  41s  status=running  iteration 180/200 · err 3.013e+07
t=  46s  status=running  iteration 200/200 · err 2.961e+07
t=  51s  status=running  ML refinement
t=  52s  status=succeeded  completed
```

## 6. Result — physics, ML and hybrid, scored against the measured data
```
{
 "dataset": {
  "name": "clam_dense_c128.npz_rand0.35",
  "n_patterns": 140,
  "frame_shape": [
   128,
   128
  ],
  "wavelength_nm": 635.0000000000001,
  "sample_pixel_um": 10.055954391891893,
  "object_shape": [
   417,
   416
  ]
 },
 "reconstruction": {
  "engine": "epie",
  "iters": 200,
  "elapsed_s": 46.82,
  "object_shape": [
   417,
   416
  ],
  "probe_shape": [
   128,
   128
  ],
  "sample_pixel_um": 10.055954391891893,
  "final_data_error": 29609428.041463684,
  "initial_data_error": 158465567.42071658,
  "data_error_normalised": 0.02061254366339795
 },
 "ml_refinement": {
  "applied": true,
  "model": "velos-refine-v2",
  "model_name": "velos-refine",
  "model_version": "2.0.0",
  "runtime": "onnxruntime (CPUExecutionProvider)",
  "elapsed_s": 0.65,
  "normalisation": {
   "scale": 0.8781878977322216,
   "phase0": -0.022759718534426095,
   "kx": 0.0011974589291570972,
   "ky": -0.000894740418333264,
   "ramp": true
  },
  "data_error": {
   "physics": 0.02061254366339795,
   "ml": 0.032473410370280245,
   "hybrid": 0.02605394981349725
  },
  "hybrid": {
   "iters": 10,
   "elapsed_s": 2.26,
   "files": {
    "amplitude": "hybrid_amplitude.png",
    "phase": "hybrid_phase.png",
    "preview": "hybrid_preview.png",
    "hybrid_npz": "hybrid.npz"
   }
  },
  "data_error_ratio": 1.2639851848931045,
  "consistency": "ok"
 }
}
```

## 7. Artifacts
```
comparison.png       HTTP 200   1721589 bytes
preview.png          HTTP 200    697157 bytes
refined_preview.png  HTTP 200    616332 bytes
hybrid_preview.png   HTTP 200    660071 bytes
probe.png            HTTP 200    143270 bytes
convergence.png      HTTP 200     33473 bytes
refined.npz          HTTP 200   2008211 bytes
hybrid.npz           HTTP 200   1396301 bytes
result.npz           HTTP 200   1400689 bytes
summary.json         HTTP 200      2116 bytes
```

job_id=efc7671e94ab41b5 dataset_id=c68ff5581c8f49a3
