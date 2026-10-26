## 1. Health (public)
```
{"status":"ok","version":"0.2.0","queue":{"queued":0,"running":0}}
```

## 2. Auth check — no key → 401
```
HTTP 401
```

## 3. Upload dataset
```
{
    "id": "ae1cd7569b9c4389",
    "name": "clam_sparse_every2_c128.npz",
    "filename": "clam_sparse_every2_c128.npz",
    "size_bytes": 5963511,
    "owner": "awais",
    "summary": {
        "name": "clam_sparse_every2_c128.npz",
        "n_patterns": 100,
        "frame_shape": [
            128,
            128
        ],
        "wavelength_nm": 635.0000000000001,
        "pixel_pitch_um": 29.6,
        "distance_mm": 60.0,
        "sample_pixel_um": 10.055954391891893,
        "object_shape": [
            402,
            401
        ],
        "median_scan_step_px": 27.16615541441225,
        "has_init_probe": true,
        "has_ground_truth": false,
        "dps_max": 65382.609375,
        "dps_mean": 607.9990844726562
    },
    "created_at": "2026-10-26T14:51:12.818269Z"
}
```

## 4. Submit job (returns immediately)
```
{
    "id": "563e035442554315",
    "dataset_id": "ae1cd7569b9c4389",
    "owner": "awais",
    "status": "queued",
    "params": {
        "dataset_id": "ae1cd7569b9c4389",
        "engine": "epie",
        "iters": 60,
        "seed": 0,
        "crop": null,
        "sparse_every": 1,
        "engine_params": {},
        "refine_ml": false
    },
    "progress": 0.0,
    "iteration": 0,
    "message": "queued",
    "error": "",
    "created_at": "2026-10-26T14:51:12.961997Z",
    "started_at": null,
    "finished_at": null,
    "links": {
        "self": "/api/v1/jobs/563e035442554315"
    }
}
```

## 5. Poll status (web request never blocks; worker runs the job)
```
t=  2s  status=queued     progress=0.00  queued
t=  4s  status=running    progress=0.17  iteration 10/60 · err 3.840e+07
t=  6s  status=running    progress=0.32  iteration 19/60 · err 2.891e+07
t=  8s  status=running    progress=0.55  iteration 33/60 · err 2.413e+07
t= 10s  status=running    progress=0.70  iteration 42/60 · err 2.258e+07
t= 12s  status=running    progress=0.90  iteration 54/60 · err 2.117e+07
t= 14s  status=succeeded  progress=1.00  completed
```

## 6. Result summary
```
{
    "job_id": "563e035442554315",
    "status": "succeeded",
    "summary": {
        "dataset": {
            "name": "clam_sparse_every2_c128.npz",
            "n_patterns": 100,
            "frame_shape": [
                128,
                128
            ],
            "wavelength_nm": 635.0000000000001,
            "pixel_pitch_um": 29.6,
            "distance_mm": 60.0,
            "sample_pixel_um": 10.055954391891893,
            "object_shape": [
                402,
                401
            ],
            "median_scan_step_px": 27.16615541441225,
            "has_init_probe": true,
            "has_ground_truth": false,
            "dps_max": 65382.609375,
            "dps_mean": 607.9990844726562
        },
        "reconstruction": {
            "engine": "epie",
            "iters": 60,
            "elapsed_s": 11.74,
            "object_shape": [
                402,
                401
            ],
            "probe_shape": [
                128,
                128
            ],
            "sample_pixel_um": 10.055954391891893,
            "final_data_error": 20607650.446759775,
            "initial_data_error": 120593261.29555608
        },
        "files": {
            "amplitude": "amplitude.png",
            "phase": "phase.png",
            "preview": "preview.png",
            "probe": "probe.png",
            "convergence": "convergence.png"
        }
    },
    "artifacts": [
        "amplitude.png",
        "convergence.png",
        "phase.png",
        "preview.png",
        "probe.png",
        "result.npz",
        "summary.json"
    ],
    "links": {
        "amplitude.png": "/api/v1/jobs/563e035442554315/artifacts/amplitude.png",
        "convergence.png": "/api/v1/jobs/563e035442554315/artifacts/convergence.png",
        "phase.png": "/api/v1/jobs/563e035442554315/artifacts/phase.png",
        "preview.png": "/api/v1/jobs/563e035442554315/artifacts/preview.png",
        "probe.png": "/api/v1/jobs/563e035442554315/artifacts/probe.png",
        "result.npz": "/api/v1/jobs/563e035442554315/artifacts/result.npz",
        "summary.json": "/api/v1/jobs/563e035442554315/artifacts/summary.json"
    }
}
```

## 7. Download artifacts
```
amplitude.png      230338 bytes
phase.png          439595 bytes
preview.png        682532 bytes
probe.png          152756 bytes
convergence.png     31753 bytes
summary.json          983 bytes
result.npz        1307673 bytes
```

job_id=563e035442554315 dataset_id=ae1cd7569b9c4389
