# Velos reconstruction API — reference (v1)

Base URL: `http://localhost:8000` (dev). Interactive docs: `/docs` (Swagger UI), `/redoc`.
Authentication: header `X-API-Key: <key>`; keys and roles come from `VELOS_API_KEYS` (`key:name:role`).

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | – | liveness, version, queue depth, installed model packages (`models.default`: ptychography default; `models.default_by_modality`: one default per modality) |
| POST | `/api/v1/contact` | – | website lead capture (honeypot + 10/h per IP) |
| GET | `/api/v1/contact` | admin | list captured leads |
| POST | `/api/v1/datasets` | key | upload `.mat` / `.npz` / `.h5` (multipart `file`, optional `name`) |
| GET | `/api/v1/datasets` | key | list my datasets (admin: all) |
| GET | `/api/v1/datasets/{id}` | key | metadata + geometry summary |
| POST | `/api/v1/jobs` | key | submit a reconstruction job → `202 queued` |
| GET | `/api/v1/jobs` | key | list my jobs |
| GET | `/api/v1/jobs/{id}` | key | status, progress, iteration, message |
| POST | `/api/v1/jobs/{id}/cancel` | key | cancel queued/running job |
| GET | `/api/v1/jobs/{id}/result` | key | summary + artifact links (409 until succeeded) |
| GET | `/api/v1/jobs/{id}/artifacts/{name}` | key | download PNG / NPZ / JSON |
| GET | `/api/v1/models` | key | installed ML model packages with their model cards (`modality`: ptychography or fpm; `default`: true for the default of its modality) |
| GET | `/metrics` | internal (optional bearer token) | Prometheus metrics: jobs by status, queue age, job durations, HTTP counters, worker heartbeats (M4; not exposed by the edge server) |

## Job parameters

```json
{
  "dataset_id": "…",
  "engine": "epie",          // epie | rpie | wasp | gd | map;  FPM datasets: fpm | fpm-ap | fpm-epry | fpm-rpie | fpm-all
  "iters": 100,              // iterations (default 100) — for FPM: passes after the warm-up (default 30)
  "seed": 0,
  "crop": 256,               // optional centre-crop of diffraction frames (fast preview)
  "sparse_every": 1,         // 2 → keep every 2nd grid row & column (sparse-scan experiment)
  "sparse_fraction": null,   // 0.35 → keep a random 35 % of the frames (reduced-scan experiment, M3)
  "engine_params": {},       // e.g. {"alpha": 1.0, "beta": 1.0} or {"probe_lr": 0.1}
  "refine_ml": false,        // M3: apply the packaged refinement model after the physics engine
  "model": null,             // M3: model package id from GET /api/v1/models (null: the default package of the dataset's modality)
  "hybrid_iters": 0          // M3: hybrid mode — N physics iterations seeded with the ML-refined object
}
```

### ML refinement and hybrid mode (M3)

With `refine_ml: true` the worker normalises the physics object (phase ramp, median amplitude and
phase), runs the model package with ONNX Runtime in 256×256 tiles (Hann-blended), maps the result back
to the physics frame and, if `hybrid_iters > 0`, runs that many physics iterations starting from the
refined object and the physics probe. Every estimate is scored against the **measured** diffraction
data (no ground truth needed). The result summary gains:

```json
"ml_refinement": {
  "applied": true, "model": "velos-refine-v2", "model_version": "2.0.0",
  "runtime": "onnxruntime (CPUExecutionProvider)", "elapsed_s": 0.54,
  "data_error": {"physics": 0.030, "ml": 0.037, "hybrid": 0.031},   // plant demo, 35 % of the frames
  "data_error_ratio": 1.05,
  "consistency": "ok",          // "warning: …" if the final error is > 2× physics AND > 0.02 (model card)
  "hybrid": {"iters": 10, "elapsed_s": 1.2, "files": {…}},
  "files": {"amplitude": "refined_amplitude.png", "phase": "refined_phase.png", "comparison": "comparison.png", …}
}
```

Recommended `hybrid_iters` for experimental data: **10** (longer polishing over-fits the measured frames —
see `docs/ml-pipeline.md`, M3 §6). Probe calibration: run a full scan first and pass its id as `probe_from_job`.

Artifacts added: `refined.npz` (refined object in the physics frame and in the normalised frame, plus
the normalisation parameters), `refined_*.png`, `hybrid.npz`, `hybrid_*.png`, `comparison.png`
(physics | ML | hybrid, amplitude and phase on shared colour scales).

## Example session (curl)

```bash
export KEY=dev-key API=http://localhost:8000

# 1. upload
curl -s -H "X-API-Key: $KEY" -F "file=@clam_sparse_every2_c256.npz" -F "name=clam sparse" $API/api/v1/datasets
# → {"id":"4f1c…","summary":{"n_patterns":100,"frame_shape":[256,256],"wavelength_nm":635,…}}

# 2. submit
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"dataset_id":"4f1c…","engine":"epie","iters":150}' $API/api/v1/jobs
# → 202 {"id":"9a2b…","status":"queued",…}

# 3. poll
curl -s -H "X-API-Key: $KEY" $API/api/v1/jobs/9a2b…
# → {"status":"running","progress":0.42,"iteration":63,"message":"iteration 63/150 · err 1.2e+08"}

# 4. result
curl -s -H "X-API-Key: $KEY" $API/api/v1/jobs/9a2b…/result
curl -s -H "X-API-Key: $KEY" -o phase.png $API/api/v1/jobs/9a2b…/artifacts/phase.png
```

## Status model

`queued → running → succeeded | failed | cancelled`; a running job that receives `/cancel` passes through
`cancelling` until the worker acknowledges. Errors are returned in `error` and written to `error.log` in the
job directory.

## LED-array Fourier ptychography (M4)

Datasets in the Velos FPM format (`.npz` with `modality = "fpm"`, made by `python -m velos_recon.fpm convert`)
are recognised on upload (`summary.modality = "fpm"`, `n_frames`, stored region, set-up defaults). Jobs on them
must use an FPM engine; a scanning-ptychography engine on an FPM dataset (or the reverse) returns 422.

```json
{"dataset_id": "…", "engine": "fpm", "iters": 24,
 "fpm": {"Np": 256, "z_source_mm": 59.4, "geometry_check": "report"},   // any velos_recon.fpm.FPMConfig field
 "sparse_fraction": 0.5,          // reduced LED set: all LEDs inside 1.1 × NA + 50 % of the dark-field LEDs
 "refine_ml": true, "hybrid_iters": 10}
```

Artifacts: `amplitude.png`, `phase.png`, `preview.png` (curvature and tilt removed), `convergence.png`,
`<ALGO>_amplitude.tif` / `_phase.tif` / `_pupil_phase.tif` (same format as the MATLAB script), `comparison.png`,
`frames.csv`, `summary.txt`, `fpm_log.txt`, `result.npz` (complex objects, pupils, residuals, unwrapped phase); with
`refine_ml`: `refined_*.png`, `refined.npz`, `hybrid_*.png`, `hybrid.npz`, `comparison_ml.png`. The summary reports
the residuals, the data residual of the primary (EPRY) result, and for reduced LED sets the prediction error on the
LEDs that were not used — for the physics, refined and hybrid estimates. Details: `docs/fpm.md`.

