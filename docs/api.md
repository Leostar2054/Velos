# Velos reconstruction API — reference (v1)

Base URL: `http://localhost:8000` (dev). Interactive docs: `/docs` (Swagger UI), `/redoc`.
Authentication: header `X-API-Key: <key>`; keys and roles come from `VELOS_API_KEYS` (`key:name:role`).

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | – | liveness, version, queue depth |
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

## Job parameters

```json
{
  "dataset_id": "…",
  "engine": "epie",          // epie | rpie | wasp | gd | map
  "iters": 100,
  "seed": 0,
  "crop": 256,               // optional centre-crop of diffraction frames (fast preview)
  "sparse_every": 1,         // 2 → keep every 2nd grid row & column (sparse-scan experiment)
  "engine_params": {},       // e.g. {"alpha": 1.0, "beta": 1.0} or {"probe_lr": 0.1}
  "refine_ml": false         // apply the ML refinement model (needs VELOS_ML_CHECKPOINT)
}
```

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
