# Velos BioImaging platform — architecture (M1 deliverable, updated at M2)

```
                         ┌──────────────────────────────┐
   browser ──────────────▶  Website (Next.js, static)    │   Home · Product · Team · Contact
                         │  website/                     │   contact form → POST /api/v1/contact
                         └──────────────┬───────────────┘
                                        │ NEXT_PUBLIC_API_URL
   researcher / script ─── X-API-Key ───▼──────────────────────────────────────────────┐
                         ┌──────────────────────────────┐        ┌─────────────────────┐│
                         │  API (FastAPI)  backend/      │        │  SQLite (WAL)        ││
                         │  velos_backend.main           │◀──────▶│  datasets · jobs ·   ││
                         │  /datasets /jobs /result      │        │  contact leads       ││
                         └──────────────┬───────────────┘        └──────────┬──────────┘│
                                        │ writes uploads                     │ claims    │
                         ┌──────────────▼───────────────┐        ┌──────────▼──────────┐│
                         │  Storage (local volume;       │◀──────▶│  Worker process(es)  ││
                         │  S3/GCS adapter at M4)        │ results│  velos_backend.worker││
                         │  datasets/<id>/  jobs/<id>/   │        └──────────┬──────────┘│
                         └──────────────────────────────┘                   │           │
                                                                 ┌───────────▼──────────┐│
                                                                 │ velos_recon           ││
                                                                 │ Client's physics      ││
                                                                 │ engines (core.py)     ││
                                                                 │ + velos_ml refinement ││
                                                                 └──────────────────────┘│
```

## Components

| Component | Location | Stack | Status at M2 |
|---|---|---|---|
| Website | `website/` | Next.js 15 (App Router) + Tailwind, static export, content in `content/site.json` | Home, Product, Team, Contact working; SEO metadata, sitemap, robots, analytics hook; contact form wired to API |
| API | `backend/velos_backend/main.py` | FastAPI, SQLModel/SQLite, API-key auth | Upload / job / status / result / artifact / cancel / contact endpoints, Swagger docs |
| Queue + worker | `backend/velos_backend/queue.py`, `worker.py` | DB-backed queue with atomic claim; worker is a separate process | Jobs never block the web request; progress + cancellation |
| Physics engines | `backend/velos_recon/` | Client's `ptycho_core.py` vendored unchanged (`core.py`) + driver, data loaders, metrics, viz | ePIE / rPIE / WASP / GD / MAP selectable per job |
| ML pipeline | `ml/velos_ml/` | PyTorch, YAML configs | Synthetic data generation, residual U-Net, training, evaluation vs GT & physics baseline, inference hook |
| Deployment | `docker-compose.yml`, `*/Dockerfile` | Docker | Scaffold for the M3 staging stack; not yet deployed to a cloud account |

## Key design decisions

1. **Physics logic is integrated, not rewritten.** `velos_recon/core.py` is the Client's `ptycho_core.py` with a one-line
   shim exposing the per-iteration step; a unit test proves the service reproduces the Client's engine bit-for-bit.
   Service-level behaviour (geometry, probe seeding, progress, cancellation) lives in `engine.py`.
2. **Queue without a broker for M2.** SQLite in WAL mode + an atomic conditional `UPDATE` claim gives correct
   multi-worker semantics on a laptop or a single VM. The claim/finish interface maps directly onto Celery/RQ if a
   distributed broker is needed at M4.
3. **Storage behind a 4-method interface** (`storage.py`) so an object-store backend can be added without touching
   the API or worker.
4. **Secrets only in the environment** (`VELOS_API_KEYS`, …); `.env.example` files document every variable.
5. **Static website.** No server-side rendering is needed for a marketing site; the export can be hosted on any CDN,
   nginx (see `website/Dockerfile`) or S3+CloudFront. Editable content lives in one JSON file.
6. **ML as a post-processing stage.** The first model refines the physics reconstruction from a sparse scan
   rather than replacing the physics; this keeps the quantitative reference intact and matches the SPARK mode
   model (physics-based → sparse-scanning → AI-assisted → hybrid).

## Request flow (end to end)

1. `POST /api/v1/datasets` (multipart) → file stored, parsed with `velos_recon.load_dataset` (format + geometry
   validation), summary saved → `201 {id, summary}`.
2. `POST /api/v1/jobs {dataset_id, engine, iters, …}` → row inserted with `status=queued` → `202` immediately.
3. Worker claims the job, runs `velos_recon.reconstruct` with a throttled progress callback (DB updates every
   0.5 s), checks for cancellation, writes `result.npz`, PNGs, `summary.json`, optionally runs ML refinement.
4. `GET /api/v1/jobs/{id}` shows `running · iteration k/N`; `GET …/result` lists artifacts and metrics;
   `GET …/artifacts/<name>` streams a file.

## Interfaces agreed at kickoff

* Dataset formats: MATLAB `expt` struct (`.mat`), `.npz`, `.h5` with `dps`, `positions`, `wavelength`,
  `pixel_pitch`, `distance`, optional `init_probe`, `ground_truth`.
* Job parameters: `engine`, `iters`, `seed`, `crop`, `sparse_every`, `engine_params`, `refine_ml`.
* Result bundle: `result.npz` (object, probe, mask, positions, dx, error history), `amplitude.png`, `phase.png`,
  `preview.png`, `probe.png`, `convergence.png`, `summary.json` (+ `refined_*` when ML is applied).
