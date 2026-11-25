# Handover guide — operate, maintain, retrain and redeploy the Velos platform

For a small technical team taking over the platform (Agreement §10: start / deploy / retrain with the documentation).
Everything lives in the Client-owned repository `github.com/Leostar2054/Velos`.

## 1. What is where

| Path | Content |
|---|---|
| `website/` | Next.js 15 static site (Home, Product, Team, Contact). Texts, team and contact details: `website/content/site.json` |
| `backend/velos_backend/` | FastAPI API (`main.py`), job queue (`queue.py`), worker (`worker.py`, `fpm_jobs.py`), storage, auth, config |
| `backend/velos_recon/` | reconstruction library: `core.py` (Client's ptychography engines, unchanged), `engine.py`, `data.py`, `fpm.py` (LED-array FPM, port of the MATLAB code), `refine.py` (model runtime), `normalise.py`, `metrics.py`, `viz.py` |
| `ml/` | training pipeline: simulation, real-data pair builders (`realdata.py`, `fpm_realdata.py`), U-Net, training, evaluation (`evaluate_m3.py`, `evaluate_fpm.py`), packaging; pipelines `ml/scripts/m3_pipeline.sh`, `ml/scripts/m4_fpm_pipeline.sh` |
| `scripts/` | `deploy_server.sh`, `smoke_test.sh`, `load_test.py`, `demo_m3.sh` / `demo_m4.sh` (end-to-end API demos), `website_qa.js`, `license_report.py`, diagram generators |
| `models/` | installed model packages (ONNX + model card) |
| `examples/fpm/` | 256-px crops of the real blood-smear and leaf acquisitions (API format) |
| `tools/fpm_packing/` | scripts that pack raw FPM acquisitions (225 TIFFs) into compact files for the platform (Python / MATLAB) |
| `deploy/`, `docker-compose*.yml`, `backend/Dockerfile*`, `scripts/deploy_server.sh` | deployment (runbook: `docs/deployment.md`) |
| `.github/workflows/ci.yml` | CI/CD (tests, stack rehearsal, image publishing, optional deploy) |
| `docs/` | architecture, API, data review, FPM, ML pipeline, deployment runbook, this guide, licences |
| `evidence/` | milestone evidence (figures, metrics, transcripts, QA reports) |

## 2. Run everything on a laptop (15 minutes)

```bash
git clone https://github.com/Leostar2054/Velos.git && cd Velos
cd backend && python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,ml]"
pytest -q                                                   # all tests must pass
export VELOS_MODEL_DIR=$PWD/../models VELOS_API_KEYS=dev-key:dev:admin
uvicorn velos_backend.main:app --port 8000 &                # http://localhost:8000/docs
python -m velos_backend.worker &
cd .. && FPM_DATA=examples/fpm/blood_222619_c256.npz SKIP_WEB=1 scripts/smoke_test.sh http://localhost:8000 dev-key
cd website && npm ci && npm run dev                         # http://localhost:3000
```

Or the full production stack locally: `cp deploy/.env.example deploy/.env` (set `SITE_DOMAIN=:80`,
`SITE_URL=http://localhost`, keys), then `docker compose -f docker-compose.prod.yml --env-file deploy/.env up -d --build`.

## 3. Reconstruct data

* **Scanning ptychography** (`.mat` Maiden format, `.npz`, `.h5`): upload, then a job with `engine` = `epie` | `rpie` |
  `wasp` | `gd` | `map`; options for reduced scans, probe calibration, ML refinement and hybrid mode (`docs/api.md`).
* **LED-array FPM (in-house microscope)**: convert an acquisition folder once —
  `python -m velos_recon.fpm convert <folder> my_scan_c512.npz --crop 512 --set lambda_um=0.633 z_source_mm=59.4 optical_axis_x_mm=-0.70 optical_axis_y_mm=-0.23 Np=512` —
  upload it and run `engine: "fpm"` (EPRY), `fpm-ap`, `fpm-rpie` or `fpm-all`. On the command line:
  `python -m velos_recon.fpm run <folder> --Np 512 --z 55 --passes 30 --geometry-check report`.
* **New set-up geometry**: run once with `--geometry-check apply` on a full scan; it reports the LED distance and
  optical-axis offset that fit the frames (`summary.txt`); store them as the dataset defaults (`--set …`).
* **End-to-end FPM demo through the API** (reduced LED set → AI model → hybrid, all scored against the frames):
  `API=http://localhost:8000 KEY=<key> scripts/demo_m4.sh <fpm.npz> 0.3 24 10 demo_out` (transcripts in `evidence/api_m4/`).
* **Large data**: the platform reconstructs patches (`Np`); for a full field run several patches
  (`crop_center_x_px`, `crop_center_y_px`). Raw 225-frame scans are ~0.9 GB; keep them on lab storage and upload crops.

## 4. Deploy and operate

See `docs/deployment.md`: first deployment (`scripts/deploy_server.sh`), updates and rollback, monitoring (Prometheus,
alert rules), logs, backups and recovery procedures, GPU worker, CI/CD settings, security checklist.

## 5. Retrain the AI models

| Model | Data | Pipeline | Time (CPU, 2 cores) |
|---|---|---|---|
| `velos-refine-v2` (scanning ptychography) | simulated phantoms + 635 nm clam scan | `ml/scripts/m3_pipeline.sh` (`MAT_DIR=<folder with the .mat datasets>`) | several hours (GPU: < 1 h) |
| `velos-refine-fpm-v1` (LED-array FPM) | blood-smear and leaf scans of 23 Nov 2026 (+ side regions for testing) | `PACK=<packed crops> ml/scripts/m4_fpm_pipeline.sh` (stages `data`, `train`, `eval`, `package`) | ~1.5 h data + 50 min training + 30 min evaluation (GPU: much less) |

To add new FPM samples: pack the acquisitions (`tools/fpm_packing/`: `pack_fpm_for_velos.py` / `.m` keeps all frames
of the central N×N pixels; `pack_fpm_side_regions.m` packs the regions left and right of it), add them to the `data` stage of
`m4_fpm_pipeline.sh` (training draws `--draws 3`; other regions of an already calibrated scan: `--config-from
<centre>/meta.json`), add the folder to `data.real.root` in `ml/configs/refine_fpm_v1.yaml`, rerun `train`, `eval`,
`package`, then copy the new package to `models/` with a new version and check the evaluation in its model card
**before** making it the default. Keep the rule used so far: evaluate only on regions, frame draws and acquisitions
never used for training, and always report the physics result (and the same number of extra physics passes) next to
the model output.

## 6. Routine tasks

| Task | How |
|---|---|
| Add / rotate API keys | edit `VELOS_API_KEYS` in `deploy/.env`, `docker compose … up -d api worker` |
| Website text, team, contact address | `website/content/site.json` (address: `CONTACT_EMAIL` in `deploy/.env`), redeploy |
| Install / roll back a model | copy / remove a directory in `models/`, restart `api worker` (§5 of the runbook) |
| Read contact-form leads | `GET /api/v1/contact` with an admin key |
| Dependency updates | update `backend/pyproject.toml` or `website/package.json`, push; CI must stay green |
| Licences | `docs/THIRD_PARTY_LICENSES.md` (regenerate with `scripts/license_report.py`) |

## 7. Known limitations and open items

* No Velos cloud account yet: the production stack is rehearsed on every push in CI; the server rollout follows the
  runbook (§4 B of `docs/deployment.md`). TLS on the real domain and the GPU worker are untested until then.
* The FPM AI model is validated for the **blood smear in the hybrid mode** (`hybrid_iters` 5) only. Its output alone
  removes reduced-LED artefacts but does not predict unmeasured LED images better (for viewing, not for quantitative
  values); for the **leaf section it is not validated** (output inconsistent with the measured images). It was trained
  on one field of view per specimen type: re-evaluate it on new samples before use (model card, `docs/ml-pipeline.md`).
* The four blood scans (and the leaf scans) of 23 Nov repeat the same field of view; the evaluation therefore uses
  spatial hold-out regions, unseen LED draws and sensor regions outside the training crop.
* `run_info.json` labels the LEDs GREEN while the analysis uses 633 nm (red, as in the set-up photo) — confirm.
* No dark frames are recorded; the camera offset is estimated from the darkest frame when possible.
* SQLite is adequate for a single server; move to PostgreSQL (`VELOS_DB_URL`) before running several servers.
* Warranty: 30 days after final acceptance for reproducible material bugs (Agreement §12).
