# Velos BioImaging platform

Software stack of **Velos QPI / SPARK** — label-free quantitative phase imaging with physics-based and AI-assisted
reconstruction: company website, reconstruction API with asynchronous jobs, AI training pipeline, trained models,
and the production deployment (HTTPS edge server, API, workers, monitoring).

[![ci](https://github.com/Leostar2054/Velos/actions/workflows/ci.yml/badge.svg)](https://github.com/Leostar2054/Velos/actions/workflows/ci.yml)

| Folder | What is inside |
|---|---|
| `website/` | Next.js 15 + Tailwind static site: Home, Product, Team, Contact; editable `content/site.json`; SEO, sitemap; contact form → API |
| `backend/` | FastAPI API, job queue, workers, storage, API-key auth, Prometheus metrics; `velos_recon`: scanning ptychography (the Client's engines, unchanged) and **LED-array Fourier ptychography** (Python port of the in-house MATLAB AP/EPRY/rPIE code), model runtime (ONNX Runtime / TorchScript) |
| `ml/` | config-driven PyTorch pipeline: simulated and real training pairs, refinement U-Net, two-stage training, evaluation against ground truth and full-data references, ONNX + TorchScript packaging with model cards |
| `models/` | trained packages: `velos-refine-v2` (scanning ptychography), `velos-refine-fpm-v1` (LED-array FPM, real samples) |
| `examples/fpm/` | real data: 256-px crops of the in-house microscope's blood-smear and leaf scans (225 LEDs each) |
| `tools/fpm_packing/` | packing scripts for raw FPM acquisitions (Python / MATLAB / Windows launcher) |
| `deploy/`, `docker-compose.prod.yml`, `docker-compose.gpu.yml`, `scripts/deploy_server.sh` | production deployment (Caddy TLS edge, API, worker, Prometheus; optional GPU worker) |
| `.github/workflows/ci.yml` | tests, ML smoke run, website QA, **full production stack on every push** (smoke + real FPM job, load test, monitoring, redeploy), image publishing, optional server deploy |
| `docs/` | [architecture](docs/architecture.md) · [API](docs/api.md) · [FPM](docs/fpm.md) · [ML pipeline](docs/ml-pipeline.md) · [data review](docs/data-review.md) · [deployment runbook](docs/deployment.md) · [handover guide](docs/handover.md) · [licences](docs/THIRD_PARTY_LICENSES.md) |
| `evidence/` | milestone evidence: figures, metrics, API transcripts, QA reports, CI logs |

## Quick start

```bash
cd backend && pip install -e ".[dev,ml]" && pytest -q
export VELOS_MODEL_DIR=$PWD/../models VELOS_API_KEYS=dev-key:dev:admin
uvicorn velos_backend.main:app --port 8000 &        # API docs: http://localhost:8000/docs
python -m velos_backend.worker &
cd .. && FPM_DATA=examples/fpm/blood_222619_c256.npz SKIP_WEB=1 scripts/smoke_test.sh http://localhost:8000 dev-key

# LED-array FPM from the command line (AP / EPRY / rPIE, MATLAB-compatible outputs)
python -m velos_recon.fpm run examples/fpm/leaf_214040_c256.npz --passes 30 --out results/leaf

# website
cd website && npm ci && npm run dev                   # http://localhost:3000
```

Production: see [docs/deployment.md](docs/deployment.md) (`scripts/deploy_server.sh` on a Velos server; the same
stack is rehearsed by CI on every push). Handover, retraining and routine tasks: [docs/handover.md](docs/handover.md).

## Project history

| Milestone | Scope |
|---|---|
| M1–M2 (Oct 2026) | architecture, repositories, data review; backend core (queue, storage, auth, Client's engines); website; first ML baseline |
| M3 (Nov 2026) | refinement model v2 (synthetic + experimental fine-tuning), ONNX/TorchScript packaging, backend integration (ML + hybrid modes), end-to-end inference, website QA, deployment-ready package |
| M4 (Nov 2026) | in-house LED-array FPM engine (MATLAB → Python, validated) and real biological samples (blood smear, leaf) through the API; FPM AI model fine-tuned and evaluated on real data; production stack rehearsed in CI with load test and monitoring; GHCR images; runbook, handover guide, licences |

Configuration and secrets are environment variables only (`deploy/.env.example`, `backend/.env.example`).
Development used an AI coding assistant; such commits carry a `Co-Authored-By` trailer.
