# Velos BioImaging platform — monorepo

End-to-end software stack for Velos QPI / SPARK (Contractor Services Agreement, 12 Oct – 27 Nov 2026):
company website, reconstruction backend/API with asynchronous jobs, ML training pipeline, and deployment files.

| Folder | Module (Schedule A) | What is inside |
|---|---|---|
| `website/` | 1 — Website | Next.js 15 + Tailwind static site: Home, Product, Team, Contact; editable `content/site.json`; SEO, sitemap, analytics hook; contact form → API |
| `backend/` | 2 — Backend/API | FastAPI API, SQLite job queue, worker, storage, API-key auth; `velos_recon` physics-engine integration; tests |
| `ml/` | 3 — ML pipeline | Config-driven PyTorch pipeline: synthetic + experimental training data, refinement U-Net (v2), two-stage training, evaluation vs ground truth / dense reference, ONNX + TorchScript packaging with model card |
| `docker-compose.yml`, `docker-compose.prod.yml`, `deploy/`, `*/Dockerfile`, `.github/workflows/` | 4 — Deployment | Local stack; production/staging stack with Caddy edge (TLS), hardened images, CI/CD workflow, smoke test — runbook in `docs/deployment.md` |
| `docs/` | — | Architecture, data review, API reference, ML pipeline notes, deployment runbook |
| `evidence/` | — | Milestone evidence: screenshots, API transcripts, figures, metrics, QA reports |
| `scripts/` | — | `demo_m2.sh`, `demo_m3.sh` (end-to-end demos), `smoke_test.sh`, `website_qa.js`, report builders |

## Run everything locally (5 minutes)

```bash
# backend (API + worker)
cd backend && pip install -e ".[dev]" && cp .env.example .env && set -a && . ./.env && set +a
uvicorn velos_backend.main:app --port 8000 &          # http://localhost:8000/docs
python -m velos_backend.worker &
pytest -q                                              # 20 tests

# website
cd ../website && npm install && npm run build && npx serve out   # or: npm run dev

# ML (needs torch; CPU is fine for the baseline)
cd ../ml && pip install -e . 
python -m velos_ml.simulate --config configs/data_synthetic.yaml --out data/synthetic
python -m velos_ml.train    --config configs/baseline_unet.yaml --data data/synthetic --out runs/baseline_unet
python -m velos_ml.evaluate --data data/synthetic --checkpoint runs/baseline_unet/best.pt --out results/baseline_unet

# M3 model (refinement v2): full pipeline in ml/scripts/m3_pipeline.sh; package → backend
python -m velos_ml.package runs/refine_v2/best.pt --out ../backend/models/velos-refine-v2 --version 2.0.0

# end-to-end API demos (API + worker running; VELOS_MODEL_DIR=backend/models for M3)
API=http://localhost:8000 KEY=dev-key scripts/demo_m2.sh path/to/dataset.npz epie 60 demo_out
API=http://localhost:8000 KEY=dev-key scripts/demo_m3.sh path/to/clam_dense_c128.npz 0.35 200 30 demo_m3_out
scripts/smoke_test.sh http://localhost:8000 dev-key     # SKIP_WEB=1 when only the API runs
```

## Milestone status

| Milestone | Scope | Status |
|---|---|---|
| M1 (12–19 Oct) | architecture, repositories, website/API scaffold, data review, ML scaffold | delivered with M2 (see `docs/architecture.md`, `docs/data-review.md`) |
| M2 (20–28 Oct) | backend core (queue/storage/auth), website pages, first ML baseline, demo + baseline comparison | delivered 28 Oct — see the M2 report |
| M3 (29 Oct–12 Nov) | model refinement, packaging (ONNX/TorchScript) and backend integration (ML + hybrid modes), end-to-end inference demo, website QA, deployment-ready package | **this delivery** — see the M3 report; cloud staging moved to M4 (agreed 11 Nov) |
| M4 (13–27 Nov) | staging + production deployment on a Velos-owned cloud account (GPU worker), biological sample data, real setup figure, monitoring, handover | next |

Secrets are never stored in the repository; see the `.env.example` files.
