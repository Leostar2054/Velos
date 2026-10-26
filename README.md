# Velos BioImaging platform — monorepo

End-to-end software stack for Velos QPI / SPARK (Contractor Services Agreement, 12 Oct – 27 Nov 2026):
company website, reconstruction backend/API with asynchronous jobs, ML training pipeline, and deployment files.

| Folder | Module (Schedule A) | What is inside |
|---|---|---|
| `website/` | 1 — Website | Next.js 15 + Tailwind static site: Home, Product, Team, Contact; editable `content/site.json`; SEO, sitemap, analytics hook; contact form → API |
| `backend/` | 2 — Backend/API | FastAPI API, SQLite job queue, worker, storage, API-key auth; `velos_recon` physics-engine integration; tests |
| `ml/` | 3 — ML pipeline | Config-driven PyTorch pipeline: phantom data generation, residual U-Net, training, evaluation vs ground truth and physics baseline, inference/export |
| `docker-compose.yml`, `*/Dockerfile` | 4 — Deployment | Containerised website + API + worker (staging stack scaffold for M3) |
| `docs/` | — | Architecture, data review, API reference, ML pipeline notes |
| `evidence/` | — | Milestone evidence: screenshots, API transcript, figures, metrics |
| `scripts/` | — | `demo_m2.sh` end-to-end demo script |

## Run everything locally (5 minutes)

```bash
# backend (API + worker)
cd backend && pip install -e ".[dev]" && cp .env.example .env && set -a && . ./.env && set +a
uvicorn velos_backend.main:app --port 8000 &          # http://localhost:8000/docs
python -m velos_backend.worker &
pytest -q                                              # 10 tests

# website
cd ../website && npm install && npm run build && npx serve out   # or: npm run dev

# ML (needs torch; CPU is fine for the baseline)
cd ../ml && pip install -e . 
python -m velos_ml.simulate --config configs/data_synthetic.yaml --out data/synthetic
python -m velos_ml.train    --config configs/baseline_unet.yaml --data data/synthetic --out runs/baseline_unet
python -m velos_ml.evaluate --data data/synthetic --checkpoint runs/baseline_unet/best.pt --out results/baseline_unet

# end-to-end API demo (API + worker running)
API=http://localhost:8000 KEY=dev-key scripts/demo_m2.sh path/to/dataset.npz epie 60 demo_out
```

## Milestone status

| Milestone | Scope | Status |
|---|---|---|
| M1 (12–19 Oct) | architecture, repositories, website/API scaffold, data review, ML scaffold | delivered with M2 (see `docs/architecture.md`, `docs/data-review.md`) |
| M2 (20–28 Oct) | backend core (queue/storage/auth), website pages, first ML baseline, demo + baseline comparison | **this delivery** — see `evidence/` and the M2 report |
| M3 (29 Oct–12 Nov) | model refinement/packaging, backend integration, staging deployment, website QA | next |
| M4 (13–27 Nov) | production deployment, monitoring, documentation, handover | — |

Secrets are never stored in the repository; see the `.env.example` files.
