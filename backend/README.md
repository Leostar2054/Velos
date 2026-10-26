# velos-backend — API, job queue, worker and physics-engine integration

## Quick start (development)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                   # edit keys; never commit .env
set -a; source .env; set +a                            # Windows: set each VELOS_* variable

uvicorn velos_backend.main:app --reload --port 8000    # terminal 1: API  → http://localhost:8000/docs
python -m velos_backend.worker                         # terminal 2: worker (add more for parallel jobs)
pytest -q                                              # 10 tests incl. end-to-end API flow
```

## Packages

* `velos_recon/` — reconstruction package
  * `core.py` — the Client's `ptycho_core.py`, vendored unchanged (+1-line integration shim)
  * `data.py` — loaders (`.mat` Maiden `expt` struct, `.npz`, `.h5`), geometry, crop/bin, sparse subsets
  * `engine.py` — `reconstruct(dataset, ReconConfig)` with progress/cancel; probe seeding incl. automatic
    Fresnel chirp (see `docs/data-review.md`)
  * `metrics.py` — NRMSE / PSNR / SSIM on amplitude and phase after removing ptychographic ambiguities
  * `viz.py`, `cli.py` — figures and a CLI (`velos-recon info|subset|reconstruct`)
* `velos_backend/` — FastAPI app (`main.py`), models, DB session, auth, storage, queue, worker

## CLI examples

```bash
velos-recon info "real data - clam.mat"
velos-recon subset "real data - clam.mat" --every 2 --crop 256 --out clam_sparse.npz
velos-recon reconstruct clam_sparse.npz --engine epie --iters 150 --out results/clam_sparse
```

## Configuration

All settings are environment variables (documented in `velos_backend/config.py` and `.env.example`).
`VELOS_API_KEYS` must be set in any shared environment; without it the server starts with the development key
`dev-key` and logs a warning.

## Docker

`docker build -t velos-backend .` then run the same image as API (default command) and as worker
(`python -m velos_backend.worker`) sharing a volume mounted at `/data` — see `../docker-compose.yml`.
