# Deployment runbook — Velos platform (website + API + worker + ML model)

Status (M3, 12 Nov 2026): **deployment-ready package**. The images, production compose file,
edge-server configuration, CI/CD workflow and smoke test are in the repository and were
validated as far as possible without a cloud account and without a Docker daemon (see
§9). The actual staging/production deployment on a **Velos-owned** cloud account is
scheduled for **M4** (agreed with the Client on 11 Nov 2026).

## 1. Topology

```
                    Internet (HTTPS 443 / HTTP 80 → redirect)
                                   │
                        ┌──────────▼──────────┐
                        │ edge  (Caddy 2.8)   │  TLS certificates (automatic), gzip/zstd,
                        │ static website      │  security headers, JSON access log
                        │ /api/* /health /docs│──────────┐
                        └─────────────────────┘          │ reverse proxy
                                                ┌────────▼────────┐
                                                │ api (FastAPI)   │  uploads, jobs, results,
                                                │ uvicorn :8000   │  auth, model cards
                                                └────────┬────────┘
                               SQLite (WAL) job queue +  │  volume velos-data:/data
                               datasets + job outputs    │
                                                ┌────────▼────────┐
                                                │ worker (1..n)   │  physics engines (client's
                                                │ CPU now, GPU M4 │  ptycho_core) + ONNX model
                                                └─────────────────┘  read-only volume ./models
```

* One VM is enough for staging and early production (e.g. 4 vCPU / 16 GB; for GPU
  inference a single T4/L4-class GPU instance).
* Datasets, results and the job database live on the `velos-data` volume (back it up, §6).
* Model packages are plain directories under `./models` on the host (mounted read-only).

## 2. Prerequisites (Client side, M4)

| Item | Owner | Notes |
|---|---|---|
| Cloud account (AWS / GCP / Azure / NCP) | **Velos BioImaging** | Created and paid by Velos; the contractor gets a limited IAM user. No deployment runs under a personal account. |
| VM with Docker Engine + Compose v2 | Velos / contractor | Ubuntu 22.04/24.04 LTS; ports 80/443 open; SSH restricted to admin IPs |
| DNS record | Velos | `A` record of the chosen domain (or sub-domain such as `app.`) → VM public IP |
| Mailbox for certificate notices | Velos | used as `ACME_EMAIL` |
| Container registry (optional) | Velos | GHCR / ECR / Artifact Registry — needed only for the CI deploy job |
| GitHub repository settings | Velos | secrets for the deploy job (§7) |

## 3. Configuration and secrets

All configuration comes from environment variables; nothing secret is stored in git.

| File | Purpose |
|---|---|
| `deploy/.env.example` | template for the server's `deploy/.env` (domain, API keys, model, tag) |
| `backend/.env.example` | template for local development |

Generate API keys with `python -c "import secrets; print(secrets.token_urlsafe(32))"` and
set `VELOS_API_KEYS=key1:lab:user,key2:admin:admin`. Rotate keys by editing the file and
running `docker compose … up -d` (the API re-reads the environment on restart).

Main variables (full list in `backend/velos_backend/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `SITE_DOMAIN` | — | public domain served by the edge (":8080" = plain HTTP rehearsal) |
| `VELOS_API_KEYS` | dev key (insecure) | API keys and roles |
| `VELOS_MODEL_DIR` | `/models` | model packages |
| `VELOS_DEFAULT_MODEL` | first package | package used when a job does not name one |
| `VELOS_ML_RUNTIME` | `auto` | `onnxruntime` (preferred) or `torchscript` |
| `VELOS_MAX_UPLOAD_MB` | 2048 | upload limit (Caddy also limits bodies to 2 GB) |

## 4. First deployment

```bash
# on the server
sudo mkdir -p /opt/velos && cd /opt/velos
git clone <velos repository> .            # or copy the release bundle
cp deploy/.env.example deploy/.env && nano deploy/.env
mkdir -p models && cp -r <package>/velos-refine-v2 models/   # from 03_Model_Package
docker compose -f docker-compose.prod.yml --env-file deploy/.env up -d --build
docker compose -f docker-compose.prod.yml ps                 # api must be "healthy"
scripts/smoke_test.sh https://$SITE_DOMAIN <user-api-key>    # all checks must PASS
```

The edge obtains the TLS certificate automatically on first start (DNS must already point
to the VM). The smoke test checks health, the four website pages, authentication, the model
list, an upload, a reconstruction job with ML refinement + hybrid mode, the result and an
artifact download.

## 5. Updates, model releases and rollback

* **Code update:** `git pull && docker compose -f docker-compose.prod.yml --env-file deploy/.env up -d --build`,
  then the smoke test. Running jobs finish on the old worker (stop it with `docker compose stop worker`
  first if a long job must not be interrupted).
* **Model update:** copy the new package directory into `models/` (e.g. `velos-refine-v3/`),
  check its `model_card.json` (verification passed, evaluation, limitations), set
  `VELOS_DEFAULT_MODEL` and restart `api worker`. Older packages stay selectable per job
  (`"model": "<id>"`), so a model rollback is a one-line change.
* **Rollback (images):** set `VELOS_TAG` to the previous tag in `deploy/.env` and `up -d`.

## 6. Backups, logs and monitoring

* **Backup:** nightly copy of the `velos-data` volume (SQLite database + datasets + results):
  `docker run --rm -v velos_velos-data:/d -v /backup:/b alpine tar czf /b/velos-$(date +%F).tgz -C /d .`
  plus the `models/` directory and `deploy/.env` (store the latter encrypted).
* **Logs:** `docker compose logs -f api worker edge` (edge logs are JSON). Configure the Docker
  `json-file` driver with rotation (`max-size: 50m`, `max-file: 5`) or ship to the cloud's log service.
* **Monitoring:** an external uptime check on `https://<domain>/health` (status, version,
  queue depth, installed models); alert when the queue stays non-empty with no running job.
* **Disk:** alert at 80 % on the data volume (uploads up to 2 GB each).

## 7. CI/CD (`.github/workflows/ci.yml`)

| Job | What it does | When |
|---|---|---|
| backend | installs the backend with ONNX Runtime, runs the 19 tests | every push / PR |
| ml | tiny simulate → train → package, with the ONNX/TorchScript verification | every push / PR |
| website | build + QA crawl (3 viewports, links, images, duplicate ids, axe-core) — fails on any issue | every push / PR |
| images | `docker compose config`, builds both images, starts the stack, runs the smoke test | every push / PR |
| deploy | pushes images to the Velos registry, rolls out on staging over SSH, smoke-tests | `main`, only when the repository variable `DEPLOY_ENABLED=true` (M4) |

Deploy job settings (M4): variables `REGISTRY`, `DEPLOY_HOST`, `STAGING_DOMAIN`,
`DEPLOY_ENABLED`; secrets `REGISTRY_USER`, `REGISTRY_TOKEN`, `DEPLOY_SSH_KEY`, `SMOKE_API_KEY`.

## 8. GPU inference worker (M4)

The worker uses ONNX Runtime and picks the CUDA execution provider automatically when it is
available (`velos_recon.refine`). For a GPU VM: install the NVIDIA driver + container toolkit,
build the backend image on a CUDA runtime base with `--build-arg ML_EXTRA=ml-gpu`, and enable
the `deploy.resources.reservations.devices` block of the `worker` service. `/health` and each
job's `ml_refinement.runtime` field report which provider ran (e.g. `onnxruntime (CUDAExecutionProvider)`).
On CPU, one 256×256 tile takes about 0.5 s with ONNX Runtime (2 vCPU; 3–4× faster than eager PyTorch).

## 9. What was verified in M3 (no cloud account, no Docker daemon in the build environment)

| Check | Result |
|---|---|
| `docker compose -f docker-compose.prod.yml --env-file deploy/.env.example config` | valid |
| `docker compose -f docker-compose.yml config` | valid |
| CI workflow against the GitHub Actions JSON schema | valid |
| Backend test suite incl. ONNX runtime, hybrid mode, models endpoint | 19 passed |
| Smoke test script against a running API + worker (without and with a model package) | all checks passed |
| End-to-end demo through the API with the packaged model (ONNX Runtime) | see `01_Evidence/api_m3/` |
| Website QA crawl (4 pages × 3 viewports, axe-core WCAG 2.1 AA) | 0 issues after fixes |

Not yet verified (requires M4 resources): building and running the images (`docker build`
and `caddy validate` run in CI and in the edge image build), TLS issuance on the real
domain, GPU execution, backups on the cloud volume.

## 10. Security checklist

- [ ] `deploy/.env` readable only by the deploy user (`chmod 600`); never committed
- [ ] default dev key absent (`/health` works, but the API logs a warning if the insecure default is used)
- [ ] SSH key-only login, firewall: 22 (admin IPs), 80, 443
- [ ] automatic security updates on the VM; images rebuilt monthly
- [ ] backups tested by a restore at least once
- [ ] contact-form leads are personal data: access limited to admin keys; retention policy agreed by Velos
