# Deployment and operations runbook — Velos platform (website + API + worker + AI models)

Status (M4, 25 Nov 2026). The complete production stack is built, started and tested automatically on every
push to `main` of the Client-owned repository (`github.com/Leostar2054/Velos`, GitHub Actions job **stack**):
HTTPS edge server with the website, API, worker, Prometheus monitoring; smoke test including an FPM job on
real blood-smear data, a load/stability test, a monitoring check and a redeployment test. Images are published
to the GitHub Container Registry. The rollout on a Velos-owned cloud or on-premises server uses the same files
with `scripts/deploy_server.sh` (§4); a GPU worker is one override file away (§8). No Velos cloud account existed
at M4; see the M4 scope-change confirmation.

## 1. Topology

```
                    Internet (HTTPS 443 / HTTP 80 → redirect)
                                   │
                        ┌──────────▼──────────┐
                        │ edge  (Caddy 2.8)   │  automatic TLS certificates, gzip/zstd, security headers,
                        │ static website      │  JSON access log; /metrics is NOT exposed
                        │ /api/* /health /docs│──────────┐
                        └─────────────────────┘          │ reverse proxy
                                                ┌────────▼────────┐        ┌──────────────────┐
                                                │ api (FastAPI)   │◄───────│ prometheus       │  scrapes /metrics,
                                                │ uvicorn :8000   │ /metrics│ (profile         │  alert rules,
                                                └────────┬────────┘        │  "monitoring")   │  127.0.0.1:9090
                               SQLite (WAL) job queue +  │                 └──────────────────┘
                               datasets + job outputs    │  volume velos-data:/data
                                                ┌────────▼────────┐
                                                │ worker (1..n)   │  ptychography engines (Client's ptycho_core),
                                                │ CPU or GPU      │  LED-array FPM engine (port of the MATLAB code),
                                                └─────────────────┘  ONNX models from ./models (read-only)
```

* One VM is enough for the product-validation stage (4 vCPU / 16 GB RAM; for GPU inference one T4/L4-class GPU).
* Datasets, results and the job database live on the `velos-data` volume (back it up, §6).
* Model packages are directories under `./models` (tracked in git: `velos-refine-v2` for scanning ptychography,
  `velos-refine-fpm-v1` for LED-array FPM; the TorchScript copies are in the release package).

## 2. Prerequisites (Client side)

| Item | Owner | Notes |
|---|---|---|
| Cloud account (AWS / GCP / Azure / NCP) or an on-premises Linux server | **Velos BioImaging** | Paid by Velos; contractors get a limited user. No deployment runs under a personal account. |
| VM with Docker Engine + Compose v2 | Velos | Ubuntu 22.04/24.04 LTS; ports 80/443 open; SSH key-only, restricted to admin IPs |
| DNS record | Velos | `A` record of the domain (or a sub-domain such as `app.`) → VM public IP |
| Mailbox for certificate notices | Velos | `ACME_EMAIL` |
| GitHub repository | Velos (done) | `Leostar2054/Velos`; secrets/variables for the deploy job (§7) |

## 3. Configuration and secrets

Everything comes from environment variables; nothing secret is stored in git (`deploy/.env.example` is the template
for the server's `deploy/.env`, `chmod 600`).

| Variable | Default | Meaning |
|---|---|---|
| `SITE_DOMAIN` | — | domain served by the edge (`:80` = plain HTTP, used in CI) |
| `SITE_URL` | `https://velos.example.com` | public address used in canonical URLs and the sitemap |
| `ACME_EMAIL` | — | certificate-authority contact |
| `CONTACT_EMAIL` | empty (hidden) | contact address shown on the website |
| `VELOS_API_KEYS` | — (dev key if unset) | `key:name:role` list; generate keys with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `VELOS_DEFAULT_MODEL` | first package of the modality | ptychography default package; FPM jobs pick an FPM package automatically |
| `VELOS_ML_RUNTIME` | `auto` | `onnxruntime` (CPU or CUDA) or `torchscript` |
| `VELOS_MAX_UPLOAD_MB` | 2048 | upload limit (the edge also limits bodies to 2 GB) |
| `VELOS_LOG_FORMAT` | `json` in production | `json` (one object per line) or `text` |
| `VELOS_METRICS_TOKEN` | empty | if set, `/metrics` needs `Authorization: Bearer <token>` (add it to `deploy/prometheus/prometheus.yml`) |
| `VELOS_TAG` | `0.4.0` | image tag |

## 4. Deployment

**A. Automatic rehearsal on every push (already running).** The CI job *stack* (`.github/workflows/ci.yml`) copies
`deploy/.env.example`, sets CI values, then runs exactly the production commands: `docker compose -f
docker-compose.prod.yml --profile monitoring build/up`, `scripts/smoke_test.sh` (with `FPM_DATA=examples/fpm/
blood_222619_c256.npz`), `scripts/load_test.py`, the Prometheus checks and a redeployment (`up -d --force-recreate api
worker`). Its log is the deployment evidence (Actions tab → run → *Production stack*).

**B. First deployment on a Velos server.**

```bash
sudo mkdir -p /opt/velos && sudo chown $USER /opt/velos && cd /opt/velos
git clone https://github.com/Leostar2054/Velos.git .
cp deploy/.env.example deploy/.env && chmod 600 deploy/.env && nano deploy/.env   # domain, keys, e-mails
MONITORING=1 SMOKE_API_KEY=<a user key> scripts/deploy_server.sh                  # build, start, wait, smoke test
```

`deploy_server.sh` validates the compose file, builds the images for the real domain, starts or updates the stack,
waits for `/health` and runs the smoke test. The edge obtains the TLS certificate on first start (DNS must point to
the VM). Alternatively pull the published images: `ghcr.io/leostar2054/velos-backend:<sha>` and
`ghcr.io/leostar2054/velos-edge:<sha>` (the edge image is built for the placeholder site URL; build it on the server
for the real domain).

## 5. Updates, model releases and rollback

* **Code update:** `git pull && MONITORING=1 scripts/deploy_server.sh`. Running jobs finish on the old worker
  (`docker compose stop worker` first if a long job must not be interrupted). With `DEPLOY_ENABLED=true` the CI deploy
  job does this over SSH after every green build of `main`.
* **Model update:** copy the new package directory into `models/` (e.g. `velos-refine-fpm-v2/`), check its
  `model_card.json` (verification passed, evaluation, limitations), restart `api worker`. Older packages stay
  selectable per job (`"model": "<id>"`), so a model rollback is a one-line change.
* **Rollback (code):** `git checkout <previous tag or commit> && scripts/deploy_server.sh`.

## 6. Monitoring, logging, backups and recovery

**Metrics.** `GET /metrics` (Prometheus text format, inside the compose network only): jobs by status, queue age,
job durations, datasets, installed models, per-route HTTP counts and latency, worker heartbeats. Start Prometheus with
`MONITORING=1` (profile `monitoring`); UI on `127.0.0.1:9090` (SSH tunnel: `ssh -L 9090:localhost:9090 <server>`).
Alert rules (`deploy/prometheus/alerts.yml`): API down, worker stalled (no heartbeat for 2 min), queue backlog
(> 15 min), more than 3 failed jobs per hour, API 5xx errors. Connect Alertmanager or Grafana for e-mail/chat delivery.
External uptime check: `https://<domain>/health` (status, version, queue depth, models).

**Health checks.** `api`: HTTP `/health` every 30 s; `worker`: heartbeat file `/data/heartbeats/<id>.json` younger
than 120 s; `docker compose ps` shows both as *healthy*.

**Logs.** `docker compose logs -f api worker edge`; API and worker write JSON lines (`VELOS_LOG_FORMAT=json`), the edge
writes JSON access logs; the Docker json-file driver rotates at 20 MB × 5 files.

**Backups.** Nightly copy of the `velos-data` volume (database, datasets, results) plus `models/` and `deploy/.env`
(encrypted):

```bash
docker run --rm -v velos_velos-data:/d -v /backup:/b alpine tar czf /b/velos-$(date +%F).tgz -C /d .
```

**Recovery procedures.**

| Situation | Action |
|---|---|
| API unhealthy / 5xx alerts | `docker compose logs --tail=200 api`; `docker compose restart api`; if a bad release: roll back (§5) |
| Worker stalled (heartbeat alert) or a job stuck in "running" | `docker compose restart worker`; jobs that were running are reported failed/cancelled by the queue (fixed in M3); resubmit |
| Queue backlog | add workers: `docker compose up -d --scale worker=2` (SQLite queue claims jobs atomically) |
| Disk full | delete old job outputs (`/data/jobs/<id>`) after export, extend the volume; alert at 80 % |
| Lost server | new VM → §4 B → restore: `docker run --rm -v velos_velos-data:/d -v /backup:/b alpine tar xzf /b/velos-<date>.tgz -C /d` → `scripts/deploy_server.sh` |
| Certificate problem | check DNS and port 80/443, `docker compose logs edge`; Caddy retries automatically |

## 7. CI/CD (`.github/workflows/ci.yml`)

| Job | What it does | When |
|---|---|---|
| backend | installs the backend with ONNX Runtime, runs the test suite (scanning ptychography, FPM, ML runtime, metrics) | every push / PR |
| ml | tiny simulate → train → package with ONNX/TorchScript verification; FPM engine on the real blood sample; real-sample FPM pipeline (training pairs → evaluation with model, control and hybrid → summary) | every push / PR |
| website | build + QA crawl (3 viewports, links, images, duplicate ids, axe-core) — fails on any issue | every push / PR |
| stack | production stack on the runner: build, start, smoke test incl. real FPM data, load test, Prometheus checks, redeployment; evidence artifact | every push / PR |
| publish | images to `ghcr.io/<owner>/velos-backend` and `velos-edge` (tags: commit, `latest`) | push to `main` |
| deploy | SSH to the Velos server, checkout the commit, `MONITORING=1 scripts/deploy_server.sh`, smoke test | `main`, only with the repository variable `DEPLOY_ENABLED=true` |

Deploy job settings: variables `DEPLOY_ENABLED`, `DEPLOY_HOST` (`user@host`), `SITE_DOMAIN`; secrets `DEPLOY_SSH_KEY`,
`SMOKE_API_KEY`; GitHub environment `production` (add required reviewers to approve each rollout).

## 8. GPU inference worker

`docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml --env-file deploy/.env up -d --build` (or
`GPU=1 scripts/deploy_server.sh`) builds the worker from `backend/Dockerfile.gpu` (CUDA 12.4 runtime + onnxruntime-gpu)
and reserves one GPU; the NVIDIA driver and container toolkit must be installed on the VM. ONNX Runtime then uses the
CUDA execution provider automatically; each job reports the runtime it used (`ml_refinement.runtime`). The physics
engines run on the CPU in both images.

## 9. What was verified in M4

| Check | Result |
|---|---|
| Backend tests (API, queue, worker, FPM engine, ML runtime, metrics, JSON logs) | 30 passed (local and in CI) |
| Production stack on the GitHub runner: build, start, health, website pages, auth, models, ptychography job with ML + hybrid, FPM job on real blood data (116 of 221 LEDs) | all checks passed |
| Load / stability test (24 jobs, 6 parallel submitters + 2 polling clients, 2 cancelled on purpose) | 22 succeeded, 2 cancelled, 0 request errors, 32.8 jobs/min with one CPU worker; p95 latency 9.8 ms (`/health`), 4.9 ms (job status), 43 ms (job submit) |
| Monitoring | Prometheus target *up*, 5 alert rules loaded, `velos_jobs` query answered, `/metrics` not reachable through the edge |
| Redeployment (recreate api + worker) | 26 jobs still visible afterwards, smoke test passed again |
| Website QA crawl (4 pages × 3 viewports, axe-core) | 0 issues |

Not verified (needs Velos resources): TLS issuance on the real domain, GPU execution, backups on the cloud volume.

## 10. Security checklist

- [ ] `deploy/.env` readable only by the deploy user (`chmod 600`); never committed
- [ ] no insecure default key (the API logs a warning if `VELOS_API_KEYS` is unset)
- [ ] SSH key-only login; firewall 22 (admin IPs), 80, 443; Prometheus only on 127.0.0.1
- [ ] automatic security updates on the VM; images rebuilt monthly (CI does this on every push)
- [ ] backups tested by one restore
- [ ] contact-form leads are personal data: admin keys only; retention policy agreed by Velos
