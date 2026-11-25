"""
FastAPI application — Velos BioImaging reconstruction API (v1).

Endpoints
---------
GET  /health                              liveness + queue depth (public)
POST /api/v1/contact                      website lead capture (public, honeypot + rate limit)
POST /api/v1/datasets                     upload a microscopy dataset (.mat/.npz/.h5)   [API key]
GET  /api/v1/datasets                     list my datasets                               [API key]
GET  /api/v1/datasets/{id}                dataset metadata + geometry summary            [API key]
POST /api/v1/jobs                         submit a reconstruction job (returns at once)  [API key]
GET  /api/v1/jobs                         list my jobs                                   [API key]
GET  /api/v1/jobs/{id}                    status / progress                              [API key]
POST /api/v1/jobs/{id}/cancel             request cancellation                           [API key]
GET  /api/v1/jobs/{id}/result             result summary + artifact links                [API key]
GET  /api/v1/jobs/{id}/artifacts/{name}   download an artifact (png / npz / json)        [API key]
GET  /api/v1/models                       installed ML model packages (model cards)      [API key]
GET  /metrics                             Prometheus metrics (jobs, queue, HTTP, workers) [internal / token]

Interactive docs: /docs (Swagger UI), /redoc.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlmodel import Session, func, select

from velos_recon import load_dataset

from . import __version__
from .auth import can_access, require_admin, require_api_key
from .config import ApiKeyInfo, get_settings
from .db import get_engine, get_session
from .models import (ContactIn, ContactLead, Dataset, DatasetOut, Job, JobCreate, JobOut,
                     JobResultOut, JobStatus)
from .queue import enqueue, request_cancel
from .storage import get_storage

log = logging.getLogger("velos.api")
ALLOWED_EXT = {".mat", ".npz", ".h5", ".hdf5"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    from .config import configure_logging
    configure_logging(settings.log_level)
    get_engine()           # creates tables
    get_storage()          # creates folders
    if settings.insecure_default_key:
        log.warning("VELOS_API_KEYS not set: using the insecure development key 'dev-key'")
    log.info("Velos API %s ready · storage=%s", __version__, settings.storage_dir)
    yield


app = FastAPI(
    title="Velos BioImaging Reconstruction API",
    version=__version__,
    description="Upload ptychographic microscopy data, run SPARK physics-based (and ML-refined) "
                "reconstructions asynchronously, and retrieve amplitude/phase results.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins,
                   allow_methods=["*"], allow_headers=["*"])


_http_count: dict = defaultdict(int)
_http_seconds: dict = defaultdict(float)


@app.middleware("http")
async def _access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = time.perf_counter() - t0
    route = request.scope.get("route")
    key = (request.method, getattr(route, "path", "unmatched"), f"{response.status_code // 100}xx")
    _http_count[key] += 1
    _http_seconds[key] += dt
    log.info("%s %s → %s (%.0f ms)", request.method, request.url.path, response.status_code, dt * 1000)
    return response


# ------------------------------------------------------------------ helpers
def _dataset_out(d: Dataset) -> DatasetOut:
    return DatasetOut(id=d.id, name=d.name, filename=d.filename, size_bytes=d.size_bytes,
                      owner=d.owner, summary=json.loads(d.summary_json or "{}"), created_at=d.created_at)


def _job_out(j: Job) -> JobOut:
    links = {"self": f"/api/v1/jobs/{j.id}"}
    if j.status == JobStatus.succeeded:
        links["result"] = f"/api/v1/jobs/{j.id}/result"
    return JobOut(id=j.id, dataset_id=j.dataset_id, owner=j.owner, status=j.status,
                  params=json.loads(j.params_json or "{}"), progress=j.progress, iteration=j.iteration,
                  message=j.message, error=j.error, created_at=j.created_at, started_at=j.started_at,
                  finished_at=j.finished_at, links=links)


def _get_job_or_404(session: Session, job_id: str, who: ApiKeyInfo) -> Job:
    job = session.get(Job, job_id)
    if job is None or not can_access(who, job.owner):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job


# ------------------------------------------------------------------ public
@app.get("/health", tags=["system"])
def health(session: Session = Depends(get_session)):
    queued = session.exec(select(func.count()).select_from(Job).where(Job.status == JobStatus.queued)).one()
    running = session.exec(select(func.count()).select_from(Job).where(Job.status == JobStatus.running)).one()
    from velos_recon.refine import list_packages
    settings = get_settings()
    cards = list_packages(settings.model_dir)
    defaults = _default_models(cards, settings.default_model)
    models = {"installed": [c["id"] for c in cards],
              "default": defaults.get("ptychography"),          # unchanged meaning (M3): ptychography default
              "default_by_modality": defaults}
    return {"status": "ok", "version": __version__, "queue": {"queued": queued, "running": running},
            "models": models}


def _default_models(cards: list, configured: str | None) -> dict:
    """Default package per modality, as the worker resolves it (``VELOS_DEFAULT_MODEL`` if it is of that
    modality, otherwise the first installed package of the modality)."""
    out: dict = {}
    for c in cards:
        mod = c.get("modality") or "ptychography"
        out.setdefault(mod, c["id"])
        if c["id"] == configured:
            out[mod] = c["id"]
    return out


@app.get("/metrics", tags=["system"], include_in_schema=True)
def metrics(request: Request, session: Session = Depends(get_session)):
    """Prometheus text format: job counts by status, queue depth, job durations, datasets,
    installed models, worker heartbeats and per-route HTTP counters / latency sums.
    Protected by ``VELOS_METRICS_TOKEN`` (Bearer) when set; the edge server does not expose it."""
    from fastapi.responses import PlainTextResponse
    import os
    token = os.getenv("VELOS_METRICS_TOKEN")
    if token and request.headers.get("authorization", "") != f"Bearer {token}":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "metrics token required")
    from velos_recon.refine import list_packages
    settings = get_settings()
    lines = ["# HELP velos_build_info Velos API version.", "# TYPE velos_build_info gauge",
             f'velos_build_info{{version="{__version__}"}} 1',
             "# HELP velos_jobs Jobs by status.", "# TYPE velos_jobs gauge"]
    counts = dict(session.exec(select(Job.status, func.count()).group_by(Job.status)).all())
    for st in JobStatus:
        lines.append(f'velos_jobs{{status="{st.value}"}} {int(counts.get(st, 0))}')
    rows = session.exec(select(Job.started_at, Job.finished_at).where(Job.status == JobStatus.succeeded)).all()
    durs = [(f - s).total_seconds() for s, f in rows if s and f]
    lines += ["# HELP velos_job_duration_seconds Run time of succeeded jobs.", "# TYPE velos_job_duration_seconds summary",
              f"velos_job_duration_seconds_sum {sum(durs):.3f}", f"velos_job_duration_seconds_count {len(durs)}"]
    oldest = session.exec(select(func.min(Job.created_at)).where(Job.status == JobStatus.queued)).one()
    age = 0.0
    if oldest is not None:
        from datetime import datetime, timezone
        o = oldest if oldest.tzinfo else oldest.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - o).total_seconds()
    lines += ["# HELP velos_queue_oldest_seconds Age of the oldest queued job.", "# TYPE velos_queue_oldest_seconds gauge",
              f"velos_queue_oldest_seconds {age:.1f}"]
    n_ds = session.exec(select(func.count()).select_from(Dataset)).one()
    lines += ["# TYPE velos_datasets gauge", f"velos_datasets {int(n_ds)}",
              "# TYPE velos_models_installed gauge", f"velos_models_installed {len(list_packages(settings.model_dir))}"]
    lines += ["# HELP velos_worker_heartbeat_age_seconds Seconds since each worker last polled the queue.",
              "# TYPE velos_worker_heartbeat_age_seconds gauge"]
    hb_dir = settings.storage_dir / "heartbeats"
    now = time.time()
    if hb_dir.is_dir():
        for f in sorted(hb_dir.glob("*.json")):
            lines.append(f'velos_worker_heartbeat_age_seconds{{worker="{f.stem}"}} {now - f.stat().st_mtime:.1f}')
    lines += ["# HELP velos_http_requests_total HTTP requests by method, route and status class.",
              "# TYPE velos_http_requests_total counter"]
    for (m, r, c), n in sorted(_http_count.items()):
        lines.append(f'velos_http_requests_total{{method="{m}",route="{r}",code="{c}"}} {n}')
    lines += ["# HELP velos_http_request_seconds_total Summed request latency.", "# TYPE velos_http_request_seconds_total counter"]
    for (m, r, c), v in sorted(_http_seconds.items()):
        lines.append(f'velos_http_request_seconds_total{{method="{m}",route="{r}",code="{c}"}} {v:.4f}')
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/api/v1/models", tags=["models"])
def list_models(who: ApiKeyInfo = Depends(require_api_key)):
    """Installed refinement-model packages with their model cards (version, training data,
    evaluation summary, intended use, limitations). Use the ``id`` as ``model`` in a job."""
    from velos_recon.refine import list_packages
    settings = get_settings()
    cards = list_packages(settings.model_dir)
    defaults = set(_default_models(cards, settings.default_model).values())
    for c in cards:
        c.setdefault("modality", "ptychography")      # M3 packages predate the field
        c["default"] = c["id"] in defaults
    return cards


_contact_hits: dict[str, deque] = defaultdict(deque)


@app.post("/api/v1/contact", status_code=201, tags=["website"])
def contact(payload: ContactIn, request: Request, session: Session = Depends(get_session)):
    if payload.website:                                   # honeypot filled → bot
        return {"ok": True}
    ip = request.client.host if request.client else "?"
    now = time.time(); hits = _contact_hits[ip]
    while hits and now - hits[0] > 3600:
        hits.popleft()
    if len(hits) >= 10:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many messages, try later")
    hits.append(now)
    lead = ContactLead(name=payload.name, email=payload.email, organisation=payload.organisation,
                       message=payload.message)
    session.add(lead); session.commit()
    log.info("contact lead %s from %s <%s>", lead.id, lead.name, lead.email)
    return {"ok": True, "id": lead.id}


@app.get("/api/v1/contact", tags=["website"])
def list_contacts(who: ApiKeyInfo = Depends(require_admin), session: Session = Depends(get_session)):
    rows = session.exec(select(ContactLead).order_by(ContactLead.created_at.desc()).limit(500)).all()
    return [{"id": r.id, "name": r.name, "email": r.email, "organisation": r.organisation,
             "message": r.message, "created_at": r.created_at} for r in rows]


# ------------------------------------------------------------------ datasets
@app.post("/api/v1/datasets", response_model=DatasetOut, status_code=201, tags=["datasets"])
def upload_dataset(file: UploadFile = File(...), name: str | None = Form(None),
                   who: ApiKeyInfo = Depends(require_api_key), session: Session = Depends(get_session)):
    from pathlib import Path
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"unsupported file type '{ext}'; use {sorted(ALLOWED_EXT)}")
    settings = get_settings(); storage = get_storage()
    row = Dataset(name=name or Path(file.filename).stem, filename=file.filename, path="", size_bytes=0, owner=who.name)
    try:
        path, size = storage.save_upload(row.id, file.filename, file.file, settings.max_upload_mb * 1024 * 1024)
    except ValueError as e:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(e))
    try:
        from velos_recon.fpm import is_fpm_npz, load_fpm_npz
        if is_fpm_npz(path):                          # LED-array FPM dataset (M4)
            acq = load_fpm_npz(path, name=row.name)
            summary = acq.summary()
            summary["n_patterns"] = acq.n_frames
        else:
            ds = load_dataset(path, name=row.name)   # validates format + geometry
            summary = ds.summary()
            summary.setdefault("modality", "ptychography")
    except Exception as e:
        import shutil; shutil.rmtree(storage.dataset_dir(row.id), ignore_errors=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"dataset could not be parsed: {e}")
    row.path, row.size_bytes, row.summary_json = str(path), size, json.dumps(summary)
    session.add(row); session.commit(); session.refresh(row)
    log.info("dataset %s uploaded by %s (%d bytes, %s patterns)", row.id, who.name, size, summary["n_patterns"])
    return _dataset_out(row)


@app.get("/api/v1/datasets", response_model=List[DatasetOut], tags=["datasets"])
def list_datasets(who: ApiKeyInfo = Depends(require_api_key), session: Session = Depends(get_session)):
    q = select(Dataset).order_by(Dataset.created_at.desc())
    if who.role != "admin":
        q = q.where(Dataset.owner == who.name)
    return [_dataset_out(d) for d in session.exec(q).all()]


@app.get("/api/v1/datasets/{dataset_id}", response_model=DatasetOut, tags=["datasets"])
def get_dataset(dataset_id: str, who: ApiKeyInfo = Depends(require_api_key), session: Session = Depends(get_session)):
    d = session.get(Dataset, dataset_id)
    if d is None or not can_access(who, d.owner):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset not found")
    return _dataset_out(d)


# ------------------------------------------------------------------ jobs
@app.post("/api/v1/jobs", response_model=JobOut, status_code=202, tags=["jobs"])
def submit_job(payload: JobCreate, who: ApiKeyInfo = Depends(require_api_key), session: Session = Depends(get_session)):
    d = session.get(Dataset, payload.dataset_id)
    if d is None or not can_access(who, d.owner):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset not found")
    from .fpm_jobs import FPM_ENGINES
    modality = json.loads(d.summary_json or "{}").get("modality", "ptychography")
    if (modality == "fpm") != (payload.engine in FPM_ENGINES):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"engine '{payload.engine}' does not fit a {modality} dataset "
                            f"({'use fpm, fpm-ap, fpm-epry, fpm-rpie or fpm-all' if modality == 'fpm' else 'use epie, rpie, wasp, gd or map'})")
    if payload.probe_from_job:
        ref = session.get(Job, payload.probe_from_job)
        if ref is None or not can_access(who, ref.owner):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "probe_from_job: job not found")
        if ref.status != JobStatus.succeeded:
            raise HTTPException(status.HTTP_409_CONFLICT, f"probe_from_job: job is {ref.status.value}, not succeeded")
    job = enqueue(session, d.id, who.name, payload.model_dump())
    log.info("job %s queued by %s on dataset %s", job.id, who.name, d.id)
    return _job_out(job)


@app.get("/api/v1/jobs", response_model=List[JobOut], tags=["jobs"])
def list_jobs(who: ApiKeyInfo = Depends(require_api_key), session: Session = Depends(get_session)):
    q = select(Job).order_by(Job.created_at.desc()).limit(200)
    if who.role != "admin":
        q = q.where(Job.owner == who.name)
    return [_job_out(j) for j in session.exec(q).all()]


@app.get("/api/v1/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
def get_job(job_id: str, who: ApiKeyInfo = Depends(require_api_key), session: Session = Depends(get_session)):
    return _job_out(_get_job_or_404(session, job_id, who))


@app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobOut, tags=["jobs"])
def cancel_job(job_id: str, who: ApiKeyInfo = Depends(require_api_key), session: Session = Depends(get_session)):
    job = _get_job_or_404(session, job_id, who)
    if job.status not in (JobStatus.queued, JobStatus.running):
        raise HTTPException(status.HTTP_409_CONFLICT, f"job is {job.status.value}; cannot cancel")
    return _job_out(request_cancel(session, job))


@app.get("/api/v1/jobs/{job_id}/result", response_model=JobResultOut, tags=["jobs"])
def get_result(job_id: str, who: ApiKeyInfo = Depends(require_api_key), session: Session = Depends(get_session)):
    job = _get_job_or_404(session, job_id, who)
    if job.status != JobStatus.succeeded:
        raise HTTPException(status.HTTP_409_CONFLICT, f"job is {job.status.value}; result not available")
    storage = get_storage()
    names = [p.name for p in storage.job_artifacts(job_id)]
    links = {n: f"/api/v1/jobs/{job_id}/artifacts/{n}" for n in names}
    return JobResultOut(job_id=job_id, status=job.status, summary=json.loads(job.result_json or "{}"),
                        artifacts=names, links=links)


@app.get("/api/v1/jobs/{job_id}/artifacts/{name}", tags=["jobs"])
def get_artifact(job_id: str, name: str, who: ApiKeyInfo = Depends(require_api_key), session: Session = Depends(get_session)):
    _get_job_or_404(session, job_id, who)
    try:
        path = get_storage().artifact_path(job_id, name)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid artifact name")
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "artifact not found")
    return FileResponse(path)
