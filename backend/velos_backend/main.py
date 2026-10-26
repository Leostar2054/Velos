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
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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


@app.middleware("http")
async def _access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s → %s (%.0f ms)", request.method, request.url.path, response.status_code,
             (time.perf_counter() - t0) * 1000)
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
    return {"status": "ok", "version": __version__, "queue": {"queued": queued, "running": running}}


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
        ds = load_dataset(path, name=row.name)       # validates format + geometry
        summary = ds.summary()
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
