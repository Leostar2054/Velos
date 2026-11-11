"""
Database-backed job queue.

Jobs are rows in the ``job`` table. The API only *enqueues* (status=queued) and
returns immediately; a separate worker process *claims* the oldest queued job
with an atomic conditional UPDATE, so several workers can run safely against
the same database. This keeps M2 free of extra infrastructure (no Redis
broker needed on a laptop) while the claim/ack semantics map 1:1 onto Celery
or RQ if the deployment (M4) needs a distributed broker.
"""
from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Session, select, update

from .models import Job, JobStatus


def _now() -> datetime:
    return datetime.now(timezone.utc)


def enqueue(session: Session, dataset_id: str, owner: str, params: dict) -> Job:
    job = Job(dataset_id=dataset_id, owner=owner, params_json=json.dumps(params),
              status=JobStatus.queued, message="queued")
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def claim_next(session: Session, worker_id: Optional[str] = None) -> Optional[Job]:
    """Atomically move the oldest queued job to 'running' and return it."""
    worker_id = worker_id or f"{socket.gethostname()}"
    candidate = session.exec(
        select(Job).where(Job.status == JobStatus.queued).order_by(Job.created_at).limit(1)
    ).first()
    if candidate is None:
        return None
    result = session.exec(
        update(Job)
        .where(Job.id == candidate.id, Job.status == JobStatus.queued)
        .values(status=JobStatus.running, started_at=_now(), worker_id=worker_id,
                message="running")
    )
    session.commit()
    if result.rowcount != 1:          # another worker won the race
        return None
    session.refresh(candidate)
    return candidate


def update_progress(session: Session, job_id: str, iteration: int, total: int, message: str = "") -> None:
    session.exec(update(Job).where(Job.id == job_id).values(
        progress=min(1.0, iteration / max(total, 1)), iteration=iteration,
        message=message or f"iteration {iteration}/{total}"))
    session.commit()


def is_cancel_requested(session: Session, job_id: str) -> bool:
    session.expire_all()
    job = session.get(Job, job_id)
    return job is not None and job.status == JobStatus.cancelling


def finish(session: Session, job_id: str, status: JobStatus, result: dict | None = None,
           error: str = "") -> None:
    values = dict(status=status, finished_at=_now(), error=error[:4000],
                  result_json=json.dumps(result) if result else "",
                  message={JobStatus.succeeded: "completed", JobStatus.failed: "failed",
                           JobStatus.cancelled: "cancelled"}.get(status, ""))
    if status == JobStatus.succeeded:
        values["progress"] = 1.0
    # failed / cancelled jobs keep the progress they reached (the column is NOT NULL; the M2
    # code wrote None here, which made a failing or cancelled *running* job stay "running")
    session.rollback()                       # discard a half-finished transaction from the failure
    session.exec(update(Job).where(Job.id == job_id).values(**values))
    session.commit()


def request_cancel(session: Session, job: Job) -> Job:
    if job.status == JobStatus.queued:
        job.status = JobStatus.cancelled; job.finished_at = _now(); job.message = "cancelled"
    elif job.status == JobStatus.running:
        job.status = JobStatus.cancelling; job.message = "cancellation requested"
    session.add(job); session.commit(); session.refresh(job)
    return job
