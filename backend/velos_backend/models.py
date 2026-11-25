"""Database tables (SQLModel) and API schemas (pydantic)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field as PField, field_validator
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return uuid.uuid4().hex[:16]


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelling = "cancelling"
    cancelled = "cancelled"


# ------------------------------------------------------------------ tables
class Dataset(SQLModel, table=True):
    id: str = Field(default_factory=_uid, primary_key=True)
    name: str
    filename: str
    path: str
    size_bytes: int
    owner: str
    summary_json: str = "{}"
    created_at: datetime = Field(default_factory=_now)


class Job(SQLModel, table=True):
    id: str = Field(default_factory=_uid, primary_key=True)
    dataset_id: str = Field(index=True)
    owner: str
    status: JobStatus = Field(default=JobStatus.queued, index=True)
    params_json: str = "{}"
    progress: float = 0.0
    iteration: int = 0
    message: str = ""
    error: str = ""
    result_json: str = ""
    worker_id: str = ""
    created_at: datetime = Field(default_factory=_now, index=True)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class ContactLead(SQLModel, table=True):
    id: str = Field(default_factory=_uid, primary_key=True)
    name: str
    email: str
    organisation: str = ""
    message: str
    source: str = "website"
    created_at: datetime = Field(default_factory=_now)


# ------------------------------------------------------------------ API schemas
class DatasetOut(BaseModel):
    id: str
    name: str
    filename: str
    size_bytes: int
    owner: str
    summary: Dict[str, Any]
    created_at: datetime


class JobCreate(BaseModel):
    dataset_id: str
    engine: str = PField(default="epie", description=(
        "scanning ptychography: epie | rpie | wasp | gd | map;  LED-array FPM datasets: "
        "fpm (= fpm-epry) | fpm-ap | fpm-rpie | fpm-all"))
    iters: Optional[int] = PField(default=None, ge=1, le=5000,
                                  description="iterations (ptychography, default 100) or passes (FPM, default 30)")
    seed: int = 0
    crop: Optional[int] = PField(default=None, ge=32, description="centre-crop frames to N px (fast preview)")
    sparse_every: int = PField(default=1, ge=1, le=8, description="keep every k-th grid row/col (sparse-scan test)")
    engine_params: Dict[str, Any] = PField(default_factory=dict)
    sparse_fraction: Optional[float] = PField(default=None, gt=0.0, le=1.0,
                                              description="keep a random fraction of the frames (reduced-scan test)")
    refine_ml: bool = PField(default=False, description="apply the ML refinement model after reconstruction")
    model: Optional[str] = PField(default=None, description="model package id (GET /api/v1/models); default if omitted")
    hybrid_iters: int = PField(default=0, ge=0, le=1000,
                               description="hybrid mode: physics iterations seeded with the ML-refined object (0 = off)")
    fpm: Dict[str, Any] = PField(default_factory=dict, description=(
        "FPM settings that override the defaults stored with the dataset, e.g. {\"Np\": 256, "
        "\"z_source_mm\": 55, \"lambda_um\": 0.633, \"n_warmup\": 0, \"geometry_check\": \"report\"} "
        "(field names as in velos_recon.fpm.FPMConfig)"))
    probe_from_job: Optional[str] = PField(default=None, description=(
        "probe calibration: start from the probe reconstructed by this earlier (succeeded) job, e.g. a dense "
        "scan of the same illumination — needed for strongly reduced scans of some samples"))

    @field_validator("engine")
    @classmethod
    def _engine_ok(cls, v: str) -> str:
        from velos_recon.engine import ENGINE_NAMES
        from .fpm_jobs import FPM_ENGINES
        v = v.lower()
        if v not in ENGINE_NAMES and v not in FPM_ENGINES:
            raise ValueError(f"engine must be one of {sorted(ENGINE_NAMES) + sorted(FPM_ENGINES)}")
        return v

    @field_validator("fpm")
    @classmethod
    def _fpm_ok(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if v:
            from velos_recon.fpm import FPMConfig
            FPMConfig.from_dict(v)          # unknown keys -> 422
        return v


class JobOut(BaseModel):
    id: str
    dataset_id: str
    owner: str
    status: JobStatus
    params: Dict[str, Any]
    progress: float
    iteration: int
    message: str
    error: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    links: Dict[str, str] = PField(default_factory=dict)


class JobResultOut(BaseModel):
    job_id: str
    status: JobStatus
    summary: Dict[str, Any]
    artifacts: List[str]
    links: Dict[str, str]


class ContactIn(BaseModel):
    name: str = PField(min_length=1, max_length=120)
    email: str = PField(min_length=3, max_length=200)
    organisation: str = PField(default="", max_length=200)
    message: str = PField(min_length=1, max_length=4000)
    website: str = PField(default="", description="honeypot: must stay empty")

    @field_validator("email")
    @classmethod
    def _email_ok(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("invalid email")
        return v
