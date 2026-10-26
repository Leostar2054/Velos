"""
Runtime configuration. Everything comes from environment variables (or a
``.env`` file loaded by the process manager) so that no secret is ever stored
in the repository (Schedule A: configuration and secrets separated).

Variable                 Default                      Meaning
-----------------------  ---------------------------  -----------------------------------------
VELOS_STORAGE_DIR        ./storage                    Root for uploaded datasets and job outputs
VELOS_DB_URL             sqlite:///<storage>/velos.db SQLAlchemy URL (SQLite for M2; Postgres later)
VELOS_API_KEYS           dev-key:dev:admin            Comma list of key:name:role (role = admin|user)
VELOS_MAX_UPLOAD_MB      2048                         Upload size limit
VELOS_CORS_ORIGINS       http://localhost:3000        Comma list of allowed website origins
VELOS_ML_CHECKPOINT      (unset)                      TorchScript/PyTorch checkpoint for ML refinement
VELOS_WORKER_POLL_S      1.0                          Worker poll interval
VELOS_LOG_LEVEL          INFO
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ApiKeyInfo:
    name: str
    role: str  # "admin" | "user"


@dataclass
class Settings:
    storage_dir: Path
    db_url: str
    api_keys: Dict[str, ApiKeyInfo]
    max_upload_mb: int
    cors_origins: List[str]
    ml_checkpoint: str | None
    worker_poll_s: float
    log_level: str
    insecure_default_key: bool = field(default=False)

    @classmethod
    def from_env(cls) -> "Settings":
        storage = Path(os.getenv("VELOS_STORAGE_DIR", "./storage")).resolve()
        storage.mkdir(parents=True, exist_ok=True)
        db_url = os.getenv("VELOS_DB_URL") or f"sqlite:///{storage / 'velos.db'}"
        raw_keys = os.getenv("VELOS_API_KEYS")
        insecure = False
        if not raw_keys:
            raw_keys = "dev-key:dev:admin"
            insecure = True
        keys: Dict[str, ApiKeyInfo] = {}
        for item in raw_keys.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) < 2:
                raise ValueError("VELOS_API_KEYS entries must be key:name[:role]")
            key, name = parts[0], parts[1]
            role = parts[2] if len(parts) > 2 else "user"
            keys[key] = ApiKeyInfo(name=name, role=role)
        origins = [o.strip() for o in os.getenv("VELOS_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
        return cls(
            storage_dir=storage, db_url=db_url, api_keys=keys,
            max_upload_mb=int(os.getenv("VELOS_MAX_UPLOAD_MB", "2048")),
            cors_origins=origins, ml_checkpoint=os.getenv("VELOS_ML_CHECKPOINT") or None,
            worker_poll_s=float(os.getenv("VELOS_WORKER_POLL_S", "1.0")),
            log_level=os.getenv("VELOS_LOG_LEVEL", "INFO"), insecure_default_key=insecure,
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """Used by tests to re-read the environment."""
    global _settings
    _settings = None
