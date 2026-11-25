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
VELOS_MODEL_DIR          ./models                     Model packages (one sub-directory each, see velos_recon.refine)
VELOS_DEFAULT_MODEL      (first package found)        Package used when a job does not name one
VELOS_ML_RUNTIME         auto                         auto | onnxruntime | torchscript
VELOS_WORKER_POLL_S      1.0                          Worker poll interval
VELOS_LOG_LEVEL          INFO
VELOS_LOG_FORMAT         text                         text | json (one JSON object per line, for log collectors)
VELOS_METRICS_TOKEN      (unset)                      if set, /metrics requires "Authorization: Bearer <token>"
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
    model_dir: Path
    default_model: str | None
    ml_runtime: str
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
            cors_origins=origins,
            model_dir=Path(os.getenv("VELOS_MODEL_DIR", "./models")).resolve(),
            default_model=os.getenv("VELOS_DEFAULT_MODEL") or None,
            ml_runtime=os.getenv("VELOS_ML_RUNTIME", "auto"),
            worker_poll_s=float(os.getenv("VELOS_WORKER_POLL_S", "1.0")),
            log_level=os.getenv("VELOS_LOG_LEVEL", "INFO"), insecure_default_key=insecure,
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def configure_logging(level: str = "INFO") -> None:
    """Plain text (default) or JSON lines (VELOS_LOG_FORMAT=json) on stdout."""
    import json as _json
    import logging
    import time as _time

    class _JsonFormatter(logging.Formatter):
        def format(self, record):
            d = {"ts": _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
                 "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
            if record.exc_info:
                d["exc"] = self.formatException(record.exc_info)
            return _json.dumps(d, ensure_ascii=False)

    handler = logging.StreamHandler()
    if os.getenv("VELOS_LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def reset_settings() -> None:
    """Used by tests to re-read the environment."""
    global _settings
    _settings = None
