"""
Persistent storage for inputs and results.

M2 uses the local filesystem under ``VELOS_STORAGE_DIR`` (a mounted volume in
Docker). The interface is deliberately small so an S3/GCS backend can be added
at deployment time (M3/M4) without touching the API or the worker.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import BinaryIO, Iterable

from .config import get_settings

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    name = Path(name).name
    name = _SAFE.sub("_", name).strip("._") or "upload.bin"
    return name[:120]


class LocalStorage:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or get_settings().storage_dir)
        (self.root / "datasets").mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)

    # datasets --------------------------------------------------------------
    def dataset_dir(self, dataset_id: str) -> Path:
        return self.root / "datasets" / dataset_id

    def save_upload(self, dataset_id: str, filename: str, stream: BinaryIO,
                    max_bytes: int) -> tuple[Path, int]:
        d = self.dataset_dir(dataset_id)
        d.mkdir(parents=True, exist_ok=True)
        target = d / safe_filename(filename)
        size = 0
        with open(target, "wb") as f:
            while True:
                chunk = stream.read(8 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    f.close(); shutil.rmtree(d, ignore_errors=True)
                    raise ValueError(f"upload exceeds limit of {max_bytes} bytes")
                f.write(chunk)
        return target, size

    # jobs ------------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        d = self.root / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def job_artifacts(self, job_id: str) -> Iterable[Path]:
        d = self.root / "jobs" / job_id
        if not d.exists():
            return []
        return sorted(p for p in d.iterdir() if p.is_file())

    def artifact_path(self, job_id: str, name: str) -> Path:
        p = (self.root / "jobs" / job_id / safe_filename(name)).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("invalid artifact path")
        return p


def get_storage() -> LocalStorage:
    return LocalStorage()
