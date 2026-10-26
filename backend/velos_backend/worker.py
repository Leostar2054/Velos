"""
Reconstruction worker.

    python -m velos_backend.worker            # run forever (poll the queue)
    python -m velos_backend.worker --once     # process at most one job (tests / cron)

For each claimed job the worker loads the dataset, runs the Client's physics
engine through :func:`velos_recon.reconstruct` with progress reporting, optionally
applies the ML refinement model, writes artifacts into the job directory and
marks the job succeeded / failed / cancelled. The GPU inference worker of
M3/M4 is the same process with CUDA available.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

from velos_recon import ReconConfig, load_dataset, reconstruct
from velos_recon.engine import ReconCancelled
from velos_recon.viz import save_convergence, save_object_images, save_probe_image

from .config import get_settings
from .db import session_scope
from .models import Dataset, JobStatus
from .queue import claim_next, finish, is_cancel_requested, update_progress
from .storage import get_storage

log = logging.getLogger("velos.worker")


def _run_ml_refinement(result, out_dir: Path) -> dict:
    """Apply the ML refinement model if a checkpoint is configured (M3 integration hook)."""
    settings = get_settings()
    if not settings.ml_checkpoint:
        return {"applied": False, "reason": "VELOS_ML_CHECKPOINT not configured"}
    try:
        from velos_ml.infer import refine_object  # optional dependency (ml/ package)
    except Exception as e:  # pragma: no cover
        return {"applied": False, "reason": f"velos_ml unavailable: {e}"}
    t0 = time.perf_counter()
    refined = refine_object(result.obj, result.mask, settings.ml_checkpoint)
    files = save_object_images(refined, result.mask, out_dir, prefix="refined_",
                               dx_um=result.dx_m * 1e6, title="ML-refined")
    import numpy as np
    np.savez_compressed(out_dir / "refined.npz", obj=refined.astype(np.complex64))
    return {"applied": True, "checkpoint": settings.ml_checkpoint,
            "elapsed_s": round(time.perf_counter() - t0, 2), "files": files}


def process_job(job_id: str) -> JobStatus:
    storage = get_storage()
    with session_scope() as session:
        from .models import Job
        job = session.get(Job, job_id)
        params = json.loads(job.params_json or "{}")
        ds_row = session.get(Dataset, job.dataset_id)
        if ds_row is None:
            finish(session, job_id, JobStatus.failed, error="dataset not found")
            return JobStatus.failed
        out_dir = storage.job_dir(job_id)
        log.info("job %s: start (%s)", job_id, params)
        try:
            ds = load_dataset(ds_row.path, name=ds_row.name)
            if int(params.get("sparse_every", 1)) > 1:
                ds = ds.sparse_subset(every=int(params["sparse_every"]))
            cfg = ReconConfig(engine=params.get("engine", "wasp"), iters=int(params.get("iters", 100)),
                              seed=int(params.get("seed", 0)), crop=params.get("crop"),
                              engine_params=params.get("engine_params") or {},
                              progress_every=max(1, int(params.get("iters", 100)) // 50))
            last = {"t": 0.0}

            def progress(it, n, info):
                now = time.perf_counter()
                if now - last["t"] > 0.5 or it == n:       # throttle DB writes
                    update_progress(session, job_id, it, n, f"iteration {it}/{n} · err {info['err']:.3e}")
                    last["t"] = now

            def cancelled():
                return is_cancel_requested(session, job_id)

            res = reconstruct(ds, cfg, progress=progress, should_cancel=cancelled)
            res.to_npz(out_dir / "result.npz")
            files = save_object_images(res.obj, res.mask, out_dir, dx_um=res.dx_m * 1e6,
                                       title=f"{ds.name} · {cfg.engine.upper()} · {cfg.iters} it")
            files["probe"] = save_probe_image(res.probe, out_dir)
            files["convergence"] = save_convergence(res.history, out_dir)
            summary = {"dataset": res.dataset_summary, "reconstruction": res.summary(),
                       "files": {k: Path(v).name for k, v in files.items()}}
            if params.get("refine_ml"):
                summary["ml_refinement"] = _run_ml_refinement(res, out_dir)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            finish(session, job_id, JobStatus.succeeded, result=summary)
            log.info("job %s: succeeded in %.1fs", job_id, res.elapsed_s)
            return JobStatus.succeeded
        except ReconCancelled as e:
            finish(session, job_id, JobStatus.cancelled, error=str(e))
            log.info("job %s: cancelled", job_id)
            return JobStatus.cancelled
        except Exception as e:  # noqa: BLE001 - report any failure to the client
            tb = traceback.format_exc()
            (out_dir / "error.log").write_text(tb)
            finish(session, job_id, JobStatus.failed, error=f"{type(e).__name__}: {e}")
            log.exception("job %s: failed", job_id)
            return JobStatus.failed


def run_once(worker_id: str | None = None) -> str | None:
    with session_scope() as session:
        job = claim_next(session, worker_id)
        if job is None:
            return None
        job_id = job.id
    process_job(job_id)
    return job_id


def main(argv=None):
    p = argparse.ArgumentParser(prog="velos-worker")
    p.add_argument("--once", action="store_true", help="process at most one job and exit")
    p.add_argument("--poll", type=float, default=None, help="poll interval (s)")
    p.add_argument("--worker-id", default=None)
    a = p.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    poll = a.poll or settings.worker_poll_s
    log.info("worker started (storage=%s, poll=%.1fs)", settings.storage_dir, poll)
    while True:
        job_id = run_once(a.worker_id)
        if a.once:
            return 0 if job_id else 1
        if job_id is None:
            time.sleep(poll)


if __name__ == "__main__":
    raise SystemExit(main())
