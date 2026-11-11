"""
Reconstruction worker.

    python -m velos_backend.worker            # run forever (poll the queue)
    python -m velos_backend.worker --once     # process at most one job (tests / cron)

For each claimed job the worker loads the dataset, runs the Client's physics
engine through :func:`velos_recon.reconstruct` with progress reporting, optionally
applies the packaged ML refinement model (ONNX Runtime / TorchScript) and the
optional hybrid physics polish, writes artifacts into the job directory and
marks the job succeeded / failed / cancelled. The GPU inference worker (M4) is
the same process with onnxruntime-gpu / CUDA available.
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


def resolve_model(name: str | None):
    """Path of the requested (or default) model package, or None if none is installed."""
    settings = get_settings()
    from velos_recon.refine import list_packages
    cards = list_packages(settings.model_dir)
    if not cards:
        return None
    wanted = name or settings.default_model
    if wanted is None:
        return settings.model_dir / cards[0]["id"]
    if not (settings.model_dir / wanted / "model_card.json").exists():
        raise ValueError(f"unknown model '{wanted}' (installed: {[c['id'] for c in cards]})")
    return settings.model_dir / wanted


def _run_ml_refinement(ds, cfg, res, params: dict, out_dir: Path, progress=None, should_cancel=None) -> dict:
    """M3: apply the packaged refinement model (ONNX/TorchScript runtime) and, optionally,
    the hybrid physics polish. Every estimate is scored against the measured data."""
    import numpy as np
    from velos_recon.engine import data_error
    from velos_recon.refine import hybrid, load_package, refine

    settings = get_settings()
    path = resolve_model(params.get("model"))
    if path is None:
        return {"applied": False, "reason": f"no model package installed in {settings.model_dir}"}
    pkg = load_package(str(path), settings.ml_runtime)
    ref = refine(res.obj, res.mask, pkg)
    np.savez_compressed(out_dir / "refined.npz", obj=ref.obj.astype(np.complex64),
                        obj_normalised=ref.z.astype(np.complex64), **{f"norm_{k}": v for k, v in ref.params.to_dict().items()})
    dx_um = res.dx_m * 1e6
    files = save_object_images(ref.z, res.mask, out_dir, prefix="refined_", dx_um=dx_um,
                               title=f"ML-refined · {pkg.card.get('name')} {pkg.card.get('version')}")
    e_phys = data_error(res.obj, res.probe, ds, crop=cfg.crop)
    e_ml = data_error(ref.obj, res.probe, ds, crop=cfg.crop)
    out = {"applied": True, "model": path.name, "model_name": pkg.card.get("name"),
           "model_version": pkg.card.get("version"), "runtime": ref.runtime,
           "elapsed_s": round(ref.elapsed_s, 3),
           "normalisation": ref.params.to_dict(),
           "data_error": {"physics": e_phys, "ml": e_ml},
           "files": {k: Path(v).name for k, v in files.items()} | {"refined_npz": "refined.npz"}}
    warn = float(pkg.card.get("data_error_ratio_warn", 2.0))
    floor = float(pkg.card.get("data_error_abs_floor", 0.02))
    n_h = int(params.get("hybrid_iters", 0) or 0)
    if n_h > 0:
        hres = hybrid(ds, res, ref.obj, cfg, n_h, progress=progress, should_cancel=should_cancel)
        e_h = data_error(hres.obj, hres.probe, ds, crop=cfg.crop)
        from velos_recon.normalise import normalise
        zh, _ = normalise(hres.obj, hres.mask)
        hres.to_npz(out_dir / "hybrid.npz")
        hf = save_object_images(zh, hres.mask, out_dir, prefix="hybrid_", dx_um=dx_um,
                                title=f"Hybrid · ML + {n_h} physics it")
        out["hybrid"] = {"iters": n_h, "elapsed_s": round(hres.elapsed_s, 2),
                         "files": {k: Path(v).name for k, v in hf.items()} | {"hybrid_npz": "hybrid.npz"}}
        out["data_error"]["hybrid"] = e_h
        final_err = e_h
    else:
        final_err = e_ml
    final_ratio = final_err / max(e_phys, 1e-30)
    out["data_error_ratio"] = final_ratio
    # flag only a gross misfit: above `warn` × the physics error AND above an absolute floor (on high-SNR
    # data the physics fit is near perfect, so a ratio alone would flag harmless differences)
    out["consistency"] = "ok" if (final_ratio <= warn or final_err <= floor) else (
        "warning: the refined object fits the measured data much worse than the physics result "
        f"(error {final_err:.3f} = {final_ratio:.1f}× physics); prefer the physics result")
    try:
        from velos_recon.viz import save_comparison
        panels = [("Physics", np.where(res.mask, ref.z_in, 0)), ("ML-refined", ref.z)]
        if n_h > 0:
            panels.append((f"Hybrid (+{n_h} it)", zh))
        out["files"]["comparison"] = Path(save_comparison(panels, res.mask, out_dir / "comparison.png", dx_um)).name
    except Exception as e:  # pragma: no cover - figure only
        log.warning("comparison figure failed: %s", e)
    return out


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
            if params.get("sparse_fraction") and float(params["sparse_fraction"]) < 1.0:
                ds = ds.sparse_subset(fraction=float(params["sparse_fraction"]), seed=int(params.get("seed", 0)))
            cfg = ReconConfig(engine=params.get("engine", "epie"), iters=int(params.get("iters", 100)),
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

            init_probe = None
            if params.get("probe_from_job"):                 # probe calibration from an earlier job
                import numpy as np
                ref_file = storage.job_dir(params["probe_from_job"]) / "result.npz"
                init_probe = np.load(ref_file)["probe"]
                m = int(cfg.crop) if cfg.crop else min(ds.frame_shape)
                if init_probe.shape != (m, m):
                    raise ValueError(f"probe_from_job: probe {init_probe.shape} does not match frames ({m}, {m}); "
                                     "use the same crop as the calibration job")
            res = reconstruct(ds, cfg, progress=progress, should_cancel=cancelled, init_probe=init_probe)
            res.to_npz(out_dir / "result.npz")
            files = save_object_images(res.obj, res.mask, out_dir, dx_um=res.dx_m * 1e6,
                                       title=f"{ds.name} · {cfg.engine.upper()} · {cfg.iters} it")
            files["probe"] = save_probe_image(res.probe, out_dir)
            files["convergence"] = save_convergence(res.history, out_dir)
            from velos_recon.engine import data_error
            rsum = res.summary()
            # normalised data-consistency error Σ(|F(P·O)| − √I)² / Σ I (comparable across datasets and
            # with the ML / hybrid errors); final_data_error above is the engine's raw sum
            rsum["data_error_normalised"] = data_error(res.obj, res.probe, ds, crop=cfg.crop)
            if init_probe is not None:
                rsum["probe_from_job"] = params["probe_from_job"]
            summary = {"dataset": res.dataset_summary, "reconstruction": rsum,
                       "files": {k: Path(v).name for k, v in files.items()}}
            if params.get("refine_ml"):
                update_progress(session, job_id, cfg.iters, cfg.iters, "ML refinement")
                summary["ml_refinement"] = _run_ml_refinement(ds, cfg, res, params, out_dir,
                                                              should_cancel=cancelled)
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
