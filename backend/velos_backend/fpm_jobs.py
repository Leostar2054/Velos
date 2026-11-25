"""
FPM jobs (M4): LED-array Fourier ptychography on the in-house microscope.

Called by the worker for datasets whose summary says ``modality = fpm`` (a Velos FPM ``.npz``,
see :func:`velos_recon.fpm.save_fpm_npz`). Steps:

1. optional reduced-LED test (``sparse_fraction``): all bright-field LEDs are kept, a random
   fraction of the dark-field LEDs is kept — what a faster acquisition would record;
2. AP / EPRY / rPIE reconstruction of the requested patch (Python port of the MATLAB engine);
3. artifacts: amplitude / phase PNGs (curvature and tilt removed), the MATLAB-style TIFFs,
   comparison figure, frames.csv, summary.txt, result.npz;
4. data consistency: residual against the frames used and — for a reduced set — the
   prediction error on the frames that were NOT used;
5. optional ML refinement with an FPM model package (``modality = fpm`` in its card) and the
   hybrid physics polish (EPRY seeded with the refined object and the recovered pupil).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from velos_recon import fpm

log = logging.getLogger("velos.worker.fpm")

FPM_ENGINES = {"fpm": ["EPRY"], "fpm-epry": ["EPRY"], "fpm-ap": ["AP"], "fpm-rpie": ["rPIE"],
               "fpm-all": ["AP", "EPRY", "rPIE"]}
DEFAULT_PASSES = 30


def job_config(acq: fpm.FPMAcquisition, params: dict) -> fpm.FPMConfig:
    d = dict(acq.defaults or {})
    d.update(params.get("fpm") or {})
    d.setdefault("geometry_check", "off")          # the check costs ~1 min; ask for it explicitly
    d["algorithms"] = FPM_ENGINES[params.get("engine", "fpm")]
    if params.get("iters"):
        d["n_passes"] = int(params["iters"])
    else:
        d.setdefault("n_passes", DEFAULT_PASSES)
    return fpm.FPMConfig.from_dict(d)


reduced_led_set = fpm.reduced_led_set   # kept here for backwards compatibility


def _prediction_error(obj, pupil, prep_full: fpm.Prepared, used_led_index: set) -> float | None:
    unused = [m for m, li in enumerate(prep_full.frames_table["led_index"]) if li not in used_led_index]
    if not unused:
        return None
    return fpm.fpm_data_error(obj, pupil, prep_full, frames=unused)


def run_fpm_job(dataset_path: str, params: dict, out_dir: Path, progress=None, should_cancel=None,
                resolve_model=None, ml_runtime: str = "auto") -> dict:
    from velos_recon.viz import save_comparison, save_convergence, save_object_images
    acq = fpm.load_fpm_npz(dataset_path)
    cfg = job_config(acq, params)
    t0 = time.perf_counter()
    prep_full = None
    used = None
    if params.get("sparse_fraction") and float(params["sparse_fraction"]) < 1.0:
        # the full set fixes the reconstruction grid and lets us score the frames left out
        prep_full = fpm.prepare(acq, replace(cfg, geometry_check="off"))
        cfg = replace(cfg, upsample=prep_full.geometry.N_obj // cfg.Np)
        keep = reduced_led_set(acq, cfg, float(params["sparse_fraction"]), int(params.get("seed", 0)))
        acq_run = acq.subset(keep, name_suffix=f"_leds{keep.size}")
        used = {int(i) + 1 for i in keep}
    else:
        acq_run = acq
    logs: list[str] = []
    res = fpm.reconstruct_fpm(acq_run, cfg, log=logs.append, progress=progress, should_cancel=should_cancel)
    res.prepared.patch["dataset"] = acq.name
    files = fpm.save_fpm_results(res, out_dir)
    primary = next((a for a in res.algorithms if a.name.upper() == "EPRY"), res.algorithms[0])
    fp = fpm.flatten_params(primary.obj, res.pixel_um, res.curvature_rad_um2)
    flat = fpm.apply_flatten(primary.obj, fp)
    mask = np.ones(flat.shape, bool)
    title = f"{acq.name} · FPM {primary.name} · {res.prepared.A.shape[0]} LEDs · {cfg.n_passes} passes"
    imgs = save_object_images(flat, mask, out_dir, dx_um=res.pixel_um, title=title)
    files.update({k: Path(v).name for k, v in imgs.items()})
    files["convergence"] = Path(save_convergence({"err": primary.residual}, out_dir)).name
    (out_dir / "fpm_log.txt").write_text("\n".join(logs) + "\n")
    files["log"] = "fpm_log.txt"
    rsum = res.summary()
    rsum["primary"] = primary.name
    rsum["flatten"] = fp
    rsum["data_error_normalised"] = fpm.fpm_data_error(primary.obj, primary.pupil, res.prepared, primary.intensity_factor)
    if prep_full is not None:
        rsum["reduced_leds"] = {"fraction_darkfield_kept": float(params["sparse_fraction"]),
                                "leds_used": int(res.prepared.A.shape[0]), "leds_available": int(prep_full.A.shape[0])}
        rsum["prediction_error_unused_frames"] = _prediction_error(primary.obj, primary.pupil, prep_full, used)
    summary = {"dataset": acq.summary(), "reconstruction": rsum, "files": files,
               "elapsed_s": round(time.perf_counter() - t0, 2)}
    if params.get("refine_ml"):
        summary["ml_refinement"] = _fpm_refinement(res, primary, fp, flat, params, out_dir, prep_full, used,
                                                   resolve_model, ml_runtime, progress, should_cancel)
    return summary


def _fpm_refinement(res, primary, fp, flat, params, out_dir: Path, prep_full, used, resolve_model, ml_runtime,
                    progress=None, should_cancel=None) -> dict:
    from velos_recon.refine import load_package, refine
    from velos_recon.viz import save_comparison, save_object_images
    path = resolve_model(params.get("model"), "fpm") if resolve_model else None
    if path is None:
        return {"applied": False, "reason": "no FPM model package installed"}
    pkg = load_package(str(path), ml_runtime)
    mask = np.ones(flat.shape, bool)
    ref = refine(flat, mask, pkg)                         # model works in the flattened frame
    refined_obj = fpm.undo_flatten(ref.obj, fp)
    np.savez_compressed(out_dir / "refined.npz", obj=refined_obj.astype(np.complex64),
                        obj_flat=ref.obj.astype(np.complex64))
    files = save_object_images(ref.obj, mask, out_dir, prefix="refined_", dx_um=res.pixel_um,
                               title=f"ML-refined · {pkg.card.get('name')} {pkg.card.get('version')}")
    prep = res.prepared
    e_phys = fpm.fpm_data_error(primary.obj, primary.pupil, prep)
    e_ml = fpm.fpm_data_error(refined_obj, primary.pupil, prep)
    out = {"applied": True, "model": Path(path).name, "model_name": pkg.card.get("name"),
           "model_version": pkg.card.get("version"), "runtime": ref.runtime, "elapsed_s": round(ref.elapsed_s, 3),
           "data_error": {"physics": e_phys, "ml": e_ml},
           "files": {k: Path(v).name for k, v in files.items()} | {"refined_npz": "refined.npz"}}
    if prep_full is not None:
        out["prediction_error_unused_frames"] = {"physics": _prediction_error(primary.obj, primary.pupil, prep_full, used),
                                                 "ml": _prediction_error(refined_obj, primary.pupil, prep_full, used)}
    n_h = int(params.get("hybrid_iters", 0) or 0)
    panels = [("Physics", flat), ("ML-refined", ref.obj)]
    final_err = e_ml
    if n_h > 0:
        h = fpm.hybrid_fpm(prep, refined_obj, primary.pupil, n_h, "EPRY", should_cancel=should_cancel)
        hz = fpm.apply_flatten(h.obj, fp)
        np.savez_compressed(out_dir / "hybrid.npz", obj=h.obj.astype(np.complex64), pupil=h.pupil.astype(np.complex64))
        hf = save_object_images(hz, mask, out_dir, prefix="hybrid_", dx_um=res.pixel_um,
                                title=f"Hybrid · ML + {n_h} EPRY passes")
        e_h = fpm.fpm_data_error(h.obj, h.pupil, prep)
        out["hybrid"] = {"passes": n_h, "elapsed_s": round(h.time_s, 2),
                         "files": {k: Path(v).name for k, v in hf.items()} | {"hybrid_npz": "hybrid.npz"}}
        out["data_error"]["hybrid"] = e_h
        if prep_full is not None:
            out["prediction_error_unused_frames"]["hybrid"] = _prediction_error(h.obj, h.pupil, prep_full, used)
        panels.append((f"Hybrid (+{n_h})", hz))
        final_err = e_h
    warn = float(pkg.card.get("data_error_ratio_warn", 2.0))
    floor = float(pkg.card.get("data_error_abs_floor", 0.02))
    ratio = final_err / max(e_phys, 1e-30)
    out["data_error_ratio"] = ratio
    out["consistency"] = "ok" if (ratio <= warn or final_err <= floor) else (
        "warning: the refined object fits the measured frames much worse than the physics result "
        f"(error {final_err:.3f} = {ratio:.1f}× physics); prefer the physics result")
    try:
        out["files"]["comparison_ml"] = Path(save_comparison(panels, mask, out_dir / "comparison_ml.png",
                                                             res.pixel_um)).name
    except Exception as e:  # pragma: no cover
        log.warning("comparison figure failed: %s", e)
    return out
