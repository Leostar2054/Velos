"""
Real-sample FPM training / test pairs (M4) from the in-house LED-array microscope.

Same idea as :mod:`velos_ml.realdata` (M3, scanning ptychography): there is no ground truth for
experimental data, so the reference is the reconstruction from ALL usable LEDs ("dense"), and the
inputs are reconstructions from reduced LED sets — every LED inside 1.1 × the objective NA plus a
random fraction of the dark-field LEDs, i.e. what a faster acquisition would record:

    python -m velos_ml.fpm_realdata --source <..._c512_meta.mat> --name blood_222619 \
        --out data/fpm/blood_222619 --fractions 0.30 0.50 --draws 4

Steps per dataset
1. geometry calibration on the full LED set (geometry check, "apply": LED distance and
   optical-axis offset that fit the frames best) — then fixed for every reduced set;
2. dense reference: EPRY on all usable LEDs;
3. reduced sets: EPRY on each LED subset, same grid (upsample of the dense run), same settings;
4. every reconstruction is stored raw and "flattened" (point-source curvature, tilt and mean
   phase removed with a refined fit), plus its pupil and its data errors on the frames used and
   — for reduced sets — the prediction error on the frames NOT used.

Output layout (``dense.npz``, ``subsets/sub_XX.npz``, ``meta.json``) is the one read by
:class:`velos_ml.dataset.RealPatchDataset`, so the M3 training code is reused unchanged.
Draw numbering: for each fraction f_i, draws d = 0..n-1 give subset index k = d·n_f + i; draws
0 (k = 0, 1) are the TEST draws and never used for training.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from velos_recon import fpm


def _border_mask(n: int, border: int) -> np.ndarray:
    m = np.zeros((n, n), bool)
    m[border:n - border, border:n - border] = True
    return m


def calibrate(acq: fpm.FPMAcquisition, base: fpm.FPMConfig, log=print) -> fpm.FPMConfig:
    prep = fpm.prepare(acq, replace(base, geometry_check="apply"), log=log)
    return replace(prep.cfg, geometry_check="off")


def run(source: str, name: str, out: str, lambda_um: float = 0.633, z_mm: float = 55.0, Np: int = 512,
        passes: int = 30, warmup: int = 2, algo: str = "EPRY", fractions=(0.30, 0.50), draws: int = 4,
        seed: int = 2026, calibrate_geometry: bool = True, border: int = 24, config_from: str | None = None,
        log=print) -> dict:
    out = Path(out); (out / "subsets").mkdir(parents=True, exist_ok=True)
    acq = fpm.load_fpm(source, name=name)
    # the patch is the stored region itself (packed crops: centre crop or side regions of the sensor);
    # for the centre crop this equals the default image-centre patch
    h, w = acq.frames.shape[1:]
    centre = {"crop_center_x_px": acq.crop_col0 + (w + 1) / 2, "crop_center_y_px": acq.crop_row0 + (h + 1) / 2}
    base = fpm.FPMConfig(lambda_um=lambda_um, objective_NA=0.10, magnification=4.0, z_source_mm=z_mm, Np=Np,
                         n_warmup=warmup, n_passes=passes, algorithms=[algo], geometry_check="off", **centre)
    meta_file = out / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    if "config" in meta:
        cfg = fpm.FPMConfig.from_dict(meta["config"])
        log(f"[{name}] calibrated configuration reused")
    elif config_from:
        # another region of the same acquisition: the set-up geometry (LED distance, optical axis,
        # orientation) is a property of the microscope, so the calibration of the centre region is reused
        ref = fpm.FPMConfig.from_dict(json.loads(Path(config_from).read_text())["config"])
        cfg = replace(ref, Np=Np, n_warmup=warmup, n_passes=passes, algorithms=[algo], geometry_check="off",
                      upsample=None, **centre)
        meta.update({"name": name, "source": str(source), "config": cfg.to_dict(), "calibration_s": 0.0,
                     "config_from": str(config_from), "acquisition": acq.summary()})
        meta_file.write_text(json.dumps(meta, indent=2, default=float))
        log(f"[{name}] geometry from {config_from}: z {cfg.z_source_mm:.2f} mm, axis ({cfg.optical_axis_x_mm:+.2f}, "
            f"{cfg.optical_axis_y_mm:+.2f}) mm; patch centre ({centre['crop_center_x_px']:.1f}, {centre['crop_center_y_px']:.1f}) px")
    else:
        t = time.time()
        cfg = calibrate(acq, base, log=log) if calibrate_geometry else base
        meta.update({"name": name, "source": str(source), "config": cfg.to_dict(),
                     "calibration_s": round(time.time() - t, 1), "acquisition": acq.summary()})
        meta_file.write_text(json.dumps(meta, indent=2, default=float))
        log(f"[{name}] geometry: z {cfg.z_source_mm:.2f} mm, axis ({cfg.optical_axis_x_mm:+.2f}, "
            f"{cfg.optical_axis_y_mm:+.2f}) mm, swap {cfg.swap_led_xy}, y sign {cfg.led_y_sign:+d}")
    dense_file = out / "dense.npz"
    prep_full = fpm.prepare(acq, cfg)
    if not dense_file.exists():
        res = fpm.reconstruct_fpm(acq, cfg, prepared=prep_full)
        a = res.algorithms[0]
        fp = fpm.flatten_params(a.obj, res.pixel_um, res.curvature_rad_um2, refine=True)
        flat = fpm.apply_flatten(a.obj, fp)
        n = flat.shape[0]
        np.savez_compressed(dense_file, obj=flat.astype(np.complex64), obj_raw=a.obj.astype(np.complex64),
                            pupil=a.pupil.astype(np.complex64), mask=_border_mask(n, border * 3),
                            residual=a.residual, factor=a.intensity_factor, flatten=json.dumps(fp),
                            pixel_um=res.pixel_um, leds=np.array(prep_full.frames_table["led_index"]),
                            data_error=float(a.residual[-1]), elapsed_s=a.time_s)
        meta["dense"] = {"leds": int(prep_full.A.shape[0]), "residual": float(a.residual[-1]),
                         "elapsed_s": round(a.time_s, 1), "N_obj": int(res.prepared.geometry.N_obj),
                         "pixel_um": res.pixel_um, "synthetic_NA": res.prepared.geometry.synthetic_NA}
        meta_file.write_text(json.dumps(meta, indent=2, default=float))
        log(f"[{name}] dense: {prep_full.A.shape[0]} LEDs, residual {a.residual[-1]:.4f}, {a.time_s:.0f} s")
    up = prep_full.geometry.N_obj // Np
    cfg_sub = replace(cfg, upsample=up)
    meta.setdefault("subsets", {})
    nf = len(fractions)
    for d in range(draws):
        for i, f in enumerate(fractions):
            k = d * nf + i
            fn = out / "subsets" / f"sub_{k:02d}.npz"
            if fn.exists():
                continue
            keep = fpm.reduced_led_set(acq, cfg_sub, f, seed + 1000 * d + i)
            sub = acq.subset(keep)
            prep = fpm.prepare(sub, cfg_sub)
            res = fpm.reconstruct_fpm(sub, cfg_sub, prepared=prep)
            a = res.algorithms[0]
            used = {int(x) + 1 for x in keep}
            unused = [m for m, li in enumerate(prep_full.frames_table["led_index"]) if li not in used]
            pred = fpm.fpm_data_error(a.obj, a.pupil, prep_full, frames=unused) if unused else None
            fp = fpm.flatten_params(a.obj, res.pixel_um, res.curvature_rad_um2, refine=True)
            flat = fpm.apply_flatten(a.obj, fp)
            n = flat.shape[0]
            np.savez_compressed(fn, obj=flat.astype(np.complex64), obj_raw=a.obj.astype(np.complex64),
                                pupil=a.pupil.astype(np.complex64), mask=_border_mask(n, border * 3),
                                keep=keep, fraction=f, draw=d, residual=a.residual, factor=a.intensity_factor,
                                flatten=json.dumps(fp), data_error_used=float(a.residual[-1]),
                                prediction_error=np.nan if pred is None else pred, elapsed_s=a.time_s,
                                leds_used=int(prep.A.shape[0]))
            meta["subsets"][f"{k:02d}"] = {"fraction_darkfield": f, "draw": d, "leds_used": int(prep.A.shape[0]),
                                           "residual": float(a.residual[-1]), "prediction_error": pred,
                                           "elapsed_s": round(a.time_s, 1), "test_draw": d == 0}
            meta_file.write_text(json.dumps(meta, indent=2, default=float))
            log(f"[{name}] sub_{k:02d}: fraction {f:.2f} draw {d}: {prep.A.shape[0]} LEDs, residual "
                f"{a.residual[-1]:.4f}, prediction error {pred if pred is None else round(pred, 4)}, {a.time_s:.0f} s")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.633)
    ap.add_argument("--z", type=float, default=55.0)
    ap.add_argument("--Np", type=int, default=512)
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.30, 0.50])
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--config-from", default=None,
                    help="meta.json of another region of the same acquisition: reuse its calibrated geometry")
    a = ap.parse_args(argv)
    run(a.source, a.name, a.out, a.lam, a.z, a.Np, a.passes, a.warmup, "EPRY", tuple(a.fractions), a.draws,
        calibrate_geometry=not a.no_calibration, config_from=a.config_from, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
