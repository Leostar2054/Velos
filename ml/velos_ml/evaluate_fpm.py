"""
M4 evaluation on REAL samples (blood smear, leaf section) from the in-house LED-array FPM.

    python -m velos_ml.evaluate_fpm --data data/fpm/blood_222619:holdout data/fpm/leaf_214040_L:full ... \
        --model fpm_v1=runs/refine_fpm_v1/best.pt --model v2=runs/refine_v2/best.pt \
        --hybrid-model fpm_v1 --hybrid 5 10 --out results/m4

Every test case is a reduced LED set (test draws only: sub_00, sub_01) reconstructed with EPRY;
the reference is the reconstruction from ALL usable LEDs of the same acquisition.

Methods: ``physics`` (EPRY on the reduced set — the input of every model), ``physics+k`` (k more EPRY
passes from the physics result: the control for the hybrid mode), each ``<model>`` applied to the
flattened physics result exactly as the backend does (normalise → tiles → model), and ``hybrid@k``
(the chosen model's output seeding k further EPRY passes on the reduced set).

Regions: ``holdout`` = the right 35 % of the high-resolution field (the models were trained on the
left 65 % only); ``full`` = an acquisition region never used for training (whole field).

Metrics: registered NRMSE / phase and amplitude PSNR & SSIM against the reference (global complex
scalar, phase ramp and sub-pixel shift removed), the data residual on the LEDs used, and the
prediction error on the LEDs that were NOT used (needs no reference at all).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from velos_recon import fpm
from velos_recon.metrics import amplitude_phase_metrics

from .evaluate_m3 import METRICS, TorchTiled, aggregate

SPLIT = 0.65


def _metrics(est, ref, mask):
    # no sub-pixel registration: reconstructions of one acquisition share the grid (offsets <= 0.1 px);
    # the global complex scalar and phase ramp are still removed
    m = amplitude_phase_metrics(np.asarray(est, np.complex128), np.asarray(ref, np.complex128), mask, register=False)
    return {k: float(m[k]) for k in METRICS}


def _apply_model(mdl, S, mask, c_split: int):
    """Model output (flattened frame). In hold-out mode the model only ever sees hold-out pixels."""
    if c_split:
        sub = np.ascontiguousarray(S[:, c_split:]); msub = mask[:, c_split:]
        out, _, _ = mdl.refine_full(sub, msub)
        full = S.copy(); full[:, c_split:] = out
        return full
    out, _, _ = mdl.refine_full(S, mask)
    return out


def eval_dataset(root: Path, mode: str, models: dict, hybrid_model: str | None, hybrid_passes, subsets=(0, 1)):
    meta = json.loads((root / "meta.json").read_text())
    acq = fpm.load_fpm(meta["source"], name=meta["name"])
    cfg = fpm.FPMConfig.from_dict(meta["config"])
    prep_full = fpm.prepare(acq, cfg)
    dense = np.load(root / "dense.npz")
    D = dense["obj"].astype(np.complex128)
    H, W = D.shape
    border = dense["mask"].astype(bool)
    c_split = int(SPLIT * W) if mode == "holdout" else 0
    region = border.copy()
    if c_split:
        region[:, :c_split] = False
    rows, examples = [], []
    for k in subsets:
        f = root / "subsets" / f"sub_{k:02d}.npz"
        if not f.exists():
            continue
        s = np.load(f)
        S = s["obj"].astype(np.complex128)
        raw = s["obj_raw"].astype(np.complex128)
        P = s["pupil"].astype(np.complex128)
        fp = json.loads(str(s["flatten"]))
        keep = s["keep"]
        up = prep_full.geometry.N_obj // cfg.Np
        sub = acq.subset(keep)
        prep_sub = fpm.prepare(sub, replace(cfg, upsample=up))
        used = {int(x) + 1 for x in keep}
        unused = [m for m, li in enumerate(prep_full.frames_table["led_index"]) if li not in used]
        est_flat = {"physics": S}
        est_raw = {"physics": raw}
        pup = {"physics": P}
        for name, mdl in models.items():
            o = _apply_model(mdl, S, border, c_split)
            est_flat[name] = o
            est_raw[name] = fpm.undo_flatten(o, fp)
            pup[name] = P
        for n_h in hybrid_passes:     # control: the same number of extra EPRY passes without a model
            h = fpm.hybrid_fpm(prep_sub, raw, P, n_h, "EPRY")
            est_raw[f"physics+{n_h}"] = h.obj
            est_flat[f"physics+{n_h}"] = fpm.apply_flatten(h.obj, fp)
            pup[f"physics+{n_h}"] = h.pupil
        if hybrid_model:
            for n_h in hybrid_passes:
                h = fpm.hybrid_fpm(prep_sub, est_raw[hybrid_model], P, n_h, "EPRY")
                est_raw[f"hybrid@{n_h}"] = h.obj
                est_flat[f"hybrid@{n_h}"] = fpm.apply_flatten(h.obj, fp)
                pup[f"hybrid@{n_h}"] = h.pupil
        for name in est_flat:
            r = {"case": f"{root.name}/sub_{k:02d}", "dataset": root.name, "mode": mode,
                 "fraction": float(s["fraction"]), "leds": int(s["leds_used"]), "method": name,
                 **_metrics(est_flat[name], D, region),
                 "data_error_used": fpm.fpm_data_error(est_raw[name], pup[name], prep_sub),
                 "data_error_predict": fpm.fpm_data_error(est_raw[name], pup[name], prep_full, frames=unused)
                 if unused else float("nan"), "n_predict_leds": len(unused)}
            rows.append(r)
        examples.append({"case": f"{root.name}/sub_{k:02d}", "mask": region, "c_split": c_split,
                         "dense": D, "leds": int(s["leds_used"]), "leds_full": int(prep_full.A.shape[0]),
                         **{m: v for m, v in est_flat.items()}})
        print(f"  {root.name} sub_{k:02d} ({int(s['leds_used'])} LEDs): " +
              " · ".join(f"{r['method']} {r['psnr_phase']:.2f}dB/{r['data_error_predict']:.4f}"
                         for r in rows[-len(est_flat):]), flush=True)
    return rows, examples


def repeatability(roots) -> list:
    """Reference noise floor: dense reconstructions of repeated acquisitions of the same field."""
    out = []
    for a, b in roots:
        A = np.load(Path(a) / "dense.npz"); B = np.load(Path(b) / "dense.npz")
        m = A["mask"].astype(bool)
        W = m.shape[1]
        reg = m.copy(); reg[:, :int(SPLIT * W)] = False
        out.append({"pair": f"{Path(a).name} vs {Path(b).name}",
                    **_metrics(B["obj"].astype(np.complex128), A["obj"].astype(np.complex128), reg)})
    return out


def save_example_figure(ex: dict, methods, path, crop: int = 360):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from velos_recon.metrics import align
    D = ex["dense"]; c0 = ex["c_split"]
    H, W = D.shape
    r0 = (H - crop) // 2
    cs = c0 + max(0, ((W - c0) - crop) // 2) if c0 else (W - crop) // 2
    win = (slice(r0, r0 + crop), slice(cs, cs + crop))
    m = np.zeros_like(ex["mask"]); m[win] = True
    cols = methods + ["dense"]
    fig, axs = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6.6), squeeze=False)
    ref = D[win]
    for j, name in enumerate(cols):
        z = ex[name]
        if name != "dense":       # same grid as the reference (no registration, as in the metrics)
            z = align(z, D, m, ramp=True)
        z = z[win]
        ph = np.angle(z * np.exp(-1j * np.angle(np.mean(ref))))
        lo, hi = np.percentile(np.angle(ref * np.exp(-1j * np.angle(np.mean(ref)))), [1, 99])
        axs[0, j].imshow(np.abs(z), cmap="gray", vmin=np.percentile(np.abs(ref), 1), vmax=np.percentile(np.abs(ref), 99))
        axs[1, j].imshow(ph, cmap="twilight", vmin=lo, vmax=hi)
        title = {"dense": f"all {ex['leds_full']} LEDs (reference)", "physics": f"physics, {ex['leds']} LEDs",
                 "v2": "M3 model (no FPM training)", "fpm_v1": "FPM model (AI only)"}.get(name, name)
        if name.startswith("physics+"):
            title = f"physics + {name.split('+')[1]} passes"
        elif name.startswith("hybrid@"):
            title = f"hybrid: AI + {name.split('@')[1]} passes"
        axs[0, j].set_title(title, fontsize=10)
        for i in range(2):
            axs[i, j].set_axis_off()
    axs[0, 0].text(-0.05, 0.5, "amplitude", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right")
    axs[1, 0].text(-0.05, 0.5, "phase", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle(ex["case"], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _read_rows(csv_path: Path) -> list:
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if k in ("case", "dataset", "mode", "method"):
                    out[k] = v
                elif k in ("leds", "n_predict_leds"):
                    out[k] = int(float(v)) if v not in ("", None) else None
                else:
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        out[k] = v
            rows.append(out)
    return rows


def write_summary(out: Path, rows: list, repeat_pairs) -> dict:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("case", "method", "leds"), k))
    with open(out / "metrics_fpm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary = {"by_case": aggregate(rows, ("case", "method")),
               "by_dataset": aggregate(rows, ("dataset", "method")),
               "by_mode_fraction": aggregate(rows, ("mode", "fraction", "method")),
               "overall": aggregate(rows, ("method",)),
               "repeatability": repeatability(repeat_pairs)}
    (out / "summary_fpm.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs="*", default=None,
                    help="combine the metrics_fpm.csv of several partial runs (dirs) into --out (no evaluation)")
    ap.add_argument("--data", nargs="+", default=[], help="dir:holdout | dir:full")
    ap.add_argument("--model", action="append", default=[], help="name=checkpoint.pt")
    ap.add_argument("--hybrid-model", default=None)
    ap.add_argument("--hybrid", type=int, nargs="*", default=[5, 10])
    ap.add_argument("--repeat", nargs="*", default=[], help="dirA,dirB pairs (repeated acquisitions)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.merge is not None:          # partial runs (e.g. two processes on two cores) → one summary
        import shutil
        rows = []
        for d in a.merge:
            rows += _read_rows(Path(d) / "metrics_fpm.csv")
            for fig_file in Path(d).glob("fig_*.png"):
                if fig_file.parent.resolve() != out.resolve():
                    shutil.copy(fig_file, out / fig_file.name)
        summary = write_summary(out, rows, [p.split(",") for p in a.repeat])
        print(json.dumps(summary["overall"], indent=1, default=float))
        return
    models = {}
    for spec in a.model:
        n, ck = spec.split("=", 1)
        models[n] = TorchTiled(ck)
    hmax = max(a.hybrid) if a.hybrid else None
    methods = ["physics"] + ([f"physics+{hmax}"] if hmax else []) + list(models) + \
        ([f"hybrid@{hmax}"] if (hmax and a.hybrid_model) else [])
    rows = []
    for spec in a.data:
        d, mode = spec.rsplit(":", 1)
        r, examples = eval_dataset(Path(d), mode, models, a.hybrid_model, a.hybrid)
        rows += r
        write_summary(out, rows, [])            # partial results are kept if a later dataset fails
        for ex in examples:                     # figures now, so the full-size arrays are not kept in memory
            save_example_figure(ex, [m for m in methods if m in ex], out / f"fig_{ex['case'].replace('/', '_')}.png")
        del examples
    summary = write_summary(out, rows, [p.split(",") for p in a.repeat])
    print(json.dumps(summary["overall"], indent=1, default=float))


if __name__ == "__main__":
    main()
