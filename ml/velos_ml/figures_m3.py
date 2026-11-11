"""
Figures for the M3 evaluation (reads the output folder of ``velos_ml.evaluate_m3``).

    python -m velos_ml.figures_m3 results/m3 [--runs runs/refine_v2_synthetic runs/refine_v2]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from velos_recon.metrics import align, estimate_shift, shift_image  # noqa: E402

LABEL = {"physics": "Physics (reduced scan)", "physics+300": "Physics +300 it", "m2": "M2 model",
         "v2_syn": "v2 (synthetic only)", "v2": "v2 model", "hybrid@10": "Hybrid (+10 it)",
         "hybrid@30": "Hybrid (+30 it)", "hybrid@100": "Hybrid (+100 it)", "dense": "Dense scan (reference)",
         "gt": "Ground truth"}
COLOR = {"physics": "#64748b", "physics+300": "#94a3b8", "m2": "#f59e0b", "v2_syn": "#a78bfa", "v2": "#7c3aed",
         "hybrid@10": "#67e8f9", "hybrid@30": "#06b6d4", "hybrid@100": "#0e7490", "dense": "#16a34a"}


def _bbox(mask):
    r = np.where(mask.any(1))[0]; c = np.where(mask.any(0))[0]
    return slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1)


def _phase_panel(est, ref, mask, register=False):
    est = np.asarray(est, np.complex128)
    if register and est is not ref:
        est = shift_image(est, estimate_shift(est, ref, mask))
    a = align(est, np.asarray(ref, np.complex128), mask)
    rs, cs = _bbox(mask)
    ph = np.angle(a[rs, cs]); m = mask[rs, cs]
    return np.where(m, ph, np.nan)


def grid(rows, cols, ref_key, out, title, cmap="twilight_shifted", row_labels=None, col_w=2.3, register=False):
    """rows: list of dicts with arrays (complex) + 'mask'; cols: keys to show (ref first)."""
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(col_w * len(cols), col_w * len(rows) + 0.5), squeeze=False)
    for i, ex in enumerate(rows):
        ref = ex[ref_key]; mask = ex["mask"]
        ref_ph = _phase_panel(ref, ref, mask)
        lo, hi = np.nanpercentile(ref_ph, 1), np.nanpercentile(ref_ph, 99)
        for j, k in enumerate(cols):
            ax = axs[i, j]
            if k in ex:
                ax.imshow(_phase_panel(ex[k], ref, mask, register and k != ref_key), cmap=cmap, vmin=lo, vmax=hi,
                          interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(LABEL.get(k, k), fontsize=9)
            if j == 0 and row_labels:
                ax.set_ylabel(row_labels[i], fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def _read_csv(path):
    with open(path) as f:
        return [{k: (float(v) if k not in ("case", "method", "bin") and v not in ("",) else v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def metrics_vs_frames(rows, out, methods=("physics", "v2", "hybrid@30", "dense")):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, lab in ((axs[0], "psnr_phase", "phase PSNR (dB) ↑"), (axs[1], "nrmse_complex", "NRMSE ↓")):
        for m in methods:
            pts = sorted((r["frames"], r[key]) for r in rows if r["method"] == m)
            if not pts:
                continue
            x, y = np.array(pts).T
            ax.scatter(x, y, s=12, alpha=0.45, color=COLOR.get(m), label=None)
            edges = [35, 50, 65, 82]
            bx, by = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (x >= lo) & (x < hi)
                if sel.any():
                    bx.append(x[sel].mean()); by.append(np.median(y[sel]))
            ax.plot(bx, by, "-o", color=COLOR.get(m), label=LABEL.get(m, m), lw=2, ms=5)
        ax.set_xlabel("frames in the reduced scan (dense scan = 121)"); ax.set_ylabel(lab); ax.grid(alpha=.3)
        if key == "nrmse_complex":
            ax.set_yscale("log")
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Synthetic test set: quality vs scan density (dots = cases, lines = bin medians)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def hybrid_curve(rows, out, hybrid_iters=(10, 30, 100), base="v2"):
    xs = [0] + list(hybrid_iters)
    keys = [base] + [f"hybrid@{k}" for k in hybrid_iters]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, lab in ((axs[0], "psnr_phase", "phase PSNR (dB) ↑"), (axs[1], "data_error_predict", "prediction error on unseen frames ↓")):
        def mean(m):
            v = [r[key] for r in rows if r["method"] == m and r.get(key) not in (None, "")]
            return float(np.mean(v)) if v else np.nan
        ax.plot(xs, [mean(k) for k in keys], "-o", color=COLOR["hybrid@30"], label="ML (0) → hybrid (+k physics it)")
        for m, ls in (("physics", "--"), ("physics+300", ":"), ("dense", "-.")):
            val = mean(m)
            if np.isfinite(val):
                ax.axhline(val, ls=ls, color=COLOR[m], label=LABEL[m])
        ax.set_xlabel("physics iterations after the ML step"); ax.set_ylabel(lab); ax.grid(alpha=.3)
        ax.set_xscale("symlog", linthresh=10)
    axs[0].legend(fontsize=8)
    fig.suptitle("Hybrid mode on the synthetic test set (means)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def bars(summary_rows, out, methods, title):
    keys = (("psnr_phase", "phase PSNR (dB) ↑"), ("ssim_phase", "phase SSIM ↑"), ("nrmse_complex", "NRMSE ↓"))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (k, lab) in zip(axs, keys):
        vals = [next((g for g in summary_rows if g["method"] == m), None) for m in methods]
        ms = [m for m, v in zip(methods, vals) if v]
        vals = [v for v in vals if v]
        ax.bar(range(len(ms)), [v[k] for v in vals], yerr=[v.get(k + "_std", 0) for v in vals],
               color=[COLOR.get(m, "#999") for m in ms], capsize=3)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([LABEL.get(m, m) for m in ms], rotation=35, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10); ax.grid(axis="y", alpha=.3)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def training_curves(run_dirs, out):
    fig, ax = plt.subplots(figsize=(7, 3.6)); off = 0
    for rd, c in zip(run_dirs, ("#7c3aed", "#06b6d4")):
        rows = list(csv.DictReader(open(Path(rd) / "log.csv")))
        ep = [off + int(r["epoch"]) for r in rows]
        ax.plot(ep, [float(r["train_loss"]) for r in rows], color=c, alpha=.6, label=f"{Path(rd).name} train")
        ax.plot(ep, [float(r["val_loss"]) for r in rows], color=c, lw=2, label=f"{Path(rd).name} val (synthetic)")
        off = ep[-1]
    ax.set_xlabel("epoch (stage 1 → stage 2)"); ax.set_ylabel("phase-aware loss"); ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args(argv)
    res = Path(a.results)
    ex = np.load(res / "_examples.npy", allow_pickle=True).item()
    summary = json.loads((res / "summary.json").read_text())
    made = []
    syn = ex.get("synthetic") or []
    if syn:
        syn = sorted(syn, key=lambda e: e["frames"])
        pick = [syn[0], syn[len(syn) // 2], syn[-1]]
        for e in pick:
            e["mask"] = e["mask"]
        cols = [c for c in ("gt", "physics", "m2", "v2", "hybrid@30", "dense") if c in pick[0]]
        made.append(grid(pick, cols, "gt", res / "fig_synthetic_examples.png",
                         "Synthetic test cases — phase (rad), aligned to the ground truth",
                         row_labels=[f"{e['frames']} frames" for e in pick]))
        rows = _read_csv(res / "metrics_synthetic.csv")
        made.append(metrics_vs_frames(rows, res / "fig_metrics_vs_frames.png"))
        made.append(hybrid_curve(rows, res / "fig_hybrid_curve.png"))
        order = [m for m in ("physics", "physics+300", "m2", "v2_syn", "v2", "hybrid@10", "hybrid@30", "hybrid@100", "dense")
                 if any(g["method"] == m for g in summary["synthetic"]["overall"])]
        made.append(bars(summary["synthetic"]["overall"], res / "fig_metrics_bars.png", order,
                         f"Synthetic test set (n = {summary['synthetic']['n_cases']}), mean ± std"))
    real = ex.get("real") or []
    for name in ("real_clam", "real_plant"):
        rs = [e for e in real if e["case"].startswith(name)]
        if not rs:
            continue
        for e in rs:
            e["mask"] = e["mask"]
        cols = [c for c in ("dense", "physics", "m2", "v2", "hybrid@30") if c in rs[0]]
        title = ("Clam (635 nm), spatial hold-out region — phase (rad), aligned to the all-frames reconstruction"
                 if name == "real_clam" else "Plant (675 nm, specimen never used in training) — phase (rad), aligned to the all-frames reconstruction")
        made.append(grid(rs, cols, "dense", res / f"fig_{name}.png", title,
                         row_labels=[f"{round(100 * e['fraction'])} % of frames" for e in rs], col_w=2.6, register=True))
    if a.runs:
        made.append(training_curves(a.runs, res / "fig_training_curves.png"))
    print("\n".join(str(m) for m in made))


if __name__ == "__main__":
    main()
