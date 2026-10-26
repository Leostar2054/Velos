"""
Evaluation against ground truth AND the Client's physics baseline (Contract §3.3 / §4).

    python -m velos_ml.evaluate --data data/synthetic --checkpoint runs/baseline_unet/best.pt \
                                --out results/baseline_unet [--control-iters 300]

Rows reported (mean ± std over the test split, inside the evaluation mask):
  physics_sparse        the Client's engine on the sparse scan (the ML input)  ← baseline
  physics_sparse_long   same, with --control-iters iterations (does more compute close the gap?)
  ml_refined            physics_sparse followed by the trained model
  physics_dense         the Client's engine on the dense scan (reference ceiling)
Outputs: metrics.csv (per sample), summary.json, summary.md, comparison grid and bar chart.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from velos_recon import ReconConfig, core, reconstruct
from velos_recon.metrics import align, amplitude_phase_metrics

from .dataset import SparseRefineDataset
from .infer import load_model, refine_object
from .phantoms import make_phantom, make_probe_variant
from .simulate import _dataset_from_sim

METRICS = ["nrmse_complex", "psnr_amplitude", "ssim_amplitude", "psnr_phase", "ssim_phase", "mae_phase_rad"]


def _control(sample: dict, gen_cfg: dict, iters: int) -> np.ndarray:
    """Re-run the physics engine on the same sparse scan with more iterations."""
    seed = int(sample["seed"]); N, Np = gen_cfg["size"], gen_cfg["probe_px"]
    O = make_phantom(seed); P = make_probe_variant(Np, seed + 7, core.make_probe)
    pos = core.make_positions(N, Np, overlap=gen_cfg["sparse_overlap"], jitter=gen_cfg["jitter_px"], seed=seed)
    amps = core.simulate(O, P, pos, photons=float(sample["photons"]), seed=seed)
    res = reconstruct(_dataset_from_sim(O, P, pos, amps),
                      ReconConfig(engine=gen_cfg["engine"], iters=iters, seed=seed,
                                  engine_params=gen_cfg.get("engine_params", {})))
    return res.obj


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--control-iters", type=int, default=0); ap.add_argument("--n-fig", type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gen_cfg = json.load(open(Path(a.data) / "config.json"))
    ds = SparseRefineDataset(a.data, a.split)
    model, cfg = load_model(a.checkpoint)
    rows = []; examples = []
    for i in range(len(ds)):
        s = ds.raw(i); gt = s["gt"].astype(np.complex128); m = s["eval_mask"]
        variants = {"physics_sparse": s["sparse_recon"].astype(np.complex128)}
        if a.control_iters:
            variants["physics_sparse_long"] = _control(s, gen_cfg, a.control_iters)
        variants["ml_refined"] = refine_object(s["sparse_recon"], s["sparse_mask"], a.checkpoint)
        variants["physics_dense"] = s["dense_recon"].astype(np.complex128)
        for name, z in variants.items():
            mm = amplitude_phase_metrics(z, gt, m)
            rows.append({"sample": i, "seed": int(s["seed"]), "photons": float(s["photons"]), "variant": name, **mm})
        if i < a.n_fig:
            examples.append((gt, m, {k: align(v, gt, m) for k, v in variants.items()}))
        print(f"  sample {i + 1}/{len(ds)}", flush=True)
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    summary = {}
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        summary[v] = {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs]))} for k in METRICS}
    summary["_meta"] = {"n_samples": len(ds), "split": a.split, "checkpoint": a.checkpoint,
                        "sparse_frames": int(ds.raw(0)["sparse_n_frames"]), "dense_frames": int(ds.raw(0)["dense_n_frames"]),
                        "control_iters": a.control_iters, "generator": gen_cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_md(summary, variants, out / "summary.md")
    _grid(examples, variants, out / "comparison_grid.png")
    _bars(summary, variants, out / "metrics_bars.png")
    print(open(out / "summary.md").read())


LABELS = {"physics_sparse": "Physics (sparse scan)", "physics_sparse_long": "Physics (sparse, more iters)",
          "ml_refined": "Physics (sparse) + ML", "physics_dense": "Physics (dense scan)"}


def _write_md(summary, variants, path):
    meta = summary["_meta"]
    lines = [f"# Baseline comparison — {meta['split']} split, n = {meta['n_samples']}",
             f"Sparse scan: {meta['sparse_frames']} frames · dense scan: {meta['dense_frames']} frames "
             f"({100 * (1 - meta['sparse_frames'] / meta['dense_frames']):.0f} % fewer frames).", "",
             "| Method | NRMSE (complex) ↓ | PSNR amp (dB) ↑ | SSIM amp ↑ | PSNR phase (dB) ↑ | SSIM phase ↑ | MAE phase (rad) ↓ |",
             "|---|---|---|---|---|---|---|"]
    for v in variants:
        s = summary[v]
        cell = lambda k, d=3: f"{s[k]['mean']:.{d}f} ± {s[k]['std']:.{d}f}"
        lines.append(f"| {LABELS.get(v, v)} | {cell('nrmse_complex')} | {cell('psnr_amplitude', 1)} | {cell('ssim_amplitude')} | "
                     f"{cell('psnr_phase', 1)} | {cell('ssim_phase')} | {cell('mae_phase_rad')} |")
    lines += ["", "Metrics are computed inside the evaluation mask (illuminated by both scans) after removing the "
                  "global complex scalar and linear phase ramp, which are unobservable in ptychography."]
    Path(path).write_text("\n".join(lines) + "\n")


def _grid(examples, variants, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = ["gt"] + variants
    nrow = len(examples) * 3; ncol = len(cols)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow))
    axs = np.atleast_2d(axs)
    for e, (gt, m, var) in enumerate(examples):
        imgs = {"gt": gt, **var}
        for c, name in enumerate(cols):
            z = imgs[name]
            amp = np.where(m, np.abs(z), np.nan); pha = np.where(m, np.angle(z), np.nan)
            perr = np.where(m, np.angle(z * np.conj(gt)), np.nan)
            axs[3 * e, c].imshow(amp, cmap="gray", vmin=0.6, vmax=1.05)
            axs[3 * e + 1, c].imshow(pha, cmap="twilight", vmin=-3.2, vmax=3.2)
            axs[3 * e + 2, c].imshow(perr, cmap="coolwarm", vmin=-0.5, vmax=0.5)
            if e == 0:
                axs[0, c].set_title("Ground truth" if name == "gt" else LABELS.get(name, name), fontsize=9)
        for r, lab in enumerate(("amplitude", "phase (rad)", "phase error (rad)")):
            axs[3 * e + r, 0].set_ylabel(lab, fontsize=9)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _bars(summary, variants, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = [("psnr_phase", "PSNR phase (dB)"), ("ssim_phase", "SSIM phase"), ("psnr_amplitude", "PSNR amp (dB)"),
            ("ssim_amplitude", "SSIM amp"), ("nrmse_complex", "NRMSE (complex)")]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.4))
    colors = ["#94a3b8", "#cbd5e1", "#7c3aed", "#0891b2"]
    for ax, (k, lab) in zip(axs, keys):
        means = [summary[v][k]["mean"] for v in variants]; stds = [summary[v][k]["std"] for v in variants]
        ax.bar(range(len(variants)), means, yerr=stds, capsize=3, color=[colors[i % 4] for i in range(len(variants))])
        ax.set_xticks(range(len(variants))); ax.set_xticklabels([LABELS.get(v, v).replace(" (", "\n(") for v in variants], fontsize=7)
        ax.set_title(lab, fontsize=9); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
