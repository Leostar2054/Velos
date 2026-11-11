"""
M3 evaluation: synthetic test set (ground truth) + experimental hold-out data.

    python -m velos_ml.evaluate_m3 --synthetic data/synthetic_v2 \
        --real data/real_clam:holdout data/real_plant:full \
        --model m2=runs/baseline_unet/best.pt --model v2_syn=runs/refine_v2_synthetic/best.pt \
        --model v2=runs/refine_v2/best.pt --hybrid-model v2 --out results/m3

Methods compared on every test case
-----------------------------------
physics        ePIE on the reduced scan (converged, 300 it) — the input of every model
physics+300    the same, 300 further iterations (does more physics alone help?)
<model>        a refinement model applied to the physics result (tiled exactly as in the backend)
hybrid@k       the final model's output seeding k further ePIE iterations (backend "hybrid" mode)
dense          ePIE on the full scan (121 frames synthetic / all frames real) — not available
               to a reduced-scan user; shown as the upper reference

Metrics: NRMSE (complex), PSNR/SSIM of phase and amplitude after removing the global
complex scalar and phase ramp (ground truth for synthetic; full-scan reconstruction for real
data), and two data-consistency errors that need no ground truth: on the frames used, and on
frames NOT used ("prediction error" — synthetic: the dense scan's 121 frames; real: the
held-out frames).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from velos_recon import ReconConfig, core, load_dataset, reconstruct
from velos_recon.engine import data_error
from velos_recon.metrics import amplitude_phase_metrics
from velos_recon.normalise import NormParams, denormalise
from velos_recon.refine import ModelPackage

from .dataset import erode, prepare_input
from .model import build_model
from .phantoms import PhantomConfig, make_phantom, make_probe_variant
from .simulate import _dataset_from_sim

METRICS = ("nrmse_complex", "psnr_phase", "ssim_phase", "psnr_amplitude", "ssim_amplitude", "mae_phase_rad")


class TorchTiled(ModelPackage):
    """A PyTorch checkpoint run through the backend's tiling code (same T, overlap, blending)."""

    def __init__(self, checkpoint: str, tile: int = 256, overlap: int = 64):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = state["config"]
        self.model = build_model(self.cfg).eval(); self.model.load_state_dict(state["model"])
        self.tile, self.overlap = tile, overlap
        self.norm = self.cfg.get("data", {}).get("norm", "median")
        self.card, self.runtime = {"name": Path(checkpoint).parent.name}, "pytorch"

        def run(x):
            with torch.no_grad():
                return self.model(torch.from_numpy(x)).numpy()
        self._run = run

    def refine_full(self, obj, mask):
        """→ (refined object in the physics frame, normalised refined z, normalised input z)."""
        x, z_in, params = prepare_input(obj, mask, self.norm)
        y = self.apply(x)
        z = np.where(mask, y[0] + 1j * y[1], 0.0)
        if params is None:      # M2 convention (median only): recover its scalar for denormalising
            vals = obj[mask]
            params = NormParams(float(np.median(np.abs(vals))), float(np.median(np.angle(vals))), 0, 0, False)
        return np.where(mask, denormalise(z, params), obj), z, z_in


def _metrics(est, ref, mask, register=False):
    m = amplitude_phase_metrics(np.asarray(est, np.complex128), np.asarray(ref, np.complex128), mask,
                                register=register)
    return {k: float(m[k]) for k in METRICS}


# ---------------------------------------------------------------------- synthetic
def _sim_datasets(d, cfg):
    seed = int(d["seed"]); N, Np = cfg["size"], cfg["probe_px"]
    O = make_phantom(seed, PhantomConfig(size=N)); P = make_probe_variant(Np, seed + 7, core.make_probe)
    out = {}
    for tag, ov in (("sparse", float(d["sparse_overlap"])), ("dense", cfg["dense_overlap"])):
        pos = core.make_positions(N, Np, overlap=ov, jitter=cfg["jitter_px"], seed=seed)
        amps = core.simulate(O, P, pos, photons=float(d["photons"]), seed=seed)
        out[tag] = _dataset_from_sim(O, P, pos, amps)
    assert np.array_equal(out["sparse"].positions_px(), d["sparse_pos"]), "regenerated scan differs"
    return out


def eval_synthetic(root, models, hybrid_model, hybrid_iters, long_iters, limit=None):
    root = Path(root); cfg = json.loads((root / "config.json").read_text())
    files = sorted((root / "test").glob("sample_*.npz"))[:limit]
    rcfg = ReconConfig(engine=cfg["engine"], engine_params=cfg.get("engine_params", {}))
    rows, examples = [], []
    for i, f in enumerate(files):
        t0 = time.time()
        d = dict(np.load(f)); gt = d["gt"].astype(np.complex128); em = d["eval_mask"]
        dsets = _sim_datasets(d, cfg)
        S, PS, MS = d["sparse_recon"].astype(np.complex128), d["sparse_probe"].astype(np.complex128), d["sparse_mask"]
        est = {"physics": S}
        long = reconstruct(dsets["sparse"], ReconConfig(**{**rcfg.__dict__, "iters": long_iters, "seed": int(d["seed"])}),
                           init_object=S, init_probe=PS)
        est[f"physics+{long_iters}"] = long.obj
        norm_z = {}
        for name, mdl in models.items():
            obj_ml, z, _ = mdl.refine_full(S, MS)
            est[name] = obj_ml; norm_z[name] = z
        for k in hybrid_iters:
            h = reconstruct(dsets["sparse"], ReconConfig(**{**rcfg.__dict__, "iters": k, "seed": int(d["seed"])}),
                            init_object=est[hybrid_model], init_probe=PS)
            est[f"hybrid@{k}"] = h.obj
        est["dense"] = d["dense_recon"].astype(np.complex128)
        for name, o in est.items():
            r = {"case": f.stem, "frames": int(d["sparse_n_frames"]), "photons": float(d["photons"]),
                 "method": name, **_metrics(o, gt, em, register=True)}
            probe = PS if name != "dense" else None
            if probe is not None:
                r["data_error_used"] = data_error(o, probe, dsets["sparse"])
                r["data_error_predict"] = data_error(o, probe, dsets["dense"])
            rows.append(r)
        examples.append({"file": f.stem, "frames": int(d["sparse_n_frames"]), "gt": gt, "mask": em,
                         **{k: v for k, v in est.items()}})
        print(f"  synthetic {i + 1}/{len(files)} ({int(d['sparse_n_frames'])} frames): "
              + " · ".join(f"{r['method']} {r['psnr_phase']:.1f}" for r in rows[-len(est):]) + f"  [{time.time() - t0:.0f}s]",
              flush=True)
    return rows, examples


# ---------------------------------------------------------------------- real data
def eval_real(root, mode, models, hybrid_model, hybrid_iters, subsets, mat_path, crop=128, split=0.65):
    root = Path(root); meta = json.loads((root / "meta.json").read_text()) if (root / "meta.json").exists() else {}
    dense = np.load(root / "dense.npz"); D, MD = dense["obj"].astype(np.complex128), dense["mask"]
    ds = load_dataset(mat_path).center_crop(crop)
    H, W = D.shape
    region = np.ones((H, W), bool)
    c_split = 0
    if mode == "holdout":
        c_split = int(split * W); region[:, :c_split] = False
    pos = ds.positions_px()
    rows, examples = [], []
    for k in subsets:
        s = np.load(root / "subsets" / f"sub_{k:02d}.npz")
        S, PS, MS, keep = s["obj"].astype(np.complex128), s["probe"].astype(np.complex128), s["mask"], s["keep"]
        held = np.setdiff1d(np.arange(ds.n_patterns), keep)
        if mode == "holdout":   # only frames whose window lies (almost) entirely in the hold-out columns
            held = held[pos[held, 1] >= c_split - 16]
            used = keep[pos[keep, 1] >= c_split - 16]
        else:
            used = keep
        em = erode(MS & MD, 4) & region
        sub_ds = ds.subset(keep)
        est = {"physics": S}
        for name, mdl in models.items():
            if mode == "holdout":   # the model only ever sees hold-out pixels
                x, z_in, params = prepare_input(S, MS, mdl.norm)
                y = mdl.apply(np.ascontiguousarray(x[:, :, c_split:]))
                zfull = np.zeros_like(S); zfull[:, c_split:] = y[0] + 1j * y[1]
                zfull = np.where(MS, zfull, 0)
                if params is None:
                    vals = S[MS]; params = NormParams(float(np.median(np.abs(vals))), float(np.median(np.angle(vals))), 0, 0, False)
                obj = denormalise(zfull, params)
                obj = np.where(MS & region, obj, S)
            else:
                obj, _, _ = mdl.refine_full(S, MS)
            est[name] = obj
        for it in hybrid_iters:
            h = reconstruct(sub_ds, ReconConfig(engine="epie", iters=it, seed=k), init_object=est[hybrid_model], init_probe=PS)
            est[f"hybrid@{it}"] = h.obj
        if (root / "dense_repeat.npz").exists():     # repeatability floor of the reference itself
            rep = np.load(root / "dense_repeat.npz")
            est["dense_repeat"] = rep["obj"].astype(np.complex128)
        for name, o in est.items():
            probe = rep["probe"].astype(np.complex128) if name == "dense_repeat" else PS
            r = {"case": f"{root.name}/sub_{k:02d}", "fraction": float(s["fraction"]), "frames": int(len(keep)),
                 "method": name, **_metrics(o, D, em, register=True),
                 "data_error_used": data_error(o, probe, ds, used),
                 "data_error_predict": data_error(o, probe, ds, held), "n_predict_frames": int(len(held))}
            rows.append(r)
        examples.append({"case": f"{root.name}/sub_{k:02d}", "fraction": float(s["fraction"]), "mask": em,
                         "region_cols": c_split, "dense": D, **est})
        print(f"  {root.name} sub_{k:02d} ({len(keep)} frames): " +
              " · ".join(f"{r['method']} {r['psnr_phase']:.1f}/{r['data_error_predict']:.3f}" for r in rows[-len(est):]), flush=True)
    return rows, examples


# ---------------------------------------------------------------------- reporting
def aggregate(rows, key_fields=("method",)):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in key_fields), []).append(r)
    out = []
    for key, rs in groups.items():
        g = dict(zip(key_fields, key)); g["n"] = len(rs)
        for m in METRICS + ("data_error_used", "data_error_predict"):
            vals = [r[m] for r in rs if m in r and r[m] is not None and np.isfinite(r[m])]
            if vals:
                g[m] = float(np.mean(vals)); g[m + "_std"] = float(np.std(vals))
        out.append(g)
    return out


def _bin(frames):
    return "36-49 frames" if frames < 50 else ("50-64 frames" if frames < 65 else "65-81 frames")


def write_outputs(out, syn_rows, real_rows, method_order):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("synthetic", syn_rows), ("real", real_rows)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("case", "method", "frames"), k))
        with open(out / f"metrics_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary = {"synthetic": {}, "real": {}}
    if syn_rows:
        for r in syn_rows:
            r["bin"] = _bin(r["frames"])
        summary["synthetic"]["overall"] = aggregate(syn_rows)
        summary["synthetic"]["by_density"] = aggregate(syn_rows, ("bin", "method"))
        summary["synthetic"]["n_cases"] = len({r["case"] for r in syn_rows})
    if real_rows:
        summary["real"]["by_case"] = aggregate(real_rows, ("case", "method"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default=None)
    ap.add_argument("--real", nargs="*", default=[], help="root:mode:mat:subsets (mode holdout|full)")
    ap.add_argument("--model", action="append", default=[], help="name=checkpoint")
    ap.add_argument("--hybrid-model", default="v2")
    ap.add_argument("--hybrid-iters", type=int, nargs="*", default=[10, 30, 100])
    ap.add_argument("--long-iters", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    models = {}
    for item in a.model:
        n, p = item.split("=", 1); models[n] = TorchTiled(p)
    syn_rows, syn_ex, real_rows, real_ex = [], [], [], []
    if a.synthetic:
        syn_rows, syn_ex = eval_synthetic(a.synthetic, models, a.hybrid_model, a.hybrid_iters, a.long_iters, a.limit)
    for spec in a.real:
        root, mode, mat, subs = spec.split("::")
        r, e = eval_real(root, mode, models, a.hybrid_model, a.hybrid_iters, [int(s) for s in subs.split(",")], mat)
        real_rows += r; real_ex += e
    order = ["physics", f"physics+{a.long_iters}", *models, *[f"hybrid@{k}" for k in a.hybrid_iters], "dense"]
    write_outputs(a.out, syn_rows, real_rows, order)
    np.save(Path(a.out) / "_examples.npy", np.array({"synthetic": syn_ex[:30], "real": real_ex}, dtype=object),
            allow_pickle=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
