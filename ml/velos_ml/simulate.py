"""
Dataset generation: phantom → diffraction frames (Client's forward model) →
physics reconstruction from a *sparse* scan (ML input / baseline) and from a
*dense* scan (physics reference) → normalised arrays saved as .npz shards.

    python -m velos_ml.simulate --config configs/data_synthetic.yaml --out data/synthetic

Every sample stores: gt (complex), sparse_recon, dense_recon, masks, metadata
(seed, overlap, photons, engine, iterations). All reconstructions use
``velos_recon.reconstruct`` (the Client's engines) — nothing is re-implemented.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from velos_recon import ReconConfig, core, reconstruct
from velos_recon.metrics import amplitude_phase_metrics
from velos_recon.data import PtychoDataset

from .phantoms import PhantomConfig, make_phantom, make_probe_variant

DEFAULT = {
    "size": 256, "probe_px": 64, "n_train": 240, "n_val": 30, "n_test": 30,
    "dense_overlap": 0.70, "sparse_overlap": 0.60, "jitter_px": 2.0,
    "photons": [5e3, 5e4],
    "engine": "wasp", "engine_params": {"probe_lr": 0.1},
    "sparse_iters": 100, "dense_iters": 150, "seed": 1234,
}


def normalise(obj: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Self-normalisation usable without ground truth: median amplitude → 1, median phase → 0
    (background medium). Removes the global complex-scalar ambiguity of ptychography."""
    vals = obj[mask]
    scale = np.median(np.abs(vals)) + 1e-12
    phase0 = np.median(np.angle(vals))
    out = obj / scale * np.exp(-1j * phase0)
    return np.where(mask, out, 0.0)


def erode(mask: np.ndarray, r: int = 4) -> np.ndarray:
    """Binary erosion by a (2r+1)² square (pure numpy)."""
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out &= np.roll(np.roll(mask, dy, 0), dx, 1)
    out[:r, :] = out[-r:, :] = out[:, :r] = out[:, -r:] = False
    return out


def _dataset_from_sim(O, P, pos, amps) -> PtychoDataset:
    Np = P.shape[0]
    dx, wl, dist = 1e-6, 1e-6, 1e-2
    pitch = wl * dist / (Np * dx)
    positions_m = np.stack([pos[:, 1], pos[:, 0]], 1).astype(float) * dx
    return PtychoDataset(dps=(amps ** 2).astype(np.float32), positions_m=positions_m,
                         wavelength=wl, pixel_pitch=pitch, distance=dist,
                         init_probe=P, ground_truth=O, name="synthetic")


def make_sample(args):
    idx, seed, cfg = args
    rng = np.random.default_rng(seed)
    N, Np = cfg["size"], cfg["probe_px"]
    O = make_phantom(seed, PhantomConfig(size=N))
    P = make_probe_variant(Np, seed + 7, core.make_probe)
    photons = float(10 ** rng.uniform(np.log10(cfg["photons"][0]), np.log10(cfg["photons"][1])))
    out = {"gt": O.astype(np.complex64), "seed": seed, "photons": photons}
    rc = ReconConfig(engine=cfg["engine"], engine_params=cfg.get("engine_params", {}), seed=seed)
    for tag, overlap, iters in (("sparse", cfg["sparse_overlap"], cfg["sparse_iters"]),
                                ("dense", cfg["dense_overlap"], cfg["dense_iters"])):
        pos = core.make_positions(N, Np, overlap=overlap, jitter=cfg["jitter_px"], seed=seed)
        amps = core.simulate(O, P, pos, photons=photons, seed=seed)
        ds = _dataset_from_sim(O, P, pos, amps)
        rc.iters = iters
        res = reconstruct(ds, rc)
        out[f"{tag}_recon"] = res.obj.astype(np.complex64)
        out[f"{tag}_mask"] = res.mask                      # illuminated region (coverage > 5 % of max)
        out[f"{tag}_n_frames"] = len(pos)
        out[f"{tag}_err"] = float(res.history["err"][-1])
    # evaluation region: illuminated by BOTH scans, eroded to exclude the soft probe edge
    out["eval_mask"] = erode(out["sparse_mask"] & out["dense_mask"], 4)
    for tag in ("sparse", "dense"):
        out[f"{tag}_nrmse"] = amplitude_phase_metrics(out[f"{tag}_recon"].astype(np.complex128), O,
                                                      out["eval_mask"])["nrmse_complex"]
    return idx, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = dict(DEFAULT); cfg.update(yaml.safe_load(open(a.config)) or {})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    splits = {"train": cfg["n_train"], "val": cfg["n_val"], "test": cfg["n_test"]}
    if a.limit:
        splits = {k: min(v, a.limit) for k, v in splits.items()}
    t0 = time.time(); base = cfg["seed"]; log = []
    for split, n in splits.items():
        d = out / split; d.mkdir(exist_ok=True)
        seeds = [base + {"train": 0, "val": 100000, "test": 200000}[split] + i for i in range(n)]
        jobs = [(i, s, cfg) for i, s in enumerate(seeds)]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for idx, sample in ex.map(make_sample, jobs):
                np.savez_compressed(d / f"sample_{idx:04d}.npz", **sample)
                log.append({"split": split, "idx": idx, "seed": sample["seed"], "photons": sample["photons"],
                            "sparse_frames": sample["sparse_n_frames"], "dense_frames": sample["dense_n_frames"],
                            "sparse_nrmse": sample["sparse_nrmse"], "dense_nrmse": sample["dense_nrmse"]})
                if len(log) % 10 == 0:
                    print(f"  [{split}] {idx + 1}/{n}  sparse NRMSE {sample['sparse_nrmse']:.3f}  "
                          f"dense NRMSE {sample['dense_nrmse']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    (out / "generation_log.json").write_text(json.dumps(log, indent=1))
    sp = np.array([l["sparse_nrmse"] for l in log]); de = np.array([l["dense_nrmse"] for l in log])
    print(f"done: {len(log)} samples in {time.time() - t0:.0f}s · physics NRMSE sparse {sp.mean():.3f}±{sp.std():.3f} "
          f"· dense {de.mean():.3f}±{de.std():.3f}")


if __name__ == "__main__":
    main()
