"""
Real-data training/test set from an experimental scan (M3).

There is no ground truth for experimental data, so the reference is the physics
reconstruction from ALL frames ("dense"), and the inputs are physics reconstructions
from random subsets of the frames ("reduced scans"), exactly as a user of the
sparse-scanning mode would produce them:

    python -m velos_ml.realdata --mat "real data - clam.mat" --name clam --crop 128 \
        --out data/real_clam --n-subsets 14 --fractions 0.30 0.60

Outputs ``dense.npz`` (object, probe, mask, data error), ``subsets/sub_XX.npz`` (object,
probe, mask, kept frame indices, fraction, data errors on used and held-out frames) and
``meta.json``. Subsets keep the full scan's pixel grid (``PtychoDataset.subset`` preserves
the scan origin), so every reconstruction is directly comparable with the reference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from velos_recon import ReconConfig, load_dataset, reconstruct
from velos_recon.engine import data_error


def run(mat: str, name: str, out: str, crop: int = 128, engine: str = "epie",
        dense_iters: int = 200, sub_iters: int = 200, n_subsets: int = 14,
        fractions=(0.30, 0.60), fixed=(0.35, 0.50), seed: int = 2026,
        probe_from_dense: bool = False, repeat_seed: int | None = None) -> dict:
    """``probe_from_dense``: start every reduced-scan reconstruction from the probe of the
    full-scan reconstruction ("probe calibration": calibrate once with a dense scan, then scan
    sparsely). ``repeat_seed``: also save a second full-scan reconstruction with another seed
    (``dense_repeat.npz``) — the repeatability floor of the reference itself."""
    out = Path(out); (out / "subsets").mkdir(parents=True, exist_ok=True)
    ds = load_dataset(mat).center_crop(crop)
    ds.name = name
    t0 = time.time(); log = {"name": name, "crop": crop, "engine": engine, "n_frames": ds.n_patterns,
                             "dataset": ds.summary(), "subsets": []}
    dense_file = out / "dense.npz"
    if dense_file.exists():
        d = np.load(dense_file); D, PD, MD = d["obj"], d["probe"], d["mask"]
        print(f"[{name}] dense reference reused")
    else:
        res = reconstruct(ds, ReconConfig(engine=engine, iters=dense_iters, seed=seed))
        D, PD, MD = res.obj, res.probe, res.mask
        err = data_error(D, PD, ds)
        np.savez_compressed(dense_file, obj=D.astype(np.complex64), probe=PD.astype(np.complex64), mask=MD,
                            data_error=err, iters=dense_iters, elapsed_s=res.elapsed_s,
                            err_hist=np.asarray(res.history["err"]))
        print(f"[{name}] dense: {ds.n_patterns} frames, {dense_iters} it, data error {err:.4f}, "
              f"{res.elapsed_s:.0f}s", flush=True)
    log["dense"] = {"shape": list(D.shape), "iters": dense_iters}
    if repeat_seed is not None and not (out / "dense_repeat.npz").exists():
        rr = reconstruct(ds, ReconConfig(engine=engine, iters=dense_iters, seed=repeat_seed))
        np.savez_compressed(out / "dense_repeat.npz", obj=rr.obj.astype(np.complex64), probe=rr.probe.astype(np.complex64),
                            mask=rr.mask, seed=repeat_seed, data_error=data_error(rr.obj, rr.probe, ds))
        print(f"[{name}] repeat full-scan reconstruction (seed {repeat_seed}) saved", flush=True)
    log["probe_from_dense"] = bool(probe_from_dense)
    rng = np.random.default_rng(seed)
    # a few fixed fractions first (reported test cases, comparable with M2: 35 % and 50 %),
    # then random fractions in the training range
    fr_list = list(fixed) + list(rng.uniform(fractions[0], fractions[1], max(0, n_subsets - len(fixed))))
    for k, f in enumerate(fr_list):
        fpath = out / "subsets" / f"sub_{k:02d}.npz"
        if fpath.exists():                      # already built: keep its statistics in meta.json
            z = np.load(fpath)
            log["subsets"].append({"k": k, "fraction": float(z["fraction"]), "n_frames": int(len(z["keep"])),
                                   "data_error_used": float(z["data_error_used"]),
                                   "data_error_heldout": float(z["data_error_heldout"]),
                                   "elapsed_s": round(float(z["elapsed_s"]), 1)})
            continue
        r = np.random.default_rng(seed + 1000 + k)
        keep = np.sort(r.choice(ds.n_patterns, int(round(f * ds.n_patterns)), replace=False))
        held = np.setdiff1d(np.arange(ds.n_patterns), keep)
        sub = ds.subset(keep, f"_sub{k:02d}")
        res = reconstruct(sub, ReconConfig(engine=engine, iters=sub_iters, seed=seed + k),
                          init_probe=PD if probe_from_dense else None)
        if res.obj.shape != D.shape:
            raise RuntimeError(f"subset object {res.obj.shape} != dense {D.shape} (grid not preserved)")
        e_used = data_error(res.obj, res.probe, ds, keep)
        e_held = data_error(res.obj, res.probe, ds, held)
        np.savez_compressed(fpath, obj=res.obj.astype(np.complex64), probe=res.probe.astype(np.complex64),
                            mask=res.mask, keep=keep.astype(np.int32), fraction=float(f), iters=sub_iters,
                            data_error_used=e_used, data_error_heldout=e_held, elapsed_s=res.elapsed_s)
        log["subsets"].append({"k": k, "fraction": float(f), "n_frames": int(len(keep)),
                               "data_error_used": e_used, "data_error_heldout": e_held,
                               "elapsed_s": round(res.elapsed_s, 1)})
        print(f"[{name}] subset {k:02d}: {len(keep)} frames ({f:.2f}), err used {e_used:.4f} "
              f"held-out {e_held:.4f}, {res.elapsed_s:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
    (out / "meta.json").write_text(json.dumps(log, indent=1, default=float))
    return log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--engine", default="epie"); ap.add_argument("--dense-iters", type=int, default=200)
    ap.add_argument("--sub-iters", type=int, default=200); ap.add_argument("--n-subsets", type=int, default=14)
    ap.add_argument("--fractions", type=float, nargs=2, default=(0.30, 0.60))
    ap.add_argument("--fixed", type=float, nargs="*", default=(0.35, 0.50))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--probe-from-dense", action="store_true")
    ap.add_argument("--repeat-seed", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.mat, a.name, a.out, a.crop, a.engine, a.dense_iters, a.sub_iters, a.n_subsets,
        tuple(a.fractions), tuple(a.fixed), a.seed, a.probe_from_dense, a.repeat_seed)


if __name__ == "__main__":
    main()
