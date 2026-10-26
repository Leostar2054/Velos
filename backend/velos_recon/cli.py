"""Command-line interface.

    python -m velos_recon.cli info <dataset>
    python -m velos_recon.cli reconstruct <dataset> --engine wasp --iters 100 --crop 256 --out results/
    python -m velos_recon.cli subset <dataset> --every 2 --out sparse.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .data import load_dataset, save_npz
from .engine import ReconConfig, reconstruct, ENGINE_NAMES
from .viz import save_object_images, save_probe_image, save_convergence


def _cmd_info(a):
    ds = load_dataset(a.dataset)
    print(json.dumps(ds.summary(), indent=2))


def _cmd_subset(a):
    ds = load_dataset(a.dataset)
    sub = ds.sparse_subset(every=a.every, fraction=a.fraction, seed=a.seed)
    if a.crop:
        sub = sub.center_crop(a.crop)
    save_npz(sub, a.out)
    print(f"wrote {a.out}: {sub.n_patterns} of {ds.n_patterns} patterns")


def _cmd_reconstruct(a):
    ds = load_dataset(a.dataset)
    cfg = ReconConfig(engine=a.engine, iters=a.iters, seed=a.seed, crop=a.crop,
                      engine_params=json.loads(a.params) if a.params else {},
                      progress_every=max(1, a.iters // 20))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def prog(it, n, info):
        print(f"  iter {it:4d}/{n}  err={info['err']:.4e}  t={info['elapsed_s']:.1f}s", flush=True)

    print(f"[velos_recon] {ds.name}: {ds.n_patterns} patterns, frame {ds.frame_shape}, engine={a.engine}")
    res = reconstruct(ds, cfg, progress=prog)
    res.to_npz(out / "result.npz")
    files = save_object_images(res.obj, res.mask, out, dx_um=res.dx_m * 1e6,
                               title=f"{ds.name} · {ENGINE_NAMES[a.engine]} · {a.iters} it")
    files["probe"] = save_probe_image(res.probe, out)
    files["convergence"] = save_convergence(res.history, out)
    summary = {"dataset": ds.summary(), "result": res.summary(), "files": files}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["result"], indent=2))
    print(f"done in {time.time() - t0:.1f}s → {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="velos_recon")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("info"); s.add_argument("dataset"); s.set_defaults(fn=_cmd_info)
    s = sp.add_parser("subset"); s.add_argument("dataset"); s.add_argument("--every", type=int, default=2)
    s.add_argument("--fraction", type=float, default=None); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--crop", type=int, default=None)
    s.add_argument("--out", required=True); s.set_defaults(fn=_cmd_subset)
    s = sp.add_parser("reconstruct"); s.add_argument("dataset")
    s.add_argument("--engine", default="epie", choices=sorted(ENGINE_NAMES))
    s.add_argument("--iters", type=int, default=100); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--crop", type=int, default=None); s.add_argument("--params", default=None,
                                                                     help='JSON dict of engine params')
    s.add_argument("--out", required=True); s.set_defaults(fn=_cmd_reconstruct)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
