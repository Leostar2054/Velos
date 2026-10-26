"""
Config-driven training.

    python -m velos_ml.train --config configs/baseline_unet.yaml [--data DIR] [--out DIR]
                             [--set train.epochs=5 --set model.base=16 ...]

Writes to the run directory: config.json (resolved), log.csv (per epoch), best.pt /
last.pt (model + config + epoch + metrics), and training_curves.png. Seeds are fixed
and recorded; the run is reproducible on CPU (bit-exact) and on GPU up to cuDNN nondeterminism.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import SparseRefineDataset
from .model import build_model, count_params

DEFAULT = {
    "data": {"root": "data/synthetic", "crop": 128, "limit_train": None},
    "model": {"base": 32, "depth": 3, "residual": True},
    "train": {"epochs": 30, "batch_size": 8, "lr": 1e-3, "weight_decay": 1e-4, "seed": 0,
              "num_workers": 0, "loss": "l1", "amp_weight": 1.0},
    "name": "baseline_unet",
}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        d[k] = deep_update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


def parse_sets(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def masked_loss(pred, target, mask, kind="l1"):
    """Loss on (Re, Im) inside the evaluation mask; L1 is robust to the few blown-up pixels
    of low-coverage regions in the physics input."""
    diff = pred - target
    if kind == "l2":
        per = (diff ** 2).sum(1, keepdim=True)
    else:
        per = diff.abs().sum(1, keepdim=True)
    return (per * mask).sum() / (mask.sum() * 2 + 1e-8)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval(); tot = 0.0; n = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        tot += masked_loss(model(x), y, m).item() * x.shape[0]; n += x.shape[0]
    return tot / max(n, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = json.loads(json.dumps(DEFAULT))
    deep_update(cfg, yaml.safe_load(open(a.config)) or {})
    deep_update(cfg, parse_sets(a.set))
    if a.data:
        cfg["data"]["root"] = a.data
    out = Path(a.out or f"runs/{cfg['name']}"); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(cfg["train"]["seed"]))
    torch.set_num_threads(max(1, torch.get_num_threads()))

    tr = SparseRefineDataset(cfg["data"]["root"], "train", crop=cfg["data"]["crop"], augment=True,
                             limit=cfg["data"].get("limit_train"), seed=cfg["train"]["seed"])
    va = SparseRefineDataset(cfg["data"]["root"], "val", crop=None, augment=False)
    tl = DataLoader(tr, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                    num_workers=int(cfg["train"]["num_workers"]), drop_last=True)
    vl = DataLoader(va, batch_size=4, shuffle=False)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg["train"]["lr"]),
                                                total_steps=epochs * len(tl), pct_start=0.15)
    cfg["_resolved"] = {"device": str(device), "n_train": len(tr), "n_val": len(va),
                        "params": count_params(model), "torch": torch.__version__}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[train] {cfg['name']}: {len(tr)} train / {len(va)} val · {count_params(model):,} params · {device}")

    best = float("inf"); t0 = time.time(); rows = []
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])
        for ep in range(1, epochs + 1):
            model.train(); tot = 0.0; n = 0
            for x, y, m in tl:
                x, y, m = x.to(device), y.to(device), m.to(device)
                loss = masked_loss(model(x), y, m, cfg["train"]["loss"])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tot += loss.item() * x.shape[0]; n += x.shape[0]
            train_loss = tot / max(n, 1); val_loss = evaluate_loader(model, vl, device)
            lr = opt.param_groups[0]["lr"]; el = time.time() - t0
            rows.append((ep, train_loss, val_loss, lr, el)); w.writerow([ep, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.2e}", f"{el:.0f}"]); f.flush()
            state = {"model": model.state_dict(), "config": cfg, "epoch": ep, "val_loss": val_loss}
            torch.save(state, out / "last.pt")
            if val_loss < best:
                best = val_loss; torch.save(state, out / "best.pt")
            print(f"  epoch {ep:3d}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}  lr {lr:.1e}  ({el:.0f}s)"
                  + ("  *" if val_loss == best else ""), flush=True)
    _plot(rows, out / "training_curves.png")
    print(f"[train] done in {time.time() - t0:.0f}s · best val loss {best:.4f} → {out}")


def _plot(rows, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ep = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ep, [r[1] for r in rows], label="train"); ax.plot(ep, [r[2] for r in rows], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked L1 (Re, Im)"); ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
