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

from .dataset import RealPatchDataset, SparseRefineDataset
from .model import build_model, count_params

DEFAULT = {
    # norm/target_frame: "median"/"absolute" = M2 convention; "median_ramp"/"input" = M3 (see dataset.py)
    "data": {"root": "data/synthetic", "crop": 128, "limit_train": None, "norm": "median",
             "target_frame": "absolute", "preload": False, "real": None},
    "model": {"base": 32, "depth": 3, "residual": True},
    "train": {"epochs": 30, "batch_size": 8, "lr": 1e-3, "weight_decay": 1e-4, "seed": 0,
              "num_workers": 0, "loss": "l1", "w_phase": 1.0, "w_grad": 0.5, "init_from": None,
              "pct_start": 0.15, "select": "best"},
    "name": "baseline_unet",
}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        d[k] = deep_update(d.get(k) or {}, v) if isinstance(v, dict) else v
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


def phase_aware_loss(pred, target, mask, w_phase=1.0, w_grad=0.5):
    """M3 loss = masked L1 on (Re, Im)
               + w_phase · amplitude-weighted |wrapped phase error|   (QPI cares about phase;
                 in Re/Im space phase errors of weak-amplitude pixels are under-weighted)
               + w_grad · L1 of spatial gradients of (Re, Im)          (edges / fine detail)."""
    l1 = masked_loss(pred, target, mask, "l1")
    m = mask[:, 0]
    pr, pi, tr, ti = pred[:, 0], pred[:, 1], target[:, 0], target[:, 1]
    re = pr * tr + pi * ti + 1e-6            # pred · conj(target); epsilon keeps atan2 finite at 0
    im = pi * tr - pr * ti
    dphi = torch.atan2(im, re)
    w = torch.sqrt(tr ** 2 + ti ** 2 + 1e-12).clamp(max=1.5)
    l_phase = (w * dphi.abs() * m).sum() / (m.sum() + 1e-8)
    mx = mask[..., :, 1:] * mask[..., :, :-1]; my = mask[..., 1:, :] * mask[..., :-1, :]
    gx = ((pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])).abs().sum(1, keepdim=True)
    gy = ((pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs().sum(1, keepdim=True)
    l_grad = (gx * mx).sum() / (2 * mx.sum() + 1e-8) + (gy * my).sum() / (2 * my.sum() + 1e-8)
    return l1 + w_phase * l_phase + w_grad * l_grad


def compute_loss(pred, target, mask, tcfg: dict):
    if tcfg.get("loss", "l1") == "phase_aware":
        return phase_aware_loss(pred, target, mask, float(tcfg.get("w_phase", 1.0)), float(tcfg.get("w_grad", 0.5)))
    return masked_loss(pred, target, mask, tcfg.get("loss", "l1"))


@torch.no_grad()
def evaluate_loader(model, loader, device, tcfg=None):
    model.eval(); tot = 0.0; n = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        tot += compute_loss(model(x), y, m, tcfg or {}).item() * x.shape[0]; n += x.shape[0]
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

    dcfg, tcfg = cfg["data"], cfg["train"]
    common = dict(norm=dcfg.get("norm", "median"), target_frame=dcfg.get("target_frame", "absolute"),
                  preload=bool(dcfg.get("preload", False)))
    tr = SparseRefineDataset(dcfg["root"], "train", crop=dcfg["crop"], augment=True,
                             limit=dcfg.get("limit_train"), seed=tcfg["seed"], **common)
    va = SparseRefineDataset(dcfg["root"], "val", crop=None, augment=False, **common)
    train_sets = [tr]
    if dcfg.get("real"):                      # M3: mix in experimental patches (spatial hold-out)
        rc = dcfg["real"]
        for root in ([rc["root"]] if isinstance(rc["root"], str) else rc["root"]):
            train_sets.append(RealPatchDataset(root, rc["subsets"], crop=int(rc.get("crop", dcfg["crop"])),
                                               col_range=tuple(rc.get("col_range", (0.0, 0.65))),
                                               n_per_epoch=int(rc.get("n_per_epoch", len(tr))),
                                               seed=tcfg["seed"] + 1, norm=common["norm"],
                                               target_frame=common["target_frame"]))
    from torch.utils.data import ConcatDataset
    tr_all = ConcatDataset(train_sets) if len(train_sets) > 1 else tr
    tl = DataLoader(tr_all, batch_size=int(tcfg["batch_size"]), shuffle=True,
                    num_workers=int(tcfg["num_workers"]), drop_last=True)
    vl = DataLoader(va, batch_size=4, shuffle=False)
    model = build_model(cfg).to(device)
    if tcfg.get("init_from"):                 # fine-tuning stage
        state = torch.load(tcfg["init_from"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"[train] initialised from {tcfg['init_from']} (epoch {state.get('epoch')})")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg["train"]["lr"]),
                                                total_steps=epochs * len(tl), pct_start=float(tcfg.get("pct_start", 0.15)))
    cfg["_resolved"] = {"device": str(device), "n_train": len(tr_all), "n_val": len(va),
                        "n_train_synthetic": len(tr), "n_train_real_per_epoch": len(tr_all) - len(tr),
                        "params": count_params(model), "torch": torch.__version__}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[train] {cfg['name']}: {len(tr_all)} train / {len(va)} val · {count_params(model):,} params · {device}")

    best = float("inf"); t0 = time.time(); rows = []
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])
        for ep in range(1, epochs + 1):
            model.train(); tot = 0.0; n = 0
            for x, y, m in tl:
                x, y, m = x.to(device), y.to(device), m.to(device)
                loss = compute_loss(model(x), y, m, tcfg)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tot += loss.item() * x.shape[0]; n += x.shape[0]
            train_loss = tot / max(n, 1); val_loss = evaluate_loader(model, vl, device, tcfg)
            lr = opt.param_groups[0]["lr"]; el = time.time() - t0
            rows.append((ep, train_loss, val_loss, lr, el)); w.writerow([ep, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.2e}", f"{el:.0f}"]); f.flush()
            state = {"model": model.state_dict(), "config": cfg, "epoch": ep, "val_loss": val_loss}
            torch.save(state, out / "last.pt")
            # select "best" (lowest synthetic validation loss) or "last" (fixed schedule — used for the
            # fine-tuning stage, where the synthetic validation loss alone does not measure the
            # adaptation to experimental data)
            if tcfg.get("select", "best") == "last":
                if ep == epochs:
                    best = val_loss; torch.save(state, out / "best.pt")
            elif val_loss < best:
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
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
