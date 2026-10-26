"""
Inference utilities (used by the backend worker and the evaluation script).

    refine_object(obj, mask, checkpoint) -> complex ndarray   # same shape as obj
    export_torchscript(checkpoint, out_path)                  # packaged model for M3
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .dataset import to_channels
from .model import build_model


@lru_cache(maxsize=4)
def load_model(checkpoint: str, device: str = "cpu"):
    ckpt = Path(checkpoint)
    if ckpt.suffix == ".ts" or ckpt.name.endswith(".torchscript.pt"):
        model = torch.jit.load(str(ckpt), map_location=device)
        return model.eval(), {"torchscript": True}
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(state["config"])
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state["config"]


@torch.no_grad()
def refine_channels(x: np.ndarray, model, device: str = "cpu", tile: int = 512) -> np.ndarray:
    """x[3,H,W] → y[2,H,W]; large fields are processed in overlapping tiles."""
    H, W = x.shape[-2:]
    if max(H, W) <= tile:
        return model(torch.from_numpy(x[None]).to(device))[0].cpu().numpy()
    out = np.zeros((2, H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    ov = tile // 4
    for r0 in range(0, H, tile - ov):
        for c0 in range(0, W, tile - ov):
            r1, c1 = min(r0 + tile, H), min(c0 + tile, W)
            y = model(torch.from_numpy(x[None, :, r0:r1, c0:c1]).to(device))[0].cpu().numpy()
            out[:, r0:r1, c0:c1] += y; wsum[r0:r1, c0:c1] += 1
    return out / np.maximum(wsum, 1)


def refine_object(obj: np.ndarray, mask: np.ndarray, checkpoint: str, device: str = "cpu") -> np.ndarray:
    """Apply the refinement model to a physics reconstruction.

    The input is self-normalised (median amplitude → 1, median phase → 0 over ``mask``),
    so the output lives in the same normalised frame: background amplitude ≈ 1, phase ≈ 0.
    Pixels outside ``mask`` are returned as 0.
    """
    model, _ = load_model(checkpoint, device)
    x = to_channels(obj, mask)
    y = refine_channels(x, model, device)
    z = (y[0] + 1j * y[1]).astype(np.complex128)
    return np.where(mask, z, 0.0)


def export_torchscript(checkpoint: str, out_path: str, size: int = 256) -> str:
    model, cfg = load_model(checkpoint)
    example = torch.zeros(1, 3, size, size)
    ts = torch.jit.trace(model, example)
    ts.save(out_path)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("checkpoint"); ap.add_argument("--export", default=None)
    a = ap.parse_args()
    if a.export:
        print("wrote", export_torchscript(a.checkpoint, a.export))
