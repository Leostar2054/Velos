"""PyTorch dataset over the generated .npz shards."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .simulate import normalise


def to_channels(obj: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Complex object → float32 channels [Re, Im, mask] after self-normalisation."""
    z = normalise(obj.astype(np.complex128), mask)
    return np.stack([z.real, z.imag, mask.astype(np.float64)], 0).astype(np.float32)


class SparseRefineDataset(Dataset):
    """Input: physics reconstruction from the sparse scan; target: ground truth.

    Returns (x[3,H,W], y[2,H,W], eval_mask[1,H,W]). Training uses random crops and
    flips/rotations (all metrics are evaluated on the full field).
    """

    def __init__(self, root: str | Path, split: str, crop: Optional[int] = None,
                 augment: bool = False, limit: Optional[int] = None, seed: int = 0):
        self.files = sorted(Path(root, split).glob("sample_*.npz"))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"no samples under {Path(root, split)}")
        self.crop, self.augment = crop, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x = to_channels(d["sparse_recon"], d["sparse_mask"])
        gt = d["gt"].astype(np.complex128)
        y = np.stack([gt.real, gt.imag], 0).astype(np.float32)
        m = d["eval_mask"][None].astype(np.float32)
        if self.crop and self.crop < x.shape[-1]:
            c = self.crop; H = x.shape[-1]
            r0, c0 = self.rng.integers(0, H - c + 1, 2)
            x, y, m = x[:, r0:r0 + c, c0:c0 + c], y[:, r0:r0 + c, c0:c0 + c], m[:, r0:r0 + c, c0:c0 + c]
        if self.augment:
            k = int(self.rng.integers(0, 4))
            x, y, m = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2)), np.rot90(m, k, (1, 2))
            if self.rng.uniform() < 0.5:
                x, y, m = x[:, :, ::-1], y[:, :, ::-1], m[:, :, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m)))

    def raw(self, i) -> dict:
        return dict(np.load(self.files[i]))
