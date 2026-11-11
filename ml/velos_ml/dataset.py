"""
PyTorch datasets.

* :class:`SparseRefineDataset` — synthetic shards from ``velos_ml.simulate``
  (input = physics reconstruction of a reduced scan, target = ground truth).
* :class:`RealPatchDataset` — experimental scans from ``velos_ml.realdata``
  (input = reconstruction from a random subset of frames, target = reconstruction from
  all frames), restricted to a spatial training region so that the remaining columns
  stay unseen for testing.

Two input conventions exist (recorded in every checkpoint's config as ``data.norm``):

* ``"median"`` (M2): median amplitude → 1, median phase → 0; target = ground truth as is.
* ``"median_ramp"`` (M3): additionally removes the linear phase ramp
  (:func:`velos_recon.normalise.normalise`), and the target is expressed in the *input's*
  frame (``data.target_frame = "input"``: ramp + complex scalar fitted to the input), so the
  network only learns local corrections, never the unobservable global ambiguities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from velos_recon.metrics import align, estimate_shift, shift_image
from velos_recon.normalise import legacy_median_normalise, normalise


def erode(mask: np.ndarray, r: int = 4) -> np.ndarray:
    """Binary erosion by a (2r+1)² square (pure numpy)."""
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out &= np.roll(np.roll(mask, dy, 0), dx, 1)
    out[:r, :] = out[-r:, :] = out[:, :r] = out[:, -r:] = False
    return out


def prepare_input(obj: np.ndarray, mask: np.ndarray, norm: str = "median_ramp"):
    """Complex object → (x[3,H,W] float32 = Re, Im, mask; normalised complex z; params)."""
    obj = np.asarray(obj, dtype=np.complex128); mask = np.asarray(mask, dtype=bool)
    if norm == "median":
        z, params = legacy_median_normalise(obj, mask), None
    elif norm == "median_ramp":
        z, params = normalise(obj, mask, ramp=True)
    else:
        raise ValueError(f"unknown normalisation '{norm}'")
    x = np.stack([z.real, z.imag, mask.astype(np.float64)], 0).astype(np.float32)
    return x, z, params


def to_channels(obj: np.ndarray, mask: np.ndarray, norm: str = "median") -> np.ndarray:
    """Backwards-compatible helper (M2 default normalisation)."""
    return prepare_input(obj, mask, norm)[0]


def target_in_frame(ref: np.ndarray, z: np.ndarray, mask: np.ndarray, frame: str,
                    register: bool = True) -> np.ndarray:
    """Express the reference (ground truth or full-scan reconstruction) as the training target.

    ``frame="input"`` (M3): the reference is moved into the input's frame — sub-pixel translation
    (``register``), linear phase ramp and complex scalar — so the network only learns local
    corrections and never the unobservable global ambiguities. (ePIE reconstructions are offset
    from the ground truth by up to ~1 px even when started from the true probe.)"""
    ref = np.asarray(ref, dtype=np.complex128)
    if frame == "absolute":
        return ref
    if frame == "input":
        if register:
            ref = shift_image(ref, estimate_shift(ref, z, mask))
        return align(ref, z, mask, ramp=True)
    raise ValueError(f"unknown target frame '{frame}'")


def _prepare_synthetic(args):
    f, norm, frame = args
    d = np.load(f)
    x, z, _ = prepare_input(d["sparse_recon"], d["sparse_mask"], norm)
    m = d["eval_mask"]
    t = target_in_frame(d["gt"], z, m, frame)
    y = np.stack([t.real, t.imag], 0).astype(np.float32)
    return x, y, m[None].astype(np.float32)


def _augment(rng, *arrs):
    k = int(rng.integers(0, 4)); flip = rng.uniform() < 0.5
    out = []
    for a in arrs:
        a = np.rot90(a, k, (1, 2))
        if flip:
            a = a[:, :, ::-1]
        out.append(np.ascontiguousarray(a))
    return out


class SparseRefineDataset(Dataset):
    """Synthetic pairs. Returns (x[3,H,W], y[2,H,W], eval_mask[1,H,W]).

    Training uses random crops and flips/rotations; metrics are evaluated on the full field.
    With ``preload`` the normalised arrays are prepared once and kept in memory.
    """

    def __init__(self, root: str | Path, split: str, crop: Optional[int] = None,
                 augment: bool = False, limit: Optional[int] = None, seed: int = 0,
                 norm: str = "median", target_frame: str = "absolute", preload: bool = False,
                 workers: int = 2):
        self.files = sorted(Path(root, split).glob("sample_*.npz"))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"no samples under {Path(root, split)}")
        self.crop, self.augment = crop, augment
        self.norm, self.target_frame = norm, target_frame
        self.rng = np.random.default_rng(seed)
        self._cache = self._preload(Path(root), split, workers) if preload else None

    def __len__(self):
        return len(self.files)

    def _preload(self, root: Path, split: str, workers: int):
        """Prepare every pair once (normalisation + target registration take ~1-2 s each) and
        keep a disk cache next to the data so later runs / stages start immediately."""
        cache = root / f"_prepared_{split}_{self.norm}_{self.target_frame}_{len(self.files)}.npz"
        if cache.exists():
            c = np.load(cache)
            return list(zip(c["x"], c["y"], c["m"]))
        jobs = [(f, self.norm, self.target_frame) for f in self.files]
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                items = list(ex.map(_prepare_synthetic, jobs, chunksize=4))
        else:
            items = [_prepare_synthetic(j) for j in jobs]
        np.savez(cache, x=np.stack([i[0] for i in items]), y=np.stack([i[1] for i in items]),
                 m=np.stack([i[2] for i in items]))
        return items

    def _prepare(self, i):
        return _prepare_synthetic((self.files[i], self.norm, self.target_frame))

    def __getitem__(self, i):
        x, y, m = self._cache[i] if self._cache is not None else self._prepare(i)
        if self.crop and self.crop < x.shape[-1]:
            c = self.crop; H, W = x.shape[-2:]
            r0 = int(self.rng.integers(0, H - c + 1)); c0 = int(self.rng.integers(0, W - c + 1))
            x, y, m = x[:, r0:r0 + c, c0:c0 + c], y[:, r0:r0 + c, c0:c0 + c], m[:, r0:r0 + c, c0:c0 + c]
        if self.augment:
            x, y, m = _augment(self.rng, x, y, m)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m)))

    def raw(self, i) -> dict:
        return dict(np.load(self.files[i]))


class RealPatchDataset(Dataset):
    """Experimental pairs from ``velos_ml.realdata`` output.

    Only columns in ``col_range`` (fractions of the object width) are ever cropped, and the
    frame alignment of the target is fitted on that region only, so the remaining columns
    are a clean spatial hold-out. ``subsets`` selects which random frame draws are used.
    """

    def __init__(self, root: str | Path, subsets: Sequence[int], crop: int = 128,
                 col_range=(0.0, 0.65), n_per_epoch: int = 240, augment: bool = True,
                 seed: int = 0, norm: str = "median_ramp", target_frame: str = "input"):
        root = Path(root)
        dense = np.load(root / "dense.npz")
        D, MD = dense["obj"].astype(np.complex128), dense["mask"]
        H, W = D.shape
        self.c_lo, self.c_hi = int(col_range[0] * W), int(col_range[1] * W)
        region = np.zeros((H, W), bool); region[:, self.c_lo:self.c_hi] = True
        self.items = []
        for k in subsets:
            s = np.load(root / "subsets" / f"sub_{k:02d}.npz")
            x, z, _ = prepare_input(s["obj"], s["mask"], norm)
            m = erode(s["mask"] & MD, 4)
            # reduced-scan and full-scan reconstructions differ by a sub-pixel translation
            # (0.1-0.7 px on the clam data); register the target to the input first
            Dk = shift_image(D, estimate_shift(D, z, m & region))
            t = target_in_frame(Dk, z, m & region, target_frame)
            y = np.stack([t.real, t.imag], 0).astype(np.float32)
            self.items.append((x, y, (m & region)[None].astype(np.float32)))
        self.crop, self.n, self.augment = crop, n_per_epoch, augment
        self.rng = np.random.default_rng(seed)
        if self.c_hi - self.c_lo < crop or H < crop:
            raise ValueError("training region smaller than the crop")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x, y, m = self.items[int(self.rng.integers(0, len(self.items)))]
        c = self.crop; H = x.shape[-2]
        for _ in range(20):          # avoid crops that are mostly outside the coverage mask
            r0 = int(self.rng.integers(0, H - c + 1)); c0 = int(self.rng.integers(self.c_lo, self.c_hi - c + 1))
            mc = m[:, r0:r0 + c, c0:c0 + c]
            if mc.mean() > 0.3:
                break
        xc, yc = x[:, r0:r0 + c, c0:c0 + c], y[:, r0:r0 + c, c0:c0 + c]
        if self.augment:
            xc, yc, mc = _augment(self.rng, xc, yc, mc)
        return (torch.from_numpy(np.ascontiguousarray(xc)), torch.from_numpy(np.ascontiguousarray(yc)),
                torch.from_numpy(np.ascontiguousarray(mc)))
