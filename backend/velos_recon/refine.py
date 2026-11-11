"""
ML refinement at inference time — the runtime used by the backend worker.

It needs only NumPy plus ONE inference runtime (``onnxruntime`` preferred, CPU or
CUDA; TorchScript via ``torch`` as fallback). No training code is imported.

Model package (a directory, produced by ``python -m velos_ml.package``)::

    model.onnx         input "x" float32 [1, 3, T, T] (Re, Im, mask) → output "y" [1, 2, T, T]
    model.ts           optional TorchScript equivalent
    model_card.json    name, version, tile size T, overlap, normalisation, training data,
                       evaluation summary, intended use and limitations, file hashes

Pipeline (:func:`refine`)::

    physics object ──normalise (ramp, median scale/phase)──► z ──tiles T×T, Hann-blended──►
    model ──► z' ──denormalise──► refined object in the physics result's own frame

:func:`hybrid` then runs a short physics "polish" seeded with the refined object (and the
physics probe) so that the final estimate is again consistent with the measured data.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .normalise import NormParams, denormalise, normalise


@dataclass
class RefineOutput:
    obj: np.ndarray          # refined object, physics frame (outside mask: physics values)
    z: np.ndarray            # refined object, normalised frame (0 outside mask)
    z_in: np.ndarray         # normalised physics input (for side-by-side display)
    params: NormParams
    runtime: str
    elapsed_s: float
    card: dict


class ModelPackage:
    """A loaded model package with a uniform ``predict(x[1,3,T,T]) -> y[1,2,T,T]``."""

    def __init__(self, path: str | Path, runtime: str = "auto"):
        self.path = Path(path)
        card_file = self.path / "model_card.json"
        if not card_file.exists():
            raise FileNotFoundError(f"no model_card.json in {self.path}")
        self.card = json.loads(card_file.read_text())
        self.tile = int(self.card.get("tile", 256))
        self.overlap = int(self.card.get("tile_overlap", self.tile // 4))
        self.norm = self.card.get("normalisation", "median_ramp")
        self.runtime, self._run = self._load(runtime)

    # ------------------------------------------------------------------ runtimes
    def _load(self, runtime: str):
        onnx_file, ts_file = self.path / "model.onnx", self.path / "model.ts"
        errors = []
        if runtime in ("auto", "onnxruntime") and onnx_file.exists():
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = int(os.getenv("VELOS_ML_THREADS", "0")) or (os.cpu_count() or 1)
                providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                             if p in ort.get_available_providers()]
                sess = ort.InferenceSession(str(onnx_file), so, providers=providers)
                name_in = sess.get_inputs()[0].name
                used = sess.get_providers()[0]
                return f"onnxruntime ({used})", lambda x: sess.run(None, {name_in: x})[0]
            except Exception as e:  # pragma: no cover - fall through to TorchScript
                errors.append(f"onnxruntime: {e}")
        if runtime in ("auto", "torchscript") and ts_file.exists():
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                ts = torch.jit.load(str(ts_file), map_location=dev).eval()

                def run(x):
                    with torch.no_grad():
                        return ts(torch.from_numpy(x).to(dev)).cpu().numpy()
                return f"torchscript ({dev})", run
            except Exception as e:  # pragma: no cover
                errors.append(f"torchscript: {e}")
        raise RuntimeError(f"no usable runtime for {self.path}: {errors or 'model files missing'}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._run(np.ascontiguousarray(x, dtype=np.float32))

    # ------------------------------------------------------------------ tiling
    def _starts(self, n: int) -> list:
        t, step = self.tile, self.tile - self.overlap
        if n <= t:
            return [0]
        s = list(range(0, n - t, step)) + [n - t]
        return sorted(set(s))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """x[3,H,W] → y[2,H,W] with fixed T×T tiles and Hann-window blending.

        Fields smaller than T are zero-padded (mask channel 0 = "not measured", exactly
        how unmeasured pixels look in training)."""
        _, H, W = x.shape
        t = self.tile
        Hp, Wp = max(H, t), max(W, t)
        if (Hp, Wp) != (H, W):
            xp = np.zeros((3, Hp, Wp), np.float32); xp[:, :H, :W] = x
        else:
            xp = x.astype(np.float32, copy=False)
        w1 = np.sin(np.pi * (np.arange(t) + 0.5) / t) ** 2          # strictly positive Hann
        w2 = np.outer(w1, w1).astype(np.float32)
        out = np.zeros((2, Hp, Wp), np.float32); wsum = np.zeros((Hp, Wp), np.float32)
        for r0 in self._starts(Hp):
            for c0 in self._starts(Wp):
                tile = xp[None, :, r0:r0 + t, c0:c0 + t]
                if tile[0, 2].max() <= 0:                            # nothing measured here
                    y = tile[0, :2]
                else:
                    y = self.predict(tile)[0]
                out[:, r0:r0 + t, c0:c0 + t] += y * w2; wsum[r0:r0 + t, c0:c0 + t] += w2
        return (out / wsum)[:, :H, :W]


@lru_cache(maxsize=4)
def load_package(path: str, runtime: str = "auto") -> ModelPackage:
    return ModelPackage(path, runtime)


def list_packages(model_dir: str | Path) -> list:
    """Model cards of every package under ``model_dir`` (one sub-directory per model)."""
    root = Path(model_dir)
    out = []
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if (p / "model_card.json").exists()):
        card = json.loads((d / "model_card.json").read_text())
        card["id"] = d.name
        card["files"] = sorted(f.name for f in d.iterdir() if f.is_file())
        out.append(card)
    return out


def refine(obj: np.ndarray, mask: np.ndarray, package: ModelPackage | str | Path) -> RefineOutput:
    """Apply the refinement model to a physics reconstruction (any size)."""
    pkg = package if isinstance(package, ModelPackage) else load_package(str(package))
    t0 = time.perf_counter()
    mask = np.asarray(mask, bool)
    if pkg.norm != "median_ramp":
        raise ValueError(f"package normalisation '{pkg.norm}' not supported by this runtime")
    z_in, params = normalise(obj, mask, ramp=True)
    x = np.stack([z_in.real, z_in.imag, mask.astype(np.float64)], 0).astype(np.float32)
    y = pkg.apply(x)
    z = np.where(mask, y[0] + 1j * y[1], 0.0)
    refined = np.where(mask, denormalise(z, params), obj)
    return RefineOutput(obj=refined, z=z, z_in=z_in, params=params, runtime=pkg.runtime,
                        elapsed_s=time.perf_counter() - t0, card=pkg.card)


def hybrid(ds, physics_result, refined_obj: np.ndarray, cfg, iters: int,
           progress=None, should_cancel=None):
    """Physics polish seeded with the ML-refined object and the physics probe."""
    from .engine import reconstruct
    cfg2 = dataclasses.replace(cfg, iters=int(iters), progress_every=max(1, int(iters) // 20))
    return reconstruct(ds, cfg2, progress=progress, should_cancel=should_cancel,
                       init_object=refined_obj, init_probe=physics_result.probe)
