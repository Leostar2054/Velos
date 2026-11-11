"""
Self-normalisation of complex reconstructions (no ground truth needed).

Ptychography recovers the object only up to (i) a global complex scalar and (ii) a
linear phase ramp (the object-ramp / probe-shift ambiguity, strong for ePIE on raster
scans). The ML refinement model therefore works in a *normalised frame*:

    z = obj · exp(-i(kx·x + ky·y)) / (s · exp(iφ0))

with the ramp (kx, ky) estimated from the median of wrapped neighbour phase differences
(dominated by the flat background), s = median amplitude and φ0 = circular median phase
over the coverage mask. In this frame the background is ≈ 1 + 0i, whatever engine or
instrument produced the reconstruction.

:func:`denormalise` maps a model output back to the reconstruction's own frame, so it can
be written next to the physics result or used to seed further physics iterations
("hybrid" mode). Used by the ML pipeline (training data) and the backend (inference),
so both see exactly the same transform.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import estimate_phase_ramp


@dataclass
class NormParams:
    scale: float = 1.0      # median amplitude over the mask
    phase0: float = 0.0     # circular-median phase over the mask (rad), after ramp removal
    kx: float = 0.0         # removed linear phase ramp (rad / pixel) along columns
    ky: float = 0.0         # … along rows
    ramp: bool = True

    def to_dict(self) -> dict:
        return {k: float(v) if not isinstance(v, bool) else v for k, v in asdict(self).items()}


def _ramp(shape, kx: float, ky: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    return np.exp(1j * (kx * xx + ky * yy))


def _circular_median_phase(z: np.ndarray) -> float:
    """Median phase robust to wrapping: centre on the mean phasor direction first."""
    if z.size == 0:
        return 0.0
    c = float(np.angle(np.sum(z / (np.abs(z) + 1e-12))))
    return c + float(np.median(np.angle(z * np.exp(-1j * c))))


def normalise(obj: np.ndarray, mask: np.ndarray, ramp: bool = True):
    """Return (z, params): ``obj`` in the normalised frame (0 outside ``mask``)."""
    obj = np.asarray(obj, dtype=np.complex128)
    mask = np.asarray(mask, dtype=bool)
    kx = ky = 0.0
    if ramp and mask.sum() > 16:
        kx, ky = estimate_phase_ramp(obj, mask)
        obj = obj * np.conj(_ramp(obj.shape, kx, ky))
    vals = obj[mask]
    scale = float(np.median(np.abs(vals))) + 1e-12 if vals.size else 1.0
    phase0 = _circular_median_phase(vals)
    z = obj / scale * np.exp(-1j * phase0)
    return np.where(mask, z, 0.0), NormParams(scale, phase0, kx, ky, ramp)


def denormalise(z: np.ndarray, params: NormParams | dict) -> np.ndarray:
    """Inverse of :func:`normalise` (back to the reconstruction's own frame)."""
    p = params if isinstance(params, NormParams) else NormParams(**params)
    out = np.asarray(z, dtype=np.complex128) * p.scale * np.exp(1j * p.phase0)
    if p.ramp and (p.kx or p.ky):
        out = out * _ramp(out.shape, p.kx, p.ky)
    return out


def legacy_median_normalise(obj: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """M2 normalisation (median amplitude/phase only, no ramp) — kept so the M2 baseline
    checkpoint can be evaluated exactly as it was trained."""
    vals = obj[mask]
    scale = np.median(np.abs(vals)) + 1e-12
    phase0 = np.median(np.angle(vals))
    out = obj / scale * np.exp(-1j * phase0)
    return np.where(mask, out, 0.0)
