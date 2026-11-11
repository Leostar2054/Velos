"""
Engine driver: runs the Client's validated reconstruction engines on a
:class:`~velos_recon.data.PtychoDataset` with progress reporting.

The per-iteration update rules live in :mod:`velos_recon.core` and are used
exactly as written (``core.ENGINES[name].step_fn``). This module only adds the
"plumbing" a service needs: geometry conversion, probe/object initialisation,
progress callbacks, cancellation, timing and a serialisable result object.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional

import numpy as np

from . import core
from .data import PtychoDataset

ENGINE_NAMES = {
    "wasp": "WASP",
    "rpie": "rPIE",
    "epie": "ePIE",
    "gd": "GD",
    "map": "MAP",
}

# Default hyper-parameters per engine (kwargs forwarded to the core step fn).
ENGINE_DEFAULTS: Dict[str, dict] = {
    # probe_lr=1.0 (the Client's paper default) oscillates when the number of positions is
    # small or the probe is uncertain (observed on the 635 nm clam data); 0.1 was stable and
    # more accurate in all our checks (see docs/data-review.md), so it is the service default.
    "wasp": {"probe_lr": 0.1},
    "rpie": {"a_obj": 0.15, "a_prb": 0.15, "beta": 0.5},
    # alpha=beta=1 (Maiden & Rodenburg 2009 defaults) converged on the 635 nm clam data within
    # 40-60 iterations; the batch "wasp" engine stalled on that data (see docs/data-review.md).
    "epie": {"alpha": 1.0, "beta": 1.0},
    "gd":   {"probe_lr": 0.3},
    "map":  {"lr": 1.5e-2, "probe_lr": 1.5e-3, "mode": "stochastic", "anneal": True},
}


@dataclass
class ReconConfig:
    engine: str = "epie"
    iters: int = 100
    seed: int = 0
    crop: Optional[int] = None          # centre-crop diffraction frames (fast preview)
    engine_params: dict = field(default_factory=dict)
    probe_init: str = "dataset"         # "dataset" (use init_probe if present) | "disk"
    probe_disk_fraction: float = 0.5    # disk diameter as a fraction of the frame
    probe_chirp: str | float = "auto"   # "auto" | "none" | z (m): seed exp(+iπr²/λz) on the probe
    mask_threshold: float = 0.05        # coverage mask: denO > thr * max(denO)
    progress_every: int = 1

    def resolved_engine_params(self) -> dict:
        p = dict(ENGINE_DEFAULTS.get(self.engine, {}))
        p.update(self.engine_params or {})
        return p


@dataclass
class ReconResult:
    obj: np.ndarray                 # complex object estimate [H, W]
    probe: np.ndarray               # complex probe estimate [Np, Np]
    mask: np.ndarray                # bool coverage mask [H, W]
    positions_px: np.ndarray        # [J, 2] (row, col) top-left frame offsets
    dx_m: float                     # sample-plane pixel size (m)
    history: dict                   # per-iteration 'err', 'time' (+ 'nrmse' if ground truth)
    config: dict
    dataset_summary: dict
    elapsed_s: float

    def to_npz(self, path) -> None:
        np.savez_compressed(path, obj=self.obj.astype(np.complex64),
                            probe=self.probe.astype(np.complex64), mask=self.mask,
                            positions_px=self.positions_px, dx_m=self.dx_m,
                            err=np.asarray(self.history.get("err", [])),
                            time=np.asarray(self.history.get("time", [])),
                            nrmse=np.asarray(self.history.get("nrmse", [])))

    def summary(self) -> dict:
        err = self.history.get("err", [])
        out = {
            "engine": self.config.get("engine"),
            "iters": self.config.get("iters"),
            "elapsed_s": round(self.elapsed_s, 2),
            "object_shape": list(self.obj.shape),
            "probe_shape": list(self.probe.shape),
            "sample_pixel_um": self.dx_m * 1e6,
            "final_data_error": float(err[-1]) if len(err) else None,
            "initial_data_error": float(err[0]) if len(err) else None,
        }
        if len(self.history.get("nrmse", [])):
            out["final_nrmse"] = float(self.history["nrmse"][-1])
        return out


class ReconCancelled(RuntimeError):
    pass


def data_error(obj: np.ndarray, probe: np.ndarray, ds: PtychoDataset,
               index: Optional[np.ndarray] = None, crop: Optional[int] = None) -> float:
    """Normalised data-consistency error  Σ(|F(P·O_j)| − √I_j)² / Σ I_j  over frames ``index``
    (all frames by default), using the engines' own forward model (``core.FT``).

    It needs no ground truth, so it can score any object estimate — physics, ML-refined or
    hybrid — against the measurements. Evaluated on frames that were *not* used for the
    reconstruction it becomes a cross-validation (prediction) error.
    """
    if crop:
        ds = ds.center_crop(int(crop))
    amps = ds.amplitudes()
    pos = ds.positions_px()
    idx = np.arange(len(pos)) if index is None else np.asarray(index)
    npix = probe.shape[0]
    num = den = 0.0
    for k in idx:
        r0, c0 = pos[k]
        patch = obj[r0:r0 + npix, c0:c0 + npix]
        if patch.shape != probe.shape:
            continue
        a = np.abs(core.FT(probe * patch))
        num += float(np.sum((a - amps[k]) ** 2)); den += float(np.sum(amps[k] ** 2))
    return num / max(den, 1e-30)


ProgressCB = Callable[[int, int, dict], None]


def _init_probe(ds: PtychoDataset, cfg: ReconConfig) -> np.ndarray:
    m, n = ds.frame_shape
    if cfg.probe_init == "dataset" and ds.init_probe is not None:
        p0 = np.asarray(ds.init_probe)
        if p0.shape != (m, n):
            raise ValueError(f"init_probe shape {p0.shape} does not match frame {(m, n)}")
        p0 = p0.astype(np.complex128)
    else:
        yy, xx = np.mgrid[0:m, 0:n]
        r = np.hypot(yy - m / 2 + 0.5, xx - n / 2 + 0.5)
        p0 = (r <= cfg.probe_disk_fraction * min(m, n) / 2).astype(np.complex128)
    p0 = _apply_probe_chirp(p0, ds, cfg)
    # Power normalisation (as in the Client's MATLAB WASP pre-processing): with the
    # unitary FT used in core, sum|P|^2 should match the brightest pattern's photons.
    bright = float(ds.dps.sum(axis=(1, 2)).max())
    power = float(np.sum(np.abs(p0) ** 2))
    if power <= 0:
        raise ValueError("initial probe has zero power")
    return p0 * math.sqrt(bright / power)


def probe_fresnel_number(p0: np.ndarray, ds: PtychoDataset) -> float:
    """N_F = a² / (λ z) for the probe support radius a (from the initial probe)."""
    support = np.abs(p0) > 0.5 * np.abs(p0).max()
    a = math.sqrt(support.sum() / math.pi) * ds.dx
    return a * a / (ds.wavelength * ds.distance)


def _apply_probe_chirp(p0: np.ndarray, ds: PtychoDataset, cfg: ReconConfig) -> np.ndarray:
    """Seed the quadratic (Fresnel) phase on the initial probe when the detector is not in
    the far field of the probe aperture.

    Physics: Fresnel propagation over z equals a Fourier transform of ψ(r)·exp(iπr²/λz) up to
    a detector-plane phase that intensity measurements do not see. The far-field FFT model
    used by the engines is therefore exact *provided the probe carries this chirp*. Starting
    from a flat aperture the algorithms must discover it, which stalls or takes hundreds of
    iterations (observed on the 635 nm clam data, N_F ≈ 1.6: flat start → featureless object;
    chirped start → tissue structure within 40 iterations). "auto" applies the chirp when
    N_F > 0.3; a float sets the distance z explicitly; "none" disables it.
    """
    mode = cfg.probe_chirp
    if mode == "none" or mode is None:
        return p0
    if mode == "auto":
        if probe_fresnel_number(p0, ds) <= 0.3:
            return p0
        z = ds.distance
    else:
        z = float(mode)
    m, n = p0.shape
    yy, xx = np.mgrid[0:m, 0:n]
    r2 = ((yy - m / 2) ** 2 + (xx - n / 2) ** 2) * ds.dx ** 2
    return p0 * np.exp(1j * math.pi * r2 / (ds.wavelength * z))


def reconstruct(ds: PtychoDataset, cfg: ReconConfig,
                progress: Optional[ProgressCB] = None,
                should_cancel: Optional[Callable[[], bool]] = None,
                init_object: Optional[np.ndarray] = None,
                init_probe: Optional[np.ndarray] = None) -> ReconResult:
    """Run ``cfg.iters`` iterations of the chosen engine on ``ds``.

    ``init_object`` / ``init_probe`` (complex, object-frame and frame-sized) start the
    engine from a given estimate instead of the Client's default initialisation — used by
    the hybrid mode, where the ML-refined object seeds a short physics "polish" that
    restores consistency with the measured data. The update rules are unchanged.
    """
    if cfg.engine not in ENGINE_NAMES:
        raise ValueError(f"unknown engine '{cfg.engine}'; choose from {list(ENGINE_NAMES)}")
    if cfg.crop:
        ds = ds.center_crop(int(cfg.crop))
    m, n = ds.frame_shape
    if m != n:                       # core.FT normalises with the last axis; keep frames square
        ds = ds.center_crop(min(m, n))

    amps = ds.amplitudes()                               # [J, Np, Np]
    pos = ds.positions_px()                              # [J, 2] (row, col)
    shape = ds.object_shape()
    P0 = _init_probe(ds, cfg)
    O, P = core._init(shape, P0, cfg.seed)               # Client's initialisation
    if init_probe is not None:
        if np.shape(init_probe) != P.shape:
            raise ValueError(f"init_probe shape {np.shape(init_probe)} != frame {P.shape}")
        P = np.array(init_probe, dtype=np.complex128)
    if init_object is not None:
        if np.shape(init_object) != tuple(shape):
            raise ValueError(f"init_object shape {np.shape(init_object)} != object {tuple(shape)}")
        O = np.array(init_object, dtype=np.complex128)
    den = core.compute_denO(P0, pos, shape)
    mask = den > cfg.mask_threshold * den.max()

    step = core.ENGINES[ENGINE_NAMES[cfg.engine]].step_fn   # validated update rule
    params = cfg.resolved_engine_params()
    state = {"n_total": cfg.iters}
    hist = {"err": [], "time": [], "nrmse": []}
    gt = ds.ground_truth
    t0 = time.perf_counter()
    for it in range(1, cfg.iters + 1):
        if should_cancel is not None and should_cancel():
            raise ReconCancelled(f"cancelled at iteration {it}")
        O, P, err = step(O, P, amps, pos, it, state, **params)
        hist["err"].append(float(err))
        hist["time"].append(time.perf_counter() - t0)
        if gt is not None and gt.shape == O.shape:
            hist["nrmse"].append(core.nrmse(O, gt, mask))
        if progress is not None and (it % cfg.progress_every == 0 or it == cfg.iters):
            progress(it, cfg.iters, {"err": float(err), "elapsed_s": hist["time"][-1]})
    elapsed = time.perf_counter() - t0
    if not hist["nrmse"]:
        hist.pop("nrmse")
    return ReconResult(obj=O, probe=P, mask=mask, positions_px=pos, dx_m=ds.dx,
                       history=hist, config=asdict(cfg), dataset_summary=ds.summary(),
                       elapsed_s=elapsed)
