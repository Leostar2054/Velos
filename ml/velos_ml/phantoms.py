"""
Synthetic label-free cell phantoms (complex transmission functions).

Each phantom is O(r) = A(r)·exp(iφ(r)) on an N×N grid:
* background medium: A = 1, φ = 0 (+ a weak smooth phase background);
* 3–8 adherent-cell-like ellipses with dome-shaped phase (thickness profile),
  a denser nucleus, weak absorption and fine intracellular texture;
* optional cell-cell contact/overlap and random orientation.

Phase scale is in radians at 630 nm (typical live cells: 0.5–3 rad), matching the
Velos QPI operating regime. Everything is seeded for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via FFT (no scipy dependency; same approach as the Client's core)."""
    n0, n1 = a.shape
    fy = np.fft.fftfreq(n0)[:, None]
    fx = np.fft.fftfreq(n1)[None, :]
    H = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * H))


@dataclass
class PhantomConfig:
    size: int = 256
    n_cells: tuple = (3, 8)
    radius_px: tuple = (14, 40)          # semi-axis range
    aspect: tuple = (0.55, 1.0)
    phase_max: tuple = (0.6, 2.6)        # rad
    nucleus_phase: tuple = (0.3, 1.0)    # extra rad
    absorption: tuple = (0.03, 0.15)     # amplitude drop inside cells
    texture_rad: tuple = (0.03, 0.12)
    background_phase: tuple = (0.0, 0.25)
    edge_sigma: float = 1.0


def make_phantom(seed: int, cfg: PhantomConfig | None = None) -> np.ndarray:
    cfg = cfg or PhantomConfig()
    rng = np.random.default_rng(seed)
    N = cfg.size
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    phase = np.zeros((N, N)); absorb = np.zeros((N, N)); cellmask = np.zeros((N, N), bool)

    n_cells = rng.integers(cfg.n_cells[0], cfg.n_cells[1] + 1)
    for _ in range(n_cells):
        a = rng.uniform(*cfg.radius_px); b = a * rng.uniform(*cfg.aspect)
        th = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(a, N - a), rng.uniform(a, N - a)
        xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        rho2 = (xr / a) ** 2 + (yr / b) ** 2
        inside = rho2 < 1.0
        dome = np.sqrt(np.clip(1.0 - rho2, 0, 1))                 # hemispherical thickness
        pmax = rng.uniform(*cfg.phase_max)
        phase = np.maximum(phase, pmax * dome)                     # overlapping cells: max thickness
        # nucleus
        na, nb = a * rng.uniform(0.25, 0.45), b * rng.uniform(0.25, 0.45)
        ncx, ncy = rng.uniform(-0.3, 0.3) * a, rng.uniform(-0.3, 0.3) * b
        nrho2 = ((xr - ncx) / na) ** 2 + ((yr - ncy) / nb) ** 2
        phase += rng.uniform(*cfg.nucleus_phase) * np.sqrt(np.clip(1 - nrho2, 0, 1))
        absorb = np.maximum(absorb, rng.uniform(*cfg.absorption) * dome)
        cellmask |= inside
    # intracellular texture (organelles / granularity)
    tex = _smooth(rng.standard_normal((N, N)), rng.uniform(1.5, 3.0))
    tex = tex / (tex.std() + 1e-12) * rng.uniform(*cfg.texture_rad)
    phase += tex * cellmask
    # weak smooth background phase (medium / substrate)
    bg = _smooth(rng.standard_normal((N, N)), 25.0)
    bg = bg / (np.abs(bg).max() + 1e-12) * rng.uniform(*cfg.background_phase)
    phase += bg
    # soften edges slightly (finite resolution of real cells)
    phase = _smooth(phase, cfg.edge_sigma)
    absorb = _smooth(absorb, cfg.edge_sigma)
    amp = np.clip(1.0 - absorb, 0.6, 1.0)
    return (amp * np.exp(1j * phase)).astype(np.complex128)


def make_probe_variant(Np: int, seed: int, base_fn) -> np.ndarray:
    """Perturb the Client's probe model (aperture radius / defocus) for robustness."""
    rng = np.random.default_rng(seed)
    if rng.uniform() < 0.5:
        return base_fn(Np)
    u, v = np.meshgrid(np.linspace(-1, 1, Np), np.linspace(-1, 1, Np))
    r = np.hypot(u, v)
    rad = rng.uniform(0.5, 0.7); defocus = rng.uniform(1.0, 3.5)
    ap = 1.0 / (1.0 + np.exp((r - rad) / 0.035))
    P = ap * np.exp(1j * defocus * r ** 2)
    return (P / np.abs(P).max()).astype(np.complex128)
