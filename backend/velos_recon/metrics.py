"""
Image-quality metrics for complex reconstructions.

All metrics first remove the global complex-scalar ambiguity of ptychography
(``rec ← c·rec`` with ``c = <rec, ref>/<rec, rec>`` over the valid mask, the
same alignment used by the Client's ``core.nrmse``), then evaluate amplitude
and phase separately inside the mask's bounding box.

* PSNR  – 10·log10(range² / MSE); range = peak-to-peak of the reference.
* SSIM  – scikit-image structural similarity (7×7 window); falls back to the
  Client's global SSIM formula if scikit-image is unavailable.
* NRMSE – complex normalised RMSE (Client's definition).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import core

try:  # optional dependency
    from skimage.metrics import structural_similarity as _ssim_sk
except Exception:  # pragma: no cover
    _ssim_sk = None


def psnr(x: np.ndarray, ref: np.ndarray, data_range: float | None = None) -> float:
    x = np.asarray(x, float); ref = np.asarray(ref, float)
    if data_range is None:
        data_range = float(np.ptp(ref)) or 1.0
    mse = float(np.mean((x - ref) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(x: np.ndarray, ref: np.ndarray, data_range: float | None = None) -> float:
    x = np.asarray(x, float); ref = np.asarray(ref, float)
    if data_range is None:
        data_range = float(np.ptp(ref)) or 1.0
    if _ssim_sk is not None and min(x.shape) >= 7:
        return float(_ssim_sk(x, ref, data_range=data_range, win_size=7))
    # global-statistics fallback (same form as the Client's core.ssim_amp)
    mx, my, vx, vy = x.mean(), ref.mean(), x.var(), ref.var()
    vxy = float(((x - mx) * (ref - my)).mean())
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return float(((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


def _bbox(mask: np.ndarray):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def estimate_phase_ramp(z: np.ndarray, mask: np.ndarray, n_iter: int = 3):
    """Robustly estimate a linear phase ramp exp(i(kx·x + ky·y)) in ``z`` over ``mask``.

    Uses the median of wrapped nearest-neighbour phase differences, so it is dominated
    by the (flat) background and insensitive to the steep phase edges of cells.
    Returns (kx, ky) in rad/pixel.
    """
    z = z.copy(); kx_tot = ky_tot = 0.0
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(float)
    for _ in range(n_iter):
        mx = mask[:, 1:] & mask[:, :-1]; my = mask[1:, :] & mask[:-1, :]
        gx = np.angle(z[:, 1:] * np.conj(z[:, :-1]))[mx]
        gy = np.angle(z[1:, :] * np.conj(z[:-1, :]))[my]
        if gx.size == 0 or gy.size == 0:
            break
        kx, ky = float(np.median(gx)), float(np.median(gy))
        z = z * np.exp(-1j * (kx * xx + ky * yy))
        kx_tot += kx; ky_tot += ky
    return kx_tot, ky_tot


def remove_phase_ramp(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kx, ky = estimate_phase_ramp(z, mask)
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(float)
    return z * np.exp(-1j * (kx * xx + ky * yy))


def align(rec: np.ndarray, ref: np.ndarray, mask: np.ndarray, ramp: bool = True) -> np.ndarray:
    """Return ``rec`` with the ptychographic ambiguities removed relative to ``ref``:
    the linear phase ramp (raster-scan/probe-shift ambiguity, if ``ramp``) and the
    global complex scalar (``c = <rec, ref>/<rec, rec>`` over ``mask``)."""
    if ramp:
        kx, ky = estimate_phase_ramp(rec * np.conj(ref), mask)
        yy, xx = np.mgrid[0:rec.shape[0], 0:rec.shape[1]].astype(float)
        rec = rec * np.exp(-1j * (kx * xx + ky * yy))
    c = core._align(rec, ref, mask)
    return c * rec


def estimate_shift(moving: np.ndarray, fixed: np.ndarray, mask: np.ndarray, upsample: int = 20,
                   refine: bool = True):
    """Sub-pixel translation (dy, dx) that registers ``moving`` onto ``fixed`` (complex images,
    compared inside ``mask``). Ptychographic reconstructions of the same scan from different frame
    subsets can be offset by a fraction of a pixel (joint object/probe translation ambiguity).

    Phase correlation gives the starting estimate; with ``refine`` the shift is then optimised
    directly for the agreement measured by :func:`core.nrmse` after removing the phase ramp and
    complex scalar (robust when the masked edges or noise bias the correlation peak).
    """
    from skimage.registration import phase_cross_correlation
    a = np.where(mask, fixed - fixed[mask].mean(), 0)
    b = np.where(mask, moving - moving[mask].mean(), 0)
    shift = phase_cross_correlation(a, b, upsample_factor=upsample, normalization=None)[0]
    if refine:
        from scipy.ndimage import fourier_shift
        from scipy.optimize import minimize
        F = np.fft.fft2(moving)

        def cost(sh):
            r = np.fft.ifft2(fourier_shift(F, sh))
            return core.nrmse(align(r, fixed, mask), fixed, mask)
        best = min((np.asarray(shift, float), np.zeros(2)), key=cost)       # never worse than no shift
        res = minimize(cost, best, method="Nelder-Mead",
                       options={"xatol": 0.02, "fatol": 1e-6, "maxiter": 80, "initial_simplex":
                                [best, best + [0.3, 0], best + [0, 0.3]]})
        shift = res.x if res.fun <= cost(best) else best
    return float(shift[0]), float(shift[1])


def shift_image(z: np.ndarray, shift) -> np.ndarray:
    """Fourier (sub-pixel) shift of a complex image by (dy, dx)."""
    from scipy.ndimage import fourier_shift
    out = np.fft.ifft2(fourier_shift(np.fft.fft2(z), shift))
    return out if np.iscomplexobj(z) else out.real


def amplitude_phase_metrics(rec: np.ndarray, ref: np.ndarray, mask: np.ndarray,
                            register: bool = False) -> Dict[str, float]:
    """Metrics of ``rec`` against ``ref`` (both complex, same shape) inside ``mask``.
    The linear phase ramp and global complex scalar — both unobservable in
    ptychography — are removed from ``rec`` first (see :func:`align`). With ``register``
    a sub-pixel translation is removed as well (used for experimental data, where the
    reference is another reconstruction)."""
    if register:
        rec = shift_image(rec, estimate_shift(rec, ref, mask))
    rec = align(rec, ref, mask)
    rs, cs = _bbox(mask)
    m = mask[rs, cs]
    a_rec, a_ref = np.abs(rec[rs, cs]), np.abs(ref[rs, cs])
    # phase difference wrapped to (-pi, pi]; evaluate phase relative to the reference
    dphi = np.angle(rec[rs, cs] * np.conj(ref[rs, cs]))
    p_ref = np.angle(ref[rs, cs])
    p_rec = p_ref + dphi
    # pixels outside the mask are not reconstructed: substitute the reference (zero error)
    a_rec = np.where(m, a_rec, a_ref)
    p_rec = np.where(m, p_rec, p_ref)
    out = {
        "nrmse_complex": core.nrmse(rec, ref, mask),
        "psnr_amplitude": psnr(a_rec, a_ref),
        "ssim_amplitude": ssim(a_rec, a_ref),
        "psnr_phase": psnr(p_rec, p_ref),
        "ssim_phase": ssim(p_rec, p_ref),
        "mae_phase_rad": float(np.mean(np.abs(dphi[m]))),
        "mae_amplitude": float(np.mean(np.abs(a_rec - a_ref)[m])),
    }
    return out
