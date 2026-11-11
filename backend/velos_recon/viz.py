"""Figure / PNG export helpers (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def crop_to_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 0) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, arr.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + 1 + pad, arr.shape[1])
    return arr[r0:r1, c0:c1]


def _robust_range(x: np.ndarray, lo=1.0, hi=99.0):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def save_object_images(obj: np.ndarray, mask: np.ndarray, out_dir: Path,
                       prefix: str = "", dx_um: Optional[float] = None,
                       title: Optional[str] = None) -> dict:
    """Write amplitude.png, phase.png and preview.png; return the file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    o = crop_to_mask(obj, mask)
    m = crop_to_mask(mask, mask)
    amp = np.abs(o); pha = np.angle(o)
    amp = np.where(m, amp, np.nan); pha = np.where(m, pha, np.nan)
    files = {}
    for name, img, cmap in (("amplitude", amp, "gray"), ("phase", pha, "twilight")):
        vmin, vmax = _robust_range(img[np.isfinite(img)])
        fig, ax = plt.subplots(figsize=(6, 6 * img.shape[0] / img.shape[1]))
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_axis_off()
        if dx_um:
            _scalebar(ax, img.shape[1], dx_um)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("amplitude (a.u.)" if name == "amplitude" else "phase (rad)")
        fig.tight_layout()
        p = out_dir / f"{prefix}{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files[name] = str(p)
    # side-by-side preview
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (name, img, cmap) in zip(axs, (("Amplitude", amp, "gray"), ("Phase (rad)", pha, "twilight"))):
        vmin, vmax = _robust_range(img[np.isfinite(img)])
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name); ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    p = out_dir / f"{prefix}preview.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    files["preview"] = str(p)
    return files


def save_probe_image(probe: np.ndarray, out_dir: Path, prefix: str = "") -> str:
    out_dir = Path(out_dir)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    im0 = axs[0].imshow(np.abs(probe), cmap="inferno"); axs[0].set_title("Probe amplitude")
    im1 = axs[1].imshow(np.angle(probe), cmap="twilight"); axs[1].set_title("Probe phase (rad)")
    for ax, im in zip(axs, (im0, im1)):
        ax.set_axis_off(); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    p = out_dir / f"{prefix}probe.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return str(p)


def save_convergence(history: dict, out_dir: Path, prefix: str = "") -> str:
    out_dir = Path(out_dir)
    err = np.asarray(history.get("err", []))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogy(np.arange(1, len(err) + 1), err / max(err[0], 1e-30), lw=2)
    ax.set_xlabel("iteration"); ax.set_ylabel("data error (normalised)")
    ax.grid(alpha=0.3)
    if "nrmse" in history and len(history["nrmse"]):
        ax2 = ax.twinx(); ax2.plot(np.arange(1, len(err) + 1), history["nrmse"], color="C3", lw=1.5)
        ax2.set_ylabel("NRMSE vs ground truth", color="C3")
    fig.tight_layout()
    p = out_dir / f"{prefix}convergence.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return str(p)


def _scalebar(ax, width_px: int, dx_um: float):
    bar_um = 10 ** np.floor(np.log10(width_px * dx_um / 5))
    bar_px = bar_um / dx_um
    x0, y0 = width_px * 0.05, ax.get_ylim()[0] * 0.95
    ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=3)
    ax.text(x0, y0 - width_px * 0.02, f"{bar_um:g} µm", color="white", fontsize=9)


def save_comparison(panels, mask: np.ndarray, path, dx_um: Optional[float] = None) -> str:
    """Side-by-side amplitude (top) and phase (bottom) of several complex estimates on a
    shared colour scale per row (range taken from the first panel), e.g. physics / ML / hybrid."""
    m = crop_to_mask(mask, mask)
    imgs = [(name, crop_to_mask(z, mask)) for name, z in panels]
    n = len(imgs)
    h, w = m.shape
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 7.2 * h / w + 0.6), squeeze=False)
    ref_amp = np.abs(imgs[0][1])[m]; ref_pha = np.angle(imgs[0][1])[m]
    ranges = (_robust_range(ref_amp), _robust_range(ref_pha))
    for j, (name, z) in enumerate(imgs):
        for i, (img, cmap) in enumerate(((np.abs(z), "gray"), (np.angle(z), "twilight"))):
            ax = axs[i, j]
            ax.imshow(np.where(m, img, np.nan), cmap=cmap, vmin=ranges[i][0], vmax=ranges[i][1],
                      interpolation="nearest")
            ax.set_axis_off()
            if i == 0:
                ax.set_title(name, fontsize=11)
            if dx_um and j == 0:
                _scalebar(ax, w, dx_um)
    axs[0, 0].text(-0.04, 0.5, "amplitude", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right")
    axs[1, 0].text(-0.04, 0.5, "phase", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)
