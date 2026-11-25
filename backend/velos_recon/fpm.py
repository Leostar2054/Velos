"""
LED-array Fourier ptychographic microscopy (FPM) — AP / EPRY / rPIE.

Python port of the Client's MATLAB quick-check engine
``main_led_fpm_AP_EPRY_rPIE_CS2100M.m`` (version 2, 19 Nov 2026), written for the
in-house LED-panel FPM microscope (Thorlabs CS2100M-USB camera, 4×/0.10 objective,
15×15 LEDs at 3 mm pitch about 5-6 cm below the sample). The update rules, frame
order, spectrum windows, brightness factors, transition-frame detection, hot-pixel
repair and geometry check follow the MATLAB code line by line so that results can be
compared directly; the differences are only technical (NumPy/SciPy FFTs instead of
gpuArray, NPZ/TIFF output instead of .mat).

    psi = F^-1{O_m P},  psi' = psi*sqrt(I_m)/|psi|,  dF = F{psi'} - O_m P
    AP    O_m <- O_m + conj(P) dF / |P|^2max                        pupil fixed
    EPRY  as AP, and P <- P + conj(O_m) dF / |O_m|^2max             (support kept)
    rPIE  O_m <- O_m + conj(P) dF / ((1-a)|P|^2 + a|P|^2max)
          P   <- P + conj(O_m) dF / ((1-b)|O_m|^2 + b|O_m|^2max),  a = b = 0.5

Inputs
------
* an acquisition folder (``positions.csv``, ``run_info.json``, ``led_###_x##_y##.tif``),
* a packed crop made by ``pack_fpm_for_velos.py`` (``<name>_c512_meta.npz`` + parts), or
* a single ``.npz`` in the Velos FPM layout (see :func:`save_fpm_npz`) — the format the
  backend API accepts for FPM jobs.

Command line::

    python -m velos_recon.fpm run <folder | *_meta.npz | dataset.npz> --Np 512 --z 55 \
        --lambda 0.633 --warmup 0 --passes 50 --out results/leaf
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

try:  # SciPy's FFT keeps complex64 and is multi-threaded
    import scipy.fft as _sfft

    def _fft2(x):
        return _sfft.fft2(x, workers=-1)

    def _ifft2(x):
        return _sfft.ifft2(x, workers=-1)
except Exception:  # pragma: no cover - NumPy fallback
    _fft2, _ifft2 = np.fft.fft2, np.fft.ifft2

_fftshift, _ifftshift = np.fft.fftshift, np.fft.ifftshift
TINY = 1e-20
ALGORITHMS = ("AP", "EPRY", "RPIE")


def _mround(x):
    """MATLAB round (half away from zero)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ============================================================================ configuration
@dataclass
class FPMConfig:
    # optics
    lambda_um: float = 0.632
    objective_NA: float = 0.10
    camera_pixel_um: float = 5.04            # Thorlabs CS2100M-USB pixel pitch
    magnification: float = 4.0
    # LED panel
    z_source_mm: Optional[float] = None       # None = led_to_sample_mm of run_info.json
    led_pitch_mm: Optional[float] = 3.0
    optical_axis_x_mm: float = 0.0
    optical_axis_y_mm: float = 0.0
    swap_led_xy: bool = False
    led_x_sign: int = 1
    led_y_sign: int = -1
    pupil_distance_mm: float = math.inf       # inf = telecentric objective
    defocus_um: float = 0.0
    # patch (centre in 1-indexed full-frame pixel coordinates, as in MATLAB; None = image centre)
    Np: int = 512
    crop_center_x_px: Optional[float] = None
    crop_center_y_px: Optional[float] = None
    # pre-processing
    background_offset: Optional[float] = None
    saturation_level: Optional[float] = 65535.0
    normalize_exposure: bool = True
    remove_hot_pixels: bool = True
    skip_clipped_frames: bool = True
    clipped_fraction: float = 0.60
    skip_transition_frames: bool = True
    transition_band: float = 0.10
    max_illumination_NA: float = math.inf
    # reconstruction
    algorithms: Sequence[str] = ("AP", "EPRY", "rPIE")
    n_passes: int = 10
    n_warmup: int = 2
    rpie_alpha: float = 0.5
    rpie_beta: float = 0.5
    intensity_correction: bool = True
    intensity_limit: float = 3.0
    init_spherical_phase: bool = True
    upsample: Optional[int] = None
    geometry_check: str = "report"           # off | report | apply
    precision: str = "single"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["algorithms"] = list(self.algorithms)
        for k, v in list(d.items()):
            if isinstance(v, float) and math.isinf(v):
                d[k] = "inf"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FPMConfig":
        known = {f for f in cls.__dataclass_fields__}
        kw = {}
        for k, v in (d or {}).items():
            if k not in known:
                raise ValueError(f"unknown FPM setting '{k}'")
            if isinstance(v, str) and v.lower() in ("inf", "infinity"):
                v = math.inf
            kw[k] = v
        return cls(**kw)


# ============================================================================ acquisition data
@dataclass
class FPMAcquisition:
    """Raw frames of one LED scan (all frames, possibly only a crop of the sensor)."""
    frames: np.ndarray                         # [M, h, w] raw grey values (uint16 or float)
    files: List[str]
    pixel_x: np.ndarray                        # LED panel pixel coordinates
    pixel_y: np.ndarray
    led_x_mm: np.ndarray                       # LED positions [mm], +x right, +y up (positions.csv)
    led_y_mm: np.ndarray
    exposure_us: np.ndarray
    gain: np.ndarray
    frame_height: int                          # full sensor frame
    frame_width: int
    crop_row0: int = 0                         # position of `frames` in the full frame (0-indexed)
    crop_col0: int = 0
    run_info: dict = field(default_factory=dict)
    name: str = "fpm"
    defaults: dict = field(default_factory=dict)   # setup defaults stored with the data (FPMConfig fields)
    centre_full: Optional[np.ndarray] = None   # full-frame image of the centre LED (overview only)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    def summary(self) -> dict:
        return {"modality": "fpm", "name": self.name, "n_frames": self.n_frames,
                "frame_shape": [int(self.frame_height), int(self.frame_width)],
                "stored_region": [int(self.crop_row0), int(self.crop_col0),
                                  int(self.frames.shape[1]), int(self.frames.shape[2])],
                "led_pitch_mm": float(self.run_info.get("led_pitch_mm", 3.0) or 3.0),
                "led_to_sample_mm": self.run_info.get("led_to_sample_mm"),
                "led_color": self.run_info.get("led_color"),
                "camera": self.run_info.get("camera_model"),
                "exposure_us": sorted({float(t) for t in np.unique(self.exposure_us)})[:4],
                "defaults": self.defaults}

    def subset(self, keep: np.ndarray, name_suffix: str = "") -> "FPMAcquisition":
        keep = np.asarray(keep)
        if keep.dtype == bool:
            keep = np.flatnonzero(keep)
        return replace(self, frames=self.frames[keep], files=[self.files[i] for i in keep],
                       pixel_x=self.pixel_x[keep], pixel_y=self.pixel_y[keep],
                       led_x_mm=self.led_x_mm[keep], led_y_mm=self.led_y_mm[keep],
                       exposure_us=self.exposure_us[keep], gain=self.gain[keep],
                       name=self.name + name_suffix)

    def crop(self, size: int, center_row: Optional[int] = None, center_col: Optional[int] = None) -> "FPMAcquisition":
        """Keep a size×size region of the stored frames (full-frame coordinates, 0-indexed centre)."""
        h, w = self.frames.shape[1:]
        cr = self.crop_row0 + h // 2 if center_row is None else center_row
        cc = self.crop_col0 + w // 2 if center_col is None else center_col
        r0 = int(np.clip(cr - size // 2 - self.crop_row0, 0, h - size))
        c0 = int(np.clip(cc - size // 2 - self.crop_col0, 0, w - size))
        return replace(self, frames=self.frames[:, r0:r0 + size, c0:c0 + size],
                       crop_row0=self.crop_row0 + r0, crop_col0=self.crop_col0 + c0)


def _read_tif(path: Path) -> np.ndarray:
    try:
        import tifffile
        return np.asarray(tifffile.imread(str(path)))
    except Exception:
        from PIL import Image
        with Image.open(path) as im:
            return np.array(im)


def _positions_table(text: str) -> Dict[str, list]:
    text = text.lstrip("﻿")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("positions.csv contains no data rows")
    cols = {k.strip(): [r.get(k, "") for r in rows] for k in rows[0].keys()}
    for need in ("index", "pixel_x", "pixel_y", "file"):
        if need not in cols:
            raise ValueError(f'positions.csv is missing the column "{need}"')
    return cols


def _num(col: Optional[list], n: int) -> np.ndarray:
    if col is None:
        return np.full(n, np.nan)
    out = np.full(n, np.nan)
    for i, v in enumerate(col):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _from_tables(frames, positions_text: str, run_info: dict, **kw) -> FPMAcquisition:
    T = _positions_table(positions_text)
    n = len(T["file"])
    idx = _num(T["index"], n)
    order = np.argsort(idx, kind="stable")
    if not np.array_equal(idx[order], np.arange(1, n + 1)):
        raise ValueError("positions.csv must contain the indices 1..N once each")
    take = lambda name: _num(T.get(name), n)[order]
    pixel_x, pixel_y = take("pixel_x"), take("pixel_y")
    files = [T["file"][i].strip() for i in order]
    pitch = float(run_info.get("led_pitch_mm") or 3.0)
    led_x, led_y = take("led_x_mm"), take("led_y_mm")
    if not (np.all(np.isfinite(led_x)) and np.all(np.isfinite(led_y))):
        led_x = (pixel_x - pixel_x[0]) * pitch
        led_y = (pixel_y[0] - pixel_y) * pitch
    t_us = take("exposure_us")
    if not np.all(np.isfinite(t_us)):
        t_us = take("configured_exposure_us")
    gain = take("gain")
    if not np.all(np.isfinite(gain)):
        gain = take("configured_gain")
    if not np.all(np.isfinite(t_us)) or np.any(t_us <= 0):
        t_us = np.ones(n)
    if not np.all(np.isfinite(gain)) or np.any(gain <= 0):
        gain = np.ones(n)
    if frames is not None and frames.shape[0] == n:
        frames = frames[order]
    return FPMAcquisition(frames=frames, files=files, pixel_x=pixel_x, pixel_y=pixel_y,
                          led_x_mm=led_x, led_y_mm=led_y, exposure_us=t_us, gain=gain,
                          run_info=run_info, **kw)


def load_fpm_folder(folder: str | Path, crop: Optional[int] = None, name: Optional[str] = None) -> FPMAcquisition:
    """Read an acquisition folder written by the LED-scan acquisition software."""
    folder = Path(folder)
    pos = (folder / "positions.csv").read_text(encoding="utf-8-sig")
    info = {}
    if (folder / "run_info.json").exists():
        info = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
    acq = _from_tables(None, pos, info, frame_height=0, frame_width=0, name=name or folder.name)
    first = _read_tif(folder / acq.files[0])
    if first.ndim == 3:
        raise ValueError("colour frames: pick a channel before loading (not needed for the CS2100M camera)")
    H, W = first.shape
    n = min(crop or max(H, W), H, W)
    r0, c0 = ((H - n) // 2, (W - n) // 2) if crop else (0, 0)
    h, w = (n, n) if crop else (H, W)
    frames = np.empty((acq.n_frames if acq.frames is not None else len(acq.files), h, w), dtype=first.dtype)
    centre = int(info.get("center_led_index", 0) or 0)
    centre_full = None
    for m, fn in enumerate(acq.files):
        a = first if m == 0 else _read_tif(folder / fn)
        frames[m] = a[r0:r0 + h, c0:c0 + w]
        if centre and m == centre - 1:
            centre_full = a
    return replace(acq, frames=frames, frame_height=H, frame_width=W, crop_row0=r0, crop_col0=c0,
                   centre_full=centre_full)


def _mat_str(v) -> str:
    a = np.asarray(v)
    if a.dtype.kind in "US":
        return "".join(a.ravel().tolist()) if a.ndim else str(a)
    return str(a)


def load_fpm_packed(meta_path: str | Path, name: Optional[str] = None) -> FPMAcquisition:
    """Read the files written by pack_fpm_for_velos.py (``*_meta.npz`` + ``*_partKofN.npz``) or by
    pack_fpm_for_velos.m (``*_meta.mat`` + ``*_partKofN.mat``, MATLAB v7)."""
    meta_path = Path(meta_path)
    is_mat = meta_path.name.endswith("_meta.mat")
    stem = meta_path.name[: -len("_meta.mat" if is_mat else "_meta.npz")]
    if is_mat:
        from scipy.io import loadmat
        m = loadmat(str(meta_path), squeeze_me=True, chars_as_strings=True)
        parts = int(m["parts"])
        chunks = []
        for k in range(parts):
            p = loadmat(str(meta_path.with_name(f"{stem}_part{k + 1}of{parts}.mat")), squeeze_me=False)
            f = p["frames"]
            chunks.append(np.moveaxis(f if f.ndim == 3 else f[:, :, None], 2, 0))     # MATLAB [n, n, k] -> [k, n, n]
        files_packed = [str(x) for x in np.atleast_1d(m["files"])]
        positions, run_info_txt = _mat_str(m["positions_csv"]), _mat_str(m["run_info_json"])
        H, W, r0, c0 = (int(m[k]) for k in ("frame_height", "frame_width", "crop_row0", "crop_col0"))
        cf = np.asarray(m["centre_full"]) if "centre_full" in m and np.asarray(m["centre_full"]).size else None
    else:
        m = np.load(meta_path, allow_pickle=False)
        parts = int(m["parts"])
        chunks = [np.load(meta_path.with_name(f"{stem}_part{k + 1}of{parts}.npz"))["frames"] for k in range(parts)]
        files_packed = [str(f) for f in m["files"]]
        positions, run_info_txt = str(m["positions_csv"]), str(m["run_info_json"])
        H, W, r0, c0 = (int(m[k]) for k in ("frame_height", "frame_width", "crop_row0", "crop_col0"))
        cf = m["centre_full"] if m["centre_full"].size else None
    frames = np.concatenate(chunks, axis=0)
    if frames.shape[0] != len(files_packed):
        raise ValueError(f"{meta_path.name}: {frames.shape[0]} frames for {len(files_packed)} files")
    info = json.loads(run_info_txt) if run_info_txt.strip() else {}
    acq = _from_tables(None, positions, info, frame_height=H, frame_width=W, crop_row0=r0, crop_col0=c0,
                       name=name or stem)
    pos_index = {f: i for i, f in enumerate(files_packed)}      # packed in positions.csv row order
    frames = frames[[pos_index[f] for f in acq.files]]
    return replace(acq, frames=frames, centre_full=cf)


def save_fpm_npz(acq: FPMAcquisition, path: str | Path, defaults: Optional[dict] = None) -> Path:
    """Single-file FPM dataset (the layout the backend API accepts for ``modality = fpm``).

    keys: ``modality`` = "fpm", ``frames`` [M, h, w] (raw grey values, index order), ``files``,
    ``pixel_x``, ``pixel_y``, ``led_x_mm``, ``led_y_mm``, ``exposure_us``, ``gain``,
    ``frame_height``, ``frame_width``, ``crop_row0``, ``crop_col0``, ``run_info_json``,
    ``fpm_defaults_json`` (FPMConfig fields that describe the set-up, e.g. lambda_um, z_source_mm).
    """
    path = Path(path)
    np.savez_compressed(path, modality="fpm", frames=acq.frames, files=np.array(acq.files),
                        pixel_x=acq.pixel_x, pixel_y=acq.pixel_y, led_x_mm=acq.led_x_mm, led_y_mm=acq.led_y_mm,
                        exposure_us=acq.exposure_us, gain=acq.gain, frame_height=acq.frame_height,
                        frame_width=acq.frame_width, crop_row0=acq.crop_row0, crop_col0=acq.crop_col0,
                        run_info_json=json.dumps(acq.run_info), name=acq.name,
                        fpm_defaults_json=json.dumps(defaults if defaults is not None else acq.defaults))
    return path


def is_fpm_npz(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return "modality" in z.files and str(z["modality"]) == "fpm"
    except Exception:
        return False


def load_fpm_npz(path: str | Path, name: Optional[str] = None) -> FPMAcquisition:
    z = np.load(path, allow_pickle=False)
    if "modality" not in z.files or str(z["modality"]) != "fpm":
        raise ValueError("not a Velos FPM dataset (.npz with modality = 'fpm')")
    frames = z["frames"]
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError("frames must be [M, h, w] with M >= 2")
    n = frames.shape[0]
    arr = lambda k, default: (np.asarray(z[k], dtype=float) if k in z.files else default)
    acq = FPMAcquisition(
        frames=frames, files=[str(f) for f in z["files"]] if "files" in z.files else [f"frame_{i + 1:03d}" for i in range(n)],
        pixel_x=arr("pixel_x", np.zeros(n)), pixel_y=arr("pixel_y", np.zeros(n)),
        led_x_mm=arr("led_x_mm", None), led_y_mm=arr("led_y_mm", None),
        exposure_us=arr("exposure_us", np.ones(n)), gain=arr("gain", np.ones(n)),
        frame_height=int(z["frame_height"]) if "frame_height" in z.files else frames.shape[1],
        frame_width=int(z["frame_width"]) if "frame_width" in z.files else frames.shape[2],
        crop_row0=int(z["crop_row0"]) if "crop_row0" in z.files else 0,
        crop_col0=int(z["crop_col0"]) if "crop_col0" in z.files else 0,
        run_info=json.loads(str(z["run_info_json"])) if "run_info_json" in z.files and str(z["run_info_json"]) else {},
        name=name or (str(z["name"]) if "name" in z.files else Path(path).stem),
        defaults=json.loads(str(z["fpm_defaults_json"])) if "fpm_defaults_json" in z.files else {})
    if acq.led_x_mm is None or acq.led_y_mm is None:
        raise ValueError("led_x_mm / led_y_mm missing")
    for k in ("pixel_x", "pixel_y", "led_x_mm", "led_y_mm", "exposure_us", "gain"):
        if getattr(acq, k).shape != (n,):
            raise ValueError(f"{k} must have one value per frame")
    return acq


def load_fpm(path: str | Path, crop: Optional[int] = None, name: Optional[str] = None) -> FPMAcquisition:
    p = Path(path)
    if p.is_dir():
        return load_fpm_folder(p, crop=crop, name=name)
    if p.name.endswith("_meta.npz") or p.name.endswith("_meta.mat"):
        acq = load_fpm_packed(p, name=name)
    else:
        acq = load_fpm_npz(p, name=name)
    return acq.crop(crop) if crop and crop < min(acq.frames.shape[1:]) else acq


# ============================================================================ geometry
@dataclass
class Geometry:
    illumination_na: np.ndarray
    du: float
    pupil_radius_px: float
    shift_row_exact: np.ndarray
    shift_col_exact: np.ndarray
    shift_row: np.ndarray
    shift_col: np.ndarray
    synthetic_NA: float
    N_obj: int
    row1: np.ndarray            # 0-indexed first row of every Np×Np spectrum window
    col1: np.ndarray


def led_geometry(cfg: FPMConfig, led_x_mm, led_y_mm, patch_x_mm, patch_y_mm, Np, dpix_um, upsample=None) -> Geometry:
    """LED positions -> illumination angles at the patch centre -> integer spectrum windows."""
    x = np.asarray(led_x_mm, float) - cfg.optical_axis_x_mm
    y = np.asarray(led_y_mm, float) - cfg.optical_axis_y_mm
    if cfg.swap_led_xy:
        x, y = y, x
    ff = 1.0 + cfg.z_source_mm / cfg.pupil_distance_mm
    sx = cfg.led_x_sign * x - ff * patch_x_mm
    sy = cfg.led_y_sign * y - ff * patch_y_mm
    dist = np.sqrt(sx ** 2 + sy ** 2 + cfg.z_source_mm ** 2)
    sin_x, sin_y = -sx / dist, -sy / dist
    ill = np.hypot(sin_x, sin_y)
    du = 1.0 / (Np * dpix_um)
    sce = (sin_x / cfg.lambda_um) / du
    sre = (sin_y / cfg.lambda_um) / du
    sc, sr = _mround(sce).astype(int), _mround(sre).astype(int)
    syn = cfg.objective_NA + float(np.max(ill))
    if upsample is None:
        max_shift = int(max(np.max(np.abs(sc)), np.max(np.abs(sr))))
        need = max(Np, Np + 2 * max_shift + 4, int(math.ceil(2 * (syn / cfg.lambda_um) / du)))
        N_obj = int(math.ceil(need / Np) * Np)
    else:
        N_obj = int(round(upsample) * Np)
    row1 = N_obj // 2 - sr - Np // 2
    col1 = N_obj // 2 - sc - Np // 2
    if np.any(row1 < 0) or np.any(col1 < 0) or np.any(row1 + Np > N_obj) or np.any(col1 + Np > N_obj):
        raise ValueError("Spectrum windows exceed the reconstruction grid: increase upsample or check the geometry.")
    return Geometry(ill, du, (cfg.objective_NA / cfg.lambda_um) / du, sre, sce, sr, sc, syn, N_obj, row1, col1)


def start_pupil(cfg: FPMConfig, Np: int, du: float, radius_px: float):
    coord = np.arange(Np) - Np / 2
    xx, yy = np.meshgrid(coord, coord)
    supp = (np.hypot(xx, yy) < radius_px).astype(float)
    P0 = supp.astype(complex)
    if cfg.defocus_um != 0:
        f2 = (xx * du) ** 2 + (yy * du) ** 2
        kz = 2 * np.pi * np.sqrt(np.maximum(1 / cfg.lambda_um ** 2 - f2, 0))
        P0 = supp * np.exp(1j * cfg.defocus_um * (kz - 2 * np.pi / cfg.lambda_um))
    return P0, supp


def spherical_phase(cfg: FPMConfig, Np: int, dpix_um: float) -> np.ndarray:
    """Phase curvature of the point-source illumination across the patch."""
    coord = (np.arange(Np) - Np / 2) * dpix_um
    xx, yy = np.meshgrid(coord, coord)
    ff = 1.0 + cfg.z_source_mm / cfg.pupil_distance_mm
    return np.pi * ff * (xx ** 2 + yy ** 2) / (cfg.lambda_um * cfg.z_source_mm * 1000)


# ============================================================================ engine
@dataclass
class EngineOptions:
    n_passes: int = 10
    n_warmup: int = 2
    rpie_alpha: float = 0.5
    rpie_beta: float = 0.5
    intensity_correction: bool = True
    intensity_limit: float = 3.0
    init_phase: Optional[np.ndarray] = None
    subpixel: Optional[dict] = None          # geometry check: {'dr','dc','radius'}


class FPMCancelled(RuntimeError):
    pass


def fpm_recover(A: np.ndarray, row1, col1, P0, supp, N_obj: int, algo: str, opt: EngineOptions,
                n_bright: int, want0: bool = False, init_spectrum: Optional[np.ndarray] = None,
                progress: Optional[Callable] = None, should_cancel: Optional[Callable] = None):
    """Port of ``fpm_recover``.

    A       [M, Np, Np] measured AMPLITUDES (frames low -> high illumination angle), NaN = ignored pixel
    row1/col1  0-indexed first row / column of every Np×Np window in the (fftshifted) spectrum
    returns obj, P, resid, fac, tsec, resid0
    """
    M, Np, _ = A.shape
    a = algo.upper()
    if a == "AP":
        alpha, beta, gP = 1.0, 1.0, 0.0
    elif a == "EPRY":
        alpha, beta, gP = 1.0, 1.0, 1.0
    elif a == "RPIE":
        alpha, beta, gP = opt.rpie_alpha, opt.rpie_beta, 1.0
    else:
        raise ValueError(f'Unknown algorithm "{algo}" (use AP, EPRY or rPIE).')
    rdt = np.float64 if A.dtype == np.float64 else np.float32
    cdt = np.complex64 if rdt is np.float32 else np.complex128
    nanmask = np.isnan(A)
    has_nan = bool(nanmask.any())
    frame_has_nan = nanmask.reshape(M, -1).any(axis=1) if has_nan else np.zeros(M, bool)
    E = np.empty(M)
    for m in range(M):
        am = A[m]
        if frame_has_nan[m]:
            ok = ~nanmask[m]
            E[m] = float(np.sum(am[ok].astype(float) ** 2)) / max(int(ok.sum()), 1)
        else:
            E[m] = float(np.mean(am.astype(float) ** 2))

    P = P0.astype(cdt)
    suppc = supp.astype(rdt)
    pad = (N_obj - Np) // 2
    if init_spectrum is not None:
        O = init_spectrum.astype(cdt).copy()
    else:
        a0 = A[0].astype(rdt).copy()
        if frame_has_nan[0]:
            bad = nanmask[0]
            a0[bad] = a0[~bad].mean()
        a0 = a0.astype(cdt)
        if opt.init_phase is not None:
            a0 = a0 * np.exp(1j * opt.init_phase).astype(cdt)
        O = np.zeros((N_obj, N_obj), cdt)
        O[pad:pad + Np, pad:pad + Np] = _fftshift(_fft2(a0))
        cover = np.zeros((N_obj, N_obj), rdt)
        for m in range(M):
            r, c = int(row1[m]), int(col1[m])
            np.maximum(cover[r:r + Np, c:c + Np], suppc, out=cover[r:r + Np, c:c + Np])
        O *= cover
        del cover

    n_total = opt.n_warmup + opt.n_passes
    resid = np.zeros(n_total)
    resid0 = np.zeros(n_total)
    fac = np.ones(M)
    use_ic = opt.intensity_correction
    sub = opt.subpixel is not None and gP == 0
    if sub:
        coord = np.arange(Np) - Np / 2
        xx, yy = np.meshgrid(coord, coord)
        xx = xx.astype(rdt); yy = yy.astype(rdt)
    Afill = np.where(nanmask, 0, A).astype(rdt) if has_nan else A.astype(rdt, copy=False)
    t0 = time.perf_counter()
    conjP = np.conj(P); Pabs2 = (P.real ** 2 + P.imag ** 2).astype(rdt); Pmax2 = float(Pabs2.max())
    for it in range(n_total):
        warm = it < opt.n_warmup
        num = den = num0 = den0 = 0.0
        for m in range(M):
            if should_cancel is not None and m == 0 and should_cancel():
                raise FPMCancelled("cancelled")
            r, c = int(row1[m]), int(col1[m])
            Ow = O[r:r + Np, c:c + Np]
            if sub:
                s = opt.subpixel
                P = np.clip(s["radius"] + 0.5 - np.sqrt((xx - s["dc"][m]) ** 2 + (yy - s["dr"][m]) ** 2), 0, 1).astype(cdt)
                conjP = np.conj(P); Pabs2 = (P.real ** 2).astype(rdt); Pmax2 = float(Pabs2.max())
            lowFT = Ow * P
            psi = _ifft2(_ifftshift(lowFT))
            aest = np.abs(psi)
            Am = Afill[m]
            ok = None
            if frame_has_nan[m]:
                ok = ~nanmask[m]
            if want0:
                d0 = Am - aest
                if ok is not None:
                    d0 = np.where(ok, d0, 0)
                num0 += float(np.sum(d0.astype(np.float64) ** 2))
                den0 += float(np.sum(Am.astype(np.float64) ** 2))
            if use_ic and not warm:
                if ok is not None:
                    s2 = float(np.sum(aest[ok].astype(np.float64) ** 2)) / max(int(ok.sum()), 1)
                else:
                    s2 = float(np.mean(aest.astype(np.float64) ** 2))
                f = math.sqrt(s2 / max(E[m], TINY))
                fac[m] = min(max(f, 1.0 / opt.intensity_limit), opt.intensity_limit)
            Am = Am * rdt(fac[m])
            if ok is not None:
                Am = np.where(ok, Am, aest)
            d = Am - aest
            num += float(np.sum(d.astype(np.float64) ** 2))
            den += float(np.sum(Am.astype(np.float64) ** 2))
            ratio = Am / np.maximum(aest, TINY)
            psi2 = psi * ratio + Am * (aest <= TINY)
            dF = _fftshift(_fft2(psi2)) - lowFT
            if warm:
                O[r:r + Np, c:c + Np] = Ow + conjP * dF / Pmax2
            else:
                Onew = Ow + conjP * dF / ((1 - alpha) * Pabs2 + alpha * Pmax2)
                if gP > 0:
                    Oabs2 = (Ow.real ** 2 + Ow.imag ** 2)
                    Omax2 = float(Oabs2.max())
                    P = (P + gP * np.conj(Ow) * dF / ((1 - beta) * Oabs2 + beta * Omax2 + TINY)) * suppc
                    conjP = np.conj(P); Pabs2 = (P.real ** 2 + P.imag ** 2).astype(rdt); Pmax2 = float(Pabs2.max())
                O[r:r + Np, c:c + Np] = Onew
        resid[it] = math.sqrt(num / max(den, TINY))
        if want0:
            resid0[it] = math.sqrt(num0 / max(den0, TINY))
        if not math.isfinite(resid[it]):
            resid, resid0 = resid[: it + 1], resid0[: it + 1]
            break
        if use_ic and not warm:
            g = math.exp(float(np.mean(np.log(fac[: min(n_bright, M)]))))
            O /= g
            fac /= g
        if progress is not None:
            progress(it + 1, n_total, {"err": resid[it]})
    obj = _ifft2(_ifftshift(O)) * (N_obj / Np) ** 2
    return obj, P, resid, fac, time.perf_counter() - t0, resid0, O


# ============================================================================ pre-processing helpers
def _mean_valid(x) -> float:
    x = np.asarray(x, float).ravel()
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def _median_valid(x) -> float:
    x = np.asarray(x, float).ravel()
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else 0.0


_OFF = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if not (dr == 0 and dc == 0)]


def _neighbour_median_full(f: np.ndarray) -> np.ndarray:
    """Median of the 24 neighbours (5×5 without the centre, edges clamped) for every pixel."""
    n1, n2 = f.shape
    fp = np.pad(f, 2, mode="edge")
    v = np.stack([fp[2 + dr:2 + dr + n1, 2 + dc:2 + dc + n2] for dr, dc in _OFF], 0)
    v = np.sort(v, axis=0)                       # NaN sort last, as in MATLAB
    return 0.5 * (v[11] + v[12])


def _neighbour_median_at(f: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    n1, n2 = f.shape
    v = np.stack([f[np.clip(rr + dr, 0, n1 - 1), np.clip(cc + dc, 0, n2 - 1)] for dr, dc in _OFF], 1)
    v = np.sort(v, axis=1)
    return 0.5 * (v[:, 11] + v[:, 12])


def fix_hot_pixels(I: np.ndarray, frame_mean: np.ndarray):
    """Isolated bright sensor pixels: found in the darkest frames (5×5 neighbourhood),
    replaced in every frame by the median of the 24 neighbours."""
    M, n1, n2 = I.shape
    k = min(8, M)
    idx = np.argsort(frame_mean, kind="stable")
    D = I[idx[:k]].astype(float)
    D[np.isnan(D)] = 0
    D = np.median(D, axis=0)
    S = (D - _neighbour_median_full(D)).ravel()
    sig = 1.4826 * float(np.median(np.abs(S - np.median(S))))
    ref = _median_valid(I[idx[-1]])
    hot = np.flatnonzero(S > max(6 * sig, 0.02 * ref))
    n_hot = hot.size
    if n_hot == 0 or n_hot > 0.05 * n1 * n2:
        return I, 0
    rr, cc = np.unravel_index(hot, (n1, n2))
    for m in range(M):
        f = I[m].astype(float)
        I[m][rr, cc] = _neighbour_median_at(f, rr, cc)
    return I, n_hot


def _box_mean3(a):
    s = np.zeros_like(a, dtype=float)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            s += np.roll(a, (dr, dc), axis=(0, 1))
    return s / 9


def direct_beam(I: np.ndarray, fmean: np.ndarray, sat_floor: np.ndarray):
    """Level of the direct beam of every frame relative to bright field (1 = BF, 0 = DF),
    measured in the clear background of the patch. Returns (direct, n_pix, unknown)."""
    M, n1, n2 = I.shape
    direct = np.full(M, np.nan); unknown = np.zeros(M, bool)
    if not np.any(np.isfinite(fmean)):
        return direct, 0, unknown
    ib = int(np.nanargmax(np.where(np.isfinite(fmean), fmean, -np.inf)))
    a = I[ib].astype(float)
    bad = np.isnan(a)
    if bad.all():
        return direct, 0, unknown
    a[bad] = 0
    s = _box_mean3(a)
    near_bad = _box_mean3(bad.astype(float)) > 0
    lo = s.copy(); hi = s.copy()
    for d in range(-3, 4):
        t = np.roll(s, d, axis=0)
        lo = np.minimum(lo, t); hi = np.maximum(hi, t)
    lo1, hi1 = lo.copy(), hi.copy()
    for d in range(-3, 4):
        lo = np.minimum(lo, np.roll(lo1, d, axis=1)); hi = np.maximum(hi, np.roll(hi1, d, axis=1))
    v = np.sort(a[~bad])
    top = v[max(1, int(_mround(0.99 * v.size))) - 1]
    cand = (~near_bad) & (s >= 0.5 * top) & ((hi - lo) <= 0.18 * s)
    cand[:4, :] = False; cand[-4:, :] = False; cand[:, :4] = False; cand[:, -4:] = False
    if cand.sum() < 50:
        return direct, 0, unknown
    lev = float(np.median(s[cand]))
    flat = cand & (np.abs(s - lev) <= 0.10 * lev)
    mask = flat.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            mask &= np.roll(flat, (dr, dc), axis=(0, 1))
    n = int(mask.sum())
    if n < 100:
        return direct, 0, unknown
    level = np.zeros(M)
    for m in range(M):
        vv = I[m][mask].astype(float)
        ok = ~np.isnan(vv)
        if ok.sum() >= 0.95 * n:
            level[m] = np.median(vv[ok])
        elif ok.sum() >= 0.5 * n:
            level[m] = max(float(np.median(vv[ok])), float(sat_floor[m]))
        else:
            level[m] = sat_floor[m]; unknown[m] = True
    lv = np.sort(level[~unknown])[::-1]
    if lv.size == 0 or not lv[0] > 0:
        return direct, 0, unknown
    lv = lv[lv >= 0.5 * lv[0]]
    ref = lv[min(3, lv.size) - 1]
    return level / ref, n, unknown


# ============================================================================ preparation + full run
@dataclass
class Prepared:
    cfg: FPMConfig
    A: np.ndarray                 # [M, Np, Np] amplitudes (NaN = ignored), low -> high angle
    order: np.ndarray             # indices into the acquisition (index order)
    ill_na: np.ndarray
    geometry: Geometry
    P0: np.ndarray
    supp: np.ndarray
    init_phase: Optional[np.ndarray]
    n_bright: int
    dpix_um: float
    patch: dict
    notes: List[str]
    frames_table: dict
    transition: dict
    geometry_check: dict
    raw_lowest_angle: np.ndarray
    log: List[str]


def prepare(acq: FPMAcquisition, cfg: FPMConfig, log: Optional[Callable[[str], None]] = None) -> Prepared:
    """Sections 3-7 of the MATLAB script: patch, geometry, frames, pre-processing, frame selection,
    geometry check. Returns everything the engine needs."""
    lines: List[str] = []

    def say(msg: str):
        lines.append(msg)
        if log:
            log(msg)

    notes: List[str] = []

    def note(msg: str):
        notes.append(msg)
        say("  ! " + msg)

    cfg = replace(cfg)
    info = acq.run_info or {}
    if cfg.z_source_mm is None:
        if info.get("led_to_sample_mm") is not None:
            cfg.z_source_mm = float(info["led_to_sample_mm"])
        else:
            raise ValueError("z_source_mm is empty and run_info.json has no led_to_sample_mm")
    if cfg.led_pitch_mm is None:
        cfg.led_pitch_mm = float(info.get("led_pitch_mm") or 3.0)
    if not (cfg.lambda_um > 0 and cfg.z_source_mm > 0 and cfg.led_pitch_mm > 0):
        raise ValueError("lambda_um, z_source_mm and led_pitch_mm must be positive")
    if not (0 < cfg.objective_NA < 1):
        raise ValueError("invalid objective NA")
    if cfg.Np < 32 or cfg.Np % 2:
        raise ValueError("choose an even Np >= 32")
    if info.get("led_to_sample_mm") is not None and abs(float(info["led_to_sample_mm"]) - cfg.z_source_mm) > 0.5:
        note(f"run_info.json says led_to_sample_mm = {float(info['led_to_sample_mm']):.1f}, "
             f"this run uses z_source_mm = {cfg.z_source_mm:.1f}.")
    rdt = np.float32 if cfg.precision == "single" else np.float64
    fh, fw = int(acq.frame_height), int(acq.frame_width)
    Np = int(cfg.Np)
    if Np > min(fh, fw):
        raise ValueError(f"Np = {Np} is larger than the frame ({fw} x {fh})")
    dpix_um = cfg.camera_pixel_um / cfg.magnification
    cx = (fw + 1) / 2 if cfg.crop_center_x_px is None else cfg.crop_center_x_px
    cy = (fh + 1) / 2 if cfg.crop_center_y_px is None else cfg.crop_center_y_px
    col_start = int(min(max(_mround(cx - Np / 2), 1), fw - Np + 1))     # 1-indexed, as in MATLAB
    row_start = int(min(max(_mround(cy - Np / 2), 1), fh - Np + 1))
    cx = col_start + (Np - 1) / 2
    cy = row_start + (Np - 1) / 2
    patch_x_mm = (cx - (fw + 1) / 2) * dpix_um / 1000
    patch_y_mm = (cy - (fh + 1) / 2) * dpix_um / 1000
    r0 = row_start - 1 - acq.crop_row0
    c0 = col_start - 1 - acq.crop_col0
    h, w = acq.frames.shape[1:]
    if r0 < 0 or c0 < 0 or r0 + Np > h or c0 + Np > w:
        raise ValueError(f"the {Np}-px patch at ({cy:.1f}, {cx:.1f}) is outside the stored region "
                         f"(rows {acq.crop_row0}..{acq.crop_row0 + h - 1}, cols {acq.crop_col0}..{acq.crop_col0 + w - 1})")
    say(f"Patch: {Np} x {Np} px, centre (x {cx:.1f}, y {cy:.1f}) = ({patch_x_mm:.3f}, {patch_y_mm:.3f}) mm from the image centre")
    say(f"Sample-plane pixel: {dpix_um:.4f} um, patch width {Np * dpix_um:.1f} um")

    # ---- geometry -> order (low -> high angle)
    G = led_geometry(cfg, acq.led_x_mm, acq.led_y_mm, patch_x_mm, patch_y_mm, Np, dpix_um, None)
    order = np.argsort(G.illumination_na, kind="stable")
    order = order[G.illumination_na[order] <= cfg.max_illumination_NA]
    M = order.size
    if M < 2:
        raise ValueError("fewer than two LEDs within max_illumination_NA")
    ill_na = G.illumination_na[order]
    NBF = int(np.sum(ill_na < cfg.objective_NA))
    say(f"Pupil radius: {G.pupil_radius_px:.2f} spectrum pixels; bright-field LEDs {NBF} of {M} (predicted), "
        f"max. illumination NA {ill_na.max():.4f}")
    if NBF == 0:
        raise ValueError("No bright-field frame predicted: check optical_axis_*_mm, z_source_mm and objective_NA.")

    # ---- frames
    I = np.empty((M, Np, Np), dtype=rdt)
    raw_max = np.zeros(M); raw_min = np.zeros(M); frac_min = np.zeros(M); sat_frac = np.zeros(M)
    for m, k in enumerate(order):
        v = acq.frames[k, r0:r0 + Np, c0:c0 + Np]
        raw_max[m] = v.max(); raw_min[m] = v.min()
        frac_min[m] = float(np.mean(v == raw_min[m]))
        I[m] = v
    sat_level = cfg.saturation_level if cfg.saturation_level is not None else np.inf
    sat_thr = sat_level
    for m in range(M):
        s = I[m] >= sat_thr
        sat_frac[m] = float(s.mean())
        I[m][s] = np.nan
    global_min = raw_min.min()
    clipped = (raw_min == global_min) & (frac_min > cfg.clipped_fraction)
    fmean_raw = np.array([_mean_valid(I[m]) for m in range(M)])
    if cfg.background_offset is not None:
        bg = float(cfg.background_offset)
    else:
        idark = int(np.nanargmin(fmean_raw))
        dmin = fmean_raw[idark]
        if dmin - global_min < 0.02 * (np.nanmax(fmean_raw) - global_min):
            bg = _median_valid(I[idark])
            say(f"Background offset: {bg:.2f} (median of the darkest frame; no dark frame given)")
        else:
            bg = 0.0
            note("No dark frame and no frame without light: the camera offset is NOT removed "
                 "(give dark_frame_file or background_offset).")
    I -= rdt(bg)
    n_hot = 0
    if cfg.remove_hot_pixels:
        I, n_hot = fix_hot_pixels(I, fmean_raw)
        if n_hot:
            say(f"Hot pixels: {n_hot} replaced in every frame ({100 * n_hot / Np ** 2:.2f} % of the patch)")
    np.maximum(I, 0, out=I, where=~np.isnan(I))
    t_us = acq.exposure_us[order]; gain = acq.gain[order]
    norm_fac = np.ones(M)
    if cfg.normalize_exposure:
        norm_fac = (t_us[0] / t_us) * (gain[0] / gain)
        for m in range(M):
            if norm_fac[m] != 1:
                I[m] *= rdt(norm_fac[m])
    fmean = np.array([_mean_valid(I[m]) for m in range(M)])

    # ---- frame selection
    use = np.isfinite(fmean) & (fmean > 0)
    if cfg.skip_clipped_frames:
        use &= ~clipped
    if not use[0]:
        raise ValueError("The lowest-angle frame is unusable (clipped or empty). Check channel, exposure and the patch position.")
    n_clip = int(clipped.sum())
    if n_clip:
        msg = (f"{n_clip} of {M} frames have more than {100 * cfg.clipped_fraction:.0f} % of their pixels at the minimum grey "
               f"value ({global_min:g}): the dark level is clipped and weak dark-field signal is lost.")
        if cfg.skip_clipped_frames:
            msg += " These frames are NOT used."
        note(msg + " Expose the dark-field LEDs longer and keep a positive camera offset (black level).")
    if np.sum(sat_frac[use] > 0.01):
        note(f"{int(np.sum(sat_frac[use] > 0.01))} used frames have more than 1 % saturated pixels in the patch (ignored pixel by pixel).")
    if not np.any(use & (ill_na >= cfg.objective_NA)):
        note("No usable dark-field frame: the result is limited to the bright-field synthetic aperture.")

    fm = fmean.copy()
    fm[~use | (sat_frac > 0.02)] = -np.inf
    direct, n_bgpix, bg_saturated = direct_beam(I, fm, np.maximum(sat_thr - bg, 0) * norm_fac)
    have_direct = n_bgpix > 0
    side = np.zeros(M, int)
    if have_direct:
        side[direct >= 0.65] = 1
        side[(direct <= 0.35) & ~bg_saturated] = -1
        trans = use & (direct < 0.80) & ((direct > 0.10) | bg_saturated)
        bright = use & (direct >= 0.80)
        say(f"Direct beam: clear background of the patch ({n_bgpix} pixels): {int(np.sum(direct >= 0.80))} bright-field, "
            f"{int(np.sum((direct < 0.80) & ((direct > 0.10) | bg_saturated)))} transition, "
            f"{int(np.sum(~bg_saturated & (direct <= 0.10)))} dark-field LEDs")
        n_wrong_df = int(np.sum((side > 0) & (ill_na >= 1.05 * cfg.objective_NA)))
        n_wrong_bf = int(np.sum((side < 0) & (ill_na < 0.95 * cfg.objective_NA)))
        if n_wrong_df + n_wrong_bf:
            note(f"{n_wrong_df} LED(s) with a bright background are dark-field LEDs for these settings and {n_wrong_bf} LED(s) "
                 "with a dark background are bright-field LEDs: check z_source_mm, optical_axis_*_mm, objective_NA and "
                 "the orientation flags.")
    else:
        say(f"Direct beam: no clear background in this patch; transition LEDs are taken from the geometry "
            f"(+-{100 * cfg.transition_band:.0f} % of the NA)")
        trans = use & (np.abs(ill_na / cfg.objective_NA - 1) < cfg.transition_band)
        bright = use & (fmean > 0.25 * np.nanmax(np.where(use, fmean, np.nan)))
        n_pred_bf_dark = int(np.sum((ill_na < 0.9 * cfg.objective_NA) & (fmean < 0.25 * np.nanmax(fmean))))
        n_pred_df_bright = int(np.sum((ill_na > 1.15 * cfg.objective_NA) & (fmean > 0.5 * np.nanmax(fmean))))
        if n_pred_bf_dark + n_pred_df_bright:
            note(f"{n_pred_bf_dark} LED(s) predicted bright-field are dark and {n_pred_df_bright} predicted dark-field are "
                 "bright in this patch: check z_source_mm, optical_axis_*_mm and the orientation flags.")
    if not np.any(bright):
        bright = use & (fmean > 0.25 * np.nanmax(np.where(use, fmean, np.nan)))
    kt = order[trans]
    transition = {"pixel_x": acq.pixel_x[kt].tolist(), "pixel_y": acq.pixel_y[kt].tolist(),
                  "direct_beam": [None if not np.isfinite(x) else float(x) for x in direct[trans]],
                  "used": not cfg.skip_transition_frames}
    if kt.size:
        lst = " ".join(f"({int(acq.pixel_x[k])},{int(acq.pixel_y[k])})" for k in kt[:12]) + (" ..." if kt.size > 12 else "")
        if have_direct:
            lo_, hi_ = 100 * np.min(direct[trans]), 100 * np.max(direct[trans])
            lvl = f"{lo_:.0f}" + (f"-{hi_:.0f}" if hi_ - lo_ > 0.5 else "")
            msg = (f"{kt.size} LED(s) light this patch through the edge of the objective aperture (clear background at {lvl} % "
                   f"of the bright-field level): {lst} - bright/dark transition frames, which do not fit one coherent pupil.")
        else:
            msg = (f"{kt.size} LED(s) are within {100 * cfg.transition_band:.0f} % of the objective NA (bright/dark "
                   f"transition): {lst}.")
        if cfg.skip_transition_frames:
            msg += " They are NOT used (skip_transition_frames = false keeps them)."
        note(msg)
    sel = (side != 0) & (ill_na < 2.5 * cfg.objective_NA)
    cons = {"x": acq.led_x_mm[order[sel]], "y": acq.led_y_mm[order[sel]], "side": side[sel]}
    if cfg.skip_transition_frames:
        use &= ~trans
    order = order[use]; I = I[use]; ill_na = ill_na[use]; fmean_u = fmean[use]
    bright = bright[use]; direct_u = direct[use]
    M = order.size
    if M < 2:
        raise ValueError("fewer than two usable frames")
    A = np.sqrt(I)
    del I
    nb = np.flatnonzero(~bright)
    n_lead = max(1, int(nb[0]) if nb.size else M)
    say(f"Frames used: {M} ({int(np.sum(ill_na < cfg.objective_NA))} bright-field by prediction)")

    # ---- geometry check
    gc = {"done": False}
    if cfg.geometry_check.lower() != "off":
        subm = ill_na < 1.8 * cfg.objective_NA
        if subm.sum() >= 5:
            gc, cfg, notes2 = geometry_check(cfg, acq.led_x_mm[order[subm]], acq.led_y_mm[order[subm]], patch_x_mm,
                                             patch_y_mm, dpix_um, A[subm], min(n_lead, int(subm.sum())), cons, say)
            for n_ in notes2:
                note(n_)
        else:
            say("Geometry check skipped (fewer than 5 low-angle frames).")
    if gc.get("done") and gc.get("applied") and not have_direct and cfg.skip_transition_frames:
        Gn = led_geometry(cfg, acq.led_x_mm[order], acq.led_y_mm[order], patch_x_mm, patch_y_mm, Np, dpix_um, None)
        t2 = np.abs(Gn.illumination_na / cfg.objective_NA - 1) < cfg.transition_band
        if t2.any() and (~t2).sum() >= 2 and not t2[0]:
            note(f"{int(t2.sum())} more LED(s) are within {100 * cfg.transition_band:.0f} % of the objective NA with the "
                 "applied geometry and are not used.")
            keep = ~t2
            order = order[keep]; A = A[keep]; ill_na = ill_na[keep]; fmean_u = fmean_u[keep]
            bright = bright[keep]; direct_u = direct_u[keep]
            M = order.size
            nb = np.flatnonzero(~bright)
            n_lead = max(1, int(nb[0]) if nb.size else M)

    G = led_geometry(cfg, acq.led_x_mm[order], acq.led_y_mm[order], patch_x_mm, patch_y_mm, Np, dpix_um, cfg.upsample)
    P0, supp = start_pupil(cfg, Np, G.du, G.pupil_radius_px)
    init_phase = spherical_phase(cfg, Np, dpix_um).astype(rdt) if cfg.init_spherical_phase else None
    say(f"Reconstruction grid: {G.N_obj} x {G.N_obj} (pixel {dpix_um * Np / G.N_obj:.4f} um), synthetic NA {G.synthetic_NA:.3f}")
    say(f"Resolution (full pitch): raw lambda/NA = {cfg.lambda_um / cfg.objective_NA:.2f} um -> "
        f"lambda/NA_syn = {cfg.lambda_um / G.synthetic_NA:.2f} um")
    frames_table = {"led_index": (order + 1).tolist(), "file": [acq.files[k] for k in order],
                    "pixel_x": acq.pixel_x[order].astype(int).tolist(), "pixel_y": acq.pixel_y[order].astype(int).tolist(),
                    "illumination_na": G.illumination_na.tolist(), "shift_row": G.shift_row.tolist(),
                    "shift_col": G.shift_col.tolist(), "exposure_us": acq.exposure_us[order].tolist(),
                    "mean_signal": fmean_u.tolist(),
                    "direct_beam": [None if not np.isfinite(x) else float(x) for x in direct_u]}
    patch = {"Np": Np, "row_start_1": row_start, "col_start_1": col_start, "cx": cx, "cy": cy,
             "patch_x_mm": patch_x_mm, "patch_y_mm": patch_y_mm, "background_offset": bg, "hot_pixels": n_hot}
    return Prepared(cfg, A, order, G.illumination_na, G, P0, supp, init_phase, n_lead, dpix_um, patch, notes,
                    frames_table, transition, gc, A[0] ** 2, lines)


# ---------------------------------------------------------------------------- geometry check
def _check_run(c: FPMConfig, led_x, led_y, px, py, Nc, dpix_um, A, n_bright, cons):
    try:
        G = led_geometry(c, led_x, led_y, px, py, Nc, dpix_um, None)
        n_wrong = 0
        if len(cons["side"]):
            Gc = led_geometry(c, cons["x"], cons["y"], px, py, Nc, dpix_um, None)
            n_wrong = int(np.sum((cons["side"] > 0) & (Gc.illumination_na >= 1.05 * c.objective_NA)) +
                          np.sum((cons["side"] < 0) & (Gc.illumination_na < 0.95 * c.objective_NA)))
    except ValueError:
        return float("nan"), float("nan"), 0
    P0, supp = start_pupil(c, Nc, G.du, G.pupil_radius_px)
    opt = EngineOptions(n_passes=4, n_warmup=2, intensity_correction=c.intensity_correction,
                        intensity_limit=c.intensity_limit,
                        init_phase=spherical_phase(c, Nc, dpix_um).astype(A.dtype) if c.init_spherical_phase else None,
                        subpixel={"dr": G.shift_row - G.shift_row_exact, "dc": G.shift_col - G.shift_col_exact,
                                  "radius": G.pupil_radius_px})
    out = fpm_recover(A, G.row1, G.col1, P0, supp, G.N_obj, "AP", opt, max(1, n_bright), want0=True)
    misfit = float(out[5][-1])
    return misfit + n_wrong, misfit, n_wrong


def _misfit_text(m, w, full):
    s = f"{m:.4f}"
    if w > 0 and full:
        s += f" (and {w} LED(s) on the wrong side of the aperture edge)"
    elif w > 0:
        s += f" [{w}]"
    return s


def geometry_check(cfg: FPMConfig, led_x, led_y, px, py, dpix_um, A, n_bright, cons, say):
    """Coarse check of z, panel orientation and optical axis with short AP runs on the low-angle frames
    (misfit to the frames as measured; LEDs on the wrong side of the aperture edge are penalised)."""
    notes: List[str] = []
    Np = A.shape[1]
    Nc = min(Np, 256)
    c0 = (Np - Nc) // 2
    A = np.ascontiguousarray(A[:, c0:c0 + Nc, c0:c0 + Nc])
    crit_of = lambda c: _check_run(c, led_x, led_y, px, py, Nc, dpix_um, A, n_bright, cons)
    say(f"Geometry check ({A.shape[0]} low-angle frames, short AP runs on the central {Nc} px, misfit to the frames as measured):")
    r_now, m_now, w_now = crit_of(cfg)
    best = replace(cfg); r_best = r_now

    def scan_distance(base, z_list):
        r = np.full(len(z_list), np.nan); mm = r.copy(); ww = np.zeros(len(z_list), int)
        for i, z in enumerate(z_list):
            r[i], mm[i], ww[i] = crit_of(replace(base, z_source_mm=float(z)))
        say("   z [mm]     : " + " ".join(f"{z:6.1f}" for z in z_list))
        say("   misfit     : " + " ".join(f"{x:6.4f}" for x in mm))
        return r, mm, ww

    z_list = cfg.z_source_mm * np.arange(0.80, 1.2001, 0.04)
    r_z, m_z, w_z = scan_distance(best, z_list)
    i = int(np.nanargmin(r_z)) if np.any(np.isfinite(r_z)) else 0
    r_min = r_z[i]
    m_o = [m_now, np.nan, np.nan, np.nan]; w_o = [w_now, 0, 0, 0]
    if r_min < 0.95 * r_best:
        best.z_source_mm = float(z_list[i]); r_best = r_min; m_o[0] = m_z[i]; w_o[0] = int(w_z[i])
    r_o = [r_best, np.nan, np.nan, np.nan]
    for j in (1, 2, 3):
        c = replace(best)
        if j in (1, 3):
            c.led_y_sign = -best.led_y_sign
        if j in (2, 3):
            c.swap_led_xy = not best.swap_led_xy
        r_o[j], m_o[j], w_o[j] = crit_of(c)
    say(f"   orientation at {best.z_source_mm:.1f} mm: as set {_misfit_text(m_o[0], w_o[0], False)} | led_y_sign * -1 "
        f"{_misfit_text(m_o[1], w_o[1], False)} | swap_led_xy reversed {_misfit_text(m_o[2], w_o[2], False)} | both "
        f"{_misfit_text(m_o[3], w_o[3], False)}")
    j = int(np.nanargmin(r_o)) if np.any(np.isfinite(r_o)) else 0
    if j > 0 and r_o[j] < 0.95 * r_best:
        if j in (1, 3):
            best.led_y_sign = -best.led_y_sign
        if j in (2, 3):
            best.swap_led_xy = not best.swap_led_xy
        r_best = r_o[j]
        say("   the distance once more with that orientation:")
        r_z2, _, _ = scan_distance(best, z_list)
        i = int(np.nanargmin(r_z2))
        if r_z2[i] < 0.95 * r_best:
            best.z_source_mm = float(z_list[i]); r_best = r_z2[i]
    px_mm = best.lambda_um * best.z_source_mm / (Nc * dpix_um)
    starts = [best.z_source_mm]
    if r_best < 0.95 * r_now:
        starts = [best.z_source_mm * f for f in (1, 0.93, 1.07)]
    n_eval = 0; r_s = np.inf; x_s = None
    for z0 in starts:
        x = np.array([z0, best.optical_axis_x_mm, best.optical_axis_y_mm], float)
        c = replace(best, z_source_mm=float(x[0]), optical_axis_x_mm=float(x[1]), optical_axis_y_mm=float(x[2]))
        r = crit_of(c)[0]; ne = 1
        step = np.array([0.03 * x[0], px_mm, px_mm])
        while ne < 60 and step[0] > 0.004 * x[0]:
            improved = False
            for d in range(3):
                for sgn in (-1, 1):
                    xt = x.copy(); xt[d] += sgn * step[d]
                    c = replace(best, z_source_mm=float(xt[0]), optical_axis_x_mm=float(xt[1]), optical_axis_y_mm=float(xt[2]))
                    rt = crit_of(c)[0]; ne += 1
                    if rt < 0.99 * r:
                        x, r, improved = xt, rt, True
                        break
            if not improved:
                step = step / 2
        n_eval += ne
        if r < r_s:
            r_s, x_s = r, x
    if x_s is not None and r_s < 0.95 * r_best:
        best.z_source_mm, best.optical_axis_x_mm, best.optical_axis_y_mm = (float(v) for v in x_s)
        r_best = r_s
    c = replace(best, led_y_sign=-best.led_y_sign)
    r_m = crit_of(c)[0]
    if r_m < 0.95 * r_best:
        best = c
    _, m_best, w_best = crit_of(best)
    say(f"   step search ({n_eval} runs): z = {best.z_source_mm:.1f} mm, optical axis ({best.optical_axis_x_mm:+.2f}, "
        f"{best.optical_axis_y_mm:+.2f}) mm, swap_led_xy = {int(best.swap_led_xy)}, led_y_sign = {best.led_y_sign:+d}")
    say(f"   misfit: {_misfit_text(m_now, w_now, True)} with the settings as given, {_misfit_text(m_best, w_best, True)} "
        "with the best values found")
    z_off = abs(best.z_source_mm / cfg.z_source_mm - 1) > 0.02
    o_off = best.led_y_sign != cfg.led_y_sign or best.swap_led_xy != cfg.swap_led_xy
    a_off = math.hypot(best.optical_axis_x_mm - cfg.optical_axis_x_mm, best.optical_axis_y_mm - cfg.optical_axis_y_mm) > 0.5 * px_mm
    if o_off:
        notes.append(f"Geometry check: the LED orientation swap_led_xy = {int(best.swap_led_xy)}, led_x_sign = {best.led_x_sign:+d}, "
                     f"led_y_sign = {best.led_y_sign:+d} fits the frames better than the one set.")
    if z_off:
        notes.append(f"Geometry check: z_source_mm = {best.z_source_mm:.1f} mm fits the frames better than {cfg.z_source_mm:.1f} mm "
                     f"(objective_NA = {cfg.objective_NA * best.z_source_mm / cfg.z_source_mm:.3f} at {cfg.z_source_mm:.1f} mm is "
                     "equivalent). Measure the LED-to-sample distance again.")
    if a_off:
        notes.append(f"Geometry check: optical_axis_x_mm = {best.optical_axis_x_mm:.2f}, optical_axis_y_mm = "
                     f"{best.optical_axis_y_mm:.2f} fits better than ({cfg.optical_axis_x_mm:.2f}, {cfg.optical_axis_y_mm:.2f}): "
                     "the scan centre is not exactly on the optical axis.")
    if o_off or z_off or a_off:
        notes.append(f"Geometry check: misfit {_misfit_text(m_now, w_now, True)} with the settings as given, "
                     f"{_misfit_text(m_best, w_best, True)} with the values above.")
    gc = {"done": True, "residual_as_set": m_now, "residual_best": m_best, "wrong_side_as_set": w_now,
          "wrong_side_best": w_best, "z_list": z_list.tolist(), "residual_z": m_z.tolist(),
          "z_best": best.z_source_mm, "swap_led_xy": bool(best.swap_led_xy), "led_y_sign": int(best.led_y_sign),
          "optical_axis_mm": [best.optical_axis_x_mm, best.optical_axis_y_mm], "applied": False}
    if cfg.geometry_check.lower() == "apply" and (z_off or o_off or a_off):
        cfg = replace(cfg, z_source_mm=best.z_source_mm, swap_led_xy=best.swap_led_xy, led_y_sign=best.led_y_sign,
                      optical_axis_x_mm=best.optical_axis_x_mm, optical_axis_y_mm=best.optical_axis_y_mm)
        gc["applied"] = True
        say("   APPLIED to this run.")
    return gc, cfg, notes


# ---------------------------------------------------------------------------- reduced LED sets
def reduced_led_set(acq: FPMAcquisition, cfg: FPMConfig, fraction: float, seed: int = 0) -> np.ndarray:
    """Indices of a reduced LED set: every LED inside 1.1 × the objective NA (bright field and the
    aperture edge) plus a random ``fraction`` of the dark-field LEDs."""
    z = cfg.z_source_mm or float(acq.run_info.get("led_to_sample_mm") or 50.0)
    G = led_geometry(replace(cfg, z_source_mm=z), acq.led_x_mm, acq.led_y_mm, 0.0, 0.0, 64,
                         cfg.camera_pixel_um / cfg.magnification, None)
    inner = G.illumination_na < 1.1 * cfg.objective_NA
    outer = np.flatnonzero(~inner)
    rng = np.random.default_rng(seed)
    n_keep = int(round(fraction * outer.size))
    keep = np.sort(np.concatenate([np.flatnonzero(inner), rng.choice(outer, n_keep, replace=False)]))
    return keep


# ============================================================================ results
@dataclass
class AlgoResult:
    name: str
    obj: np.ndarray
    pupil: np.ndarray
    residual: np.ndarray
    intensity_factor: np.ndarray
    time_s: float
    spectrum: Optional[np.ndarray] = None


@dataclass
class FPMResult:
    prepared: Prepared
    algorithms: List[AlgoResult]
    total_s: float

    @property
    def pixel_um(self) -> float:
        p = self.prepared
        return p.dpix_um * p.cfg.Np / p.geometry.N_obj

    @property
    def curvature_rad_um2(self) -> float:
        c = self.prepared.cfg
        return math.pi / (c.lambda_um * c.z_source_mm * 1000)

    def get(self, name: str) -> AlgoResult:
        for a in self.algorithms:
            if a.name.upper() == name.upper():
                return a
        raise KeyError(name)

    def summary(self) -> dict:
        p = self.prepared
        return {"modality": "fpm", "config": p.cfg.to_dict(), "patch": p.patch, "N_obj": p.geometry.N_obj,
                "pixel_um": self.pixel_um, "raw_pixel_um": p.dpix_um, "synthetic_NA": p.geometry.synthetic_NA,
                "frames_used": int(p.A.shape[0]), "n_bright": int(p.n_bright),
                "resolution_um": {"raw": p.cfg.lambda_um / p.cfg.objective_NA,
                                  "synthetic": p.cfg.lambda_um / p.geometry.synthetic_NA},
                "algorithms": [{"name": a.name, "final_residual": float(a.residual[-1]), "time_s": round(a.time_s, 2),
                                "passes": int(a.residual.size)} for a in self.algorithms],
                "transition_leds": p.transition, "geometry_check": p.geometry_check, "notes": p.notes,
                "total_s": round(self.total_s, 2)}


def reconstruct_fpm(acq: FPMAcquisition, cfg: FPMConfig, log: Optional[Callable[[str], None]] = None,
                    progress: Optional[Callable] = None, should_cancel: Optional[Callable] = None,
                    prepared: Optional[Prepared] = None) -> FPMResult:
    t_all = time.perf_counter()
    prep = prepared or prepare(acq, cfg, log=log)
    cfg = prep.cfg
    opt = EngineOptions(n_passes=cfg.n_passes, n_warmup=cfg.n_warmup, rpie_alpha=cfg.rpie_alpha, rpie_beta=cfg.rpie_beta,
                        intensity_correction=cfg.intensity_correction, intensity_limit=cfg.intensity_limit,
                        init_phase=prep.init_phase)
    out: List[AlgoResult] = []
    n_alg = len(cfg.algorithms)
    for ia, name in enumerate(cfg.algorithms):
        prog = None
        if progress is not None:
            prog = (lambda it, n, info, ia=ia: progress(ia * n + it, n_alg * n, {**info, "algorithm": name}))
        obj, P, resid, fac, tsec, _, O = fpm_recover(prep.A, prep.geometry.row1, prep.geometry.col1, prep.P0, prep.supp,
                                                     prep.geometry.N_obj, name, opt, prep.n_bright,
                                                     progress=prog, should_cancel=should_cancel)
        if log:
            first = resid[min(cfg.n_warmup, resid.size - 1)]
            log(f"  {name:5s} residual {first:.4f} -> {resid[-1]:.4f}   {tsec:.1f} s")
        out.append(AlgoResult(name, obj, P, resid, fac, tsec, O))
    return FPMResult(prep, out, time.perf_counter() - t_all)


# ---------------------------------------------------------------------------- display / unwrapping
def flatten_params(o: np.ndarray, pixel_um: float, kappa0: float, refine: bool = False) -> dict:
    """Fit the point-source phase curvature (kappa, rad/um^2) and the tilt (cycles/pixel) of a
    reconstructed field on a reduced copy (as in the MATLAB display_phase). ``refine`` adds a
    parabolic sub-step fit of kappa around the best grid value (0.02·kappa0 steps leave up to
    ~0.2 rad of curvature at the edge of a 512-px patch) — used by the ML pipeline."""
    N = o.shape[0]
    s = max(1, N // 256)
    od = o[::s, ::s].astype(np.complex128)
    n = od.shape[0]
    c = (np.arange(n) - n / 2) * pixel_um * s
    x, y = np.meshgrid(c, c)
    r2 = x ** 2 + y ** 2
    ks = kappa0 * np.arange(0, 3.0001, 0.02)
    v = np.array([abs(np.sum(od * np.exp(-1j * k * r2))) for k in ks])
    i = int(np.argmax(v))
    kappa = float(ks[i])
    if refine and 0 < i < ks.size - 1:
        lo, hi = float(ks[i - 1]), float(ks[i + 1])
        for _ in range(30):                                    # golden-section search in [ks[i-1], ks[i+1]]
            m1, m2 = hi - 0.618 * (hi - lo), lo + 0.618 * (hi - lo)
            f1 = abs(np.sum(od * np.exp(-1j * m1 * r2))); f2 = abs(np.sum(od * np.exp(-1j * m2 * r2)))
            if f1 > f2:
                hi = m2
            else:
                lo = m1
        kappa = 0.5 * (lo + hi)
    F = _fftshift(np.fft.fft2(od * np.exp(-1j * kappa * r2)))
    ir, ic = np.unravel_index(int(np.argmax(np.abs(F))), F.shape)
    fr = (ir - n / 2) / (n * s)
    fc = (ic - n / 2) / (n * s)
    flat = o * _flat_phasor(o.shape[0], pixel_um, kappa, fr, fc)
    return {"kappa": kappa, "fr": float(fr), "fc": float(fc), "pixel_um": float(pixel_um),
            "phase0": float(np.angle(np.mean(flat)))}


def _flat_phasor(N: int, pixel_um: float, kappa: float, fr: float, fc: float) -> np.ndarray:
    cf = np.arange(N) - N / 2
    X, Y = np.meshgrid(cf, cf)
    return np.exp(-1j * (kappa * pixel_um ** 2 * (X ** 2 + Y ** 2) + 2 * np.pi * (fc * X + fr * Y)))


def apply_flatten(o: np.ndarray, fp: dict) -> np.ndarray:
    """Field without curvature, tilt and mean phase (the frame shown in the phase images)."""
    return o * _flat_phasor(o.shape[0], fp["pixel_um"], fp["kappa"], fp["fr"], fp["fc"]) * np.exp(-1j * fp["phase0"])


def undo_flatten(z: np.ndarray, fp: dict) -> np.ndarray:
    return z * np.conj(_flat_phasor(z.shape[0], fp["pixel_um"], fp["kappa"], fp["fr"], fp["fc"])) * np.exp(1j * fp["phase0"])


def display_phase(o: np.ndarray, pixel_um: float, kappa0: float) -> np.ndarray:
    """Phase without the point-source curvature and the tilt (fitted on a reduced copy), -pi..pi."""
    return np.angle(apply_flatten(o, flatten_params(o, pixel_um, kappa0)))


def flatten_field(o: np.ndarray, pixel_um: float, kappa0: float) -> np.ndarray:
    """Complex field with the curvature, tilt and mean phase removed (as used by display_phase)."""
    return apply_flatten(o, flatten_params(o, pixel_um, kappa0))


def object_spectrum(obj: np.ndarray, Np: int) -> np.ndarray:
    """Inverse of the engine's final step obj = ifft2(ifftshift(O)) * (N_obj/Np)^2."""
    N_obj = obj.shape[0]
    return _fftshift(_fft2(obj.astype(np.complex64) / (N_obj / Np) ** 2))


def fpm_data_error(obj: np.ndarray, pupil: np.ndarray, prep: "Prepared", fac: Optional[np.ndarray] = None,
                   frames: Optional[np.ndarray] = None) -> float:
    """Data residual sqrt(sum (fac*sqrt(I) - |F^-1{O_m P}|)^2 / sum (fac*sqrt(I))^2) of a high-resolution
    object against the prepared frames (all, or the indices in ``frames``). With ``fac`` = None the
    brightness factors that fit best for this object are used (as the engine does)."""
    A = prep.A
    Np = A.shape[1]
    O = object_spectrum(obj, Np)
    P = pupil.astype(np.complex64)
    num = den = 0.0
    idx = range(A.shape[0]) if frames is None else frames
    for m in idx:
        r, c = int(prep.geometry.row1[m]), int(prep.geometry.col1[m])
        aest = np.abs(_ifft2(_ifftshift(O[r:r + Np, c:c + Np] * P)))
        am = A[m]
        ok = ~np.isnan(am)
        a_ok, e_ok = am[ok].astype(np.float64), aest[ok].astype(np.float64)
        if fac is None:
            f = math.sqrt(float(np.mean(e_ok ** 2)) / max(float(np.mean(a_ok ** 2)), TINY))
        else:
            f = float(fac[m])
        num += float(np.sum((f * a_ok - e_ok) ** 2))
        den += float(np.sum((f * a_ok) ** 2))
    return math.sqrt(num / max(den, TINY))


def hybrid_fpm(prep: "Prepared", obj_init: np.ndarray, pupil: np.ndarray, passes: int, algo: str = "EPRY",
               progress: Optional[Callable] = None, should_cancel: Optional[Callable] = None) -> AlgoResult:
    """Physics polish seeded with a (refined) high-resolution object and the recovered pupil."""
    cfg = prep.cfg
    opt = EngineOptions(n_passes=int(passes), n_warmup=0, rpie_alpha=cfg.rpie_alpha, rpie_beta=cfg.rpie_beta,
                        intensity_correction=cfg.intensity_correction, intensity_limit=cfg.intensity_limit)
    O0 = object_spectrum(obj_init, prep.A.shape[1])
    obj, P, resid, fac, tsec, _, O = fpm_recover(prep.A, prep.geometry.row1, prep.geometry.col1, pupil, prep.supp,
                                                 prep.geometry.N_obj, algo, opt, prep.n_bright, init_spectrum=O0,
                                                 progress=progress, should_cancel=should_cancel)
    return AlgoResult(f"{algo}+hybrid", obj, P, resid, fac, tsec, O)


def unwrap_phase_ls(psi: np.ndarray) -> np.ndarray:
    """Unweighted least-squares phase unwrapping (Ghiglia & Romero) with DCT Poisson solver."""
    from scipy.fft import dctn, idctn
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    dx = wrap(np.diff(psi, axis=1)); dy = wrap(np.diff(psi, axis=0))
    rho = np.zeros_like(psi)
    rho[:, :-1] += dx; rho[:, 1:] -= dx
    rho[:-1, :] += dy; rho[1:, :] -= dy
    n1, n2 = psi.shape
    R = dctn(rho, type=2, norm="ortho")
    i = np.arange(n1)[:, None]; j = np.arange(n2)[None, :]
    den = 2 * (np.cos(np.pi * i / n1) + np.cos(np.pi * j / n2) - 2)
    den[0, 0] = 1.0
    Phi = R / den
    Phi[0, 0] = 0.0
    return idctn(Phi, type=2, norm="ortho")


def _to_u16(x: np.ndarray) -> np.ndarray:
    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx > 0:
        return np.clip(np.nan_to_num(x) / mx * 65535, 0, 65535).astype(np.uint16)
    return np.zeros(x.shape, np.uint16)


def _write_tif(path: Path, a: np.ndarray):
    try:
        import tifffile
        tifffile.imwrite(str(path), a)
    except Exception:
        from PIL import Image
        Image.fromarray(a).save(path)


def save_fpm_results(res: FPMResult, out_dir: str | Path, figure: bool = True, unwrap: bool = True) -> Dict[str, str]:
    """Same files as the MATLAB script (TIFF images, frames.csv, summary.txt, comparison.png) plus result.npz."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p = res.prepared
    files: Dict[str, str] = {}
    store = {}
    for a in res.algorithms:
        amp = np.abs(a.obj)
        ph = display_phase(a.obj, res.pixel_um, res.curvature_rad_um2)
        _write_tif(out / f"{a.name}_amplitude.tif", _to_u16(amp))
        _write_tif(out / f"{a.name}_phase.tif", ((ph + np.pi) / (2 * np.pi) * 65535).astype(np.uint16))
        pp = np.angle(a.pupil) * p.supp
        _write_tif(out / f"{a.name}_pupil_phase.tif", ((pp + np.pi) / (2 * np.pi) * 65535).astype(np.uint16))
        files[f"{a.name}_amplitude"] = f"{a.name}_amplitude.tif"
        files[f"{a.name}_phase"] = f"{a.name}_phase.tif"
        store[f"{a.name}_object"] = a.obj.astype(np.complex64)
        store[f"{a.name}_pupil"] = a.pupil.astype(np.complex64)
        store[f"{a.name}_residual"] = a.residual
        store[f"{a.name}_factor"] = a.intensity_factor
        store[f"{a.name}_phase_flat"] = ph.astype(np.float32)
        if unwrap:
            uw = unwrap_phase_ls(ph.astype(np.float64))
            store[f"{a.name}_phase_unwrapped"] = uw.astype(np.float32)
    raw = np.sqrt(np.maximum(np.nan_to_num(p.raw_lowest_angle), 0))
    _write_tif(out / "raw_lowest_angle_amplitude.tif", _to_u16(raw))
    np.savez_compressed(out / "result.npz", pupil_support=p.supp.astype(np.uint8), pixel_um=res.pixel_um,
                        raw_pixel_um=p.dpix_um, raw_lowest_angle=p.raw_lowest_angle.astype(np.float32),
                        summary_json=json.dumps(res.summary(), default=float), **store)
    files["result_npz"] = "result.npz"
    F = p.frames_table
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "led_index", "file", "pixel_x", "pixel_y", "illumination_na", "shift_row_px", "shift_col_px",
                    "exposure_us", "mean_signal", "direct_beam"] + [f"factor_{a.name}" for a in res.algorithms])
        for m in range(len(F["led_index"])):
            db = F["direct_beam"][m]
            w.writerow([m + 1, F["led_index"][m], F["file"][m], F["pixel_x"][m], F["pixel_y"][m],
                        f"{F['illumination_na'][m]:.6f}", F["shift_row"][m], F["shift_col"][m], f"{F['exposure_us'][m]:g}",
                        f"{F['mean_signal'][m]:.6g}", "" if db is None else f"{db:.3f}"] +
                       [f"{a.intensity_factor[m]:.4f}" for a in res.algorithms])
    files["frames_csv"] = "frames.csv"
    c = p.cfg
    lines = ["LED Fourier ptychography (Velos platform, Python port of the MATLAB quick check): "
             + " / ".join(a.name for a in res.algorithms),
             f"Dataset: {p.patch.get('dataset', '')}",
             f"Lambda (um): {c.lambda_um:.6f}", f"Objective NA: {c.objective_NA:.6f}", f"Magnification: {c.magnification:.6f}",
             f"LED pitch (mm): {c.led_pitch_mm:.6f}", f"LED to sample (mm): {c.z_source_mm:.6f}",
             f"Optical-axis offset (mm): {c.optical_axis_x_mm:.6f} {c.optical_axis_y_mm:.6f}",
             f"swap_led_xy: {int(c.swap_led_xy)}", f"led_x_sign: {c.led_x_sign}", f"led_y_sign: {c.led_y_sign}",
             f"Np: {c.Np}", f"N_obj: {p.geometry.N_obj}", f"Pixel of the result (um): {res.pixel_um:.6f}",
             f"Synthetic NA: {p.geometry.synthetic_NA:.6f}", f"Frames used: {p.A.shape[0]}",
             f"Passes: {c.n_warmup} warm-up + {c.n_passes}", f"Intensity correction: {int(c.intensity_correction)}"]
    T = p.transition
    lines.append(f"Bright/dark transition LEDs ({len(T['pixel_x'])}, used: {int(T['used'])}):" +
                 "".join(f" ({int(x)},{int(y)}) {'NaN' if d is None else f'{d:.2f}'}"
                         for x, y, d in zip(T["pixel_x"], T["pixel_y"], T["direct_beam"])))
    lines.append("Computed on: CPU (NumPy/SciPy)")
    for a in res.algorithms:
        lines.append(f"{a.name}: final data residual {a.residual[-1]:.6f}, time {a.time_s:.2f} s")
    g = p.geometry_check
    if g.get("done"):
        lines.append(f"Geometry check: z {g['z_best']:.2f} mm, swap_led_xy {int(g['swap_led_xy'])}, led_y_sign {g['led_y_sign']:+d}, "
                     f"optical axis ({g['optical_axis_mm'][0]:.2f}, {g['optical_axis_mm'][1]:.2f}) mm, misfit "
                     f"{g['residual_best']:.4f} (as set {g['residual_as_set']:.4f}), applied {int(g['applied'])}")
    lines.append(f"Check list ({len(p.notes)}):")
    lines += [f"  {k + 1}. {n}" for k, n in enumerate(p.notes)]
    lines.append(f"Total time (s): {res.total_s:.1f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    files["summary_txt"] = "summary.txt"
    if figure:
        try:
            files["comparison"] = Path(save_comparison_figure(res, out / "comparison.png")).name
        except Exception as e:  # pragma: no cover - figure only
            (out / "figure_error.txt").write_text(str(e))
    return files


def save_comparison_figure(res: FPMResult, path: str | Path) -> str:
    """One figure like the MATLAB result window: amplitude / phase / pupil phase per algorithm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = res.prepared
    nA = len(res.algorithms)
    fig, axes = plt.subplots(3, nA + 1, figsize=(4.2 * (nA + 1), 11), dpi=110)
    raw = np.sqrt(np.maximum(np.nan_to_num(p.raw_lowest_angle), 0))
    ax = axes[0, 0]; ax.imshow(raw, cmap="gray"); ax.set_title(f"raw frame, lowest angle ({p.cfg.Np} px)"); ax.axis("off")
    amax = max(np.quantile(np.abs(a.obj), 0.999) for a in res.algorithms)
    for j, a in enumerate(res.algorithms, start=1):
        ax = axes[0, j]; ax.imshow(np.abs(a.obj), cmap="gray", vmin=0, vmax=amax)
        ax.set_title(f"{a.name} amplitude (residual {a.residual[-1]:.4f}, {a.time_s:.0f} s)", fontsize=9); ax.axis("off")
        ax = axes[1, j]; ax.imshow(display_phase(a.obj, res.pixel_um, res.curvature_rad_um2), cmap="gray", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"{a.name} phase, -π … π (curvature removed)", fontsize=9); ax.axis("off")
        ax = axes[2, j]
        pp = np.angle(a.pupil) * p.supp
        n = pp.shape[0]; c = n // 2; hw = min(n // 2 - 1, int(math.ceil(1.25 * math.sqrt(p.supp.sum() / math.pi))))
        im = ax.imshow(pp[c - hw:c + hw + 1, c - hw:c + hw + 1], cmap="jet", vmin=-np.pi / 2, vmax=np.pi / 2)
        ax.set_title(f"{a.name} pupil phase [rad]", fontsize=9); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1, 0]
    F = p.frames_table
    sc = ax.scatter(F["pixel_x"], F["pixel_y"], c=np.log10(np.maximum(np.array(F["mean_signal"]), 1e-6)), s=22)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xlabel("panel pixel x"); ax.set_ylabel("panel pixel y")
    ax.set_title("LEDs used: log10 mean signal", fontsize=9); fig.colorbar(sc, ax=ax, fraction=0.046)
    ax = axes[2, 0]
    for a, sty in zip(res.algorithms, ("-o", "-s", "-^", "-d")):
        ax.plot(np.arange(1, a.residual.size + 1), a.residual, sty, ms=3, label=a.name)
    ax.set_xlabel("pass"); ax.set_ylabel("data residual"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"convergence ({p.A.shape[0]} frames)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


# ============================================================================ simulation (tests, demos)
def simulate_fpm(n_low: int = 64, upsample: int = 3, grid: int = 9, lambda_um: float = 0.632, NA: float = 0.10,
                 mag: float = 4.0, camera_pixel_um: float = 5.04, z_mm: float = 55.0, pitch_mm: float = 3.0,
                 photons: float = 0.0, pupil_aberration: float = 0.0, seed: int = 0, obj: Optional[np.ndarray] = None,
                 read_noise: float = 1.5, texture: float = 0.25):
    """Simulate an LED scan of a phase/amplitude object with the same geometry as the engine.

    Returns (acquisition, ground-truth high-resolution object, config). Intensities are in grey
    values with an offset of 60 (like the CS2100M) and, if ``photons`` > 0, Poisson noise.
    """
    rng = np.random.default_rng(seed)
    N = n_low * upsample
    if obj is None:
        yy, xx = np.mgrid[0:N, 0:N] / N
        ph = np.zeros((N, N)); am = np.ones((N, N))
        for _ in range(14):
            cy, cx, r = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85), rng.uniform(0.03, 0.12)
            d2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / r ** 2
            ph += rng.uniform(0.3, 1.2) * np.exp(-d2 ** 2)
            am -= rng.uniform(0.0, 0.25) * np.exp(-d2 ** 2)
        if texture:
            from scipy.ndimage import gaussian_filter
            t = gaussian_filter(rng.standard_normal((N, N)), 1.2)
            ph += texture * t / (t.std() + 1e-12)
        obj = np.clip(am, 0.3, 1) * np.exp(1j * ph)
    cfg = FPMConfig(lambda_um=lambda_um, objective_NA=NA, camera_pixel_um=camera_pixel_um, magnification=mag,
                    z_source_mm=z_mm, led_pitch_mm=pitch_mm, Np=n_low, upsample=upsample, geometry_check="off",
                    init_spherical_phase=False)
    c0 = grid // 2
    pix_x, pix_y = [], []
    for py in range(c0 + grid // 2, c0 - grid // 2 - 1, -1):           # raster like the acquisition software
        for px in range(c0 + grid // 2, c0 - grid // 2 - 1, -1):
            pix_x.append(px); pix_y.append(py)
    pix_x = np.array(pix_x, float); pix_y = np.array(pix_y, float)
    led_x = (pix_x - c0) * pitch_mm
    led_y = (c0 - pix_y) * pitch_mm
    dpix = camera_pixel_um / mag
    G = led_geometry(cfg, led_x, led_y, 0.0, 0.0, n_low, dpix, upsample)
    P, supp = start_pupil(cfg, n_low, G.du, G.pupil_radius_px)
    if pupil_aberration:
        coord = (np.arange(n_low) - n_low / 2) / G.pupil_radius_px
        X, Y = np.meshgrid(coord, coord)
        P = supp * np.exp(1j * pupil_aberration * (2 * (X ** 2 + Y ** 2) - 1))
    S = _fftshift(np.fft.fft2(obj))
    frames = []
    for m in range(len(pix_x)):
        r, c = int(G.row1[m]), int(G.col1[m])
        low = np.fft.ifft2(_ifftshift(S[r:r + n_low, c:c + n_low] * P)) / (upsample ** 2)
        I = np.abs(low) ** 2
        frames.append(I)
    frames = np.array(frames)
    scale = 20000.0 / frames.max()
    frames = frames * scale
    if photons > 0:
        frames = rng.poisson(frames * photons / 20000.0) * (20000.0 / photons)
    if read_noise:
        frames = frames + rng.normal(0, read_noise, frames.shape)
    frames = np.clip(np.round(frames + 60), 0, 65535).astype(np.uint16)
    files = [f"led_{i + 1:03d}_x{int(x)}_y{int(y)}.tif" for i, (x, y) in enumerate(zip(pix_x, pix_y))]
    acq = FPMAcquisition(frames=frames, files=files, pixel_x=pix_x, pixel_y=pix_y, led_x_mm=led_x, led_y_mm=led_y,
                         exposure_us=np.ones(len(files)), gain=np.ones(len(files)), frame_height=n_low,
                         frame_width=n_low, run_info={"led_to_sample_mm": z_mm, "led_pitch_mm": pitch_mm,
                                                      "center_led_index": len(files) // 2 + 1, "simulated": True},
                         name="simulated_fpm", defaults={"lambda_um": lambda_um, "objective_NA": NA,
                                                         "magnification": mag, "z_source_mm": z_mm, "Np": n_low})
    return acq, obj, cfg


# ============================================================================ CLI
def _cfg_from_args(a, acq: FPMAcquisition) -> FPMConfig:
    d = dict(acq.defaults or {})
    for k, v in (("lambda_um", a.lam), ("z_source_mm", a.z), ("Np", a.Np), ("n_passes", a.passes), ("n_warmup", a.warmup),
                 ("objective_NA", a.NA), ("magnification", a.mag), ("geometry_check", a.geometry_check),
                 ("optical_axis_x_mm", a.axis_x), ("optical_axis_y_mm", a.axis_y), ("upsample", a.upsample)):
        if v is not None:
            d[k] = v
    if a.algorithms:
        d["algorithms"] = a.algorithms
    return FPMConfig.from_dict(d)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="velos-fpm", description="LED-array Fourier ptychography (AP / EPRY / rPIE)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="reconstruct one patch")
    r.add_argument("source", help="acquisition folder, *_meta.npz (packed) or Velos FPM .npz")
    r.add_argument("--out", default=None)
    r.add_argument("--Np", type=int, default=None)
    r.add_argument("--z", type=float, default=None, help="LED-to-sample distance [mm]")
    r.add_argument("--lambda", dest="lam", type=float, default=None, help="wavelength [um]")
    r.add_argument("--NA", type=float, default=None)
    r.add_argument("--mag", type=float, default=None)
    r.add_argument("--passes", type=int, default=None)
    r.add_argument("--warmup", type=int, default=None)
    r.add_argument("--upsample", type=int, default=None)
    r.add_argument("--axis-x", dest="axis_x", type=float, default=None)
    r.add_argument("--axis-y", dest="axis_y", type=float, default=None)
    r.add_argument("--algorithms", nargs="+", default=None)
    r.add_argument("--geometry-check", dest="geometry_check", default=None, choices=["off", "report", "apply"])
    c = sub.add_parser("convert", help="folder or packed crop -> single Velos FPM .npz (API upload format)")
    c.add_argument("source"); c.add_argument("dest"); c.add_argument("--crop", type=int, default=None)
    c.add_argument("--set", nargs="*", default=[], help="setup defaults stored with the data, e.g. lambda_um=0.633 z_source_mm=55")
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        acq = load_fpm(a.source, crop=a.crop)
        defaults = {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            defaults[k] = v
        FPMConfig.from_dict(defaults)
        save_fpm_npz(acq, a.dest, defaults)
        print(json.dumps(acq.summary(), indent=2, default=float))
        return 0
    acq = load_fpm(a.source)
    cfg = _cfg_from_args(a, acq)
    res = reconstruct_fpm(acq, cfg, log=print)
    out = Path(a.out or (Path(a.source).with_suffix("") if not Path(a.source).is_dir() else Path(a.source) / "reconstruction_velos"))
    res.prepared.patch["dataset"] = acq.name
    files = save_fpm_results(res, out)
    print(json.dumps({"out": str(out), "files": files,
                      "residuals": {x.name: float(x.residual[-1]) for x in res.algorithms}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
