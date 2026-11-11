"""
Dataset loading and geometry for scanning ptychography.

Supported input formats
-----------------------
1. **MATLAB** ``.mat`` (v5/v7, as used by the Client's existing datasets):
   a struct ``expt`` with fields ``dps`` (M×N×J intensities, centred DC),
   ``positions.x`` / ``positions.y`` (metres), ``wavelength`` (m),
   ``cameraPixelPitch`` (m), ``cameraLength`` (m); optional ``initProbe``.
2. **NumPy** ``.npz`` with keys ``dps`` (J×M×N), ``positions`` (J×2, metres,
   columns = x, y), ``wavelength``, ``pixel_pitch``, ``distance``;
   optional ``init_probe``, ``ground_truth`` (complex object).
3. **HDF5** ``.h5``/``.hdf5`` with the same key names as the ``.npz`` layout.

All loaders return a :class:`PtychoDataset` with diffraction patterns stored
as ``dps[J, M, N]`` (float32, DC centred) and positions in metres.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class PtychoDataset:
    dps: np.ndarray                     # [J, M, N] intensities, DC centred
    positions_m: np.ndarray             # [J, 2] (x, y) in metres
    wavelength: float                   # metres
    pixel_pitch: float                  # detector pixel pitch, metres
    distance: float                     # sample→detector distance (or camera length), metres
    init_probe: Optional[np.ndarray] = None       # [M, N] complex or mask
    ground_truth: Optional[np.ndarray] = None     # complex object (simulations only)
    name: str = "dataset"
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ geometry
    @property
    def n_patterns(self) -> int:
        return int(self.dps.shape[0])

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return int(self.dps.shape[1]), int(self.dps.shape[2])

    @property
    def dx(self) -> float:
        """Sample-plane pixel size (m) for a far-field geometry: λ·L / (N·p)."""
        n = self.dps.shape[-1]
        return float(self.wavelength * self.distance / (n * self.pixel_pitch))

    def positions_px(self) -> np.ndarray:
        """Integer top-left (row, col) pixel offsets of each probe frame, zero based.

        Convention: for experimental data the object frame starts at the first scan
        position (positions are shifted to a zero minimum, as in the Client's MATLAB
        code). For simulated data that carries ``ground_truth``, positions are absolute
        pixel coordinates in the ground-truth frame so metrics can be computed directly.
        """
        x = self.positions_m[:, 0]; y = self.positions_m[:, 1]
        if self.ground_truth is None:
            # a subset keeps its parent's scan origin (meta["origin_m"]) so reduced-scan
            # reconstructions land on exactly the same pixel grid as the full scan
            ox, oy = self.meta.get("origin_m", (x.min(), y.min()))
            x = x - ox; y = y - oy
        col = np.round(x / self.dx).astype(int)
        row = np.round(y / self.dx).astype(int)
        return np.stack([row, col], axis=1)

    def object_shape(self) -> Tuple[int, int]:
        if self.ground_truth is not None:
            return tuple(int(s) for s in self.ground_truth.shape)
        pos = self.positions_px()
        m, n = self.frame_shape
        rmax, cmax = int(pos[:, 0].max()), int(pos[:, 1].max())
        if "far_m" in self.meta and "origin_m" in self.meta:     # subset: parent's field of view
            (fx, fy), (ox, oy) = self.meta["far_m"], self.meta["origin_m"]
            rmax = max(rmax, int(round((fy - oy) / self.dx)))
            cmax = max(cmax, int(round((fx - ox) / self.dx)))
        return rmax + m, cmax + n

    def summary(self) -> dict:
        pos = self.positions_px()
        step = _median_step(pos)
        return {
            "name": self.name,
            "n_patterns": self.n_patterns,
            "frame_shape": list(self.frame_shape),
            "wavelength_nm": self.wavelength * 1e9,
            "pixel_pitch_um": self.pixel_pitch * 1e6,
            "distance_mm": self.distance * 1e3,
            "sample_pixel_um": self.dx * 1e6,
            "object_shape": list(self.object_shape()),
            "median_scan_step_px": step,
            "has_init_probe": self.init_probe is not None,
            "has_ground_truth": self.ground_truth is not None,
            "dps_max": float(self.dps.max()),
            "dps_mean": float(self.dps.mean()),
        }

    # ------------------------------------------------------------------ transforms
    def center_crop(self, size: int) -> "PtychoDataset":
        """Crop every diffraction pattern to the central ``size×size`` pixels.

        Cropping keeps the real-space frame extent (λL/p) and coarsens the
        sample-plane pixel (dx grows by M/size). It is used for fast previews
        on CPU; full-resolution runs use the uncropped data.
        """
        m, n = self.frame_shape
        if size >= min(m, n):
            return self
        r0, c0 = (m - size) // 2, (n - size) // 2
        dps = np.ascontiguousarray(self.dps[:, r0:r0 + size, c0:c0 + size])
        probe = None
        if self.init_probe is not None:
            # The initial probe lives in REAL space: a Fourier-plane crop keeps the frame's
            # physical extent (λL/p) but coarsens the pixel, so the probe must be resampled
            # (not cropped) to keep its physical size.
            probe = _resample_probe(np.asarray(self.init_probe), size)
        meta = dict(self.meta, crop=size, original_frame=[m, n])
        return dataclasses.replace(self, dps=dps, init_probe=probe, meta=meta)

    def bin(self, factor: int) -> "PtychoDataset":
        """Bin every diffraction pattern by ``factor×factor`` (sum of intensities).

        Binning keeps all scattering angles (same sample-plane pixel dx) and shrinks the
        real-space frame extent (λL/p) by ``factor`` — the frame must still contain the
        probe. Preferred over cropping for fast CPU runs of experimental data.
        """
        if factor <= 1:
            return self
        m, n = self.frame_shape
        mb, nb = (m // factor) * factor, (n // factor) * factor
        r0, c0 = (m - mb) // 2, (n - nb) // 2
        d = self.dps[:, r0:r0 + mb, c0:c0 + nb]
        dps = d.reshape(d.shape[0], mb // factor, factor, nb // factor, factor).sum(axis=(2, 4))
        probe = None
        if self.init_probe is not None:
            # real-space frame shrinks: keep the central part of the probe at the same dx
            p = np.asarray(self.init_probe)
            pr0, pc0 = (p.shape[0] - mb // factor) // 2, (p.shape[1] - nb // factor) // 2
            probe = np.ascontiguousarray(p[pr0:pr0 + mb // factor, pc0:pc0 + nb // factor])
        meta = dict(self.meta, bin=factor, original_frame=[m, n])
        return dataclasses.replace(self, dps=np.ascontiguousarray(dps.astype(np.float32)),
                                   pixel_pitch=self.pixel_pitch * factor, init_probe=probe, meta=meta)

    def subset(self, index: np.ndarray, name_suffix: str = "") -> "PtychoDataset":
        """Keep only the given pattern indices (used for sparse-scan experiments)."""
        index = np.asarray(index)
        x, y = self.positions_m[:, 0], self.positions_m[:, 1]
        meta = dict(self.meta, subset=len(index))
        meta.setdefault("origin_m", (float(x.min()), float(y.min())))   # keep the parent grid
        meta.setdefault("far_m", (float(x.max()), float(y.max())))
        return dataclasses.replace(
            self, dps=self.dps[index], positions_m=self.positions_m[index],
            name=self.name + name_suffix, meta=meta)

    def amplitudes(self) -> np.ndarray:
        """Measured amplitudes √I as float64 [J, M, N] (what the engines consume)."""
        return np.sqrt(np.maximum(self.dps, 0.0)).astype(np.float64)

    def grid_indices(self) -> np.ndarray:
        """Assign each scan position a (row, col) grid index for (jittered) raster scans.

        Rows are formed by sorting y into ``n_rows = round(sqrt(J))`` groups; columns by
        sorting x within each row. Works for serpentine and jittered rasters.
        """
        j = self.n_patterns
        n_rows = int(round(np.sqrt(j)))
        order_y = np.argsort(self.positions_m[:, 1], kind="stable")
        grid = np.zeros((j, 2), dtype=int)
        for r, chunk in enumerate(np.array_split(order_y, n_rows)):
            chunk = chunk[np.argsort(self.positions_m[chunk, 0], kind="stable")]
            grid[chunk, 0] = r
            grid[chunk, 1] = np.arange(len(chunk))
        return grid

    def sparse_subset(self, every: int = 2, fraction: Optional[float] = None,
                      seed: int = 0) -> "PtychoDataset":
        """Sparse-scan subset: keep every ``every``-th grid row and column, or a random
        ``fraction`` of positions when given. Returns a new dataset (positions preserved)."""
        if fraction is not None:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(self.n_patterns, int(round(fraction * self.n_patterns)), replace=False))
            return self.subset(keep, f"_rand{fraction:.2f}")
        g = self.grid_indices()
        keep = np.where((g[:, 0] % every == 0) & (g[:, 1] % every == 0))[0]
        return self.subset(keep, f"_every{every}")


def _resample_probe(p: np.ndarray, size: int) -> np.ndarray:
    """Resample a real-space probe (complex or mask) to size×size by block averaging
    (integer factors) or Fourier-domain interpolation (other factors)."""
    m, n = p.shape
    if m == size and n == size:
        return p
    if m % size == 0 and n % size == 0:
        f = m // size
        out = p.reshape(size, f, size, f).mean(axis=(1, 3))
        return out.astype(np.complex128) if np.iscomplexobj(p) else out.astype(float)
    # generic: crop the Fourier transform (band-limit) then inverse-transform at the new size
    P = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p.astype(np.complex128))))
    r0, c0 = (m - size) // 2, (n - size) // 2
    Pc = P[r0:r0 + size, c0:c0 + size]
    out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Pc))) * (size * size) / (m * n)
    return out if np.iscomplexobj(p) else np.abs(out)


def _median_step(pos: np.ndarray) -> float:
    if len(pos) < 2:
        return 0.0
    d = np.sqrt(((pos[None, :, :] - pos[:, None, :]) ** 2).sum(-1)).astype(float)
    d[d == 0] = np.inf
    return float(np.median(d.min(axis=1)))


# ---------------------------------------------------------------------- loading
def load_dataset(path: str | Path, name: Optional[str] = None) -> PtychoDataset:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        ds = _load_mat(path)
    elif suffix == ".npz":
        ds = _load_npz(path)
    elif suffix in (".h5", ".hdf5"):
        ds = _load_h5(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    ds.name = name or path.stem
    _validate(ds)
    return ds


def _load_mat(path: Path) -> PtychoDataset:
    import scipy.io as sio

    try:
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:  # MATLAB v7.3 → HDF5
        return _load_mat73(path)
    if "expt" not in d:
        raise ValueError("MAT file has no 'expt' struct")
    e = d["expt"]
    dps = np.asarray(e.dps)
    if dps.ndim != 3:
        raise ValueError("expt.dps must be M×N×J")
    dps = np.ascontiguousarray(np.moveaxis(dps, 2, 0)).astype(np.float32)
    pos = np.stack([np.asarray(e.positions.x, float).ravel(),
                    np.asarray(e.positions.y, float).ravel()], axis=1)
    probe = d.get("initProbe")
    if probe is not None:
        probe = np.asarray(probe)
    return PtychoDataset(
        dps=dps, positions_m=pos, wavelength=float(e.wavelength),
        pixel_pitch=float(e.cameraPixelPitch), distance=float(e.cameraLength),
        init_probe=probe, meta={"source": "mat"})


def _load_mat73(path: Path) -> PtychoDataset:
    import h5py

    with h5py.File(path, "r") as h:
        e = h["expt"]
        dps = np.asarray(e["dps"])            # h5py gives J×N×M (MATLAB column-major)
        dps = np.ascontiguousarray(np.transpose(dps, (0, 2, 1))).astype(np.float32)
        pos = np.stack([np.asarray(e["positions"]["x"]).ravel(),
                        np.asarray(e["positions"]["y"]).ravel()], axis=1)
        probe = np.asarray(h["initProbe"]).T if "initProbe" in h else None
        return PtychoDataset(
            dps=dps, positions_m=pos, wavelength=float(np.asarray(e["wavelength"]).ravel()[0]),
            pixel_pitch=float(np.asarray(e["cameraPixelPitch"]).ravel()[0]),
            distance=float(np.asarray(e["cameraLength"]).ravel()[0]),
            init_probe=probe, meta={"source": "mat73"})


def _load_npz(path: Path) -> PtychoDataset:
    with np.load(path, allow_pickle=False) as z:
        ds = PtychoDataset(
            dps=np.asarray(z["dps"], dtype=np.float32),
            positions_m=np.asarray(z["positions"], dtype=float),
            wavelength=float(z["wavelength"]), pixel_pitch=float(z["pixel_pitch"]),
            distance=float(z["distance"]),
            init_probe=np.asarray(z["init_probe"]) if "init_probe" in z else None,
            ground_truth=np.asarray(z["ground_truth"]) if "ground_truth" in z else None,
            meta={"source": "npz"})
    return ds


def _load_h5(path: Path) -> PtychoDataset:
    import h5py

    with h5py.File(path, "r") as h:
        ds = PtychoDataset(
            dps=np.asarray(h["dps"], dtype=np.float32),
            positions_m=np.asarray(h["positions"], dtype=float),
            wavelength=float(h["wavelength"][()]), pixel_pitch=float(h["pixel_pitch"][()]),
            distance=float(h["distance"][()]),
            init_probe=np.asarray(h["init_probe"]) if "init_probe" in h else None,
            ground_truth=np.asarray(h["ground_truth"]) if "ground_truth" in h else None,
            meta={"source": "h5"})
    return ds


def save_npz(ds: PtychoDataset, path: str | Path) -> Path:
    path = Path(path)
    extra = {}
    if ds.init_probe is not None:
        extra["init_probe"] = ds.init_probe
    if ds.ground_truth is not None:
        extra["ground_truth"] = ds.ground_truth
    np.savez_compressed(path, dps=ds.dps.astype(np.float32), positions=ds.positions_m,
                        wavelength=ds.wavelength, pixel_pitch=ds.pixel_pitch,
                        distance=ds.distance, **extra)
    return path


def _validate(ds: PtychoDataset) -> None:
    if ds.dps.ndim != 3:
        raise ValueError("dps must be [J, M, N]")
    if ds.positions_m.shape != (ds.dps.shape[0], 2):
        raise ValueError("positions must be [J, 2] and match the number of patterns")
    if not np.isfinite(ds.dps).all():
        raise ValueError("dps contain NaN/Inf")
    if ds.dps.min() < 0:
        raise ValueError("dps contain negative intensities")
    for k in ("wavelength", "pixel_pitch", "distance"):
        if getattr(ds, k) <= 0:
            raise ValueError(f"{k} must be positive")
