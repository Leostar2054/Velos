"""
velos_recon
===========
Reconstruction package for the Velos BioImaging / SPARK platform.

It integrates the Client's validated physics-based ptychography engines
(``velos_recon.core`` — vendored unchanged from ``ptycho_core.py``) behind a
small, stable interface used by the backend job worker, the CLI and the
ML pipeline:

* :mod:`velos_recon.data`    – dataset loading (.mat Maiden ``expt`` format,
  .npz, .h5), geometry (sample-plane pixel size, scan positions → pixels),
  centre-cropping for fast previews.
* :mod:`velos_recon.engine`  – ``reconstruct()`` drives a chosen engine
  (ePIE / rPIE / WASP / GD / MAP) with progress callbacks and returns a
  :class:`ReconResult` (object, probe, mask, per-iteration error, timings).
* :mod:`velos_recon.metrics` – PSNR / SSIM / NRMSE on amplitude and phase,
  with the complex-scalar ambiguity removed first.
* :mod:`velos_recon.viz`     – PNG / figure export for results.
"""

from .data import PtychoDataset, load_dataset, save_npz  # noqa: F401
from .engine import ReconConfig, ReconResult, reconstruct, ENGINE_NAMES  # noqa: F401
from .metrics import amplitude_phase_metrics, psnr, ssim  # noqa: F401

__version__ = "0.2.0"
