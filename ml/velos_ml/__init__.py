"""
velos_ml — config-driven PyTorch pipeline for ML-refined ptychographic reconstruction.

Task (first baseline, M2): *sparse-scan refinement*. The physics engine reconstructs
amplitude/phase from a sparse scan (≈60 % fewer diffraction frames than a dense scan);
a U-Net learns to map that reconstruction toward the ground truth, i.e. toward what a
dense scan would have given. This directly supports the Velos QPI value proposition
(fewer measurements, same quality) and is evaluated against ground truth AND the
Client's physics-based baseline with PSNR / SSIM (Contract §3.3, §4).
"""

__version__ = "0.2.0"
