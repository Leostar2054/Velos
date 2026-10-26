import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def tmp_root():
    d = tempfile.mkdtemp(prefix="velos-test-")
    yield Path(d)


@pytest.fixture(scope="session", autouse=True)
def _env(tmp_root):
    os.environ["VELOS_STORAGE_DIR"] = str(tmp_root / "storage")
    os.environ["VELOS_API_KEYS"] = "test-user-key:alice:user,test-admin-key:root:admin"
    os.environ["VELOS_LOG_LEVEL"] = "WARNING"
    from velos_backend.config import reset_settings
    from velos_backend.db import reset_engine
    reset_settings(); reset_engine()
    yield


def make_synthetic_npz(path: Path, N: int = 96, Np: int = 32, overlap: float = 0.6, photons: float = 1e5):
    """Small synthetic dataset built with the Client's simulation code (ground truth included)."""
    from velos_recon import core, save_npz
    from velos_recon.data import PtychoDataset
    O = core.make_object(N=N, seed=1)
    P = core.make_probe(Np=Np)
    pos = core.make_positions(N, Np, overlap=overlap, jitter=0.0, seed=1)
    amps = core.simulate(O, P, pos, photons=photons, seed=1)
    dx = 1e-6; wavelength = 1e-6; distance = 1e-2
    pitch = wavelength * distance / (Np * dx)
    positions_m = np.stack([pos[:, 1], pos[:, 0]], 1).astype(float) * dx   # (x, y)
    ds = PtychoDataset(dps=(amps ** 2).astype(np.float32), positions_m=positions_m,
                       wavelength=wavelength, pixel_pitch=pitch, distance=distance,
                       init_probe=P, ground_truth=O, name="synthetic")
    save_npz(ds, path)
    return path


@pytest.fixture(scope="session")
def synthetic_npz(tmp_root):
    return make_synthetic_npz(tmp_root / "synthetic.npz")
