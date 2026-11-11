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
    os.environ["VELOS_MODEL_DIR"] = str(tmp_root / "models")
    make_identity_package(tmp_root / "models" / "identity-test", tile=64)
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


def make_identity_package(path: Path, tile: int = 64) -> Path:
    """A model package whose ONNX graph returns its input's (Re, Im) channels unchanged.

    It exercises the complete runtime path (normalise → T×T tiles → onnxruntime → Hann blend
    → denormalise → hybrid polish) with a known answer: the refined object must equal the
    physics object."""
    import json
    import onnx
    from onnx import TensorProto, helper
    path.mkdir(parents=True, exist_ok=True)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, tile, tile])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, tile, tile])
    inits = [helper.make_tensor(n, TensorProto.INT64, [1], [v]) for n, v in
             (("starts", 0), ("ends", 2), ("axes", 1))]
    node = helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["y"])
    graph = helper.make_graph([node], "identity", [x], [y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path / "model.onnx"))
    card = {"name": "identity-test", "version": "0.0.0", "tile": tile, "tile_overlap": tile // 4,
            "normalisation": "median_ramp", "inputs": "x[1,3,T,T] (Re, Im, mask)",
            "outputs": "y[1,2,T,T] (Re, Im)", "intended_use": "tests only",
            "data_error_ratio_warn": 2.0, "data_error_abs_floor": 0.02}
    (path / "model_card.json").write_text(json.dumps(card, indent=1))
    return path
