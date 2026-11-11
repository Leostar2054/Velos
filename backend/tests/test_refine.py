"""M3: normalisation, model-package runtime (ONNX), tiling, hybrid mode, subset geometry."""
import dataclasses

import numpy as np
import pytest

from velos_recon import ReconConfig, core, load_dataset, reconstruct
from velos_recon.data import PtychoDataset
from velos_recon.engine import data_error
from velos_recon.normalise import denormalise, normalise
from velos_recon.refine import hybrid, list_packages, load_package, refine


def _test_object(h=150, w=230, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 25 ** 2))
    obj = (0.9 - 0.3 * blob) * np.exp(1j * (1.2 * blob + 0.04 * xx - 0.03 * yy + 0.7))
    obj *= 2.5 * (1 + 0.01 * rng.standard_normal((h, w)))
    mask = np.zeros((h, w), bool); mask[5:-7, 9:-4] = True
    return obj, mask


def test_normalise_roundtrip_and_ramp_removal():
    obj, mask = _test_object()
    z, p = normalise(obj, mask)
    assert abs(p.kx - 0.04) < 5e-3 and abs(p.ky + 0.03) < 5e-3          # ramp found
    assert abs(np.median(np.abs(z[mask])) - 1) < 1e-6                    # background amplitude → 1
    back = denormalise(z, p)
    assert np.allclose(back[mask], obj[mask], atol=1e-9)


def test_identity_package_tiled_onnx(tmp_root):
    cards = list_packages(tmp_root / "models")
    assert [c["id"] for c in cards] == ["identity-test"]
    pkg = load_package(str(tmp_root / "models" / "identity-test"))
    assert pkg.runtime.startswith("onnxruntime")
    obj, mask = _test_object()                     # 150×230 > 64-px tiles: many overlapping tiles
    out = refine(obj, mask, pkg)
    assert out.obj.shape == obj.shape
    assert np.allclose(out.obj[mask], obj[mask], rtol=1e-5, atol=1e-5)   # identity survives blending
    assert np.allclose(out.obj[~mask], obj[~mask])                       # unmeasured pixels untouched
    small, smask = _test_object(40, 50)                                  # smaller than one tile: padded
    assert np.allclose(refine(small, smask, pkg).obj[smask], small[smask], rtol=1e-5, atol=1e-5)


def test_reconstruct_init_object_and_hybrid(synthetic_npz, tmp_root):
    ds = load_dataset(synthetic_npz)
    cfg = ReconConfig(engine="epie", iters=8, seed=0)
    res = reconstruct(ds, cfg)
    # seeding with the true object and probe starts (and stays) near the data
    true = reconstruct(ds, dataclasses.replace(cfg, iters=1), init_object=ds.ground_truth,
                       init_probe=ds.init_probe)
    assert true.history["err"][0] < 0.2 * res.history["err"][0]
    with pytest.raises(ValueError):
        reconstruct(ds, cfg, init_object=np.ones((3, 3)))
    pkg = load_package(str(tmp_root / "models" / "identity-test"))
    ref = refine(res.obj, res.mask, pkg)
    h = hybrid(ds, res, ref.obj, cfg, 4)
    assert h.obj.shape == res.obj.shape and len(h.history["err"]) == 4
    e_phys, e_hyb = data_error(res.obj, res.probe, ds), data_error(h.obj, h.probe, ds)
    assert np.isfinite(e_phys) and np.isfinite(e_hyb) and e_hyb < 1.5 * e_phys


def test_subset_keeps_parent_grid():
    rng = np.random.default_rng(0)
    xy = np.stack(np.meshgrid(np.arange(6), np.arange(6)), -1).reshape(-1, 2) * 10e-6 + 3e-3
    ds = PtychoDataset(dps=rng.random((36, 32, 32)).astype(np.float32), positions_m=xy,
                       wavelength=1e-6, pixel_pitch=2e-4, distance=0.064, name="grid")
    keep = np.where((xy[:, 0] > xy[:, 0].min()) & (xy[:, 1] < xy[:, 1].max()))[0]   # drop an edge row/col
    sub = ds.subset(keep)
    assert sub.object_shape() == ds.object_shape()
    assert np.array_equal(sub.positions_px(), ds.positions_px()[keep])


def test_subpixel_registration_recovers_known_shift():
    from velos_recon.metrics import amplitude_phase_metrics, estimate_shift, shift_image
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    tex = gaussian_filter(rng.standard_normal((128, 128)), 3); tex /= np.abs(tex).max()
    amp = 0.8 + 0.2 * gaussian_filter(rng.standard_normal((128, 128)), 2)
    obj = amp * np.exp(1j * (1.0 * tex + 0.02 * np.arange(128)[None, :]))       # textured, with a ramp
    mask = np.zeros((128, 128), bool); mask[8:-8, 10:-6] = True
    moved = shift_image(obj, (0.4, -0.65))
    dy, dx = estimate_shift(moved, obj, mask)
    assert abs(dy + 0.4) < 0.1 and abs(dx - 0.65) < 0.1
    plain = amplitude_phase_metrics(moved, obj, mask)["nrmse_complex"]
    reg = amplitude_phase_metrics(moved, obj, mask, register=True)["nrmse_complex"]
    assert reg < 0.5 * plain
