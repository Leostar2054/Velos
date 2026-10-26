"""The service driver must reproduce the Client's engine bit-for-bit."""
import numpy as np

from velos_recon import ReconConfig, core, load_dataset, reconstruct
from velos_recon.metrics import amplitude_phase_metrics


def test_driver_matches_client_engine(synthetic_npz):
    ds = load_dataset(synthetic_npz)
    amps = ds.amplitudes(); pos = ds.positions_px(); P0 = ds.init_probe
    # the driver scales the probe to the brightest pattern's power; do the same here
    bright = ds.dps.sum(axis=(1, 2)).max(); P0 = P0 * np.sqrt(bright / np.sum(np.abs(P0) ** 2))
    O_ref, P_ref, hist = core.wasp(amps, P0, pos, ds.ground_truth, n_iter=4, seed=0, probe_lr=0.1)
    res = reconstruct(ds, ReconConfig(engine="wasp", iters=4, seed=0, engine_params={"probe_lr": 0.1}))
    assert res.obj.shape == O_ref.shape
    assert np.allclose(res.obj, O_ref) and np.allclose(res.probe, P_ref)
    assert np.allclose(res.history["err"], hist["err"])


def test_engines_reduce_error_and_recover_object(synthetic_npz):
    ds = load_dataset(synthetic_npz)
    for engine in ("wasp", "rpie", "epie"):
        res = reconstruct(ds, ReconConfig(engine=engine, iters=30, seed=0))
        err = res.history["err"]
        assert err[-1] < 0.2 * err[0], engine
        m = amplitude_phase_metrics(res.obj, ds.ground_truth, res.mask)
        assert m["ssim_amplitude"] > 0.6 and m["nrmse_complex"] < 0.5, (engine, m)


def test_sparse_subset_and_crop(synthetic_npz):
    ds = load_dataset(synthetic_npz)
    sub = ds.sparse_subset(every=2)
    assert 0 < sub.n_patterns < ds.n_patterns
    c = ds.center_crop(16)
    assert c.frame_shape == (16, 16) and abs(c.dx - 2 * ds.dx) < 1e-12
