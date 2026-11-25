"""LED-array FPM (M4): engine port, data formats and the API path."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from velos_recon import fpm
from velos_recon.metrics import amplitude_phase_metrics

USER = {"X-API-Key": "test-user-key"}


@pytest.fixture(scope="module")
def sim():
    acq, gt, cfg = fpm.simulate_fpm(n_low=48, upsample=3, grid=9, z_mm=40, seed=3)
    return acq, gt, cfg


def test_engine_recovers_simulated_object(sim):
    acq, gt, cfg = sim
    cfg = fpm.FPMConfig(**{**cfg.__dict__, "n_passes": 25, "algorithms": ["AP", "EPRY", "rPIE"]})
    res = fpm.reconstruct_fpm(acq, cfg)
    assert res.prepared.A.shape[0] == acq.n_frames            # all frames usable in the simulation
    assert res.prepared.geometry.N_obj == 3 * 48
    for a in res.algorithms:
        assert a.residual[-1] < 0.5 * a.residual[0]             # converges
        m = amplitude_phase_metrics(a.obj, gt, np.ones(gt.shape, bool))
        assert m["nrmse_complex"] < 0.15, (a.name, m)
    # data error of the result against its own frames equals the engine residual (same factors)
    ep = res.get("EPRY")
    e = fpm.fpm_data_error(ep.obj, ep.pupil, res.prepared, ep.intensity_factor)
    assert abs(e - ep.residual[-1]) < 0.02


def test_flatten_roundtrip_and_unwrap(sim):
    acq, gt, cfg = sim
    res = fpm.reconstruct_fpm(acq, fpm.FPMConfig(**{**cfg.__dict__, "n_passes": 5, "algorithms": ["AP"]}))
    o = res.algorithms[0].obj
    p = fpm.flatten_params(o, res.pixel_um, res.curvature_rad_um2)
    assert np.allclose(fpm.undo_flatten(fpm.apply_flatten(o, p), p), o, atol=1e-5)
    ramp = np.add.outer(np.linspace(0, 12, 64), np.linspace(0, 9, 64))
    wrapped = np.angle(np.exp(1j * ramp))
    uw = fpm.unwrap_phase_ls(wrapped)
    assert np.abs((uw - uw.mean()) - (ramp - ramp.mean())).max() < 1e-6


def test_npz_and_packed_formats(sim, tmp_path):
    acq, gt, cfg = sim
    p = fpm.save_fpm_npz(acq, tmp_path / "sim_fpm.npz", {"lambda_um": 0.632, "z_source_mm": 40, "Np": 48})
    assert fpm.is_fpm_npz(p)
    back = fpm.load_fpm_npz(p)
    assert back.frames.shape == acq.frames.shape and np.array_equal(back.frames, acq.frames)
    assert back.defaults["z_source_mm"] == 40 and back.files == acq.files
    # packed layout written by pack_fpm_for_velos.py (frames in positions.csv row order)
    rows = ["index,pixel_x,pixel_y,led_x_mm,led_y_mm,file"]
    perm = np.random.default_rng(0).permutation(acq.n_frames)
    for i in perm:
        rows.append(f"{i + 1},{int(acq.pixel_x[i])},{int(acq.pixel_y[i])},{acq.led_x_mm[i]},{acq.led_y_mm[i]},{acq.files[i]}")
    stem = tmp_path / "Blood__fpm_test_c48"
    np.savez_compressed(f"{stem}_part1of1.npz", frames=acq.frames[perm], first_index=0)
    np.savez_compressed(f"{stem}_meta.npz", files=np.array([acq.files[i] for i in perm]), positions_csv="\n".join(rows),
                        run_info_json=json.dumps(acq.run_info), crop_row0=0, crop_col0=0, frame_height=48,
                        frame_width=48, crop=48, colour_channel=-1, n_frames=acq.n_frames, parts=1,
                        centre_full=np.zeros((0, 0), np.uint16), centre_file="")
    packed = fpm.load_fpm_packed(f"{stem}_meta.npz")
    assert np.array_equal(packed.frames, acq.frames) and packed.files == acq.files


def test_geometry_check_finds_led_distance(sim):
    acq, gt, cfg = sim
    wrong = fpm.FPMConfig(**{**cfg.__dict__, "z_source_mm": 48.0, "geometry_check": "apply", "n_passes": 2,
                             "algorithms": ["AP"]})
    res = fpm.reconstruct_fpm(acq, wrong)
    gc = res.prepared.geometry_check
    assert gc["done"] and gc["applied"]
    assert abs(gc["z_best"] - 40.0) < 4.0                     # true distance 40 mm (coarse 4 % grid)


# ------------------------------------------------------------------ API
@pytest.fixture(scope="module")
def client():
    from velos_backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def fpm_dataset(client, sim, tmp_path_factory):
    acq, gt, cfg = sim
    p = fpm.save_fpm_npz(acq, tmp_path_factory.mktemp("fpm") / "sim_fpm.npz",
                         {"lambda_um": 0.632, "z_source_mm": 40, "Np": 48, "upsample": 3,
                          "init_spherical_phase": False})
    with open(p, "rb") as f:
        r = client.post("/api/v1/datasets", headers=USER, files={"file": ("sim_fpm.npz", f)}, data={"name": "sim FPM"})
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["summary"]["modality"] == "fpm" and d["summary"]["n_frames"] == acq.n_frames
    return d


def _run(client, payload):
    from velos_backend.worker import run_once
    r = client.post("/api/v1/jobs", headers=USER, json=payload)
    assert r.status_code == 202, r.text
    job = r.json()
    assert run_once("test-worker") == job["id"]
    r = client.get(f"/api/v1/jobs/{job['id']}", headers=USER)
    assert r.json()["status"] == "succeeded", r.json()
    return client.get(f"/api/v1/jobs/{job['id']}/result", headers=USER).json()


def test_fpm_job_end_to_end(client, fpm_dataset):
    res = _run(client, {"dataset_id": fpm_dataset["id"], "engine": "fpm-all", "iters": 8})
    s = res["summary"]["reconstruction"]
    assert [a["name"] for a in s["algorithms"]] == ["AP", "EPRY", "rPIE"]
    assert s["primary"] == "EPRY" and s["data_error_normalised"] < 0.2
    for name in ("phase.png", "amplitude.png", "preview.png", "EPRY_phase.tif", "result.npz", "summary.txt",
                 "frames.csv", "comparison.png", "convergence.png"):
        assert name in res["artifacts"], name
    r = client.get(res["links"]["summary.txt"], headers=USER)
    assert r.status_code == 200 and b"EPRY: final data residual" in r.content


def test_fpm_reduced_leds_and_ml(client, fpm_dataset, tmp_root):
    from conftest import make_identity_package
    pkg = make_identity_package(tmp_root / "models" / "identity-fpm-test", tile=64)
    card = json.loads((pkg / "model_card.json").read_text()); card["modality"] = "fpm"
    (pkg / "model_card.json").write_text(json.dumps(card))
    res = _run(client, {"dataset_id": fpm_dataset["id"], "engine": "fpm", "iters": 8, "sparse_fraction": 0.5,
                        "refine_ml": True, "model": "identity-fpm-test", "hybrid_iters": 2})
    s = res["summary"]
    red = s["reconstruction"]["reduced_leds"]
    assert red["leds_used"] < red["leds_available"]
    assert s["reconstruction"]["prediction_error_unused_frames"] > 0
    ml = s["ml_refinement"]
    assert ml["applied"] and ml["model"] == "identity-fpm-test"
    # identity model: refined object == physics object -> same data error
    assert abs(ml["data_error"]["ml"] - ml["data_error"]["physics"]) < 1e-3
    assert ml["consistency"] == "ok" and "hybrid" in ml
    assert "comparison_ml.png" in res["artifacts"]
    # two packages of different modalities: one default each, and the worker never mixes them up
    h = client.get("/health").json()["models"]
    assert h["default"] == "identity-test"
    assert h["default_by_modality"] == {"ptychography": "identity-test", "fpm": "identity-fpm-test"}
    cards = {c["id"]: c for c in client.get("/api/v1/models", headers=USER).json()}
    assert cards["identity-test"]["default"] and cards["identity-test"]["modality"] == "ptychography"
    assert cards["identity-fpm-test"]["default"] and cards["identity-fpm-test"]["modality"] == "fpm"
    from velos_backend.worker import resolve_model
    assert resolve_model(None, "ptychography").name == "identity-test"
    assert resolve_model(None, "fpm").name == "identity-fpm-test"
    with pytest.raises(ValueError, match="for fpm"):
        resolve_model("identity-test", "fpm")


def test_fpm_validation_errors(client, fpm_dataset, synthetic_npz):
    r = client.post("/api/v1/jobs", headers=USER, json={"dataset_id": fpm_dataset["id"], "engine": "epie"})
    assert r.status_code == 422 and "fpm" in r.text
    r = client.post("/api/v1/jobs", headers=USER, json={"dataset_id": fpm_dataset["id"], "engine": "fpm",
                                                       "fpm": {"no_such_setting": 1}})
    assert r.status_code == 422
    with open(synthetic_npz, "rb") as f:
        d = client.post("/api/v1/datasets", headers=USER, files={"file": ("p.npz", f)}).json()
    r = client.post("/api/v1/jobs", headers=USER, json={"dataset_id": d["id"], "engine": "fpm"})
    assert r.status_code == 422


def test_phase_ramp_estimate_robust_to_pixel_noise():
    """FPM reconstructions can have strong pixel-level phase noise; the wrapped-difference median then
    invents ramps (M4 finding). The estimate must find a true ramp and add none to a flat noisy phase."""
    from velos_recon.metrics import estimate_phase_ramp
    rng = np.random.default_rng(5)
    n = 256
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    mask = np.ones((n, n), bool)
    noise = rng.normal(0, 1.4, (n, n))                     # rad, per pixel
    flat = np.exp(1j * noise)
    kx, ky = estimate_phase_ramp(flat, mask)
    assert abs(kx) < 2e-3 and abs(ky) < 2e-3
    ramp = np.exp(1j * (0.05 * xx - 0.03 * yy + noise))
    kx, ky = estimate_phase_ramp(ramp, mask)
    assert abs(kx - 0.05) < 2e-3 and abs(ky + 0.03) < 2e-3
    smooth = np.exp(1j * (0.02 * xx + 0.01 * yy + 0.05 * rng.normal(size=(n, n))))   # the usual case
    kx, ky = estimate_phase_ramp(smooth, mask)
    assert abs(kx - 0.02) < 1e-3 and abs(ky - 0.01) < 1e-3
