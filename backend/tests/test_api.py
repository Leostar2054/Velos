"""End-to-end API test: upload → submit → worker → status → result → artifact."""
import io

import pytest
from fastapi.testclient import TestClient

from velos_backend.main import app
from velos_backend.worker import run_once

USER = {"X-API-Key": "test-user-key"}
ADMIN = {"X-API-Key": "test-admin-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_auth_required(client):
    assert client.get("/api/v1/datasets").status_code == 401
    assert client.get("/api/v1/datasets", headers={"X-API-Key": "wrong"}).status_code == 401


def test_upload_rejects_bad_files(client):
    r = client.post("/api/v1/datasets", headers=USER, files={"file": ("x.txt", b"hello")})
    assert r.status_code == 415
    r = client.post("/api/v1/datasets", headers=USER, files={"file": ("x.npz", b"not an npz")})
    assert r.status_code == 422


def test_full_workflow(client, synthetic_npz):
    # 1. upload
    with open(synthetic_npz, "rb") as f:
        r = client.post("/api/v1/datasets", headers=USER, files={"file": ("synthetic.npz", f)},
                        data={"name": "synthetic cells"})
    assert r.status_code == 201, r.text
    ds = r.json(); assert ds["summary"]["n_patterns"] > 0 and ds["summary"]["has_ground_truth"]

    # 2. submit (returns immediately with status queued)
    r = client.post("/api/v1/jobs", headers=USER,
                    json={"dataset_id": ds["id"], "engine": "wasp", "iters": 12})
    assert r.status_code == 202, r.text
    job = r.json(); assert job["status"] == "queued"

    # 3. status before processing
    r = client.get(f"/api/v1/jobs/{job['id']}", headers=USER)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert client.get(f"/api/v1/jobs/{job['id']}/result", headers=USER).status_code == 409

    # 4. worker processes the job (separate process in production)
    assert run_once("test-worker") == job["id"]

    # 5. status + result
    r = client.get(f"/api/v1/jobs/{job['id']}", headers=USER)
    assert r.json()["status"] == "succeeded", r.json()
    assert r.json()["progress"] == 1.0
    r = client.get(f"/api/v1/jobs/{job['id']}/result", headers=USER)
    assert r.status_code == 200
    res = r.json()
    assert {"amplitude.png", "phase.png", "result.npz", "summary.json"} <= set(res["artifacts"])
    assert res["summary"]["reconstruction"]["final_data_error"] < res["summary"]["reconstruction"]["initial_data_error"]

    # 6. artifact download
    r = client.get(res["links"]["amplitude.png"], headers=USER)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"

    # 7. isolation: another user cannot see it, admin can
    r = client.get(f"/api/v1/jobs/{job['id']}", headers={"X-API-Key": "test-admin-key"})
    assert r.status_code == 200


def test_cancel_queued_job(client, synthetic_npz):
    with open(synthetic_npz, "rb") as f:
        ds = client.post("/api/v1/datasets", headers=USER, files={"file": ("s.npz", f)}).json()
    job = client.post("/api/v1/jobs", headers=USER, json={"dataset_id": ds["id"], "iters": 5}).json()
    r = client.post(f"/api/v1/jobs/{job['id']}/cancel", headers=USER)
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert run_once("test-worker") is None       # nothing left to claim


def test_bad_engine_rejected(client, synthetic_npz):
    r = client.post("/api/v1/jobs", headers=USER, json={"dataset_id": "nope", "engine": "magic"})
    assert r.status_code == 422


def test_contact_endpoint(client):
    r = client.post("/api/v1/contact", json={"name": "Dr Kim", "email": "kim@lab.ac.kr",
                                             "organisation": "GIST", "message": "Interested in a demo"})
    assert r.status_code == 201 and r.json()["ok"]
    r = client.post("/api/v1/contact", json={"name": "bot", "email": "b@b.io", "message": "x", "website": "spam"})
    assert r.status_code == 201   # silently accepted, not stored
    r = client.post("/api/v1/contact", json={"name": "x", "email": "bad", "message": "x"})
    assert r.status_code == 422
    assert client.get("/api/v1/contact", headers=USER).status_code == 403      # admin only
    leads = client.get("/api/v1/contact", headers=ADMIN).json()
    assert any(l["email"] == "kim@lab.ac.kr" for l in leads) and not any(l["name"] == "bot" for l in leads)


# ---------------------------------------------------------------- M3: model packages
def test_models_endpoint_and_health(client):
    assert client.get("/api/v1/models").status_code == 401
    r = client.get("/api/v1/models", headers=USER)
    assert r.status_code == 200, r.text
    cards = r.json()
    assert cards[0]["id"] == "identity-test" and cards[0]["default"] and "model.onnx" in cards[0]["files"]
    assert client.get("/health").json()["models"]["default"] == "identity-test"


def _upload(client, path):
    with open(path, "rb") as f:
        return client.post("/api/v1/datasets", headers=USER, files={"file": ("s.npz", f)}).json()


def test_job_with_ml_refinement_and_hybrid(client, synthetic_npz):
    ds = _upload(client, synthetic_npz)
    job = client.post("/api/v1/jobs", headers=USER, json={
        "dataset_id": ds["id"], "engine": "epie", "iters": 6, "refine_ml": True, "hybrid_iters": 3}).json()
    assert job["params"]["refine_ml"] and job["params"]["hybrid_iters"] == 3
    assert run_once("test-worker") == job["id"]
    r = client.get(f"/api/v1/jobs/{job['id']}/result", headers=USER)
    assert r.status_code == 200, client.get(f"/api/v1/jobs/{job['id']}", headers=USER).json()
    res = r.json(); ml = res["summary"]["ml_refinement"]
    assert ml["applied"] and ml["model"] == "identity-test" and ml["runtime"].startswith("onnxruntime")
    assert set(ml["data_error"]) == {"physics", "ml", "hybrid"}
    # identity model: refined == physics, so its data error is the physics one
    assert abs(ml["data_error"]["ml"] - ml["data_error"]["physics"]) <= 1e-6 * ml["data_error"]["physics"]
    assert ml["consistency"] == "ok" and ml["hybrid"]["iters"] == 3
    for name in ("refined_phase.png", "refined.npz", "hybrid_preview.png", "hybrid.npz", "comparison.png"):
        assert client.get(res["links"][name], headers=USER).status_code == 200, name


def test_reduced_scan_fraction_and_unknown_model(client, synthetic_npz):
    ds = _upload(client, synthetic_npz)
    job = client.post("/api/v1/jobs", headers=USER, json={
        "dataset_id": ds["id"], "iters": 3, "sparse_fraction": 0.5}).json()
    assert run_once("test-worker") == job["id"]
    summ = client.get(f"/api/v1/jobs/{job['id']}/result", headers=USER).json()["summary"]
    full = client.get(f"/api/v1/datasets/{ds['id']}", headers=USER).json()["summary"]["n_patterns"]
    assert summ["dataset"]["n_patterns"] == round(0.5 * full)
    bad = client.post("/api/v1/jobs", headers=USER, json={
        "dataset_id": ds["id"], "iters": 2, "refine_ml": True, "model": "nope"}).json()
    run_once("test-worker")
    st = client.get(f"/api/v1/jobs/{bad['id']}", headers=USER).json()
    assert st["status"] == "failed" and "unknown model" in st["error"]


def test_cancel_running_job_finalises(client, synthetic_npz):
    """Regression (found in M3): cancelling a job that is already running must end in
    'cancelled' (M2 wrote progress=NULL here and the job stayed 'running')."""
    from velos_backend.db import session_scope
    from velos_backend.queue import claim_next
    from velos_backend.worker import process_job
    ds = _upload(client, synthetic_npz)
    job = client.post("/api/v1/jobs", headers=USER, json={"dataset_id": ds["id"], "iters": 50}).json()
    with session_scope() as s:
        assert claim_next(s, "w").id == job["id"]                  # now "running"
    r = client.post(f"/api/v1/jobs/{job['id']}/cancel", headers=USER)
    assert r.json()["status"] == "cancelling"
    process_job(job["id"])                                          # worker notices at iteration 1
    st = client.get(f"/api/v1/jobs/{job['id']}", headers=USER).json()
    assert st["status"] == "cancelled" and "cancelled at iteration" in st["error"]


def test_probe_calibration_from_earlier_job(client, synthetic_npz):
    ds = _upload(client, synthetic_npz)
    cal = client.post("/api/v1/jobs", headers=USER, json={"dataset_id": ds["id"], "iters": 10}).json()
    # not finished yet → 409
    r = client.post("/api/v1/jobs", headers=USER, json={"dataset_id": ds["id"], "probe_from_job": cal["id"]})
    assert r.status_code == 409
    assert run_once("test-worker") == cal["id"]
    assert client.post("/api/v1/jobs", headers=USER, json={"dataset_id": ds["id"], "probe_from_job": "nope"}).status_code == 404
    job = client.post("/api/v1/jobs", headers=USER, json={
        "dataset_id": ds["id"], "iters": 3, "sparse_fraction": 0.5, "probe_from_job": cal["id"]}).json()
    assert run_once("test-worker") == job["id"]
    st = client.get(f"/api/v1/jobs/{job['id']}", headers=USER).json()
    assert st["status"] == "succeeded", st
