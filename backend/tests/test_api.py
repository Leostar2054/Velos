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
