"""M4 operations: Prometheus metrics, optional metrics token, JSON logging, worker heartbeat."""
import json
import logging
import os

from fastapi.testclient import TestClient

from velos_backend.main import app


def test_metrics_exposes_jobs_http_and_models():
    with TestClient(app) as c:
        c.get("/health")
        r = c.get("/metrics")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
        t = r.text
        for name in ("velos_build_info", 'velos_jobs{status="queued"}', "velos_job_duration_seconds_count",
                     "velos_queue_oldest_seconds", "velos_models_installed", "velos_http_requests_total"):
            assert name in t, name
        assert 'route="/health"' in t                     # route templates, not raw paths (bounded labels)


def test_metrics_token(monkeypatch):
    monkeypatch.setenv("VELOS_METRICS_TOKEN", "s3cret")
    with TestClient(app) as c:
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_json_logging(monkeypatch, capsys):
    from velos_backend.config import configure_logging
    monkeypatch.setenv("VELOS_LOG_FORMAT", "json")
    configure_logging("INFO")
    logging.getLogger("velos.test").info("hello %s", "world")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    d = json.loads(line)
    assert d["msg"] == "hello world" and d["level"] == "INFO" and d["logger"] == "velos.test"
    monkeypatch.setenv("VELOS_LOG_FORMAT", "text")
    configure_logging(os.getenv("VELOS_LOG_LEVEL", "WARNING"))
