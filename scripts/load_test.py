#!/usr/bin/env python3
"""
Performance / stability test of a running Velos stack (M4).

    python scripts/load_test.py --base http://localhost:8000 --key <user-key> --jobs 24 --concurrency 6 \
        --out evidence/load_test

What it does (standard library + NumPy only, so it runs on any machine with Python 3.10+):

1. uploads one small dataset (synthetic scanning-ptychography data unless --dataset is given);
2. submits ``--jobs`` reconstruction jobs from ``--concurrency`` parallel clients while other
   clients keep polling job status, listing jobs and calling /health (a realistic mixed load);
3. waits until every job has finished and checks that each one succeeded and produced its
   artifacts; a few jobs are cancelled on purpose to exercise the cancel path;
4. reports API latency percentiles per endpoint, job queue waiting time and run time,
   throughput, failures, and the /metrics counters before and after.

Exit status 0 only if no request failed unexpectedly and every non-cancelled job succeeded.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np


class Client:
    def __init__(self, base: str, key: str):
        self.base, self.key = base.rstrip("/"), key
        self.lat: dict[str, list] = {}
        self.errors: list[str] = []
        self.lock = threading.Lock()

    def _req(self, method: str, path: str, tag: str, body: bytes | None = None, ctype: str | None = None,
             ok=(200, 201, 202)):
        req = urllib.request.Request(self.base + path, data=body, method=method)
        req.add_header("X-API-Key", self.key)
        if ctype:
            req.add_header("Content-Type", ctype)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data, code = r.read(), r.status
        except urllib.error.HTTPError as e:
            data, code = e.read(), e.code
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.errors.append(f"{method} {path}: {e}")
            return None, None
        dt = time.perf_counter() - t0
        with self.lock:
            self.lat.setdefault(tag, []).append(dt)
            if code not in ok:
                self.errors.append(f"{method} {path}: HTTP {code} {data[:200]!r}")
        return code, data

    def get_json(self, path, tag):
        code, data = self._req("GET", path, tag)
        return json.loads(data) if code == 200 else None

    def post_json(self, path, payload, tag):
        code, data = self._req("POST", path, tag, json.dumps(payload).encode(), "application/json")
        return json.loads(data) if code in (200, 201, 202) else None

    def upload(self, filename: str, content: bytes, name: str):
        boundary = uuid.uuid4().hex
        buf = io.BytesIO()
        for k, v in (("name", name),):
            buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                  f"Content-Type: application/octet-stream\r\n\r\n".encode())
        buf.write(content); buf.write(f"\r\n--{boundary}--\r\n".encode())
        code, data = self._req("POST", "/api/v1/datasets", "POST /datasets", buf.getvalue(),
                               f"multipart/form-data; boundary={boundary}")
        return json.loads(data) if code == 201 else None


def synthetic_dataset() -> bytes:
    """Small scanning-ptychography dataset (96 px object, 32 px probe, 81 positions)."""
    N, Np = 96, 32
    yy, xx = np.mgrid[0:N, 0:N]
    obj = np.exp(1j * 0.8 * np.exp(-((yy - 48) ** 2 + (xx - 40) ** 2) / 300.0))
    r = np.hypot(*np.mgrid[-Np / 2:Np / 2, -Np / 2:Np / 2]); probe = (r < 10).astype(complex)
    pos = [(i, j) for i in range(0, N - Np + 1, 8) for j in range(0, N - Np + 1, 8)]
    dps = np.stack([np.abs(np.fft.fftshift(np.fft.fft2(probe * obj[i:i + Np, j:j + Np]))) ** 2 / Np ** 2 * 1e4
                    for i, j in pos])
    dx, wl, z = 1e-6, 1e-6, 1e-2
    b = io.BytesIO()
    np.savez(b, dps=dps.astype(np.float32), positions=np.array([(j * dx, i * dx) for i, j in pos]),
             wavelength=wl, pixel_pitch=wl * z / (Np * dx), distance=z, init_probe=probe)
    return b.getvalue()


def pct(v, q):
    return float(np.percentile(v, q)) if v else float("nan")


def metrics_snapshot(c: Client) -> dict:
    req = urllib.request.Request(c.base + "/metrics")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001 (metrics may be internal-only behind the edge server)
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("velos_jobs{") or line.startswith("velos_job_duration") or line.startswith("velos_worker"):
            k, v = line.rsplit(" ", 1); out[k] = float(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--jobs", type=int, default=24); ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--iters", type=int, default=30); ap.add_argument("--engine", default="epie")
    ap.add_argument("--dataset", default=None, help="file to upload instead of the synthetic set")
    ap.add_argument("--job-json", default=None, help="extra job fields as JSON, e.g. '{\"engine\": \"fpm\"}'")
    ap.add_argument("--cancel", type=int, default=2, help="jobs to cancel on purpose")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    c = Client(a.base, a.key)
    t_start = time.time()
    before = metrics_snapshot(c)
    content = Path(a.dataset).read_bytes() if a.dataset else synthetic_dataset()
    ds = c.upload(Path(a.dataset).name if a.dataset else "loadtest.npz", content, "load-test")
    if ds is None:
        print("upload failed:", c.errors); return 2
    extra = json.loads(a.job_json) if a.job_json else {}
    payload = {"dataset_id": ds["id"], "engine": a.engine, "iters": a.iters, **extra}
    stop = threading.Event()

    def background_reader():
        while not stop.is_set():
            c.get_json("/health", "GET /health")
            c.get_json("/api/v1/jobs", "GET /jobs")
            time.sleep(0.2)

    readers = [threading.Thread(target=background_reader, daemon=True) for _ in range(2)]
    for t in readers:
        t.start()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        jobs = [j for j in ex.map(lambda i: c.post_json("/api/v1/jobs", {**payload, "seed": i}, "POST /jobs"),
                                  range(a.jobs)) if j]
    t_submitted = time.time()
    cancelled = set()
    for j in jobs[-a.cancel:] if a.cancel else []:
        r = c.post_json(f"/api/v1/jobs/{j['id']}/cancel", {}, "POST /jobs/{id}/cancel")
        if r:
            cancelled.add(j["id"])
    pending = {j["id"] for j in jobs}
    final = {}
    deadline = time.time() + a.timeout
    while pending and time.time() < deadline:
        for jid in list(pending):
            st = c.get_json(f"/api/v1/jobs/{jid}", "GET /jobs/{id}")
            if st and st["status"] in ("succeeded", "failed", "cancelled"):
                final[jid] = st; pending.discard(jid)
        time.sleep(0.5)
    stop.set()
    for t in readers:
        t.join(timeout=5)
    # artifacts of every succeeded job
    missing = []
    for jid, st in final.items():
        if st["status"] == "succeeded":
            res = c.get_json(f"/api/v1/jobs/{jid}/result", "GET /jobs/{id}/result")
            if not res or not res.get("artifacts"):
                missing.append(jid)
    after = metrics_snapshot(c)

    def ts(x):
        return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp() if x else None

    waits, runs = [], []
    for st in final.values():
        if st["status"] == "succeeded" and st.get("started_at") and st.get("finished_at"):
            waits.append(ts(st["started_at"]) - ts(st["created_at"]))
            runs.append(ts(st["finished_at"]) - ts(st["started_at"]))
    status_counts = {}
    for st in final.values():
        status_counts[st["status"]] = status_counts.get(st["status"], 0) + 1
    unexpected = [jid for jid, st in final.items() if st["status"] != "succeeded" and jid not in cancelled]
    total_s = time.time() - t_start
    report = {
        "base": a.base, "jobs_submitted": len(jobs), "jobs_requested": a.jobs, "concurrency": a.concurrency,
        "job": payload, "cancelled_on_purpose": len(cancelled), "status_counts": status_counts,
        "unfinished": len(pending), "unexpected_failures": unexpected, "missing_artifacts": missing,
        "request_errors": c.errors[:20], "n_request_errors": len(c.errors),
        "duration_s": round(total_s, 1), "submit_phase_s": round(t_submitted - t_start, 2),
        "throughput_jobs_per_min": round(60 * status_counts.get("succeeded", 0) / max(total_s, 1e-9), 2),
        "job_wait_s": {"median": round(statistics.median(waits), 2) if waits else None, "max": round(max(waits), 2) if waits else None},
        "job_run_s": {"median": round(statistics.median(runs), 2) if runs else None, "max": round(max(runs), 2) if runs else None},
        "latency_ms": {k: {"n": len(v), "p50": round(1e3 * pct(v, 50), 1), "p95": round(1e3 * pct(v, 95), 1),
                           "p99": round(1e3 * pct(v, 99), 1), "max": round(1e3 * max(v), 1)} for k, v in sorted(c.lat.items())},
        "metrics_before": before, "metrics_after": after,
    }
    ok = not unexpected and not pending and not missing and not [e for e in c.errors if "cancel" not in e]
    report["passed"] = bool(ok)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / "load_test.json").write_text(text)
        lines = [f"# Load / stability test — {datetime.now().isoformat(timespec='seconds')}", "",
                 f"* target: `{a.base}` · {len(jobs)} jobs ({payload['engine']}, {payload['iters']} it) from {a.concurrency} "
                 f"parallel clients + 2 polling clients · {len(cancelled)} cancelled on purpose",
                 f"* result: **{'PASSED' if ok else 'FAILED'}** · statuses {status_counts} · request errors {len(c.errors)}",
                 f"* duration {report['duration_s']} s · throughput {report['throughput_jobs_per_min']} jobs/min · "
                 f"queue wait median {report['job_wait_s']['median']} s (max {report['job_wait_s']['max']}) · "
                 f"run time median {report['job_run_s']['median']} s", "",
                 "| Endpoint | n | p50 ms | p95 ms | p99 ms | max ms |", "|---|---|---|---|---|---|"]
        for k, v in report["latency_ms"].items():
            lines.append(f"| {k} | {v['n']} | {v['p50']} | {v['p95']} | {v['p99']} | {v['max']} |")
        (out / "load_test.md").write_text("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
