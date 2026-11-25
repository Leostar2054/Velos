"""Platform architecture diagram (M4) → evidence/diagrams/platform_architecture_m4.png.  Run from the repository root:
    python scripts/make_platform_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W, H = 16.0, 9.6
fig = plt.figure(figsize=(W, H), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
C = {"client": "#e8f1fb", "edge": "#fff3e0", "api": "#e9f7ef", "worker": "#f3e8fb", "ops": "#eef0f3", "data": "#fdf6d8"}
EC = {"client": "#4a7bb7", "edge": "#d08a1c", "api": "#2e8b57", "worker": "#7b4bb7", "ops": "#5b6470", "data": "#b59a1a"}

def box(x, y, w, h, kind, title, lines=(), fs=10.5, tfs=12):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=C[kind], ec=EC[kind], lw=1.6)
    ax.add_patch(p)
    ax.text(x + 0.15, y + h - 0.16, title, ha="left", va="top", fontsize=tfs, fontweight="bold", color="#1b1f24")
    for i, l in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.55 - i * 0.30, l, ha="left", va="top", fontsize=fs, color="#2b2f36")
    return (x, y, w, h)

def arrow(p, q, label=None, color="#444", ls="-", lw=1.6, off=(0, 0.12), fs=9.5, ha="center", rad=0.0):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        mx, my = (p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1]
        ax.text(mx, my, label, ha=ha, va="bottom", fontsize=fs, color=color)

ax.text(0.3, H - 0.35, "Velos platform — software architecture (M4)", fontsize=17, fontweight="bold", va="top", color="#1b1f24")
ax.text(0.3, H - 0.85, "Everything below the edge server runs as containers from docker-compose.prod.yml; the same stack is "
        "built, started and tested by GitHub Actions on every push.", fontsize=10.5, va="top", color="#444")

# column labels
for x, t in ((0.3, "USERS AND DATA"), (3.75, "EDGE"), (6.55, "API AND STORAGE"), (10.2, "WORKERS (1…n, CPU or GPU)")):
    ax.text(x, H - 1.35, t, fontsize=10, fontweight="bold", color="#6a717b")

b_web = box(0.3, 6.55, 2.85, 1.45, "client", "Website visitors", ["browser → Home, Product,", "Team, Contact (lead form)"])
b_res = box(0.3, 4.55, 2.85, 1.65, "client", "Researchers / scripts", ["REST API with an API key:", "upload, submit, poll,", "download results"])
b_mic = box(0.3, 1.75, 2.85, 2.45, "data", "Microscope data", ["LED-array FPM (in-house):", "  folder → pack → .npz", "scanning ptychography:", "  .mat / .npz / .h5", "(see docs/fpm.md, data-review)"])

b_edge = box(3.75, 4.2, 2.35, 3.8, "edge", "Edge server", ["Caddy 2", "HTTPS (Let's Encrypt)", "static website", "(Next.js export)", "reverse proxy:", "/api, /health, /docs", "/metrics → 404", "JSON access log"])

b_api = box(6.55, 5.2, 3.2, 2.8, "api", "API (FastAPI)", ["API-key auth, roles", "uploads + validation", "jobs: submit / cancel", "results + artifacts", "model cards, contact leads", "/metrics (Prometheus)"])
b_db = box(6.55, 3.45, 3.2, 1.35, "data", "Job queue + metadata", ["SQLite (WAL), atomic claims", "(PostgreSQL for many servers)"])
b_vol = box(6.55, 1.75, 3.2, 1.35, "data", "Storage volume", ["datasets, job outputs,", "heartbeats (backed up nightly)"])

b_wk = box(10.2, 1.75, 3.0, 6.25, "worker", "Reconstruction worker", [
    "Physics engines", "  scanning ptychography:", "  ePIE · rPIE · WASP · GD · MAP", "  (Client's code, unchanged)",
    "  LED-array FPM: AP · EPRY · rPIE", "  (port of the MATLAB code)", "AI refinement", "  ONNX Runtime / TorchScript",
    "Hybrid polish", "  physics passes seeded", "  with the AI output", "Consistency check", "  fit to measured frames,", "  error on frames not used"], fs=10)
b_models = box(13.55, 5.55, 2.2, 2.45, "worker", "Model packages", ["models/ (read-only)", "velos-refine-v2", "  (ptychography)", "velos-refine-fpm-v1", "  (LED-array FPM)"], fs=10)
b_prom = box(3.75, 1.75, 2.35, 1.95, "ops", "Prometheus", ["scrapes /metrics", "5 alert rules", "(API down, worker stalled,", "backlog, failures, 5xx)"], fs=9.5)
b_ml = box(13.55, 1.75, 2.2, 2.75, "ops", "ML pipeline (ml/)", ["real + simulated", "training pairs →", "train → evaluate", "→ package", "(ONNX, TorchScript,", "model card)"], fs=9.5)

# operations band
b_ci = box(0.3, 0.2, 15.45, 1.3, "ops", "GitHub repository Leostar2054/Velos → GitHub Actions on every push", [
    "tests (backend, FPM, metrics) · ML smoke run · website QA · production-stack rehearsal (smoke test incl. a real FPM job, load test,",
    "monitoring checks, redeployment) → images in the GitHub Container Registry → deploy job (scripts/deploy_server.sh over SSH)"], fs=10, tfs=11.5)

# arrows
arrow((3.15, 7.25), (3.75, 7.25), None, color=EC["client"])
arrow((3.15, 5.35), (3.75, 5.35), None, color=EC["client"])
ax.text(3.45, 7.33, "HTTPS", fontsize=8.5, ha="center", color=EC["client"]); ax.text(3.45, 5.43, "HTTPS", fontsize=8.5, ha="center", color=EC["client"])
arrow((1.8, 4.2), (1.8, 4.55), None, color=EC["data"])
ax.text(1.9, 4.33, "upload", fontsize=8.5, color=EC["data"])
arrow((6.1, 6.6), (6.55, 6.6), None, color=EC["edge"])
arrow((8.15, 5.2), (8.15, 4.8), None, color=EC["api"])
ax.text(8.25, 4.93, "enqueue", fontsize=8.5, color=EC["api"])
arrow((9.75, 4.12), (10.2, 4.12), None, color=EC["worker"])
ax.text(9.98, 4.2, "claim", fontsize=8.5, ha="center", color=EC["worker"])
arrow((10.2, 2.42), (9.75, 2.42), None, color=EC["worker"])
ax.text(9.98, 2.5, "results", fontsize=8.5, ha="center", color=EC["worker"])
arrow((13.55, 6.8), (13.2, 6.8), None, color=EC["worker"])
arrow((6.1, 3.0), (6.55, 5.45), None, color=EC["ops"], ls="--")

arrow((14.65, 4.5), (14.65, 5.55), None, color=EC["ops"], ls="--")
ax.text(14.75, 4.85, "new version", fontsize=8.5, color=EC["ops"])
fig.savefig("evidence/diagrams/platform_architecture_m4.png", dpi=150, facecolor="white")
print("ok")
