"""Render the live OpenAPI specification to a static HTML page (offline alternative to Swagger UI,
whose assets load from a CDN) and optionally screenshot it.

    python scripts/render_openapi.py http://localhost:8000 out_dir [--screenshot]
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

CSS = """body{font-family:Inter,Arial,sans-serif;background:#0b1020;color:#e2e8f0;margin:0;padding:28px 36px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:26px 0 8px;color:#67e8f9}
.sub{color:#94a3b8;font-size:13px;margin-bottom:18px}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid #1e293b;padding:7px 8px;text-align:left;vertical-align:top}th{color:#94a3b8;font-weight:600}
.m{font-weight:700;font-family:monospace;padding:2px 7px;border-radius:5px}.get{background:#0e7490}.post{background:#6d28d9}
code{font-family:monospace;color:#c4b5fd}.new{color:#22d3ee;font-weight:700;font-size:11px;margin-left:6px}"""
NEW_PATHS = {"/api/v1/models"}
NEW_FIELDS = {"sparse_fraction", "model", "hybrid_iters", "probe_from_job"}


def main():
    base, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    spec = json.load(urllib.request.urlopen(f"{base}/openapi.json"))
    (out / "openapi.json").write_text(json.dumps(spec, indent=1))
    rows = []
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            new = '<span class="new">NEW in M3</span>' if path in NEW_PATHS else ""
            rows.append(f'<tr><td><span class="m {method}">{method.upper()}</span></td><td><code>{html.escape(path)}</code>{new}</td>'
                        f'<td>{html.escape(op.get("summary", ""))}</td><td>{html.escape(", ".join(op.get("tags", [])))}</td></tr>')
    job = spec["components"]["schemas"]["JobCreate"]["properties"]
    req = set(spec["components"]["schemas"]["JobCreate"].get("required", []))
    frows = []
    for name, p in job.items():
        typ = p.get("type") or " | ".join(x.get("type", "") for x in p.get("anyOf", []) if x.get("type") != "null")
        new = '<span class="new">NEW in M3</span>' if name in NEW_FIELDS else ""
        frows.append(f'<tr><td><code>{name}</code>{new}</td><td>{typ}{" (required)" if name in req else ""}</td>'
                     f'<td>{html.escape(str(p.get("default", "")))}</td><td>{html.escape(p.get("description", ""))}</td></tr>')
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>Velos API</title><style>{CSS}</style></head><body>"
            f"<h1>{html.escape(spec['info']['title'])} — v{html.escape(spec['info']['version'])}</h1>"
            f"<div class='sub'>Rendered from the live OpenAPI specification ({html.escape(base)}/openapi.json)</div>"
            f"<h2>Endpoints</h2><table><tr><th>Method</th><th>Path</th><th>Summary</th><th>Tag</th></tr>{''.join(rows)}</table>"
            f"<h2>Job parameters (POST /api/v1/jobs)</h2><table><tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>"
            f"{''.join(frows)}</table></body></html>")
    (out / "openapi_rendered.html").write_text(page, encoding="utf-8")
    if "--screenshot" in sys.argv:
        js = (f"const {{chromium}}=require('playwright');(async()=>{{const b=await chromium.launch();"
              f"const p=await b.newPage({{viewport:{{width:1280,height:900}},deviceScaleFactor:2}});"
              f"await p.goto('file://{(out / 'openapi_rendered.html').resolve()}');"
              f"await p.screenshot({{path:'{(out / 'openapi_docs_m3.png').resolve()}',fullPage:true}});await b.close();}})();")
        subprocess.run(["node", "-e", js], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
