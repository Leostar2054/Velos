"""Markdown summary of the M3 evaluation (tables only; interpretation lives in docs/ml-pipeline.md).

    python -m velos_ml.summary_m3 results/m3 > results/m3/summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LABEL = {"physics": "Physics, reduced scan (ePIE 300 it)", "physics+300": "Physics + 300 further it",
         "m2": "M2 model", "v2_syn": "v2, synthetic only (stage 1)", "v2": "**v2 model (final)**",
         "hybrid@10": "Hybrid v2 + 10 it", "hybrid@30": "**Hybrid v2 + 30 it**", "hybrid@100": "Hybrid v2 + 100 it",
         "dense": "Dense scan, 121 frames (reference)", "dense_repeat": "All frames, 2nd run (repeatability)"}
ORDER = ["physics", "physics+300", "m2", "v2_syn", "v2", "hybrid@10", "hybrid@30", "hybrid@100", "dense", "dense_repeat"]


def _f(g, k, nd=3):
    v = g.get(k)
    return "–" if v is None else f"{v:.{nd}f}"


def render(res: Path) -> str:
    S = json.loads((res / "summary.json").read_text())
    out = ["# M3 evaluation — summary tables", ""]
    syn = S.get("synthetic", {})
    if syn:
        out += [f"## Synthetic test set (n = {syn['n_cases']}, ground truth; registered, ramp and scalar removed)", "",
                "| Method | NRMSE ↓ | Phase PSNR (dB) ↑ | Phase SSIM ↑ | Amp PSNR (dB) ↑ | Amp SSIM ↑ | Data err. used ↓ | Prediction err. (unmeasured frames) ↓ |",
                "|---|---|---|---|---|---|---|---|"]
        by = {g["method"]: g for g in syn["overall"]}
        for m in ORDER:
            if m in by:
                g = by[m]
                out.append(f"| {LABEL[m]} | {_f(g, 'nrmse_complex')} ± {_f(g, 'nrmse_complex_std')} | {_f(g, 'psnr_phase', 1)} ± "
                           f"{_f(g, 'psnr_phase_std', 1)} | {_f(g, 'ssim_phase')} | {_f(g, 'psnr_amplitude', 1)} | "
                           f"{_f(g, 'ssim_amplitude')} | {_f(g, 'data_error_used', 4)} | {_f(g, 'data_error_predict', 4)} |")
        out += ["", "### By number of frames (phase PSNR dB / NRMSE)", ""]
        bins = sorted({g["bin"] for g in syn["by_density"]}, key=lambda b: int(b.split()[0]))
        out += ["| Method | " + " | ".join(f"{b} (n={next(g['n'] for g in syn['by_density'] if g['bin'] == b)})" for b in bins) + " |",
                "|---|" + "---|" * len(bins)]
        for m in ORDER:
            cells = []
            for b in bins:
                g = next((g for g in syn["by_density"] if g["bin"] == b and g["method"] == m), None)
                cells.append(f"{g['psnr_phase']:.1f} / {g['nrmse_complex']:.3f}" if g else "–")
            if any(c != "–" for c in cells):
                out.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    real = S.get("real", {})
    if real:
        out += ["", "## Experimental data (reference = reconstruction from all frames; registered)", "",
                "| Case | Method | NRMSE ↓ | Phase PSNR ↑ | Phase SSIM ↑ | Data err. used ↓ | Prediction err. (frames not used) ↓ |",
                "|---|---|---|---|---|---|---|"]
        for g in real["by_case"]:
            out.append(f"| {g['case']} | {LABEL.get(g['method'], g['method'])} | {_f(g, 'nrmse_complex')} | {_f(g, 'psnr_phase', 1)} | "
                       f"{_f(g, 'ssim_phase')} | {_f(g, 'data_error_used', 4)} | {_f(g, 'data_error_predict', 4)} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(render(Path(sys.argv[1])))
