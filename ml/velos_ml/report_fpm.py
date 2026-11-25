"""Markdown tables from the M4 FPM evaluation (results/m4/summary_fpm.json).

    python -m velos_ml.report_fpm results/m4/summary_fpm.json > results/m4/summary.md
    python -m velos_ml.report_fpm results/m4/summary_fpm.json --compact results/m4/eval_compact.json   (model card)
"""
from __future__ import annotations

import json
import sys

LABEL = {"physics": "Physics (EPRY, reduced LED set)", "physics+5": "Physics + 5 passes",
         "physics+10": "Physics + 10 passes", "v2": "M3 model v2 (no FPM training)",
         "fpm_v1": "**FPM model v1** (fine-tuned)", "hybrid@5": "Hybrid: FPM model + 5 passes",
         "hybrid@10": "Hybrid: FPM model + 10 passes"}
ORDER = ["physics", "physics+5", "physics+10", "v2", "fpm_v1", "hybrid@5", "hybrid@10"]
MODE = {"holdout": "hold-out region (right 35 % of the trained field, unseen LED draws)",
        "full": "side regions of the sensor (never used for training)"}


def _order(methods) -> list:
    """Known methods in report order, then any other method names (e.g. other models) alphabetically."""
    methods = set(methods)
    return [m for m in ORDER if m in methods] + sorted(methods - set(ORDER))


def _f(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def tables(S: dict) -> str:
    out = []
    groups = {}
    for g in S["by_mode_fraction"]:
        groups.setdefault((g["mode"], g["fraction"]), {})[g["method"]] = g
    for (mode, frac), by in sorted(groups.items(), key=lambda kv: (kv[0][0] != "holdout", kv[0][1])):
        n = max(g["n"] for g in by.values())
        out.append(f"#### {MODE.get(mode, mode)} — dark-field fraction {frac:.0%} (n = {n} cases)\n")
        out.append("| Method | Phase PSNR (dB) ↑ | NRMSE ↓ | Phase SSIM ↑ | Amp PSNR (dB) ↑ | Data err. used ↓ | Prediction err. (LEDs not used) ↓ |")
        out.append("|---|---|---|---|---|---|---|")
        base = by.get("physics", {})
        for m in _order(by):
            g = by.get(m)
            if not g:
                continue
            d = g.get("psnr_phase", 0) - base.get("psnr_phase", 0)
            pe = g.get("data_error_predict"); pb = base.get("data_error_predict")
            rel = f" ({(pe / pb - 1) * 100:+.0f} %)" if (pe and pb and m != "physics") else ""
            out.append(f"| {LABEL.get(m, m)} | {g.get('psnr_phase', float('nan')):.2f}"
                       f"{'' if m == 'physics' else f' ({d:+.2f})'} | {_f(g.get('nrmse_complex'))} | "
                       f"{_f(g.get('ssim_phase'))} | {_f(g.get('psnr_amplitude'), 2)} | {_f(g.get('data_error_used'), 4)} | "
                       f"{_f(pe, 4)}{rel} |")
        out.append("")
    out.append("#### Per scan (mean of the test LED draws)\n")
    present = _order({g["method"] for g in S["by_dataset"]})
    out.append("| Scan | " + " | ".join(LABEL.get(m, m).replace("**", "") for m in present) + " |")
    out.append("|---|" + "---|" * len(present))
    ds = {}
    for g in S["by_dataset"]:
        ds.setdefault(g["dataset"], {})[g["method"]] = g
    for name, by in ds.items():
        cells = [f"{by[m]['psnr_phase']:.2f} / {by[m]['data_error_predict']:.3f}" if m in by else "–" for m in present]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("\n(phase PSNR dB / prediction error on the LEDs not used)\n")
    if S.get("repeatability"):
        out.append("#### Repeatability of the reference (full-LED reconstructions of repeated scans, hold-out region)\n")
        out.append("| Pair | Phase PSNR (dB) | NRMSE | Phase SSIM |")
        out.append("|---|---|---|---|")
        for r in S["repeatability"]:
            out.append(f"| {r['pair']} | {r['psnr_phase']:.2f} | {r['nrmse_complex']:.3f} | {r['ssim_phase']:.3f} |")
        out.append("")
    return "\n".join(out)


PROTOCOL = ("real LED-array FPM data (blood smear, leaf section; 23 Nov 2026). Input: EPRY reconstruction from a reduced "
            "LED set (all LEDs inside 1.1 x NA + 30 % or 50 % of the dark-field LEDs); reference: EPRY reconstruction from "
            "all usable LEDs of the same scan (no ground truth exists). Test data never used for training: right 35 % of the "
            "field with unseen LED draws (holdout), and sensor regions left/right of the training region (full). Metrics "
            "after removing the global complex scalar and the linear phase ramp; data_error_predict = misfit on the LEDs "
            "that were NOT used.")


def compact(S: dict, nd: int = 4) -> dict:
    keep = ("n", "nrmse_complex", "psnr_phase", "ssim_phase", "psnr_amplitude", "data_error_used", "data_error_predict")
    r = lambda g: {k: (round(g[k], nd) if isinstance(g.get(k), float) else g.get(k)) for k in keep if k in g}
    by_mf = {}
    for g in S["by_mode_fraction"]:
        by_mf.setdefault(f"{g['mode']}@{g['fraction']:.2f}", {})[g["method"]] = r(g)
    return {"protocol": PROTOCOL,
            "overall": {g["method"]: r(g) for g in S["overall"]},
            "by_mode_fraction": by_mf,
            "repeatability": [{k: (round(v, nd) if isinstance(v, float) else v) for k, v in x.items()}
                              for x in S.get("repeatability", [])]}


def main(argv=None):
    argv = argv or sys.argv[1:]
    S = json.loads(open(argv[0]).read())
    if len(argv) > 2 and argv[1] == "--compact":
        open(argv[2], "w").write(json.dumps(compact(S), indent=1))
        return
    print(tables(S))


if __name__ == "__main__":
    main()
