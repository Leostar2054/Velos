"""
Package a trained checkpoint for inference (M3): ONNX + TorchScript + model card.

    python -m velos_ml.package runs/refine_v2/best.pt --out models/velos-refine-v2 \
        --version 2.0.0 --eval results/m3/summary.json

The backend runtime (``velos_recon.refine``) loads the package with ONNX Runtime (CPU or
CUDA) or TorchScript, applies fixed T×T tiles with Hann blending, and needs no training
code. The export is verified numerically against the PyTorch model before the card is
written; the verification numbers are stored in the card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .model import build_model, count_params


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _realistic_input(tile: int, seed: int = 0) -> np.ndarray:
    """A normalised-looking test tile: background ≈ 1+0i, a few phase objects, partial mask."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:tile, 0:tile]
    phase = np.zeros((tile, tile))
    for _ in range(6):
        cy, cx, r = rng.uniform(0, tile, 2).tolist() + [rng.uniform(8, 30)]
        phase += rng.uniform(0.3, 1.5) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    z = (1 - 0.2 * phase / (phase.max() + 1e-9)) * np.exp(1j * phase) + 0.05 * (
        rng.standard_normal((tile, tile)) + 1j * rng.standard_normal((tile, tile)))
    mask = np.ones((tile, tile)); mask[:, : tile // 6] = 0
    z = np.where(mask > 0, z, 0)
    return np.stack([z.real, z.imag, mask], 0)[None].astype(np.float32)


FPM_CARD = {
    "task": "refinement of LED-array Fourier ptychographic (FPM) reconstructions from reduced LED sets",
    "inputs": "x float32 [1, 3, T, T] = (Re, Im, mask) of the FLATTENED, NORMALISED EPRY object "
              "(point-source curvature, tilt and mean phase removed: velos_recon.fpm.apply_flatten; then "
              "velos_recon.normalise.normalise)",
    "outputs": "y float32 [1, 2, T, T] = (Re, Im) of the refined object in the same frame",
    "intended_use": "Research and development use inside the Velos/SPARK platform: optional refinement of EPRY "
                    "reconstructions of the in-house LED-array microscope (633 nm, 4x/0.10, 15x15 LEDs at 3 mm, "
                    "about 55-60 mm below the sample) acquired with a reduced LED set (all LEDs inside 1.1 x NA "
                    "plus 30-50 % of the dark-field LEDs), followed by the hybrid physics polish. The physics "
                    "result is always kept and returned alongside.",
    "limitations": [
        "Fine-tuned on ONE field of view per specimen type (blood smear; leaf section), left 65 % of a "
        "645 um patch, five acquisitions; tested on the right 35 %, on unseen LED draws and on sensor "
        "regions never used for training. Other specimens, objectives, wavelengths or LED geometries are "
        "out of distribution.",
        "Validated for the BLOOD SMEAR in the hybrid mode (model + 5 EPRY passes): closer to the full-LED "
        "reconstruction and slightly lower prediction error on the LEDs not used. The model output alone "
        "removes reduced-LED artefacts but does not predict unmeasured LED images better: use it for viewing, "
        "not for quantitative values. NOT validated for the leaf section (its output contradicts the measured "
        "images; prediction error +34 to +105 %).",
        "Outputs are estimates, not measurements: use the data-consistency check (residual on the LEDs used "
        "and prediction error on LEDs not used) and prefer the physics or hybrid result when it disagrees.",
        "References are full-LED EPRY reconstructions, not ground truth; their own repeatability is reported "
        "in the evaluation summary.",
    ],
}


def export(checkpoint: str, out: str, tile: int = 256, overlap: int = 64, version: str = "2.0.0",
           name: str = "velos-refine", eval_summary: str | None = None, opset: int = 17,
           extra_inputs: list | None = None, modality: str = "ptychography") -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = build_model(cfg).eval()
    model.load_state_dict(state["model"])
    example = torch.from_numpy(_realistic_input(tile))

    # TorchScript (trace at the fixed tile size: no padding branch is taken at T = 2^k)
    ts_path = out / "model.ts"
    with torch.no_grad():
        ts = torch.jit.trace(model, example)
    ts.save(str(ts_path))

    # ONNX (static [1,3,T,T] → [1,2,T,T]; the runtime handles any field size by tiling)
    onnx_path = out / "model.onnx"
    exporter = "torch.onnx (TorchScript-based)"
    try:
        torch.onnx.export(model, (example,), str(onnx_path), input_names=["x"], output_names=["y"],
                          opset_version=opset, dynamo=False)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, (example,), str(onnx_path), input_names=["x"], output_names=["y"],
                          opset_version=opset)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    # numerical verification: eager PyTorch vs ONNX Runtime vs TorchScript
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tests = [_realistic_input(tile, s) for s in range(3)] + list(extra_inputs or [])
    diffs = {"onnxruntime": [], "torchscript": []}
    timing = {"pytorch_ms": [], "onnxruntime_ms": [], "torchscript_ms": []}
    ts_loaded = torch.jit.load(str(ts_path))
    for x in tests:
        xt = torch.from_numpy(x)
        with torch.no_grad():
            t0 = time.perf_counter(); ref = model(xt).numpy(); timing["pytorch_ms"].append(1e3 * (time.perf_counter() - t0))
            t0 = time.perf_counter(); y_ts = ts_loaded(xt).numpy(); timing["torchscript_ms"].append(1e3 * (time.perf_counter() - t0))
        t0 = time.perf_counter(); y_ort = sess.run(None, {"x": x})[0]; timing["onnxruntime_ms"].append(1e3 * (time.perf_counter() - t0))
        diffs["onnxruntime"].append(float(np.abs(y_ort - ref).max()))
        diffs["torchscript"].append(float(np.abs(y_ts - ref).max()))
    verification = {"n_inputs": len(tests), "max_abs_diff": {k: max(v) for k, v in diffs.items()},
                    "tolerance": 1e-4, "passed": all(max(v) < 1e-4 for v in diffs.values()),
                    "cpu_latency_ms_per_tile_median": {k: round(float(np.median(v)), 1) for k, v in timing.items()},
                    "platform": f"{platform.processor() or platform.machine()} · {torch.get_num_threads()} threads"}
    if not verification["passed"]:
        raise RuntimeError(f"export verification failed: {verification}")

    ev = json.loads(Path(eval_summary).read_text()) if eval_summary else None
    m = cfg.get("model", {})
    card = {
        "name": name, "version": version,
        "created": time.strftime("%Y-%m-%d"),
        "task": "refinement of physics-based ptychographic reconstructions from reduced (sparse) scans",
        "architecture": {"type": "RefineUNet (residual U-Net, GroupNorm, SiLU)", "base_channels": m.get("base"),
                         "depth": m.get("depth"), "parameters": count_params(model)},
        "inputs": "x float32 [1, 3, T, T] = (Re, Im, coverage mask) of the NORMALISED physics object",
        "outputs": "y float32 [1, 2, T, T] = (Re, Im) of the refined object in the same normalised frame",
        "normalisation": cfg.get("data", {}).get("norm", "median"),
        "normalisation_details": "remove linear phase ramp (median wrapped gradients), divide by median "
                                 "amplitude and circular-median phase over the coverage mask; "
                                 "velos_recon.normalise.normalise / denormalise",
        "tile": tile, "tile_overlap": overlap,
        "runtime": {"onnx_opset": opset, "exporter": exporter, "torch": torch.__version__,
                    "onnxruntime": ort.__version__},
        "training": {"config": {k: cfg.get(k) for k in ("name", "data", "model", "train")},
                     "epoch": state.get("epoch"), "val_loss": state.get("val_loss")},
        "evaluation": ev,
        "verification": verification,
        # consistency flag (backend worker): warn when the final estimate's data error exceeds BOTH
        # ratio × physics error AND the absolute floor. Calibrated on the M3 evaluation: v2 stayed ≤ 1.5×
        # physics on every experimental case; unsuitable models (M2, synthetic-only) reached 1.5–3.1×.
        # The floor avoids false alarms on high-SNR data where physics fits the frames almost perfectly.
        "data_error_ratio_warn": 2.0,
        "data_error_abs_floor": 0.02,
        "recommended_hybrid_iters": {"experimental": 10, "synthetic": 30},
        "intended_use": "Research and development use inside the Velos/SPARK platform: optional refinement "
                        "of ePIE reconstructions of reduced visible-light ptychography scans "
                        "(about 35-80 % of a dense raster), followed by the hybrid physics polish. "
                        "The physics result is always kept and returned alongside.",
        "limitations": [
            "Trained on simulated cell phantoms and ONE experimental specimen (635 nm clam section, left "
            "65 % of the field); tested on held-out regions/frame draws and one unseen specimen (675 nm "
            "plant section). Not validated on biological cell samples — required before any claim (M4).",
            "Outputs are estimates, not measurements. Use the data-consistency check: if the refined "
            "object fits the measured diffraction data much worse than the physics result, prefer the "
            "physics or hybrid result.",
            "Pixel sizes / probe sizes outside the training range (about 5-10 µm pixels at 128-px frames; "
            "64-px simulated probes) are out of distribution.",
        ],
        "owner": "Velos BioImaging (all rights reserved)",
        "files": {},
    }
    card["modality"] = modality
    if modality == "fpm":
        card.update(FPM_CARD)
        card["recommended_hybrid_iters"] = {"experimental": 5}      # M4 evaluation: 5 ≈ 10 passes, 5 better near the field edge
    for f in (onnx_path, ts_path):
        card["files"][f.name] = {"sha256": _sha256(f), "bytes": f.stat().st_size}
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=float))
    return card


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=256); ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--version", default="2.0.0"); ap.add_argument("--name", default="velos-refine")
    ap.add_argument("--eval", default=None, help="evaluation summary JSON to embed in the card")
    ap.add_argument("--modality", default="ptychography", choices=["ptychography", "fpm"])
    a = ap.parse_args(argv)
    card = export(a.checkpoint, a.out, a.tile, a.overlap, a.version, a.name, a.eval, modality=a.modality)
    print(json.dumps({"files": card["files"], "verification": card["verification"]}, indent=2))


if __name__ == "__main__":
    main()
