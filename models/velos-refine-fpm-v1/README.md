# velos-refine-fpm-v1 — refinement model for the LED-array FPM microscope (M4)

A self-describing inference package for the Velos/SPARK backend. It refines an **EPRY reconstruction from a reduced
LED set** of the in-house LED-array Fourier ptychographic microscope (633 nm, 4×/0.10, 15 × 15 LEDs at 3 mm, about
55–60 mm below the sample) toward the reconstruction from all LEDs. The physics result is always kept next to it, and
every output is scored against the measured LED images.

| File | What it is |
|---|---|
| `model.onnx` | the network for ONNX Runtime (CPU or CUDA): input `x` float32 [1, 3, 256, 256] = (Re, Im, mask) of the flattened, normalised EPRY object → output `y` [1, 2, 256, 256] |
| `model.ts` | the same network as TorchScript (fallback runtime) |
| `model_card.json` | `"modality": "fpm"`, version, training data and configuration, evaluation summary (real samples), intended use, limitations, numerical verification, SHA-256 of the files |

## Install in the backend

```bash
cp -r velos-refine-fpm-v1 models/            # next to velos-refine-v2 (scanning ptychography)
export VELOS_MODEL_DIR=$PWD/models            # docker-compose: mounted at /models
curl -s -H "X-API-Key: $KEY" localhost:8000/api/v1/models     # both packages, each with its "modality"
```

FPM jobs pick the FPM package automatically (a ptychography model is never applied to FPM data, and vice versa):

```json
{"dataset_id": "…", "engine": "fpm", "iters": 24, "sparse_fraction": 0.3, "refine_ml": true, "hybrid_iters": 5}
```

The result contains the physics, AI and hybrid estimates with their data residual on the LEDs used, the prediction
error on the LEDs not used, and a `consistency` flag.

## What it was trained and tested on

Fine-tuned from `velos-refine-v2` on reduced-LED EPRY reconstructions of five real scans (three blood smear,
two leaf section; 23 Nov 2026), left 65 % of the field only. Tested on data never used for training — the right
35 % of the field with unseen LED draws, and two sensor regions left and right of the training region:

| Test (phase PSNR vs all LEDs · prediction error on the LEDs not used) | Physics | AI only | **Hybrid (AI + 5 passes)** |
|---|---|---|---|
| Blood smear, hold-out region (n = 8) | 32.0 dB · 0.314 | +1.6 dB · +2 % | **+0.6 dB · −1 %** |
| Blood smear, unseen sensor regions (n = 4) | 24.8 dB · 0.466 | +2.2 dB · +25 % | **+2.2 dB · −1 %** |
| Leaf section, hold-out + side regions (n = 8) | 16.0 / 13.3 dB | +6.4 / +3.4 dB · +34 / +105 % | +2.0 / +2.1 dB · +3 / +11 % |

Use the **hybrid mode** (`hybrid_iters` 5). The AI output alone removes reduced-LED artefacts but does not predict the
unmeasured images better — for viewing only. **Not validated for the leaf section.** Details: `docs/ml-pipeline.md`
(M4), `evidence/ml_m4/`.

## Check integrity · rebuild

`sha256sum model.onnx model.ts` → compare with `model_card.json` → `files`. `PACK=<_velos_packed> ml/scripts/m4_fpm_pipeline.sh`
reproduces the training pairs, the fine-tuning (from `velos-refine-v2`), the evaluation and this package.

Owner: Velos BioImaging. Research use; see `limitations` in the card.
