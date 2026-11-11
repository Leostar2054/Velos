# velos-refine-v2 — refinement model package (M3)

A self-describing inference package for the Velos/SPARK backend. It refines a physics-based (ePIE)
reconstruction from a **reduced** ptychographic scan; the physics result is always kept next to it.

| File | What it is |
|---|---|
| `model.onnx` | the network for ONNX Runtime (CPU or CUDA): input `x` float32 [1, 3, 256, 256] = (Re, Im, coverage mask) of the normalised physics object → output `y` [1, 2, 256, 256] |
| `model.ts` | the same network as TorchScript (fallback runtime) |
| `model_card.json` | version, training data and configuration, evaluation summary, intended use, limitations, numerical verification, data-consistency warning threshold, SHA-256 of the files |

## Install in the backend

```bash
mkdir -p backend/models && cp -r velos-refine-v2 backend/models/
export VELOS_MODEL_DIR=$PWD/backend/models            # docker-compose: mounted at /models
curl -s localhost:8000/health                          # "models": {"installed": ["velos-refine-v2"], ...}
curl -s -H "X-API-Key: $KEY" localhost:8000/api/v1/models
```

Use it in a job (the physics result, the ML-refined result and the hybrid result are all returned, each
scored against the measured data):

```json
{"dataset_id": "…", "engine": "epie", "iters": 200, "sparse_fraction": 0.35,
 "refine_ml": true, "hybrid_iters": 30}
```

For samples whose reduced scans do not converge on their own (e.g. the 675 nm plant data), first run a
full scan of the same illumination and pass its job id as `"probe_from_job"` (probe calibration).

## Check integrity

```bash
sha256sum model.onnx model.ts        # compare with model_card.json → files
```

## Rebuild from source

`ml/scripts/m3_pipeline.sh` reproduces the data, both training stages, the evaluation and this package;
`python -m velos_ml.package <checkpoint> --out <dir>` re-exports and re-verifies an existing checkpoint.

Owner: Velos BioImaging. Not validated on biological cell samples yet (M4) — see `limitations` in the card.
