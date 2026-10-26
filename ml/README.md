# velos-ml — ML training pipeline (Module 3)

First baseline (M2): **sparse-scan refinement** — a residual U-Net that maps the physics reconstruction of a
sparse scan (−47 % diffraction frames) toward the ground truth. Everything is config-driven and seeded.

```bash
pip install -e ../backend -e .          # velos_recon (Client's engines) + velos_ml (+ torch)

# 1. data: phantoms → Client's forward model → Client's engine on sparse and dense scans
python -m velos_ml.simulate --config configs/data_synthetic.yaml --out data/synthetic --workers 4

# 2. train (CPU ok for the baseline; GPU picks up automatically)
python -m velos_ml.train --config configs/baseline_unet.yaml --data data/synthetic --out runs/baseline_unet
#    override anything: --set train.epochs=60 --set model.base=48

# 3. evaluate against ground truth and the physics baseline (PSNR/SSIM/NRMSE), with figures
python -m velos_ml.evaluate --data data/synthetic --checkpoint runs/baseline_unet/best.pt \
                            --out results/baseline_unet --control-iters 300

# 4. package for inference (M3) and use from the backend worker
python -m velos_ml.infer runs/baseline_unet/best.pt --export runs/baseline_unet/model.torchscript.pt
export VELOS_ML_CHECKPOINT=$PWD/runs/baseline_unet/best.pt     # worker applies it when a job sets refine_ml=true
```

## Layout
| File | Role |
|---|---|
| `velos_ml/phantoms.py` | synthetic cell phantoms (dome-shaped phase, nucleus, absorption, texture), probe variants |
| `velos_ml/simulate.py` | dataset generation; writes `sample_XXXX.npz` (gt, sparse/dense recon, masks, metadata) + `generation_log.json` |
| `velos_ml/dataset.py` | torch `Dataset` with self-normalisation (median amplitude → 1, median phase → 0), crops, flips/rotations |
| `velos_ml/model.py` | `RefineUNet` (3 → 2 channels, residual, GroupNorm/SiLU; 1.9 M params at base=32, depth=3) |
| `velos_ml/train.py` | AdamW + OneCycle, masked L1, checkpoints (`best.pt`, `last.pt`), `log.csv`, curves |
| `velos_ml/evaluate.py` | metrics table (markdown/json/csv), comparison grid, bar chart; optional "more iterations" physics control |
| `velos_ml/infer.py` | `refine_object(obj, mask, checkpoint)` used by the backend worker; TorchScript export; tiling for large fields |
| `configs/` | `data_synthetic.yaml`, `baseline_unet.yaml` |

## Experiment log (M2)
See `../evidence/ml/` for the trained baseline: `summary.md` (metrics vs ground truth and physics baseline),
`metrics.csv` (per sample), `comparison_grid.png`, `metrics_bars.png`, `training_curves.png`, `log.csv`, `best.pt`.
Unsuccessful/negative observations are recorded in `../docs/data-review.md` (engine behaviour, overlap regimes).

## Limitations (reported, not hidden)
* CPU training, small model and dataset — a first baseline, not a tuned model.
* Synthetic training domain; real-data behaviour shown qualitatively only until experimental training data exist (M3).
* Metrics are computed after removing ptychographic ambiguities (global scalar, linear phase ramp) inside the region
  illuminated by both scans.
