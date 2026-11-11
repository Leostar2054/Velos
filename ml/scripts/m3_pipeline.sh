#!/usr/bin/env bash
# M3 pipeline — reproduces the refinement model v2 end to end (CPU: ~3–4 h on 2 cores; GPU much faster).
#
#   cd ml && MAT_DIR="<folder with 'real data - clam.mat' and 'real data - plant.mat'>" scripts/m3_pipeline.sh
#
# Steps: synthetic data v2 → experimental data (clam training/hold-out, plant unseen, probe-calibrated
# plant, repeat references) → stage-1 training (synthetic) → stage-2 fine-tuning (+ real) → evaluation →
# figures → package (ONNX + TorchScript + model card, verified).
set -euo pipefail
MAT_DIR=${MAT_DIR:?set MAT_DIR to the folder with the .mat example datasets}
W=${WORKERS:-2}
mkdir -p data runs results models

python -m velos_ml.simulate --config configs/data_synthetic_v2.yaml --out data/synthetic_v2 --workers "$W"

python -m velos_ml.realdata --mat "$MAT_DIR/real data - clam.mat" --name clam --crop 128 \
       --out data/real_clam --n-subsets 14 --repeat-seed 11
python -m velos_ml.realdata --mat "$MAT_DIR/real data - plant.mat" --name plant --crop 128 \
       --out data/real_plant --n-subsets 3 --fixed 0.20 0.35 0.50
mkdir -p data/real_plant_cal && cp data/real_plant/dense.npz data/real_plant_cal/
python -m velos_ml.realdata --mat "$MAT_DIR/real data - plant.mat" --name plant --crop 128 \
       --out data/real_plant_cal --n-subsets 3 --fixed 0.20 0.35 0.50 --probe-from-dense --repeat-seed 11

python -m velos_ml.train --config configs/refine_v2_synthetic.yaml --out runs/refine_v2_synthetic
python -m velos_ml.train --config configs/refine_v2.yaml --out runs/refine_v2

python -m velos_ml.evaluate_m3 --synthetic data/synthetic_v2 \
    --real "data/real_clam::holdout::$MAT_DIR/real data - clam.mat::0,1" \
           "data/real_plant_cal::full::$MAT_DIR/real data - plant.mat::0,1,2" \
           "data/real_plant::full::$MAT_DIR/real data - plant.mat::1,2" \
    --model m2=runs/baseline_unet/best.pt --model v2_syn=runs/refine_v2_synthetic/best.pt \
    --model v2=runs/refine_v2/best.pt --hybrid-model v2 --hybrid-iters 10 30 100 --out results/m3
python -m velos_ml.figures_m3 results/m3 --runs runs/refine_v2_synthetic runs/refine_v2

python -m velos_ml.package runs/refine_v2/best.pt --out models/velos-refine-v2 --version 2.0.0 \
       --eval results/m3/summary.json
