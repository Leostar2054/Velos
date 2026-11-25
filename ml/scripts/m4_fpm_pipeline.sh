#!/usr/bin/env bash
# M4 real-sample pipeline (LED-array FPM): data -> fine-tuning -> evaluation -> package.
#   PACK=<folder with the *_c512_meta.mat files> ml/scripts/m4_fpm_pipeline.sh
# Run from ml/. Each step skips work that already exists, so the script can be resumed.
set -euo pipefail
PACK=${PACK:?set PACK to the _velos_packed folder}
OUT=${OUT:-data/fpm}
PASSES=${PASSES:-24}
STAGE=${STAGE:-all}           # data | train | eval | package | all
B=Blood__fpm_cs2100m_raster_20261123
L=LEAF__fpm_cs2100m_raster_20261123
if [[ $STAGE == data || $STAGE == all ]]; then
  # centre region of every complete scan: test draws sub_00/sub_01 + training draws sub_02..sub_05
  for spec in "${B}_222619_c512:blood_222619:3" "${L}_214040_c512:leaf_214040:3" "${B}_223858_c512:blood_223858:3" \
              "${L}_215808_c512:leaf_215808:3" "${B}_225108_c512:blood_225108:3" "${B}_230302_2000000exp_c512:blood_230302_2s:1"; do
    IFS=: read -r src name draws <<< "$spec"
    python -m velos_ml.fpm_realdata --source "$PACK/${src}_meta.mat" --name "$name" --out "$OUT/$name" \
      --draws "$draws" --passes "$PASSES"
  done
  # side regions (never used for training): test draws only
  for side in L512 R512; do
    for spec in "${B}_222619_${side}:blood_222619_${side}" "${L}_214040_${side}:leaf_214040_${side}"; do
      IFS=: read -r src name <<< "$spec"
      [[ -f "$PACK/${src}_meta.mat" ]] || { echo "skip $src (not packed)"; continue; }
      python -m velos_ml.fpm_realdata --source "$PACK/${src}_meta.mat" --name "$name" --out "$OUT/$name" \
        --draws 1 --passes "$PASSES"
    done
  done
fi
if [[ $STAGE == train || $STAGE == all ]]; then
  python -m velos_ml.train --config configs/refine_fpm_v1.yaml --out runs/refine_fpm_v1
fi
if [[ $STAGE == eval || $STAGE == all ]]; then
  DATA=()
  for n in blood_222619 leaf_214040 blood_223858 leaf_215808 blood_225108 blood_230302_2s; do DATA+=("$OUT/$n:holdout"); done
  for n in blood_222619_L512 blood_222619_R512 leaf_214040_L512 leaf_214040_R512; do
    [[ -d "$OUT/$n" ]] && DATA+=("$OUT/$n:full")
  done
  python -m velos_ml.evaluate_fpm --data "${DATA[@]}" \
    --model v2=runs/refine_v2/best.pt --model fpm_v1=runs/refine_fpm_v1/best.pt \
    --hybrid-model fpm_v1 --hybrid 5 10 \
    --repeat "$OUT/blood_222619,$OUT/blood_223858" "$OUT/blood_222619,$OUT/blood_225108" "$OUT/leaf_214040,$OUT/leaf_215808" \
    --out results/m4
fi
if [[ $STAGE == package || $STAGE == all ]]; then
  python -m velos_ml.package runs/refine_fpm_v1/best.pt --out models/velos-refine-fpm-v1 --version 1.0.0 \
    --name velos-refine-fpm --eval results/m4/summary_fpm.json --modality fpm
fi
