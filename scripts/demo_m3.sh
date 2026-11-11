#!/usr/bin/env bash
# M3 demo: end-to-end inference through the backend — reduced scan → physics engine →
# packaged ML model (ONNX Runtime) → hybrid physics polish → data-consistency check.
#
#   usage: scripts/demo_m3.sh <dataset.npz|.mat> <fraction> [iters] [hybrid_iters] [out_dir]
#
# Requires a running API + worker with VELOS_MODEL_DIR containing a model package.
# Set API and KEY in the environment if not default. Output is Markdown (pipe to a file).
# CALIBRATE=1: first reconstruct the full scan and reuse its probe for the reduced scan
# ("probe calibration" workflow: calibrate once, then scan sparsely).
set -euo pipefail
DATA=${1:?dataset path}; FRAC=${2:-0.35}; ITERS=${3:-200}; HYB=${4:-30}; OUT=${5:-demo_m3_out}
API=${API:-http://localhost:8000}; KEY=${KEY:-dev-key}
mkdir -p "$OUT"
j() { python3 -c 'import sys, json
d = json.load(sys.stdin)
for k in sys.argv[1].split("."):
    d = d[int(k)] if isinstance(d, list) else d.get(k)
print(json.dumps(d) if isinstance(d, (dict, list)) else d)' "$1"; }

echo "## 1. Health — API version and installed model packages"; echo '```'
curl -s "$API/health" | python3 -m json.tool; echo '```'

echo; echo "## 2. Model card of the default package (GET /api/v1/models)"; echo '```'
curl -s -H "X-API-Key: $KEY" "$API/api/v1/models" | python3 -c '
import sys, json
for c in json.load(sys.stdin):
    ev = c.get("verification", {})
    print(json.dumps({k: c.get(k) for k in ("id", "name", "version", "created", "default", "normalisation", "tile",
                                             "tile_overlap", "architecture", "files")}, indent=1))
    print("verification:", json.dumps(ev.get("max_abs_diff")), "passed:", ev.get("passed"))
    print("intended use:", c.get("intended_use"))'
echo '```'

echo; echo "## 3. Upload the dataset"; echo '```'
UP=$(curl -s -H "X-API-Key: $KEY" -F "file=@$DATA" -F "name=$(basename "$DATA")" "$API/api/v1/datasets")
echo "$UP" | python3 -m json.tool; echo '```'
DS=$(echo "$UP" | j id)

CAL_FIELD=""
if [[ "${CALIBRATE:-0}" == "1" ]]; then
  echo; echo "## 3b. Probe calibration: full scan (all frames), physics only"; echo '```'
  CJ=$(curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
       -d "{\"dataset_id\":\"$DS\",\"engine\":\"epie\",\"iters\":$ITERS}" "$API/api/v1/jobs")
  CJID=$(echo "$CJ" | j id); echo "calibration job $CJID queued"
  for i in $(seq 1 1200); do
    ST=$(curl -s -H "X-API-Key: $KEY" "$API/api/v1/jobs/$CJID" | j status)
    [[ "$ST" == "succeeded" || "$ST" == "failed" ]] && break; sleep 1
  done
  echo "calibration job $CJID: $ST after ~${i}s"
  curl -s -H "X-API-Key: $KEY" "$API/api/v1/jobs/$CJID/result" | python3 -c 'import sys, json; s = json.load(sys.stdin)["summary"]["reconstruction"]; print(json.dumps(s, indent=1))'
  curl -s -H "X-API-Key: $KEY" -o "$OUT/calibration_preview.png" "$API/api/v1/jobs/$CJID/artifacts/preview.png"
  echo '```'
  CAL_FIELD=",\"probe_from_job\":\"$CJID\""
fi

echo; echo "## 4. Submit: reduced scan (${FRAC} of the frames) + ML refinement + hybrid (${HYB} it)"; echo '```'
BODY="{\"dataset_id\":\"$DS\",\"engine\":\"epie\",\"iters\":$ITERS,\"sparse_fraction\":$FRAC,\"refine_ml\":true,\"hybrid_iters\":$HYB$CAL_FIELD}"
echo "POST /api/v1/jobs $BODY"
JOB=$(curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d "$BODY" "$API/api/v1/jobs")
echo "$JOB" | python3 -m json.tool; echo '```'
JID=$(echo "$JOB" | j id)

echo; echo "## 5. Status polling"; echo '```'
LAST=""
for i in $(seq 1 900); do
  S=$(curl -s -H "X-API-Key: $KEY" "$API/api/v1/jobs/$JID")
  ST=$(echo "$S" | j status); MSG=$(echo "$S" | j message)
  LINE="status=$ST  $MSG"
  if [[ "$LINE" != "$LAST" && ( $((i % 5)) -eq 1 || "$ST" != "running" ) ]]; then printf "t=%4ds  %s\n" "$i" "$LINE"; LAST="$LINE"; fi
  [[ "$ST" == "succeeded" || "$ST" == "failed" || "$ST" == "cancelled" ]] && break
  sleep 1
done; echo '```'

echo; echo "## 6. Result — physics, ML and hybrid, scored against the measured data"; echo '```'
RES=$(curl -s -H "X-API-Key: $KEY" "$API/api/v1/jobs/$JID/result")
echo "$RES" > "$OUT/result.json"
echo "$RES" | python3 -c '
import sys, json
r = json.load(sys.stdin); s = r["summary"]
print(json.dumps({"dataset": {k: s["dataset"][k] for k in ("name", "n_patterns", "frame_shape", "wavelength_nm", "sample_pixel_um", "object_shape")},
                  "reconstruction": s["reconstruction"],
                  "ml_refinement": {k: v for k, v in s["ml_refinement"].items() if k != "files"}}, indent=1))'
echo '```'

echo; echo "## 7. Artifacts"; echo '```'
for A in comparison.png preview.png refined_preview.png hybrid_preview.png probe.png convergence.png refined.npz hybrid.npz result.npz summary.json; do
  code=$(curl -s -H "X-API-Key: $KEY" -o "$OUT/$A" -w "%{http_code}" "$API/api/v1/jobs/$JID/artifacts/$A")
  printf "%-20s HTTP %s %9d bytes\n" "$A" "$code" "$(stat -c %s "$OUT/$A")"
done; echo '```'
echo; echo "job_id=$JID dataset_id=$DS" | tee "$OUT/ids.txt"
