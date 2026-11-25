#!/usr/bin/env bash
# M4 demo: real LED-array FPM data end to end through the API — upload → full-LED reference job →
# reduced-LED job with the FPM AI model (ONNX Runtime) and the hybrid EPRY polish → every estimate scored
# against the measured frames, incl. the LEDs that were NOT used.
#
#   usage: scripts/demo_m4.sh <fpm_dataset.npz> [fraction] [passes] [hybrid_passes] [out_dir]
#
# Requires a running API + worker with VELOS_MODEL_DIR containing an FPM model package
# (modality "fpm"). Set API and KEY in the environment if not default. Output is Markdown.
set -euo pipefail
DATA=${1:?FPM dataset (.npz)}; FRAC=${2:-0.3}; PASSES=${3:-24}; HYB=${4:-10}; OUT=${5:-demo_m4_out}
API=${API:-http://localhost:8000}; KEY=${KEY:-dev-key}
mkdir -p "$OUT"
j() { python3 -c 'import sys, json
d = json.load(sys.stdin)
for k in sys.argv[1].split("."):
    d = d[int(k)] if isinstance(d, list) else d.get(k)
print(json.dumps(d) if isinstance(d, (dict, list)) else d)' "$1"; }
wait_job() {   # $1 job id → prints the status lines, returns when finished
  local JID=$1 LAST="" S ST MSG LINE
  for i in $(seq 1 1800); do
    S=$(curl -s -H "X-API-Key: $KEY" "$API/api/v1/jobs/$JID")
    ST=$(echo "$S" | j status); MSG=$(echo "$S" | j message)
    LINE="status=$ST  $MSG"
    if [[ "$LINE" != "$LAST" && ( $((i % 10)) -eq 1 || "$ST" != "running" ) ]]; then printf "t=%4ds  %s\n" "$i" "$LINE"; LAST="$LINE"; fi
    [[ "$ST" == "succeeded" || "$ST" == "failed" || "$ST" == "cancelled" ]] && break
    sleep 1
  done
}

echo "## 1. Health and installed model packages"; echo '```'
curl -s "$API/health" | python3 -m json.tool
curl -s -H "X-API-Key: $KEY" "$API/api/v1/models" | python3 -c '
import sys, json
for c in json.load(sys.stdin):
    print(json.dumps({k: c.get(k) for k in ("id", "name", "version", "modality", "default", "normalisation", "tile")}))'
echo '```'

echo; echo "## 2. Upload the FPM dataset"; echo '```'
UP=$(curl -s -H "X-API-Key: $KEY" -F "file=@$DATA" -F "name=$(basename "$DATA")" "$API/api/v1/datasets")
echo "$UP" | python3 -c '
import sys, json
d = json.load(sys.stdin); s = d.get("summary", {})
print(json.dumps({"id": d.get("id"), "name": d.get("name"), "size_bytes": d.get("size_bytes"),
                  "summary": {k: s.get(k) for k in ("modality", "n_patterns", "frame_shape", "stored_region", "led_color",
                                                     "camera", "exposure_us")}}, indent=1))'
echo '```'
DS=$(echo "$UP" | j id)

echo; echo "## 3. Reference: all LEDs, physics only (EPRY, $PASSES passes)"; echo '```'
BODY="{\"dataset_id\":\"$DS\",\"engine\":\"fpm\",\"iters\":$PASSES}"
echo "POST /api/v1/jobs $BODY"
RJ=$(curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d "$BODY" "$API/api/v1/jobs")
RJID=$(echo "$RJ" | j id); echo "job $RJID: $(echo "$RJ" | j status)"
wait_job "$RJID"
curl -s -H "X-API-Key: $KEY" "$API/api/v1/jobs/$RJID/result" > "$OUT/reference_result.json"
python3 -c '
import sys, json
s = json.load(open(sys.argv[1]))["summary"]; r = s["reconstruction"]
print(json.dumps({"leds_used": r.get("frames_used"), "grid_px": r.get("N_obj"), "pixel_um": r.get("pixel_um"),
                  "synthetic_NA": r.get("synthetic_NA"), "resolution_um": r.get("resolution_um"),
                  "final_residual": [a["final_residual"] for a in r.get("algorithms", [])],
                  "data_error_normalised": r.get("data_error_normalised"), "elapsed_s": s.get("elapsed_s")}, indent=1, default=str))' "$OUT/reference_result.json"
for A in preview.png result.npz; do
  curl -s -H "X-API-Key: $KEY" -o "$OUT/reference_$A" "$API/api/v1/jobs/$RJID/artifacts/$A"
done
echo '```'

echo; echo "## 4. Reduced LED set (bright field + ${FRAC} of the dark-field LEDs) → FPM model → hybrid ($HYB passes)"; echo '```'
BODY="{\"dataset_id\":\"$DS\",\"engine\":\"fpm\",\"iters\":$PASSES,\"sparse_fraction\":$FRAC,\"refine_ml\":true,\"hybrid_iters\":$HYB}"
echo "POST /api/v1/jobs $BODY"
JOB=$(curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d "$BODY" "$API/api/v1/jobs")
echo "$JOB" | python3 -m json.tool
JID=$(echo "$JOB" | j id)
wait_job "$JID"
echo '```'

echo; echo "## 5. Result — physics, AI model and hybrid, each scored against the measured frames"; echo '```'
RES=$(curl -s -H "X-API-Key: $KEY" "$API/api/v1/jobs/$JID/result")
echo "$RES" > "$OUT/result.json"
echo "$RES" | python3 -c '
import sys, json
s = json.load(sys.stdin)["summary"]; r = s["reconstruction"]; m = s.get("ml_refinement", {})
print(json.dumps({"dataset": {k: s["dataset"].get(k) for k in ("name", "n_patterns", "stored_region")},
                  "reduced_leds": r.get("reduced_leds"),
                  "data_error_used_leds": m.get("data_error"),
                  "prediction_error_unused_leds": m.get("prediction_error_unused_frames") or {"physics": r.get("prediction_error_unused_frames")},
                  "model": {k: m.get(k) for k in ("model", "model_version", "runtime", "elapsed_s")},
                  "hybrid": {k: v for k, v in (m.get("hybrid") or {}).items() if k != "files"},
                  "consistency": m.get("consistency"), "elapsed_s": s.get("elapsed_s")}, indent=1, default=str))'
echo '```'

echo; echo "## 6. Artifacts"; echo '```'
for A in comparison_ml.png preview.png refined_preview.png hybrid_preview.png convergence.png result.npz refined.npz hybrid.npz summary.json; do
  code=$(curl -s -H "X-API-Key: $KEY" -o "$OUT/$A" -w "%{http_code}" "$API/api/v1/jobs/$JID/artifacts/$A")
  printf "%-20s HTTP %s %9d bytes\n" "$A" "$code" "$(stat -c %s "$OUT/$A")"
done; echo '```'
echo; echo "reference_job=$RJID job=$JID dataset=$DS" | tee "$OUT/ids.txt"
