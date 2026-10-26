#!/usr/bin/env bash
# M2 mid-project demo: end-to-end API flow against a running API + worker.
#   usage: scripts/demo_m2.sh <dataset.npz|.mat> [engine] [iters] [out_dir]
# Requires: curl, python3 (for JSON parsing). Set API and KEY in the environment if not default.
set -euo pipefail
DATA=${1:?dataset path}; ENGINE=${2:-epie}; ITERS=${3:-60}; OUT=${4:-demo_out}
API=${API:-http://localhost:8000}; KEY=${KEY:-dev-key}
mkdir -p "$OUT"
j() { python3 -c "import sys,json; print(json.load(sys.stdin)$1)"; }

echo "## 1. Health (public)"; echo '```'; curl -s "$API/health"; echo; echo '```'
echo; echo "## 2. Auth check — no key → 401"; echo '```'
curl -s -o /dev/null -w "HTTP %{http_code}\n" "$API/api/v1/datasets"; echo '```'

echo; echo "## 3. Upload dataset"; echo '```'
UP=$(curl -s -H "X-API-Key: $KEY" -F "file=@$DATA" -F "name=$(basename "$DATA")" "$API/api/v1/datasets")
echo "$UP" | python3 -m json.tool; echo '```'
DS=$(echo "$UP" | j "['id']")

echo; echo "## 4. Submit job (returns immediately)"; echo '```'
JOB=$(curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
      -d "{\"dataset_id\":\"$DS\",\"engine\":\"$ENGINE\",\"iters\":$ITERS}" "$API/api/v1/jobs")
echo "$JOB" | python3 -m json.tool; echo '```'
JID=$(echo "$JOB" | j "['id']")

echo; echo "## 5. Poll status (web request never blocks; worker runs the job)"; echo '```'
for i in $(seq 1 600); do
  S=$(curl -s -H "X-API-Key: $KEY" "$API/api/v1/jobs/$JID")
  ST=$(echo "$S" | j "['status']"); MSG=$(echo "$S" | j "['message']"); PR=$(echo "$S" | j "['progress']")
  printf "t=%3ds  status=%-10s progress=%.2f  %s\n" "$((i*2))" "$ST" "$PR" "$MSG"
  [[ "$ST" == "succeeded" || "$ST" == "failed" || "$ST" == "cancelled" ]] && break
  sleep 2
done; echo '```'

echo; echo "## 6. Result summary"; echo '```'
RES=$(curl -s -H "X-API-Key: $KEY" "$API/api/v1/jobs/$JID/result"); echo "$RES" | python3 -m json.tool; echo '```'

echo; echo "## 7. Download artifacts"; echo '```'
for A in amplitude.png phase.png preview.png probe.png convergence.png summary.json result.npz; do
  curl -s -H "X-API-Key: $KEY" -o "$OUT/$A" "$API/api/v1/jobs/$JID/artifacts/$A"
  printf "%-16s %8d bytes\n" "$A" "$(stat -c %s "$OUT/$A")"
done; echo '```'
echo; echo "job_id=$JID dataset_id=$DS" | tee "$OUT/ids.txt"
