#!/usr/bin/env bash
# Post-deployment smoke test (staging / production / CI).
#
#   scripts/smoke_test.sh BASE_URL API_KEY [dataset.npz]
#
# BASE_URL is the public address (edge server), e.g. https://velos.example.com or
# http://localhost:8000 for an API-only run (set SKIP_WEB=1). Without a dataset a small
# synthetic one is generated. Exits non-zero at the first failed check.
set -euo pipefail
BASE=${1:?usage: smoke_test.sh BASE_URL API_KEY [dataset.npz]}
KEY=${2:?API key required}
DATA=${3:-}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; exit 1; }
json() { python3 -c 'import sys, json
d = json.load(sys.stdin)
for k in sys.argv[1].split("."):
    d = d[int(k)] if isinstance(d, list) else d.get(k)
print("" if d is None else (json.dumps(d) if isinstance(d, (dict, list)) else d))' "$1"; }

# 1. health (public)
H=$(curl -fsS "$BASE/health") || fail "GET /health"
[ "$(echo "$H" | json status)" = "ok" ] || fail "health status: $H"
pass "health ok · version $(echo "$H" | json version) · default model: $(echo "$H" | json models.default)"

# 2. website pages (edge only)
if [ "${SKIP_WEB:-0}" != "1" ]; then
  for p in / /product/ /team/ /contact/; do
    code=$(curl -s -o "$TMP/page.html" -w '%{http_code}' "$BASE$p")
    [ "$code" = "200" ] && grep -q "Velos" "$TMP/page.html" || fail "website $p → HTTP $code"
  done
  pass "website pages / /product/ /team/ /contact/"
fi

# 3. authentication enforced
[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/v1/datasets")" = "401" ] || fail "unauthenticated request not rejected"
pass "API rejects requests without a key"

# 4. models
M=$(curl -fsS -H "X-API-Key: $KEY" "$BASE/api/v1/models") || fail "GET /api/v1/models"
NMODELS=$(echo "$M" | python3 -c 'import sys,json; print(len(json.load(sys.stdin)))')
pass "models endpoint · $NMODELS package(s) installed"

# 5. upload
if [ -z "$DATA" ]; then
  DATA="$TMP/smoke.npz"
  python3 - "$DATA" <<'PY'
import sys, numpy as np
rng = np.random.default_rng(0); N, Np = 96, 32
yy, xx = np.mgrid[0:N, 0:N]
obj = np.exp(1j * 0.8 * np.exp(-((yy - 48) ** 2 + (xx - 40) ** 2) / 300.0))
r = np.hypot(*np.mgrid[-Np/2:Np/2, -Np/2:Np/2]); probe = (r < 10).astype(complex)
pos = [(i, j) for i in range(0, N - Np + 1, 8) for j in range(0, N - Np + 1, 8)]
dps = np.stack([np.abs(np.fft.fftshift(np.fft.fft2(probe * obj[i:i+Np, j:j+Np]))) ** 2 / Np ** 2 * 1e4 for i, j in pos])
dx, wl, z = 1e-6, 1e-6, 1e-2
np.savez(sys.argv[1], dps=dps.astype(np.float32), positions=np.array([(j * dx, i * dx) for i, j in pos]),
         wavelength=wl, pixel_pitch=wl * z / (Np * dx), distance=z, init_probe=probe)
PY
fi
U=$(curl -fsS -H "X-API-Key: $KEY" -F "file=@$DATA" -F "name=smoke-test" "$BASE/api/v1/datasets") || fail "upload"
DS=$(echo "$U" | json id)
pass "upload · dataset $DS · $(echo "$U" | json summary.n_patterns) patterns"

# 6. job (ML refinement + hybrid when a model is installed)
REFINE=false; [ "$NMODELS" -gt 0 ] && REFINE=true
J=$(curl -fsS -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d "{\"dataset_id\": \"$DS\", \"engine\": \"epie\", \"iters\": 20, \"refine_ml\": $REFINE, \"hybrid_iters\": $([ $REFINE = true ] && echo 5 || echo 0)}" \
     "$BASE/api/v1/jobs") || fail "submit job"
JOB=$(echo "$J" | json id)
pass "job submitted · $JOB · refine_ml=$REFINE"
for i in $(seq 1 ${TIMEOUT_S:-300}); do
  S=$(curl -fsS -H "X-API-Key: $KEY" "$BASE/api/v1/jobs/$JOB" | json status)
  case "$S" in succeeded) break ;; failed|cancelled) fail "job ended $S: $(curl -fsS -H "X-API-Key: $KEY" "$BASE/api/v1/jobs/$JOB" | json error)" ;; esac
  sleep 1
done
[ "$S" = "succeeded" ] || fail "job not finished in time (status $S)"
pass "job succeeded after ~${i}s"

# 7. result + artifact
R=$(curl -fsS -H "X-API-Key: $KEY" "$BASE/api/v1/jobs/$JOB/result") || fail "result"
code=$(curl -s -o "$TMP/phase.png" -w '%{http_code}' -H "X-API-Key: $KEY" "$BASE/api/v1/jobs/$JOB/artifacts/phase.png")
[ "$code" = "200" ] && file "$TMP/phase.png" 2>/dev/null | grep -qi png || [ "$code" = "200" ] || fail "artifact download → $code"
pass "result + artifact download (phase.png)"
if [ "$REFINE" = true ]; then
  C=$(echo "$R" | json summary.ml_refinement.consistency)
  pass "ML refinement · model $(echo "$R" | json summary.ml_refinement.model) · runtime $(echo "$R" | json summary.ml_refinement.runtime) · consistency: $C"
fi
echo "ALL CHECKS PASSED ($BASE)"
