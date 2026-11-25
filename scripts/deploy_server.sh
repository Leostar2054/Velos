#!/usr/bin/env bash
# (Re)deploy the Velos stack on a server (cloud VM or on-premises), from the repository checkout.
#
#   ssh <server>  →  cd /opt/velos && git pull --ff-only && scripts/deploy_server.sh
#
# Builds the images on the server (the website is built for the real domain from deploy/.env),
# starts / updates the stack with zero manual steps, waits for /health and runs the smoke test.
# MONITORING=1 also starts Prometheus (compose profile "monitoring"); GPU=1 adds the GPU worker
# override (docker-compose.gpu.yml, needs the NVIDIA container toolkit).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f deploy/.env ] || { echo "deploy/.env missing: cp deploy/.env.example deploy/.env and edit it"; exit 1; }
set -a; . deploy/.env; set +a
FILES=(-f docker-compose.prod.yml)
[ "${GPU:-0}" = "1" ] && FILES+=(-f docker-compose.gpu.yml)
PROFILES=()
[ "${MONITORING:-0}" = "1" ] && PROFILES+=(--profile monitoring)
C=(docker compose "${FILES[@]}" --env-file deploy/.env "${PROFILES[@]}")
"${C[@]}" config --quiet
"${C[@]}" build
"${C[@]}" up -d --remove-orphans
for i in $(seq 1 90); do
  curl -fsS "https://${SITE_DOMAIN}/health" > /dev/null 2>&1 && break
  curl -fsS "http://localhost/health" > /dev/null 2>&1 && break
  sleep 2
done
"${C[@]}" ps
if [ -n "${SMOKE_API_KEY:-}" ]; then
  scripts/smoke_test.sh "https://${SITE_DOMAIN}" "$SMOKE_API_KEY"
fi
echo "deployed $(git rev-parse --short HEAD) on ${SITE_DOMAIN}"
