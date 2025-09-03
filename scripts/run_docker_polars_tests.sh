#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-docker_polars_test.yml}
COMPOSE="docker compose"
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE="docker-compose"
fi

echo "[+] Bringing up test stack..."
$COMPOSE -f "$COMPOSE_FILE" up -d --build

echo "[+] Validating Polars across containers..."
python validation/validate_polars_docker.py || true

echo "[+] Running pytest docker-marked tests from host..."
pytest -q -m docker || true

echo "[+] Optionally running tests inside pytest-runner (if enabled)..."
if $COMPOSE -f "$COMPOSE_FILE" ps | grep -q pytest-runner; then
  $COMPOSE -f "$COMPOSE_FILE" logs -f pytest-runner || true
fi

echo "[+] Done. To tear down: $COMPOSE -f $COMPOSE_FILE down"

