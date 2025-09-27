#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo '{"error":"project_id parameter is required","count":"-1","instances":""}'
  exit 0
fi

# Fetch GKE Autopilot maintenance instances (prefixed with gk3-)
# Using name regex filter per gcloud compute docs.
readarray -t INSTANCES < <(gcloud compute instances list \
  --project="${PROJECT_ID}" \
  --filter="name~'gk3-'" \
  --format="value(name)" 2>/dev/null || true)

if (( ${#INSTANCES[@]} == 1 )) && [[ -z "${INSTANCES[0]}" ]]; then
  INSTANCES=()
fi

COUNT=${#INSTANCES[@]}

# Build comma-separated list while ensuring JSON-safe escaping for names
INSTANCE_LIST=""
for NAME in "${INSTANCES[@]}"; do
  if [[ -n "${INSTANCE_LIST}" ]]; then
    INSTANCE_LIST+=";"
  fi
  INSTANCE_LIST+="${NAME}"
done

echo "{\"count\":\"${COUNT}\",\"instances\":\"${INSTANCE_LIST}\"}"
