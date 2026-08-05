#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)_model3_o2_dynamic_object}"

cd "${PROJECT_ROOT}"
exec python3 -m model3_o2_dynamic.launch \
  --config "${STAGE2_CONFIG:?Set STAGE2_CONFIG to the materialized Stage-2 config}" \
  --run-id "${RUN_ID}"
