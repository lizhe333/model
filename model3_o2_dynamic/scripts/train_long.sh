#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)_model3_o2_long_from_model3_80k}"

cd "${PROJECT_ROOT}"
exec python3 -m model3_o2.launch \
  --config model3_o2/configs/libero_long_fast.json \
  --run-id "${RUN_ID}"
