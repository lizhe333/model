#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)_model5_o2_stage1_model5_long_80k}"

cd "${PROJECT_ROOT}"
exec python3 -m model5_o2.launch \
  --config model5_o2/configs/libero_long_stage1_model5_80k.json \
  --run-id "${RUN_ID}"
