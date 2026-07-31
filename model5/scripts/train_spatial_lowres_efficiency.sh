#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${MODEL5_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
export MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-False}"

cd "${PROJECT_ROOT}"
exec python3 -m model5.launch \
  --config "${PROJECT_ROOT}/model5/configs/libero_spatial_lowres_efficiency.json" \
  --run-id "${RUN_ID}" \
  "$@"
