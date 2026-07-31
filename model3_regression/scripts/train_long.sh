#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${MODEL3_REGRESSION_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

cd "${PROJECT_ROOT}"
exec python3 -m model3_regression.launch \
  --config "${PROJECT_ROOT}/model3_regression/configs/libero_long.json" \
  --run-id "${RUN_ID}" \
  "$@"
