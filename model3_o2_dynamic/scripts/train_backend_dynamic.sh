#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# The shared backend launcher remains unchanged by default.  This opt-in
# entrypoint only switches its Hydra train function to DynamicWan22Trainer.
export TRAIN_ENTRYPOINT="${PROJECT_ROOT}/model3_o2_dynamic/backend_train.py"
exec bash "${PROJECT_ROOT}/model3/scripts/train_backend.sh" "$@"
