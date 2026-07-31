#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <generated-stage2-config.json> <run-id>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="$1"
RUN_ID="$2"

cd "${PROJECT_ROOT}"
exec python3 -m model5_o2.launch --config "${CONFIG_PATH}" --run-id "${RUN_ID}"
