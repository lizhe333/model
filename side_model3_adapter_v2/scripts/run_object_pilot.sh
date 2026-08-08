#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CACHE_DIR="${LATENT_CACHE_DIR:-${PROJECT_ROOT}/Light-WAM/data/latent_cache_Wan2.1-T2V-1.3B/libero_object_2cam224_side_model3_independent_v1}"
RUN_PREFIX="${RUN_PREFIX:-$(date +%Y-%m-%d_%H-%M-%S)_side_model3_adapter_v2_object}"
PILOT_DIR="${PROJECT_ROOT}/runs/I-003/side_model3_adapter_v2/object_pilot/${RUN_PREFIX}"

SMOKE_B16_ID="${RUN_PREFIX}_b16_ga1_smoke"
SMOKE_B8_ID="${RUN_PREFIX}_b8_ga2_smoke"
FORMAL_B16_ID="${RUN_PREFIX}_b16_ga1_40k"
FORMAL_B8_ID="${RUN_PREFIX}_b8_ga2_40k"

mkdir -p "${PILOT_DIR}"

validate_cache() {
  env PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n lightwam-libero-eval \
    python -c \
      "from side_model3_adapter_v2.data import validate_complete_side_observation_cache; validate_complete_side_observation_cache('${CACHE_DIR}', expected_samples=67309)"
}

run_smoke() {
  local run_id=$1
  local batch_size=$2
  local grad_acc=$3
  local port=$4
  local output_dir="${PROJECT_ROOT}/runs/I-003/side_model3_adapter_v2/backend_runs/${run_id}"
  local log_path="${PILOT_DIR}/${run_id}.log"

  [[ ! -e "${output_dir}" ]] || {
    echo "Refusing to reuse existing smoke output: ${output_dir}" >&2
    return 2
  }
  if env \
    RUN_ID="${run_id}" \
    OUTPUT_DIR="${output_dir}" \
    GPU_IDS="0,1,2,3" \
    NUM_PROCESSES="4" \
    MAIN_PROCESS_PORT="${port}" \
    BATCH_SIZE="${batch_size}" \
    GRAD_ACC="${grad_acc}" \
    MAX_STEPS="1" \
    SAVE_EVERY="1000" \
    WARMUP_STEPS="1" \
    WANDB_MODE="offline" \
    bash "${SCRIPT_DIR}/train_object.sh" > "${log_path}" 2>&1; then
    return 0
  fi
  if rg -qi "CUDA out of memory|OutOfMemoryError|CUDA error: out of memory" "${log_path}"; then
    return 10
  fi
  return 2
}

launch_formal() {
  local run_id=$1
  local batch_size=$2
  local grad_acc=$3
  local port=$4
  local output_dir="${PROJECT_ROOT}/runs/I-003/side_model3_adapter_v2/backend_runs/${run_id}"
  local log_path="${PILOT_DIR}/${run_id}.log"

  [[ ! -e "${output_dir}" ]] || {
    echo "Refusing to reuse existing formal output: ${output_dir}" >&2
    return 1
  }
  exec env \
    RUN_ID="${run_id}" \
    OUTPUT_DIR="${output_dir}" \
    GPU_IDS="0,1,2,3" \
    NUM_PROCESSES="4" \
    MAIN_PROCESS_PORT="${port}" \
    BATCH_SIZE="${batch_size}" \
    GRAD_ACC="${grad_acc}" \
    MAX_STEPS="40000" \
    SAVE_EVERY="5000" \
    WARMUP_STEPS="1000" \
    WANDB_MODE="offline" \
    bash "${SCRIPT_DIR}/train_object.sh" >> "${log_path}" 2>&1
}

validate_cache

smoke_status=0
run_smoke "${SMOKE_B16_ID}" 16 1 29651 || smoke_status=$?
if [[ "${smoke_status}" -eq 0 ]]; then
  launch_formal "${FORMAL_B16_ID}" 16 1 29653
fi
if [[ "${smoke_status}" -ne 10 ]]; then
  exit "${smoke_status}"
fi

smoke_status=0
run_smoke "${SMOKE_B8_ID}" 8 2 29652 || smoke_status=$?
if [[ "${smoke_status}" -eq 0 ]]; then
  launch_formal "${FORMAL_B8_ID}" 8 2 29654
fi
exit "${smoke_status}"
