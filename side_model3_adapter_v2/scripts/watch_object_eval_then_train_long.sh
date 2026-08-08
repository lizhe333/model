#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_SESSION="${EVAL_SESSION:-side_adapter_object_eval_2x4}"
EVAL_RUN="${EVAL_RUN:-${PROJECT_ROOT}/runs/I-003/side_model3_adapter_v2/2026-08-07_object_15k_20k_30k_40k_parallel2x4_waves_eval500}"
QUEUE_ID="${QUEUE_ID:-2026-08-07_object_eval_then_side_model3_adapter_v2_long}"
QUEUE_DIR="${PROJECT_ROOT}/runs/I-003/side_model3_adapter_v2/${QUEUE_ID}"
QUEUE_LOG="${QUEUE_DIR}/monitor.log"
STATUS_FILE="${QUEUE_DIR}/status.txt"
RUN_ID="${RUN_ID:-2026-08-07_side_model3_adapter_v2_long_cached_b16_ga1_90k}"
OUTPUT_DIR="${PROJECT_ROOT}/runs/I-003/side_model3_adapter_v2/backend_runs/${RUN_ID}"
CACHE_DIR="${PROJECT_ROOT}/Light-WAM/data/latent_cache_Wan2.1-T2V-1.3B/libero_10_2cam224_side_model3_independent_v1"
DATASET_DIR="${PROJECT_ROOT}/Light-WAM/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot"

mkdir -p "${QUEUE_DIR}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

set_status() {
  printf '%s\n' "$1" > "${STATUS_FILE}"
}

gpus_0_to_3_are_free() {
  local free_values
  free_values="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F, '$1 + 0 >= 0 && $1 + 0 <= 3 {gsub(/ /, "", $2); print $2 + 0}')"
  [[ "$(wc -l <<< "${free_values}")" -eq 4 ]] || return 1
  while read -r free_mib; do
    [[ "${free_mib}" -ge 45000 ]] || return 1
  done <<< "${free_values}"
}

main() {
  cd "${PROJECT_ROOT}"
  if [[ -e "${OUTPUT_DIR}" ]]; then
    log "refusing_existing_output output=${OUTPUT_DIR}"
    set_status "failed_existing_output"
    exit 1
  fi
  if [[ ! -d "${DATASET_DIR}" ]]; then
    log "missing_long_dataset dataset=${DATASET_DIR}"
    set_status "failed_missing_dataset"
    exit 1
  fi
  env PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n lightwam-libero-eval \
    python -c \
      "from side_model3_adapter_v2.data import validate_complete_side_observation_cache; validate_complete_side_observation_cache('${CACHE_DIR}', expected_samples=104280)"

  set_status "waiting_for_object_evaluation"
  log "waiting_for_evaluation session=${EVAL_SESSION}"
  while tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; do
    sleep 60
  done

  if ! rg -q 'status=summary_complete_terminal_review_pending' "${EVAL_RUN}/evaluation_status.txt"; then
    log "evaluation_did_not_complete status_file=${EVAL_RUN}/evaluation_status.txt"
    set_status "failed_evaluation_incomplete"
    exit 1
  fi

  set_status "waiting_for_gpus_0_3"
  log "waiting_for_gpus_0_3_to_reach_45000_mib_free_each"
  while ! gpus_0_to_3_are_free; do
    sleep 15
  done

  set_status "launching_long_training"
  log "long_training_launch run=${RUN_ID} batch=16 grad_acc=1 max_steps=90000"
  exec env \
    RUN_ID="${RUN_ID}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    GPU_IDS="0,1,2,3" \
    NUM_PROCESSES="4" \
    MAIN_PROCESS_PORT="29661" \
    BATCH_SIZE="16" \
    GRAD_ACC="1" \
    NUM_WORKERS="16" \
    MAX_STEPS="90000" \
    SAVE_EVERY="5000" \
    WARMUP_STEPS="1000" \
    WANDB_MODE="offline" \
    bash "${PROJECT_ROOT}/side_model3_adapter_v2/scripts/train_long.sh"
}

main "$@"
