#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_side_model3_adapter_v2_long}"

export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29661}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/I-003/side_model3_adapter_v2/backend_runs/${RUN_ID}}"
export DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/Light-WAM/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot}"
export TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PROJECT_ROOT}/Light-WAM/data/text_embeds_cache/libero_wan21t2b}"
export LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-${PROJECT_ROOT}/Light-WAM/data/latent_cache_Wan2.1-T2V-1.3B/libero_10_2cam224_side_model3_independent_v1}"
export WANDB_NAME="${WANDB_NAME:-side_model3_adapter_v2_long_${RUN_ID}}"

exec conda run --no-capture-output -n lightwam-libero-eval \
  bash "${PROJECT_ROOT}/side_model3_adapter_v2/scripts/train_backend.sh" "$@"
