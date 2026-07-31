#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_NAME="${SUITE_NAME:-libero_10}" \
TASK_NAME="${TASK_NAME:-libero_uncond_2cam224_1e-4}" \
RUN_TAG="${RUN_TAG:-lightwam_libero_10}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29580}" \
DATASET_DIR="${DATASET_DIR:-./data/libero_mujoco3.3.2/libero_10_no_noops_lerobot}" \
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-./data/latent_cache_Wan2.1-T2V-1.3B/libero_10_2cam224}" \
LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
SAVE_EVERY="${SAVE_EVERY:-5000}" \
bash "${SCRIPT_DIR}/train_libero_core.sh" "$@"
