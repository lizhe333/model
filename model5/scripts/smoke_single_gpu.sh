#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPU_ID="${MODEL5_SMOKE_GPU_ID:?Set MODEL5_SMOKE_GPU_ID to one confirmed-idle GPU in 4,5,6,7}"
case "${GPU_ID}" in
  4|5|6|7) ;;
  *)
    echo "MODEL5_SMOKE_GPU_ID must be one of 4,5,6,7; got ${GPU_ID}" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_model5_single_gpu_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/${RUN_ID}}"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_ROOT}/Light-WAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/runs/I-003/model5/cache/hf_datasets}"
export PIP_CONFIG_FILE=/dev/null
export FASTWAM_STRICT_DATASET_ERRORS=true

export GPU_IDS="${GPU_ID}"
export NUM_PROCESSES=1
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29585}"
export RUN_ID
export RUN_TAG=model5_single_gpu_smoke
export OUTPUT_DIR
export MODEL5_LOG_DIR="${OUTPUT_DIR}/logs"
export MODEL5_LOG_FILE="${MODEL5_LOG_DIR}/smoke.log"

export DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/Light-WAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot}"
export LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-${PROJECT_ROOT}/Light-WAM/data/latent_cache_Wan2.1-T2V-1.3B/libero_spatial_2cam224}"
export TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PROJECT_ROOT}/Light-WAM/data/text_embeds_cache/libero_wan21t2b}"

export BATCH_SIZE=1
export GRAD_ACC=1
export NUM_WORKERS=0
export EVAL_EVERY=0
export MAX_STEPS=1
export SAVE_EVERY=0
export WARMUP_STEPS=0
export NUM_EPOCHS=1
export WANDB_ENABLED=false
export WANDB_MODE=offline
export PARAMETER_REPORT_ENABLED=false
export TIMING_BREAKDOWN_ENABLED=true
export TRAIN_VISUALIZATION_ENABLED=false

export ACTION_FEATURE_TEMPORAL_SCOPE=current_plus_noisy_future
export FIXED_FEATURE_TIMESTEP=1000
export FUTURE_FEATURE_LATENT_SLOTS="${FUTURE_FEATURE_LATENT_SLOTS:-1}"
export ACTION_FEATURE_SPATIAL_DOWNSAMPLE_FACTOR=1

exec conda run --no-capture-output -n lightwam-libero-eval \
  bash "${PROJECT_ROOT}/model5/scripts/train_backend.sh" \
  "model=model5_tri_timestep_query_flow" \
  "benchmark.enabled=true" \
  "benchmark.warmup_steps=0" \
  "benchmark.measure_steps=1" \
  "benchmark.output_filename=model5_smoke_benchmark.json" \
  "benchmark.description=model5_real_weight_real_data_single_gpu_smoke" \
  "smoke_validation.enabled=true" \
  "smoke_validation.seed=123" \
  "smoke_validation.action_inference_steps=2" \
  "log_every=1" \
  "train_visualization.enabled=false"
