#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKEND_ROOT="${PROJECT_ROOT}/model3/third_party/light_wam"
HYDRA_CONFIG_ROOT="${PROJECT_ROOT}/side_model3_adapter_v2/configs/hydra"
TRAIN_ENTRYPOINT="${PROJECT_ROOT}/side_model3_adapter_v2/backend_train.py"

cd "${BACKEND_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_ROOT}/Light-WAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/runs/I-003/side_model3_adapter_v2/cache/hf_datasets}"
export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29651}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR is required}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:?TEXT_EMBED_CACHE_DIR is required}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:?LATENT_CACHE_DIR is required}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_STEPS="${MAX_STEPS:-40000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-25}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WANDB_NAME="${WANDB_NAME:-side_model3_adapter_v2_object}"
WANDB_MODE="${WANDB_MODE:-offline}"
V1_WARMSTART_PATH="${V1_WARMSTART_PATH:-null}"

echo "[side-model3-adapter] gpus=${GPU_IDS} processes=${NUM_PROCESSES} batch=${BATCH_SIZE} grad_acc=${GRAD_ACC}"
echo "[side-model3-adapter] max_steps=${MAX_STEPS} output_dir=${OUTPUT_DIR}"
echo "[side-model3-adapter] latent_cache=${LATENT_CACHE_DIR} dataset=${DATASET_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
  --config_file "${BACKEND_ROOT}/scripts/accelerate_configs/accelerate_zero1_ds.yaml" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${TRAIN_ENTRYPOINT}" \
  "hydra.searchpath=[file://${HYDRA_CONFIG_ROOT}]" \
  "task=libero_uncond_2cam224_1e-4" \
  "model=side_model3_adapter_v2" \
  "output_dir=${OUTPUT_DIR}" \
  "resume=null" \
  "wandb.enabled=true" \
  "wandb.project=i003-side-model3-adapter" \
  "wandb.name=${WANDB_NAME}" \
  "wandb.mode=${WANDB_MODE}" \
  "batch_size=${BATCH_SIZE}" \
  "gradient_accumulation_steps=${GRAD_ACC}" \
  "num_workers=${NUM_WORKERS}" \
  "eval_every=0" \
  "learning_rate=${LEARNING_RATE}" \
  "lr_scheduler_type=${LR_SCHEDULER_TYPE}" \
  "mixed_precision=${MIXED_PRECISION}" \
  "seed=${SEED}" \
  "max_grad_norm=${MAX_GRAD_NORM}" \
  "weight_decay=${WEIGHT_DECAY}" \
  "max_steps=${MAX_STEPS}" \
  "save_every=${SAVE_EVERY}" \
  "warmup_steps=${WARMUP_STEPS}" \
  "num_epochs=${NUM_EPOCHS}" \
  "parameter_report.enabled=true" \
  "timing_breakdown.enabled=false" \
  "train_visualization.enabled=false" \
  "model.load_text_encoder=false" \
  "model.model3_action_dit_warmstart_path=null" \
  "model.side_model3_adapter_v1_warmstart_path=${V1_WARMSTART_PATH}" \
  "model.mot_checkpoint_mixed_attn=false" \
  "model.video_latent_spatial_downsample_factor=1" \
  "model.apply_video_latent_downsample_to_action_branch=false" \
  "data.train.dataset_dirs=['${DATASET_DIR}']" \
  "data.train._target_=side_model3_adapter_v2.data.SideModel3AdapterV2CachedRobotVideoDataset" \
  "data.train.use_latent_cache=true" \
  "data.train.latent_cache_dir='${LATENT_CACHE_DIR}'" \
  "data.train.text_embedding_cache_dir='${TEXT_EMBED_CACHE_DIR}'" \
  "data.train.num_frames=33" \
  "data.train.global_sample_stride=1" \
  "data.train.action_video_freq_ratio=4" \
  "data.train.video_size=[224,448]" \
  "data.train.concat_multi_camera=horizontal" \
  "data.train.processor.num_output_cameras=2" \
  "$@"
