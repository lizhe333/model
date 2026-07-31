#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKEND_ROOT="${PROJECT_ROOT}/model5/third_party/light_wam"
HYDRA_CONFIG_ROOT="${HYDRA_CONFIG_ROOT:-${PROJECT_ROOT}/model5/configs/hydra}"
MODEL_PACKAGE_LABEL="${MODEL_PACKAGE_LABEL:-model5}"
cd "${BACKEND_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

TASK_NAME="${TASK_NAME:-libero_uncond_2cam224_1e-4}"
SUITE_NAME="${SUITE_NAME:-libero}"
EXTRA_ARGS=("$@")

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29577}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_$$}"
RUN_TAG="${RUN_TAG:-${SUITE_NAME}_lightwam}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/I-003/model5/backend_runs/${RUN_ID}}"
LOG_DIR="${MODEL5_LOG_DIR:-${OUTPUT_DIR}/logs}"
LOG_FILE="${MODEL5_LOG_FILE:-${LOG_DIR}/training.log}"
WANDB_PROJECT="${WANDB_PROJECT:-light-wam}"
WANDB_NAME="${WANDB_NAME:-${RUN_TAG}_${RUN_ID}}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
RESUME="${RESUME:-null}"

DATASET_DIR="${DATASET_DIR:-}"
DATASET_DIRS="${DATASET_DIRS:-}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PROJECT_ROOT}/Light-WAM/data/text_embeds_cache/libero}"

NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-4}"
ACTION_FEATURE_TEMPORAL_SCOPE="${ACTION_FEATURE_TEMPORAL_SCOPE:-current_plus_noisy_future}"
FIXED_FEATURE_TIMESTEP="${FIXED_FEATURE_TIMESTEP:-1000}"
FUTURE_FEATURE_LATENT_SLOTS="${FUTURE_FEATURE_LATENT_SLOTS:-1}"
ACTION_FEATURE_SPATIAL_DOWNSAMPLE_FACTOR="${ACTION_FEATURE_SPATIAL_DOWNSAMPLE_FACTOR:-1}"
VIDEO_SIZE="${VIDEO_SIZE:-[224,448]}"
CONCAT_MULTI_CAMERA="${CONCAT_MULTI_CAMERA:-horizontal}"
NUM_OUTPUT_CAMERAS="${NUM_OUTPUT_CAMERAS:-2}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EVAL_EVERY="${EVAL_EVERY:-0}"
MAX_STEPS="${MAX_STEPS:-150000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-25}"

USE_WAM_ADAPTER="${USE_WAM_ADAPTER:-true}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
REMOVE_ORIGINAL_ACTION_EXPERT="${REMOVE_ORIGINAL_ACTION_EXPERT:-true}"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-[8,16,24]}"
ADAPTER_DIM="${ADAPTER_DIM:-256}"
ADAPTER_SCALE="${ADAPTER_SCALE:-1.0}"

USE_BACKBONE_LORA="${USE_BACKBONE_LORA:-true}"
LORA_LAYER_INDICES="${LORA_LAYER_INDICES:-[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-['self_attn.q','self_attn.k','self_attn.v','self_attn.o','cross_attn.q','cross_attn.k','cross_attn.v','cross_attn.o','ffn.0','ffn.2']}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

TEMPORAL_WEIGHTING_ENABLED="${TEMPORAL_WEIGHTING_ENABLED:-true}"
TEMPORAL_PREFIX_STEPS="${TEMPORAL_PREFIX_STEPS:-8}"
TEMPORAL_PREFIX_WEIGHT="${TEMPORAL_PREFIX_WEIGHT:-1.0}"
TEMPORAL_TAIL_WEIGHT="${TEMPORAL_TAIL_WEIGHT:-0.0}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
PARAMETER_REPORT_ENABLED="${PARAMETER_REPORT_ENABLED:-true}"
MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-True}"
TIMING_BREAKDOWN_ENABLED="${TIMING_BREAKDOWN_ENABLED:-true}"
TRAIN_VISUALIZATION_ENABLED="${TRAIN_VISUALIZATION_ENABLED:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[$(date --iso-8601=seconds)] [model5] [launch] log_file=${LOG_FILE}"

DATASET_OVERRIDE_ARGS=()
if [[ -n "${DATASET_DIRS}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.dataset_dirs=${DATASET_DIRS}")
elif [[ -n "${DATASET_DIR}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.dataset_dirs=['${DATASET_DIR}']")
fi
if [[ -n "${LATENT_CACHE_DIR}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.latent_cache_dir=${LATENT_CACHE_DIR}")
fi
if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
  DATASET_OVERRIDE_ARGS+=("data.train.text_embedding_cache_dir='${TEXT_EMBED_CACHE_DIR}'")
fi

echo "[launch] task=${TASK_NAME}"
echo "[launch] mode=model5_on_vendored_lightwam gpus=${GPU_IDS} num_processes=${NUM_PROCESSES} port=${MAIN_PROCESS_PORT}"
echo "[launch] suite=${SUITE_NAME}"
echo "[launch] run_tag=${RUN_TAG}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] log_file=${LOG_FILE}"
echo "[launch] resume=${RESUME}"
echo "[launch] dataset_dir=${DATASET_DIR:-<config default>}"
echo "[launch] dataset_dirs=${DATASET_DIRS:-<config default>}"
echo "[launch] latent_cache_dir=${LATENT_CACHE_DIR:-<config default>}"
echo "[launch] text_embed_cache_dir=${TEXT_EMBED_CACHE_DIR}"
echo "[launch] wandb.project=${WANDB_PROJECT} wandb.name=${WANDB_NAME} wandb.mode=offline"
echo "[launch] batch_size=${BATCH_SIZE} grad_acc=${GRAD_ACC} num_workers=${NUM_WORKERS} eval_every=${EVAL_EVERY}"
echo "[launch] max_steps=${MAX_STEPS} save_every=${SAVE_EVERY} warmup_steps=${WARMUP_STEPS} num_epochs=${NUM_EPOCHS}"
echo "[launch] learning_rate=${LEARNING_RATE} lr_scheduler_type=${LR_SCHEDULER_TYPE}"
echo "[launch] mixed_precision=${MIXED_PRECISION} seed=${SEED} max_grad_norm=${MAX_GRAD_NORM} weight_decay=${WEIGHT_DECAY}"
echo "[launch] video_gradient_checkpointing=${MOT_CHECKPOINT_MIXED_ATTN}"
echo "[launch] model.loss.action_temporal_weighting.num_prefix_steps=${TEMPORAL_PREFIX_STEPS}"
echo "[launch] model_package=${PROJECT_ROOT}/${MODEL_PACKAGE_LABEL}"
echo "[launch] action_feature_scope=${ACTION_FEATURE_TEMPORAL_SCOPE} tau_f=${FIXED_FEATURE_TIMESTEP} future_slots=${FUTURE_FEATURE_LATENT_SLOTS} downsample=${ACTION_FEATURE_SPATIAL_DOWNSAMPLE_FACTOR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
  --config_file "${BACKEND_ROOT}/scripts/accelerate_configs/accelerate_zero1_ds.yaml" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${BACKEND_ROOT}/scripts/train.py" \
  "hydra.searchpath=[file://${HYDRA_CONFIG_ROOT}]" \
  "hydra/job_logging=stdout" \
  "task=${TASK_NAME}" \
  "output_dir=${OUTPUT_DIR}" \
  "resume=${RESUME}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${WANDB_NAME}" \
  "wandb.mode=offline" \
  "batch_size=${BATCH_SIZE}" \
  "gradient_accumulation_steps=${GRAD_ACC}" \
  "num_workers=${NUM_WORKERS}" \
  "eval_every=${EVAL_EVERY}" \
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
  "parameter_report.enabled=${PARAMETER_REPORT_ENABLED}" \
  "model.mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}" \
  "model.loss.use_first_frame_residual_video_target=false" \
  "model.loss.action_temporal_weighting.enabled=${TEMPORAL_WEIGHTING_ENABLED}" \
  "model.loss.action_temporal_weighting.num_prefix_steps=${TEMPORAL_PREFIX_STEPS}" \
  "model.loss.action_temporal_weighting.prefix_weight=${TEMPORAL_PREFIX_WEIGHT}" \
  "model.loss.action_temporal_weighting.tail_weight=${TEMPORAL_TAIL_WEIGHT}" \
  "model.wam_adapter.use_wam_adapter=${USE_WAM_ADAPTER}" \
  "model.wam_adapter.freeze_backbone=${FREEZE_BACKBONE}" \
  "model.wam_adapter.remove_original_action_expert=${REMOVE_ORIGINAL_ACTION_EXPERT}" \
  "model.wam_adapter.adapter_layer_indices=${ADAPTER_LAYERS}" \
  "model.wam_adapter.adapter_dim=${ADAPTER_DIM}" \
  "model.wam_adapter.adapter_scale=${ADAPTER_SCALE}" \
  "model.wam_adapter.use_backbone_lora=${USE_BACKBONE_LORA}" \
  "model.wam_adapter.lora_layer_indices=${LORA_LAYER_INDICES}" \
  "model.wam_adapter.lora_target_modules=${LORA_TARGET_MODULES}" \
  "model.wam_adapter.lora_rank=${LORA_RANK}" \
  "model.wam_adapter.lora_alpha=${LORA_ALPHA}" \
  "model.wam_adapter.lora_dropout=${LORA_DROPOUT}" \
  "model.video_backbone_type=wan2_1_t2v" \
  "model.video_backbone_name=Wan-AI/Wan2.1-T2V-1.3B" \
  "model.video_latent_spatial_downsample_factor=2" \
  "model.apply_video_latent_downsample_to_action_branch=false" \
  "model.action_feature_config.temporal_scope=${ACTION_FEATURE_TEMPORAL_SCOPE}" \
  "model.action_feature_config.fixed_future_timestep=${FIXED_FEATURE_TIMESTEP}" \
  "model.action_feature_config.num_future_latent_slots=${FUTURE_FEATURE_LATENT_SLOTS}" \
  "model.action_feature_config.spatial_downsample_factor=${ACTION_FEATURE_SPATIAL_DOWNSAMPLE_FACTOR}" \
  "timing_breakdown.enabled=${TIMING_BREAKDOWN_ENABLED}" \
  "timing_breakdown.sync_cuda=false" \
  "train_visualization.enabled=${TRAIN_VISUALIZATION_ENABLED}" \
  "train_visualization.every=500" \
  "train_visualization.fps=8" \
  "train_visualization.tiled=false" \
  "data.train.use_latent_cache=true" \
  "data.train.num_frames=${NUM_FRAMES}" \
  "data.train.action_video_freq_ratio=${ACTION_VIDEO_FREQ_RATIO}" \
  "data.train.video_size=${VIDEO_SIZE}" \
  "data.train.concat_multi_camera=${CONCAT_MULTI_CAMERA}" \
  "data.train.processor.num_output_cameras=${NUM_OUTPUT_CAMERAS}" \
  "${DATASET_OVERRIDE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
