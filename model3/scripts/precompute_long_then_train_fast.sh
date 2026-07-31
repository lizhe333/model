#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKEND_ROOT="${PROJECT_ROOT}/model3/third_party/light_wam"
DATASET_DIR="/data/public/datasets/FastWAM/libero_mujoco3.3.2/libero_10_no_noops_lerobot"
CACHE_DIR="${PROJECT_ROOT}/runs/I-003/model3/cache/latent/Wan2.1-T2V-1.3B/libero_10_2cam224"
RUN_ID="${MODEL3_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
EVIDENCE_DIR="${PROJECT_ROOT}/runs/I-003/model3/${RUN_ID}"
EXPECTED_SAMPLES=104280
PIPELINE_START_SECONDS=${SECONDS}

format_duration() {
  local total_seconds="$1"
  printf '%02d:%02d:%02d' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

log_stage() {
  local stage="$1"
  shift
  printf '[pipeline] timestamp=%s elapsed=%s stage=%s' \
    "$(date --iso-8601=seconds)" \
    "$(format_duration "$((SECONDS - PIPELINE_START_SECONDS))")" \
    "${stage}"
  if (( $# > 0 )); then
    printf ' %s' "$*"
  fi
  printf '\n'
}

export PYTHONPATH="${BACKEND_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_ROOT}/Light-WAM/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PIP_CONFIG_FILE=/dev/null
export HF_DATASETS_CACHE="${PROJECT_ROOT}/runs/I-003/model3/cache/hf_datasets"

mkdir -p "$(dirname "${CACHE_DIR}")"
cd "${BACKEND_ROOT}"

validate_cache() {
  test -s "${CACHE_DIR}/index.pt" || return 1
  test -s "${CACHE_DIR}/meta.json" || return 1
  /data/miniconda3/envs/lightwam-libero-eval/bin/python -c \
    "import torch; p=torch.load('${CACHE_DIR}/index.pt', map_location='cpu', weights_only=False); assert p['storage_format']=='sharded_v1'; assert int(p['num_samples'])==${EXPECTED_SAMPLES}; assert int(p['sample_to_shard'].numel())==${EXPECTED_SAMPLES}; assert int(p['sample_to_offset'].numel())==${EXPECTED_SAMPLES}; print('[pipeline] cache_validation=pass samples=%d shards=%d' % (p['num_samples'], len(p['shard_paths'])))"
}

if validate_cache; then
  log_stage precompute_skipped_valid_cache \
    "cache_dir=${CACHE_DIR} expected_samples=${EXPECTED_SAMPLES}"
else
  log_stage precompute_long_latents "cache_dir=${CACHE_DIR} expected_samples=${EXPECTED_SAMPLES}"
  PRECOMPUTE_START_SECONDS=${SECONDS}
  CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n lightwam-libero-eval \
    torchrun --standalone --nproc_per_node=4 \
    scripts/precompute_video_latents.py \
    task=libero_uncond_2cam224_1e-4 \
    overwrite=false \
    model.video_backbone_type=wan2_1_t2v \
    model.video_backbone_name=Wan-AI/Wan2.1-T2V-1.3B \
    precompute_storage_format=sharded_v1 \
    precompute_video_only=true \
    precompute_shard_size=1024 \
    precompute_batch_size=32 \
    precompute_num_workers=8 \
    precompute_cache_dtype=model \
    precompute_resume=true \
    precompute_timing.enabled=true \
    precompute_timing.sync_cuda=false \
    precompute_timing.log_every=10 \
    "data.train.dataset_dirs=['${DATASET_DIR}']" \
    "data.train.latent_cache_dir=${CACHE_DIR}"
  validate_cache
  log_stage precompute_complete \
    "precompute_duration=$(format_duration "$((SECONDS - PRECOMPUTE_START_SECONDS))")"
fi

cd "${PROJECT_ROOT}"
if [[ -e "${EVIDENCE_DIR}" ]]; then
  log_stage duplicate_run_id_refused "evidence_dir=${EVIDENCE_DIR}"
  exit 2
fi

TRAIN_START_SECONDS=${SECONDS}
log_stage train_long_cached_fast "run_id=${RUN_ID}"
if env MODEL3_RUN_ID="${RUN_ID}" "${PROJECT_ROOT}/model3/scripts/train_long_fast.sh"; then
  TRAIN_STATUS=0
else
  TRAIN_STATUS=$?
fi
log_stage training_finished \
  "run_id=${RUN_ID} status=${TRAIN_STATUS} train_duration=$(format_duration "$((SECONDS - TRAIN_START_SECONDS))")"
exit "${TRAIN_STATUS}"
