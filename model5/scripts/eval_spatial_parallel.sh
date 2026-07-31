#!/usr/bin/env bash
set -euo pipefail

ROOT=${MODEL5_ROOT:-/home/lizhe/data/I-003_fast_video_dit_to_wam}
VENDOR=${MODEL5_VENDOR:-${ROOT}/model5/third_party/light_wam}
LIBERO_ROOT=${MODEL5_LIBERO_ROOT:-/data/users/lizhe/LIBERO}
TRAIN_RUN=${MODEL5_TRAIN_RUN:?MODEL5_TRAIN_RUN must point to a Model5 backend training run}
CKPT=${MODEL5_CKPT:-${TRAIN_RUN}/checkpoints/weights/step_035000.pt}
RUN_ROOT=${MODEL5_EVAL_RUN_ROOT:?MODEL5_EVAL_RUN_ROOT must point to a model5 evaluation evidence directory}
NUM_TRIALS=${MODEL5_EVAL_NUM_TRIALS:-50}
OUTPUT_DIR=${RUN_ROOT}/eval_output
LOG_DIR=${RUN_ROOT}/logs/task_logs
MASTER_LOG=${MODEL5_EVAL_LOG_FILE:-${RUN_ROOT}/logs/evaluation.log}
FAILED_TASKS=${RUN_ROOT}/failed_tasks.txt
PYTHON_BIN=${MODEL5_PYTHON_BIN:-/data/miniconda3/envs/lightwam-libero-eval/bin/python}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RUN_ROOT}/cache/mpl"
: > "${FAILED_TASKS}"
exec > >(tee -a "${MASTER_LOG}") 2>&1
echo "[$(date --iso-8601=seconds)] [model5] [eval] log_file=${MASTER_LOG}"

export PIP_CONFIG_FILE=/dev/null
export DIFFSYNTH_MODEL_BASE_PATH=${MODEL5_DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/Light-WAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HOME=${MODEL5_HF_HOME:-/data/cache/huggingface}
export LIBERO_ROOT
export MPLCONFIGDIR=${RUN_ROOT}/cache/mpl
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=${ROOT}:${VENDOR}/src:${VENDOR}:${LIBERO_ROOT}
export CKPT
export CONFIG=${TRAIN_RUN}/config.yaml
unset PYTHONHOME VIRTUAL_ENV

for required_path in \
  "${PYTHON_BIN}" \
  "${CKPT}" \
  "${TRAIN_RUN}/config.yaml" \
  "${TRAIN_RUN}/dataset_stats.json" \
  "${VENDOR}/experiments/libero/eval_libero_single.py"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[preflight] missing required path: ${required_path}" >&2
    exit 2
  fi
done

preflight_cache=${RUN_ROOT}/cache/numba_preflight
mkdir -p "${preflight_cache}"
NUMBA_CACHE_DIR=${preflight_cache} "${PYTHON_BIN}" - "${CKPT}" <<'PY'
import importlib
import pathlib
import sys

import hydra
import libero
import pandas
import robosuite
import torch

evaluator = importlib.import_module("experiments.libero.eval_libero_single")
summarizer = importlib.import_module("experiments.libero.summarize_results")

module_paths = {
    "evaluator": pathlib.Path(evaluator.__file__).resolve(),
    "hydra": pathlib.Path(hydra.__file__).resolve(),
    "libero": pathlib.Path(libero.__file__).resolve(),
    "pandas": pathlib.Path(pandas.__file__).resolve(),
    "robosuite": pathlib.Path(robosuite.__file__).resolve(),
    "summarizer": pathlib.Path(summarizer.__file__).resolve(),
}
contaminated = {name: path for name, path in module_paths.items() if "ImageWAM" in str(path)}
if contaminated:
    raise RuntimeError(f"ImageWAM module contamination detected: {contaminated}")

checkpoint_path = pathlib.Path(sys.argv[1]).resolve()
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
expected_method = "model5_asymmetric_tri_timestep_query_flow_v1"
if payload.get("method_id") != expected_method or payload.get("model_class") != "Model5WAM":
    raise RuntimeError(
        "Checkpoint identity mismatch: "
        f"method_id={payload.get('method_id')!r}, model_class={payload.get('model_class')!r}"
    )

print(f"[preflight] python={pathlib.Path(sys.executable).resolve()}")
for name, path in module_paths.items():
    print(f"[preflight] {name}={path}")
print(
    f"[preflight] checkpoint={checkpoint_path} "
    f"method_id={payload['method_id']} model_class={payload['model_class']} step={payload.get('step')}"
)
PY

if [[ ${MODEL5_PREFLIGHT_ONLY:-0} == 1 ]]; then
  exit 0
fi

cd "${VENDOR}"

pids=()
stop_workers() {
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap stop_workers INT TERM EXIT

run_gpu_tasks() {
  local gpu=$1
  shift
  local task_id log_file
  local worker_status=0
  local numba_cache=${RUN_ROOT}/cache/numba_gpu${gpu}
  mkdir -p "${numba_cache}"

  for task_id in "$@"; do
    log_file=${LOG_DIR}/libero_spatial_task${task_id}.log
    echo "[eval] gpu=${gpu} suite=libero_spatial task=${task_id} trials=${NUM_TRIALS} log=${log_file}"
    if ! CUDA_VISIBLE_DEVICES=${gpu} NUMBA_CACHE_DIR=${numba_cache} "${PYTHON_BIN}" \
      experiments/libero/eval_libero_single.py \
      task=libero_uncond_2cam224_1e-4 \
      ckpt="${CKPT}" \
      gpu_id=${gpu} \
      EVALUATION.task_suite_name=libero_spatial \
      EVALUATION.task_id=${task_id} \
      EVALUATION.output_dir="${OUTPUT_DIR}" \
      EVALUATION.num_trials=${NUM_TRIALS} \
      EVALUATION.use_training_run_config=true \
      EVALUATION.training_config_path="${TRAIN_RUN}/config.yaml" \
      EVALUATION.dataset_stats_path="${TRAIN_RUN}/dataset_stats.json" \
      EVALUATION.action_horizon=8 \
      EVALUATION.replan_steps=8 \
      EVALUATION.num_inference_steps=10 \
      EVALUATION.visualize_future_video=false \
      EVALUATION.print_action_latency=true \
      2>&1 | tee -a "${log_file}"; then
      printf '%s,gpu%s,libero_spatial,%s,%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${gpu}" "${task_id}" "${log_file}" \
        >> "${FAILED_TASKS}"
      worker_status=1
    fi
  done
  return "${worker_status}"
}

run_gpu_tasks 0 0 4 8 & pid0=$!; pids+=("${pid0}")
run_gpu_tasks 1 1 5 9 & pid1=$!; pids+=("${pid1}")
run_gpu_tasks 2 2 6 & pid2=$!; pids+=("${pid2}")
run_gpu_tasks 3 3 7 & pid3=$!; pids+=("${pid3}")

status=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
pids=()
trap - INT TERM EXIT

"${PYTHON_BIN}" experiments/libero/summarize_results.py --output_dir="${OUTPUT_DIR}"

if [[ -s "${FAILED_TASKS}" ]]; then
  status=1
fi

echo "[summary] output_dir=${OUTPUT_DIR}"
echo "[summary] failed_tasks=${FAILED_TASKS}"
exit "${status}"
