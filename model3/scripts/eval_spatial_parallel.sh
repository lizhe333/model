#!/usr/bin/env bash
set -euo pipefail

ROOT=${MODEL3_ROOT:-/home/lizhe/data/I-003_fast_video_dit_to_wam}
VENDOR=${MODEL3_VENDOR:-${ROOT}/model3/third_party/light_wam}
LIBERO_ROOT=${MODEL3_LIBERO_ROOT:-/data/users/lizhe/LIBERO}
TRAIN_RUN=${MODEL3_TRAIN_RUN:-${ROOT}/runs/I-003/model3/backend_runs/2026-07-24_model3_spatial_long_b16_ga1_v2}
CKPT=${MODEL3_CKPT:-${TRAIN_RUN}/checkpoints/weights/step_060000.pt}
RUN_ROOT=${MODEL3_EVAL_RUN_ROOT:?MODEL3_EVAL_RUN_ROOT must point to a model3 evaluation evidence directory}
NUM_TRIALS=${MODEL3_EVAL_NUM_TRIALS:-50}
SUCCESS_VIDEO_STRIDE=10
OUTPUT_DIR=${RUN_ROOT}/eval_output
LOG_DIR=${RUN_ROOT}/logs/task_logs
FAILED_TASKS=${RUN_ROOT}/failed_tasks.txt
PYTHON_BIN=${MODEL3_PYTHON_BIN:-/data/miniconda3/envs/lightwam-libero-eval/bin/python}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RUN_ROOT}/cache/mpl"
: > "${FAILED_TASKS}"

export PIP_CONFIG_FILE=/dev/null
export DIFFSYNTH_MODEL_BASE_PATH=${MODEL3_DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/Light-WAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HOME=${MODEL3_HF_HOME:-/data/cache/huggingface}
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
  "${RUN_ROOT}/run_manifest.json" \
  "${VENDOR}/experiments/libero/eval_libero_single.py"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[preflight] missing required path: ${required_path}" >&2
    exit 2
  fi
done

preflight_cache=${RUN_ROOT}/cache/numba_preflight
mkdir -p "${preflight_cache}"
NUMBA_CACHE_DIR=${preflight_cache} "${PYTHON_BIN}" - \
  "${CKPT}" "${RUN_ROOT}/run_manifest.json" "${SUCCESS_VIDEO_STRIDE}" <<'PY'
import importlib
import json
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
manifest_path = pathlib.Path(sys.argv[2]).resolve()
success_video_stride = int(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
protocol = manifest.get("protocol")
if not isinstance(protocol, dict):
    raise RuntimeError("run_manifest.json is missing the protocol mapping")
if protocol.get("success_video_stride") != success_video_stride:
    raise RuntimeError(
        "run manifest success_video_stride mismatch: "
        f"expected {success_video_stride}, got {protocol.get('success_video_stride')!r}"
    )
if protocol.get("save_all_failure_videos") is not True:
    raise RuntimeError("run manifest must set save_all_failure_videos=true")
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
expected_method = "model3_vla_recurrent_query_flow_v1"
if payload.get("method_id") != expected_method or payload.get("model_class") != "Model3WAM":
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
print(
    f"[preflight] video_retention=success_stride_{success_video_stride} "
    "save_all_failures=true"
)
PY

if [[ ${MODEL3_PREFLIGHT_ONLY:-0} == 1 ]]; then
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
      EVALUATION.success_video_stride=${SUCCESS_VIDEO_STRIDE} \
      EVALUATION.save_all_failure_videos=true \
      EVALUATION.print_action_latency=true \
      > "${log_file}" 2>&1; then
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
