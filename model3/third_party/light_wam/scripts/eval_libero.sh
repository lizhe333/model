#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

EXTRA_ARGS=("$@")

TASK_NAME="${TASK_NAME:-libero_uncond_2cam224_1e-4}"
CKPT="${CKPT:-}"
GPU_ID="${GPU_ID:-0}"
# NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_TRIALS="${NUM_TRIALS:-5}"
USE_TRAINING_RUN_CONFIG="${USE_TRAINING_RUN_CONFIG:-true}"
TASK_SUITES_RAW="${TASK_SUITES:-libero_10,libero_goal,libero_spatial,libero_object}"
TASK_ORDERS_RAW="${TASK_ORDERS:-0,1,2,3,4,5,6,7,8,9}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/third_party/LIBERO}"

strip_brackets() {
  local value="$1"
  value="${value#[}"
  value="${value%]}"
  echo "$value"
}

trim_spaces() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  echo "$value"
}

if [[ -z "${CKPT}" ]]; then
  echo "Error: CKPT is required." >&2
  echo "Example: CKPT=/path/to/checkpoints/weights/step_072000.pt bash scripts/eval_libero.sh" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Error: checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${LIBERO_ROOT}" ]]; then
  echo "Error: LIBERO_ROOT not found: ${LIBERO_ROOT}" >&2
  exit 1
fi

TASK_SUITES_RAW="$(strip_brackets "${TASK_SUITES_RAW}")"
TASK_ORDERS_RAW="$(strip_brackets "${TASK_ORDERS_RAW}")"
IFS=',' read -r -a SUITE_ARRAY <<< "${TASK_SUITES_RAW}"
IFS=',' read -r -a TASK_ARRAY <<< "${TASK_ORDERS_RAW}"

CKPT_ABS="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")"
RUN_DIR="$(cd "$(dirname "${CKPT_ABS}")/../.." && pwd)"
RUN_TAG="$(basename "${RUN_DIR}")"
CKPT_FILE="$(basename "${CKPT_ABS}")"
CKPT_TAG="${CKPT_FILE%.*}"
OUTPUT_DIR_BASE="${OUTPUT_DIR:-${RUN_DIR}/eval_libero_single_${RUN_TAG}}"
if [[ "${OUTPUT_DIR_BASE}" == *"${CKPT_TAG}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR_BASE}"
else
  OUTPUT_DIR="${OUTPUT_DIR_BASE}_${CKPT_TAG}"
fi
LOG_DIR="${OUTPUT_DIR}/task_logs"
FAILED_TASKS_FILE="${OUTPUT_DIR}/failed_tasks.txt"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
: > "${FAILED_TASKS_FILE}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"

if ! "${PYTHON_BIN}" -c "import libero"; then
  echo "Error: python cannot import libero." >&2
  exit 1
fi

total_tasks=0
completed_tasks=0
failed_tasks=0

for suite_raw in "${SUITE_ARRAY[@]}"; do
  suite="$(trim_spaces "${suite_raw}")"
  [[ -z "${suite}" ]] && continue
  for task_raw in "${TASK_ARRAY[@]}"; do
    task_id="$(trim_spaces "${task_raw}")"
    [[ -z "${task_id}" ]] && continue
    total_tasks=$((total_tasks + 1))
    log_file="${LOG_DIR}/${suite}_task${task_id}.log"
    echo "[eval] suite=${suite} task_id=${task_id} log=${log_file}"
    if "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
      "task=${TASK_NAME}" \
      "ckpt=${CKPT_ABS}" \
      "gpu_id=${GPU_ID}" \
      "EVALUATION.task_suite_name=${suite}" \
      "EVALUATION.task_id=${task_id}" \
      "EVALUATION.output_dir=${OUTPUT_DIR}" \
      "EVALUATION.num_trials=${NUM_TRIALS}" \
      "EVALUATION.use_training_run_config=${USE_TRAINING_RUN_CONFIG}" \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "${log_file}"; then
      completed_tasks=$((completed_tasks + 1))
    else
      failed_tasks=$((failed_tasks + 1))
      printf '%s,%s,%s,%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${suite}" "${task_id}" "${log_file}" >> "${FAILED_TASKS_FILE}"
    fi
  done
done

if compgen -G "${OUTPUT_DIR}/*/gpu*_task*_results.json" > /dev/null; then
  "${PYTHON_BIN}" experiments/libero/summarize_results.py --output_dir="${OUTPUT_DIR}"
fi

echo "[summary] completed=${completed_tasks} failed=${failed_tasks} total=${total_tasks}"
echo "[summary] output_dir=${OUTPUT_DIR}"
if [[ "${failed_tasks}" -gt 0 ]]; then
  echo "[summary] failed task list: ${FAILED_TASKS_FILE}"
  exit 1
fi
