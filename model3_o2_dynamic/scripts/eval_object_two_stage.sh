#!/usr/bin/env bash
set -euo pipefail

ROOT=${MODEL3_O2_ROOT:-/home/lizhe/data/I-003_fast_video_dit_to_wam}
VENDOR=${ROOT}/model3/third_party/light_wam
LIBERO_ROOT=${MODEL3_O2_LIBERO_ROOT:-/data/users/lizhe/LIBERO}
TRAIN_RUN=${MODEL3_O2_TRAIN_RUN:-${ROOT}/runs/I-003/model3_o2/backend_runs/2026-07-30_model3_o2_object_warmstart20k_b16_ga1_150k}
RUN_ROOT=${MODEL3_O2_EVAL_RUN_ROOT:?MODEL3_O2_EVAL_RUN_ROOT is required}
PYTHON_BIN=${MODEL3_O2_PYTHON_BIN:-/data/miniconda3/envs/lightwam-libero-eval/bin/python}
NUM_TRIALS=50
SUCCESS_VIDEO_STRIDE=10
STEPS=(10000 20000 35000)

export PIP_CONFIG_FILE=/dev/null
export DIFFSYNTH_MODEL_BASE_PATH=${ROOT}/Light-WAM/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HOME=${MODEL3_O2_HF_HOME:-/data/cache/huggingface}
export LIBERO_ROOT
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${ROOT}:${VENDOR}/src:${VENDOR}:${LIBERO_ROOT}
unset PYTHONHOME VIRTUAL_ENV

checkpoint_for_step() {
  printf '%s/checkpoints/weights/step_%06d.pt\n' "${TRAIN_RUN}" "$1"
}

solver10_root_for_step() {
  printf '%s/solver10_step_%06d\n' "${RUN_ROOT}" "$1"
}

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/cache/numba_preflight" "${RUN_ROOT}/cache/mpl_preflight"

for required_path in \
  "${PYTHON_BIN}" \
  "${TRAIN_RUN}/config.yaml" \
  "${TRAIN_RUN}/dataset_stats.json" \
  "${VENDOR}/experiments/libero/eval_libero_single.py" \
  "${VENDOR}/experiments/libero/summarize_results.py" \
  "${ROOT}/model3_o2/scripts/finalize_object_eval.py"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[preflight] missing required path: ${required_path}" >&2
    exit 2
  fi
done
for step in "${STEPS[@]}"; do
  if [[ ! -f $(checkpoint_for_step "${step}") ]]; then
    echo "[preflight] missing checkpoint: $(checkpoint_for_step "${step}")" >&2
    exit 2
  fi
done

REPO_COMMIT=$(git -C "${ROOT}" rev-parse HEAD)
REPO_DIRTY=false
if [[ -n $(git -C "${ROOT}" status --short) ]]; then
  REPO_DIRTY=true
fi

NUMBA_CACHE_DIR=${RUN_ROOT}/cache/numba_preflight \
MPLCONFIGDIR=${RUN_ROOT}/cache/mpl_preflight \
"${PYTHON_BIN}" - \
  "${RUN_ROOT}" "${TRAIN_RUN}" "${REPO_COMMIT}" "${REPO_DIRTY}" <<'PY'
import datetime as dt
import hashlib
import importlib
import json
import pathlib
import sys

import torch
from libero.libero import benchmark

run_root = pathlib.Path(sys.argv[1]).resolve()
train_run = pathlib.Path(sys.argv[2]).resolve()
repo_commit = sys.argv[3]
repo_dirty = sys.argv[4].lower() == "true"
steps = (10000, 20000, 35000)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


evaluator = importlib.import_module("experiments.libero.eval_libero_single")
summarizer = importlib.import_module("experiments.libero.summarize_results")
if "libero_object" not in benchmark.get_benchmark_dict():
    raise RuntimeError("libero_object benchmark is unavailable")
for name, module in {"evaluator": evaluator, "summarizer": summarizer}.items():
    path = pathlib.Path(module.__file__).resolve()
    if "model3/third_party/light_wam" not in str(path):
        raise RuntimeError(f"unexpected {name} provenance: {path}")

started_at = dt.datetime.now(dt.timezone.utc).isoformat()
checkpoint_records = []
for step in steps:
    checkpoint = train_run / "checkpoints" / "weights" / f"step_{step:06d}.pt"
    checkpoint_sha = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("method_id") != "model3_o2_layer_aware_query_flow_v1":
        raise RuntimeError(f"step {step}: unexpected method_id {payload.get('method_id')!r}")
    if payload.get("model_class") != "Model3O2WAM":
        raise RuntimeError(f"step {step}: unexpected model_class {payload.get('model_class')!r}")
    if int(payload.get("step", -1)) != step:
        raise RuntimeError(f"step {step}: checkpoint payload step mismatch")
    checkpoint_records.append({"step": step, "path": str(checkpoint), "sha256": checkpoint_sha})

coordinator = {
    "schema_version": 1,
    "track_id": "model3_o2",
    "run_type": "object_checkpoint_selection_then_solver_diagnostic",
    "selection_label": "best_observed_on_predeclared_checkpoint_set",
    "selection_rule": "highest solver-10 successes over 500 episodes; earliest step breaks an exact tie",
    "selection_set_steps": list(steps),
    "stage_order": ["solver10_checkpoint_selection", "solver5_selected_checkpoint_diagnostic"],
    "checkpoints": checkpoint_records,
    "protocol": {
        "task_suite": "libero_object",
        "tasks": 10,
        "trials_per_task": 50,
        "episodes_per_condition": 500,
        "seed": 42,
        "action_horizon": 8,
        "replan_steps": 8,
        "max_episode_steps": 400,
        "success_video_stride": 10,
        "save_all_failure_videos": True,
    },
    "resource_layout": {
        "gpu_ids": [0, 1, 2, 3],
        "stage1_model_processes_per_gpu": 3,
        "stage2_model_processes_per_gpu": 1,
        "valid_for_latency_comparison": False,
    },
    "repo_commit": repo_commit,
    "repo_dirty": repo_dirty,
    "started_at": started_at,
    "completed_at": None,
    "status": "preflight_passed",
}
(run_root / "run_manifest.json").write_text(
    json.dumps(coordinator, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(run_root / "commands.txt").write_text(
    "MODEL3_O2_EVAL_RUN_ROOT=" + str(run_root) +
    " bash model3_o2/scripts/eval_object_two_stage.sh\n",
    encoding="utf-8",
)
(run_root / "environment.md").write_text(
    "# Environment\n\n"
    "- conda_env: `lightwam-libero-eval`\n"
    "- precision: `bf16`\n"
    "- simulator: `LIBERO` with EGL\n"
    "- evaluator: vendored Model3 Light-WAM infrastructure with O2 runtime from training config\n"
    "- GPUs: `0,1,2,3`; stage 1 uses three model workers per GPU\n"
    "- shared-GPU timing is invalid for latency claims\n",
    encoding="utf-8",
)
(run_root / "run_report.md").write_text(
    "# Model3 O2 Object Checkpoint And Solver Evaluation\n\n"
    "Status: preflight passed. Solver-10 checkpoint selection is pending.\n",
    encoding="utf-8",
)

for record in checkpoint_records:
    step = record["step"]
    condition = run_root / f"solver10_step_{step:06d}"
    (condition / "eval_output").mkdir(parents=True, exist_ok=True)
    (condition / "logs" / "task_logs").mkdir(parents=True, exist_ok=True)
    (condition / "cache").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "track_id": "model3_o2",
        "parent_track": "model3",
        "run_type": "checkpoint_selection_eval",
        "evidence_scope": "local_training_eval",
        "dataset": "LIBERO",
        "suites": ["libero_object"],
        "selection_label": "best_observed_on_predeclared_checkpoint_set",
        "selection_set_steps": list(steps),
        "checkpoint_path": record["path"],
        "checkpoint_sha256": record["sha256"],
        "checkpoint_step": step,
        "protocol": {
            "tasks": 10,
            "trials_per_task": 50,
            "total_episodes": 500,
            "gpu_ids": [0, 1, 2, 3],
            "seed": 42,
            "action_horizon": 8,
            "replan_steps": 8,
            "action_inference_steps": 10,
            "max_episode_steps": 400,
            "success_video_stride": 10,
            "save_all_failure_videos": True,
        },
        "started_at": started_at,
        "completed_at": None,
        "status": "preflight_passed",
    }
    (condition / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (condition / "failed_tasks.txt").write_text("", encoding="utf-8")
    (condition / "run_report.md").write_text(
        f"# O2 Object Step {step} Solver-10 Evaluation\n\nStatus: pending.\n",
        encoding="utf-8",
    )

print("[preflight] O2 checkpoints=10K,20K,35K solver10_then_selected_solver5")
PY

if [[ ${MODEL3_O2_PREFLIGHT_ONLY:-0} == 1 ]]; then
  exit 0
fi

for step in "${STEPS[@]}"; do
  condition=$(solver10_root_for_step "${step}")
  if find "${condition}/eval_output" -type f -name '*_results.json' -print -quit | grep -q .; then
    echo "[launch] existing results found in ${condition}; refusing to overwrite" >&2
    exit 3
  fi
  printf 'running start=%s\n' "$(date --iso-8601=seconds)" > "${condition}/evaluation_status.txt"
done

cd "${VENDOR}"
pids=()
stop_workers() {
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap stop_workers INT TERM EXIT

run_jobs() {
  local step=$1 gpu=$2
  shift 2
  local condition checkpoint task_id log_file status=0
  condition=$(solver10_root_for_step "${step}")
  checkpoint=$(checkpoint_for_step "${step}")
  mkdir -p "${condition}/cache/numba_gpu${gpu}" "${condition}/cache/mpl_gpu${gpu}"
  for task_id in "$@"; do
    log_file=${condition}/logs/task_logs/libero_object_task${task_id}.log
    echo "[eval] solver=10 step=${step} gpu=${gpu} task=${task_id} trials=${NUM_TRIALS}"
    if ! CUDA_VISIBLE_DEVICES=${gpu} \
      NUMBA_CACHE_DIR=${condition}/cache/numba_gpu${gpu} \
      MPLCONFIGDIR=${condition}/cache/mpl_gpu${gpu} \
      "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
        task=libero_uncond_2cam224_1e-4 \
        ckpt="${checkpoint}" \
        gpu_id=${gpu} \
        EVALUATION.task_suite_name=libero_object \
        EVALUATION.task_id=${task_id} \
        EVALUATION.output_dir="${condition}/eval_output" \
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
        EVALUATION.print_action_latency=false \
        > "${log_file}" 2>&1; then
      printf '%s,gpu%s,step%s,task%s,%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${gpu}" "${step}" "${task_id}" "${log_file}" \
        >> "${condition}/failed_tasks.txt"
      status=1
    fi
  done
  return "${status}"
}

for step in "${STEPS[@]}"; do
  run_jobs "${step}" 0 0 4 8 & pids+=("$!")
  run_jobs "${step}" 1 1 5 9 & pids+=("$!")
  run_jobs "${step}" 2 2 6 & pids+=("$!")
  run_jobs "${step}" 3 3 7 & pids+=("$!")
done

stage1_status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    stage1_status=1
  fi
done
pids=()
trap - INT TERM EXIT

for step in "${STEPS[@]}"; do
  condition=$(solver10_root_for_step "${step}")
  "${PYTHON_BIN}" experiments/libero/summarize_results.py --output_dir="${condition}/eval_output"
  if [[ -s "${condition}/failed_tasks.txt" ]]; then
    stage1_status=1
  fi
  printf 'finished exit_code=%s end=%s\n' "${stage1_status}" "$(date --iso-8601=seconds)" \
    > "${condition}/evaluation_status.txt"
done
if [[ ${stage1_status} -ne 0 ]]; then
  echo "[stage1] worker failure; solver5 will not start" >&2
  exit "${stage1_status}"
fi

cd "${ROOT}"
for step in "${STEPS[@]}"; do
  "${PYTHON_BIN}" -m model3_o2.scripts.finalize_object_eval \
    --run-root "$(solver10_root_for_step "${step}")"
done

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
steps = (10000, 20000, 35000)
results = []
for step in steps:
    validation = json.loads(
        (root / f"solver10_step_{step:06d}" / "result_validation.json").read_text(encoding="utf-8")
    )
    results.append({
        "step": step,
        "successes": int(validation["results"]["successes"]),
        "episodes": int(validation["results"]["valid_episodes"]),
        "success_rate": float(validation["results"]["success_rate"]),
        "checkpoint_sha256": validation["checkpoint"]["sha256"],
    })
selected = sorted(results, key=lambda item: (-item["successes"], item["step"]))[0]
payload = {
    "schema_version": 1,
    "status": "pass",
    "selection_label": "best_observed_on_predeclared_checkpoint_set",
    "selection_rule": "highest solver-10 successes; earliest step breaks an exact tie",
    "results": results,
    "selected": selected,
}
(root / "selection.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "selected_step.txt").write_text(f"{selected['step']}\n", encoding="utf-8")
print(f"[selection] selected_step={selected['step']} successes={selected['successes']}/500")
PY

SELECTED_STEP=$(<"${RUN_ROOT}/selected_step.txt")
SELECTED_CKPT=$(checkpoint_for_step "${SELECTED_STEP}")
SOLVER5_ROOT=${RUN_ROOT}/solver5_selected_step_$(printf '%06d' "${SELECTED_STEP}")
mkdir -p "${SOLVER5_ROOT}/eval_output" "${SOLVER5_ROOT}/logs/task_logs" "${SOLVER5_ROOT}/cache"

"${PYTHON_BIN}" - "${RUN_ROOT}" "${SOLVER5_ROOT}" "${SELECTED_STEP}" "${SELECTED_CKPT}" <<'PY'
import datetime as dt
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
condition = pathlib.Path(sys.argv[2]).resolve()
step = int(sys.argv[3])
checkpoint = pathlib.Path(sys.argv[4]).resolve()
selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
if int(selection["selected"]["step"]) != step:
    raise RuntimeError("selected step changed before solver5 launch")
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
checkpoint_sha = digest.hexdigest()
if checkpoint_sha != selection["selected"]["checkpoint_sha256"]:
    raise RuntimeError("selected checkpoint SHA mismatch")
manifest = {
    "schema_version": 1,
    "track_id": "model3_o2",
    "parent_track": "model3",
    "run_type": "selected_checkpoint_solver_diagnostic",
    "evidence_scope": "diagnostic_only",
    "formal_benchmark_eligible": False,
    "dataset": "LIBERO",
    "suites": ["libero_object"],
    "selection_source": str(root / "selection.json"),
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_sha,
    "checkpoint_step": step,
    "protocol": {
        "tasks": 10,
        "trials_per_task": 50,
        "total_episodes": 500,
        "gpu_ids": [0, 1, 2, 3],
        "seed": 42,
        "action_horizon": 8,
        "replan_steps": 8,
        "action_inference_steps": 5,
        "max_episode_steps": 400,
        "success_video_stride": 10,
        "save_all_failure_videos": True,
    },
    "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "completed_at": None,
    "status": "preflight_passed",
}
(condition / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(condition / "failed_tasks.txt").write_text("", encoding="utf-8")
(condition / "run_report.md").write_text(
    f"# O2 Object Step {step} Solver-5 Diagnostic\n\nStatus: pending.\n",
    encoding="utf-8",
)
PY

printf 'running start=%s\n' "$(date --iso-8601=seconds)" > "${SOLVER5_ROOT}/evaluation_status.txt"
cd "${VENDOR}"
pids=()
trap stop_workers INT TERM EXIT
run_solver5_jobs() {
  local gpu=$1
  shift
  local task_id log_file status=0
  mkdir -p "${SOLVER5_ROOT}/cache/numba_gpu${gpu}" "${SOLVER5_ROOT}/cache/mpl_gpu${gpu}"
  for task_id in "$@"; do
    log_file=${SOLVER5_ROOT}/logs/task_logs/libero_object_task${task_id}.log
    echo "[eval] solver=5 step=${SELECTED_STEP} gpu=${gpu} task=${task_id} trials=${NUM_TRIALS}"
    if ! CUDA_VISIBLE_DEVICES=${gpu} \
      NUMBA_CACHE_DIR=${SOLVER5_ROOT}/cache/numba_gpu${gpu} \
      MPLCONFIGDIR=${SOLVER5_ROOT}/cache/mpl_gpu${gpu} \
      "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
        task=libero_uncond_2cam224_1e-4 \
        ckpt="${SELECTED_CKPT}" \
        gpu_id=${gpu} \
        EVALUATION.task_suite_name=libero_object \
        EVALUATION.task_id=${task_id} \
        EVALUATION.output_dir="${SOLVER5_ROOT}/eval_output" \
        EVALUATION.num_trials=${NUM_TRIALS} \
        EVALUATION.use_training_run_config=true \
        EVALUATION.training_config_path="${TRAIN_RUN}/config.yaml" \
        EVALUATION.dataset_stats_path="${TRAIN_RUN}/dataset_stats.json" \
        EVALUATION.action_horizon=8 \
        EVALUATION.replan_steps=8 \
        EVALUATION.num_inference_steps=5 \
        EVALUATION.visualize_future_video=false \
        EVALUATION.success_video_stride=${SUCCESS_VIDEO_STRIDE} \
        EVALUATION.save_all_failure_videos=true \
        EVALUATION.print_action_latency=false \
        > "${log_file}" 2>&1; then
      printf '%s,gpu%s,step%s,task%s,%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${gpu}" "${SELECTED_STEP}" "${task_id}" "${log_file}" \
        >> "${SOLVER5_ROOT}/failed_tasks.txt"
      status=1
    fi
  done
  return "${status}"
}

run_solver5_jobs 0 0 4 8 & pids+=("$!")
run_solver5_jobs 1 1 5 9 & pids+=("$!")
run_solver5_jobs 2 2 6 & pids+=("$!")
run_solver5_jobs 3 3 7 & pids+=("$!")
stage2_status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    stage2_status=1
  fi
done
pids=()
trap - INT TERM EXIT

"${PYTHON_BIN}" experiments/libero/summarize_results.py --output_dir="${SOLVER5_ROOT}/eval_output"
if [[ -s "${SOLVER5_ROOT}/failed_tasks.txt" ]]; then
  stage2_status=1
fi
printf 'finished exit_code=%s end=%s\n' "${stage2_status}" "$(date --iso-8601=seconds)" \
  > "${SOLVER5_ROOT}/evaluation_status.txt"
if [[ ${stage2_status} -ne 0 ]]; then
  exit "${stage2_status}"
fi

cd "${ROOT}"
"${PYTHON_BIN}" -m model3_o2.scripts.finalize_object_eval --run-root "${SOLVER5_ROOT}"

"${PYTHON_BIN}" - "${RUN_ROOT}" "${SOLVER5_ROOT}" "${SELECTED_STEP}" <<'PY'
import csv
import datetime as dt
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
solver5_root = pathlib.Path(sys.argv[2]).resolve()
step = int(sys.argv[3])
solver10_root = root / f"solver10_step_{step:06d}"


def outcomes(path: pathlib.Path) -> dict[tuple[int, int], bool]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["task_id"]), int(row["episode_id"])): row["success"].lower() == "true"
            for row in csv.DictReader(handle)
        }


o10 = outcomes(solver10_root / "episode_metrics.csv")
o5 = outcomes(solver5_root / "episode_metrics.csv")
if set(o10) != set(o5) or len(o10) != 500:
    raise RuntimeError("solver 10 and solver 5 episode identities do not match")
both_success = sum(o10[key] and o5[key] for key in o10)
solver10_only = sum(o10[key] and not o5[key] for key in o10)
solver5_only = sum(not o10[key] and o5[key] for key in o10)
both_fail = sum(not o10[key] and not o5[key] for key in o10)
discordant = solver10_only + solver5_only
if discordant:
    tail = sum(math.comb(discordant, k) for k in range(min(solver10_only, solver5_only) + 1)) / (2 ** discordant)
    p_value = min(1.0, 2.0 * tail)
else:
    p_value = 1.0
v10 = json.loads((solver10_root / "result_validation.json").read_text(encoding="utf-8"))
v5 = json.loads((solver5_root / "result_validation.json").read_text(encoding="utf-8"))
comparison = {
    "schema_version": 1,
    "status": "pass",
    "checkpoint_step": step,
    "checkpoint_sha256": v10["checkpoint"]["sha256"],
    "episodes": 500,
    "solver10_successes": v10["results"]["successes"],
    "solver5_successes": v5["results"]["successes"],
    "both_success": both_success,
    "solver10_only": solver10_only,
    "solver5_only": solver5_only,
    "both_fail": both_fail,
    "exact_mcnemar_two_sided_p": p_value,
    "reporting_boundary": "solver5 is diagnostic_only and did not participate in checkpoint selection",
}
(root / "solver_comparison.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
manifest_path = root / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = "complete"
manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
manifest["selected"] = selection["selected"]
manifest["solver5_diagnostic"] = comparison
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
rows = [
    "| O2 step | Solver 10 successes | Selected |",
    "|---:|---:|:---:|",
]
for result in selection["results"]:
    rows.append(
        f"| {result['step']:,} | {result['successes']}/500 | "
        f"{'yes' if result['step'] == step else 'no'} |"
    )
(root / "run_report.md").write_text(
    "# Model3 O2 Object Checkpoint And Solver Evaluation\n\n"
    "Status: complete and validated.\n\n"
    + "\n".join(rows)
    + "\n\n"
    + f"Selected step: `{step}`. Solver 5: `{comparison['solver5_successes']}/500`; "
      f"solver 10: `{comparison['solver10_successes']}/500`.\n\n"
    + "Solver 5 is diagnostic-only and did not participate in checkpoint selection.\n",
    encoding="utf-8",
)
print(json.dumps(comparison, sort_keys=True))
PY

echo "[complete] run_root=${RUN_ROOT}"
