#!/usr/bin/env python3
"""Evaluate Dynamic O2 Long steps 5K/10K and compare to the Model3 parent."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


METHOD_ID = "model3_o2_dynamic_response_prewarm_v1"
MODEL_CLASS = "Model3O2DynamicWAM"
SUMMARY_CONFIG_LABEL = "libero_flux2_klein_4b_base_imagewam"
STEPS = (5_000, 10_000)
TASK_GROUPS = {0: (0, 4, 8), 1: (1, 5, 9), 2: (2, 6), 3: (3, 7)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_summary_identity(summary_path: Path, checkpoint_path: Path) -> None:
    """Replace legacy summarizer labels with the audited checkpoint identity."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("config") not in {"", SUMMARY_CONFIG_LABEL}:
        raise RuntimeError(
            f"unexpected summary config label in {summary_path}: "
            f"{summary.get('config')!r}"
        )
    summary["ckpt"] = str(checkpoint_path)
    summary["config"] = SUMMARY_CONFIG_LABEL
    _write_json(summary_path, summary)


def _load_outcomes(path: Path) -> dict[tuple[int, int], bool]:
    with path.open(newline="", encoding="utf-8") as handle:
        outcomes: dict[tuple[int, int], bool] = {}
        for row in csv.DictReader(handle):
            key = (int(row["task_id"]), int(row["episode_id"]))
            value = row["success"].strip().lower()
            if value not in {"true", "false"}:
                raise RuntimeError(f"invalid success value {value!r} in {path}")
            if key in outcomes:
                raise RuntimeError(f"duplicate episode identity {key} in {path}")
            outcomes[key] = value == "true"
    return outcomes


def _exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _matched_comparison(
    selected_csv: Path,
    parent_csv: Path,
    *,
    selected_step: int,
    selected_sha256: str,
) -> dict[str, Any]:
    selected = _load_outcomes(selected_csv)
    parent = _load_outcomes(parent_csv)
    if len(selected) != 500 or set(selected) != set(parent):
        raise RuntimeError("selected O2 and parent Model3 episode identities do not match")
    both_success = sum(selected[key] and parent[key] for key in selected)
    o2_only = sum(selected[key] and not parent[key] for key in selected)
    model3_only = sum(not selected[key] and parent[key] for key in selected)
    both_fail = sum(not selected[key] and not parent[key] for key in selected)
    return {
        "schema_version": 1,
        "status": "pass",
        "episodes": 500,
        "o2_checkpoint_step": selected_step,
        "o2_checkpoint_sha256": selected_sha256,
        "o2_successes": both_success + o2_only,
        "model3_successes": both_success + model3_only,
        "both_success": both_success,
        "o2_only": o2_only,
        "model3_only": model3_only,
        "both_fail": both_fail,
        "exact_mcnemar_two_sided_p": _exact_mcnemar(o2_only, model3_only),
        "parent_result": "Model3 Long step 80K fixed result",
    }


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--train-run",
        type=Path,
        default=root
        / "runs/I-003/model3_o2/backend_runs/2026-07-31_model3_o2_long_from_model3_80k_b16_ga1_10k",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--parent-run",
        type=Path,
        default=root / "runs/I-003/model3/2026-07-26_model3_step080000_libero10_eval500",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/data/miniconda3/envs/lightwam-libero-eval/bin/python"),
    )
    parser.add_argument("--libero-root", type=Path, default=Path("/data/users/lizhe/LIBERO"))
    parser.add_argument(
        "--model-processes-per-gpu",
        type=int,
        choices=(1, 2),
        default=2,
        help="Concurrent evaluator processes per physical GPU; use 1 when sharing GPUs.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    train_run = args.train_run.resolve()
    run_root = args.run_root.resolve()
    parent_run = args.parent_run.resolve()
    python_bin = args.python_bin.resolve()
    vendor = root / "model3/third_party/light_wam"
    evaluator = vendor / "experiments/libero/eval_libero_single.py"
    summarizer = vendor / "experiments/libero/summarize_results.py"
    finalizer = root / "model3_o2_dynamic/scripts/finalize_object_eval.py"
    parent_csv = parent_run / "episode_metrics.csv"
    required = [
        python_bin,
        evaluator,
        summarizer,
        finalizer,
        train_run / "config.yaml",
        train_run / "dataset_stats.json",
        parent_run / "result_validation.json",
        parent_csv,
    ]
    checkpoints = {
        step: train_run / "checkpoints/weights" / f"step_{step:06d}.pt" for step in STEPS
    }
    required.extend(checkpoints.values())
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing required paths:\n- " + "\n- ".join(map(str, missing)))
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"run root is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    import torch

    checkpoint_records: list[dict[str, Any]] = []
    for step, checkpoint in checkpoints.items():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if payload.get("method_id") != METHOD_ID or payload.get("model_class") != MODEL_CLASS:
            raise RuntimeError(f"step {step}: O2 checkpoint identity mismatch")
        if int(payload.get("step", -1)) != step:
            raise RuntimeError(f"step {step}: checkpoint payload step mismatch")
        checkpoint_records.append(
            {"step": step, "path": str(checkpoint), "sha256": _sha256(checkpoint)}
        )

    parent_validation = json.loads(
        (parent_run / "result_validation.json").read_text(encoding="utf-8")
    )
    if parent_validation.get("status") != "pass":
        raise RuntimeError("parent Model3 Long result is not terminally validated")
    if int(parent_validation["results"]["successes"]) != 478:
        raise RuntimeError("parent Model3 Long result is not the fixed 478/500 reference")

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    repo_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    repo_dirty = bool(
        subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip()
    )
    coordinator = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "run_type": "long_checkpoint_selection_and_parent_comparison",
        "selection_label": "best_observed_on_predeclared_checkpoint_set",
        "selection_rule": "highest successes over 500 episodes; earliest step breaks an exact tie",
        "selection_set_steps": list(STEPS),
        "checkpoints": checkpoint_records,
        "parent_result": {
            "track_id": "model3",
            "run_root": str(parent_run),
            "checkpoint_step": 80_000,
            "successes": 478,
            "episodes": 500,
        },
        "protocol": {
            "task_suite": "libero_10",
            "tasks": 10,
            "trials_per_task": 50,
            "episodes_per_condition": 500,
            "seed": 42,
            "action_horizon": 8,
            "replan_steps": 8,
            "action_inference_steps": 10,
            "max_episode_steps": 700,
            "success_video_stride": 10,
            "save_all_failure_videos": True,
        },
        "resource_layout": {
            "gpu_ids": [0, 1, 2, 3],
            "model_processes_per_gpu": args.model_processes_per_gpu,
            "valid_for_latency_comparison": False,
        },
        "repo_commit": repo_commit,
        "repo_dirty": repo_dirty,
        "started_at": started_at,
        "completed_at": None,
        "status": "preflight_passed",
    }
    _write_json(run_root / "run_manifest.json", coordinator)
    (run_root / "commands.txt").write_text(
        f"{python_bin} -m model3_o2_dynamic.scripts.eval_long_two_checkpoint "
        f"--train-run {train_run} --run-root {run_root} "
        f"--model-processes-per-gpu {args.model_processes_per_gpu}\n",
        encoding="utf-8",
    )
    (run_root / "environment.md").write_text(
        "# Environment\n\n"
        "- conda_env: `lightwam-libero-eval`\n"
        "- simulator: `LIBERO` with EGL\n"
        f"- GPUs: `0,1,2,3`; {args.model_processes_per_gpu} model process(es) per GPU\n"
        "- shared-GPU timing is invalid for latency claims\n",
        encoding="utf-8",
    )
    (run_root / "run_report.md").write_text(
        "# Model3 O2 Long 5K/10K Evaluation\n\n"
        "Status: preflight passed; closed-loop evaluation pending.\n",
        encoding="utf-8",
    )
    for record in checkpoint_records:
        step = int(record["step"])
        condition = run_root / f"solver10_step_{step:06d}"
        for subdir in ("eval_output", "logs/task_logs", "cache"):
            (condition / subdir).mkdir(parents=True, exist_ok=True)
        (condition / "failed_tasks.txt").write_text("", encoding="utf-8")
        _write_json(
            condition / "run_manifest.json",
            {
                "schema_version": 1,
                "track_id": "model3_o2_dynamic",
                "parent_track": "model3",
                "run_type": "long_checkpoint_selection_eval",
                "evidence_scope": "local_training_eval",
                "dataset": "LIBERO",
                "suites": ["libero_10"],
                "selection_label": "best_observed_on_predeclared_checkpoint_set",
                "selection_set_steps": list(STEPS),
                "checkpoint_path": record["path"],
                "checkpoint_sha256": record["sha256"],
                "checkpoint_step": step,
                "summary_config_label": SUMMARY_CONFIG_LABEL,
                "summary_config_note": (
                    "Legacy Hydra label retained by the vendor summarizer; task logs and "
                    "summary checkpoint identity are normalized to this registered run."
                ),
                "protocol": {
                    "tasks": 10,
                    "trials_per_task": 50,
                    "total_episodes": 500,
                    "gpu_ids": [0, 1, 2, 3],
                    "seed": 42,
                    "action_horizon": 8,
                    "replan_steps": 8,
                    "action_inference_steps": 10,
                    "max_episode_steps": 700,
                    "success_video_stride": 10,
                    "save_all_failure_videos": True,
                },
                "resource_layout": {
                    "gpu_ids": [0, 1, 2, 3],
                    "model_processes_per_gpu": args.model_processes_per_gpu,
                    "shared_gpu": True,
                },
                "started_at": started_at,
                "completed_at": None,
                "status": "preflight_passed",
            },
        )
        (condition / "run_report.md").write_text(
            f"# O2 Long Step {step} Solver-10 Evaluation\n\nStatus: pending.\n",
            encoding="utf-8",
        )
    print("[preflight] O2 Long checkpoints=5K,10K; parent=Model3 Long-80K 478/500")
    if args.preflight_only:
        return 0

    child_env = os.environ.copy()
    child_env.update(
        {
            "PIP_CONFIG_FILE": "/dev/null",
            "DIFFSYNTH_MODEL_BASE_PATH": str(root / "Light-WAM/checkpoints"),
            "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            "HF_HOME": child_env.get("MODEL3_O2_HF_HOME", "/data/cache/huggingface"),
            "LIBERO_ROOT": str(args.libero_root.resolve()),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "TOKENIZERS_PARALLELISM": "false",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": ":".join(
                [str(root), str(vendor / "src"), str(vendor), str(args.libero_root.resolve())]
            ),
        }
    )
    child_env.pop("PYTHONHOME", None)
    child_env.pop("VIRTUAL_ENV", None)
    active: list[subprocess.Popen[Any]] = []
    active_lock = threading.Lock()
    ledger_lock = threading.Lock()

    def run_group(step: int, gpu: int, task_ids: tuple[int, ...]) -> bool:
        condition = run_root / f"solver10_step_{step:06d}"
        cache_env = child_env.copy()
        cache_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cache_env["NUMBA_CACHE_DIR"] = str(condition / f"cache/numba_gpu{gpu}")
        cache_env["MPLCONFIGDIR"] = str(condition / f"cache/mpl_gpu{gpu}")
        Path(cache_env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(cache_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        ok = True
        for task_id in task_ids:
            log_path = condition / f"logs/task_logs/libero_10_task{task_id}.log"
            command = [
                str(python_bin),
                "experiments/libero/eval_libero_single.py",
                "task=libero_uncond_2cam224_1e-4",
                f"ckpt={checkpoints[step]}",
                f"gpu_id={gpu}",
                "EVALUATION.task_suite_name=libero_10",
                f"EVALUATION.task_id={task_id}",
                f"EVALUATION.output_dir={condition / 'eval_output'}",
                "EVALUATION.num_trials=50",
                "EVALUATION.use_training_run_config=true",
                f"EVALUATION.training_config_path={train_run / 'config.yaml'}",
                f"EVALUATION.dataset_stats_path={train_run / 'dataset_stats.json'}",
                # Deployment restores the self-contained Dynamic checkpoint below.
                # Re-loading the Stage 1 export during model construction would make
                # inference depend on an intermediate training artifact and on the
                # RNG consumed by optional evaluator-only components such as T5.
                "+model.response_adapter_export_path=",
                "+model.response_adapter_export_sha256=",
                "EVALUATION.action_horizon=8",
                "EVALUATION.replan_steps=8",
                "EVALUATION.num_inference_steps=10",
                "EVALUATION.visualize_future_video=false",
                "EVALUATION.success_video_stride=10",
                "EVALUATION.save_all_failure_videos=true",
                "EVALUATION.print_action_latency=false",
            ]
            print(f"[eval] step={step} gpu={gpu} task={task_id} trials=50")
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=vendor,
                    env=cache_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with active_lock:
                    active.append(process)
                return_code = process.wait()
                with active_lock:
                    active.remove(process)
            if return_code != 0:
                ok = False
                with ledger_lock:
                    with (condition / "failed_tasks.txt").open("a", encoding="utf-8") as ledger:
                        ledger.write(
                            f"{dt.datetime.now().astimezone().isoformat()},gpu{gpu},"
                            f"step{step},task{task_id},{log_path}\n"
                        )
        return ok

    def run_gpu_sequential(gpu: int, task_ids: tuple[int, ...]) -> bool:
        """Run both checkpoint conditions serially on one shared physical GPU."""

        return all(run_group(step, gpu, task_ids) for step in STEPS)

    futures: list[concurrent.futures.Future[bool]] = []
    try:
        max_workers = len(TASK_GROUPS) * args.model_processes_per_gpu
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            if args.model_processes_per_gpu == 1:
                for gpu, task_ids in TASK_GROUPS.items():
                    futures.append(executor.submit(run_gpu_sequential, gpu, task_ids))
            else:
                for step in STEPS:
                    for gpu, task_ids in TASK_GROUPS.items():
                        futures.append(executor.submit(run_group, step, gpu, task_ids))
            worker_ok = all(future.result() for future in futures)
    except KeyboardInterrupt:
        with active_lock:
            for process in active:
                process.terminate()
        raise
    if not worker_ok:
        raise RuntimeError("at least one Long evaluation worker failed")

    results: list[dict[str, Any]] = []
    for record in checkpoint_records:
        step = int(record["step"])
        condition = run_root / f"solver10_step_{step:06d}"
        subprocess.run(
            [str(python_bin), str(summarizer), f"--output_dir={condition / 'eval_output'}"],
            cwd=vendor,
            env=child_env,
            check=True,
        )
        _normalize_summary_identity(
            condition / "eval_output/summary.json",
            Path(str(record["path"])),
        )
        subprocess.run(
            [
                str(python_bin),
                "-m",
                "model3_o2_dynamic.scripts.finalize_object_eval",
                "--run-root",
                str(condition),
            ],
            cwd=root,
            env=child_env,
            check=True,
        )
        validation = json.loads(
            (condition / "result_validation.json").read_text(encoding="utf-8")
        )
        results.append(
            {
                "step": step,
                "successes": int(validation["results"]["successes"]),
                "episodes": int(validation["results"]["valid_episodes"]),
                "success_rate": float(validation["results"]["success_rate"]),
                "checkpoint_sha256": validation["checkpoint"]["sha256"],
            }
        )

    selected = sorted(results, key=lambda item: (-item["successes"], item["step"]))[0]
    selection = {
        "schema_version": 1,
        "status": "pass",
        "selection_label": "best_observed_on_predeclared_checkpoint_set",
        "selection_rule": "highest successes; earliest step breaks an exact tie",
        "results": results,
        "selected": selected,
    }
    _write_json(run_root / "selection.json", selection)
    selected_root = run_root / f"solver10_step_{int(selected['step']):06d}"
    comparison = _matched_comparison(
        selected_root / "episode_metrics.csv",
        parent_csv,
        selected_step=int(selected["step"]),
        selected_sha256=str(selected["checkpoint_sha256"]),
    )
    _write_json(run_root / "parent_comparison.json", comparison)
    coordinator["status"] = "complete"
    coordinator["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    coordinator["selected"] = selected
    coordinator["parent_comparison"] = comparison
    _write_json(run_root / "run_manifest.json", coordinator)
    report_rows = [
        "# Model3 O2 Dynamic Long 5K/10K Evaluation",
        "",
        "Status: complete and terminally validated.",
        "",
        "| O2-local step | Successes | Selected |",
        "|---:|---:|:---:|",
    ]
    for result in results:
        report_rows.append(
            f"| {int(result['step']):,} | {int(result['successes'])}/500 | "
            f"{'yes' if result['step'] == selected['step'] else 'no'} |"
        )
    report_rows.extend(
        [
            "",
            f"Selected O2 versus fixed Model3 Long: {comparison['o2_successes']}/500 "
            f"versus {comparison['model3_successes']}/500.",
            f"Matched outcomes: both success {comparison['both_success']}, "
            f"O2 only {comparison['o2_only']}, Model3 only {comparison['model3_only']}, "
            f"both fail {comparison['both_fail']}.",
            f"Exact two-sided McNemar p={comparison['exact_mcnemar_two_sided_p']}.",
            "",
            "The selected result is best observed on the predeclared 5K/10K set, not an untouched final test.",
            "Shared-GPU timing is invalid for latency claims.",
        ]
    )
    (run_root / "run_report.md").write_text("\n".join(report_rows) + "\n", encoding="utf-8")
    print(
        f"[complete] selected_step={selected['step']} "
        f"successes={selected['successes']}/500 model3=478/500"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
