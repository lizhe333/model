#!/usr/bin/env python3
"""Evaluate O2 Spatial steps 5K/10K and publish the predeclared selection."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

from .eval_long_two_checkpoint import (
    METHOD_ID,
    MODEL_CLASS,
    STEPS,
    SUMMARY_CONFIG_LABEL,
    TASK_GROUPS,
    _normalize_summary_identity,
    _sha256,
    _write_json,
)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--train-run",
        type=Path,
        default=root
        / "runs/I-003/model3_o2/backend_runs/2026-07-31_model3_o2_spatial_from_model3_60k_b16_ga1_10k",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/data/miniconda3/envs/lightwam-libero-eval/bin/python"),
    )
    parser.add_argument("--libero-root", type=Path, default=Path("/data/users/lizhe/LIBERO"))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    train_run = args.train_run.resolve()
    run_root = args.run_root.resolve()
    python_bin = args.python_bin.resolve()
    vendor = root / "model3/third_party/light_wam"
    evaluator = vendor / "experiments/libero/eval_libero_single.py"
    summarizer = vendor / "experiments/libero/summarize_results.py"
    finalizer = root / "model3_o2/scripts/finalize_object_eval.py"
    checkpoints = {
        step: train_run / "checkpoints/weights" / f"step_{step:06d}.pt"
        for step in STEPS
    }
    required = [
        python_bin,
        evaluator,
        summarizer,
        finalizer,
        train_run / "config.yaml",
        train_run / "dataset_stats.json",
        *checkpoints.values(),
    ]
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

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    coordinator: dict[str, Any] = {
        "schema_version": 1,
        "track_id": "model3_o2",
        "run_type": "spatial_checkpoint_selection",
        "selection_label": "best_observed_on_predeclared_checkpoint_set",
        "selection_rule": "highest successes over 500 episodes; earliest step breaks an exact tie",
        "selection_set_steps": list(STEPS),
        "checkpoints": checkpoint_records,
        "protocol": {
            "task_suite": "libero_spatial",
            "tasks": 10,
            "trials_per_task": 50,
            "episodes_per_condition": 500,
            "seed": 42,
            "action_horizon": 8,
            "replan_steps": 8,
            "action_inference_steps": 10,
            "max_episode_steps": 400,
            "success_video_stride": 10,
            "save_all_failure_videos": True,
        },
        "resource_layout": {
            "gpu_ids": [0, 1, 2, 3],
            "model_processes_per_gpu": 2,
            "valid_for_latency_comparison": False,
        },
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "repo_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=root, text=True
            ).strip()
        ),
        "started_at": started_at,
        "completed_at": None,
        "status": "preflight_passed",
    }
    _write_json(run_root / "run_manifest.json", coordinator)
    (run_root / "commands.txt").write_text(
        f"{python_bin} -m model3_o2.scripts.eval_spatial_two_checkpoint "
        f"--train-run {train_run} --run-root {run_root}\n",
        encoding="utf-8",
    )
    (run_root / "environment.md").write_text(
        "# Environment\n\n"
        "- conda_env: `lightwam-libero-eval`\n"
        "- simulator: `LIBERO` with EGL\n"
        "- GPUs: `0,1,2,3`; two model processes per GPU\n"
        "- shared-GPU timing is invalid for latency claims\n",
        encoding="utf-8",
    )
    (run_root / "run_report.md").write_text(
        "# Model3 O2 Spatial 5K/10K Evaluation\n\n"
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
                "track_id": "model3_o2",
                "parent_track": "model3",
                "run_type": "spatial_checkpoint_selection_eval",
                "evidence_scope": "local_training_eval",
                "dataset": "LIBERO",
                "suites": ["libero_spatial"],
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
                    "max_episode_steps": 400,
                    "success_video_stride": 10,
                    "save_all_failure_videos": True,
                },
                "started_at": started_at,
                "completed_at": None,
                "status": "preflight_passed",
            },
        )
        (condition / "run_report.md").write_text(
            f"# O2 Spatial Step {step} Solver-10 Evaluation\n\nStatus: pending.\n",
            encoding="utf-8",
        )
    print("[preflight] O2 Spatial checkpoints=5K,10K; 500 episodes each")
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
            log_path = condition / f"logs/task_logs/libero_spatial_task{task_id}.log"
            command = [
                str(python_bin),
                "experiments/libero/eval_libero_single.py",
                "task=libero_uncond_2cam224_1e-4",
                f"ckpt={checkpoints[step]}",
                f"gpu_id={gpu}",
                "EVALUATION.task_suite_name=libero_spatial",
                f"EVALUATION.task_id={task_id}",
                f"EVALUATION.output_dir={condition / 'eval_output'}",
                "EVALUATION.num_trials=50",
                "EVALUATION.use_training_run_config=true",
                f"EVALUATION.training_config_path={train_run / 'config.yaml'}",
                f"EVALUATION.dataset_stats_path={train_run / 'dataset_stats.json'}",
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

    futures: list[concurrent.futures.Future[bool]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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
        raise RuntimeError("at least one Spatial evaluation worker failed")

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
                "model3_o2.scripts.finalize_object_eval",
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
    coordinator["status"] = "complete"
    coordinator["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    coordinator["selected"] = selected
    _write_json(run_root / "run_manifest.json", coordinator)
    report_rows = [
        "# Model3 O2 Spatial 5K/10K Evaluation",
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
            "The selected result is best observed on the predeclared 5K/10K set, not an untouched final test.",
            "The historical Model3 Spatial 488/500 result is not used as a formal paired parent because its episode ledger was deleted.",
            "Shared-GPU timing is invalid for latency claims.",
        ]
    )
    (run_root / "run_report.md").write_text("\n".join(report_rows) + "\n", encoding="utf-8")
    print(f"[complete] selected_step={selected['step']} successes={selected['successes']}/500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
