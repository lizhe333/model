#!/usr/bin/env python3
"""Evaluate Model5 Object 10K/15K/20K at both solver 10 and solver 5."""

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
from pathlib import Path
from typing import Any


STEPS = (10_000, 15_000, 20_000)
SOLVER5_SPLIT = ((0, (0, 4, 8)), (1, (1, 5, 9)), (2, (2, 6)), (3, (3, 7)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ObjectEvaluation:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = args.root.resolve()
        self.vendor = args.vendor.resolve()
        self.train_run = args.train_run.resolve()
        self.libero_root = args.libero_root.resolve()
        self.python = args.python.resolve()
        self.run_root = args.run_root.resolve()
        self.log_path = self.run_root / "logs" / "evaluation.log"
        self.records: dict[int, dict[str, Any]] = {}

    def log(self, message: str) -> None:
        line = f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] [model5] [eval] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def checkpoint(self, step: int) -> Path:
        return self.train_run / "checkpoints" / "weights" / f"step_{step:06d}.pt"

    def condition(self, solver: int, step: int) -> Path:
        prefix = "solver10_step" if solver == 10 else "solver5_step"
        return self.run_root / f"{prefix}_{step:06d}"

    def environment(self, gpu: int, condition: Path, step: int) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "PIP_CONFIG_FILE": "/dev/null",
                "DIFFSYNTH_MODEL_BASE_PATH": str(self.root / "Light-WAM" / "checkpoints"),
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                "LIBERO_ROOT": str(self.libero_root),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "TOKENIZERS_PARALLELISM": "false",
                "HYDRA_FULL_ERROR": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": ":".join((str(self.root), str(self.vendor / "src"), str(self.vendor), str(self.libero_root))),
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "NUMBA_CACHE_DIR": str(condition / "cache" / f"numba_gpu{gpu}"),
                "MPLCONFIGDIR": str(condition / "cache" / f"mpl_gpu{gpu}"),
                # The upstream summarizer records these from its environment.
                # Bind them to this condition rather than inheriting shell state.
                "CKPT": str(self.checkpoint(step)),
                "CONFIG": str(self.train_run / "config.yaml"),
            }
        )
        return env

    def prepare(self, *, resume: bool = False) -> None:
        required = (
            self.python,
            self.train_run / "config.yaml",
            self.train_run / "dataset_stats.json",
            self.vendor / "experiments" / "libero" / "eval_libero_single.py",
            self.vendor / "experiments" / "libero" / "summarize_results.py",
            self.root / "model5" / "scripts" / "finalize_object_eval.py",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing evaluation prerequisites: " + ", ".join(missing))
        if resume:
            manifest_path = self.run_root / "run_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"cannot resume without run manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded_steps = manifest.get("checkpoint_steps", manifest.get("selection_set_steps"))
            if manifest.get("track_id") != "model5" or recorded_steps != list(STEPS):
                raise RuntimeError("resume manifest is not the expected Model5 Object checkpoint sweep")
            checkpoints = manifest.get("checkpoints")
            if not isinstance(checkpoints, list) or len(checkpoints) != len(STEPS):
                raise RuntimeError("resume manifest has an invalid checkpoint list")
            for record in checkpoints:
                if not isinstance(record, dict):
                    raise RuntimeError("resume checkpoint record is not a mapping")
                step = record.get("step")
                if type(step) is not int or step not in STEPS or record.get("path") != str(self.checkpoint(step)):
                    raise RuntimeError("resume checkpoint identity does not match this training run")
                checkpoint = self.checkpoint(step)
                if not checkpoint.is_file() or record.get("sha256") != _sha256(checkpoint):
                    raise RuntimeError(f"resume checkpoint is missing or changed: {checkpoint}")
                self.records[step] = {"step": step, "path": str(checkpoint), "sha256": record["sha256"]}
            self.log("resuming after solver-10 rollout completion; regenerating bound summaries and validation")
            return
        self.run_root.mkdir(parents=True, exist_ok=False)
        (self.run_root / "logs").mkdir()
        (self.run_root / "cache").mkdir()
        for step in STEPS:
            checkpoint = self.checkpoint(step)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
            payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            if payload.get("method_id") != "model5_asymmetric_tri_timestep_query_flow_v1":
                raise RuntimeError(f"step {step}: method identity mismatch")
            if payload.get("model_class") != "Model5WAM" or int(payload.get("step", -1)) != step:
                raise RuntimeError(f"step {step}: model class or step mismatch")
            self.records[step] = {"step": step, "path": str(checkpoint), "sha256": _sha256(checkpoint)}
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(
            self.run_root / "run_manifest.json",
            {
                "schema_version": 1,
                "track_id": "model5",
                "run_type": "object_checkpoint_two_solver_sweep",
                "checkpoint_steps": list(STEPS),
                "solver_execution_order": [10, 5],
                "checkpoints": [self.records[step] for step in STEPS],
                "protocol": {"task_suite": "libero_object", "tasks": 10, "trials_per_task": 50, "episodes_per_condition": 500, "seed": 42, "action_horizon": 8, "replan_steps": 8, "max_episode_steps": 400, "success_video_stride": 10, "save_all_failure_videos": True},
                "resource_layout": {"solver10": {"gpu_ids": [0, 1, 2, 3], "model_workers_per_gpu": 3}, "solver5": {"gpu_ids": [0, 1, 2, 3], "model_workers_per_gpu": 3}, "shared_gpu_training": False, "valid_for_latency_comparison": False},
                "started_at": started,
                "completed_at": None,
                "status": "preflight_passed",
            },
        )
        (self.run_root / "commands.txt").write_text(
            f"MODEL5_EVAL_RUN_ROOT={self.run_root} python -m model5.scripts.eval_object_two_stage --train-run {self.train_run}\n",
            encoding="utf-8",
        )
        (self.run_root / "environment.md").write_text(
            "# Environment\n\n- conda_env: `lightwam-libero-eval`\n- simulator: `LIBERO` with EGL\n- shared GPU timing is not latency evidence\n",
            encoding="utf-8",
        )
        for step in STEPS:
            self.make_condition(self.condition(10, step), step, 10, started)
        self.log("preflight passed for 10K, 15K, and 20K checkpoints")

    def make_condition(self, root: Path, step: int, solver: int, started: str | None = None) -> None:
        for subdir in ("eval_output", "logs/task_logs", "cache"):
            (root / subdir).mkdir(parents=True, exist_ok=True)
        record = self.records.get(step)
        if record is None:
            checkpoint = self.checkpoint(step)
            record = {"step": step, "path": str(checkpoint), "sha256": _sha256(checkpoint)}
            self.records[step] = record
        _write_json(
            root / "run_manifest.json",
            {
                "schema_version": 1,
                "track_id": "model5",
                "parent_track": "model3",
                "run_type": "per_checkpoint_solver_eval",
                "evidence_scope": "local_training_eval",
                "dataset": "LIBERO",
                "suites": ["libero_object"],
                "checkpoint_path": record["path"],
                "checkpoint_sha256": record["sha256"],
                "checkpoint_step": step,
                "protocol": {"tasks": 10, "trials_per_task": 50, "total_episodes": 500, "gpu_ids": [0, 1, 2, 3], "seed": 42, "action_horizon": 8, "replan_steps": 8, "action_inference_steps": solver, "max_episode_steps": 400, "success_video_stride": 10, "save_all_failure_videos": True},
                "started_at": started or dt.datetime.now(dt.timezone.utc).isoformat(),
                "completed_at": None,
                "status": "preflight_passed",
            },
        )
        (root / "failed_tasks.txt").write_text("", encoding="utf-8")
        (root / "run_report.md").write_text(f"# Model5 Object step {step} solver {solver}\n\nStatus: pending.\n", encoding="utf-8")

    def worker(self, condition: Path, step: int, solver: int, gpu: int, tasks: tuple[int, ...]) -> bool:
        failed = False
        env = self.environment(gpu, condition, step)
        for task in tasks:
            log = condition / "logs" / "task_logs" / f"libero_object_task{task}.log"
            command = [
                str(self.python), "experiments/libero/eval_libero_single.py", "task=libero_uncond_2cam224_1e-4",
                f"ckpt={self.checkpoint(step)}", f"gpu_id={gpu}", "EVALUATION.task_suite_name=libero_object",
                f"EVALUATION.task_id={task}", f"EVALUATION.output_dir={condition / 'eval_output'}",
                "EVALUATION.num_trials=50", "EVALUATION.use_training_run_config=true",
                f"EVALUATION.training_config_path={self.train_run / 'config.yaml'}",
                f"EVALUATION.dataset_stats_path={self.train_run / 'dataset_stats.json'}",
                "EVALUATION.action_horizon=8", "EVALUATION.replan_steps=8",
                f"EVALUATION.num_inference_steps={solver}", "EVALUATION.visualize_future_video=false",
                "EVALUATION.print_action_latency=false",
            ]
            self.log(f"launch solver={solver} step={step} gpu={gpu} task={task}")
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(command, cwd=self.vendor, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                failed = True
                with (condition / "failed_tasks.txt").open("a", encoding="utf-8") as handle:
                    handle.write(f"gpu={gpu},step={step},solver={solver},task={task},log={log},return_code={completed.returncode}\n")
        return not failed

    def finalize(self, condition: Path) -> dict[str, Any]:
        if (condition / "failed_tasks.txt").read_text(encoding="utf-8").strip():
            raise RuntimeError(f"worker failure in {condition}")
        manifest = json.loads((condition / "run_manifest.json").read_text(encoding="utf-8"))
        step = manifest.get("checkpoint_step")
        if type(step) is not int or step not in STEPS:
            raise RuntimeError(f"invalid checkpoint step in condition manifest: {condition}")
        summary = subprocess.run(
            [str(self.python), "experiments/libero/summarize_results.py", f"--output_dir={condition / 'eval_output'}"],
            cwd=self.vendor,
            env=self.environment(0, condition, step),
            check=False,
        )
        if summary.returncode:
            raise RuntimeError(f"summarizer failed for {condition}")
        result = subprocess.run(
            [str(self.python), "-m", "model5.scripts.finalize_object_eval", "--run-root", str(condition)],
            cwd=self.root,
            env=self.environment(0, condition, step),
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"terminal validation failed for {condition}")
        return json.loads((condition / "result_validation.json").read_text(encoding="utf-8"))

    def run(self, *, resume: bool = False) -> None:
        if not resume:
            stage1 = tuple(
                (self.condition(10, step), step, 10, gpu, tasks)
                for step in STEPS
                for gpu, tasks in SOLVER5_SPLIT
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                results = [pool.submit(self.worker, *job) for job in stage1]
                if not all(future.result() for future in results):
                    raise RuntimeError("solver-10 worker failure; solver-5 will not start")
        validations10 = {step: self.finalize(self.condition(10, step)) for step in STEPS}
        self.log("solver-10 results finalized for all checkpoints")

        for step in STEPS:
            self.make_condition(self.condition(5, step), step, 5)
        stage2 = tuple(
            (self.condition(5, step), step, 5, gpu, tasks)
            for step in STEPS
            for gpu, tasks in SOLVER5_SPLIT
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            results = [pool.submit(self.worker, *job) for job in stage2]
            if not all(future.result() for future in results):
                raise RuntimeError("solver-5 worker failure")
        validations5 = {step: self.finalize(self.condition(5, step)) for step in STEPS}
        comparisons = {
            step: self.compare(step, validations10[step], validations5[step], self.condition(5, step))
            for step in STEPS
        }
        self.complete(validations10, validations5, comparisons)

    def compare(self, step: int, validation10: dict[str, Any], validation5: dict[str, Any], solver5: Path) -> dict[str, Any]:
        def outcomes(path: Path) -> dict[tuple[int, int], bool]:
            with path.open(newline="", encoding="utf-8") as handle:
                return {(int(row["task_id"]), int(row["episode_id"])): row["success"].lower() == "true" for row in csv.DictReader(handle)}
        o10 = outcomes(self.condition(10, step) / "episode_metrics.csv")
        o5 = outcomes(solver5 / "episode_metrics.csv")
        if set(o10) != set(o5) or len(o10) != 500:
            raise RuntimeError("solver episode identities mismatch")
        both_success = sum(o10[key] and o5[key] for key in o10)
        solver10_only = sum(o10[key] and not o5[key] for key in o10)
        solver5_only = sum(not o10[key] and o5[key] for key in o10)
        both_fail = 500 - both_success - solver10_only - solver5_only
        discordant = solver10_only + solver5_only
        p_value = 1.0 if not discordant else min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(min(solver10_only, solver5_only) + 1)) / (2**discordant))
        comparison = {"schema_version": 1, "status": "pass", "checkpoint_step": step, "episodes": 500, "solver10_successes": validation10["results"]["successes"], "solver5_successes": validation5["results"]["successes"], "both_success": both_success, "solver10_only": solver10_only, "solver5_only": solver5_only, "both_fail": both_fail, "exact_mcnemar_two_sided_p": p_value, "reporting_boundary": "matched solver comparison; neither solver is automatically diagnostic"}
        _write_json(self.run_root / f"solver_comparison_step_{step:06d}.json", comparison)
        return comparison

    def complete(
        self,
        validations10: dict[int, dict[str, Any]],
        validations5: dict[int, dict[str, Any]],
        comparisons: dict[int, dict[str, Any]],
    ) -> None:
        def best_steps(validations: dict[int, dict[str, Any]]) -> list[int]:
            highest = max(
                int(validation["results"]["successes"])
                for validation in validations.values()
            )
            return [
                step
                for step in STEPS
                if int(validations[step]["results"]["successes"]) == highest
            ]

        best_by_solver = {
            "solver10": best_steps(validations10),
            "solver5": best_steps(validations5),
        }
        path = self.run_root / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update({"status": "complete", "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "best_observed_steps_by_solver": best_by_solver, "solver_comparisons": comparisons})
        _write_json(path, manifest)
        rows = "\n".join(f"- step {step}: solver-10 `{comparison['solver10_successes']}/500`; solver-5 `{comparison['solver5_successes']}/500`" for step, comparison in sorted(comparisons.items()))
        report = (
            "# Model5 Object Checkpoint And Solver Evaluation\n\n"
            "Status: complete and validated.\n\n"
            "Solver 10 ran before solver 5 only as a resource schedule. Both are "
            "formal per-checkpoint results; this coordinator does not impose a "
            "cross-solver selection rule.\n\n"
            f"Best observed checkpoints: solver-10 `{best_by_solver['solver10']}`; "
            f"solver-5 `{best_by_solver['solver5']}`.\n\n{rows}\n"
        )
        (self.run_root / "run_report.md").write_text(report, encoding="utf-8")
        self.log(f"complete best_by_solver={best_by_solver}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/lizhe/data/I-003_fast_video_dit_to_wam"))
    parser.add_argument("--vendor", type=Path, default=None)
    parser.add_argument("--libero-root", type=Path, default=Path("/data/users/lizhe/LIBERO"))
    parser.add_argument("--python", type=Path, default=Path("/data/miniconda3/envs/lightwam-libero-eval/bin/python"))
    parser.add_argument("--resume", action="store_true", help="Reuse complete solver-10 rollouts after a coordinator/finalization failure.")
    args = parser.parse_args()
    if args.vendor is None:
        args.vendor = args.root / "model5" / "third_party" / "light_wam"
    evaluation = ObjectEvaluation(args)
    try:
        evaluation.prepare(resume=args.resume)
        evaluation.run(resume=args.resume)
    except Exception as exc:
        evaluation.log(f"failed: {type(exc).__name__}: {exc}") if evaluation.run_root.exists() else print(f"[model5 eval] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
