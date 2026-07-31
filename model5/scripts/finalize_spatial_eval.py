#!/usr/bin/env python3
"""Validate and finalize a completed model5 LIBERO Spatial evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - exercised by the CLI environment check
    imageio = None

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by the CLI environment check
    yaml = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised by the CLI environment check
    torch = None


EPISODE_COLUMNS = [
    "track_id",
    "dataset",
    "suite",
    "task_id",
    "task_name",
    "episode_id",
    "seed",
    "checkpoint_sha256",
    "success",
    "valid",
    "steps_executed",
    "max_steps",
    "termination_reason",
    "wall_seconds",
    "video_path",
    "evidence_scope",
]

TASK_COLUMNS = [
    "track_id",
    "dataset",
    "suite",
    "task_id",
    "task_name",
    "checkpoint_sha256",
    "episodes",
    "valid_episodes",
    "successes",
    "success_rate",
    "mean_steps",
    "mean_wall_seconds",
    "evidence_scope",
]

RESULT_FILE_RE = re.compile(r"gpu(?P<gpu>\d+)_task(?P<task>\d+)_results\.json")
VIDEO_FILE_RE = re.compile(
    r".+--episode=task(?P<task>\d+)_trial(?P<episode>\d+)"
    r"--success=(?P<success>True|False)--task=.+\.mp4"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ValidationError(ValueError):
    """Raised when an evaluation artifact violates the finalization contract."""


@dataclass(frozen=True)
class RunContract:
    run_root: Path
    output_dir: Path
    track_id: str
    dataset: str
    suite: str
    evidence_scope: str
    num_tasks: int
    trials_per_task: int
    total_episodes: int
    max_steps: int
    seed: int
    seed_source: str
    checkpoint_path: Path
    checkpoint_sha256: str
    training_config_path: Path
    gpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class TaskResult:
    path: Path
    gpu_id: int
    task_id: int
    task_name: str
    successes: frozenset[int]
    failures: frozenset[int]
    duration_seconds: float


@dataclass(frozen=True)
class VideoResult:
    path: Path
    success: bool
    steps_executed: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"failed to read {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain a JSON object: {path}")
    return value


def _checked_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    _require(type(value) is int, f"{label} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None:
        _require(result >= minimum, f"{label} must be >= {minimum}, got {result}")
    return result


def _checked_float(value: Any, label: str, *, positive: bool = False) -> float:
    _require(type(value) in {int, float}, f"{label} must be numeric, got {value!r}")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite, got {result}")
    if positive:
        _require(result > 0.0, f"{label} must be positive, got {result}")
    return result


def _checked_int_list(value: Any, label: str) -> list[int]:
    _require(isinstance(value, list), f"{label} must be a list")
    result = [_checked_int(item, f"{label}[{index}]", minimum=0) for index, item in enumerate(value)]
    _require(result == sorted(result), f"{label} must be sorted")
    _require(len(result) == len(set(result)), f"{label} contains duplicate episode ids")
    return result


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True) == right.resolve(strict=True)
    except OSError:
        return False


def _path_from_string(value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value.strip(), f"{label} must be a non-empty path")
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    _require(path.is_file(), f"{label} does not exist: {path}")
    return path.resolve(strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_checkpoint(
    contract: RunContract,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    actual_sha256 = _sha256_file(contract.checkpoint_path)
    _require(actual_sha256 == contract.checkpoint_sha256, "checkpoint SHA-256 does not match run_manifest.json")
    _require(torch is not None, "PyTorch is required; run this finalizer in lightwam-libero-eval")
    try:
        payload = torch.load(
            contract.checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise ValidationError(f"failed to inspect checkpoint identity: {exc}") from exc
    _require(isinstance(payload, dict), "checkpoint payload must be a mapping")

    expected_method_id = "model5_asymmetric_tri_timestep_query_flow_v1"
    expected_model_class = "Model5WAM"
    _require(payload.get("method_id") == expected_method_id, "checkpoint method_id is not model5 VLA-query flow")
    _require(payload.get("model_class") == expected_model_class, "checkpoint model_class is not Model5WAM")
    filename_match = re.fullmatch(r"step_(\d+)\.pt", contract.checkpoint_path.name)
    _require(filename_match is not None, f"unexpected checkpoint filename: {contract.checkpoint_path.name}")
    filename_step = int(filename_match.group(1))
    checkpoint_step = _checked_int(payload.get("step"), "checkpoint step", minimum=0)
    _require(checkpoint_step == filename_step, "checkpoint payload step does not match its filename")

    policy_config = payload.get("action_policy_config")
    _require(isinstance(policy_config, dict), "checkpoint is missing action_policy_config")
    _require(policy_config.get("method_id") == expected_method_id, "checkpoint action policy method_id mismatch")
    _require(
        _checked_int(policy_config.get("action_horizon"), "checkpoint action policy horizon", minimum=1) == 8,
        "checkpoint action policy horizon must be 8",
    )
    _require(isinstance(payload.get("action_policy_state_dict"), dict), "checkpoint is missing action policy state")
    _require(isinstance(payload.get("mot"), dict), "checkpoint is missing MoT state")
    _require("state_fusion_action_expert" not in payload, "checkpoint contains a StateFusion action expert")

    optional_manifest_identity = {
        "checkpoint_method_id": expected_method_id,
        "checkpoint_model_class": expected_model_class,
        "checkpoint_step": checkpoint_step,
    }
    for key, expected in optional_manifest_identity.items():
        if key in manifest:
            _require(manifest[key] == expected, f"run manifest {key} does not match the checkpoint")
    return {
        "method_id": expected_method_id,
        "model_class": expected_model_class,
        "step": checkpoint_step,
    }


def _close_float(actual: Any, expected: float, label: str) -> None:
    actual_float = _checked_float(actual, label)
    _require(
        math.isclose(actual_float, expected, rel_tol=1e-9, abs_tol=1e-6),
        f"{label} mismatch: expected {expected}, got {actual_float}",
    )


def _load_training_seed(config_path: Path) -> int:
    _require(yaml is not None, "PyYAML is required; run this finalizer in lightwam-libero-eval")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"failed to read training config {config_path}: {exc}") from exc
    _require(isinstance(config, dict), f"training config must be a mapping: {config_path}")
    seed = _checked_int(config.get("seed"), "training config seed", minimum=0)
    model = config.get("model")
    _require(isinstance(model, dict), "training config is missing model mapping")
    _require(
        model.get("_target_") == "model5.runtime.create_model5_wam",
        f"unexpected model target: {model.get('_target_')!r}",
    )
    action_policy = model.get("action_query_policy_config")
    _require(isinstance(action_policy, dict), "training config is missing action_query_policy_config")
    _require(
        _checked_int(action_policy.get("action_horizon"), "model action horizon", minimum=1) == 8,
        "model action horizon must be 8",
    )
    _require(
        model.get("state_fusion_action_expert_config") is None,
        "model5 finalization rejects StateFusion action experts",
    )
    return seed


def _load_contract(run_root: Path) -> tuple[RunContract, dict[str, Any]]:
    _require(run_root.is_dir(), f"run root is not a directory: {run_root}")
    run_root = run_root.resolve(strict=True)
    manifest = _load_json(run_root / "run_manifest.json", "run manifest")

    _require(manifest.get("schema_version") == 1, "run manifest schema_version must be 1")
    _require(manifest.get("track_id") == "model5", "run manifest track_id must be model5")
    _require(manifest.get("dataset") == "LIBERO", "run manifest dataset must be LIBERO")
    _require(
        manifest.get("suites") == ["libero_spatial"],
        "run manifest suites must be exactly ['libero_spatial']",
    )
    _require(
        manifest.get("evidence_scope") == "local_training_eval",
        "run manifest evidence_scope must be local_training_eval",
    )

    checkpoint_path = _path_from_string(manifest.get("checkpoint_path"), "manifest checkpoint_path")
    checkpoint_sha256 = manifest.get("checkpoint_sha256")
    _require(
        isinstance(checkpoint_sha256, str) and SHA256_RE.fullmatch(checkpoint_sha256) is not None,
        "manifest checkpoint_sha256 must be a lowercase SHA-256 digest",
    )
    _require(
        checkpoint_path.parent.name == "weights"
        and checkpoint_path.parent.parent.name == "checkpoints",
        f"unexpected checkpoint layout: {checkpoint_path}",
    )
    training_config_path = checkpoint_path.parent.parent.parent / "config.yaml"
    _require(training_config_path.is_file(), f"missing checkpoint training config: {training_config_path}")
    training_config_path = training_config_path.resolve(strict=True)
    training_seed = _load_training_seed(training_config_path)

    protocol = manifest.get("protocol")
    _require(isinstance(protocol, dict), "run manifest is missing protocol mapping")
    num_tasks = _checked_int(protocol.get("tasks"), "protocol.tasks", minimum=1)
    trials_per_task = _checked_int(protocol.get("trials_per_task"), "protocol.trials_per_task", minimum=1)
    total_episodes = _checked_int(protocol.get("total_episodes"), "protocol.total_episodes", minimum=1)
    max_steps = _checked_int(protocol.get("max_episode_steps"), "protocol.max_episode_steps", minimum=1)
    _require(num_tasks == 10, f"LIBERO Spatial requires exactly 10 tasks, got {num_tasks}")
    _require(total_episodes == num_tasks * trials_per_task, "protocol total_episodes is inconsistent")
    _require(max_steps == 400, f"LIBERO Spatial max_episode_steps must be 400, got {max_steps}")
    _require(_checked_int(protocol.get("action_horizon"), "protocol.action_horizon") == 8, "action horizon must be 8")
    _require(_checked_int(protocol.get("replan_steps"), "protocol.replan_steps") == 8, "replan steps must be 8")
    _require(
        _checked_int(protocol.get("action_inference_steps"), "protocol.action_inference_steps") == 10,
        "action inference steps must be 10",
    )
    gpu_ids_value = protocol.get("gpu_ids")
    _require(isinstance(gpu_ids_value, list) and gpu_ids_value, "protocol.gpu_ids must be a non-empty list")
    gpu_ids = tuple(_checked_int(value, f"protocol.gpu_ids[{index}]", minimum=0) for index, value in enumerate(gpu_ids_value))
    _require(len(gpu_ids) == len(set(gpu_ids)), "protocol.gpu_ids contains duplicates")

    if "seed" in protocol:
        seed = _checked_int(protocol["seed"], "protocol.seed", minimum=0)
        _require(seed == training_seed, "protocol seed does not match the training/evaluation config")
        seed_source = "run_manifest.protocol.seed"
    else:
        seed = training_seed
        seed_source = "checkpoint_training_config.seed"

    output_dir = run_root / "eval_output"
    _require(output_dir.is_dir(), f"missing evaluation output directory: {output_dir}")
    contract = RunContract(
        run_root=run_root,
        output_dir=output_dir,
        track_id="model5",
        dataset="LIBERO",
        suite="libero_spatial",
        evidence_scope="local_training_eval",
        num_tasks=num_tasks,
        trials_per_task=trials_per_task,
        total_episodes=total_episodes,
        max_steps=max_steps,
        seed=seed,
        seed_source=seed_source,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        training_config_path=training_config_path,
        gpu_ids=gpu_ids,
    )
    return contract, manifest


def _load_task_results(contract: RunContract) -> dict[int, TaskResult]:
    failed_tasks_path = contract.run_root / "failed_tasks.txt"
    _require(failed_tasks_path.is_file(), f"missing failed task ledger: {failed_tasks_path}")
    _require(not failed_tasks_path.read_text(encoding="utf-8").strip(), "failed_tasks.txt is not empty")

    suite_dir = contract.output_dir / contract.suite
    _require(suite_dir.is_dir(), f"missing suite output directory: {suite_dir}")
    result_paths = sorted(suite_dir.glob("gpu*_task*_results.json"))
    _require(
        len(result_paths) == contract.num_tasks,
        f"expected {contract.num_tasks} task result files, found {len(result_paths)}",
    )

    expected_episode_ids = set(range(contract.trials_per_task))
    results: dict[int, TaskResult] = {}
    for path in result_paths:
        match = RESULT_FILE_RE.fullmatch(path.name)
        _require(match is not None, f"invalid task result filename: {path.name}")
        filename_gpu = int(match.group("gpu"))
        filename_task = int(match.group("task"))
        payload = _load_json(path, "task result")
        gpu_id = _checked_int(payload.get("gpu_id"), f"{path.name}.gpu_id", minimum=0)
        task_id = _checked_int(payload.get("task_id"), f"{path.name}.task_id", minimum=0)
        _require(gpu_id == filename_gpu, f"GPU id mismatch in {path.name}")
        _require(task_id == filename_task, f"task id mismatch in {path.name}")
        _require(gpu_id in contract.gpu_ids, f"unexpected GPU id {gpu_id} in {path.name}")
        _require(0 <= task_id < contract.num_tasks, f"unexpected task id {task_id} in {path.name}")
        _require(task_id not in results, f"duplicate task result for task {task_id}")
        _require(payload.get("task_suite") == contract.suite, f"suite mismatch in {path.name}")
        _require(
            _checked_int(payload.get("total_episodes"), f"{path.name}.total_episodes", minimum=1)
            == contract.trials_per_task,
            f"trial count mismatch in {path.name}",
        )
        task_name = payload.get("task_description")
        _require(isinstance(task_name, str) and task_name.strip(), f"missing task description in {path.name}")
        success_ids = _checked_int_list(payload.get("success_episodes"), f"{path.name}.success_episodes")
        failure_ids = _checked_int_list(payload.get("failure_episodes"), f"{path.name}.failure_episodes")
        success_set = set(success_ids)
        failure_set = set(failure_ids)
        _require(success_set.isdisjoint(failure_set), f"overlapping episode outcomes in {path.name}")
        _require(
            success_set | failure_set == expected_episode_ids,
            f"episode outcomes do not partition 0..{contract.trials_per_task - 1} in {path.name}",
        )
        declared_successes = _checked_int(payload.get("successes"), f"{path.name}.successes", minimum=0)
        _require(declared_successes == len(success_ids), f"success count mismatch in {path.name}")
        duration = _checked_float(payload.get("duration"), f"{path.name}.duration", positive=True)
        start_time = payload.get("start_time")
        _require(isinstance(start_time, str) and start_time.strip(), f"missing start_time in {path.name}")
        results[task_id] = TaskResult(
            path=path,
            gpu_id=gpu_id,
            task_id=task_id,
            task_name=task_name.strip(),
            successes=frozenset(success_set),
            failures=frozenset(failure_set),
            duration_seconds=duration,
        )

    _require(set(results) == set(range(contract.num_tasks)), "task results do not cover task ids 0..9")
    _require(
        len({result.task_name for result in results.values()}) == contract.num_tasks,
        "task descriptions are not unique",
    )
    return results


def _validate_summary(contract: RunContract, results: dict[int, TaskResult]) -> dict[str, Any]:
    summary_path = contract.output_dir / "summary.json"
    summary = _load_json(summary_path, "evaluation summary")
    _require(summary.get("run_id") == contract.output_dir.name, "summary run_id mismatch")
    summary_checkpoint = _path_from_string(summary.get("ckpt"), "summary.ckpt")
    summary_config = _path_from_string(summary.get("config"), "summary.config")
    _require(_same_path(summary_checkpoint, contract.checkpoint_path), "summary checkpoint path mismatch")
    _require(_same_path(summary_config, contract.training_config_path), "summary config path mismatch")

    total_successes = sum(len(result.successes) for result in results.values())
    total_duration = sum(result.duration_seconds for result in results.values())
    max_duration = max(result.duration_seconds for result in results.values())

    suite_stats = summary.get("suite_stats")
    _require(isinstance(suite_stats, dict) and set(suite_stats) == {contract.suite}, "summary suite_stats mismatch")
    suite_stat = suite_stats[contract.suite]
    _require(isinstance(suite_stat, dict), "summary suite statistics must be a mapping")
    _require(_checked_int(suite_stat.get("total_tasks"), "summary total_tasks") == contract.num_tasks, "summary task count mismatch")
    _require(
        _checked_int(suite_stat.get("total_trials"), "summary total_trials") == contract.total_episodes,
        "summary episode count mismatch",
    )
    _require(
        _checked_int(suite_stat.get("total_successes"), "summary total_successes") == total_successes,
        "summary success count mismatch",
    )
    _close_float(suite_stat.get("total_time"), total_duration, "summary suite total_time")
    _close_float(suite_stat.get("max_time"), max_duration, "summary suite max_time")

    task_summaries = summary.get("task_results")
    expected_task_keys = {f"{contract.suite}_{task_id}" for task_id in range(contract.num_tasks)}
    _require(
        isinstance(task_summaries, dict) and set(task_summaries) == expected_task_keys,
        "summary task_results keys mismatch",
    )
    for task_id, result in results.items():
        task_summary = task_summaries[f"{contract.suite}_{task_id}"]
        _require(isinstance(task_summary, dict), f"summary task {task_id} must be a mapping")
        _require(task_summary.get("task_description") == result.task_name, f"summary task name mismatch for task {task_id}")
        _require(
            _checked_int(task_summary.get("total_episodes"), f"summary task {task_id} total_episodes")
            == contract.trials_per_task,
            f"summary trial count mismatch for task {task_id}",
        )
        _require(
            _checked_int(task_summary.get("successes"), f"summary task {task_id} successes")
            == len(result.successes),
            f"summary success count mismatch for task {task_id}",
        )
        _close_float(task_summary.get("duration"), result.duration_seconds, f"summary task {task_id} duration")
        expected_percent = 100.0 * len(result.successes) / contract.trials_per_task
        _close_float(task_summary.get("success_rate"), expected_percent, f"summary task {task_id} success_rate")

    overall = summary.get("overall")
    _require(isinstance(overall, dict), "summary overall must be a mapping")
    expected_percent = 100.0 * total_successes / contract.total_episodes
    _close_float(overall.get("average_success_rate"), expected_percent, "summary overall success rate")
    _close_float(overall.get("total_time"), total_duration, "summary overall total_time")
    _close_float(
        overall.get("average_task_time"),
        total_duration / contract.num_tasks,
        "summary overall average_task_time",
    )
    return summary


def _count_video_frames(path: Path) -> int:
    _require(imageio is not None, "imageio is required; run this finalizer in lightwam-libero-eval")
    try:
        reader = imageio.get_reader(path)
        try:
            frame_count = reader.count_frames()
        finally:
            reader.close()
    except Exception as exc:
        raise ValidationError(f"failed to count frames in {path}: {exc}") from exc
    _require(type(frame_count) is int and frame_count > 0, f"invalid frame count for {path}: {frame_count!r}")
    return frame_count


def _load_videos(
    contract: RunContract,
    results: dict[int, TaskResult],
) -> dict[tuple[int, int], VideoResult]:
    video_dir = contract.output_dir / contract.suite / "videos"
    _require(video_dir.is_dir(), f"missing rollout video directory: {video_dir}")
    video_paths = sorted(video_dir.glob("*.mp4"))
    _require(
        len(video_paths) == contract.total_episodes,
        f"expected {contract.total_episodes} rollout videos, found {len(video_paths)}",
    )

    videos: dict[tuple[int, int], VideoResult] = {}
    for path in video_paths:
        match = VIDEO_FILE_RE.fullmatch(path.name)
        _require(match is not None, f"invalid rollout video filename: {path.name}")
        task_id = int(match.group("task"))
        episode_id = int(match.group("episode"))
        key = (task_id, episode_id)
        _require(0 <= task_id < contract.num_tasks, f"unexpected task id in video: {path.name}")
        _require(0 <= episode_id < contract.trials_per_task, f"unexpected episode id in video: {path.name}")
        _require(key not in videos, f"duplicate rollout video for task {task_id} episode {episode_id}")
        success = match.group("success") == "True"
        expected_success = episode_id in results[task_id].successes
        _require(success == expected_success, f"video success flag mismatch: {path.name}")
        steps_executed = _count_video_frames(path)
        _require(
            1 <= steps_executed <= contract.max_steps,
            f"video frame count is outside 1..{contract.max_steps}: {path.name} has {steps_executed}",
        )
        if not success:
            _require(
                steps_executed == contract.max_steps,
                f"failed rollout must reach max_steps={contract.max_steps}: {path.name} has {steps_executed}",
            )
        videos[key] = VideoResult(path=path, success=success, steps_executed=steps_executed)

    expected_keys = {
        (task_id, episode_id)
        for task_id in range(contract.num_tasks)
        for episode_id in range(contract.trials_per_task)
    }
    _require(set(videos) == expected_keys, "rollout videos do not cover every expected task/episode pair")
    return videos


def _metric(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _build_metric_rows(
    contract: RunContract,
    results: dict[int, TaskResult],
    videos: dict[tuple[int, int], VideoResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for task_id in range(contract.num_tasks):
        result = results[task_id]
        task_steps: list[int] = []
        for episode_id in range(contract.trials_per_task):
            video = videos[(task_id, episode_id)]
            task_steps.append(video.steps_executed)
            episode_rows.append(
                {
                    "track_id": contract.track_id,
                    "dataset": contract.dataset,
                    "suite": contract.suite,
                    "task_id": task_id,
                    "task_name": result.task_name,
                    "episode_id": episode_id,
                    "seed": contract.seed,
                    "checkpoint_sha256": contract.checkpoint_sha256,
                    "success": str(video.success).lower(),
                    "valid": "true",
                    "steps_executed": video.steps_executed,
                    "max_steps": contract.max_steps,
                    "termination_reason": "success" if video.success else "max_steps_reached",
                    "wall_seconds": "",
                    "video_path": video.path.relative_to(contract.run_root).as_posix(),
                    "evidence_scope": contract.evidence_scope,
                }
            )
        successes = len(result.successes)
        task_rows.append(
            {
                "track_id": contract.track_id,
                "dataset": contract.dataset,
                "suite": contract.suite,
                "task_id": task_id,
                "task_name": result.task_name,
                "checkpoint_sha256": contract.checkpoint_sha256,
                "episodes": contract.trials_per_task,
                "valid_episodes": contract.trials_per_task,
                "successes": successes,
                "success_rate": _metric(successes / contract.trials_per_task),
                "mean_steps": _metric(sum(task_steps) / len(task_steps)),
                "mean_wall_seconds": "",
                "evidence_scope": contract.evidence_scope,
            }
        )
    return episode_rows, task_rows


def _csv_bytes(columns: list[str], rows: Iterable[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _build_validation_payload(
    contract: RunContract,
    results: dict[int, TaskResult],
    checkpoint_identity: dict[str, Any],
    episode_csv: bytes,
    task_csv: bytes,
) -> dict[str, Any]:
    total_successes = sum(len(result.successes) for result in results.values())
    durations = {str(task_id): results[task_id].duration_seconds for task_id in range(contract.num_tasks)}
    return {
        "schema_version": 1,
        "status": "pass",
        "track_id": contract.track_id,
        "dataset": contract.dataset,
        "suite": contract.suite,
        "evidence_scope": contract.evidence_scope,
        "checkpoint": {
            "path": str(contract.checkpoint_path),
            "sha256": contract.checkpoint_sha256,
            "sha256_verified": True,
            "training_config_path": str(contract.training_config_path),
            **checkpoint_identity,
        },
        "protocol": {
            "tasks": contract.num_tasks,
            "trials_per_task": contract.trials_per_task,
            "total_episodes": contract.total_episodes,
            "max_episode_steps": contract.max_steps,
            "seed": contract.seed,
            "seed_source": contract.seed_source,
        },
        "results": {
            "task_result_files": contract.num_tasks,
            "valid_episodes": contract.total_episodes,
            "rollout_videos": contract.total_episodes,
            "successes": total_successes,
            "success_rate": total_successes / contract.total_episodes,
        },
        "field_provenance": {
            "steps_executed": "rollout_mp4_frame_count",
            "steps_definition": "policy-controlled environment steps excluding reset settle steps",
            "per_episode_wall_seconds_available": False,
            "task_mean_wall_seconds_available": False,
        },
        "raw_task_process_durations_seconds": durations,
        "raw_task_process_duration_semantics": (
            "Evaluator process wall time per task, including model/component loading and all task episodes; "
            "not apportioned into episode wall_seconds."
        ),
        "checks": {
            "failed_task_ledger_empty": True,
            "task_results_complete_and_unique": True,
            "episode_outcomes_complete_and_disjoint": True,
            "videos_complete_unique_and_readable": True,
            "video_success_flags_match_results": True,
            "failure_videos_reach_max_steps": True,
            "checkpoint_identity_path_and_sha256_match": True,
            "summary_matches_recomputed_results": True,
        },
        "artifacts": {
            "episode_metrics.csv": {
                "rows": contract.total_episodes,
                "sha256": _sha256_bytes(episode_csv),
            },
            "task_metrics.csv": {
                "rows": contract.num_tasks,
                "sha256": _sha256_bytes(task_csv),
            },
        },
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_atomically(run_root: Path, payloads: list[tuple[str, bytes]]) -> None:
    names = [name for name, _ in payloads]
    _require(
        names == ["episode_metrics.csv", "task_metrics.csv", "result_validation.json"],
        "finalization payload order must publish result_validation.json last",
    )
    staged: dict[str, Path] = {}
    try:
        for name, payload in payloads:
            fd, raw_path = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=run_root)
            temp_path = Path(raw_path)
            staged[name] = temp_path
            try:
                os.fchmod(fd, 0o644)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

        # Invalidate any previous pass marker before replacing either CSV. If a
        # later replace fails, consumers cannot mistake a partial commit for a
        # validated evidence set.
        validation_marker = run_root / "result_validation.json"
        validation_marker.unlink(missing_ok=True)
        _fsync_directory(run_root)

        # Persist both CSV renames before publishing the new completion marker.
        for name in names[:-1]:
            temp_path = staged[name]
            os.replace(temp_path, run_root / name)
            del staged[name]
        _fsync_directory(run_root)

        marker_name = names[-1]
        os.replace(staged[marker_name], validation_marker)
        del staged[marker_name]
        _fsync_directory(run_root)
    finally:
        for temp_path in staged.values():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def finalize(run_root: Path) -> dict[str, Any]:
    contract, manifest = _load_contract(run_root)
    results = _load_task_results(contract)
    _validate_summary(contract, results)
    videos = _load_videos(contract, results)
    checkpoint_identity = _validate_checkpoint(contract, manifest)

    episode_rows, task_rows = _build_metric_rows(contract, results, videos)
    episode_csv = _csv_bytes(EPISODE_COLUMNS, episode_rows)
    task_csv = _csv_bytes(TASK_COLUMNS, task_rows)
    validation = _build_validation_payload(contract, results, checkpoint_identity, episode_csv, task_csv)
    validation_json = _json_bytes(validation)
    _publish_atomically(
        contract.run_root,
        [
            ("episode_metrics.csv", episode_csv),
            ("task_metrics.csv", task_csv),
            ("result_validation.json", validation_json),
        ],
    )
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a complete model5 LIBERO Spatial run and publish structured metrics."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validation = finalize(args.run_root)
    except ValidationError as exc:
        print(f"[finalize] validation failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[finalize] filesystem failure: {exc}", file=sys.stderr)
        return 3

    results = validation["results"]
    run_root = args.run_root.resolve()
    print(
        "[finalize] status=pass "
        f"tasks={validation['protocol']['tasks']} "
        f"episodes={results['valid_episodes']} "
        f"successes={results['successes']} "
        f"success_rate={results['success_rate']:.6f}"
    )
    for filename in ("episode_metrics.csv", "task_metrics.csv", "result_validation.json"):
        print(f"[finalize] {filename}={run_root / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
