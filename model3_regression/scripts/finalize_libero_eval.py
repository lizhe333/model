#!/usr/bin/env python3
"""Validate and finalize a Model3 Regression LIBERO evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from model3.scripts import finalize_spatial_eval as base


EXPECTED_METHOD_ID = "model3_regression_recurrent_query_l1_v1"
EXPECTED_MODEL_CLASS = "Model3RegressionWAM"
EXPECTED_MODEL_TARGET = "model3_regression.runtime.create_model3_regression_wam"


def _load_training_seed(config_path: Path) -> int:
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise base.ValidationError(
            f"failed to read training config {config_path}: {exc}"
        ) from exc
    base._require(isinstance(config, dict), "training config must be a mapping")
    seed = base._checked_int(config.get("seed"), "training config seed", minimum=0)
    model = config.get("model")
    base._require(isinstance(model, dict), "training config is missing model mapping")
    base._require(
        model.get("_target_") == EXPECTED_MODEL_TARGET,
        f"unexpected model target: {model.get('_target_')!r}",
    )
    policy = model.get("action_query_policy_config")
    base._require(
        isinstance(policy, dict),
        "training config is missing action_query_policy_config",
    )
    base._require(
        base._checked_int(policy.get("action_horizon"), "model action horizon", minimum=1) == 8,
        "model action horizon must be 8",
    )
    base._require(
        model.get("state_fusion_action_expert_config") is None,
        "Regression finalization rejects StateFusion action experts",
    )
    return seed


def _validate_checkpoint(
    contract: base.RunContract,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    actual_sha256 = base._sha256_file(contract.checkpoint_path)
    base._require(
        actual_sha256 == contract.checkpoint_sha256,
        "checkpoint SHA-256 does not match run_manifest.json",
    )
    base._require(base.torch is not None, "PyTorch is required for finalization")
    try:
        payload = base.torch.load(
            contract.checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise base.ValidationError(f"failed to inspect checkpoint identity: {exc}") from exc
    base._require(isinstance(payload, dict), "checkpoint payload must be a mapping")
    base._require(
        payload.get("method_id") == EXPECTED_METHOD_ID,
        "checkpoint method_id is not Model3 Regression",
    )
    base._require(
        payload.get("model_class") == EXPECTED_MODEL_CLASS,
        "checkpoint model_class is not Model3RegressionWAM",
    )
    filename_match = re.fullmatch(r"step_(\d+)\.pt", contract.checkpoint_path.name)
    base._require(filename_match is not None, "unexpected checkpoint filename")
    filename_step = int(filename_match.group(1))
    checkpoint_step = base._checked_int(payload.get("step"), "checkpoint step", minimum=0)
    base._require(checkpoint_step == filename_step, "checkpoint step does not match filename")
    policy = payload.get("action_policy_config")
    base._require(isinstance(policy, dict), "checkpoint is missing action_policy_config")
    base._require(policy.get("method_id") == EXPECTED_METHOD_ID, "policy method_id mismatch")
    base._require(
        base._checked_int(policy.get("action_horizon"), "policy action horizon", minimum=1) == 8,
        "checkpoint action horizon must be 8",
    )
    base._require(
        isinstance(payload.get("action_policy_state_dict"), dict),
        "checkpoint is missing action policy state",
    )
    base._require(isinstance(payload.get("mot"), dict), "checkpoint is missing MoT state")
    base._require(
        "state_fusion_action_expert" not in payload,
        "checkpoint contains a StateFusion action expert",
    )
    optional_identity = {
        "checkpoint_method_id": EXPECTED_METHOD_ID,
        "checkpoint_model_class": EXPECTED_MODEL_CLASS,
        "checkpoint_step": checkpoint_step,
    }
    for key, expected in optional_identity.items():
        if key in manifest:
            base._require(manifest[key] == expected, f"manifest {key} mismatch")
    return {
        "method_id": EXPECTED_METHOD_ID,
        "model_class": EXPECTED_MODEL_CLASS,
        "step": checkpoint_step,
    }


def _load_contract(run_root: Path) -> tuple[base.RunContract, dict[str, Any]]:
    base._require(run_root.is_dir(), f"run root is not a directory: {run_root}")
    run_root = run_root.resolve(strict=True)
    manifest = base._load_json(run_root / "run_manifest.json", "run manifest")
    base._require(manifest.get("schema_version") == 1, "manifest schema_version must be 1")
    base._require(
        manifest.get("track_id") == "model3_regression",
        "manifest track_id must be model3_regression",
    )
    base._require(manifest.get("dataset") == "LIBERO", "manifest dataset must be LIBERO")
    base._require(manifest.get("suites") == ["libero_10"], "manifest suite must be libero_10")
    base._require(
        manifest.get("evidence_scope") == "local_training_eval",
        "manifest evidence_scope must be local_training_eval",
    )

    checkpoint_path = base._path_from_string(
        manifest.get("checkpoint_path"), "manifest checkpoint_path"
    )
    checkpoint_sha256 = manifest.get("checkpoint_sha256")
    base._require(
        isinstance(checkpoint_sha256, str)
        and base.SHA256_RE.fullmatch(checkpoint_sha256) is not None,
        "manifest checkpoint_sha256 must be a lowercase SHA-256 digest",
    )
    training_config_path = checkpoint_path.parent.parent.parent / "config.yaml"
    base._require(training_config_path.is_file(), "missing checkpoint training config")
    training_config_path = training_config_path.resolve(strict=True)
    training_seed = _load_training_seed(training_config_path)

    protocol = manifest.get("protocol")
    base._require(isinstance(protocol, dict), "manifest is missing protocol mapping")
    num_tasks = base._checked_int(protocol.get("tasks"), "protocol.tasks", minimum=1)
    trials_per_task = base._checked_int(
        protocol.get("trials_per_task"), "protocol.trials_per_task", minimum=1
    )
    total_episodes = base._checked_int(
        protocol.get("total_episodes"), "protocol.total_episodes", minimum=1
    )
    max_steps = base._checked_int(
        protocol.get("max_episode_steps"), "protocol.max_episode_steps", minimum=1
    )
    base._require(num_tasks == 10, "Regression Long evaluation requires 10 tasks")
    base._require(trials_per_task == 50, "Regression Long evaluation requires 50 trials per task")
    base._require(total_episodes == 500, "Regression Long evaluation requires 500 episodes")
    base._require(max_steps == 700, "Regression Long evaluation requires max_episode_steps=700")
    base._require(base._checked_int(protocol.get("action_horizon"), "action_horizon") == 8, "action horizon must be 8")
    base._require(base._checked_int(protocol.get("replan_steps"), "replan_steps") == 8, "replan steps must be 8")
    base._require(
        base._checked_int(protocol.get("action_inference_steps"), "action_inference_steps") == 1,
        "Regression action inference steps must be 1",
    )
    gpu_values = protocol.get("gpu_ids")
    base._require(isinstance(gpu_values, list) and gpu_values, "protocol.gpu_ids is required")
    gpu_ids = tuple(
        base._checked_int(value, f"protocol.gpu_ids[{index}]", minimum=0)
        for index, value in enumerate(gpu_values)
    )
    base._require(len(gpu_ids) == len(set(gpu_ids)), "protocol.gpu_ids contains duplicates")
    success_video_stride = base._checked_int(
        protocol.get("success_video_stride"), "success_video_stride", minimum=1
    )
    base._require(success_video_stride == 10, "success_video_stride must be 10")
    base._require(
        protocol.get("save_all_failure_videos") is True,
        "save_all_failure_videos must be true",
    )
    seed = base._checked_int(protocol.get("seed"), "protocol.seed", minimum=0)
    base._require(seed == training_seed, "protocol seed does not match training config")

    output_dir = run_root / "eval_output"
    base._require(output_dir.is_dir(), f"missing evaluation output directory: {output_dir}")
    contract = base.RunContract(
        run_root=run_root,
        output_dir=output_dir,
        track_id="model3_regression",
        dataset="LIBERO",
        suite="libero_10",
        evidence_scope="local_training_eval",
        num_tasks=num_tasks,
        trials_per_task=trials_per_task,
        total_episodes=total_episodes,
        max_steps=max_steps,
        seed=seed,
        seed_source="run_manifest.protocol.seed",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        training_config_path=training_config_path,
        gpu_ids=gpu_ids,
        success_video_stride=success_video_stride,
        save_all_failure_videos=True,
        video_retention_declared=True,
        summary_config_label=manifest.get("summary_config_label"),
    )
    return contract, manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _update_manifest(run_root: Path, validation: dict[str, Any]) -> None:
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = validation["results"]
    manifest["status"] = "validated"
    manifest["primary_decision"] = "completed_validated"
    manifest["completed_at"] = dt.datetime.now().astimezone().isoformat()
    manifest["terminal_validation"] = {
        "status": "pass",
        "finalized_at": dt.datetime.now().astimezone().isoformat(),
        "finalizer": "model3_regression/scripts/finalize_libero_eval.py",
        "finalizer_sha256": _sha256(Path(__file__).resolve()),
        "result_validation": "result_validation.json",
        "result_validation_sha256": _sha256(run_root / "result_validation.json"),
        "successes": results["successes"],
        "episodes": results["valid_episodes"],
        "success_rate": results["success_rate"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base._load_contract = _load_contract
    base._validate_checkpoint = _validate_checkpoint
    try:
        validation = base.finalize(args.run_root)
        _update_manifest(args.run_root.resolve(), validation)
    except base.ValidationError as exc:
        print(f"[finalize] validation failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[finalize] filesystem failure: {exc}", file=sys.stderr)
        return 3
    results = validation["results"]
    print(
        "[finalize] status=pass "
        f"tasks={validation['protocol']['tasks']} "
        f"episodes={results['valid_episodes']} "
        f"successes={results['successes']} "
        f"success_rate={results['success_rate']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
