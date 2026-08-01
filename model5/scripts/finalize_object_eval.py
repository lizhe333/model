#!/usr/bin/env python3
"""Validate and finalize a Model5 LIBERO Object evaluation condition."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from model5.scripts import finalize_spatial_eval as base


def _load_contract(run_root: Path) -> tuple[base.RunContract, dict[str, Any]]:
    base._require(run_root.is_dir(), f"run root is not a directory: {run_root}")
    run_root = run_root.resolve(strict=True)
    manifest = base._load_json(run_root / "run_manifest.json", "run manifest")
    base._require(manifest.get("schema_version") == 1, "manifest schema_version must be 1")
    base._require(manifest.get("track_id") == "model5", "manifest track_id must be model5")
    base._require(manifest.get("dataset") == "LIBERO", "manifest dataset must be LIBERO")
    base._require(manifest.get("suites") == ["libero_object"], "suite must be libero_object")

    protocol = manifest.get("protocol")
    base._require(isinstance(protocol, dict), "manifest is missing protocol")
    evidence_scope = manifest.get("evidence_scope")
    solver_steps = base._checked_int(
        protocol.get("action_inference_steps"), "action inference steps", minimum=1
    )
    base._require(solver_steps in {5, 10}, "Object solver must be 5 or 10")
    base._require(
        evidence_scope == "local_training_eval",
        "both Object solver conditions must retain local evaluation evidence",
    )

    checkpoint_path = base._path_from_string(
        manifest.get("checkpoint_path"), "manifest checkpoint_path"
    )
    checkpoint_sha256 = manifest.get("checkpoint_sha256")
    base._require(
        isinstance(checkpoint_sha256, str) and base.SHA256_RE.fullmatch(checkpoint_sha256),
        "manifest checkpoint_sha256 must be a lowercase SHA-256 digest",
    )
    training_config_path = checkpoint_path.parent.parent.parent / "config.yaml"
    base._require(training_config_path.is_file(), "missing checkpoint training config")
    training_config_path = training_config_path.resolve(strict=True)
    training_seed = base._load_training_seed(training_config_path)

    num_tasks = base._checked_int(protocol.get("tasks"), "protocol.tasks", minimum=1)
    trials = base._checked_int(protocol.get("trials_per_task"), "protocol.trials_per_task", minimum=1)
    episodes = base._checked_int(protocol.get("total_episodes"), "protocol.total_episodes", minimum=1)
    max_steps = base._checked_int(protocol.get("max_episode_steps"), "protocol.max_episode_steps", minimum=1)
    base._require((num_tasks, trials, episodes, max_steps) == (10, 50, 500, 400), "Object protocol must be 10 tasks x 50 trials x 400 steps")
    base._require(base._checked_int(protocol.get("action_horizon"), "action_horizon") == 8, "action horizon must be 8")
    base._require(base._checked_int(protocol.get("replan_steps"), "replan_steps") == 8, "replan steps must be 8")
    seed = base._checked_int(protocol.get("seed"), "seed", minimum=0)
    base._require(seed == training_seed, "protocol seed does not match training config")
    gpu_values = protocol.get("gpu_ids")
    base._require(isinstance(gpu_values, list) and gpu_values, "protocol.gpu_ids is required")
    gpu_ids = tuple(base._checked_int(value, f"gpu_ids[{index}]", minimum=0) for index, value in enumerate(gpu_values))
    base._require(len(gpu_ids) == len(set(gpu_ids)), "protocol.gpu_ids has duplicates")

    output_dir = run_root / "eval_output"
    base._require(output_dir.is_dir(), f"missing evaluation output directory: {output_dir}")
    return base.RunContract(
        run_root=run_root,
        output_dir=output_dir,
        track_id="model5",
        dataset="LIBERO",
        suite="libero_object",
        evidence_scope=str(evidence_scope),
        num_tasks=num_tasks,
        trials_per_task=trials,
        total_episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        seed_source="run_manifest.protocol.seed",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        training_config_path=training_config_path,
        gpu_ids=gpu_ids,
    ), manifest


def _mark_complete(run_root: Path, validation: dict[str, Any]) -> None:
    path = run_root / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "validated"
    manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["terminal_validation"] = {
        "status": "pass",
        "finalizer": "model5/scripts/finalize_object_eval.py",
        "successes": validation["results"]["successes"],
        "episodes": validation["results"]["valid_episodes"],
        "success_rate": validation["results"]["success_rate"],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    base._load_contract = _load_contract
    try:
        validation = base.finalize(args.run_root)
        _mark_complete(args.run_root.resolve(), validation)
    except base.ValidationError as exc:
        print(f"[finalize] validation failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[finalize] filesystem failure: {exc}", file=sys.stderr)
        return 3
    results = validation["results"]
    print(f"[finalize] status=pass successes={results['successes']}/{results['valid_episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
