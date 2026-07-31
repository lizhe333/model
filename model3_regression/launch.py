"""Validated launcher for Model3 direct-regression LIBERO controls."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from model3.launch import (
    _git_output,
    _suite_label,
    _write_json,
    build_backend_command,
    build_backend_environment as build_model3_environment,
)

from .config import load_config
from .contracts import validate_contract


def build_backend_environment(config, backend_output: Path, run_id: str) -> dict[str, str]:
    environment = build_model3_environment(config, backend_output, run_id)
    hydra_root = config.project_root / "model3_regression/configs/hydra"
    environment.update(
        {
            "RUN_TAG": f"model3_regression_{config.evaluation.suite}",
            "WANDB_PROJECT": "i003-model3-regression",
            "WANDB_NAME": f"model3_regression_{config.evaluation.suite}_{run_id}",
            "HYDRA_CONFIG_ROOT": str(hydra_root),
            "MODEL_PACKAGE_LABEL": "model3_regression",
            "MODEL3_REGRESSION_POLICY_ACTION_HORIZON": str(
                config.data.policy_action_horizon
            ),
        }
    )
    return environment


def _prepare_run(config, config_path: Path, run_id: str) -> tuple[Path, Path, dict[str, Any]]:
    evidence_dir = config.evidence_root / run_id
    backend_output = config.backend_runs_root / run_id
    evidence_dir.mkdir(parents=True, exist_ok=False)
    backend_output.parent.mkdir(parents=True, exist_ok=True)

    contract = validate_contract(config)
    _write_json(evidence_dir / "contract_validation.json", contract)
    (evidence_dir / "config_snapshot.json").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    commit = _git_output(config.project_root, "rev-parse", "HEAD")
    dirty = _git_output(config.project_root, "status", "--short")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "track_id": "model3_regression",
        "parent_track": "model3",
        "run_type": "formal_training",
        "dataset": "LIBERO",
        "suites": [config.evaluation.suite],
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "repo_commit": commit,
        "repo_dirty": bool(dirty),
        "environment_name": config.backend.conda_env,
        "primary_decision": "training_active",
        "evidence_scope": "local_training_eval",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_at": None,
        "backend": "model3_vendored_light_wam",
        "method_lineage": "model3_same_query_direct_regression",
        "expected_model_class": "Model3RegressionWAM",
        "expected_action_policy_class": "VLAQueryRegressionActionExpert",
        "expected_method_id": "model3_regression_recurrent_query_l1_v1",
        "action_objective": "masked_l1",
        "action_inference_calls": 1,
        "backend_output": str(backend_output),
    }
    _write_json(evidence_dir / "run_manifest.json", manifest)
    (evidence_dir / "repo_snapshot.md").write_text(
        f"# Repository Snapshot\n\n- commit: `{commit}`\n- dirty: `{bool(dirty)}`\n\n"
        f"```text\n{dirty}\n```\n",
        encoding="utf-8",
    )
    (evidence_dir / "environment.md").write_text(
        "# Environment\n\n"
        f"- host: `{socket.gethostname()}`\n"
        f"- launcher_python: `{sys.version.split()[0]}`\n"
        f"- conda_env: `{config.backend.conda_env}`\n"
        f"- backend: `{config.backend.repo}`\n"
        f"- dataset: `{config.data.dataset_dir}`\n"
        f"- latent_cache: `{config.data.latent_cache_dir}`\n"
        "- action_objective: `masked_l1`\n"
        "- action_inference_calls: `1`\n",
        encoding="utf-8",
    )
    (evidence_dir / "run_report.md").write_text(
        f"# Model3 Regression {_suite_label(config)} Training\n\n"
        "Status: starting.\n\n"
        f"Backend output: `{backend_output}`\n",
        encoding="utf-8",
    )
    return evidence_dir, backend_output, manifest


def launch(config_path: Path, run_id: str, *, dry_run: bool) -> int:
    config = load_config(config_path)
    config.backend.hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    validate_contract(config)
    backend_output = config.backend_runs_root / run_id
    env_overrides = build_backend_environment(config, backend_output, run_id)
    command = build_backend_command(config)
    command_text = " ".join(
        [
            *(f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())),
            shlex.join(command),
        ]
    )
    if dry_run:
        print(command_text)
        return 0

    evidence_dir, backend_output, manifest = _prepare_run(config, config_path, run_id)
    (evidence_dir / "commands.txt").write_text(command_text + "\n", encoding="utf-8")
    (evidence_dir / "run_report.md").write_text(
        f"# Model3 Regression {_suite_label(config)} Training\n\n"
        "Status: active.\n\n"
        f"Backend output: `{backend_output}`\n",
        encoding="utf-8",
    )
    process_env = os.environ.copy()
    process_env.update(env_overrides)
    try:
        result = subprocess.run(
            command,
            cwd=config.backend.repo,
            env=process_env,
            check=False,
        )
    except KeyboardInterrupt:
        manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["primary_decision"] = "training_cancelled_user_requested"
        manifest["return_code"] = 130
        _write_json(evidence_dir / "run_manifest.json", manifest)
        (evidence_dir / "run_report.md").write_text(
            f"# Model3 Regression {_suite_label(config)} Training\n\n"
            "Status: cancelled by user.\n\nReturn code: `130`\n\n"
            f"Backend output: `{backend_output}`\n",
            encoding="utf-8",
        )
        return 130

    manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["primary_decision"] = (
        "training_completed" if result.returncode == 0 else "training_failed"
    )
    manifest["return_code"] = int(result.returncode)
    _write_json(evidence_dir / "run_manifest.json", manifest)
    (evidence_dir / "run_report.md").write_text(
        f"# Model3 Regression {_suite_label(config)} Training\n\n"
        f"Status: {'completed' if result.returncode == 0 else 'failed'}.\n\n"
        f"Return code: `{result.returncode}`\n\n"
        f"Backend output: `{backend_output}`\n",
        encoding="utf-8",
    )
    return int(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and launch Model3 Regression LIBERO training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return launch(args.config.resolve(), run_id, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
