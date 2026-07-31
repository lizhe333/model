"""Validated launcher for registered Model3 O2 treatments."""

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
    _write_json,
    build_backend_command,
    build_backend_environment as build_model3_environment,
)

from .config import Model3O2Config, load_config
from .contracts import validate_contract


def build_backend_environment(
    config: Model3O2Config,
    backend_output: Path,
    run_id: str,
) -> dict[str, str]:
    environment = build_model3_environment(config.base, backend_output, run_id)
    suite = config.evaluation.suite
    environment.update(
        {
            "RUN_TAG": f"model3_o2_{suite}",
            "WANDB_PROJECT": "i003-model3-o2",
            "WANDB_NAME": f"model3_o2_{suite}_{run_id}",
            "HYDRA_CONFIG_ROOT": str(config.project_root / "model3_o2/configs/hydra"),
            "MODEL_PACKAGE_LABEL": "model3_o2",
            "MODEL3_O2_WARMSTART_PATH": str(config.initialization.model3_checkpoint),
            "MODEL3_O2_WARMSTART_SHA256": config.initialization.model3_checkpoint_sha256,
            "MODEL3_O2_WARMSTART_STEP": str(config.initialization.model3_checkpoint_step),
        }
    )
    return environment


def _prepare_run(
    config: Model3O2Config,
    config_path: Path,
    run_id: str,
) -> tuple[Path, Path, dict[str, Any]]:
    evidence_dir = config.evidence_root / run_id
    backend_output = config.backend_runs_root / run_id
    evidence_dir.mkdir(parents=True, exist_ok=False)
    backend_output.parent.mkdir(parents=True, exist_ok=True)
    contract = validate_contract(config)
    _write_json(evidence_dir / "contract_validation.json", contract)
    (evidence_dir / "config_snapshot.json").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    commit = _git_output(config.project_root, "rev-parse", "HEAD")
    dirty = _git_output(config.project_root, "status", "--short")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "track_id": "model3_o2",
        "parent_track": "model3",
        "run_type": "formal_training",
        "dataset": "LIBERO",
        "suites": [config.evaluation.suite],
        "repo_commit": commit,
        "repo_dirty": bool(dirty),
        "environment_name": config.backend.conda_env,
        "primary_decision": "training_active",
        "evidence_scope": "experimental_local_training",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_at": None,
        "method_lineage": "model3_layer_aware_query_flow",
        "expected_model_class": "Model3O2WAM",
        "expected_action_policy_class": "VLAQueryLayerAwareDiTActionExpert",
        "expected_method_id": "model3_o2_layer_aware_query_flow_v1",
        "action_objective": "flow_matching",
        "action_inference_steps": config.evaluation.num_inference_steps,
        "model3_warmstart_path": str(config.initialization.model3_checkpoint),
        "model3_warmstart_sha256": config.initialization.model3_checkpoint_sha256,
        "model3_warmstart_step": config.initialization.model3_checkpoint_step,
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
        f"- parent_checkpoint: `{config.initialization.model3_checkpoint}`\n"
        f"- parent_sha256: `{config.initialization.model3_checkpoint_sha256}`\n",
        encoding="utf-8",
    )
    return evidence_dir, backend_output, manifest


def launch(config_path: Path, run_id: str, *, dry_run: bool) -> int:
    config = load_config(config_path)
    config.backend.hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    validate_contract(config)
    backend_output = config.backend_runs_root / run_id
    environment = build_backend_environment(config, backend_output, run_id)
    command = build_backend_command(config.base)
    command_text = " ".join(
        [
            *(f"{key}={shlex.quote(value)}" for key, value in sorted(environment.items())),
            shlex.join(command),
        ]
    )
    if dry_run:
        print(command_text)
        return 0

    evidence_dir, _, manifest = _prepare_run(config, config_path, run_id)
    (evidence_dir / "commands.txt").write_text(command_text + "\n", encoding="utf-8")
    suite_label = "LIBERO Object" if config.evaluation.suite == "libero_object" else "LIBERO Long"
    (evidence_dir / "run_report.md").write_text(
        f"# Model3 O2 {suite_label} Training\n\nStatus: active.\n", encoding="utf-8"
    )
    process_env = os.environ.copy()
    process_env.update(environment)
    try:
        result = subprocess.run(
            command, cwd=config.backend.repo, env=process_env, check=False
        )
        return_code = int(result.returncode)
    except KeyboardInterrupt:
        return_code = 130
    manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["primary_decision"] = (
        "training_completed" if return_code == 0 else
        "training_cancelled_user_requested" if return_code == 130 else
        "training_failed"
    )
    manifest["return_code"] = return_code
    _write_json(evidence_dir / "run_manifest.json", manifest)
    suite_label = "LIBERO Object" if config.evaluation.suite == "libero_object" else "LIBERO Long"
    (evidence_dir / "run_report.md").write_text(
        f"# Model3 O2 {suite_label} Training\n\n"
        f"Status: {'completed' if return_code == 0 else 'stopped'}.\n\n"
        f"Return code: `{return_code}`\n",
        encoding="utf-8",
    )
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and launch Model3 O2")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_id = args.run_id or dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return launch(args.config.resolve(), run_id, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
