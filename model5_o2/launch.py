"""Validated launcher for the staged Model5 O2 experiment."""

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

from model5.launch import (
    _git_output,
    _write_json,
    build_backend_command,
    build_backend_environment as build_model5_environment,
)

from .config import STAGE1, STAGE2_CONTROL, STAGE2_O2, Model5O2Config, load_config
from .contracts import validate_contract


def build_backend_environment(
    config: Model5O2Config,
    backend_output: Path,
    run_id: str,
) -> dict[str, str]:
    environment = build_model5_environment(config.base, backend_output, run_id)
    environment.update(
        {
            "RUN_TAG": f"model5_o2_{config.stage_role}",
            "WANDB_PROJECT": "i003-model5-o2",
            "WANDB_NAME": f"model5_o2_{config.stage_role}_{run_id}",
            "HYDRA_CONFIG_ROOT": str(config.project_root / "model5_o2/configs/hydra"),
            "MODEL_PACKAGE_LABEL": "model5_o2",
            "MODEL5_O2_STAGE_ROLE": config.stage_role,
        }
    )
    if config.stage_role != STAGE1:
        initialization = config.initialization
        if initialization.model5_checkpoint is None:
            raise ValueError("Stage 2 requires a Model5 parent path")
        environment.update(
            {
                "MODEL5_O2_PARENT_PATH": str(initialization.model5_checkpoint),
                "MODEL5_O2_PARENT_SHA256": str(
                    initialization.model5_checkpoint_sha256
                ),
                "MODEL5_O2_PARENT_STEP": str(initialization.model5_checkpoint_step),
            }
        )
    return environment


def _identity(config: Model5O2Config) -> dict[str, str]:
    if config.stage_role in {STAGE1, STAGE2_CONTROL}:
        return {
            "model_class": "Model5WAM",
            "action_policy_class": "VLAQueryDiTActionExpert",
            "method_id": "model5_asymmetric_tri_timestep_query_flow_v1",
        }
    return {
        "model_class": "Model5O2WAM",
        "action_policy_class": "VLAQueryLayerAwareTemporalDiTActionExpert",
        "method_id": "model5_o2_layer_aware_temporal_query_flow_v1",
    }


def _prepare_run(
    config: Model5O2Config,
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
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    commit = _git_output(config.project_root, "rev-parse", "HEAD")
    dirty = _git_output(config.project_root, "status", "--short")
    identity = _identity(config)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "track_id": "model5_o2",
        "parent_track": "model5",
        "stage_role": config.stage_role,
        "run_type": "formal_training",
        "dataset": "LIBERO",
        "suites": ["libero_10"],
        "repo_commit": commit,
        "repo_dirty": bool(dirty),
        "environment_name": config.backend.conda_env,
        "primary_decision": "training_active",
        "evidence_scope": "experimental_local_training",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_at": None,
        "method_lineage": "model5_then_exact_q3_layer_aware_readout",
        "expected_model_class": identity["model_class"],
        "expected_action_policy_class": identity["action_policy_class"],
        "expected_method_id": identity["method_id"],
        "action_feature_temporal_scope": "current_plus_noisy_future",
        "future_feature_latent_slots": 8,
        "fixed_feature_timestep": 1000,
        "model5_parent": {
            "path": (
                None
                if config.initialization.model5_checkpoint is None
                else str(config.initialization.model5_checkpoint)
            ),
            "sha256": config.initialization.model5_checkpoint_sha256,
            "step": config.initialization.model5_checkpoint_step,
        },
        "backend_output": str(backend_output),
        "progress_log": str(backend_output / "logs" / "training.log"),
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
        f"- stage_role: `{config.stage_role}`\n",
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
    (evidence_dir / "run_report.md").write_text(
        f"# Model5 O2 {config.stage_role}\n\nStatus: active.\n",
        encoding="utf-8",
    )
    process_env = os.environ.copy()
    process_env.update(environment)
    try:
        result = subprocess.run(
            command,
            cwd=config.backend.repo,
            env=process_env,
            check=False,
        )
        return_code = int(result.returncode)
    except KeyboardInterrupt:
        return_code = 130
    manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["primary_decision"] = (
        "training_completed"
        if return_code == 0
        else "training_cancelled_user_requested"
        if return_code == 130
        else "training_failed"
    )
    manifest["return_code"] = return_code
    _write_json(evidence_dir / "run_manifest.json", manifest)
    (evidence_dir / "run_report.md").write_text(
        f"# Model5 O2 {config.stage_role}\n\n"
        f"Status: {'completed' if return_code == 0 else 'stopped'}.\n\n"
        f"Return code: `{return_code}`\n",
        encoding="utf-8",
    )
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and launch Model5 O2 stages")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_id = args.run_id or dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return launch(args.config.resolve(), run_id, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
