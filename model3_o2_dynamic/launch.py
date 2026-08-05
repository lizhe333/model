"""Validated Stage-2 launcher for the Dynamic response-prewarmed treatment."""

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

from .config import STAGE2_JOINT, Model3O2DynamicConfig, load_config
from .contracts import validate_contract


def build_backend_environment(
    config: Model3O2DynamicConfig,
    backend_output: Path,
    run_id: str,
    *,
    resume: Path | None = None,
) -> dict[str, str]:
    if config.initialization.stage_role != STAGE2_JOINT:
        raise ValueError("Dynamic launcher only accepts a materialized Stage 2 config")
    export = config.initialization.response_adapter_export
    export_sha = config.initialization.response_adapter_export_sha256
    if export is None or export_sha is None:
        raise ValueError("Dynamic Stage 2 config is missing adapter export identity")
    environment = build_model3_environment(config.base, backend_output, run_id)
    suite = config.evaluation.suite
    environment.update(
        {
            "RUN_TAG": f"model3_o2_dynamic_{suite}",
            "WANDB_PROJECT": "i003-model3-o2-dynamic",
            "WANDB_NAME": f"model3_o2_dynamic_{suite}_{run_id}",
            "HYDRA_CONFIG_ROOT": str(config.project_root / "model3_o2_dynamic/configs/hydra"),
            "MODEL_PACKAGE_LABEL": "model3_o2_dynamic",
            "MODEL3_O2_WARMSTART_PATH": str(config.initialization.model3_checkpoint),
            "MODEL3_O2_WARMSTART_SHA256": config.initialization.model3_checkpoint_sha256,
            "MODEL3_O2_WARMSTART_STEP": str(config.initialization.model3_checkpoint_step),
            "MODEL3_O2_DYNAMIC_ADAPTER_EXPORT": str(export),
            "MODEL3_O2_DYNAMIC_ADAPTER_EXPORT_SHA256": export_sha,
            "DYNAMIC_RESPONSE_ADAPTER_FREEZE_THROUGH_STEP": str(
                config.schedule.freeze_through_step
            ),
            "DYNAMIC_RESPONSE_FIRST_ADAPTER_UPDATE_STEP": str(
                config.schedule.first_adapter_update_step
            ),
            "DYNAMIC_RESPONSE_ADAPTER_LR_SCALE": str(config.schedule.adapter_lr_scale),
            "DYNAMIC_O2_GATE_FREEZE_THROUGH_STEP": str(
                config.schedule.gate_freeze_through_step
            ),
            "DYNAMIC_O2_FIRST_GATE_UPDATE_STEP": str(
                config.schedule.first_gate_update_step
            ),
            "DYNAMIC_O2_GATE_LR_SCALE": str(config.schedule.gate_lr_scale),
        }
    )
    if resume is not None:
        resume_path = Path(resume).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Dynamic Stage-2 resume path does not exist: {resume_path}")
        environment["RESUME"] = str(resume_path)
    return environment


def _prepare_run(
    config: Model3O2DynamicConfig,
    config_path: Path,
    run_id: str,
    *,
    resume: Path | None,
) -> tuple[Path, Path, dict[str, Any]]:
    evidence_dir = config.evidence_root / run_id
    backend_output = config.backend_runs_root / run_id
    existing_manifest_path = evidence_dir / "run_manifest.json"
    if evidence_dir.exists():
        if resume is None:
            raise FileExistsError(f"Dynamic Stage-2 evidence directory already exists: {evidence_dir}")
        if not existing_manifest_path.is_file():
            raise FileExistsError(
                "refusing to resume into an unrecognized Dynamic evidence directory: "
                f"{evidence_dir}"
            )
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("track_id") != "model3_o2_dynamic"
            or existing.get("stage1_adapter_export_sha256")
            != config.initialization.response_adapter_export_sha256
        ):
            raise ValueError("Dynamic Stage-2 resume evidence identity does not match this adapter export")
        manifest = dict(existing)
        manifest["resumed_from"] = str(Path(resume).expanduser().resolve())
        manifest["resumed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["completed_at"] = None
        manifest["primary_decision"] = "training_active"
        backend_output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(existing_manifest_path, manifest)
        return evidence_dir, backend_output, manifest
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
        "track_id": "model3_o2_dynamic",
        "parent_tracks": ["model3_o2", "action_response_local"],
        "run_type": "stage2_original_o2_joint_training",
        "dataset": "LIBERO",
        "suites": [config.evaluation.suite],
        "repo_commit": commit,
        "repo_dirty": bool(dirty),
        "environment_name": config.backend.conda_env,
        "primary_decision": "training_active",
        "evidence_scope": "experimental_local_training",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_at": None,
        "method_lineage": "model3_o2_dynamic_response_prewarm",
        "expected_model_class": "Model3O2DynamicWAM",
        "expected_action_policy_class": "VLAQueryLayerAwareDiTActionExpert",
        "expected_method_id": "model3_o2_dynamic_response_prewarm_v1",
        "action_objective": "flow_matching",
        "action_inference_steps": config.evaluation.num_inference_steps,
        "model3_warmstart_path": str(config.initialization.model3_checkpoint),
        "model3_warmstart_sha256": config.initialization.model3_checkpoint_sha256,
        "model3_warmstart_step": config.initialization.model3_checkpoint_step,
        "stage1_adapter_export": str(config.initialization.response_adapter_export),
        "stage1_adapter_export_sha256": config.initialization.response_adapter_export_sha256,
        "stage1_predictor_input": "adapter_residual_only",
        "stage2_response_adapter_scale": 1.0,
        "stage2_freeze_through_step": config.schedule.freeze_through_step,
        "stage2_first_adapter_update_step": config.schedule.first_adapter_update_step,
        "stage2_adapter_lr_scale": config.schedule.adapter_lr_scale,
        "stage2_gate_freeze_through_step": config.schedule.gate_freeze_through_step,
        "stage2_first_gate_update_step": config.schedule.first_gate_update_step,
        "stage2_gate_lr_scale": config.schedule.gate_lr_scale,
        "backend_output": str(backend_output),
        "resumed_from": None if resume is None else str(Path(resume).expanduser().resolve()),
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
        f"- adapter_export: `{config.initialization.response_adapter_export}`\n"
        f"- stage1_predictor_input: `adapter_residual_only`\n"
        f"- response_adapter_scale: `1.0`\n"
        f"- adapter_transition: step `{config.schedule.freeze_through_step}` -> `{config.schedule.first_adapter_update_step}`\n"
        f"- o2_gate_transition: step `{config.schedule.gate_freeze_through_step}` -> `{config.schedule.first_gate_update_step}`\n",
        encoding="utf-8",
    )
    return evidence_dir, backend_output, manifest


def launch(
    config_path: Path,
    run_id: str,
    *,
    dry_run: bool,
    resume: Path | None = None,
) -> int:
    config = load_config(config_path)
    validate_contract(config)
    backend_output = config.backend_runs_root / run_id
    environment = build_backend_environment(config, backend_output, run_id, resume=resume)
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
    config.backend.hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    evidence_dir, _, manifest = _prepare_run(config, config_path, run_id, resume=resume)
    (evidence_dir / "commands.txt").write_text(command_text + "\n", encoding="utf-8")
    (evidence_dir / "run_report.md").write_text(
        "# Model3 O2 Dynamic Stage 2\n\nStatus: active.\n", encoding="utf-8"
    )
    process_env = os.environ.copy()
    process_env.update(environment)
    try:
        result = subprocess.run(command, cwd=config.backend.repo, env=process_env, check=False)
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
        "# Model3 O2 Dynamic Stage 2\n\n"
        f"Status: {'completed' if return_code == 0 else 'stopped'}.\n\n"
        f"Return code: `{return_code}`\n",
        encoding="utf-8",
    )
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and launch Dynamic O2 Stage 2")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run_id = args.run_id or dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return launch(
        args.config.resolve(),
        run_id,
        dry_run=bool(args.dry_run),
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
