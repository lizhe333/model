"""Validated launcher for the model2-derived model3 backend plugin."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import Model3Config, load_config
from .contracts import validate_contract


def build_backend_environment(config: Model3Config, backend_output: Path, run_id: str) -> dict[str, str]:
    training = config.training
    data = config.data
    suite = config.evaluation.suite
    environment = {
        "DIFFSYNTH_MODEL_BASE_PATH": str(config.backend.checkpoint_root),
        "DIFFSYNTH_SKIP_DOWNLOAD": "true",
        "HF_DATASETS_CACHE": str(config.backend.hf_datasets_cache),
        "PIP_CONFIG_FILE": "/dev/null",
        "GPU_IDS": ",".join(str(value) for value in training.gpu_ids),
        "NUM_PROCESSES": str(training.num_processes),
        "MAIN_PROCESS_PORT": str(training.main_process_port),
        "BATCH_SIZE": str(training.batch_size),
        "GRAD_ACC": str(training.gradient_accumulation_steps),
        "NUM_WORKERS": str(training.num_workers),
        "MAX_STEPS": str(training.max_steps),
        "SAVE_EVERY": str(training.save_every),
        "WARMUP_STEPS": str(training.warmup_steps),
        "NUM_EPOCHS": str(training.num_epochs),
        "LEARNING_RATE": str(training.learning_rate),
        "LR_SCHEDULER_TYPE": training.lr_scheduler_type,
        "MIXED_PRECISION": training.mixed_precision,
        "SEED": str(training.seed),
        "MAX_GRAD_NORM": str(training.max_grad_norm),
        "WEIGHT_DECAY": str(training.weight_decay),
        "RUN_ID": run_id,
        "RUN_TAG": f"model3_{suite}",
        "SUITE_NAME": suite,
        "OUTPUT_DIR": str(backend_output),
        "WANDB_PROJECT": "i003-model3",
        "WANDB_NAME": f"model3_{suite}_{run_id}",
        "WANDB_MODE": training.wandb_mode,
        "DATASET_DIR": str(data.dataset_dir),
        "USE_LATENT_CACHE": str(data.use_latent_cache).lower(),
        "TEXT_EMBED_CACHE_DIR": str(data.text_embedding_cache_dir),
        "TEMPORAL_PREFIX_STEPS": str(data.policy_action_horizon),
        "MODEL3_POLICY_ACTION_HORIZON": str(data.policy_action_horizon),
        "NUM_FRAMES": str(data.num_frames),
        "ACTION_VIDEO_FREQ_RATIO": str(data.action_video_freq_ratio),
        "VIDEO_SIZE": "[224,448]",
        "NUM_OUTPUT_CAMERAS": "2",
    }
    if data.latent_cache_dir is not None:
        environment["LATENT_CACHE_DIR"] = str(data.latent_cache_dir)
    return environment


def _suite_label(config: Model3Config) -> str:
    return {
        "libero_spatial": "LIBERO Spatial",
        "libero_object": "LIBERO Object",
        "libero_goal": "LIBERO Goal",
        "libero_10": "LIBERO Long",
    }[config.evaluation.suite]


def build_backend_command(config: Model3Config) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        config.backend.conda_env,
        "bash",
        str(config.backend.train_script),
        f"model={config.backend.hydra_model}",
        "train_visualization.enabled=true",
    ]


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _prepare_run(config: Model3Config, config_path: Path, run_id: str) -> tuple[Path, Path, dict[str, Any]]:
    evidence_dir = config.evidence_root / run_id
    backend_output = config.backend_runs_root / run_id
    evidence_dir.mkdir(parents=True, exist_ok=False)
    backend_output.parent.mkdir(parents=True, exist_ok=True)

    contract = validate_contract(config)
    _write_json(evidence_dir / "contract_validation.json", contract)
    (evidence_dir / "config_snapshot.json").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    commit = _git_output(config.project_root, "rev-parse", "HEAD")
    dirty = _git_output(config.project_root, "status", "--short")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "track_id": "model3",
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
        "backend": "vendored_light_wam",
        "method_lineage": "model2_vla_adapter_action_query",
        "expected_model_class": "Model3WAM",
        "expected_action_policy_class": "VLAQueryDiTActionExpert",
        "expected_method_id": "model3_vla_recurrent_query_flow_v1",
        "backend_role": "vendored_clean_light_wam_infrastructure",
        "backend_output": str(backend_output),
    }
    _write_json(evidence_dir / "run_manifest.json", manifest)
    (evidence_dir / "repo_snapshot.md").write_text(
        f"# Repository Snapshot\n\n- commit: `{commit}`\n- dirty: `{bool(dirty)}`\n\n```text\n{dirty}\n```\n",
        encoding="utf-8",
    )
    (evidence_dir / "environment.md").write_text(
        "# Environment\n\n"
        f"- host: `{socket.gethostname()}`\n"
        f"- launcher_python: `{sys.version.split()[0]}`\n"
        f"- conda_env: `{config.backend.conda_env}`\n"
        f"- backend: `{config.backend.repo}`\n"
        f"- checkpoint_root: `{config.backend.checkpoint_root}`\n"
        f"- hf_datasets_cache: `{config.backend.hf_datasets_cache}`\n"
        f"- dataset: `{config.data.dataset_dir}`\n"
        f"- latent_mode: `{'cached' if config.data.use_latent_cache else 'online_frozen_vae'}`\n"
        f"- latent_cache: `{config.data.latent_cache_dir}`\n",
        encoding="utf-8",
    )
    suite_label = _suite_label(config)
    (evidence_dir / "run_report.md").write_text(
        f"# Model3 {suite_label} Training\n\n"
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
        [*(f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())), shlex.join(command)]
    )
    if dry_run:
        print(command_text)
        return 0

    evidence_dir, backend_output, manifest = _prepare_run(config, config_path, run_id)
    (evidence_dir / "commands.txt").write_text(command_text + "\n", encoding="utf-8")
    manifest["primary_decision"] = "training_active"
    _write_json(evidence_dir / "run_manifest.json", manifest)
    suite_label = _suite_label(config)
    (evidence_dir / "run_report.md").write_text(
        f"# Model3 {suite_label} Training\n\n"
        "Status: active.\n\n"
        f"Backend output: `{backend_output}`\n",
        encoding="utf-8",
    )

    process_env = os.environ.copy()
    process_env.update(env_overrides)
    try:
        result = subprocess.run(command, cwd=config.backend.repo, env=process_env, check=False)
    except KeyboardInterrupt:
        manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["primary_decision"] = "training_cancelled_user_requested"
        manifest["return_code"] = 130
        _write_json(evidence_dir / "run_manifest.json", manifest)
        (evidence_dir / "run_report.md").write_text(
            f"# Model3 {suite_label} Training\n\n"
            "Status: cancelled by user.\n\n"
            "Return code: `130`\n\n"
            f"Backend output: `{backend_output}`\n",
            encoding="utf-8",
        )
        return 130

    manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["primary_decision"] = "training_completed" if result.returncode == 0 else "training_failed"
    manifest["return_code"] = int(result.returncode)
    _write_json(evidence_dir / "run_manifest.json", manifest)
    (evidence_dir / "run_report.md").write_text(
        f"# Model3 {suite_label} Training\n\n"
        f"Status: {'completed' if result.returncode == 0 else 'failed'}.\n\n"
        f"Return code: `{result.returncode}`\n\n"
        f"Backend output: `{backend_output}`\n",
        encoding="utf-8",
    )
    return int(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and launch Model3 LIBERO training.")
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
