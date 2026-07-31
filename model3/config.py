"""Typed model3 configuration with no backend implementation imports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackendConfig:
    repo: Path
    conda_env: str
    train_script: Path
    hydra_model: str
    checkpoint_root: Path
    hf_datasets_cache: Path


@dataclass(frozen=True)
class DataConfig:
    dataset_dir: Path
    use_latent_cache: bool
    latent_cache_dir: Path | None
    text_embedding_cache_dir: Path
    camera_keys: tuple[str, ...]
    camera_resolution: tuple[int, int]
    concat_multi_camera: str
    num_frames: int
    action_video_freq_ratio: int
    video_action_horizon: int
    policy_action_horizon: int


@dataclass(frozen=True)
class ArchitectureConfig:
    parent_track: str
    design_lineage: str
    conditioner_type: str
    uses_state_fusion: bool
    video_backbone: str
    freeze_backbone: bool
    use_backbone_lora: bool
    lora_rank: int
    lora_layers: tuple[int, ...]
    hidden_state_layers: tuple[int, ...]
    adapter_layers: tuple[int, ...]
    action_query_count: int
    action_query_hidden_dim: int
    action_query_heads: int
    action_query_bridge_depth: int
    action_decoder: str
    action_dit_layers: int
    action_dit_hidden_dim: int
    future_video_flow_loss: bool
    action_flow_loss: bool
    uses_privileged_future_latent_as_input: bool


@dataclass(frozen=True)
class TrainingConfig:
    gpu_ids: tuple[int, ...]
    num_processes: int
    main_process_port: int
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    max_steps: int
    save_every: int
    warmup_steps: int
    num_epochs: int
    learning_rate: float
    lr_scheduler_type: str
    mixed_precision: str
    seed: int
    max_grad_norm: float
    weight_decay: float
    wandb_mode: str


@dataclass(frozen=True)
class EvaluationConfig:
    suite: str
    tasks: int
    trials_per_task: int
    min_success_rate: float
    num_inference_steps: int
    max_episode_steps: int


@dataclass(frozen=True)
class Model3Config:
    project_root: Path
    track_id: str
    backend: BackendConfig
    data: DataConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    evidence_root: Path
    backend_runs_root: Path


def _path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _optional_path(root: Path, value: str | None) -> Path | None:
    return None if value is None else _path(root, value)


def load_config(path: str | Path) -> Model3Config:
    config_path = Path(path).expanduser().resolve()
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = _path(config_path.parent, raw["project_root"])
    backend_raw = raw["backend"]
    data_raw = raw["data"]
    architecture_raw = raw["architecture"]
    training_raw = raw["training"]
    evaluation_raw = raw["evaluation"]
    backend_repo = _path(project_root, backend_raw["repo"])

    return Model3Config(
        project_root=project_root,
        track_id=str(raw["track_id"]),
        backend=BackendConfig(
            repo=backend_repo,
            conda_env=str(backend_raw["conda_env"]),
            train_script=_path(project_root, backend_raw["train_script"]),
            hydra_model=str(backend_raw["hydra_model"]),
            checkpoint_root=_path(project_root, backend_raw["checkpoint_root"]),
            hf_datasets_cache=_path(project_root, backend_raw["hf_datasets_cache"]),
        ),
        data=DataConfig(
            dataset_dir=_path(project_root, data_raw["dataset_dir"]),
            use_latent_cache=bool(data_raw.get("use_latent_cache", True)),
            latent_cache_dir=_optional_path(project_root, data_raw.get("latent_cache_dir")),
            text_embedding_cache_dir=_path(project_root, data_raw["text_embedding_cache_dir"]),
            camera_keys=tuple(str(value) for value in data_raw["camera_keys"]),
            camera_resolution=tuple(int(value) for value in data_raw["camera_resolution"]),
            concat_multi_camera=str(data_raw["concat_multi_camera"]),
            num_frames=int(data_raw["num_frames"]),
            action_video_freq_ratio=int(data_raw["action_video_freq_ratio"]),
            video_action_horizon=int(data_raw["video_action_horizon"]),
            policy_action_horizon=int(data_raw["policy_action_horizon"]),
        ),
        architecture=ArchitectureConfig(
            parent_track=str(architecture_raw["parent_track"]),
            design_lineage=str(architecture_raw["design_lineage"]),
            conditioner_type=str(architecture_raw["conditioner_type"]),
            uses_state_fusion=bool(architecture_raw["uses_state_fusion"]),
            video_backbone=str(architecture_raw["video_backbone"]),
            freeze_backbone=bool(architecture_raw["freeze_backbone"]),
            use_backbone_lora=bool(architecture_raw["use_backbone_lora"]),
            lora_rank=int(architecture_raw["lora_rank"]),
            lora_layers=tuple(int(value) for value in architecture_raw["lora_layers"]),
            hidden_state_layers=tuple(int(value) for value in architecture_raw["hidden_state_layers"]),
            adapter_layers=tuple(int(value) for value in architecture_raw["adapter_layers"]),
            action_query_count=int(architecture_raw["action_query_count"]),
            action_query_hidden_dim=int(architecture_raw["action_query_hidden_dim"]),
            action_query_heads=int(architecture_raw["action_query_heads"]),
            action_query_bridge_depth=int(architecture_raw["action_query_bridge_depth"]),
            action_decoder=str(architecture_raw["action_decoder"]),
            action_dit_layers=int(architecture_raw["action_dit_layers"]),
            action_dit_hidden_dim=int(architecture_raw["action_dit_hidden_dim"]),
            future_video_flow_loss=bool(architecture_raw["future_video_flow_loss"]),
            action_flow_loss=bool(architecture_raw["action_flow_loss"]),
            uses_privileged_future_latent_as_input=bool(
                architecture_raw["uses_privileged_future_latent_as_input"]
            ),
        ),
        training=TrainingConfig(
            gpu_ids=tuple(int(value) for value in training_raw["gpu_ids"]),
            num_processes=int(training_raw["num_processes"]),
            main_process_port=int(training_raw["main_process_port"]),
            batch_size=int(training_raw["batch_size"]),
            gradient_accumulation_steps=int(training_raw["gradient_accumulation_steps"]),
            num_workers=int(training_raw["num_workers"]),
            max_steps=int(training_raw["max_steps"]),
            save_every=int(training_raw["save_every"]),
            warmup_steps=int(training_raw["warmup_steps"]),
            num_epochs=int(training_raw["num_epochs"]),
            learning_rate=float(training_raw["learning_rate"]),
            lr_scheduler_type=str(training_raw["lr_scheduler_type"]),
            mixed_precision=str(training_raw.get("mixed_precision", "bf16")),
            seed=int(training_raw.get("seed", 42)),
            max_grad_norm=float(training_raw.get("max_grad_norm", 1.0)),
            weight_decay=float(training_raw.get("weight_decay", 0.01)),
            wandb_mode=str(training_raw["wandb_mode"]),
        ),
        evaluation=EvaluationConfig(
            suite=str(evaluation_raw["suite"]),
            tasks=int(evaluation_raw["tasks"]),
            trials_per_task=int(evaluation_raw["trials_per_task"]),
            min_success_rate=float(evaluation_raw["min_success_rate"]),
            num_inference_steps=int(evaluation_raw["num_inference_steps"]),
            max_episode_steps=int(evaluation_raw["max_episode_steps"]),
        ),
        evidence_root=_path(project_root, raw["evidence_root"]),
        backend_runs_root=_path(project_root, raw["backend_runs_root"]),
    )
