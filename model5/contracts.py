"""Fail-fast checks for the scientific and runtime model5 contract."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Model5Config


class ContractError(ValueError):
    """Raised when a launch would no longer represent the model5 method."""


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_contract(config: Model5Config, *, check_paths: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    architecture = config.architecture
    data = config.data
    training = config.training
    evaluation = config.evaluation

    _require(config.track_id == "model5", "track_id must be model5", errors)
    _require(data.camera_keys == ("image", "wrist_image"), "dual camera order must be image,wrist_image", errors)
    _require(data.camera_resolution == (224, 224), "each camera must be 224x224", errors)
    _require(data.concat_multi_camera == "horizontal", "camera concatenation must be horizontal", errors)
    _require(data.num_frames == 33, "video/action window must contain 33 environment frames", errors)
    _require(data.action_video_freq_ratio == 4, "action/video frequency ratio must be 4", errors)
    _require(data.video_action_horizon == 32, "the future-video window must retain 32 action transitions", errors)
    _require(data.policy_action_horizon == 8, "model5 must preserve Model3's 8-step action chunk", errors)

    _require(architecture.parent_track == "model3", "model5 parent_track must be model3", errors)
    _require(
        architecture.design_lineage == "asymmetric_tri_timestep_vla_query_b1",
        "model5 must declare the asymmetric tri-timestep action-query lineage",
        errors,
    )
    _require(
        architecture.conditioner_type == "layerwise_recurrent_action_query",
        "model5 must use the recurrent layer-wise action-query conditioner",
        errors,
    )
    _require(not architecture.uses_state_fusion, "StateFusion is not the model5 conditioner", errors)
    _require(architecture.video_backbone == "Wan-AI/Wan2.1-T2V-1.3B", "unexpected Video-DiT backbone", errors)
    _require(architecture.freeze_backbone, "base Wan weights must stay frozen", errors)
    _require(architecture.use_backbone_lora, "Wan LoRA must be enabled", errors)
    _require(architecture.lora_rank == 64, "Wan LoRA rank must be 64", errors)
    _require(architecture.lora_layers == tuple(range(30)), "Wan LoRA must cover all 30 blocks", errors)
    _require(architecture.hidden_state_layers == (8, 16, 24), "action queries must consume Wan layers 8,16,24", errors)
    _require(architecture.adapter_layers == architecture.hidden_state_layers, "adapter and hidden-state layers must align", errors)
    _require(architecture.action_query_count == 64, "model5 must preserve 64 VLA action queries", errors)
    _require(architecture.action_query_hidden_dim == 512, "action-query width must be 512", errors)
    _require(architecture.action_query_heads == 8, "action-query attention must use 8 heads", errors)
    _require(architecture.action_query_bridge_depth > 0, "action-query bridge depth must be positive", errors)
    _require(architecture.action_decoder == "vla_query_dit_flow", "action decoder must be VLA-query-conditioned flow", errors)
    _require(architecture.action_dit_layers == 16, "action DiT must have 16 layers", errors)
    _require(architecture.action_dit_hidden_dim == 512, "action DiT width must be 512", errors)
    _require(architecture.future_video_flow_loss, "future video flow loss must be enabled", errors)
    _require(architecture.action_flow_loss, "action flow loss must be enabled", errors)
    _require(
        not architecture.uses_privileged_future_latent_as_input,
        "expert future latent cannot be an online policy input",
        errors,
    )
    _require(
        architecture.action_feature_temporal_scope
        in {"current_only", "current_plus_noisy_future"},
        "invalid action feature temporal scope",
        errors,
    )
    _require(
        architecture.fixed_feature_timestep == 1000,
        "model5 fixed feature timestep must be 1000",
        errors,
    )
    _require(
        architecture.future_feature_latent_slots == 8,
        "model5 must configure eight future feature latent slots",
        errors,
    )
    _require(
        architecture.action_feature_spatial_downsample_factor in {1, 2},
        "model5 action-feature downsample factor must be 1 or 2",
        errors,
    )

    _require(training.gpu_ids == (0, 1, 2, 3), "formal training must use GPUs 0,1,2,3", errors)
    _require(training.num_processes == len(training.gpu_ids), "one process per configured GPU is required", errors)
    _require(training.max_steps >= 60_000, "formal training must run for at least 60000 steps", errors)
    _require(training.batch_size > 0 and training.gradient_accumulation_steps > 0, "invalid effective batch", errors)
    _require(training.mixed_precision == "bf16", "formal training must use BF16", errors)
    _require(training.seed == 42, "formal training seed must be 42", errors)
    _require(training.max_grad_norm == 1.0, "max gradient norm must be 1.0", errors)
    _require(training.weight_decay == 0.01, "AdamW weight decay must be 0.01", errors)
    _require(training.lr_scheduler_type == "cosine", "learning-rate schedule must be cosine", errors)
    _require(
        evaluation.suite in {"libero_spatial", "libero_10"},
        "model5 formal suite must be libero_spatial or libero_10",
        errors,
    )
    _require(evaluation.tasks == 10, "evaluation must cover all 10 suite tasks", errors)
    _require(evaluation.trials_per_task == 50, "primary evaluation needs 50 trials per task", errors)
    _require(evaluation.min_success_rate >= 0.90, "success threshold cannot be below 90%", errors)

    effective_global_batch = (
        training.batch_size * training.gradient_accumulation_steps * training.num_processes
    )
    _require(effective_global_batch == 64, "effective global batch must be 64", errors)

    if evaluation.suite == "libero_spatial":
        _require(
            data.dataset_dir.name == "libero_spatial_no_noops_lerobot",
            "Spatial config must use the no-noops Spatial dataset",
            errors,
        )
        _require(training.max_steps == 60_000, "Spatial training budget must be 60000 steps", errors)
        _require(training.save_every == 5_000, "Spatial checkpoint cadence must be 5000 steps", errors)
        _require(training.warmup_steps == 1_000, "Spatial warmup must be 1000 steps", errors)
        _require(training.learning_rate == 2e-4, "Spatial learning rate must be 2e-4", errors)
        if architecture.action_feature_spatial_downsample_factor == 1:
            _require(
                training.gradient_checkpointing,
                "high-resolution Spatial Model5 requires gradient checkpointing",
                errors,
            )
    elif evaluation.suite == "libero_10":
        _require(
            data.dataset_dir.name == "libero_10_no_noops_lerobot",
            "Long config must use the no-noops LIBERO-10 dataset",
            errors,
        )
        _require(
            data.latent_cache_dir.name == "libero_10_2cam224",
            "Long config must use the registered dual-camera latent cache",
            errors,
        )
        _require(
            architecture.action_feature_temporal_scope == "current_plus_noisy_future",
            "formal Long training must use the Model5 temporal treatment",
            errors,
        )
        _require(
            architecture.action_feature_spatial_downsample_factor == 1,
            "formal Long training must preserve high-resolution action features",
            errors,
        )
        _require(
            (training.batch_size, training.gradient_accumulation_steps) == (8, 2),
            "Long Model5 profile must be B8/GA2",
            errors,
        )
        _require(training.gradient_checkpointing, "Long Model5 requires gradient checkpointing", errors)
        _require(training.max_steps == 150_000, "Long training budget must be 150000 steps", errors)
        _require(training.save_every == 5_000, "Long checkpoint cadence must be 5000 steps", errors)
        _require(training.warmup_steps == 1_000, "Long warmup must be 1000 steps", errors)
        _require(training.learning_rate == 1e-4, "Long learning rate must be 1e-4", errors)

    checked_paths: list[Path] = []
    if check_paths:
        checked_paths = [
            config.backend.repo,
            config.backend.train_script,
            config.backend.checkpoint_root,
            config.backend.hf_datasets_cache,
            config.data.dataset_dir,
            config.data.latent_cache_dir,
            config.data.text_embedding_cache_dir,
        ]
        for path in checked_paths:
            _require(path.exists(), f"required path does not exist: {path}", errors)
        if evaluation.suite == "libero_10" and config.data.latent_cache_dir.exists():
            _require(
                (config.data.latent_cache_dir / "index.pt").is_file(),
                "Long latent cache is missing index.pt",
                errors,
            )
            _require(
                (config.data.latent_cache_dir / "meta.json").is_file(),
                "Long latent cache is missing meta.json",
                errors,
            )
            manifests = list((config.data.latent_cache_dir / "manifests").glob("rank*.pt"))
            _require(len(manifests) == 4, "Long latent cache must retain four rank manifests", errors)

    if errors:
        raise ContractError("Model5 contract validation failed:\n- " + "\n- ".join(errors))
    return {
        "passed": True,
        "track_id": config.track_id,
        "checked_paths": [str(path) for path in checked_paths],
        "architecture": asdict(architecture),
        "data": asdict(data),
        "training": asdict(training),
        "evaluation": asdict(evaluation),
    }
