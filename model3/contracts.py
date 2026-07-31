"""Fail-fast checks for the scientific and runtime model3 contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Model3Config


class ContractError(ValueError):
    """Raised when a launch would no longer represent the model3 method."""


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _require_complete_text_cache(config: Model3Config, errors: list[str]) -> None:
    """Ensure each LIBERO task prompt has the frozen embedding used by the dataset."""

    tasks_path = config.data.dataset_dir / "meta" / "tasks.jsonl"
    cache_dir = config.data.text_embedding_cache_dir
    if not tasks_path.exists():
        _require(False, f"missing LIBERO tasks metadata: {tasks_path}", errors)
        return

    expected_filenames: set[str] = set()
    with tasks_path.open(encoding="utf-8") as tasks_file:
        for line_number, line in enumerate(tasks_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "task" not in record:
                _require(False, f"missing task text at {tasks_path}:{line_number}", errors)
                continue
            prompt = "A video recorded from a robot's point of view executing the following instruction: {task}".format(
                task=str(record["task"])
            )
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            expected_filenames.add(f"{prompt_hash}.t5_len128.wan21t2v13b.pt")

    missing = sorted(filename for filename in expected_filenames if not (cache_dir / filename).is_file())
    _require(
        not missing,
        f"text embedding cache is incomplete for {config.evaluation.suite}: "
        f"{len(missing)} missing (for example {missing[0] if missing else 'n/a'})",
        errors,
    )


def validate_contract(config: Model3Config, *, check_paths: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    architecture = config.architecture
    data = config.data
    training = config.training
    evaluation = config.evaluation

    _require(config.track_id == "model3", "track_id must be model3", errors)
    _require(data.camera_keys == ("image", "wrist_image"), "dual camera order must be image,wrist_image", errors)
    _require(data.camera_resolution == (224, 224), "each camera must be 224x224", errors)
    _require(data.concat_multi_camera == "horizontal", "camera concatenation must be horizontal", errors)
    _require(data.num_frames == 33, "video/action window must contain 33 environment frames", errors)
    _require(data.action_video_freq_ratio == 4, "action/video frequency ratio must be 4", errors)
    _require(data.video_action_horizon == 32, "the future-video window must retain 32 action transitions", errors)
    _require(data.policy_action_horizon == 8, "model3 must preserve model2's 8-step action chunk", errors)

    _require(architecture.parent_track == "model2", "model3 parent_track must be model2", errors)
    _require(
        architecture.design_lineage == "vla_adapter_action_query",
        "model3 must declare the VLA-Adapter action-query lineage",
        errors,
    )
    _require(
        architecture.conditioner_type == "layerwise_recurrent_action_query",
        "model3 must use the recurrent layer-wise action-query conditioner",
        errors,
    )
    _require(not architecture.uses_state_fusion, "StateFusion is not the model3 conditioner", errors)
    _require(architecture.video_backbone == "Wan-AI/Wan2.1-T2V-1.3B", "unexpected Video-DiT backbone", errors)
    _require(architecture.freeze_backbone, "base Wan weights must stay frozen", errors)
    _require(architecture.use_backbone_lora, "Wan LoRA must be enabled", errors)
    _require(architecture.lora_rank == 64, "Wan LoRA rank must be 64", errors)
    _require(architecture.lora_layers == tuple(range(30)), "Wan LoRA must cover all 30 blocks", errors)
    _require(architecture.hidden_state_layers == (8, 16, 24), "action queries must consume Wan layers 8,16,24", errors)
    _require(architecture.adapter_layers == architecture.hidden_state_layers, "adapter and hidden-state layers must align", errors)
    _require(architecture.action_query_count == 64, "model3 must preserve 64 VLA action queries", errors)
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

    _require(training.gpu_ids == (0, 1, 2, 3), "formal training must use GPUs 0,1,2,3", errors)
    _require(training.num_processes == len(training.gpu_ids), "one process per configured GPU is required", errors)
    _require(training.batch_size > 0 and training.gradient_accumulation_steps > 0, "invalid effective batch", errors)
    _require(training.mixed_precision == "bf16", "formal training must use BF16", errors)
    _require(training.seed == 42, "formal training seed must be 42", errors)
    _require(training.max_grad_norm == 1.0, "max gradient norm must be 1.0", errors)
    _require(training.weight_decay == 0.01, "AdamW weight decay must be 0.01", errors)
    _require(training.lr_scheduler_type == "cosine", "learning-rate schedule must be cosine", errors)
    _require(
        evaluation.suite in {"libero_spatial", "libero_object", "libero_goal", "libero_10"},
        "unsupported LIBERO suite",
        errors,
    )
    _require(evaluation.tasks == 10, "evaluation must cover all 10 suite tasks", errors)
    _require(evaluation.trials_per_task == 50, "formal evaluation needs 50 trials per task", errors)
    _require(evaluation.min_success_rate >= 0.90, "success threshold cannot be below 90%", errors)

    if evaluation.suite == "libero_spatial":
        _require(
            data.dataset_dir.name == "libero_spatial_no_noops_lerobot",
            "Spatial config must use the no-noops Spatial dataset",
            errors,
        )
        _require(data.use_latent_cache, "Spatial training must use its registered latent cache", errors)
        _require(data.latent_cache_dir is not None, "Spatial latent cache path is required", errors)
        _require(training.max_steps >= 60_000, "Spatial training must run for at least 60000 steps", errors)
    elif evaluation.suite in {"libero_object", "libero_goal", "libero_10"}:
        effective_global_batch = (
            training.batch_size * training.gradient_accumulation_steps * training.num_processes
        )
        expected_dataset_name = {
            "libero_object": "libero_object_no_noops_lerobot",
            "libero_goal": "libero_goal_no_noops_lerobot",
            "libero_10": "libero_10_no_noops_lerobot",
        }[evaluation.suite]
        _require(data.dataset_dir.name == expected_dataset_name, "unexpected no-noops LIBERO dataset", errors)
        if evaluation.suite in {"libero_object", "libero_goal"}:
            _require(data.use_latent_cache, "Object and Goal training must use their registered latent caches", errors)
            _require(data.latent_cache_dir is not None, "cached training requires a latent cache path", errors)
        elif data.use_latent_cache:
            _require(data.latent_cache_dir is not None, "cached Long training requires a latent cache path", errors)
        else:
            _require(data.latent_cache_dir is None, "online-VAE Long latent cache path must be null", errors)
        _require(
            (training.batch_size, training.gradient_accumulation_steps) in {(8, 2), (16, 1)},
            "batch profile must be reference B8/GA2 or fast B16/GA1",
            errors,
        )
        _require(effective_global_batch == 64, "effective global batch must be 64", errors)
        if evaluation.suite in {"libero_object", "libero_goal"}:
            _require(training.num_workers == 16, "Object and Goal training must use 16 data-loader workers", errors)
            _require_complete_text_cache(config, errors)

        expected_training = {
            "libero_object": (150_000, 5_000, 1_000, 1e-4),
            "libero_goal": (150_000, 5_000, 1_000, 2e-4),
            "libero_10": (80_000, 5_000, 1_000, 1e-4),
        }[evaluation.suite]
        expected_steps, expected_save_every, expected_warmup, expected_lr = expected_training
        _require(training.max_steps == expected_steps, f"{evaluation.suite} training budget is incorrect", errors)
        _require(training.save_every == expected_save_every, f"{evaluation.suite} checkpoint cadence is incorrect", errors)
        _require(training.warmup_steps == expected_warmup, f"{evaluation.suite} warmup is incorrect", errors)
        _require(training.learning_rate == expected_lr, f"{evaluation.suite} learning rate is incorrect", errors)

    checked_paths: list[Path] = []
    if check_paths:
        checked_paths = [
            config.backend.repo,
            config.backend.train_script,
            config.backend.checkpoint_root,
            config.backend.hf_datasets_cache,
            config.data.dataset_dir,
            config.data.text_embedding_cache_dir,
        ]
        if config.data.use_latent_cache and config.data.latent_cache_dir is not None:
            checked_paths.append(config.data.latent_cache_dir)
        for path in checked_paths:
            _require(path.exists(), f"required path does not exist: {path}", errors)

    if errors:
        raise ContractError("Model3 contract validation failed:\n- " + "\n- ".join(errors))
    return {
        "passed": True,
        "track_id": config.track_id,
        "checked_paths": [str(path) for path in checked_paths],
        "architecture": asdict(architecture),
        "data": asdict(data),
        "training": asdict(training),
        "evaluation": asdict(evaluation),
    }
