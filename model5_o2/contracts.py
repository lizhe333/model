"""Fail-fast contract for the staged Model5 O2 comparison."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import (
    STAGE1,
    STAGE2_CONTROL,
    STAGE2_O2,
    SUPPORTED_STAGE_ROLES,
    Model5O2Config,
)


class ContractError(ValueError):
    """Raised when a run no longer represents the Model5 O2 experiment."""


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(
    config: Model5O2Config,
    *,
    check_paths: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    base = config.base
    architecture = base.architecture
    data = base.data
    training = base.training
    evaluation = base.evaluation
    initialization = config.initialization

    _require(base.track_id == "model5_o2", "track_id must be model5_o2", errors)
    _require(
        config.stage_role in SUPPORTED_STAGE_ROLES,
        f"unsupported stage_role: {config.stage_role}",
        errors,
    )
    _require(architecture.parent_track == "model3", "Model5 lineage must remain model3", errors)
    _require(
        architecture.design_lineage == "asymmetric_tri_timestep_vla_query_b1",
        "Model5 temporal design lineage must remain unchanged",
        errors,
    )
    _require(
        architecture.conditioner_type == "layerwise_recurrent_action_query",
        "the recurrent Model5 query encoder must be preserved",
        errors,
    )
    _require(not architecture.uses_state_fusion, "StateFusion is not allowed", errors)
    _require(architecture.freeze_backbone, "Wan base weights must remain frozen", errors)
    _require(architecture.use_backbone_lora, "Wan LoRA must remain enabled", errors)
    _require(architecture.lora_rank == 64, "Wan LoRA rank must remain 64", errors)
    _require(
        architecture.lora_layers == tuple(range(30)),
        "Wan LoRA must remain on all 30 blocks",
        errors,
    )
    _require(
        architecture.hidden_state_layers == (8, 16, 24),
        "Model5 O2 must read Wan layers 8,16,24",
        errors,
    )
    _require(
        architecture.adapter_layers == (8, 16, 24),
        "WAM adapters must remain at layers 8,16,24",
        errors,
    )
    _require(architecture.action_query_count == 64, "query count must remain 64", errors)
    _require(
        architecture.action_query_hidden_dim == 512,
        "query width must remain 512",
        errors,
    )
    _require(
        architecture.action_decoder == "vla_query_dit_flow",
        "the flow Action-DiT must be preserved",
        errors,
    )
    _require(architecture.action_dit_layers == 16, "Action-DiT must retain 16 layers", errors)
    _require(architecture.action_dit_hidden_dim == 512, "Action-DiT width must be 512", errors)
    _require(architecture.future_video_flow_loss, "video flow loss must remain enabled", errors)
    _require(architecture.action_flow_loss, "action flow loss must remain enabled", errors)
    _require(
        not architecture.uses_privileged_future_latent_as_input,
        "expert future latents cannot enter the action-feature path",
        errors,
    )
    _require(
        architecture.action_feature_temporal_scope == "current_plus_noisy_future",
        "Model5 O2 must preserve the full noisy-future temporal scope",
        errors,
    )
    _require(
        architecture.fixed_feature_timestep == 1000,
        "fixed action-feature timestep must remain 1000",
        errors,
    )
    _require(
        architecture.future_feature_latent_slots == 8,
        "Model5 O2 must preserve all eight future feature slots",
        errors,
    )
    _require(
        architecture.action_feature_spatial_downsample_factor == 1,
        "action features must remain high resolution",
        errors,
    )

    _require(data.camera_keys == ("image", "wrist_image"), "camera order changed", errors)
    _require(data.camera_resolution == (224, 224), "camera resolution changed", errors)
    _require(data.num_frames == 33, "training window must contain 33 frames", errors)
    _require(data.action_video_freq_ratio == 4, "action/video ratio must remain 4", errors)
    _require(data.video_action_horizon == 32, "video horizon must remain 32", errors)
    _require(data.policy_action_horizon == 8, "action horizon must remain 8", errors)
    _require(evaluation.suite == "libero_10", "first Model5 O2 test must use Long", errors)
    _require(evaluation.tasks == 10, "evaluation must cover all ten Long tasks", errors)
    _require(evaluation.trials_per_task == 50, "evaluation needs 50 trials per task", errors)
    _require(evaluation.num_inference_steps == 10, "action solver must remain 10 steps", errors)
    _require(evaluation.max_episode_steps == 700, "Long episode limit must remain 700", errors)

    _require(training.gpu_ids == (0, 1, 2, 3), "formal runs use GPUs 0,1,2,3", errors)
    _require(training.num_processes == 4, "formal runs require four ranks", errors)
    _require(
        (training.batch_size, training.gradient_accumulation_steps) == (8, 2),
        "all stages must use B8/GA2",
        errors,
    )
    _require(training.num_workers == 8, "all stages must use eight workers", errors)
    _require(training.gradient_checkpointing, "Wan gradient checkpointing is required", errors)
    _require(training.mixed_precision == "bf16", "formal runs require bf16", errors)
    _require(training.seed == 42, "formal seed must be 42", errors)
    _require(training.learning_rate == 1e-4, "learning rate must be 1e-4", errors)
    _require(training.lr_scheduler_type == "cosine", "scheduler must be cosine", errors)
    _require(training.warmup_steps == 1_000, "warmup must be 1K", errors)
    _require(
        training.batch_size * training.gradient_accumulation_steps * training.num_processes == 64,
        "effective global batch must be 64",
        errors,
    )

    if config.stage_role == STAGE1:
        _require(initialization.mode == "fresh", "Stage 1 must initialize fresh", errors)
        _require(initialization.model5_checkpoint is None, "Stage 1 cannot load Model5", errors)
        _require(base.backend.hydra_model == "model5_o2_stage1_model5_query_flow", "wrong Stage 1 Hydra model", errors)
        _require(training.max_steps == 80_000, "Stage 1 budget must be 80K", errors)
        _require(training.save_every == 20_000, "Stage 1 checkpoint cadence must be 20K", errors)
        _require(config.readout is None, "Stage 1 must not instantiate O2 readout", errors)
    else:
        _require(
            initialization.mode == "model_only_warmstart",
            "Stage 2 arms require matched model-only warm starts",
            errors,
        )
        _require(initialization.model5_checkpoint is not None, "Stage 2 parent is required", errors)
        sha = initialization.model5_checkpoint_sha256 or ""
        _require(
            len(sha) == 64 and all(character in "0123456789abcdef" for character in sha),
            "Stage 2 requires a lowercase 64-character parent SHA-256",
            errors,
        )
        _require(initialization.model5_checkpoint_step == 80_000, "Stage 2 parent must be Model5-80K", errors)
        _require(training.max_steps == 10_000, "Stage 2 local budget must be 10K", errors)
        _require(training.save_every == 5_000, "Stage 2 must save local 5K/10K", errors)
        expected_hydra = {
            STAGE2_CONTROL: "model5_o2_stage2_model5_control_query_flow",
            STAGE2_O2: "model5_o2_layer_aware_temporal_query_flow",
        }[config.stage_role]
        _require(base.backend.hydra_model == expected_hydra, "wrong Stage 2 Hydra model", errors)
        if config.stage_role == STAGE2_CONTROL:
            _require(config.readout is None, "control arm cannot instantiate O2 readout", errors)
        else:
            readout = config.readout
            _require(readout is not None, "O2 treatment readout config is required", errors)
            if readout is not None:
                _require(
                    readout.query_trace_readout == "layer_separable_gated_residual",
                    "O2 readout type changed",
                    errors,
                )
                _require(readout.readout_num_layers == 3, "O2 must read q1/q2/q3", errors)
                _require(readout.readout_query_dim == 512, "O2 readout width must be 512", errors)
                _require(readout.readout_gate_type == "querywise_scalar", "O2 gate type changed", errors)
                _require(readout.readout_identity_init, "O2 must initialize as exact q3", errors)

    checked_paths: list[Path] = []
    if check_paths:
        checked_paths = [
            base.backend.repo,
            base.backend.train_script,
            base.backend.checkpoint_root,
            base.data.dataset_dir,
            base.data.latent_cache_dir,
            base.data.text_embedding_cache_dir,
        ]
        for path in checked_paths:
            _require(path.exists(), f"required path does not exist: {path}", errors)
        if config.stage_role != STAGE1 and initialization.model5_checkpoint is not None:
            _require(
                initialization.model5_checkpoint.is_file(),
                f"Model5-80K parent does not exist: {initialization.model5_checkpoint}",
                errors,
            )
            if initialization.model5_checkpoint.is_file() and initialization.model5_checkpoint_sha256:
                actual_sha = _sha256(initialization.model5_checkpoint)
                _require(
                    actual_sha == initialization.model5_checkpoint_sha256,
                    f"Model5-80K SHA mismatch: expected {initialization.model5_checkpoint_sha256}, got {actual_sha}",
                    errors,
                )

    if errors:
        raise ContractError("Model5 O2 contract validation failed:\n- " + "\n- ".join(errors))
    return {
        "passed": True,
        "track_id": base.track_id,
        "stage_role": config.stage_role,
        "checked_paths": [str(path) for path in checked_paths],
        "architecture": asdict(architecture),
        "readout": None if config.readout is None else asdict(config.readout),
        "initialization": {
            **asdict(initialization),
            "model5_checkpoint": (
                None
                if initialization.model5_checkpoint is None
                else str(initialization.model5_checkpoint)
            ),
        },
        "data": asdict(data),
        "training": asdict(training),
        "evaluation": asdict(evaluation),
    }
