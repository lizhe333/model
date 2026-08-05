"""Hydra factory for the response-prewarmed Dynamic O2 treatment."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from .models import Model3O2DynamicWAM


def _as_dict(name: str, value: Any, *, required: bool = False) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required for model3_o2_dynamic")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return dict(value)


def create_model3_o2_dynamic_wam(
    *,
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    action_query_policy_config,
    model3_warmstart_path: str,
    model3_warmstart_sha256: str,
    model3_warmstart_step: int,
    response_adapter_config=None,
    response_adapter_export_path: str = "",
    response_adapter_export_sha256: str = "",
    dynamic_response_schedule=None,
    video_backbone_type: str = "wan2_2_ti2v",
    video_backbone_name: str | None = None,
    video_latent_spatial_downsample_factor: int = 1,
    apply_video_latent_downsample_to_action_branch: bool = False,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    wam_adapter=None,
    state_fusion_action_expert_config=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Model3O2DynamicWAM:
    video_dit = _as_dict("video_dit_config", video_dit_config, required=True)
    action_dit = _as_dict("action_dit_config", action_dit_config)
    query_policy = _as_dict(
        "action_query_policy_config", action_query_policy_config, required=True
    )
    video_schedule = _as_dict("video_scheduler", video_scheduler)
    action_schedule = _as_dict("action_scheduler", action_scheduler, required=True)
    loss_config = _as_dict("loss", loss)
    adapter_config = _as_dict("wam_adapter", wam_adapter)
    legacy_state_fusion = _as_dict(
        "state_fusion_action_expert_config", state_fusion_action_expert_config
    )
    if legacy_state_fusion:
        raise ValueError("Dynamic O2 cannot consume a StateFusion action-expert config")
    if not model3_warmstart_path or not model3_warmstart_sha256:
        raise ValueError("Dynamic O2 requires the pinned Model3 warm start")

    required_action_schedule = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_schedule = required_action_schedule - set(action_schedule)
    if missing_schedule:
        raise ValueError(f"`action_scheduler` is missing keys: {sorted(missing_schedule)}")
    temporal_weighting = _as_dict(
        "loss.action_temporal_weighting",
        loss_config.get("action_temporal_weighting"),
    )

    response_config = _as_dict("response_adapter_config", response_adapter_config)
    schedule_config = _as_dict("dynamic_response_schedule", dynamic_response_schedule)
    if schedule_config:
        required_schedule_keys = {
            "freeze_through_step",
            "first_adapter_update_step",
            "adapter_lr_scale",
            "gate_freeze_through_step",
            "first_gate_update_step",
            "gate_lr_scale",
        }
        if set(schedule_config) != required_schedule_keys:
            raise ValueError(
                "Dynamic response schedule must declare both the adapter and O2-gate transitions"
            )
        normalized_schedule = {
            "freeze_through_step": int(schedule_config["freeze_through_step"]),
            "first_adapter_update_step": int(schedule_config["first_adapter_update_step"]),
            "adapter_lr_scale": float(schedule_config["adapter_lr_scale"]),
            "gate_freeze_through_step": int(schedule_config["gate_freeze_through_step"]),
            "first_gate_update_step": int(schedule_config["first_gate_update_step"]),
            "gate_lr_scale": float(schedule_config["gate_lr_scale"]),
        }
        if (
            normalized_schedule["freeze_through_step"] != 5000
            or normalized_schedule["first_adapter_update_step"] != 5001
            or normalized_schedule["adapter_lr_scale"] != 0.1
            or normalized_schedule["gate_lr_scale"] != 1.0
            or (
                normalized_schedule["gate_freeze_through_step"],
                normalized_schedule["first_gate_update_step"],
            )
                not in {(0, 1), (30000, 30001)}
        ):
            raise ValueError(
                "Dynamic response schedule differs from the frozen adapter/O2-gate contract"
            )
    model = Model3O2DynamicWAM.from_wan22_pretrained(
        action_query_policy_config=query_policy,
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        video_backbone_type=video_backbone_type,
        video_backbone_name=video_backbone_name,
        video_latent_spatial_downsample_factor=int(video_latent_spatial_downsample_factor),
        apply_video_latent_downsample_to_action_branch=bool(
            apply_video_latent_downsample_to_action_branch
        ),
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit,
        action_dit_config=action_dit,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_schedule.get("train_shift", 5.0)),
        video_infer_shift=float(video_schedule.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_schedule.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_schedule["train_shift"]),
        action_infer_shift=float(action_schedule["infer_shift"]),
        action_num_train_timesteps=int(action_schedule["num_train_timesteps"]),
        loss_lambda_video=float(loss_config.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss_config.get("lambda_action", 1.0)),
        use_first_frame_residual_video_target=bool(
            loss_config.get("use_first_frame_residual_video_target", False)
        ),
        action_temporal_weighting_enabled=bool(temporal_weighting.get("enabled", False)),
        action_temporal_weighting_num_prefix_steps=temporal_weighting.get("num_prefix_steps"),
        action_temporal_weighting_prefix_weight=float(
            temporal_weighting.get("prefix_weight", 1.0)
        ),
        action_temporal_weighting_tail_weight=float(
            temporal_weighting.get("tail_weight", 1.0)
        ),
        wam_adapter=adapter_config,
        response_adapter_config=response_config,
    )
    model.load_model3_warmstart(
        model3_warmstart_path,
        expected_sha256=model3_warmstart_sha256,
        expected_step=int(model3_warmstart_step),
    )
    # This hash captures the exact inherited Model3 tensors plus the freshly
    # initialized exact-q3 O2 gate.  Stage 1 must not change any of them.
    model.o2_gate_initialization_sha256 = model.original_o2_tensor_sha256()
    if response_adapter_export_path:
        export = model.load_response_adapter_export(
            response_adapter_export_path,
            expected_sha256=(response_adapter_export_sha256 or None),
        )
        source = export.get("source_identity")
        if not isinstance(source, dict):
            raise ValueError("Stage 1 adapter export lacks source_identity")
        if source.get("model3_warmstart_sha256") != model.model3_warmstart_identity["sha256"]:
            raise ValueError("Stage 1 adapter export was not built from the pinned Model3 parent")
        if source.get("original_o2_tensor_sha256") != model.o2_gate_initialization_sha256:
            raise ValueError("Stage 1 adapter export O2 carrier identity does not match Stage 2")
    # The Dynamic trainer places these parameters in dedicated optimizer groups
    # before freezing A through step 5K. The Long gate is activated on the first
    # Stage 2 forward; Object retains its separately registered late boundary.
    model.set_response_adapters_trainable(False)
    model.set_o2_gate_trainable(False)
    return model


# Keeping a named compatibility alias prevents old copied scripts from silently
# instantiating Model3O2WAM.  All Dynamic Hydra configs use the explicit target.
create_model3_o2_wam = create_model3_o2_dynamic_wam
