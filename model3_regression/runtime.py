"""Hydra factory for Model3 with a direct-regression action policy."""

from __future__ import annotations

import torch

from model3.runtime import _as_dict

from .models import Model3RegressionWAM


def create_model3_regression_wam(
    *,
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    action_query_policy_config,
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
) -> Model3RegressionWAM:
    video_dit = _as_dict("video_dit_config", video_dit_config, required=True)
    action_dit = _as_dict("action_dit_config", action_dit_config)
    query_policy = _as_dict(
        "action_query_policy_config",
        action_query_policy_config,
        required=True,
    )
    video_schedule = _as_dict("video_scheduler", video_scheduler)
    action_schedule = _as_dict("action_scheduler", action_scheduler)
    loss_config = _as_dict("loss", loss)
    adapter_config = _as_dict("wam_adapter", wam_adapter)
    legacy_state_fusion = _as_dict(
        "state_fusion_action_expert_config",
        state_fusion_action_expert_config,
    )
    if legacy_state_fusion:
        raise ValueError("Model3 Regression cannot consume a StateFusion action-expert config")

    temporal_weighting = _as_dict(
        "loss.action_temporal_weighting",
        loss_config.get("action_temporal_weighting"),
    )
    return Model3RegressionWAM.from_wan22_pretrained(
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
        action_train_shift=float(action_schedule.get("train_shift", 5.0)),
        action_infer_shift=float(action_schedule.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(action_schedule.get("num_train_timesteps", 1000)),
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
    )
