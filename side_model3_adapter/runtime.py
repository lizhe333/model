"""Hydra factory copied from Side-Model3 and changed only for Wan adapters."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from .models import SideModel3AdapterWAM


def _as_dict(name: str, value: Any, *, required: bool = False) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required for Side-Model3-Adapter")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return dict(value)


def _require_value(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def create_side_model3_adapter_wam(
    *,
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    side_encoder_config,
    action_policy_config,
    visual_anchor_config,
    transition_config,
    latent_change_config,
    video_backbone_type: str = "wan2_1_t2v",
    video_backbone_name: str | None = "Wan-AI/Wan2.1-T2V-1.3B",
    video_latent_spatial_downsample_factor: int = 1,
    apply_video_latent_downsample_to_action_branch: bool = False,
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    model3_action_dit_warmstart_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    ema_decay: float = 0.996,
    wam_adapter=None,
    state_fusion_action_expert_config=None,
    mot_checkpoint_mixed_attn: bool = False,
    redirect_common_files: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> SideModel3AdapterWAM:
    """Construct the direct Side-Model3 copy with exactly three Wan adapters."""

    video_dit = _as_dict("video_dit_config", video_dit_config, required=True)
    action_dit = _as_dict("action_dit_config", action_dit_config)
    side_encoder = _as_dict("side_encoder_config", side_encoder_config, required=True)
    action_policy = _as_dict("action_policy_config", action_policy_config, required=True)
    visual_anchor = _as_dict("visual_anchor_config", visual_anchor_config, required=True)
    transition = _as_dict("transition_config", transition_config, required=True)
    latent_change = _as_dict("latent_change_config", latent_change_config, required=True)
    action_schedule = _as_dict("action_scheduler", action_scheduler, required=True)
    _as_dict("video_scheduler", video_scheduler)
    loss_config = _as_dict("loss", loss, required=True)
    adapter_config = _as_dict("wam_adapter", wam_adapter, required=True)
    state_fusion = _as_dict(
        "state_fusion_action_expert_config",
        state_fusion_action_expert_config,
    )

    if state_fusion:
        raise ValueError("Side-Model3-Adapter cannot instantiate StateFusion")
    if not bool(adapter_config.get("use_wam_adapter", False)):
        raise ValueError("Side-Model3-Adapter requires Wan residual adapters")
    if bool(adapter_config.get("use_backbone_lora", False)):
        raise ValueError("Side-Model3-Adapter cannot instantiate Wan LoRA")
    _require_value(
        tuple(int(value) for value in adapter_config.get("adapter_layer_indices", ()))
        == (8, 16, 24),
        "Side-Model3-Adapter requires adapter layers 8,16,24",
    )
    _require_value(
        int(adapter_config.get("adapter_dim", -1)) == 256,
        "Side-Model3-Adapter requires adapter_dim=256",
    )
    _require_value(
        float(adapter_config.get("adapter_scale", -1.0)) == 1.0,
        "Side-Model3-Adapter requires adapter_scale=1.0",
    )
    _require_value(
        bool(adapter_config.get("freeze_backbone", False)),
        "Side-Model3-Adapter freezes original Wan parameters",
    )
    _require_value(
        not bool(adapter_config.get("remove_original_action_expert", False)),
        "Side-Model3-Adapter uses its copied Action-DiT instead of StateFusion mode",
    )
    if bool(video_dit.get("use_backbone_lora", False)):
        raise ValueError("video_dit_config cannot enable Wan LoRA")
    _require_value(
        video_dit.get("video_attention_mask_mode") == "first_frame_causal",
        "Side-Model3-Adapter v1 requires first_frame_causal Wan attention",
    )

    _require_value(model_id == "Wan-AI/Wan2.1-T2V-1.3B", "Side-Model3-Adapter v1 requires Wan2.1-T2V-1.3B")
    _require_value(video_backbone_type == "wan2_1_t2v", "Side-Model3-Adapter v1 requires the wan2_1_t2v backend")
    _require_value(
        video_backbone_name in {None, "Wan-AI/Wan2.1-T2V-1.3B"},
        "unexpected Side-Model3 video backbone",
    )
    _require_value(
        tokenizer_model_id == "Wan-AI/Wan2.1-T2V-1.3B",
        "Side-Model3 tokenizer must match the frozen Wan",
    )
    _require_value(
        int(video_latent_spatial_downsample_factor) == 1,
        "Side-Model3 keeps the full frozen-Wan latent grid",
    )
    _require_value(
        not apply_video_latent_downsample_to_action_branch,
        "Side-Model3 does not downsample the action-side Wan states",
    )
    _require_value(not mot_checkpoint_mixed_attn, "frozen Wan must not use gradient checkpointing")
    _require_value(proprio_dim is not None and int(proprio_dim) > 0, "proprio_dim is required")
    if action_dit_pretrained_path:
        raise ValueError(
            "use model3_action_dit_warmstart_path for the optional Action-DiT-only warm start"
        )

    required_schedule = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_schedule = required_schedule - set(action_schedule)
    _require_value(
        not missing_schedule,
        f"`action_scheduler` is missing keys: {sorted(missing_schedule)}",
    )
    _require_value(
        tuple(int(value) for value in side_encoder.get("layer_indices", ()))
        == (8, 16, 20, 24, 29),
        "Side-Model3-Adapter v1 requires Wan layers 8,16,20,24,29",
    )
    _require_value(int(side_encoder.get("num_slots", -1)) == 64, "Side-Model3-Adapter requires 64 control slots")
    _require_value(int(side_encoder.get("hidden_dim", -1)) == 512, "Side-Model3 side width must be 512")
    _require_value(int(side_encoder.get("num_heads", -1)) == 8, "Side-Model3 side attention must use eight heads")
    _require_value(int(side_encoder.get("ffn_dim", -1)) == 2048, "Side-Model3 side FFN width must be 2048")
    _require_value(
        float(side_encoder.get("residual_gate_init", -1.0)) == 0.1,
        "Ladder residual gates must initialize to 0.1",
    )
    _require_value(int(action_policy.get("hidden_dim", -1)) == 512, "Action-DiT width must be 512")
    _require_value(int(action_policy.get("ffn_dim", -1)) == 2048, "Action-DiT FFN width must be 2048")
    _require_value(int(action_policy.get("num_layers", -1)) == 16, "Action-DiT must have 16 layers")
    _require_value(int(action_policy.get("num_heads", -1)) == 8, "Action-DiT must use eight heads")
    _require_value(int(action_policy.get("attn_head_dim", -1)) == 64, "Action-DiT head width must be 64")
    _require_value(int(action_policy.get("action_horizon", -1)) == 8, "Action-DiT horizon must be eight")
    _require_value(
        not bool(action_policy.get("use_gradient_checkpointing", False)),
        "Side-Model3-Adapter v1 keeps Action-DiT checkpointing disabled",
    )
    _require_value(int(visual_anchor.get("num_anchors", -1)) == 16, "the visual route requires 16 anchors")
    _require_value(int(visual_anchor.get("num_heads", -1)) == 8, "the visual route requires eight heads")
    _require_value(int(visual_anchor.get("ffn_dim", -1)) == 2048, "the visual FFN width must be 2048")
    _require_value(int(transition.get("num_blocks", -1)) == 2, "the transition predictor requires two blocks")
    _require_value(int(transition.get("num_heads", -1)) == 8, "the transition predictor requires eight heads")
    _require_value(int(transition.get("ffn_dim", -1)) == 2048, "the transition FFN width must be 2048")
    _require_value(
        (int(latent_change.get("grid_height", 0)), int(latent_change.get("grid_width", 0)))
        == (14, 28),
        "the pooled 224x448 VAE latent grid must be 14x28",
    )
    _require_value(int(latent_change.get("num_heads", -1)) == 8, "the latent head requires eight heads")
    _require_value(int(latent_change.get("ffn_dim", -1)) == 2048, "the latent-head FFN width must be 2048")
    _require_value(float(action_schedule["train_shift"]) == 5.0, "action train shift must be 5.0")
    _require_value(float(action_schedule["infer_shift"]) == 5.0, "action infer shift must be 5.0")
    _require_value(
        int(action_schedule["num_train_timesteps"]) == 1000,
        "action scheduler must use 1000 training timesteps",
    )

    loss_weights = {
        "action": float(loss_config.get("action", -1.0)),
        "state_4": float(loss_config.get("state_4", -1.0)),
        "state_8": float(loss_config.get("state_8", -1.0)),
        "latent_4": float(loss_config.get("latent_4", -1.0)),
        "latent_8": float(loss_config.get("latent_8", -1.0)),
    }
    _require_value(
        loss_weights
        == {
            "action": 1.0,
            "state_4": 0.25,
            "state_8": 0.5,
            "latent_4": 0.1,
            "latent_8": 0.2,
        },
        "Side-Model3-Adapter requires the copied five-loss weights",
    )
    _require_value(float(ema_decay) == 0.996, "Side-Model3 EMA decay must be 0.996")
    _require_value(
        not bool(loss_config.get("future_video_flow_loss", False))
        and float(loss_config.get("lambda_video", 0.0)) == 0.0,
        "Side-Model3 cannot enable future-video flow loss",
    )

    video_dit["use_wam_adapter"] = True
    video_dit["adapter_layer_indices"] = [8, 16, 24]
    video_dit["adapter_dim"] = 256
    video_dit["adapter_scale"] = 1.0
    video_dit["use_backbone_lora"] = False
    video_dit["use_gradient_checkpointing"] = False
    model = SideModel3AdapterWAM.from_wan21_pretrained(
        side_encoder_config=side_encoder,
        action_policy_config=action_policy,
        visual_anchor_config=visual_anchor,
        transition_config=transition,
        latent_change_config=latent_change,
        wam_adapter_config=adapter_config,
        loss_weights=loss_weights,
        ema_decay=float(ema_decay),
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        video_backbone_type=video_backbone_type,
        video_backbone_name=video_backbone_name,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit,
        action_dit_config=action_dit,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=False,
        action_train_shift=float(action_schedule["train_shift"]),
        action_infer_shift=float(action_schedule["infer_shift"]),
        action_num_train_timesteps=int(action_schedule["num_train_timesteps"]),
    )
    if model3_action_dit_warmstart_path:
        model.load_model3_action_dit_warmstart(model3_action_dit_warmstart_path)
    return model


def create_side_model3_adapter_trainer(*, model, train_dataset, val_dataset=None, cfg):
    """Create the trainer that performs EMA only after executed optimizer steps."""

    from .trainer import SideModel3AdapterTrainer

    return SideModel3AdapterTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
    )
