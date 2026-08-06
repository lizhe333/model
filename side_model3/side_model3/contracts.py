"""Fail-fast checks for the frozen Side-Model3 v1 method contract."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from .config import METHOD_ID, SideModel3Config, default_config


class ContractError(ValueError):
    """Raised when a configuration or live model is not Side-Model3 v1."""


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _all_frozen(module: Any) -> bool:
    return all(not parameter.requires_grad for parameter in module.parameters())


def validate_training_data_config(data_cfg: Any) -> dict[str, Any]:
    """Validate the sampled-frame-to-environment-offset contract."""

    errors: list[str] = []
    _require(int(data_cfg.get("num_frames", -1)) == 33, "num_frames must be 33", errors)
    _require(
        int(data_cfg.get("global_sample_stride", -1)) == 1,
        "global_sample_stride must be 1",
        errors,
    )
    _require(
        int(data_cfg.get("action_video_freq_ratio", -1)) == 4,
        "action_video_freq_ratio must be 4",
        errors,
    )
    _require(
        not bool(data_cfg.get("use_latent_cache", False)),
        "Side-Model3 training requires raw video instead of latent cache",
        errors,
    )
    _require(
        tuple(data_cfg.get("video_size", ())) == (224, 448),
        "video_size must be 224 by 448",
        errors,
    )
    _require(
        data_cfg.get("concat_multi_camera") == "horizontal",
        "dual cameras must be concatenated horizontally",
        errors,
    )
    if errors:
        raise ContractError(
            "Side-Model3 training data contract failed:\n- " + "\n- ".join(errors)
        )
    return {
        "passed": True,
        "num_frames": 33,
        "global_sample_stride": 1,
        "action_video_freq_ratio": 4,
        "environment_offsets": [0, 4, 8],
        "sampled_video_positions": [0, 1, 2],
    }


def _validate_live_model(model: Any, config: SideModel3Config, errors: list[str]) -> None:
    architecture = config.architecture
    _require(type(model).__name__ == "SideModel3WAM", "wrong live model class", errors)
    _require(getattr(model, "method_id", None) == METHOD_ID, "wrong live method id", errors)
    _require(
        tuple(getattr(model, "selected_wan_layers", ()))
        == architecture.hidden_state_layers,
        "live Wan layer selection differs from the contract",
        errors,
    )

    video_expert = getattr(model, "video_expert", None)
    _require(video_expert is not None, "live model has no Wan video expert", errors)
    if video_expert is not None:
        _require(
            not bool(getattr(video_expert, "use_wam_adapter", False)),
            "live Wan contains a residual adapter",
            errors,
        )
        has_lora = getattr(video_expert, "has_backbone_lora", None)
        _require(
            not bool(has_lora() if callable(has_lora) else False),
            "live Wan contains LoRA",
            errors,
        )
        _require(_all_frozen(video_expert), "live Wan has trainable parameters", errors)

    vae = getattr(model, "vae", None)
    _require(vae is not None and _all_frozen(vae), "live VAE must be frozen", errors)
    _require(
        getattr(model, "state_fusion_action_expert", None) is None,
        "StateFusion is not part of Side-Model3",
        errors,
    )

    target = getattr(model, "target_predictive_encoder", None)
    _require(
        target is not None and _all_frozen(target),
        "EMA target encoder must be present and gradient-frozen",
        errors,
    )
    if target is not None:
        _require(
            math.isclose(float(getattr(target, "decay", -1.0)), config.predictive.ema_decay),
            "EMA decay differs from the contract",
            errors,
        )
        _require(
            all(
                str(parameter.dtype) == "torch.float32"
                for parameter in target.parameters()
            ),
            "EMA target parameters must accumulate in FP32",
            errors,
        )
    _require(
        getattr(model, "loss_weights", None) == config.loss.weights(),
        "live loss weights differ from the contract",
        errors,
    )


def validate_contract(
    config: SideModel3Config | None = None,
    *,
    model: Any | None = None,
) -> dict[str, Any]:
    """Validate the method without loading Wan; optionally inspect a live instance."""

    config = default_config() if config is None else config
    architecture = config.architecture
    data = config.data
    predictive = config.predictive
    errors: list[str] = []

    _require(config.track_id == "side_model3", "track_id must be side_model3", errors)
    _require(config.method_id == METHOD_ID, "method_id must identify Side-Model3 v1", errors)
    _require(config.runtime_package == "side_model3", "runtime package must be side_model3", errors)
    _require(config.hydra_model == "side_model3_v1", "wrong Hydra model name", errors)

    _require(architecture.parent_track == "model3", "parent track must be model3", errors)
    _require(
        architecture.video_backbone == "Wan-AI/Wan2.1-T2V-1.3B",
        "Side-Model3 v1 requires Wan2.1-T2V-1.3B",
        errors,
    )
    _require(architecture.freeze_wan, "Wan must be frozen", errors)
    _require(architecture.wan_forward_no_grad, "Wan forward must run without gradients", errors)
    _require(not architecture.use_backbone_lora, "Wan LoRA is forbidden", errors)
    _require(not architecture.use_wam_adapter, "Wan residual adapters are forbidden", errors)
    _require(
        not architecture.write_side_state_to_wan,
        "side states must not be written back into Wan",
        errors,
    )
    _require(architecture.current_only_wan, "Wan must receive only one clean current observation", errors)
    _require(
        architecture.hidden_state_layers == (8, 16, 20, 24, 29),
        "Wan states must come from layers 8,16,20,24,29",
        errors,
    )
    _require(architecture.ladder_stages == 5, "the Ladder encoder must have five stages", errors)
    _require(architecture.slot_count == 64, "the control state must contain 64 slots", errors)
    _require(architecture.hidden_dim == 512, "side and action widths must be 512", errors)
    _require(architecture.attention_heads == 8, "side attention must use eight heads", errors)
    _require(architecture.ffn_dim == 2048, "side FFNs must have width 2048", errors)
    _require(
        math.isclose(architecture.ladder_residual_gate_init, 0.1),
        "Ladder residual gates must initialize to 0.1",
        errors,
    )
    _require(
        architecture.trace_fusion == "final_identity_gated_early_residual",
        "trace fusion must initialize as the final Ladder state identity",
        errors,
    )
    _require(architecture.visual_anchor_count == 16, "the visual route must use 16 anchors", errors)
    _require(
        architecture.action_decoder == "model3_action_dit_flow",
        "the decoder must preserve the Model3 Action-DiT flow path",
        errors,
    )
    _require(architecture.action_dit_layers == 16, "Action-DiT must retain 16 layers", errors)
    _require(architecture.action_horizon == 8, "Action-DiT must predict eight steps", errors)

    _require(data.raw_video_required, "training requires raw processed video", errors)
    _require(
        not data.use_joint_video_latent_cache,
        "the joint-video latent cache is incompatible with independent observations",
        errors,
    )
    _require(data.camera_keys == ("image", "wrist_image"), "dual-camera order changed", errors)
    _require(data.camera_resolution == (224, 224), "each camera must be 224 by 224", errors)
    _require(data.concat_multi_camera == "horizontal", "camera concatenation must be horizontal", errors)
    _require(data.sampled_video_positions == (0, 1, 2), "the first three sampled frames are required", errors)
    _require(data.environment_offsets == (0, 4, 8), "RGB targets must use offsets 0,4,8", errors)
    _require(data.proprio_offsets == (0, 4, 8), "proprioception must use offsets 0,4,8", errors)
    _require(data.action_horizon == 8, "the action target must contain eight steps", errors)

    _require(predictive.horizons == (4, 8), "predictive horizons must be 4 and 8", errors)
    _require(predictive.transition_blocks == 2, "the transition predictor must use two blocks", errors)
    _require(math.isclose(predictive.ema_decay, 0.996), "EMA decay must be 0.996", errors)
    _require(predictive.latent_pool_kernel == (2, 2), "latent pooling kernel must be 2 by 2", errors)
    _require(predictive.latent_pool_stride == (2, 2), "latent pooling stride must be 2 by 2", errors)
    _require(
        config.loss.weights()
        == {
            "action": 1.0,
            "state_4": 0.25,
            "state_8": 0.5,
            "latent_4": 0.1,
            "latent_8": 0.2,
        },
        "loss weights differ from the frozen five-loss objective",
        errors,
    )

    if model is not None:
        _validate_live_model(model, config, errors)
    if errors:
        raise ContractError("Side-Model3 contract validation failed:\n- " + "\n- ".join(errors))

    return {
        "passed": True,
        "track_id": config.track_id,
        "method_id": config.method_id,
        "model_checked": model is not None,
        "architecture": asdict(architecture),
        "data": asdict(data),
        "predictive": asdict(predictive),
        "loss_weights": config.loss.weights(),
        "online_shapes": {
            "rgb": ["B", 3, 1, 224, 448],
            "ladder_trace": ["B", 5, 64, 512],
            "control_state": ["B", 64, 512],
            "action": ["B", 8, 7],
        },
    }
