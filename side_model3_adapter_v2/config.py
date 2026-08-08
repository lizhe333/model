"""Typed, suite-agnostic configuration for Side-Model3-Adapter-v2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


METHOD_ID = "side_model3_adapter_v2_direct_wm_action_flow_v1"
HYDRA_MODEL = "side_model3_adapter_v2"
LATENT_CACHE_FORMAT = "side_model3_independent_observation_latent_cache_v1"


@dataclass(frozen=True)
class ArchitectureConfig:
    direct_code_parent: str = "side_model3"
    parent_track: str = "model3"
    video_backbone: str = "Wan-AI/Wan2.1-T2V-1.3B"
    freeze_wan: bool = True
    wan_forward_no_grad: bool = False
    target_wan_forward_no_grad: bool = True
    use_backbone_lora: bool = False
    use_wam_adapter: bool = True
    adapter_layer_indices: tuple[int, ...] = (8, 16, 24)
    adapter_dim: int = 256
    adapter_scale: float = 1.0
    ema_target_adapters: bool = True
    write_side_state_to_wan: bool = False
    current_only_wan: bool = True
    hidden_state_layers: tuple[int, ...] = (8, 16, 20, 24, 29)
    ladder_stages: int = 5
    slot_count: int = 64
    hidden_dim: int = 512
    attention_heads: int = 8
    ffn_dim: int = 2048
    ladder_residual_gate_init: float = 0.1
    trace_fusion: str = "final_identity_gated_early_residual"
    visual_anchor_count: int = 16
    action_decoder: str = "model3_action_dit_flow"
    action_dit_layers: int = 16
    action_horizon: int = 8


@dataclass(frozen=True)
class DataConfig:
    raw_video_required: bool = False
    latent_cache_required: bool = True
    latent_cache_format: str = LATENT_CACHE_FORMAT
    independent_single_frame_encoding: bool = True
    use_joint_video_latent_cache: bool = False
    camera_keys: tuple[str, ...] = ("image", "wrist_image")
    camera_resolution: tuple[int, int] = (224, 224)
    concat_multi_camera: str = "horizontal"
    sampled_video_positions: tuple[int, ...] = (0, 1, 2)
    environment_offsets: tuple[int, ...] = (0, 4, 8)
    proprio_offsets: tuple[int, ...] = (0, 4, 8)
    action_horizon: int = 8


@dataclass(frozen=True)
class PredictiveConfig:
    horizons: tuple[int, ...] = (4, 8)
    transition_blocks: int = 2
    ema_decay: float = 0.996
    latent_pool_kernel: tuple[int, int] = (2, 2)
    latent_pool_stride: tuple[int, int] = (2, 2)


@dataclass(frozen=True)
class LossConfig:
    action: float = 1.0
    state_4: float = 0.25
    state_8: float = 0.50
    latent_4: float = 0.10
    latent_8: float = 0.20

    def weights(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class SideModel3AdapterV2Config:
    track_id: str = "side_model3_adapter_v2"
    method_id: str = METHOD_ID
    runtime_package: str = "side_model3_adapter_v2"
    hydra_model: str = HYDRA_MODEL
    architecture: ArchitectureConfig = ArchitectureConfig()
    data: DataConfig = DataConfig()
    predictive: PredictiveConfig = PredictiveConfig()
    loss: LossConfig = LossConfig()


def default_config() -> SideModel3AdapterV2Config:
    """Return the frozen method contract without selecting a dataset or suite."""

    return SideModel3AdapterV2Config()


def _tuple_values(raw: dict[str, Any], *names: str) -> dict[str, Any]:
    normalized = dict(raw)
    for name in names:
        if name in normalized:
            normalized[name] = tuple(normalized[name])
    return normalized


def config_from_dict(raw: dict[str, Any]) -> SideModel3AdapterV2Config:
    """Build a typed contract from a partial JSON-compatible mapping."""

    config = default_config()
    architecture_raw = _tuple_values(
        raw.get("architecture", {}),
        "hidden_state_layers",
        "adapter_layer_indices",
    )
    data_raw = _tuple_values(
        raw.get("data", {}),
        "camera_keys",
        "camera_resolution",
        "sampled_video_positions",
        "environment_offsets",
        "proprio_offsets",
    )
    predictive_raw = _tuple_values(
        raw.get("predictive", {}),
        "horizons",
        "latent_pool_kernel",
        "latent_pool_stride",
    )
    return replace(
        config,
        track_id=str(raw.get("track_id", config.track_id)),
        method_id=str(raw.get("method_id", config.method_id)),
        runtime_package=str(raw.get("runtime_package", config.runtime_package)),
        hydra_model=str(raw.get("hydra_model", config.hydra_model)),
        architecture=replace(config.architecture, **architecture_raw),
        data=replace(config.data, **data_raw),
        predictive=replace(config.predictive, **predictive_raw),
        loss=replace(config.loss, **raw.get("loss", {})),
    )


def load_config(path: str | Path) -> SideModel3AdapterV2Config:
    """Load an optional method-only override; suite execution is intentionally absent."""

    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Side-Model3 configuration must be a JSON object")
    return config_from_dict(raw)


def config_dict(config: SideModel3AdapterV2Config) -> dict[str, Any]:
    """Return a JSON-ready representation used by the preflight entry point."""

    return asdict(config)
