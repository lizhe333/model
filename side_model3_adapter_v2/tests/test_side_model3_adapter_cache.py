"""Cached independent-observation contracts for Side-Model3-Adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from side_model3_adapter_v2.config import LATENT_CACHE_FORMAT
from side_model3_adapter_v2.contracts import ContractError, validate_training_data_config
from side_model3_adapter_v2.data import (
    validate_complete_side_observation_cache,
    validate_side_observation_cache_meta,
)
from side_model3_adapter_v2.models.side_model3_adapter_v2_wam import SideModel3AdapterV2WAM


def _valid_meta() -> dict:
    return {
        "format": LATENT_CACHE_FORMAT,
        "storage_format": "sharded_v1",
        "video_only": True,
        "side_model3": {
            "encoding": "independent_single_frame",
            "sampled_video_positions": [0, 1, 2],
            "environment_offsets": [0, 4, 8],
            "latent_layout": "C,T,H,W",
            "latent_time": 3,
        },
    }


def _write_meta(cache_dir: Path, payload: dict) -> None:
    cache_dir.mkdir()
    (cache_dir / "meta.json").write_text(json.dumps(payload), encoding="utf-8")


def test_adapter_accepts_the_shared_independent_cache_contract(tmp_path: Path) -> None:
    cache_dir = tmp_path / "side_cache"
    _write_meta(cache_dir, _valid_meta())
    torch.save(
        {
            "storage_format": "sharded_v1",
            "sample_to_shard": torch.tensor([0, 0, 1], dtype=torch.int32),
            "sample_to_offset": torch.tensor([0, 1, 0], dtype=torch.int32),
            "shard_paths": ["shards/a.pt", "shards/b.pt"],
        },
        cache_dir / "index.pt",
    )

    assert validate_side_observation_cache_meta(cache_dir)["format"] == LATENT_CACHE_FORMAT
    assert validate_complete_side_observation_cache(cache_dir, expected_samples=3)["format"] == LATENT_CACHE_FORMAT


def test_adapter_rejects_model3_joint_video_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "joint_video"
    _write_meta(cache_dir, {"format": "lightwam_video_latent_cache_sharded_v1"})

    with pytest.raises(ValueError, match="joint-video caches are incompatible"):
        validate_side_observation_cache_meta(cache_dir)


def test_adapter_training_data_requires_cached_latents() -> None:
    config = {
        "num_frames": 33,
        "global_sample_stride": 1,
        "action_video_freq_ratio": 4,
        "use_latent_cache": True,
        "latent_cache_dir": "/tmp/side-cache",
        "video_size": [224, 448],
        "concat_multi_camera": "horizontal",
    }
    assert validate_training_data_config(config)["passed"]

    config["use_latent_cache"] = False
    with pytest.raises(ContractError, match="latent cache must be enabled"):
        validate_training_data_config(config)


class _InputBuilderStub:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.target_action_offsets = (4, 8)
        self.vae_calls = 0

    def _encode_video_latents(self, video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        del tiled
        self.vae_calls += 1
        return video[:, :2] + self.vae_calls * 10.0


def _sample() -> dict[str, torch.Tensor]:
    return {
        "context": torch.randn(2, 4, 6),
        "context_mask": torch.ones(2, 4, dtype=torch.bool),
        "action": torch.randn(2, 8, 7),
        "proprio": torch.randn(2, 9, 8),
        "image_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        "proprio_is_pad": torch.zeros(2, 9, dtype=torch.bool),
    }


def test_adapter_cached_inputs_match_three_independent_online_encodes() -> None:
    video = torch.randn(2, 3, 3, 2, 2)
    online_model = _InputBuilderStub()
    online_sample = {**_sample(), "video": video}
    online = SideModel3AdapterV2WAM.build_inputs(online_model, online_sample)
    assert online_model.vae_calls == 3

    packed = torch.cat(
        [online["current_latents"], online["future_latents"][4], online["future_latents"][8]],
        dim=2,
    )
    cached_model = _InputBuilderStub()
    cached_sample = {key: value for key, value in online_sample.items() if key != "video"}
    cached_sample["video_latents"] = packed
    cached = SideModel3AdapterV2WAM.build_inputs(cached_model, cached_sample)

    assert cached_model.vae_calls == 0
    assert torch.equal(cached["current_latents"], online["current_latents"])
    assert torch.equal(cached["future_latents"][4], online["future_latents"][4])
    assert torch.equal(cached["future_latents"][8], online["future_latents"][8])


def test_adapter_hydra_composes_with_the_cached_object_backend() -> None:
    root = Path(__file__).resolve().parents[2]
    backend_config_dir = root / "model3/third_party/light_wam/configs"
    adapter_config_dir = root / "side_model3_adapter_v2/configs/hydra"
    with initialize_config_dir(
        version_base=None,
        config_dir=str(backend_config_dir.resolve()),
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                f"hydra.searchpath=[file://{adapter_config_dir.resolve()}]",
                "model=side_model3_adapter_v2",
                "task=libero_uncond_2cam224_1e-4",
            ],
        )

    assert cfg.model._target_ == "side_model3_adapter_v2.runtime.create_side_model3_adapter_v2_wam"
    assert cfg.model.wam_adapter.use_wam_adapter
    assert list(cfg.model.wam_adapter.adapter_layer_indices) == [8, 16, 24]
