"""Side-Model3-Adapter dataset boundary for independent-observation caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from model3.third_party.light_wam.src.lightwam.datasets.lerobot.robot_video_dataset import (
    RobotVideoDataset,
)

from .config import LATENT_CACHE_FORMAT


SIDE_CACHE_ENCODING = "independent_single_frame"
SIDE_CACHE_VIDEO_POSITIONS = (0, 1, 2)
SIDE_CACHE_ENVIRONMENT_OFFSETS = (0, 4, 8)


def validate_side_observation_cache_meta(cache_dir: str | Path) -> dict[str, Any]:
    """Validate the shared pre-Wan cache and reject joint-video layouts."""

    cache_path = Path(cache_dir).expanduser().resolve()
    meta_path = cache_path / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"Side-Model3-Adapter cache metadata is missing: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    semantics = payload.get("side_model3")
    if payload.get("format") != LATENT_CACHE_FORMAT or not isinstance(semantics, dict):
        raise ValueError(
            f"Cache {cache_path} is not {LATENT_CACHE_FORMAT}; "
            "Model3 joint-video caches are incompatible"
        )
    expected = {
        "encoding": SIDE_CACHE_ENCODING,
        "sampled_video_positions": list(SIDE_CACHE_VIDEO_POSITIONS),
        "environment_offsets": list(SIDE_CACHE_ENVIRONMENT_OFFSETS),
        "latent_layout": "C,T,H,W",
        "latent_time": 3,
    }
    for key, value in expected.items():
        if semantics.get(key) != value:
            raise ValueError(
                f"Side-Model3-Adapter cache metadata field {key!r} must be {value!r}, "
                f"got {semantics.get(key)!r}"
            )
    return payload


def validate_complete_side_observation_cache(
    cache_dir: str | Path,
    *,
    expected_samples: int,
) -> dict[str, Any]:
    """Require a complete indexed cache before an Adapter training run."""

    payload = validate_side_observation_cache_meta(cache_dir)
    index_path = Path(cache_dir).expanduser().resolve() / "index.pt"
    if not index_path.is_file():
        raise ValueError(f"Side-Model3-Adapter cache index is not complete: {index_path}")
    index = torch.load(index_path, map_location="cpu")
    sample_to_shard = index.get("sample_to_shard") if isinstance(index, dict) else None
    sample_to_offset = index.get("sample_to_offset") if isinstance(index, dict) else None
    if (
        not isinstance(sample_to_shard, torch.Tensor)
        or not isinstance(sample_to_offset, torch.Tensor)
        or sample_to_shard.numel() != expected_samples
        or sample_to_offset.numel() != expected_samples
        or bool((sample_to_shard < 0).any())
        or bool((sample_to_offset < 0).any())
    ):
        raise ValueError(
            f"Side-Model3-Adapter cache does not cover all {expected_samples} training samples"
        )
    return payload


class SideModel3AdapterV2CachedRobotVideoDataset(RobotVideoDataset):
    """RobotVideoDataset with a strict shared pre-Wan cache identity check."""

    def __init__(
        self,
        *args: Any,
        use_latent_cache: bool = False,
        latent_cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not use_latent_cache or not latent_cache_dir:
            raise ValueError(
                "Side-Model3-Adapter training requires the independent-observation latent cache"
            )
        validate_side_observation_cache_meta(latent_cache_dir)
        super().__init__(
            *args,
            use_latent_cache=True,
            latent_cache_dir=latent_cache_dir,
            **kwargs,
        )
        validate_complete_side_observation_cache(
            latent_cache_dir,
            expected_samples=len(self),
        )
