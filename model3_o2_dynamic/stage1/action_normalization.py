"""Exact LIBERO action/proprio normalization used by the shared Model3 parent.

Stage 1 never trains on the raw simulator action chunk.  This module mirrors
``SingleFieldLinearNormalizer(mode='min/max')`` from the vendored Light-WAM
processor, while sealing the parent ``dataset_stats.json`` identity into the
cache.  Keeping the small implementation local makes the Stage-1 pipeline
usable without constructing a training dataset or silently recomputing stats.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _MinMaxField:
    key: str
    minimum: torch.Tensor
    maximum: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor

    @classmethod
    def from_stats(cls, *, key: str, values: dict[str, Any]) -> "_MinMaxField":
        try:
            minimum = torch.as_tensor(values["global_min"], dtype=torch.float32)
            maximum = torch.as_tensor(values["global_max"], dtype=torch.float32)
        except KeyError as error:
            raise ValueError(f"dataset stats lacks global min/max for {key}") from error
        if minimum.shape != maximum.shape or minimum.ndim != 1:
            raise ValueError(f"normalizer stats for {key} must be matching one-dimensional vectors")
        if not torch.isfinite(minimum).all() or not torch.isfinite(maximum).all():
            raise ValueError(f"normalizer stats for {key} contain non-finite values")
        input_range = maximum - minimum
        ignore = input_range < 1.0e-4
        adjusted_range = input_range.clone()
        adjusted_range[ignore] = 2.0
        scale = 2.0 / adjusted_range
        offset = -1.0 - scale * minimum
        offset[ignore] = -minimum[ignore]
        return cls(key=key, minimum=minimum, maximum=maximum, scale=scale, offset=offset)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if int(value.shape[-1]) != int(self.scale.numel()):
            raise ValueError(
                f"{self.key} expected final width {self.scale.numel()}, got {tuple(value.shape)}"
            )
        scale = self.scale.to(device=value.device, dtype=torch.float32)
        offset = self.offset.to(device=value.device, dtype=torch.float32)
        return (value.float() * scale + offset).clamp(-5.0, 5.0)

    def identity(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "global_min": self.minimum.tolist(),
            "global_max": self.maximum.tolist(),
            "scale": self.scale.tolist(),
            "offset": self.offset.tolist(),
            "mode": "min/max",
            "clamp": [-5.0, 5.0],
        }


@dataclass(frozen=True)
class OfficialO2Normalizer:
    """Sealed action + proprio normalization identity for Dynamic Stage 1."""

    action: _MinMaxField
    state: _MinMaxField
    stats_path: Path
    stats_sha256: str
    parent_config_path: Path
    parent_config_sha256: str

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action.forward(action)

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.state.forward(proprio)

    def identity(self) -> dict[str, Any]:
        return {
            "normalizer_kind": "lightwam_single_field_minmax_v1",
            "dataset_stats_path": str(self.stats_path),
            "dataset_stats_sha256": self.stats_sha256,
            "parent_config_path": str(self.parent_config_path),
            "parent_config_sha256": self.parent_config_sha256,
            "action": self.action.identity(),
            "state": self.state.identity(),
            "action_state_transforms": None,
            "use_stepwise_action_norm": False,
        }


def _require_parent_processor_contract(parent: dict[str, Any]) -> tuple[str, str]:
    processor = parent.get("data", {}).get("train", {}).get("processor", {})
    if not isinstance(processor, dict):
        raise ValueError("parent Hydra config lacks data.train.processor")
    if processor.get("action_state_transforms") is not None:
        raise ValueError("Dynamic Stage 1 only supports the shared O2 null action/state transform")
    if bool(processor.get("use_stepwise_action_norm", False)):
        raise ValueError("Dynamic Stage 1 requires global, not stepwise, action normalization")
    if processor.get("norm_default_mode") != "min/max" or processor.get("norm_exception_mode") is not None:
        raise ValueError("Dynamic Stage 1 requires the shared O2 min/max normalizer without exceptions")
    action_meta = processor.get("shape_meta", {}).get("action", [])
    state_meta = processor.get("shape_meta", {}).get("state", [])
    if len(action_meta) != 1 or len(state_meta) != 1:
        raise ValueError("Dynamic Stage 1 requires one merged action key and one merged state key")
    action = action_meta[0]
    state = state_meta[0]
    if (action.get("key"), action.get("raw_shape"), action.get("shape")) != ("default", 7, 7):
        raise ValueError("shared O2 action shape contract is not raw/transformed width 7")
    if (state.get("key"), state.get("raw_shape"), state.get("shape")) != ("default", 8, 8):
        raise ValueError("shared O2 proprio shape contract is not raw/transformed width 8")
    return str(action["key"]), str(state["key"])


def load_official_o2_normalizer(
    *,
    parent_config_path: str | Path,
    dataset_stats_path: str | Path | None = None,
) -> OfficialO2Normalizer:
    """Load only the already-frozen parent normalizer; never fit new stats."""

    config_path = Path(parent_config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"missing shared Model3 parent config: {config_path}")
    parent = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(parent, dict):
        raise ValueError("shared Model3 parent config must be a mapping")
    action_key, state_key = _require_parent_processor_contract(parent)
    stats_path = (
        Path(dataset_stats_path).expanduser().resolve()
        if dataset_stats_path is not None
        else config_path.parent / "dataset_stats.json"
    )
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing shared Model3 dataset stats: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(stats, dict):
        raise ValueError("dataset stats must be a mapping")
    try:
        action_values = stats["action"][action_key]
        state_values = stats["state"][state_key]
    except KeyError as error:
        raise ValueError("dataset stats lacks the shared O2 action/state keys") from error
    action = _MinMaxField.from_stats(key=f"action.{action_key}", values=action_values)
    state = _MinMaxField.from_stats(key=f"state.{state_key}", values=state_values)
    if action.scale.numel() != 7 or state.scale.numel() != 8:
        raise ValueError("shared O2 action/proprio normalizer widths changed")
    return OfficialO2Normalizer(
        action=action,
        state=state,
        stats_path=stats_path,
        stats_sha256=sha256_file(stats_path),
        parent_config_path=config_path,
        parent_config_sha256=sha256_file(config_path),
    )
