"""Typed configuration for the Model3 direct-regression ablation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from model3.config import ArchitectureConfig, Model3Config, load_config as load_model3_config


@dataclass(frozen=True)
class RegressionArchitectureConfig(ArchitectureConfig):
    action_regression_loss: bool
    regression_loss_type: str
    regression_decoder_layers: int
    regression_decoder_hidden_dim: int
    regression_decoder_heads: int


def load_config(path: str | Path) -> Model3Config:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = load_model3_config(config_path)
    architecture_raw = raw["architecture"]
    architecture = RegressionArchitectureConfig(
        **asdict(base.architecture),
        action_regression_loss=bool(architecture_raw["action_regression_loss"]),
        regression_loss_type=str(architecture_raw["regression_loss_type"]),
        regression_decoder_layers=int(architecture_raw["regression_decoder_layers"]),
        regression_decoder_hidden_dim=int(
            architecture_raw["regression_decoder_hidden_dim"]
        ),
        regression_decoder_heads=int(architecture_raw["regression_decoder_heads"]),
    )
    return replace(base, architecture=architecture)
