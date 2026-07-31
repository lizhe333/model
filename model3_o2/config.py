"""Typed configuration for the Model3 O2 treatment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from model3.config import ArchitectureConfig, Model3Config, load_config as load_model3_config


@dataclass(frozen=True)
class O2ArchitectureConfig(ArchitectureConfig):
    query_trace_readout: str
    readout_num_layers: int
    readout_query_dim: int
    readout_gate_type: str
    readout_identity_init: bool


@dataclass(frozen=True)
class O2InitializationConfig:
    require_model3_warmstart: bool
    model3_checkpoint: Path
    model3_checkpoint_sha256: str
    model3_checkpoint_step: int


@dataclass(frozen=True)
class Model3O2Config:
    base: Model3Config
    initialization: O2InitializationConfig

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> Model3O2Config:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = load_model3_config(config_path)
    architecture_raw = raw["architecture"]
    architecture = O2ArchitectureConfig(
        **asdict(base.architecture),
        query_trace_readout=str(architecture_raw["query_trace_readout"]),
        readout_num_layers=int(architecture_raw["readout_num_layers"]),
        readout_query_dim=int(architecture_raw["readout_query_dim"]),
        readout_gate_type=str(architecture_raw["readout_gate_type"]),
        readout_identity_init=bool(architecture_raw["readout_identity_init"]),
    )
    base = replace(base, architecture=architecture)
    initialization_raw = raw["initialization"]
    initialization = O2InitializationConfig(
        require_model3_warmstart=bool(initialization_raw["require_model3_warmstart"]),
        model3_checkpoint=_resolve(base.project_root, initialization_raw["model3_checkpoint"]),
        model3_checkpoint_sha256=str(initialization_raw["model3_checkpoint_sha256"]).lower(),
        model3_checkpoint_step=int(initialization_raw["model3_checkpoint_step"]),
    )
    return Model3O2Config(base=base, initialization=initialization)
