"""Typed configuration for the staged Model5-to-Model5-O2 experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from model5.config import Model5Config, load_config as load_model5_config


STAGE1 = "stage1_model5_parent"
STAGE2_CONTROL = "stage2_model5_control"
STAGE2_O2 = "stage2_model5_o2"
SUPPORTED_STAGE_ROLES = {STAGE1, STAGE2_CONTROL, STAGE2_O2}


@dataclass(frozen=True)
class ParentInitializationConfig:
    mode: str
    model5_checkpoint: Path | None
    model5_checkpoint_sha256: str | None
    model5_checkpoint_step: int | None


@dataclass(frozen=True)
class O2ReadoutConfig:
    query_trace_readout: str
    readout_num_layers: int
    readout_query_dim: int
    readout_gate_type: str
    readout_identity_init: bool


@dataclass(frozen=True)
class Model5O2Config:
    base: Model5Config
    stage_role: str
    initialization: ParentInitializationConfig
    readout: O2ReadoutConfig | None

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def _resolve(root: Path, value: str | None) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> Model5O2Config:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = load_model5_config(config_path)
    stage_role = str(raw["stage_role"])
    initialization_raw = raw.get("initialization", {})
    initialization = ParentInitializationConfig(
        mode=str(initialization_raw.get("mode", "fresh")),
        model5_checkpoint=_resolve(
            base.project_root,
            initialization_raw.get("model5_checkpoint"),
        ),
        model5_checkpoint_sha256=(
            None
            if initialization_raw.get("model5_checkpoint_sha256") in {None, ""}
            else str(initialization_raw["model5_checkpoint_sha256"]).lower()
        ),
        model5_checkpoint_step=(
            None
            if initialization_raw.get("model5_checkpoint_step") is None
            else int(initialization_raw["model5_checkpoint_step"])
        ),
    )
    architecture_raw = raw["architecture"]
    readout = None
    if stage_role == STAGE2_O2:
        readout = O2ReadoutConfig(
            query_trace_readout=str(architecture_raw["query_trace_readout"]),
            readout_num_layers=int(architecture_raw["readout_num_layers"]),
            readout_query_dim=int(architecture_raw["readout_query_dim"]),
            readout_gate_type=str(architecture_raw["readout_gate_type"]),
            readout_identity_init=bool(architecture_raw["readout_identity_init"]),
        )
    return Model5O2Config(
        base=base,
        stage_role=stage_role,
        initialization=initialization,
        readout=readout,
    )
