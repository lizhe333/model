"""Typed configuration for the three-part Dynamic O2 lineage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from model3.config import ArchitectureConfig, Model3Config, load_config as load_model3_config


PIPELINE_TEMPLATE = "pipeline_template"
STAGE2_JOINT = "stage2_joint"
SUPPORTED_STAGE_ROLES = {PIPELINE_TEMPLATE, STAGE2_JOINT}


@dataclass(frozen=True)
class DynamicArchitectureConfig(ArchitectureConfig):
    query_trace_readout: str
    readout_num_layers: int
    readout_query_dim: int
    readout_gate_type: str
    readout_identity_init: bool
    response_adapter_layers: tuple[int, ...]
    response_adapter_hidden_dim: int
    response_adapter_bottleneck_dim: int
    response_adapter_activation: str
    response_adapter_zero_init: bool
    response_adapter_placement: str


@dataclass(frozen=True)
class DynamicInitializationConfig:
    require_model3_warmstart: bool
    model3_checkpoint: Path
    model3_checkpoint_sha256: str
    model3_checkpoint_step: int
    stage_role: str
    response_adapter_export: Path | None
    response_adapter_export_sha256: str | None


@dataclass(frozen=True)
class DynamicScheduleConfig:
    freeze_through_step: int
    first_adapter_update_step: int
    adapter_lr_scale: float
    gate_freeze_through_step: int
    first_gate_update_step: int
    gate_lr_scale: float


@dataclass(frozen=True)
class Model3O2DynamicConfig:
    base: Model3Config
    initialization: DynamicInitializationConfig
    schedule: DynamicScheduleConfig
    stage1: dict

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def _resolve(root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> Model3O2DynamicConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = load_model3_config(config_path)
    architecture_raw = raw["architecture"]
    architecture = DynamicArchitectureConfig(
        **asdict(base.architecture),
        query_trace_readout=str(architecture_raw["query_trace_readout"]),
        readout_num_layers=int(architecture_raw["readout_num_layers"]),
        readout_query_dim=int(architecture_raw["readout_query_dim"]),
        readout_gate_type=str(architecture_raw["readout_gate_type"]),
        readout_identity_init=bool(architecture_raw["readout_identity_init"]),
        response_adapter_layers=tuple(int(value) for value in architecture_raw["response_adapter_layers"]),
        response_adapter_hidden_dim=int(architecture_raw["response_adapter_hidden_dim"]),
        response_adapter_bottleneck_dim=int(architecture_raw["response_adapter_bottleneck_dim"]),
        response_adapter_activation=str(architecture_raw["response_adapter_activation"]),
        response_adapter_zero_init=bool(architecture_raw["response_adapter_zero_init"]),
        response_adapter_placement=str(architecture_raw["response_adapter_placement"]),
    )
    base = replace(base, architecture=architecture)
    initialization_raw = raw["initialization"]
    initialization = DynamicInitializationConfig(
        require_model3_warmstart=bool(initialization_raw["require_model3_warmstart"]),
        model3_checkpoint=_resolve(base.project_root, initialization_raw["model3_checkpoint"]),
        model3_checkpoint_sha256=str(initialization_raw["model3_checkpoint_sha256"]).lower(),
        model3_checkpoint_step=int(initialization_raw["model3_checkpoint_step"]),
        stage_role=str(initialization_raw.get("stage_role", PIPELINE_TEMPLATE)),
        response_adapter_export=_resolve(base.project_root, initialization_raw.get("response_adapter_export")),
        response_adapter_export_sha256=(
            None
            if initialization_raw.get("response_adapter_export_sha256") in (None, "")
            else str(initialization_raw["response_adapter_export_sha256"]).lower()
        ),
    )
    schedule_raw = raw.get("stage2_schedule", {})
    schedule = DynamicScheduleConfig(
        freeze_through_step=int(schedule_raw["freeze_through_step"]),
        first_adapter_update_step=int(schedule_raw["first_adapter_update_step"]),
        adapter_lr_scale=float(schedule_raw["adapter_lr_scale"]),
        gate_freeze_through_step=int(schedule_raw["gate_freeze_through_step"]),
        first_gate_update_step=int(schedule_raw["first_gate_update_step"]),
        gate_lr_scale=float(schedule_raw["gate_lr_scale"]),
    )
    stage1 = raw.get("stage1", {})
    if not isinstance(stage1, dict):
        raise ValueError("Dynamic `stage1` configuration must be an object")
    return Model3O2DynamicConfig(
        base=base,
        initialization=initialization,
        schedule=schedule,
        stage1=dict(stage1),
    )
