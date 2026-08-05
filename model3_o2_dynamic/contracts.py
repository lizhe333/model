"""Fail-fast legality checks for Dynamic response-prewarmed O2."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from typing import Any

from model3.contracts import ContractError
from model3_o2.config import Model3O2Config, O2ArchitectureConfig, O2InitializationConfig
from model3_o2.contracts import validate_contract as validate_o2_contract

from .config import (
    PIPELINE_TEMPLATE,
    STAGE2_JOINT,
    DynamicArchitectureConfig,
    Model3O2DynamicConfig,
    SUPPORTED_STAGE_ROLES,
)
from .stage1.contracts import Stage1TrainConfig


_SUITE_CONTRACTS = {
    "libero_object": {
        "parent_step": 20_000,
        "parent_sha256": "391978a158a99aeca9d425c6313451550d44165d97b01a2c75b3529381d019c1",
        "max_steps": 35_000,
        "num_workers": 16,
        "max_episode_steps": 400,
        "gate_freeze_through_step": 30_000,
        "first_gate_update_step": 30_001,
    },
    "libero_10": {
        "parent_step": 80_000,
        "parent_sha256": "65680089b942e1e01b30cf51f707079bd0404956c63a166737e10b1984971d68",
        "max_steps": 10_000,
        "num_workers": 8,
        "max_episode_steps": 700,
        "gate_freeze_through_step": 0,
        "first_gate_update_step": 1,
    },
}


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(
    config: Model3O2DynamicConfig,
    *,
    check_paths: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    base = config.base
    architecture = base.architecture
    initialization = config.initialization
    schedule = config.schedule
    suite = base.evaluation.suite
    suite_contract = _SUITE_CONTRACTS.get(suite)
    if not isinstance(architecture, DynamicArchitectureConfig):
        raise ContractError("Dynamic O2 requires DynamicArchitectureConfig")
    _require(base.track_id == "model3_o2_dynamic", "track_id must be model3_o2_dynamic", errors)
    _require(architecture.parent_track == "model3_o2", "Dynamic parent_track must be model3_o2", errors)
    _require(
        architecture.design_lineage == "model3_o2_dynamic_response_prewarm",
        "unexpected Dynamic design lineage",
        errors,
    )
    _require(architecture.response_adapter_layers == (8, 16, 24), "response adapters must be at 8/16/24", errors)
    _require(architecture.response_adapter_hidden_dim == 1536, "response adapter width must be 1536", errors)
    _require(architecture.response_adapter_bottleneck_dim == 64, "response adapter bottleneck must be 64", errors)
    _require(architecture.response_adapter_activation == "gelu", "response adapter must use GELU", errors)
    _require(architecture.response_adapter_zero_init, "response adapter up projection must be zero initialized", errors)
    _require(
        architecture.response_adapter_placement == "hidden_to_memory_projection",
        "response adapter must be between hidden and memory projection",
        errors,
    )
    _require(initialization.stage_role in SUPPORTED_STAGE_ROLES, "unsupported Dynamic stage role", errors)
    _require(
        suite_contract is not None,
        "Dynamic supports only the registered LIBERO Object and LIBERO Long suites",
        errors,
    )
    _require(base.evaluation.num_inference_steps == 10, "Dynamic must retain the O2 solver-10 protocol", errors)
    _require(base.training.gpu_ids == (0, 1, 2, 3), "Dynamic formal run uses GPUs 0,1,2,3", errors)
    _require(base.training.num_processes == 4, "Dynamic formal run needs four ranks", errors)
    _require(
        (base.training.batch_size, base.training.gradient_accumulation_steps) == (16, 1),
        "Dynamic must retain O2 B16/GA1",
        errors,
    )
    _require(base.training.save_every == 5_000, "Dynamic Stage 2 must save every 5K", errors)
    _require(base.training.warmup_steps == 1_000, "Dynamic Stage 2 warmup must remain 1K", errors)
    _require(base.training.learning_rate == 1e-4, "Dynamic Stage 2 learning rate must remain 1e-4", errors)
    _require(schedule.freeze_through_step == 5000, "adapter freeze boundary must be step 5000", errors)
    _require(schedule.first_adapter_update_step == 5001, "first adapter update must be step 5001", errors)
    _require(schedule.adapter_lr_scale == 0.1, "adapter LR scale must be 0.1", errors)
    _require(schedule.gate_lr_scale == 1.0, "O2 layer_readout LR scale must be 1.0", errors)
    _require(initialization.require_model3_warmstart, "Dynamic requires pinned Model3 parent", errors)
    if suite_contract is not None:
        _require(
            base.training.max_steps == suite_contract["max_steps"],
            f"Dynamic {suite} Stage 2 local budget must be {suite_contract['max_steps']}",
            errors,
        )
        _require(
            base.training.num_workers == suite_contract["num_workers"],
            f"Dynamic {suite} must use {suite_contract['num_workers']} data-loader workers",
            errors,
        )
        _require(
            base.evaluation.max_episode_steps == suite_contract["max_episode_steps"],
            f"Dynamic {suite} episode limit must be {suite_contract['max_episode_steps']}",
            errors,
        )
        _require(
            initialization.model3_checkpoint_step == suite_contract["parent_step"],
            f"Dynamic {suite} parent must be Model3 step {suite_contract['parent_step']}",
            errors,
        )
        _require(
            initialization.model3_checkpoint_sha256 == suite_contract["parent_sha256"],
            f"Dynamic {suite} parent SHA-256 does not match the registered shared parent",
            errors,
        )
        _require(
            schedule.gate_freeze_through_step == suite_contract["gate_freeze_through_step"],
            f"Dynamic {suite} O2 gate freeze boundary changed",
            errors,
        )
        _require(
            schedule.first_gate_update_step == suite_contract["first_gate_update_step"],
            f"Dynamic {suite} first O2 gate update changed",
            errors,
        )
    sha = initialization.model3_checkpoint_sha256
    _require(len(sha) == 64 and all(char in "0123456789abcdef" for char in sha), "invalid Model3 parent SHA-256", errors)
    try:
        Stage1TrainConfig.from_mapping(config.stage1)
    except Exception as error:
        errors.append(f"invalid frozen Stage 1 configuration: {error}")
    if initialization.stage_role == STAGE2_JOINT:
        _require(initialization.response_adapter_export is not None, "Stage 2 requires Stage 1 adapter export", errors)
        export_sha = initialization.response_adapter_export_sha256 or ""
        _require(len(export_sha) == 64 and all(char in "0123456789abcdef" for char in export_sha), "Stage 2 requires adapter-export SHA-256", errors)
    else:
        _require(initialization.response_adapter_export is None, "pipeline template cannot bind a Stage 1 export", errors)

    # Re-use O2's strict inherited architecture validation with a reference view
    # of the Dynamic config.  Only its identity/lineage labels are substituted.
    o2_architecture = O2ArchitectureConfig(
        **{
            key: value
            for key, value in asdict(architecture).items()
            if key in O2ArchitectureConfig.__dataclass_fields__
        },
    )
    o2_architecture = replace(
        o2_architecture,
        parent_track="model3",
        design_lineage="model3_layer_aware_query_flow",
    )
    # The parent-contract checker validates the original Model3 schedule rather
    # than the Dynamic O2-local budget.  O2 itself already converts Long's
    # 10K local budget to the 80K shared-parent budget; Object needs the same
    # explicit 150K reference conversion here.
    inherited_max_steps = 150_000 if suite == "libero_object" else base.training.max_steps
    o2_base = replace(
        base,
        track_id="model3_o2",
        architecture=o2_architecture,
        training=replace(base.training, max_steps=inherited_max_steps),
    )
    o2_initialization = O2InitializationConfig(
        require_model3_warmstart=initialization.require_model3_warmstart,
        model3_checkpoint=initialization.model3_checkpoint,
        model3_checkpoint_sha256=initialization.model3_checkpoint_sha256,
        model3_checkpoint_step=initialization.model3_checkpoint_step,
    )
    if not errors:
        try:
            o2_result = validate_o2_contract(
                Model3O2Config(base=o2_base, initialization=o2_initialization),
                check_paths=check_paths,
            )
        except ContractError as error:
            errors.append(str(error))
            o2_result = None
    else:
        o2_result = None
    if check_paths and initialization.stage_role == STAGE2_JOINT and initialization.response_adapter_export is not None:
        _require(initialization.response_adapter_export.is_file(), "Stage 1 adapter export does not exist", errors)
        if initialization.response_adapter_export.is_file() and initialization.response_adapter_export_sha256:
            _require(
                _sha256(initialization.response_adapter_export) == initialization.response_adapter_export_sha256,
                "Stage 1 adapter export SHA mismatch",
                errors,
            )
    if errors:
        raise ContractError("Dynamic O2 contract validation failed:\n- " + "\n- ".join(errors))
    return {
        "passed": True,
        "track_id": base.track_id,
        "stage_role": initialization.stage_role,
        "shared_o2_contract_passed": bool(o2_result and o2_result["passed"]),
        "architecture": asdict(architecture),
        "initialization": {
            **asdict(initialization),
            "model3_checkpoint": str(initialization.model3_checkpoint),
            "response_adapter_export": None if initialization.response_adapter_export is None else str(initialization.response_adapter_export),
        },
        "stage2_schedule": asdict(schedule),
        "stage1": config.stage1,
        "training": asdict(base.training),
        "evaluation": asdict(base.evaluation),
    }
