"""Fail-fast scientific contract for Model3 O2."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from typing import Any

from model3.contracts import ContractError, validate_contract as validate_model3_contract

from .config import Model3O2Config, O2ArchitectureConfig


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(config: Model3O2Config, *, check_paths: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    base = config.base
    architecture = base.architecture
    initialization = config.initialization
    if not isinstance(architecture, O2ArchitectureConfig):
        raise ContractError("Model3 O2 requires O2ArchitectureConfig")

    _require(base.track_id == "model3_o2", "track_id must be model3_o2", errors)
    _require(architecture.parent_track == "model3", "O2 parent_track must be model3", errors)
    _require(
        architecture.design_lineage == "model3_layer_aware_query_flow",
        "unexpected O2 design lineage",
        errors,
    )
    _require(
        architecture.conditioner_type == "layerwise_recurrent_action_query",
        "O2 must preserve the recurrent Model3 query encoder",
        errors,
    )
    _require(not architecture.uses_state_fusion, "StateFusion is not allowed", errors)
    _require(
        architecture.action_decoder == "vla_query_dit_flow",
        "O2 must preserve the Model3 Action-DiT flow decoder",
        errors,
    )
    _require(architecture.action_dit_layers == 16, "O2 Action-DiT must retain 16 layers", errors)
    _require(architecture.action_dit_hidden_dim == 512, "O2 Action-DiT width must remain 512", errors)
    _require(architecture.action_flow_loss, "O2 must retain action flow loss", errors)
    _require(architecture.future_video_flow_loss, "O2 must retain video flow loss", errors)
    _require(
        architecture.query_trace_readout == "layer_separable_gated_residual",
        "O2 must use the registered layer-separable gated residual readout",
        errors,
    )
    _require(architecture.readout_num_layers == 3, "O2 must explicitly read q1/q2/q3", errors)
    _require(
        architecture.readout_query_dim == architecture.action_query_hidden_dim == 512,
        "O2 readout and query memory widths must be 512",
        errors,
    )
    _require(architecture.readout_gate_type == "querywise_scalar", "unexpected O2 gate type", errors)
    _require(architecture.readout_identity_init, "O2 readout must initialize as exact q3 identity", errors)
    _require(base.evaluation.suite == "libero_object", "the initial O2 treatment is Object-only", errors)
    _require(base.evaluation.num_inference_steps == 10, "O2 must retain the 10-step flow solver", errors)

    _require(initialization.require_model3_warmstart, "O2 requires the pinned Model3 warm start", errors)
    _require(initialization.model3_checkpoint_step == 20_000, "O2 warm start must be Model3 step 20K", errors)
    sha = initialization.model3_checkpoint_sha256
    _require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid warm-start SHA-256", errors)

    if errors:
        raise ContractError("Model3 O2 contract validation failed:\n- " + "\n- ".join(errors))

    reference_architecture = replace(
        architecture,
        parent_track="model2",
        design_lineage="vla_adapter_action_query",
    )
    reference_architecture = ArchitectureConfigProxy.to_model3(reference_architecture)
    shared_reference = replace(base, track_id="model3", architecture=reference_architecture)
    shared_result = validate_model3_contract(shared_reference, check_paths=check_paths)

    if check_paths:
        if not initialization.model3_checkpoint.is_file():
            raise ContractError(
                f"Model3 O2 warm-start checkpoint does not exist: {initialization.model3_checkpoint}"
            )
        actual_sha = _sha256(initialization.model3_checkpoint)
        if actual_sha != sha:
            raise ContractError(
                f"Model3 O2 warm-start SHA mismatch: expected {sha}, got {actual_sha}"
            )

    return {
        "passed": True,
        "track_id": base.track_id,
        "shared_model3_contract_passed": bool(shared_result["passed"]),
        "checked_paths": shared_result["checked_paths"],
        "architecture": asdict(architecture),
        "initialization": {
            **asdict(initialization),
            "model3_checkpoint": str(initialization.model3_checkpoint),
        },
        "data": shared_result["data"],
        "training": shared_result["training"],
        "evaluation": shared_result["evaluation"],
    }


class ArchitectureConfigProxy:
    """Drop O2-only dataclass fields before validating the parent contract."""

    @staticmethod
    def to_model3(architecture: O2ArchitectureConfig):
        from model3.config import ArchitectureConfig

        values = asdict(architecture)
        allowed = ArchitectureConfig.__dataclass_fields__
        return ArchitectureConfig(**{key: values[key] for key in allowed})
