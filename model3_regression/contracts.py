"""Fail-fast scientific contract for the matched direct-regression treatment."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from model3.config import Model3Config
from model3.contracts import ContractError, validate_contract as validate_model3_contract

from .config import RegressionArchitectureConfig


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_contract(
    config: Model3Config,
    *,
    check_paths: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    architecture = config.architecture
    if not isinstance(architecture, RegressionArchitectureConfig):
        raise ContractError("Model3 Regression requires RegressionArchitectureConfig")

    _require(config.track_id == "model3_regression", "track_id must be model3_regression", errors)
    _require(architecture.parent_track == "model3", "parent_track must be model3", errors)
    _require(
        architecture.design_lineage == "model3_same_query_direct_regression",
        "unexpected regression design lineage",
        errors,
    )
    _require(
        architecture.conditioner_type == "layerwise_recurrent_action_query",
        "the Model3 recurrent query conditioner must be preserved",
        errors,
    )
    _require(not architecture.uses_state_fusion, "StateFusion is not allowed", errors)
    _require(
        architecture.action_decoder == "vla_query_direct_regression",
        "action decoder must be direct query regression",
        errors,
    )
    _require(architecture.action_dit_layers == 0, "the regression treatment cannot own an action DiT", errors)
    _require(architecture.action_dit_hidden_dim == 0, "the regression treatment cannot declare Action-DiT width", errors)
    _require(architecture.future_video_flow_loss, "future-video flow loss must stay enabled", errors)
    _require(not architecture.action_flow_loss, "action-flow loss must be disabled", errors)
    _require(architecture.action_regression_loss, "direct action-regression loss must be enabled", errors)
    _require(
        architecture.regression_loss_type == "masked_l1",
        "regression loss must be masked_l1",
        errors,
    )
    _require(architecture.regression_decoder_layers == 2, "regression decoder depth must be 2", errors)
    _require(
        architecture.regression_decoder_hidden_dim == architecture.action_query_hidden_dim == 512,
        "regression decoder and query memory must both use width 512",
        errors,
    )
    _require(architecture.regression_decoder_heads == 8, "regression decoder must use 8 heads", errors)
    _require(
        config.evaluation.suite in {"libero_object", "libero_10"},
        "Model3 Regression supports only registered LIBERO Object or Long controls",
        errors,
    )
    _require(
        config.evaluation.num_inference_steps == 1,
        "direct regression must declare exactly one inference call",
        errors,
    )

    if errors:
        raise ContractError("Model3 Regression contract validation failed:\n- " + "\n- ".join(errors))

    reference_architecture = replace(
        architecture,
        parent_track="model2",
        design_lineage="vla_adapter_action_query",
        action_decoder="vla_query_dit_flow",
        action_dit_layers=16,
        action_dit_hidden_dim=512,
        action_flow_loss=True,
    )
    shared_reference = replace(
        config,
        track_id="model3",
        architecture=reference_architecture,
    )
    shared_result = validate_model3_contract(
        shared_reference,
        check_paths=check_paths,
    )
    return {
        "passed": True,
        "track_id": config.track_id,
        "shared_model3_contract_passed": bool(shared_result["passed"]),
        "checked_paths": shared_result["checked_paths"],
        "architecture": asdict(architecture),
        "data": shared_result["data"],
        "training": shared_result["training"],
        "evaluation": asdict(config.evaluation),
    }
