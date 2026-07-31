"""Hydra factories for matched Model5 and Model5-O2 Stage 2 arms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model5.models import Model5WAM
from model5.runtime import create_model5_wam

from .models import Model5O2WAM


def _validate_parent_identity(
    model: Model5WAM,
    *,
    path: str,
    expected_sha256: str,
    expected_step: int,
) -> tuple[Path, dict[str, Any], str]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model5 parent does not exist: {checkpoint_path}")
    if len(expected_sha256) != 64:
        raise ValueError("Model5 parent SHA-256 must contain 64 characters")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("Model5 parent SHA-256 must be hexadecimal") from error
    actual_sha256 = Model5O2WAM._sha256(checkpoint_path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"Model5 parent SHA mismatch: expected {expected_sha256.lower()}, "
            f"got {actual_sha256}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Model5 parent payload must be a dictionary")
    if payload.get("method_id") != Model5WAM.method_id:
        raise ValueError(f"unexpected Model5 parent method: {payload.get('method_id')!r}")
    if payload.get("model_class") != Model5WAM.__name__:
        raise ValueError(f"unexpected Model5 parent class: {payload.get('model_class')!r}")
    if payload.get("step") != int(expected_step):
        raise ValueError(
            f"Model5 parent step must be {expected_step}, got {payload.get('step')!r}"
        )
    if payload.get("action_feature_config") != model.action_feature_config_dict():
        raise ValueError("Model5 parent action-feature config mismatch")
    if payload.get("action_policy_config") != model.action_policy.config_dict():
        raise ValueError("Model5 parent action-policy config mismatch")
    return checkpoint_path, payload, actual_sha256


def _require_exact_state(
    *,
    model_state: dict[str, torch.Tensor],
    parent_state: Any,
    label: str,
) -> None:
    if not isinstance(parent_state, dict):
        raise ValueError(f"Model5 parent is missing {label} state")
    missing = set(model_state) - set(parent_state)
    unexpected = set(parent_state) - set(model_state)
    if missing or unexpected:
        raise ValueError(
            f"Model5 parent {label} keys differ: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    mismatched_shapes = sorted(
        key
        for key in model_state
        if tuple(model_state[key].shape) != tuple(parent_state[key].shape)
    )
    if mismatched_shapes:
        raise ValueError(f"Model5 parent {label} tensor shapes differ: {mismatched_shapes}")


def create_model5_control_wam(
    *,
    model5_parent_path: str,
    model5_parent_sha256: str,
    model5_parent_step: int,
    **kwargs,
) -> Model5WAM:
    """Construct the q3-only Stage 2 control from Model5-80K weights."""

    model = create_model5_wam(**kwargs)
    checkpoint_path, payload, actual_sha256 = _validate_parent_identity(
        model,
        path=model5_parent_path,
        expected_sha256=model5_parent_sha256,
        expected_step=int(model5_parent_step),
    )
    _require_exact_state(
        model_state=model.mot.state_dict(),
        parent_state=payload.get("mot"),
        label="mot",
    )
    _require_exact_state(
        model_state=model.action_policy.state_dict(),
        parent_state=payload.get("action_policy_state_dict"),
        label="action_policy",
    )
    if model.proprio_encoder is not None:
        _require_exact_state(
            model_state=model.proprio_encoder.state_dict(),
            parent_state=payload.get("proprio_encoder"),
            label="proprio_encoder",
        )
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.action_policy.load_state_dict(payload["action_policy_state_dict"], strict=True)
    if model.proprio_encoder is not None:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    model.model5_parent_identity = {
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "step": int(model5_parent_step),
        "method_id": Model5WAM.method_id,
        "model_class": Model5WAM.__name__,
        "action_feature_config": payload["action_feature_config"],
        "action_policy_config": payload["action_policy_config"],
    }
    return model


def create_model5_o2_wam(
    *,
    model5_parent_path: str,
    model5_parent_sha256: str,
    model5_parent_step: int,
    **kwargs,
) -> Model5O2WAM:
    """Construct the exact-q3 O2 treatment from the same Model5-80K weights."""

    model = create_model5_wam(_model_class=Model5O2WAM, **kwargs)
    if not isinstance(model, Model5O2WAM):
        raise RuntimeError(f"expected Model5O2WAM, got {type(model).__name__}")
    model.load_model5_warmstart(
        model5_parent_path,
        expected_sha256=model5_parent_sha256,
        expected_step=int(model5_parent_step),
    )
    return model
