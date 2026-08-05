"""Adapter-only Stage 1 export used by the clean Stage 2 construction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from model3_o2_dynamic.models.response_adapter import ResponseAdapterBank


METHOD_ID = "model3_o2_dynamic_response_prewarm_v1"
MODEL_CLASS = "Model3O2DynamicWAM"


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("utf-8"))
        digest.update(repr(tuple(cpu.shape)).encode("utf-8"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_adapter_export(
    adapters: ResponseAdapterBank,
    *,
    source_identity: dict[str, Any],
    normalization_identity: dict[str, Any],
) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in adapters.state_dict().items()}
    return {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_response_adapter_export",
        "method_id": METHOD_ID,
        "model_class": MODEL_CLASS,
        "response_adapter_config": adapters.configuration(),
        "response_adapter_state_dict": state,
        "response_adapter_state_sha256": tensor_state_sha256(state),
        "source_identity": dict(source_identity),
        "normalization_identity": dict(normalization_identity),
        "stage1_predictor_input": "adapter_residual_only",
        "contains_predictors": False,
    }


def save_adapter_export(
    path: str | Path,
    adapters: ResponseAdapterBank,
    *,
    source_identity: dict[str, Any],
    normalization_identity: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    payload = build_adapter_export(
        adapters,
        source_identity=source_identity,
        normalization_identity=normalization_identity,
    )
    torch.save(payload, destination)
    payload["file_sha256"] = sha256_file(destination)
    return payload
