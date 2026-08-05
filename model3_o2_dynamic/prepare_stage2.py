"""Materialize the clean Dynamic Stage-2 config after fixed Stage-1 step 5K."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .config import STAGE2_JOINT, load_config
from .models.response_adapter import ResponseAdapterBank
from .stage1.export import MODEL_CLASS, METHOD_ID, tensor_state_sha256


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_adapter_export(
    path: str | Path,
    *,
    expected_parent_sha256: str,
) -> dict[str, Any]:
    export_path = Path(path).expanduser().resolve()
    if not export_path.is_file():
        raise FileNotFoundError(f"Stage 1 adapter export does not exist: {export_path}")
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    required = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_response_adapter_export",
        "method_id": METHOD_ID,
        "model_class": MODEL_CLASS,
        "stage1_predictor_input": "adapter_residual_only",
        "contains_predictors": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"invalid Stage 1 export {key}: expected {expected!r}")
    forbidden_predictor_keys = [
        key
        for key in payload
        if key.lower() not in {"contains_predictors", "stage1_predictor_input"}
        and (
            "predictor" in key.lower()
            or key.lower().startswith("q_")
            or key.lower().startswith("q.")
        )
    ]
    if forbidden_predictor_keys:
        raise ValueError("Stage 2 export cannot contain temporary Q state")
    bank = ResponseAdapterBank()
    if payload.get("response_adapter_config") != bank.configuration():
        raise ValueError("Stage 1 export adapter architecture does not match Dynamic contract")
    state = payload.get("response_adapter_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Stage 1 export is missing response_adapter_state_dict")
    if tensor_state_sha256(state) != payload.get("response_adapter_state_sha256"):
        raise ValueError("Stage 1 export response-adapter tensor hash mismatch")
    bank.load_state_dict(state, strict=True)
    source = payload.get("source_identity")
    if not isinstance(source, dict):
        raise ValueError("Stage 1 export is missing source_identity")
    if source.get("model3_warmstart_sha256") != expected_parent_sha256:
        raise ValueError("Stage 1 export does not identify the pinned Model3 Object-20K parent")
    if not isinstance(source.get("original_o2_tensor_sha256"), str):
        raise ValueError("Stage 1 export is missing exact O2 step-0 carrier hash")
    normalization = payload.get("normalization_identity")
    if not isinstance(normalization, dict) or normalization.get("normalization_fit_split") != "train":
        raise ValueError("Stage 1 export must identify train-only target normalization")
    return {
        "path": str(export_path),
        "sha256": sha256_file(export_path),
        "adapter_state_sha256": payload["response_adapter_state_sha256"],
        "source_identity": source,
        "normalization_identity": normalization,
    }


def prepare_stage2_config(
    *,
    template_config: str | Path,
    adapter_export: str | Path,
    output_config: str | Path,
) -> dict[str, Any]:
    template_path = Path(template_config).expanduser().resolve()
    dynamic = load_config(template_path)
    identity = verify_adapter_export(
        adapter_export,
        expected_parent_sha256=dynamic.initialization.model3_checkpoint_sha256,
    )
    raw = json.loads(template_path.read_text(encoding="utf-8"))
    output_path = Path(output_config).expanduser().resolve()
    raw = deepcopy(raw)
    raw["project_root"] = str(dynamic.project_root)
    raw["initialization"].update(
        {
            "stage_role": STAGE2_JOINT,
            "response_adapter_export": identity["path"],
            "response_adapter_export_sha256": identity["sha256"],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing materialized Stage-2 config: {output_path}"
        )
    output_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path.parent / "stage1_adapter_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity | {"stage2_config": str(output_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--adapter-export", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_stage2_config(
                template_config=args.template_config,
                adapter_export=args.adapter_export,
                output_config=args.output_config,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
