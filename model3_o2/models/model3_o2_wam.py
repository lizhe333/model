"""Strict Model3 O2 layer-aware readout treatment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import torch

from model3.models.model3_wam import Model3WAM
from model3.models.vla_query_dit_action_expert import VLAQueryDiTActionExpert
from model3.third_party.light_wam.src.lightwam.utils.logging_config import get_logger

from .vla_query_layer_aware_dit_action_expert import (
    VLAQueryLayerAwareDiTActionExpert,
)


logger = get_logger(__name__)


class Model3O2WAM(Model3WAM):
    """Model3 with an O2-only layer-aware query readout."""

    method_id = VLAQueryLayerAwareDiTActionExpert.method_id
    model3_method_id = Model3WAM.method_id
    model3_class_name = Model3WAM.__name__
    _warmstart_missing_prefix = "layer_readout."

    @classmethod
    def _build_action_policy(
        cls,
        *,
        video_hidden_dim: int,
        action_dim: int,
        num_fusion_layers: int,
        proprio_dim: Optional[int],
        action_query_policy_config: dict[str, Any],
        device: str,
        torch_dtype: torch.dtype,
    ) -> VLAQueryLayerAwareDiTActionExpert:
        return VLAQueryLayerAwareDiTActionExpert(
            video_hidden_dim=video_hidden_dim,
            action_dim=action_dim,
            num_fusion_layers=num_fusion_layers,
            proprio_dim=proprio_dim,
            **dict(action_query_policy_config),
        ).to(device=device, dtype=torch_dtype)

    @property
    def action_policy(self) -> VLAQueryLayerAwareDiTActionExpert:
        expert = self.state_fusion_action_expert
        if not isinstance(expert, VLAQueryLayerAwareDiTActionExpert):
            raise RuntimeError(
                "Model3O2WAM requires VLAQueryLayerAwareDiTActionExpert; "
                f"got {type(expert).__name__ if expert is not None else None}."
            )
        return expert

    @staticmethod
    def _sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _expected_model3_policy_config(self) -> dict[str, Any]:
        # Calling the parent implementation directly excludes O2-only config
        # fields while retaining the exact inherited Action-DiT contract.
        expected = VLAQueryDiTActionExpert.config_dict(self.action_policy)
        expected["method_id"] = self.model3_method_id
        return expected

    def load_model3_warmstart(
        self,
        path: str | Path,
        expected_sha256: str,
        expected_step: int = 20_000,
    ) -> dict[str, Any]:
        """Warm-start inherited Model3 state while leaving O2 readout new."""

        checkpoint_path = Path(path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model3 warm-start checkpoint does not exist: {checkpoint_path}"
            )
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("Model3 warm start requires a 64-character expected_sha256.")
        try:
            int(expected_sha256, 16)
        except ValueError as error:
            raise ValueError("Model3 warm-start expected_sha256 must be hexadecimal.") from error
        if not isinstance(expected_step, int) or isinstance(expected_step, bool):
            raise TypeError("Model3 warm-start expected_step must be an integer.")

        actual_sha256 = self._sha256(checkpoint_path)
        if actual_sha256 != expected_sha256.lower():
            raise ValueError(
                "Model3 warm-start SHA mismatch: "
                f"expected {expected_sha256.lower()}, got {actual_sha256}."
            )

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Model3 warm-start checkpoint payload must be a dictionary.")
        if payload.get("method_id") != self.model3_method_id:
            raise ValueError(
                f"Warm start must use method_id {self.model3_method_id!r}, "
                f"got {payload.get('method_id')!r}."
            )
        if payload.get("model_class") != self.model3_class_name:
            raise ValueError(
                f"Warm start must use model_class {self.model3_class_name!r}, "
                f"got {payload.get('model_class')!r}."
            )
        if payload.get("step") != expected_step:
            raise ValueError(
                f"Model3 warm-start step must be {expected_step}, "
                f"got {payload.get('step')!r}."
            )

        source_policy_config = payload.get("action_policy_config")
        if not isinstance(source_policy_config, dict):
            raise ValueError("Model3 warm start is missing `action_policy_config`.")
        expected_policy_config = self._expected_model3_policy_config()
        if source_policy_config != expected_policy_config:
            differing_keys = sorted(
                key
                for key in set(source_policy_config) | set(expected_policy_config)
                if source_policy_config.get(key) != expected_policy_config.get(key)
            )
            raise ValueError(
                "Model3 warm-start action policy config mismatch for keys: "
                f"{differing_keys}."
            )

        mot_state = payload.get("mot")
        if not isinstance(mot_state, dict):
            raise ValueError("Model3 warm start is missing a valid `mot` state.")
        policy_state = payload.get("action_policy_state_dict")
        if not isinstance(policy_state, dict):
            raise ValueError(
                "Model3 warm start is missing a valid `action_policy_state_dict`."
            )

        source_has_proprio = "proprio_encoder" in payload
        target_has_proprio = self.proprio_encoder is not None
        if source_has_proprio != target_has_proprio:
            raise ValueError(
                "Model3 warm-start proprio presence mismatch: "
                f"checkpoint={source_has_proprio}, model3_o2={target_has_proprio}."
            )
        proprio_state = payload.get("proprio_encoder")
        if target_has_proprio and not isinstance(proprio_state, dict):
            raise ValueError("Model3 warm start has an invalid `proprio_encoder` state.")

        target_policy_state = self.action_policy.state_dict()
        expected_missing = {
            key
            for key in target_policy_state
            if key.startswith(self._warmstart_missing_prefix)
        }
        if not expected_missing:
            raise RuntimeError(
                "Model3O2WAM action policy exposes no `layer_readout.*` parameters."
            )
        missing = set(target_policy_state) - set(policy_state)
        unexpected = set(policy_state) - set(target_policy_state)
        if missing != expected_missing or unexpected:
            raise ValueError(
                "Unexpected Model3->Model3O2 action-policy keys: "
                f"missing={sorted(missing)}, expected_missing={sorted(expected_missing)}, "
                f"unexpected={sorted(unexpected)}."
            )

        self.mot.load_state_dict(mot_state, strict=True)
        try:
            incompatible = self.action_policy.load_state_dict(policy_state, strict=False)
        except RuntimeError as error:
            raise ValueError(
                f"Model3 warm-start action policy tensors are incompatible: {error}"
            ) from error

        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing != expected_missing or unexpected:
            raise ValueError(
                "Unexpected Model3->Model3O2 action-policy incompatibility: "
                f"missing={sorted(missing)}, expected_missing={sorted(expected_missing)}, "
                f"unexpected={sorted(unexpected)}."
            )

        if target_has_proprio:
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)

        self.model3_warmstart_identity = {
            "path": str(checkpoint_path),
            "sha256": actual_sha256,
            "step": expected_step,
            "method_id": self.model3_method_id,
            "model_class": self.model3_class_name,
            "action_policy_config": source_policy_config,
        }
        logger.info(
            "Loaded strict Model3 warm start for Model3O2: path=%s step=%s sha256=%s",
            checkpoint_path,
            expected_step,
            actual_sha256,
        )
        return payload
