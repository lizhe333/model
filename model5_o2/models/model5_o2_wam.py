"""Exact-q3 O2 readout treatment on a trained Model5 carrier."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import torch

from model5.models.model5_wam import Model5WAM
from model5.models.vla_query_dit_action_expert import VLAQueryDiTActionExpert
from model5.third_party.light_wam.src.lightwam.utils.logging_config import get_logger

from .vla_query_layer_aware_temporal_dit_action_expert import (
    VLAQueryLayerAwareTemporalDiTActionExpert,
)


logger = get_logger(__name__)


class Model5O2WAM(Model5WAM):
    """Original Model5 temporal features plus an exact-q3 O2 readout."""

    method_id = VLAQueryLayerAwareTemporalDiTActionExpert.method_id
    model5_method_id = VLAQueryDiTActionExpert.method_id
    model5_class_name = Model5WAM.__name__
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
    ) -> VLAQueryLayerAwareTemporalDiTActionExpert:
        return VLAQueryLayerAwareTemporalDiTActionExpert(
            video_hidden_dim=video_hidden_dim,
            action_dim=action_dim,
            num_fusion_layers=num_fusion_layers,
            proprio_dim=proprio_dim,
            **dict(action_query_policy_config),
        ).to(device=device, dtype=torch_dtype)

    @property
    def action_policy(self) -> VLAQueryLayerAwareTemporalDiTActionExpert:
        expert = self.state_fusion_action_expert
        if not isinstance(expert, VLAQueryLayerAwareTemporalDiTActionExpert):
            raise RuntimeError(
                "Model5O2WAM requires VLAQueryLayerAwareTemporalDiTActionExpert; "
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

    def _expected_model5_policy_config(self) -> dict[str, Any]:
        # Calling the inherited Model5 implementation directly removes O2-only
        # fields while preserving every recurrent-query and Action-DiT field.
        expected = VLAQueryDiTActionExpert.config_dict(self.action_policy)
        expected["method_id"] = self.model5_method_id
        return expected

    def load_model5_warmstart(
        self,
        path: str | Path,
        expected_sha256: str,
        expected_step: int = 80_000,
    ) -> dict[str, Any]:
        """Load Model5-80K weights while leaving only the O2 readout new."""

        checkpoint_path = Path(path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model5 warm-start checkpoint does not exist: {checkpoint_path}"
            )
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("Model5 O2 requires a 64-character parent SHA-256")
        try:
            int(expected_sha256, 16)
        except ValueError as error:
            raise ValueError("Model5 parent SHA-256 must be hexadecimal") from error
        if not isinstance(expected_step, int) or isinstance(expected_step, bool):
            raise TypeError("Model5 parent expected_step must be an integer")

        actual_sha256 = self._sha256(checkpoint_path)
        if actual_sha256 != expected_sha256.lower():
            raise ValueError(
                "Model5 parent SHA mismatch: "
                f"expected {expected_sha256.lower()}, got {actual_sha256}."
            )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Model5 parent checkpoint payload must be a dictionary")
        if payload.get("method_id") != self.model5_method_id:
            raise ValueError(
                f"parent must use method_id {self.model5_method_id!r}, "
                f"got {payload.get('method_id')!r}"
            )
        if payload.get("model_class") != self.model5_class_name:
            raise ValueError(
                f"parent must use model_class {self.model5_class_name!r}, "
                f"got {payload.get('model_class')!r}"
            )
        if payload.get("step") != expected_step:
            raise ValueError(
                f"Model5 parent step must be {expected_step}, "
                f"got {payload.get('step')!r}"
            )
        if payload.get("action_feature_config") != self.action_feature_config_dict():
            raise ValueError(
                "Model5 parent action-feature config mismatch: "
                f"expected {self.action_feature_config_dict()}, "
                f"got {payload.get('action_feature_config')}"
            )

        source_policy_config = payload.get("action_policy_config")
        if not isinstance(source_policy_config, dict):
            raise ValueError("Model5 parent is missing `action_policy_config`")
        expected_policy_config = self._expected_model5_policy_config()
        if source_policy_config != expected_policy_config:
            differing_keys = sorted(
                key
                for key in set(source_policy_config) | set(expected_policy_config)
                if source_policy_config.get(key) != expected_policy_config.get(key)
            )
            raise ValueError(
                "Model5 parent action-policy config mismatch for keys: "
                f"{differing_keys}"
            )

        mot_state = payload.get("mot")
        policy_state = payload.get("action_policy_state_dict")
        if not isinstance(mot_state, dict):
            raise ValueError("Model5 parent is missing a valid `mot` state")
        if not isinstance(policy_state, dict):
            raise ValueError("Model5 parent is missing an action-policy state")

        source_has_proprio = "proprio_encoder" in payload
        target_has_proprio = self.proprio_encoder is not None
        if source_has_proprio != target_has_proprio:
            raise ValueError(
                "Model5 parent proprio presence mismatch: "
                f"checkpoint={source_has_proprio}, model5_o2={target_has_proprio}"
            )
        proprio_state = payload.get("proprio_encoder")
        if target_has_proprio and not isinstance(proprio_state, dict):
            raise ValueError("Model5 parent has an invalid proprio state")

        target_policy_state = self.action_policy.state_dict()
        expected_missing = {
            key for key in target_policy_state if key.startswith(self._warmstart_missing_prefix)
        }
        if not expected_missing:
            raise RuntimeError("Model5O2 action policy has no layer_readout parameters")
        missing = set(target_policy_state) - set(policy_state)
        unexpected = set(policy_state) - set(target_policy_state)
        if missing != expected_missing or unexpected:
            raise ValueError(
                "Unexpected Model5->Model5O2 action-policy keys: "
                f"missing={sorted(missing)}, expected_missing={sorted(expected_missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        self.mot.load_state_dict(mot_state, strict=True)
        try:
            incompatible = self.action_policy.load_state_dict(policy_state, strict=False)
        except RuntimeError as error:
            raise ValueError(
                f"Model5 parent action-policy tensors are incompatible: {error}"
            ) from error
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise ValueError(
                "Unexpected Model5->Model5O2 state incompatibility: "
                f"missing={sorted(incompatible.missing_keys)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        if target_has_proprio:
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)

        self.model5_parent_identity = {
            "path": str(checkpoint_path),
            "sha256": actual_sha256,
            "step": expected_step,
            "method_id": self.model5_method_id,
            "model_class": self.model5_class_name,
            "action_feature_config": payload["action_feature_config"],
            "action_policy_config": source_policy_config,
        }
        logger.info(
            "Loaded strict Model5 parent for Model5O2: path=%s step=%s sha256=%s",
            checkpoint_path,
            expected_step,
            actual_sha256,
        )
        return payload

    def gradient_smoke_summary(self) -> dict[str, float | bool]:
        summary = super().gradient_smoke_summary()
        squared_norm = 0.0
        tensors_with_grad = 0
        for parameter in self.action_policy.layer_readout.parameters():
            if parameter.grad is None:
                continue
            tensors_with_grad += 1
            squared_norm += float(parameter.grad.detach().float().square().sum().item())
        summary["gradient/o2_readout_has_grad"] = tensors_with_grad > 0
        summary["gradient/o2_readout_norm"] = squared_norm**0.5
        return summary

    def save_checkpoint(self, path, optimizer=None, step=None):
        parent_identity = getattr(self, "model5_parent_identity", None)
        if not isinstance(parent_identity, dict):
            raise RuntimeError("Model5 O2 checkpoints require a verified Model5 parent")
        payload = {
            "method_id": self.method_id,
            "model_class": type(self).__name__,
            "mot": self.mot.state_dict(),
            "action_policy_state_dict": self.action_policy.state_dict(),
            "action_policy_config": self.action_policy.config_dict(),
            "action_feature_config": self.action_feature_config_dict(),
            "model5_parent_identity": parent_identity,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("action_policy_config") != self.action_policy.config_dict():
            raise ValueError("Model5 O2 checkpoint action-policy config mismatch")
        expected_parent = getattr(self, "model5_parent_identity", None)
        if payload.get("model5_parent_identity") != expected_parent:
            raise ValueError("Model5 O2 checkpoint parent identity mismatch")
        return super().load_checkpoint(path, optimizer=optimizer)
