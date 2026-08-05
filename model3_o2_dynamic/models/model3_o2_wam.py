"""Strict Model3 O2 layer-aware readout treatment."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from model3.models.model3_wam import Model3WAM
from model3.models.vla_query_dit_action_expert import VLAQueryDiTActionExpert
from model3.third_party.light_wam.src.lightwam.utils.logging_config import get_logger

from .vla_query_layer_aware_dit_action_expert import (
    VLAQueryLayerAwareDiTActionExpert,
)
from .response_adapter import (
    ResponseAdapterBank,
    ResponseAdapterConfig,
    normalize_response_adapter_config,
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


class Model3O2DynamicWAM(Model3O2WAM):
    """O2 with three current-only response adapters before query projection.

    The O2 policy class, recurrent query update, readout, and Action-DiT stay
    byte-for-byte inherited.  This class changes only the layer-state records
    handed to the query encoder: ``adapted`` becomes $B_l=h_l+A_l(h_l)$ after
    the Wan forward has ended and before ``memory_projections`` reads it.
    """

    method_id = "model3_o2_dynamic_response_prewarm_v1"
    action_policy_method_id = VLAQueryLayerAwareDiTActionExpert.method_id
    response_adapter_scale = 1.0

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args: Any,
        response_adapter_config: ResponseAdapterConfig | dict[str, object] | None = None,
        **kwargs: Any,
    ) -> "Model3O2DynamicWAM":
        model = super().from_wan22_pretrained(*args, **kwargs)
        if not isinstance(model, cls):
            raise RuntimeError(f"expected {cls.__name__}, got {type(model).__name__}")
        model.install_response_adapters(response_adapter_config)
        return model

    def install_response_adapters(
        self,
        config: ResponseAdapterConfig | dict[str, object] | None = None,
    ) -> None:
        """Install the three zero-identity adapters once model dtype/device is known."""

        normalized = normalize_response_adapter_config(config)
        existing = getattr(self, "response_adapters", None)
        if existing is not None:
            if not isinstance(existing, ResponseAdapterBank):
                raise RuntimeError("response_adapters has an unexpected module type")
            if existing.configuration() != normalized.as_dict():
                raise ValueError("cannot replace an already-installed response-adapter contract")
            return
        reference = next(self.action_policy.parameters())
        self.response_adapters = ResponseAdapterBank(normalized).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        self._response_adapters_enabled = True
        self._response_adapters_trainable = True
        self._response_adapters_pending_unfreeze = False
        self._o2_gate_trainable = True
        self._o2_gate_pending_unfreeze = False
        self._last_response_adapter_metrics: dict[str, float] = {}
        self.response_adapter_export_identity: dict[str, Any] | None = None

    def _require_response_adapters(self) -> ResponseAdapterBank:
        bank = getattr(self, "response_adapters", None)
        if not isinstance(bank, ResponseAdapterBank):
            raise RuntimeError(
                "Model3O2DynamicWAM response adapters are not installed. "
                "Construct it through create_model3_o2_dynamic_wam."
            )
        return bank

    @property
    def response_adapter_config(self) -> dict[str, object]:
        return self._require_response_adapters().configuration()

    @property
    def response_adapters_trainable(self) -> bool:
        return bool(getattr(self, "_response_adapters_trainable", False))

    def set_response_adapters_trainable(self, enabled: bool) -> None:
        """Set only deployment-adapter parameter ownership, never its routing."""

        bank = self._require_response_adapters()
        self._response_adapters_trainable = bool(enabled)
        bank.set_trainable(bool(enabled))

    def schedule_response_adapter_unfreeze(self) -> None:
        """Request activation at the next action-loss forward, after step-5K save."""

        self._response_adapters_pending_unfreeze = True

    def _activate_pending_response_adapter_unfreeze(self) -> None:
        if bool(getattr(self, "_response_adapters_pending_unfreeze", False)):
            self.set_response_adapters_trainable(True)
            self._response_adapters_pending_unfreeze = False
            logger.info("Activated Dynamic response-adapter gradients for optimizer step 5001.")

    def _require_o2_layer_readout(self) -> torch.nn.Module:
        """Return exactly the original O2 gate, never the routed-query output."""

        gate = getattr(self.action_policy, "layer_readout", None)
        if not isinstance(gate, torch.nn.Module):
            raise RuntimeError("Dynamic O2 requires the original action_policy.layer_readout module")
        return gate

    @property
    def o2_gate_trainable(self) -> bool:
        return bool(getattr(self, "_o2_gate_trainable", False))

    def set_o2_gate_trainable(self, enabled: bool) -> None:
        """Change gradients only for the inherited O2 ``layer_readout`` module."""

        gate = self._require_o2_layer_readout()
        self._o2_gate_trainable = bool(enabled)
        gate.requires_grad_(bool(enabled))

    def schedule_o2_gate_unfreeze(self) -> None:
        """Activate the inherited O2 gate on the forward after its audit save."""

        self._o2_gate_pending_unfreeze = True

    def _activate_pending_o2_gate_unfreeze(self) -> None:
        if bool(getattr(self, "_o2_gate_pending_unfreeze", False)):
            self.set_o2_gate_trainable(True)
            self._o2_gate_pending_unfreeze = False
            logger.info("Activated original O2 layer_readout gradients at the gate boundary.")

    @contextmanager
    def response_adapters_disabled(self) -> Iterator[None]:
        """Temporarily expose the exact frozen Stage-1 carrier $h_l$.

        This does not change any Wan token or parameter.  It is used only by
        the carrier cache writer; regular action training and inference always
        retain the adapters in the query path.
        """

        prior = bool(getattr(self, "_response_adapters_enabled", True))
        self._response_adapters_enabled = False
        try:
            yield
        finally:
            self._response_adapters_enabled = prior

    def _build_multilayer_action_fusion_inputs(
        self,
        video_token_slice: slice | None = None,
    ) -> list[dict[str, Any]]:
        layer_states = super()._build_multilayer_action_fusion_inputs(
            video_token_slice=video_token_slice
        )
        if not bool(getattr(self, "_response_adapters_enabled", True)):
            return layer_states
        bank = self._require_response_adapters()
        bank.assert_layer_order(layer_states)
        output: list[dict[str, Any]] = []
        metrics: dict[str, float] = {}
        for state in layer_states:
            layer_idx = int(state["layer_idx"])
            backbone = state["backbone"]
            hidden = state["adapted"]
            if not isinstance(backbone, torch.Tensor) or not isinstance(hidden, torch.Tensor):
                raise TypeError("Dynamic O2 layer state tensors must be torch.Tensor values")
            # Deployment is explicitly $B_l=h_l+\alpha_l A_l(h_l)$ with
            # alpha=1.  This is before O2 memory_projection and never after
            # routed_query; no later Wan block is modified.
            residual = bank.residual(layer_idx, hidden)
            adapted = hidden + self.response_adapter_scale * residual
            if self.training:
                with torch.no_grad():
                    residual_norm = residual.detach().float().norm(dim=-1).mean()
                    carrier_norm = hidden.detach().float().norm(dim=-1).mean()
                    metrics[f"response_adapter/residual_norm_l{layer_idx}"] = float(
                        residual_norm.item()
                    )
                    metrics[f"response_adapter/carrier_norm_l{layer_idx}"] = float(
                        carrier_norm.item()
                    )
                    metrics[f"response_adapter/residual_ratio_l{layer_idx}"] = float(
                        (residual_norm / carrier_norm.clamp_min(1.0e-12)).item()
                    )
            output.append(
                {
                    **state,
                    "adapted": adapted,
                    "delta": adapted - backbone,
                }
            )
        self._last_response_adapter_metrics = metrics
        return output

    def configure_trainable_modules(self):
        super().configure_trainable_modules()
        bank = getattr(self, "response_adapters", None)
        if isinstance(bank, ResponseAdapterBank):
            bank.set_trainable(bool(getattr(self, "_response_adapters_trainable", True)))
        self._require_o2_layer_readout().requires_grad_(
            bool(getattr(self, "_o2_gate_trainable", True))
        )

    def training_loss(self, sample, tiled: bool = False):
        # The Dynamic trainer sets each request only after it saved the complete
        # frozen boundary state.  The following forward is therefore the first
        # legal adapter/gate update and keeps optimiser state continuous.
        self._activate_pending_response_adapter_unfreeze()
        self._activate_pending_o2_gate_unfreeze()
        self._last_response_adapter_metrics = {}
        loss, metrics = super().training_loss(sample, tiled=tiled)
        metrics = dict(metrics)
        metrics.update(self._last_response_adapter_metrics)
        metrics["response_adapter/scale"] = float(self.response_adapter_scale)
        return loss, metrics

    @staticmethod
    def _tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
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

    def response_adapter_state_sha256(self) -> str:
        return self._tensor_state_sha256(self._require_response_adapters().state_dict())

    def original_o2_tensor_sha256(self) -> str:
        """Hash every inherited O2 tensor while excluding only response adapters."""

        state: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("mot", self.mot),
            ("action_policy", self.action_policy),
            ("proprio_encoder", self.proprio_encoder),
        ):
            if module is None:
                continue
            for name, value in module.state_dict().items():
                state[f"{prefix}.{name}"] = value
        return self._tensor_state_sha256(state)

    def response_adapter_export_payload(
        self,
        *,
        source_identity: dict[str, Any],
        normalization_identity: dict[str, Any],
    ) -> dict[str, Any]:
        """Create the deployment-only Stage 1 hand-off; it deliberately has no Q."""

        state = {
            key: value.detach().cpu().clone()
            for key, value in self._require_response_adapters().state_dict().items()
        }
        return {
            "schema_version": 1,
            "track_id": "model3_o2_dynamic",
            "artifact_kind": "stage1_response_adapter_export",
            "method_id": self.method_id,
            "model_class": type(self).__name__,
            "stage1_predictor_input": "adapter_residual_only",
            "response_adapter_config": self.response_adapter_config,
            "response_adapter_state_dict": state,
            "response_adapter_state_sha256": self._tensor_state_sha256(state),
            "source_identity": dict(source_identity),
            "normalization_identity": dict(normalization_identity),
            "contains_predictors": False,
        }

    def save_response_adapter_export(
        self,
        path: str | Path,
        *,
        source_identity: dict[str, Any],
        normalization_identity: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self.response_adapter_export_payload(
            source_identity=source_identity,
            normalization_identity=normalization_identity,
        )
        torch.save(payload, path)
        payload["file_sha256"] = self._sha256(path)
        return payload

    def load_response_adapter_export(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        export_path = Path(path).expanduser().resolve()
        if not export_path.is_file():
            raise FileNotFoundError(f"Stage 1 adapter export does not exist: {export_path}")
        actual_sha256 = self._sha256(export_path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
            raise ValueError(
                "Stage 1 adapter export SHA mismatch: "
                f"expected {expected_sha256.lower()}, got {actual_sha256}."
            )
        payload = torch.load(export_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Stage 1 adapter export payload must be a dictionary")
        required = {
            "schema_version": 1,
            "track_id": "model3_o2_dynamic",
            "artifact_kind": "stage1_response_adapter_export",
            "method_id": self.method_id,
            "model_class": type(self).__name__,
            "stage1_predictor_input": "adapter_residual_only",
            "contains_predictors": False,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"invalid Stage 1 adapter export {key}: "
                    f"expected {expected!r}, got {payload.get(key)!r}"
                )
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
            raise ValueError("deployment adapter export must not contain Stage 1 predictors")
        if payload.get("response_adapter_config") != self.response_adapter_config:
            raise ValueError("Stage 1 adapter export architecture does not match Dynamic model")
        state = payload.get("response_adapter_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Stage 1 adapter export is missing response_adapter_state_dict")
        actual_state_sha = self._tensor_state_sha256(state)
        if actual_state_sha != payload.get("response_adapter_state_sha256"):
            raise ValueError("Stage 1 adapter export response-adapter tensor hash mismatch")
        self._require_response_adapters().load_state_dict(state, strict=True)
        self.response_adapter_export_identity = {
            "path": str(export_path),
            "sha256": actual_sha256,
            "state_sha256": actual_state_sha,
            "source_identity": payload.get("source_identity"),
            "normalization_identity": payload.get("normalization_identity"),
        }
        return payload

    def save_checkpoint(self, path, optimizer=None, step=None):
        """Save a strict deployment checkpoint with A but never a temporary Q."""

        bank = self._require_response_adapters()
        payload: dict[str, Any] = {
            "method_id": self.method_id,
            "model_class": type(self).__name__,
            "mot": self.mot.state_dict(),
            "action_policy_state_dict": self.action_policy.state_dict(),
            "action_policy_config": self.action_policy.config_dict(),
            "response_adapter_config": bank.configuration(),
            "response_adapter_state_dict": bank.state_dict(),
            "response_adapter_state_sha256": self.response_adapter_state_sha256(),
            "response_adapter_export_identity": self.response_adapter_export_identity,
            "model3_warmstart_identity": getattr(self, "model3_warmstart_identity", None),
            "original_o2_tensor_sha256": self.original_o2_tensor_sha256(),
            "contains_predictors": False,
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
        if not isinstance(payload, dict):
            raise ValueError("Dynamic checkpoint payload must be a dictionary")
        if payload.get("method_id") != self.method_id:
            raise ValueError(
                f"Dynamic checkpoint method_id must be {self.method_id!r}, "
                f"got {payload.get('method_id')!r}."
            )
        if payload.get("model_class") != type(self).__name__:
            raise ValueError(
                f"Dynamic checkpoint model_class must be {type(self).__name__!r}, "
                f"got {payload.get('model_class')!r}."
            )
        if payload.get("contains_predictors") is not False:
            raise ValueError("Dynamic deployment checkpoint must not contain Stage 1 predictors")
        if payload.get("response_adapter_config") != self.response_adapter_config:
            raise ValueError("Dynamic checkpoint response-adapter architecture mismatch")
        response_state = payload.get("response_adapter_state_dict")
        if not isinstance(response_state, dict):
            raise ValueError("Dynamic checkpoint is missing response_adapter_state_dict")
        if self._tensor_state_sha256(response_state) != payload.get("response_adapter_state_sha256"):
            raise ValueError("Dynamic checkpoint response-adapter tensor hash mismatch")
        policy_config = payload.get("action_policy_config")
        if not isinstance(policy_config, dict) or policy_config.get("method_id") != self.action_policy_method_id:
            raise ValueError("Dynamic checkpoint action policy must remain the original O2 policy")
        for key, module in (("mot", self.mot), ("action_policy_state_dict", self.action_policy)):
            state = payload.get(key)
            if not isinstance(state, dict):
                raise ValueError(f"Dynamic checkpoint is missing valid {key}")
            module.load_state_dict(state, strict=True)
        if self.proprio_encoder is not None:
            state = payload.get("proprio_encoder")
            if not isinstance(state, dict):
                raise ValueError("Dynamic checkpoint is missing proprio_encoder")
            self.proprio_encoder.load_state_dict(state, strict=True)
        self._require_response_adapters().load_state_dict(response_state, strict=True)
        self.response_adapter_export_identity = payload.get("response_adapter_export_identity")
        self.model3_warmstart_identity = payload.get("model3_warmstart_identity")
        if optimizer is not None and isinstance(payload.get("optimizer"), dict):
            optimizer.load_state_dict(payload["optimizer"])
        return payload
