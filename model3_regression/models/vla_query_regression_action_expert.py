"""Direct action regression from the unchanged Model3 recurrent query memory."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import nn

from model3.models.vla_query_dit_action_expert import (
    LayerWiseRecurrentActionQueryEncoder,
)


class DirectRegressionDecoderBlock(nn.Module):
    """Refine learned action slots against the full recurrent query memory."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.action_self_norm = nn.LayerNorm(hidden_dim)
        self.action_self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.action_cross_norm = nn.LayerNorm(hidden_dim)
        self.query_memory_norm = nn.LayerNorm(hidden_dim)
        self.query_cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )

    def forward(self, action_slots: torch.Tensor, query_memory: torch.Tensor) -> torch.Tensor:
        normalized_slots = self.action_self_norm(action_slots)
        self_out, _ = self.action_self_attention(
            normalized_slots,
            normalized_slots,
            normalized_slots,
            need_weights=False,
        )
        action_slots = action_slots + self_out

        cross_out, _ = self.query_cross_attention(
            query=self.action_cross_norm(action_slots),
            key=self.query_memory_norm(query_memory),
            value=self.query_memory_norm(query_memory),
            need_weights=False,
        )
        action_slots = action_slots + cross_out
        return action_slots + self.ffn(action_slots)


class VLAQueryRegressionActionExpert(nn.Module):
    """Predict one normalized action chunk without noise or iterative sampling."""

    method_id = "model3_regression_recurrent_query_l1_v1"
    flow_matching_layer_state_policy = False

    def __init__(
        self,
        *,
        video_hidden_dim: int,
        action_dim: int,
        num_fusion_layers: int,
        proprio_dim: Optional[int],
        query_dim: int = 512,
        num_action_queries: int = 64,
        query_num_heads: int = 8,
        query_bridge_depth: int = 2,
        query_ffn_multiplier: int = 4,
        query_dropout: float = 0.0,
        feature_source: str = "adapted",
        regression_hidden_dim: int = 512,
        regression_num_heads: int = 8,
        regression_num_layers: int = 2,
        regression_ffn_multiplier: int = 4,
        regression_dropout: float = 0.0,
        action_horizon: int = 8,
    ) -> None:
        super().__init__()
        if regression_hidden_dim != query_dim:
            raise ValueError("regression_hidden_dim must equal query_dim for the matched query ablation")
        if regression_num_layers <= 0 or regression_ffn_multiplier <= 0:
            raise ValueError("regression decoder depth and FFN multiplier must be positive")
        if regression_num_heads <= 0 or regression_hidden_dim % regression_num_heads != 0:
            raise ValueError("regression_hidden_dim must be divisible by regression_num_heads")
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")

        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.regression_hidden_dim = int(regression_hidden_dim)
        self.regression_num_heads = int(regression_num_heads)
        self.regression_num_layers = int(regression_num_layers)
        self.regression_ffn_multiplier = int(regression_ffn_multiplier)
        self.regression_dropout = float(regression_dropout)
        self.query_encoder = LayerWiseRecurrentActionQueryEncoder(
            video_hidden_dim=video_hidden_dim,
            query_dim=query_dim,
            num_fusion_layers=num_fusion_layers,
            num_action_queries=num_action_queries,
            num_heads=query_num_heads,
            bridge_depth=query_bridge_depth,
            feature_source=feature_source,
            proprio_dim=proprio_dim,
            ffn_multiplier=query_ffn_multiplier,
            dropout=query_dropout,
        )
        self.action_slots = nn.Parameter(
            torch.randn(1, self.action_horizon, self.regression_hidden_dim) * 0.02
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DirectRegressionDecoderBlock(
                    hidden_dim=self.regression_hidden_dim,
                    num_heads=self.regression_num_heads,
                    ffn_multiplier=self.regression_ffn_multiplier,
                    dropout=self.regression_dropout,
                )
                for _ in range(self.regression_num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.regression_hidden_dim)
        self.action_projection = nn.Linear(self.regression_hidden_dim, self.action_dim)

    @property
    def training_action_horizon(self) -> int:
        return self.action_horizon

    def config_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "video_hidden_dim": self.query_encoder.video_hidden_dim,
            "action_dim": self.action_dim,
            "num_fusion_layers": self.query_encoder.num_fusion_layers,
            "proprio_dim": self.query_encoder.proprio_dim,
            "query_dim": self.query_encoder.query_dim,
            "num_action_queries": self.query_encoder.num_action_queries,
            "query_num_heads": self.query_encoder.num_heads,
            "query_bridge_depth": self.query_encoder.bridge_depth,
            "query_ffn_multiplier": self.query_encoder.ffn_multiplier,
            "query_dropout": self.query_encoder.dropout,
            "feature_source": self.query_encoder.feature_source,
            "regression_hidden_dim": self.regression_hidden_dim,
            "regression_num_heads": self.regression_num_heads,
            "regression_num_layers": self.regression_num_layers,
            "regression_ffn_multiplier": self.regression_ffn_multiplier,
            "regression_dropout": self.regression_dropout,
            "action_horizon": self.action_horizon,
        }

    def encode_queries(
        self,
        layer_states: Sequence[dict[str, Any]],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_encoder(layer_states, proprio)

    def predict_from_queries(self, query_memory: torch.Tensor) -> torch.Tensor:
        if query_memory.ndim != 3:
            raise ValueError("query_memory must be [B,Q,D]")
        if int(query_memory.shape[1]) != self.query_encoder.num_action_queries:
            raise ValueError(
                f"query_memory must contain {self.query_encoder.num_action_queries} queries"
            )
        if int(query_memory.shape[2]) != self.regression_hidden_dim:
            raise ValueError(
                f"query_memory width must be {self.regression_hidden_dim}, got {query_memory.shape[2]}"
            )
        action_slots = self.action_slots.expand(query_memory.shape[0], -1, -1)
        query_memory = query_memory.to(dtype=action_slots.dtype)
        for block in self.decoder_blocks:
            action_slots = block(action_slots, query_memory)
        return self.action_projection(self.output_norm(action_slots))

    def forward(
        self,
        layer_states: Sequence[dict[str, Any]],
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_memory, _ = self.encode_queries(layer_states, proprio)
        return self.predict_from_queries(query_memory)

    @staticmethod
    def _validate_direct_call(action_horizon: int, expected_horizon: int, steps: int) -> None:
        if int(action_horizon) != int(expected_horizon):
            raise ValueError(
                f"regression policy requires action_horizon={expected_horizon}, got {action_horizon}"
            )
        if int(steps) != 1:
            raise ValueError(
                "direct regression requires num_inference_steps=1; this is one policy call, not a solver"
            )

    @torch.no_grad()
    def sample_from_queries(
        self,
        query_memory: torch.Tensor,
        *,
        action_horizon: int,
        scheduler=None,
        num_inference_steps: int = 1,
        generator: Optional[torch.Generator] = None,
        noise_device: Optional[torch.device | str] = None,
        sigma_shift: Optional[float] = None,
    ) -> torch.Tensor:
        del scheduler, generator, noise_device, sigma_shift
        self._validate_direct_call(action_horizon, self.action_horizon, num_inference_steps)
        return self.predict_from_queries(query_memory)

    @torch.no_grad()
    def sample(
        self,
        layer_states: Sequence[dict[str, Any]],
        *,
        action_horizon: int,
        scheduler=None,
        num_inference_steps: int = 1,
        proprio: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        noise_device: Optional[torch.device | str] = None,
        sigma_shift: Optional[float] = None,
    ) -> torch.Tensor:
        del scheduler, generator, noise_device, sigma_shift
        self._validate_direct_call(action_horizon, self.action_horizon, num_inference_steps)
        return self(layer_states, proprio)
