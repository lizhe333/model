"""Model3 Action-DiT with a layer-aware recurrent-query readout."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import nn

from model3.models.vla_query_dit_action_expert import VLAQueryDiTActionExpert


class LayerSeparableGatedResidualReadout(nn.Module):
    """Add independently gated earlier-layer residuals to the final query state."""

    def __init__(self, *, num_layers: int, query_dim: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("layer-aware readout requires at least two query-trace layers")
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")

        self.num_layers = int(num_layers)
        self.query_dim = int(query_dim)
        self.num_residual_layers = self.num_layers - 1

        self.earlier_norms = nn.ModuleList(
            [nn.LayerNorm(self.query_dim) for _ in range(self.num_residual_layers)]
        )
        self.value_projections = nn.ModuleList(
            [nn.Linear(self.query_dim, self.query_dim) for _ in range(self.num_residual_layers)]
        )
        self.gate_projections = nn.ModuleList(
            [nn.Linear(self.query_dim * 2, 1) for _ in range(self.num_residual_layers)]
        )
        self.final_norm = nn.LayerNorm(self.query_dim)
        self.residual_scales = nn.Parameter(torch.zeros(self.num_residual_layers))

    def _validate_trace(self, query_trace: torch.Tensor) -> None:
        if query_trace.ndim != 4:
            raise ValueError(
                "query_trace must be [B,L,Q,D], "
                f"got shape {tuple(query_trace.shape)}"
            )
        if int(query_trace.shape[1]) != self.num_layers:
            raise ValueError(
                f"query_trace must contain {self.num_layers} layers, "
                f"got {query_trace.shape[1]}"
            )
        if int(query_trace.shape[-1]) != self.query_dim:
            raise ValueError(
                f"query_trace width must be {self.query_dim}, "
                f"got {query_trace.shape[-1]}"
            )

    def forward(self, query_trace: torch.Tensor) -> torch.Tensor:
        self._validate_trace(query_trace)

        final_query = query_trace[:, -1]
        normalized_final = self.final_norm(final_query)
        routed_query = final_query

        for layer_position in range(self.num_residual_layers):
            earlier_query = query_trace[:, layer_position]
            layer_norm = self.earlier_norms[layer_position]
            delta = layer_norm(earlier_query - final_query)
            value = self.value_projections[layer_position](delta)
            gate_input = torch.cat(
                [layer_norm(earlier_query), normalized_final],
                dim=-1,
            )
            gate = torch.sigmoid(self.gate_projections[layer_position](gate_input))
            strength = torch.tanh(self.residual_scales[layer_position])
            routed_query = routed_query + strength * gate * value

        return routed_query


class VLAQueryLayerAwareDiTActionExpert(VLAQueryDiTActionExpert):
    """Preserve Model3 Action-DiT while exposing its recurrent query trace."""

    method_id = "model3_o2_layer_aware_query_flow_v1"

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
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        num_heads: int = 8,
        attn_head_dim: int = 64,
        num_layers: int = 16,
        freq_dim: int = 256,
        eps: float = 1.0e-6,
        action_horizon: int = 8,
        use_gradient_checkpointing: bool = False,
        readout_type: str = "layer_separable_gated_residual",
        readout_gate_type: str = "querywise_scalar",
        readout_identity_init: bool = True,
    ) -> None:
        if readout_type != "layer_separable_gated_residual":
            raise ValueError(f"unsupported O2 readout_type: {readout_type!r}")
        if readout_gate_type != "querywise_scalar":
            raise ValueError(f"unsupported O2 readout_gate_type: {readout_gate_type!r}")
        if not readout_identity_init:
            raise ValueError("O2 requires the exact q3 identity initialization")
        super().__init__(
            video_hidden_dim=video_hidden_dim,
            action_dim=action_dim,
            num_fusion_layers=num_fusion_layers,
            proprio_dim=proprio_dim,
            query_dim=query_dim,
            num_action_queries=num_action_queries,
            query_num_heads=query_num_heads,
            query_bridge_depth=query_bridge_depth,
            query_ffn_multiplier=query_ffn_multiplier,
            query_dropout=query_dropout,
            feature_source=feature_source,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=num_layers,
            freq_dim=freq_dim,
            eps=eps,
            action_horizon=action_horizon,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.layer_readout = LayerSeparableGatedResidualReadout(
            num_layers=self.query_encoder.num_fusion_layers,
            query_dim=self.query_encoder.query_dim,
        )

    def config_dict(self) -> dict[str, Any]:
        config = super().config_dict()
        config.update(
            {
                "query_trace_readout": "layer_separable_gated_residual",
                "readout_num_layers": self.layer_readout.num_layers,
                "readout_query_dim": self.layer_readout.query_dim,
                "readout_gate_type": "querywise_scalar",
                "readout_identity_init": True,
            }
        )
        return config

    def encode_queries(
        self,
        layer_states: Sequence[dict[str, Any]],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, query_trace = super().encode_queries(layer_states, proprio)
        routed_query = self.layer_readout(query_trace)
        return routed_query, query_trace
