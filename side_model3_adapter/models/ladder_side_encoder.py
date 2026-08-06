"""Five-stage frozen-Wan Ladder encoder and identity-initialized trace fusion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class LadderStageBlock(nn.Module):
    """Propose one cross/self/FFN slot update and apply a scalar residual gate."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        residual_gate_init: float,
    ) -> None:
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

    def forward(
        self,
        slots: torch.Tensor,
        memory: torch.Tensor,
        layer_embedding: torch.Tensor,
    ) -> torch.Tensor:
        proposal = slots + layer_embedding
        cross, _ = self.cross_attention(
            self.cross_query_norm(proposal),
            self.cross_memory_norm(memory),
            self.cross_memory_norm(memory),
            need_weights=False,
        )
        proposal = proposal + cross
        normalized = self.self_norm(proposal)
        self_update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        proposal = proposal + self_update
        proposal = proposal + self.ffn(proposal)
        return slots + self.residual_gate * (proposal - slots)


class O2StyleTraceFusion(nn.Module):
    """Fuse four early Ladder snapshots around an exact final-state identity."""

    def __init__(self, *, num_stages: int, num_slots: int, hidden_dim: int) -> None:
        super().__init__()
        if num_stages < 2:
            raise ValueError("trace fusion requires at least two stages")
        self.num_stages = int(num_stages)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.early_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_stages - 1)]
        )
        self.early_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_stages - 1)]
        )
        self.query_gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, num_slots, 1)) for _ in range(num_stages - 1)]
        )
        for projection in self.early_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        if trace.ndim != 4 or tuple(trace.shape[1:]) != (
            self.num_stages,
            self.num_slots,
            self.hidden_dim,
        ):
            raise ValueError(
                "trace must be "
                f"[B,{self.num_stages},{self.num_slots},{self.hidden_dim}], "
                f"got {tuple(trace.shape)}"
            )
        fused = trace[:, -1]
        for position, (norm, projection, gate) in enumerate(
            zip(self.early_norms, self.early_projections, self.query_gates)
        ):
            residual = projection(norm(trace[:, position]))
            fused = fused + (1.0 + torch.tanh(gate)) * residual
        return fused


class LadderSideEncoder(nn.Module):
    """Read ordered raw Wan states into one recurrent bank of control slots."""

    def __init__(
        self,
        *,
        video_hidden_dim: int,
        proprio_dim: int,
        layer_indices: Sequence[int] = (8, 16, 20, 24, 29),
        num_slots: int = 64,
        hidden_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        residual_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        normalized_layers = tuple(int(value) for value in layer_indices)
        if normalized_layers != tuple(sorted(set(normalized_layers))):
            raise ValueError("layer_indices must be unique and ordered")
        self.video_hidden_dim = int(video_hidden_dim)
        self.proprio_dim = int(proprio_dim)
        self.layer_indices = normalized_layers
        self.num_stages = len(normalized_layers)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.residual_gate_init = float(residual_gate_init)

        self.control_slots = nn.Parameter(
            torch.randn(1, self.num_slots, self.hidden_dim) * 0.02
        )
        self.proprio_norm = nn.LayerNorm(self.proprio_dim, elementwise_affine=False)
        self.proprio_projection = nn.Linear(self.proprio_dim, self.hidden_dim)
        self.layer_embeddings = nn.Parameter(
            torch.zeros(self.num_stages, 1, 1, self.hidden_dim)
        )
        self.memory_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.video_hidden_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                )
                for _ in self.layer_indices
            ]
        )
        self.stages = nn.ModuleList(
            [
                LadderStageBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    residual_gate_init=self.residual_gate_init,
                )
                for _ in self.layer_indices
            ]
        )
        self.trace_fusion = O2StyleTraceFusion(
            num_stages=self.num_stages,
            num_slots=self.num_slots,
            hidden_dim=self.hidden_dim,
        )

    def config_dict(self) -> dict[str, object]:
        return {
            "video_hidden_dim": self.video_hidden_dim,
            "proprio_dim": self.proprio_dim,
            "layer_indices": list(self.layer_indices),
            "num_slots": self.num_slots,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "residual_gate_init": self.residual_gate_init,
        }

    def forward(
        self,
        layer_states: Sequence[torch.Tensor],
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(layer_states) != self.num_stages:
            raise ValueError(
                f"expected {self.num_stages} Wan states, got {len(layer_states)}"
            )
        first = layer_states[0]
        if first.ndim != 3 or int(first.shape[-1]) != self.video_hidden_dim:
            raise ValueError(
                f"Wan states must be [B,S,{self.video_hidden_dim}], got {tuple(first.shape)}"
            )
        batch_size = int(first.shape[0])
        if proprio.ndim != 2 or tuple(proprio.shape) != (batch_size, self.proprio_dim):
            raise ValueError(
                f"proprio must be [B,{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        slots = self.control_slots.expand(batch_size, -1, -1)
        proprio_token = self.proprio_projection(
            self.proprio_norm(proprio.to(dtype=slots.dtype))
        )
        slots = slots + proprio_token.unsqueeze(1)

        trace = []
        for position, (raw_state, projection, stage) in enumerate(
            zip(layer_states, self.memory_projections, self.stages)
        ):
            if raw_state.ndim != 3 or tuple(raw_state.shape[:1]) != (batch_size,):
                raise ValueError("all Wan states must be [B,S,D] with the same batch")
            if int(raw_state.shape[-1]) != self.video_hidden_dim:
                raise ValueError("Wan state width changed across selected layers")
            # Direct Side-Model3 copy: the only boundary change is retaining
            # gradients to the three online Wan residual adapters.
            memory = projection(raw_state.to(dtype=slots.dtype))
            slots = stage(slots, memory, self.layer_embeddings[position])
            trace.append(slots)

        stacked_trace = torch.stack(trace, dim=1)
        return self.trace_fusion(stacked_trace), stacked_trace
