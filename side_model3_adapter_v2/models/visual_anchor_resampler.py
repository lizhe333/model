"""Action-only visual anchors and identity-initialized visual residual fusion."""

from __future__ import annotations

import torch
from torch import nn


class VisualAnchorResampler(nn.Module):
    def __init__(
        self,
        *,
        video_hidden_dim: int,
        hidden_dim: int = 512,
        num_anchors: int = 16,
        num_heads: int = 8,
        ffn_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.video_hidden_dim = int(video_hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_anchors = int(num_anchors)
        self.memory_projection = nn.Sequential(
            nn.Linear(self.video_hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.visual_queries = nn.Parameter(
            torch.randn(1, self.num_anchors, self.hidden_dim) * 0.02
        )
        self.cross_query_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_memory_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.self_norm = nn.LayerNorm(self.hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, self.hidden_dim),
        )

    def forward(self, h29_tokens: torch.Tensor) -> torch.Tensor:
        if h29_tokens.ndim != 3 or int(h29_tokens.shape[-1]) != self.video_hidden_dim:
            raise ValueError(
                f"H29 tokens must be [B,S,{self.video_hidden_dim}], "
                f"got {tuple(h29_tokens.shape)}"
            )
        # The adapter variant keeps this copied visual route differentiable to
        # the online Wan adapters; original Wan tensors remain frozen.
        memory = self.memory_projection(h29_tokens)
        anchors = self.visual_queries.expand(int(memory.shape[0]), -1, -1)
        cross, _ = self.cross_attention(
            self.cross_query_norm(anchors),
            self.cross_memory_norm(memory),
            self.cross_memory_norm(memory),
            need_weights=False,
        )
        anchors = anchors + cross
        normalized = self.self_norm(anchors)
        self_update, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        anchors = anchors + self_update
        return anchors + self.ffn(anchors)


class VisualAnchorActionFusion(nn.Module):
    """Read visual anchors without changing Action-DiT context length."""

    def __init__(self, *, num_slots: int = 64, hidden_dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.anchor_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.query_gate = nn.Parameter(torch.zeros(1, self.num_slots, 1))

    def forward(
        self,
        control_state: torch.Tensor,
        visual_anchors: torch.Tensor,
    ) -> torch.Tensor:
        if control_state.ndim != 3 or tuple(control_state.shape[1:]) != (
            self.num_slots,
            self.hidden_dim,
        ):
            raise ValueError(
                f"control_state must be [B,{self.num_slots},{self.hidden_dim}]"
            )
        if visual_anchors.ndim != 3 or int(visual_anchors.shape[0]) != int(
            control_state.shape[0]
        ) or int(visual_anchors.shape[-1]) != self.hidden_dim:
            raise ValueError("visual anchors must be [B,A,D] and align with control state")
        residual, _ = self.cross_attention(
            self.query_norm(control_state),
            self.anchor_norm(visual_anchors),
            self.anchor_norm(visual_anchors),
            need_weights=False,
        )
        return control_state + torch.tanh(self.query_gate) * residual
