"""Multi-horizon action encoding and latent control-state transition."""

from __future__ import annotations

import torch
from torch import nn


class MultiHorizonActionChunkEncoder(nn.Module):
    SUPPORTED_HORIZONS = (4, 8)

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 512,
        max_horizon: int = 8,
    ) -> None:
        super().__init__()
        if max_horizon != 8:
            raise ValueError("Side-Model3 v1 action encoder requires max_horizon=8")
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_horizon = int(max_horizon)
        self.action_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.max_horizon, self.hidden_dim)
        )
        self.horizon_embeddings = nn.Parameter(
            torch.zeros(len(self.SUPPORTED_HORIZONS), 1, self.hidden_dim)
        )

    def forward(self, action: torch.Tensor, *, horizon: int) -> torch.Tensor:
        if horizon not in self.SUPPORTED_HORIZONS:
            raise ValueError(f"horizon must be one of {self.SUPPORTED_HORIZONS}")
        if action.ndim != 3 or int(action.shape[-1]) != self.action_dim:
            raise ValueError(f"action must be [B,T,{self.action_dim}]")
        if int(action.shape[1]) < horizon:
            raise ValueError(f"action chunk has fewer than {horizon} steps")
        horizon_position = self.SUPPORTED_HORIZONS.index(horizon)
        tokens = self.action_projection(action[:, :horizon])
        return (
            tokens
            + self.position_embeddings[:, :horizon]
            + self.horizon_embeddings[horizon_position]
        )


class TransitionBlock(nn.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.action_cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.state_self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        cross, _ = self.action_cross_attention(
            self.state_norm(state),
            self.action_norm(action_tokens),
            self.action_norm(action_tokens),
            need_weights=False,
        )
        state = state + cross
        normalized = self.self_norm(state)
        self_update, _ = self.state_self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        state = state + self_update
        return state + self.ffn(state)


class LatentTransitionPredictor(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_blocks != 2:
            raise ValueError("Side-Model3 v1 transition predictor requires two blocks")
        self.hidden_dim = int(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransitionBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self,
        control_state: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if control_state.ndim != 3 or int(control_state.shape[-1]) != self.hidden_dim:
            raise ValueError("control_state must be [B,Q,D]")
        if action_tokens.ndim != 3 or tuple(action_tokens.shape[::2]) != (
            control_state.shape[0],
            self.hidden_dim,
        ):
            raise ValueError("action tokens must be [B,T,D] and align with state")
        state = control_state
        for block in self.blocks:
            state = block(state, action_tokens)
        return control_state + self.output_projection(state)
