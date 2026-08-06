"""Decode predicted future control state into a fixed pooled VAE latent delta."""

from __future__ import annotations

import torch
from torch import nn


class MultiHorizonFutureLatentChangeHead(nn.Module):
    SUPPORTED_HORIZONS = (4, 8)

    def __init__(
        self,
        *,
        latent_channels: int,
        grid_height: int,
        grid_width: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = self.grid_height * self.grid_width
        self.spatial_queries = nn.Parameter(
            torch.randn(1, self.num_queries, self.hidden_dim) * 0.02
        )
        self.row_embeddings = nn.Parameter(
            torch.zeros(1, self.grid_height, 1, self.hidden_dim)
        )
        self.column_embeddings = nn.Parameter(
            torch.zeros(1, 1, self.grid_width, self.hidden_dim)
        )
        self.horizon_embeddings = nn.Parameter(
            torch.zeros(len(self.SUPPORTED_HORIZONS), 1, self.hidden_dim)
        )
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)
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
        self.output_projection = nn.Linear(self.hidden_dim, self.latent_channels)

    def forward(self, predicted_state: torch.Tensor, *, horizon: int) -> torch.Tensor:
        if horizon not in self.SUPPORTED_HORIZONS:
            raise ValueError(f"horizon must be one of {self.SUPPORTED_HORIZONS}")
        if predicted_state.ndim != 3 or int(predicted_state.shape[-1]) != self.hidden_dim:
            raise ValueError("predicted_state must be [B,Q,D]")
        horizon_position = self.SUPPORTED_HORIZONS.index(horizon)
        position = (self.row_embeddings + self.column_embeddings).reshape(
            1, self.num_queries, self.hidden_dim
        )
        queries = self.spatial_queries + position + self.horizon_embeddings[horizon_position]
        queries = queries.expand(int(predicted_state.shape[0]), -1, -1)
        cross, _ = self.cross_attention(
            self.query_norm(queries),
            self.state_norm(predicted_state),
            self.state_norm(predicted_state),
            need_weights=False,
        )
        queries = queries + cross
        normalized = self.self_norm(queries)
        self_update, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + self_update
        queries = queries + self.ffn(queries)
        latent = self.output_projection(queries)
        return latent.transpose(1, 2).reshape(
            int(predicted_state.shape[0]),
            self.latent_channels,
            self.grid_height,
            self.grid_width,
        )
