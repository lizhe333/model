"""Small, training-only action-conditioned response predictors for Stage 1."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_timestep_embedding(timestep: torch.Tensor, width: int) -> torch.Tensor:
    """Deterministic scalar-timestep embedding used by every $Q_l$."""

    if timestep.ndim != 1:
        raise ValueError(f"timestep must be [B], got {tuple(timestep.shape)}")
    half = max(int(width) // 2, 1)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.float()[:, None] * frequencies[None]
    embedded = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(embedded, (0, max(int(width) - embedded.shape[-1], 0)))[:, :width]


class TokenResponsePredictor(nn.Module):
    """Predict four E0 response stages from all current-layer tokens and action.

    The predictor receives the complete $[B,392,1536]$ carrier.  It uses a
    fixed $14\times28\rightarrow4\times7$ spatial compression internally only
    to stay deliberately small; its target is the E0 global response
    $[B,4,256]$, not a raw hidden tensor.
    """

    valid_conditions = {"full", "action_only", "state_only"}

    def __init__(
        self,
        *,
        hidden_dim: int = 1536,
        action_horizon: int = 8,
        action_dim: int = 7,
        width: int = 128,
        target_dim: int = 256,
        grid_height: int = 14,
        grid_width: int = 28,
        condition: str = "full",
    ) -> None:
        super().__init__()
        if condition not in self.valid_conditions:
            raise ValueError(f"unsupported response predictor condition: {condition!r}")
        self.hidden_dim = int(hidden_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.width = int(width)
        self.target_dim = int(target_dim)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.condition = condition
        if (self.hidden_dim, self.action_horizon, self.action_dim, self.target_dim) != (
            1536,
            8,
            7,
            256,
        ):
            raise ValueError(
                "Dynamic O2 Stage 1 requires hidden=1536, action=[8,7], and E0 target=256."
            )
        self.state_norm = nn.LayerNorm(self.hidden_dim)
        self.state_projection = nn.Linear(self.hidden_dim, self.width)
        self.state_summary = nn.Sequential(
            nn.LayerNorm(self.width), nn.SiLU(), nn.Linear(self.width, self.width)
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(self.action_horizon * self.action_dim),
            nn.Linear(self.action_horizon * self.action_dim, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )
        self.timestep_projection = nn.Sequential(
            nn.Linear(self.width, self.width), nn.SiLU(), nn.Linear(self.width, self.width)
        )
        self.stage_embedding = nn.Parameter(torch.empty(4, self.width))
        self.head = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.target_dim),
        )
        nn.init.normal_(self.stage_embedding, std=0.02)

    def _validate(self, carrier: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor) -> None:
        expected_tokens = self.grid_height * self.grid_width
        if carrier.ndim != 3 or tuple(carrier.shape[1:]) != (expected_tokens, self.hidden_dim):
            raise ValueError(
                "response carrier must be [B,392,1536], "
                f"got {tuple(carrier.shape)}"
            )
        if action.ndim != 3 or tuple(action.shape[1:]) != (
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError(
                "normalized Stage 1 action must be [B,8,7], "
                f"got {tuple(action.shape)}"
            )
        if timestep.shape != (carrier.shape[0],):
            raise ValueError(
                f"timestep must be [{carrier.shape[0]}], got {tuple(timestep.shape)}"
            )

    def forward(
        self,
        carrier: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(carrier, action, timestep)
        batch = carrier.shape[0]
        if self.condition == "action_only":
            state = carrier.new_zeros((batch, self.width), dtype=torch.float32)
        else:
            token_features = self.state_projection(self.state_norm(carrier.float()))
            grid = token_features.view(
                batch, self.grid_height, self.grid_width, self.width
            ).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(grid, (4, 7)).flatten(2).mean(dim=-1)
            state = self.state_summary(pooled)
        if self.condition == "state_only":
            action_feature = state.new_zeros((batch, self.width))
        else:
            action_feature = self.action_projection(action.float().flatten(1))
        time_feature = self.timestep_projection(
            sinusoidal_timestep_embedding(timestep.float() / 1000.0, self.width)
        )
        fused = (
            state[:, None]
            + action_feature[:, None]
            + time_feature[:, None]
            + self.stage_embedding[None]
        )
        return self.head(fused)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
