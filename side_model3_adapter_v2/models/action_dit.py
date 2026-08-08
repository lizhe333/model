"""The Model3 Action-DiT isolated from the removed recurrent query encoder."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from model3.third_party.light_wam.src.lightwam.models.wan22.action_dit import (
    ActionHead,
)
from model3.third_party.light_wam.src.lightwam.models.wan22.helpers.gradient import (
    gradient_checkpoint_forward,
)
from model3.third_party.light_wam.src.lightwam.models.wan22.wan_video_dit import (
    DiTBlock,
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)


class SideModel3ActionDiT(nn.Module):
    """Predict action-flow velocity from 64 external side-path state slots."""

    def __init__(
        self,
        *,
        action_dim: int,
        context_dim: int = 512,
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        num_heads: int = 8,
        attn_head_dim: int = 64,
        num_layers: int = 16,
        freq_dim: int = 256,
        eps: float = 1.0e-6,
        action_horizon: int = 8,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim != num_heads * attn_head_dim:
            raise ValueError("hidden_dim must equal num_heads * attn_head_dim")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(num_layers)
        self.freq_dim = int(freq_dim)
        self.eps = float(eps)
        self.action_horizon = int(action_horizon)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 6),
        )
        self.context_norm = nn.LayerNorm(self.context_dim, eps=self.eps)
        self.context_modulation = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 6),
        )
        self.context_projection = nn.Linear(self.context_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=self.eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = ActionHead(
            hidden_dim=self.hidden_dim,
            out_dim=self.action_dim,
            eps=self.eps,
        )
        self.freqs = precompute_freqs_cis(
            self.attn_head_dim,
            end=self.action_horizon,
        )

    @property
    def training_action_horizon(self) -> int:
        return self.action_horizon

    def config_dict(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "attn_head_dim": self.attn_head_dim,
            "num_layers": self.num_layers,
            "freq_dim": self.freq_dim,
            "eps": self.eps,
            "action_horizon": self.action_horizon,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }

    def _validate_inputs(
        self,
        action_state: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        if noisy_action.ndim != 3 or tuple(noisy_action.shape[1:]) != (
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"noisy_action must be [B,{self.action_horizon},{self.action_dim}]"
            )
        if action_state.ndim != 3 or int(action_state.shape[0]) != int(
            noisy_action.shape[0]
        ) or int(action_state.shape[-1]) != self.context_dim:
            raise ValueError("action_state must be [B,Q,D] and align with actions")
        if timestep.ndim != 1 or int(timestep.shape[0]) != int(noisy_action.shape[0]):
            raise ValueError("timestep must be [B]")

    def forward(
        self,
        action_state: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(action_state, noisy_action, timestep)
        timestep = timestep.to(device=noisy_action.device, dtype=noisy_action.dtype)
        time_embed = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep)
        )
        normalized_context = self.context_norm(
            action_state.to(dtype=noisy_action.dtype)
        )
        context_summary = normalized_context.mean(dim=1)
        modulation = self.time_projection(time_embed) + self.context_modulation(
            context_summary
        )
        modulation = modulation.unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(noisy_action)
        context = self.context_projection(normalized_context)
        context_mask = torch.ones(
            (tokens.shape[0], tokens.shape[1], context.shape[1]),
            dtype=torch.bool,
            device=tokens.device,
        )
        freqs = self.freqs.view(self.action_horizon, 1, -1).to(tokens.device)
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                tokens = gradient_checkpoint_forward(
                    block,
                    True,
                    tokens,
                    context,
                    modulation,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                tokens = block(
                    tokens,
                    context,
                    modulation,
                    freqs,
                    context_mask=context_mask,
                )
        return self.head(tokens, time_embed)

    @torch.no_grad()
    def sample(
        self,
        action_state: torch.Tensor,
        *,
        action_horizon: int,
        scheduler,
        num_inference_steps: int,
        generator: Optional[torch.Generator] = None,
        noise_device: Optional[torch.device | str] = None,
        sigma_shift: Optional[float] = None,
    ) -> torch.Tensor:
        if int(action_horizon) != self.action_horizon:
            raise ValueError(
                f"Side-Model3 requires action_horizon={self.action_horizon}, "
                f"got {action_horizon}"
            )
        random_device = (
            action_state.device if noise_device is None else torch.device(noise_device)
        )
        action = torch.randn(
            (action_state.shape[0], self.action_horizon, self.action_dim),
            generator=generator,
            device=random_device,
            dtype=torch.float32,
        ).to(device=action_state.device, dtype=action_state.dtype)
        timesteps, deltas = scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=action_state.device,
            dtype=action_state.dtype,
            shift_override=sigma_shift,
        )
        for step_t, delta in zip(timesteps, deltas):
            timestep = step_t.expand(action_state.shape[0])
            velocity = self(action_state, action, timestep)
            action = scheduler.step(velocity, delta, action)
        return action

    def load_model3_action_dit_state(self, model3_policy_state: dict[str, torch.Tensor]) -> None:
        """Warm-start only tensors shared with Model3's Action-DiT."""

        renamed_prefixes = {
            "query_condition_norm.": "context_norm.",
            "query_modulation.": "context_modulation.",
            "query_context_projection.": "context_projection.",
        }
        own_state = self.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        for source_name, tensor in model3_policy_state.items():
            target_name = source_name
            for source_prefix, target_prefix in renamed_prefixes.items():
                if source_name.startswith(source_prefix):
                    target_name = target_prefix + source_name[len(source_prefix) :]
                    break
            if target_name in own_state and own_state[target_name].shape == tensor.shape:
                compatible[target_name] = tensor
        incompatible = self.load_state_dict(compatible, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError("unexpected tensors in Model3 Action-DiT warm start")
        missing = set(incompatible.missing_keys)
        if missing:
            raise ValueError(
                "Model3 Action-DiT warm start is incomplete: "
                + ", ".join(sorted(missing))
            )
