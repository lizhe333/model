"""Model5 action policy: recurrent VLA-style queries with flow decoding."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from model5.third_party.light_wam.src.lightwam.models.wan22.action_dit import (
    ActionHead,
)
from model5.third_party.light_wam.src.lightwam.models.wan22.helpers.gradient import (
    gradient_checkpoint_forward,
)
from model5.third_party.light_wam.src.lightwam.models.wan22.wan_video_dit import (
    DiTBlock,
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)


class RecurrentActionQueryBlock(nn.Module):
    """Update one shared query bank from one real Wan hidden state.
    每观测一层hidden信息就更新一次query
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("query hidden_dim must be positive and divisible by num_heads")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")

        self.query_cross_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
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

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    #先用query去读Wan patch
        cross, _ = self.cross_attention(
            query=self.query_cross_norm(queries),
            key=self.memory_norm(memory),
            value=self.memory_norm(memory),
            need_weights=False,
        )#最终得到的是 cross[i],形状是[512],8个head并行做，学习不同的读取方式
        queries = queries + cross #以残差的形式更新
        normalized = self.query_self_norm(queries)
        self_out, _ = self.self_attention(
            query=normalized,
            key=normalized,
            value=normalized,
            need_weights=False,
        )
        queries = queries + self_out
        return queries + self.ffn(queries) #对每个query做非线性更新


class LayerWiseRecurrentActionQueryEncoder(nn.Module):
    """Carry one action-query bank through ordered Wan layer states."""

    SUPPORTED_FEATURE_SOURCES = {"backbone", "adapted", "delta"}

    def __init__(
        self,
        *,
        video_hidden_dim: int,
        query_dim: int,
        num_fusion_layers: int,
        num_action_queries: int,
        num_heads: int,
        bridge_depth: int,
        proprio_dim: Optional[int],
        feature_source: str = "adapted",
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if video_hidden_dim <= 0 or query_dim <= 0 or num_fusion_layers <= 0:
            raise ValueError("video_hidden_dim, query_dim, and num_fusion_layers must be positive")
        if num_action_queries <= 0 or bridge_depth <= 0:
            raise ValueError("num_action_queries and bridge_depth must be positive")
        feature_source = str(feature_source).strip().lower()
        if feature_source not in self.SUPPORTED_FEATURE_SOURCES:
            raise ValueError(
                f"feature_source must be one of {sorted(self.SUPPORTED_FEATURE_SOURCES)}, got {feature_source!r}"
            )

        self.video_hidden_dim = int(video_hidden_dim)
        self.query_dim = int(query_dim)
        self.num_fusion_layers = int(num_fusion_layers)
        self.num_action_queries = int(num_action_queries)
        self.num_heads = int(num_heads)
        self.bridge_depth = int(bridge_depth)
        self.ffn_multiplier = int(ffn_multiplier)
        self.dropout = float(dropout)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.feature_source = feature_source

        self.action_queries = nn.Parameter(
            torch.randn(1, self.num_action_queries, self.query_dim) * 0.02
        )
        self.layer_embeddings = nn.Parameter(
            torch.zeros(1, self.num_fusion_layers, 1, self.query_dim)
        )
        self.memory_projections = nn.ModuleList(
            [nn.Linear(self.video_hidden_dim, self.query_dim) for _ in range(self.num_fusion_layers)]
        )
        self.memory_projection_norms = nn.ModuleList(
            [nn.LayerNorm(self.query_dim) for _ in range(self.num_fusion_layers)]
        )
        self.layer_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        RecurrentActionQueryBlock(
                            hidden_dim=self.query_dim,
                            num_heads=self.num_heads,
                            ffn_multiplier=ffn_multiplier,
                            dropout=dropout,
                        )
                        for _ in range(self.bridge_depth)
                    ]
                )
                for _ in range(self.num_fusion_layers)
            ]
        )
        self.proprio_projector = (
            None
            if self.proprio_dim is None
            else nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, self.query_dim),
            )
        )
        self.output_norm = nn.LayerNorm(self.query_dim)

    def _validate_layer_state(
        self,
        layer_state: dict[str, Any],
        layer_position: int,
    ) -> torch.Tensor:
        if not isinstance(layer_state, dict):
            raise TypeError(f"layer_states[{layer_position}] must be a dict")
        if self.feature_source not in layer_state:
            raise KeyError(
                f"layer_states[{layer_position}] is missing {self.feature_source!r}; "
                f"available={sorted(layer_state.keys())}"
            )
        memory = layer_state[self.feature_source]
        if memory.ndim != 3 or int(memory.shape[-1]) != self.video_hidden_dim:
            raise ValueError(
                f"layer_states[{layer_position}][{self.feature_source!r}] must be "
                f"[B,S,{self.video_hidden_dim}], got {tuple(memory.shape)}"
            )
        return memory

    def forward(
        self,
        layer_states: Sequence[dict[str, Any]],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(layer_states) != self.num_fusion_layers:
            raise ValueError(
                f"expected {self.num_fusion_layers} ordered Wan states, got {len(layer_states)}"
            )

        first_memory = self._validate_layer_state(layer_states[0], 0)
        batch_size = int(first_memory.shape[0])
        queries = self.action_queries.expand(batch_size, -1, -1)
        if self.proprio_projector is not None:
            if proprio is None:
                raise ValueError("proprio is required by the model5 action-query encoder")
            if proprio.ndim != 2 or tuple(proprio.shape) != (batch_size, self.proprio_dim):
                raise ValueError(
                    f"proprio must be [B,{self.proprio_dim}], got {tuple(proprio.shape)}"
                )
            proprio_condition = self.proprio_projector(proprio.to(dtype=queries.dtype))
            queries = queries + proprio_condition.unsqueeze(1)

        trace = []
        for layer_position, (layer_state, projection, projection_norm, blocks) in enumerate(
            zip(
                layer_states,
                self.memory_projections,
                self.memory_projection_norms,
                self.layer_blocks,
            )
        ):
            memory = self._validate_layer_state(layer_state, layer_position)
            if int(memory.shape[0]) != batch_size:
                raise ValueError("all Wan layer states must have the same batch size")
            memory = projection_norm(projection(memory.to(dtype=queries.dtype)))
            queries = queries + self.layer_embeddings[:, layer_position]
            for block in blocks:
                queries = block(queries, memory)
            trace.append(self.output_norm(queries))

        query_trace = torch.stack(trace, dim=1)
        return query_trace[:, -1], query_trace


class VLAQueryDiTActionExpert(nn.Module):
    """Predict action-flow velocity from recurrent VLA-style query memory."""

    method_id = "model5_asymmetric_tri_timestep_query_flow_v1"
    flow_matching_layer_state_policy = True

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
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or ffn_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim, ffn_dim, and num_layers must be positive")
        if num_heads <= 0 or attn_head_dim <= 0 or attn_head_dim % 2 != 0:
            raise ValueError("num_heads and an even positive attn_head_dim are required")
        if hidden_dim != num_heads * attn_head_dim:
            raise ValueError("hidden_dim must equal num_heads * attn_head_dim")
        if freq_dim <= 0 or freq_dim % 2 != 0:
            raise ValueError("freq_dim must be a positive even integer")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(num_layers)
        self.freq_dim = int(freq_dim)
        self.eps = float(eps)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.query_encoder = LayerWiseRecurrentActionQueryEncoder(
            video_hidden_dim=video_hidden_dim,
            query_dim=query_dim,
            num_fusion_layers=num_fusion_layers,
            num_action_queries=num_action_queries,
            num_heads=query_num_heads,
            bridge_depth=query_bridge_depth,
            proprio_dim=proprio_dim,
            feature_source=feature_source,
            ffn_multiplier=query_ffn_multiplier,
            dropout=query_dropout,
        )
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
        self.query_condition_norm = nn.LayerNorm(query_dim, eps=self.eps)
        self.query_modulation = nn.Sequential(
            nn.Linear(query_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 6),
        )
        self.query_context_projection = nn.Linear(query_dim, self.hidden_dim)
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
        self.head = ActionHead(hidden_dim=self.hidden_dim, out_dim=self.action_dim, eps=self.eps)
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=self.action_horizon)

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

    def encode_queries(
        self,
        layer_states: Sequence[dict[str, Any]],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_encoder(layer_states, proprio)

    def _validate_action(self, action: torch.Tensor, timestep: torch.Tensor) -> None:
        if action.ndim != 3 or int(action.shape[2]) != self.action_dim:
            raise ValueError(f"action must be [B,T,{self.action_dim}], got {tuple(action.shape)}")
        if int(action.shape[1]) != self.action_horizon:
            raise ValueError(
                f"model5 action horizon is fixed at {self.action_horizon}, got {action.shape[1]}"
            )
        if timestep.ndim != 1 or int(timestep.shape[0]) != int(action.shape[0]):
            raise ValueError("timestep must be [B] and align with action")

    def predict_velocity_from_queries(
        self,
        query_memory: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_action(noisy_action, timestep)
        if query_memory.ndim != 3 or int(query_memory.shape[0]) != int(noisy_action.shape[0]):
            raise ValueError("query_memory must be [B,Q,D] and align with noisy_action")
        if int(query_memory.shape[2]) != self.query_encoder.query_dim:
            raise ValueError(
                f"query_memory width must be {self.query_encoder.query_dim}, got {query_memory.shape[2]}"
            )

        timestep = timestep.to(device=noisy_action.device, dtype=noisy_action.dtype)
        time_embed = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        normalized_queries = self.query_condition_norm(query_memory.to(dtype=noisy_action.dtype))
        query_summary = normalized_queries.mean(dim=1)
        t_mod = self.time_projection(time_embed) + self.query_modulation(query_summary)
        t_mod = t_mod.unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(noisy_action)
        context = self.query_context_projection(normalized_queries)
        context_mask = torch.ones(
            (noisy_action.shape[0], noisy_action.shape[1], context.shape[1]),
            dtype=torch.bool,
            device=noisy_action.device,
        )
        freqs = self.freqs.view(self.action_horizon, 1, -1).to(device=noisy_action.device)
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                tokens = gradient_checkpoint_forward(
                    block,
                    True,
                    tokens,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                tokens = block(tokens, context, t_mod, freqs, context_mask=context_mask)
        return self.head(tokens, time_embed)

    def forward(
        self,
        layer_states: Sequence[dict[str, Any]],
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_memory, _ = self.encode_queries(layer_states, proprio)
        return self.predict_velocity_from_queries(query_memory, noisy_action, timestep)

    @torch.no_grad()
    def sample_from_queries(
        self,
        query_memory: torch.Tensor,
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
                f"model5 requires action_horizon={self.action_horizon}, got {action_horizon}"
            )
        random_device = query_memory.device if noise_device is None else torch.device(noise_device)
        action = torch.randn(
            (query_memory.shape[0], self.action_horizon, self.action_dim),
            generator=generator,
            device=random_device,
            dtype=torch.float32,
        ).to(device=query_memory.device, dtype=query_memory.dtype)
        timesteps, deltas = scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=query_memory.device,
            dtype=query_memory.dtype,
            shift_override=sigma_shift,
        )
        for step_t, delta in zip(timesteps, deltas):
            timestep = step_t.expand(query_memory.shape[0])
            velocity = self.predict_velocity_from_queries(query_memory, action, timestep)
            action = scheduler.step(velocity, delta, action)
        return action

    @torch.no_grad()
    def sample(
        self,
        layer_states: Sequence[dict[str, Any]],
        *,
        action_horizon: int,
        scheduler,
        num_inference_steps: int,
        proprio: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        noise_device: Optional[torch.device | str] = None,
        sigma_shift: Optional[float] = None,
    ) -> torch.Tensor:
        query_memory, _ = self.encode_queries(layer_states, proprio)
        return self.sample_from_queries(
            query_memory,
            action_horizon=action_horizon,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            generator=generator,
            noise_device=noise_device,
            sigma_shift=sigma_shift,
        )
