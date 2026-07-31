from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import sinusoidal_embedding_1d


FEATURE_SOURCE_ALIASES = {
    "h": "backbone",
    "backbone": "backbone",
    "base": "backbone",
    "frozen": "backbone",
    "h_prime": "adapted",
    "hprime": "adapted",
    "h'": "adapted",
    "adapted": "adapted",
    "adapter": "adapted",
    "delta": "delta",
}


def _normalize_feature_source_name(source_name: str) -> str:
    if not isinstance(source_name, str):
        raise TypeError(f"Feature source must be a string, got {type(source_name)}")
    key = source_name.strip().lower()
    if key not in FEATURE_SOURCE_ALIASES:
        raise ValueError(
            f"Unsupported feature source `{source_name}`. "
            "Expected one of: h/backbone, h_prime/adapted, delta."
        )
    return FEATURE_SOURCE_ALIASES[key]


def _normalize_feature_source_spec(
    spec: Sequence[str] | str,
    *,
    layer_idx: Optional[int] = None,
) -> tuple[str, ...]:
    if isinstance(spec, str):
        parts = [item.strip() for item in spec.split(",")]
    else:
        parts = list(spec)
    normalized = []
    seen = set()
    for item in parts:
        if item is None or (isinstance(item, str) and item.strip() == ""):
            continue
        source_name = _normalize_feature_source_name(str(item))
        if source_name in seen:
            continue
        normalized.append(source_name)
        seen.add(source_name)
    if not normalized:
        prefix = "" if layer_idx is None else f"layer_feature_sources[{layer_idx}] "
        raise ValueError(f"{prefix}must include at least one of h/backbone, h_prime/adapted, delta.")
    return tuple(normalized)


class LearnedQueryPooler(nn.Module):
    """Cross-attention pooling with a few learnable query tokens.

    The output stays at width `embed_dim` so we can replace mean pooling without
    exploding the state-fusion expert parameter count.
    """

    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_heads: int,
        merge_dim: Optional[int] = None,
        merge_num_slots: int = 2,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"`embed_dim` must be positive, got {embed_dim}")
        if num_queries <= 0:
            raise ValueError(f"`num_queries` must be positive, got {num_queries}")
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"`embed_dim` must be divisible by `num_heads`, got embed_dim={embed_dim}, num_heads={num_heads}"
            )
        if merge_dim is not None and merge_dim <= 0:
            raise ValueError(f"`merge_dim` must be positive when provided, got {merge_dim}")
        if merge_num_slots <= 0:
            raise ValueError(f"`merge_num_slots` must be positive, got {merge_num_slots}")

        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.merge_dim = None if merge_dim is None else int(merge_dim)
        self.merge_num_slots = int(merge_num_slots)
        self.output_dim = self.embed_dim if self.merge_dim is None else self.merge_dim
        self.token_norm = nn.LayerNorm(self.embed_dim)
        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, self.embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        if self.merge_dim is None:
            # Legacy path: learn one global weighted merge across query slots.
            self.query_merge_logits = nn.Parameter(torch.zeros(self.num_queries))
            self.query_slot_merge_logits = None
            self.query_merge_projector = None
        else:
            # Optional widened merge: keep a small number of merged slots, then
            # project their concatenation to a configurable width before the
            # per-layer compressor. This relaxes the single-summary bottleneck
            # without fully exposing all K query slots downstream.
            self.query_merge_logits = None
            self.query_slot_merge_logits = nn.Parameter(
                torch.zeros(self.merge_num_slots, self.num_queries)
            )
            self.query_merge_projector = nn.Sequential(
                nn.LayerNorm(self.merge_num_slots * self.embed_dim),
                nn.Linear(self.merge_num_slots * self.embed_dim, self.merge_dim),
            )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"`tokens` for LearnedQueryPooler must be [B, S, D], got shape {tuple(tokens.shape)}"
            )
        batch_size = int(tokens.shape[0])
        if int(tokens.shape[2]) != self.embed_dim:
            raise ValueError(
                f"`tokens` last dim must be {self.embed_dim}, got {tokens.shape[2]}"
            )
        key_value = self.token_norm(tokens)
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        pooled, _ = self.attn(query=query, key=key_value, value=key_value, need_weights=False)
        if self.merge_dim is None:
            merge_weights = F.softmax(self.query_merge_logits, dim=0).to(
                device=pooled.device,
                dtype=pooled.dtype,
            )
            pooled = (pooled * merge_weights.view(1, self.num_queries, 1)).sum(dim=1)
        else:
            merge_weights = F.softmax(self.query_slot_merge_logits, dim=-1).to(
                device=pooled.device,
                dtype=pooled.dtype,
            )
            pooled = torch.einsum("mk,bkd->bmd", merge_weights, pooled)
            pooled = pooled.reshape(batch_size, self.merge_num_slots * self.embed_dim)
            pooled = self.query_merge_projector(pooled)
        return self.output_norm(pooled)

    def get_attention_summary(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(
                f"`tokens` for LearnedQueryPooler must be [B, S, D], got shape {tuple(tokens.shape)}"
            )
        batch_size = int(tokens.shape[0])
        if int(tokens.shape[2]) != self.embed_dim:
            raise ValueError(
                f"`tokens` last dim must be {self.embed_dim}, got {tokens.shape[2]}"
            )
        key_value = self.token_norm(tokens)
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        _, attn_weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=False,
        )
        if attn_weights.ndim != 4:
            raise ValueError(
                f"Expected attention weights [B, H, Q, S], got {tuple(attn_weights.shape)}"
            )
        per_query_attention = attn_weights.mean(dim=1)
        if self.merge_dim is None:
            merge_weights = F.softmax(self.query_merge_logits, dim=0).to(
                device=per_query_attention.device,
                dtype=per_query_attention.dtype,
            )
            query_importance = merge_weights
            summary_attention = torch.einsum("q,bqs->bs", merge_weights, per_query_attention)
        else:
            merge_weights = F.softmax(self.query_slot_merge_logits, dim=-1).to(
                device=per_query_attention.device,
                dtype=per_query_attention.dtype,
            )
            query_importance = merge_weights.mean(dim=0)
            slot_attention = torch.einsum("mq,bqs->bms", merge_weights, per_query_attention)
            summary_attention = slot_attention.mean(dim=1)
        top_query_idx = torch.argmax(query_importance)
        return {
            "per_query_attention": per_query_attention,
            "summary_attention": summary_attention,
            "query_importance": query_importance,
            "top_query_idx": top_query_idx,
        }


class ResidualMLPBlock(nn.Module):
    """Residual MLP block used by the medium-sized direct action expert."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class LayerFusionCompressor(nn.Module):
    """Compresses one layer's [h, h', delta] pooled feature into a fixed-width vector."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class StateFusionActionExpert(nn.Module):
    """Direct action predictor over pooled multi-layer backbone/adapter states."""

    def __init__(
        self,
        video_hidden_dim: int,
        action_dim: int,
        num_fusion_layers: int,
        per_layer_dim: int = 4608,
        trunk_dim: int = 6144,
        num_trunk_blocks: int = 1,
        step_pos_dim: int = 256,
        token_pooling_type: str = "mean",
        token_pooling_num_queries: int = 4,
        token_pooling_num_heads: int = 8,
        token_pooling_merge_dim: Optional[int] = None,
        token_pooling_merge_num_slots: int = 2,
        feature_sources: Sequence[str] | str = ("backbone", "adapted", "delta"),
        layer_feature_sources: Optional[Sequence[Sequence[str] | str]] = None,
    ):
        super().__init__()
        if video_hidden_dim <= 0:
            raise ValueError(f"`video_hidden_dim` must be positive, got {video_hidden_dim}")
        if action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {action_dim}")
        if num_fusion_layers <= 0:
            raise ValueError(f"`num_fusion_layers` must be positive, got {num_fusion_layers}")
        if per_layer_dim <= 0:
            raise ValueError(f"`per_layer_dim` must be positive, got {per_layer_dim}")
        if trunk_dim <= 0:
            raise ValueError(f"`trunk_dim` must be positive, got {trunk_dim}")
        if num_trunk_blocks < 0:
            raise ValueError(f"`num_trunk_blocks` must be non-negative, got {num_trunk_blocks}")
        if step_pos_dim <= 0 or step_pos_dim % 2 != 0:
            raise ValueError(
                f"`step_pos_dim` must be a positive even integer, got {step_pos_dim}"
            )
        token_pooling_key = str(token_pooling_type).strip().lower()
        if token_pooling_key not in {"mean", "learned_query"}:
            raise ValueError(
                f"`token_pooling_type` must be one of ['mean', 'learned_query'], got {token_pooling_type}"
            )
        if token_pooling_num_queries <= 0:
            raise ValueError(
                f"`token_pooling_num_queries` must be positive, got {token_pooling_num_queries}"
            )
        if token_pooling_num_heads <= 0:
            raise ValueError(
                f"`token_pooling_num_heads` must be positive, got {token_pooling_num_heads}"
            )
        if token_pooling_merge_dim is not None and int(token_pooling_merge_dim) <= 0:
            raise ValueError(
                f"`token_pooling_merge_dim` must be positive when provided, got {token_pooling_merge_dim}"
            )
        if token_pooling_merge_num_slots <= 0:
            raise ValueError(
                f"`token_pooling_merge_num_slots` must be positive, got {token_pooling_merge_num_slots}"
            )
        if token_pooling_merge_dim is not None and token_pooling_key != "learned_query":
            raise ValueError(
                "`token_pooling_merge_dim` requires `token_pooling_type='learned_query'`."
            )

        self.video_hidden_dim = int(video_hidden_dim)
        self.action_dim = int(action_dim)
        self.num_fusion_layers = int(num_fusion_layers)
        self.per_layer_dim = int(per_layer_dim)
        self.trunk_dim = int(trunk_dim)
        self.num_trunk_blocks = int(num_trunk_blocks)
        self.step_pos_dim = int(step_pos_dim)
        self.token_pooling_type = token_pooling_key
        self.token_pooling_num_queries = int(token_pooling_num_queries)
        self.token_pooling_num_heads = int(token_pooling_num_heads)
        self.token_pooling_merge_dim = (
            None if token_pooling_merge_dim is None else int(token_pooling_merge_dim)
        )
        self.token_pooling_merge_num_slots = int(token_pooling_merge_num_slots)
        self.pooler_output_dim = (
            self.video_hidden_dim
            if self.token_pooling_merge_dim is None
            else self.token_pooling_merge_dim
        )
        global_feature_sources = _normalize_feature_source_spec(feature_sources)
        if layer_feature_sources is None:
            self.layer_feature_sources = tuple(
                global_feature_sources for _ in range(self.num_fusion_layers)
            )
        else:
            if len(layer_feature_sources) != self.num_fusion_layers:
                raise ValueError(
                    "`layer_feature_sources` must align with configured fusion layers, "
                    f"got {len(layer_feature_sources)} vs {self.num_fusion_layers}"
                )
            self.layer_feature_sources = tuple(
                _normalize_feature_source_spec(spec, layer_idx=idx)
                for idx, spec in enumerate(layer_feature_sources)
            )
        self.layer_input_dims = [
            self.pooler_output_dim * len(source_names)
            for source_names in self.layer_feature_sources
        ]
        self.fused_input_dim = self.per_layer_dim * self.num_fusion_layers

        self.layer_poolers = nn.ModuleList()
        for source_names in self.layer_feature_sources:
            source_poolers = nn.ModuleDict()
            if self.token_pooling_type == "learned_query":
                for source_name in source_names:
                    source_poolers[source_name] = LearnedQueryPooler(
                        embed_dim=self.video_hidden_dim,
                        num_queries=self.token_pooling_num_queries,
                        num_heads=self.token_pooling_num_heads,
                        merge_dim=self.token_pooling_merge_dim,
                        merge_num_slots=self.token_pooling_merge_num_slots,
                    )
            self.layer_poolers.append(source_poolers)

        self.layer_compressors = nn.ModuleList(
            [
                LayerFusionCompressor(
                    in_dim=layer_input_dim,
                    out_dim=self.per_layer_dim,
                )
                for layer_input_dim in self.layer_input_dims
            ]
        )
        self.fused_norm = nn.LayerNorm(self.fused_input_dim)
        self.fused_proj = nn.Linear(self.fused_input_dim, self.trunk_dim)
        self.trunk = nn.ModuleList(
            [ResidualMLPBlock(self.trunk_dim) for _ in range(self.num_trunk_blocks)]
        )
        self.step_pos_proj = nn.Sequential(
            nn.LayerNorm(self.step_pos_dim),
            nn.Linear(self.step_pos_dim, self.trunk_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.trunk_dim, self.trunk_dim),
        )
        self.output_norm = nn.LayerNorm(self.trunk_dim)
        self.output = nn.Sequential(
            nn.Linear(self.trunk_dim, self.trunk_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.trunk_dim, self.action_dim),
        )

    def _pool_source_tokens(self, layer_idx: int, source_name: str, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"`layer_states[{layer_idx}]['{source_name}']` must be [B, S, D], got {tuple(tokens.shape)}"
            )
        if int(tokens.shape[2]) != self.video_hidden_dim:
            raise ValueError(
                f"`layer_states[{layer_idx}]['{source_name}']` last dim must be {self.video_hidden_dim}, "
                f"got {tokens.shape[2]}"
            )
        if self.token_pooling_type == "mean":
            return tokens.mean(dim=1)
        return self.layer_poolers[layer_idx][source_name](tokens)

    def forward(self, layer_states: Sequence[dict[str, Any]], action_horizon: int) -> torch.Tensor:
        if len(layer_states) != self.num_fusion_layers:
            raise ValueError(
                "Number of layer fusion states must match configured fusion layers, "
                f"got {len(layer_states)} vs {self.num_fusion_layers}"
            )
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

        compressed = []
        batch_size = None
        for idx, (layer_state, compressor) in enumerate(zip(layer_states, self.layer_compressors)):
            if not isinstance(layer_state, dict):
                raise TypeError(
                    f"`layer_states[{idx}]` must be a dict of source tensors, got {type(layer_state)}"
                )
            pooled_sources = []
            expected_sources = self.layer_feature_sources[idx]
            for source_name in expected_sources:
                if source_name not in layer_state:
                    raise KeyError(
                        f"`layer_states[{idx}]` missing required source `{source_name}`. "
                        f"Available keys: {sorted(layer_state.keys())}"
                    )
                pooled_sources.append(
                    self._pool_source_tokens(idx, source_name, layer_state[source_name])
                )
            feature = torch.cat(pooled_sources, dim=-1)
            if feature.ndim != 2:
                raise ValueError(
                    f"`layer_states[{idx}]` pooled feature must be [B, D], got shape {tuple(feature.shape)}"
                )
            if feature.shape[1] != self.layer_input_dims[idx]:
                raise ValueError(
                    f"`layer_states[{idx}]` pooled feature last dim must be {self.layer_input_dims[idx]}, "
                    f"got {feature.shape[1]}"
                )
            if batch_size is None:
                batch_size = int(feature.shape[0])
            elif int(feature.shape[0]) != batch_size:
                raise ValueError(
                    f"`layer_states[{idx}]` batch size mismatch: {feature.shape[0]} vs {batch_size}"
                )
            compressed.append(compressor(feature))

        fused = torch.cat(compressed, dim=-1)
        state = self.fused_proj(self.fused_norm(fused))
        for block in self.trunk:
            state = block(state)

        positions = torch.arange(action_horizon, device=state.device, dtype=state.dtype)
        step_pos = sinusoidal_embedding_1d(self.step_pos_dim, positions)
        step_tokens = state.unsqueeze(1) + self.step_pos_proj(step_pos).unsqueeze(0)
        return self.output(self.output_norm(step_tokens))

    def summarize_pooling_attention(
        self,
        layer_states: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.token_pooling_type != "learned_query":
            raise ValueError(
                "`summarize_pooling_attention` requires token_pooling_type='learned_query'."
            )
        if len(layer_states) != self.num_fusion_layers:
            raise ValueError(
                "Number of layer fusion states must match configured fusion layers, "
                f"got {len(layer_states)} vs {self.num_fusion_layers}"
            )

        summaries: list[dict[str, Any]] = []
        for idx, layer_state in enumerate(layer_states):
            if not isinstance(layer_state, dict):
                raise TypeError(
                    f"`layer_states[{idx}]` must be a dict of source tensors, got {type(layer_state)}"
                )
            layer_summary = {
                "layer_idx": int(layer_state.get("layer_idx", idx)),
                "sources": {},
            }
            for source_name in self.layer_feature_sources[idx]:
                if source_name not in layer_state:
                    raise KeyError(
                        f"`layer_states[{idx}]` missing required source `{source_name}`. "
                        f"Available keys: {sorted(layer_state.keys())}"
                    )
                layer_summary["sources"][source_name] = self.layer_poolers[idx][source_name].get_attention_summary(
                    layer_state[source_name]
                )
            summaries.append(layer_summary)
        return summaries
