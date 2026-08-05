"""Current-only residual adapters used by the Dynamic O2 treatment.

The module intentionally operates on the cached Wan tokens after the Wan
forward has completed.  It is therefore not a Wan block adapter: applying it
never writes a value back into a later Video-DiT block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable
from typing import Iterable, overload

import torch
from torch import nn


DEFAULT_RESPONSE_LAYERS = (8, 16, 24)


@dataclass(frozen=True)
class ResponseAdapterConfig:
    """Architecture identity for a deployed response-adapter bank."""

    layers: tuple[int, ...] = DEFAULT_RESPONSE_LAYERS
    hidden_dim: int = 1536
    bottleneck_dim: int = 64
    activation: str = "gelu"
    zero_init_up_projection: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_response_adapter_config(
    config: ResponseAdapterConfig | dict[str, object] | None,
) -> ResponseAdapterConfig:
    """Validate and normalize the only supported first-response-adapter form."""

    if config is None:
        result = ResponseAdapterConfig()
    elif isinstance(config, ResponseAdapterConfig):
        result = config
    elif isinstance(config, dict):
        values = dict(config)
        if "layers" in values:
            values["layers"] = tuple(int(value) for value in values["layers"])
        result = ResponseAdapterConfig(**values)
    else:
        raise TypeError(
            "response_adapter_config must be a ResponseAdapterConfig, dict, or None"
        )
    if result.layers != DEFAULT_RESPONSE_LAYERS:
        raise ValueError(
            "Dynamic O2 response adapters must be exactly at layers (8, 16, 24), "
            f"got {result.layers}."
        )
    if result.hidden_dim != 1536:
        raise ValueError("Dynamic O2 response adapters require hidden_dim=1536.")
    if result.bottleneck_dim != 64:
        raise ValueError("Dynamic O2 response adapters require bottleneck_dim=64.")
    if result.activation != "gelu":
        raise ValueError("Dynamic O2 response adapters require GELU activation.")
    if not result.zero_init_up_projection:
        raise ValueError("Dynamic O2 response adapters require zero-initialized up projections.")
    return result


class TokenResidualAdapter(nn.Module):
    """A token-wise $1536 \rightarrow 64 \rightarrow 1536$ residual branch.

    ``forward`` returns the residual $A_l(h)$ rather than the sum.  Keeping the
    residual explicit makes the Stage 1 gradient and the Stage 2 audit metric
    unambiguous; callers use :meth:`apply` to form $B_l=h_l+A_l(h_l)$.
    """

    def __init__(self, *, hidden_dim: int = 1536, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        if self.hidden_dim <= 0 or self.bottleneck_dim <= 0:
            raise ValueError("response-adapter dimensions must be positive")
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.down = nn.Linear(self.hidden_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(self.bottleneck_dim, self.hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(
                "response adapter tokens must be [B,S,D], "
                f"got {tuple(tokens.shape)}"
            )
        if int(tokens.shape[-1]) != self.hidden_dim:
            raise ValueError(
                f"response adapter expected hidden width {self.hidden_dim}, "
                f"got {tokens.shape[-1]}"
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate_tokens(tokens)
        return self.up(self.activation(self.down(self.norm(tokens))))

    @overload
    def apply(self, fn: Callable[[nn.Module], None]) -> "TokenResidualAdapter":
        ...

    @overload
    def apply(self, fn: torch.Tensor) -> torch.Tensor:
        ...

    def apply(self, fn: Callable[[nn.Module], None] | torch.Tensor) -> "TokenResidualAdapter" | torch.Tensor:
        """Support both module traversal and the token residual convenience call.

        ``nn.Module.apply`` is used by DeepSpeed while configuring ZeRO leaf
        modules.  The former token-only override shadowed that protocol and
        failed before Stage 2 could initialize.
        """

        if callable(fn):
            return super().apply(fn)
        if not isinstance(fn, torch.Tensor):
            raise TypeError("response adapter apply expects a module callback or token tensor")
        return fn + self(fn)


class ResponseAdapterBank(nn.Module):
    """Three independent deployed adapters keyed by Wan layer index."""

    def __init__(
        self,
        config: ResponseAdapterConfig | dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.config = normalize_response_adapter_config(config)
        self.adapters = nn.ModuleDict(
            {
                str(layer): TokenResidualAdapter(
                    hidden_dim=self.config.hidden_dim,
                    bottleneck_dim=self.config.bottleneck_dim,
                )
                for layer in self.config.layers
            }
        )

    @property
    def layers(self) -> tuple[int, ...]:
        return self.config.layers

    def configuration(self) -> dict[str, object]:
        return self.config.as_dict()

    def adapter(self, layer_idx: int) -> TokenResidualAdapter:
        key = str(int(layer_idx))
        if key not in self.adapters:
            raise KeyError(
                f"response adapter has no layer {layer_idx}; expected {self.config.layers}"
            )
        return self.adapters[key]

    @overload
    def apply(self, fn: Callable[[nn.Module], None]) -> "ResponseAdapterBank":
        ...

    @overload
    def apply(self, fn: int, tokens: torch.Tensor) -> torch.Tensor:
        ...

    def apply(
        self,
        fn: Callable[[nn.Module], None] | int,
        tokens: torch.Tensor | None = None,
    ) -> "ResponseAdapterBank" | torch.Tensor:
        """Preserve ``nn.Module.apply(fn)`` and layer-indexed token calls."""

        if callable(fn) and tokens is None:
            return super().apply(fn)
        if not isinstance(fn, int) or not isinstance(tokens, torch.Tensor):
            raise TypeError("response adapter bank apply expects (fn) or (layer_idx, tokens)")
        return self.adapter(fn).apply(tokens)

    def residual(self, layer_idx: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.adapter(layer_idx)(tokens)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters()]

    def set_trainable(self, enabled: bool) -> None:
        self.train(bool(enabled))
        self.requires_grad_(bool(enabled))

    @staticmethod
    def assert_layer_order(layer_states: Iterable[dict[str, object]]) -> None:
        observed = tuple(int(state["layer_idx"]) for state in layer_states)
        if observed != DEFAULT_RESPONSE_LAYERS:
            raise ValueError(
                "Dynamic O2 requires layer states in exact order (8, 16, 24), "
                f"got {observed}"
            )
