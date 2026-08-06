"""FP32 EMA targets copied from Side-Model3 plus the new Wan adapters."""

from __future__ import annotations

import copy

import torch
from torch import nn


@torch.no_grad()
def _ema_update(target_module: nn.Module, online_module: nn.Module, decay: float) -> None:
    online_parameters = dict(online_module.named_parameters())
    for name, target in target_module.named_parameters():
        source = online_parameters[name]
        target.lerp_(
            source.detach().to(device=target.device, dtype=torch.float32),
            1.0 - decay,
        )

    online_buffers = dict(online_module.named_buffers())
    for name, target in target_module.named_buffers():
        source = online_buffers[name].detach().to(device=target.device)
        if target.is_floating_point():
            target.lerp_(source.to(dtype=torch.float32), 1.0 - decay)
        else:
            target.copy_(source)


class EMATargetPredictiveEncoder(nn.Module):
    def __init__(self, online_encoder: nn.Module, *, decay: float = 0.996) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.encoder = copy.deepcopy(online_encoder)
        self.encoder.float()
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        _ema_update(self.encoder, online_encoder, self.decay)

    @torch.no_grad()
    def forward(self, layer_states, proprio):
        device = next(self.encoder.parameters()).device
        prepared_states = [
            state.to(device=device, dtype=torch.float32) for state in layer_states
        ]
        prepared_proprio = proprio.to(device=device, dtype=torch.float32)
        output = self.encoder(prepared_states, prepared_proprio)
        if isinstance(output, tuple):
            return tuple(value.detach() if isinstance(value, torch.Tensor) else value for value in output)
        return output.detach() if isinstance(output, torch.Tensor) else output


class EMATargetWanAdapters(nn.Module):
    """FP32 EMA copy of only the three online Wan residual adapters."""

    def __init__(self, online_adapters: nn.Module, *, decay: float = 0.996) -> None:
        super().__init__()
        self.decay = float(decay)
        self.adapters = copy.deepcopy(online_adapters)
        self.adapters.float()
        self.adapters.requires_grad_(False)
        self.adapters.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.adapters.eval()
        return self

    @torch.no_grad()
    def update(self, online_adapters: nn.Module) -> None:
        _ema_update(self.adapters, online_adapters, self.decay)

    @torch.no_grad()
    def apply(self, layer_index: int, tokens: torch.Tensor) -> torch.Tensor:
        adapter = self.adapters[str(int(layer_index))]
        output_dtype = tokens.dtype
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            adapted, _ = adapter(tokens.float())
        return adapted.to(dtype=output_dtype)
