"""EMA target copy for the complete predictive side encoder."""

from __future__ import annotations

import copy

import torch
from torch import nn


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
        online_parameters = dict(online_encoder.named_parameters())
        for name, target in self.encoder.named_parameters():
            source = online_parameters[name]
            target.lerp_(
                source.detach().to(device=target.device, dtype=torch.float32),
                1.0 - self.decay,
            )

        online_buffers = dict(online_encoder.named_buffers())
        for name, target in self.encoder.named_buffers():
            source = online_buffers[name].detach().to(device=target.device)
            if target.is_floating_point():
                target.lerp_(source.to(dtype=torch.float32), 1.0 - self.decay)
            else:
                target.copy_(source)

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
