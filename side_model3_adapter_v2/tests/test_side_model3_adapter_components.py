"""Copied Side-Model3 component tests plus adapter-specific boundaries."""

from __future__ import annotations

import pytest
import torch

from side_model3_adapter_v2.models.action_dit import SideModel3ActionDiT
from side_model3_adapter_v2.models.ema_target import (
    EMATargetPredictiveEncoder,
    EMATargetWanAdapters,
)
from side_model3_adapter_v2.models.future_latent_change_head import (
    MultiHorizonFutureLatentChangeHead,
)
from side_model3_adapter_v2.models.ladder_side_encoder import (
    LadderSideEncoder,
    O2StyleTraceFusion,
)
from side_model3_adapter_v2.models.latent_transition import (
    LatentTransitionPredictor,
    MultiHorizonActionChunkEncoder,
)
from side_model3_adapter_v2.models.visual_anchor_resampler import VisualAnchorActionFusion
from model3.third_party.light_wam.src.lightwam.models.wan22.wan_video_dit import (
    ResidualAdapter,
)
from side_model3.models.action_dit import SideModel3ActionDiT as ParentActionDiT
from side_model3.models.ladder_side_encoder import LadderSideEncoder as ParentLadder


HIDDEN_DIM = 8
VIDEO_HIDDEN_DIM = 6
NUM_SLOTS = 4
PROPRIO_DIM = 3
ACTION_DIM = 3


def _tiny_ladder() -> LadderSideEncoder:
    return LadderSideEncoder(
        video_hidden_dim=VIDEO_HIDDEN_DIM,
        proprio_dim=PROPRIO_DIM,
        layer_indices=(8, 16, 20, 24, 29),
        num_slots=NUM_SLOTS,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        ffn_dim=16,
        residual_gate_init=0.1,
    )


def _gradient_energy(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_ladder_gate_trace_shape_and_adapter_gradient_boundary() -> None:
    torch.manual_seed(1)
    encoder = _tiny_ladder()
    states = [
        torch.randn(2, 3 + position, VIDEO_HIDDEN_DIM, requires_grad=True)
        for position in range(5)
    ]
    proprio = torch.randn(2, PROPRIO_DIM, requires_grad=True)

    control_state, trace = encoder(states, proprio)

    assert control_state.shape == (2, NUM_SLOTS, HIDDEN_DIM)
    assert trace.shape == (2, 5, NUM_SLOTS, HIDDEN_DIM)
    assert [stage.residual_gate.item() for stage in encoder.stages] == pytest.approx(
        [0.1] * 5
    )

    control_state.square().mean().backward()
    assert _gradient_energy(encoder) > 0.0
    assert proprio.grad is not None and proprio.grad.abs().sum() > 0
    assert all(state.grad is not None and state.grad.abs().sum() > 0 for state in states)


def test_trace_fusion_is_exact_final_stage_identity_at_initialization() -> None:
    torch.manual_seed(2)
    fusion = O2StyleTraceFusion(
        num_stages=5,
        num_slots=NUM_SLOTS,
        hidden_dim=HIDDEN_DIM,
    )
    trace = torch.randn(2, 5, NUM_SLOTS, HIDDEN_DIM)

    fused = fusion(trace)

    assert torch.equal(fused, trace[:, -1])
    assert all(torch.count_nonzero(gate) == 0 for gate in fusion.query_gates)
    for projection in fusion.early_projections:
        assert torch.count_nonzero(projection.weight) == 0
        assert torch.count_nonzero(projection.bias) == 0


def test_copied_ladder_and_action_dit_match_side_model3_forward() -> None:
    torch.manual_seed(71)
    parent_ladder = ParentLadder(
        video_hidden_dim=VIDEO_HIDDEN_DIM,
        proprio_dim=PROPRIO_DIM,
        layer_indices=(8, 16, 20, 24, 29),
        num_slots=NUM_SLOTS,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        ffn_dim=16,
        residual_gate_init=0.1,
    )
    torch.manual_seed(71)
    copied_ladder = _tiny_ladder()
    states = [torch.randn(2, 3, VIDEO_HIDDEN_DIM) for _ in range(5)]
    proprio = torch.randn(2, PROPRIO_DIM)
    parent_control, parent_trace = parent_ladder(states, proprio)
    copied_control, copied_trace = copied_ladder(states, proprio)
    assert torch.equal(copied_control, parent_control)
    assert torch.equal(copied_trace, parent_trace)

    policy_kwargs = {
        "action_dim": ACTION_DIM,
        "context_dim": HIDDEN_DIM,
        "hidden_dim": HIDDEN_DIM,
        "ffn_dim": 16,
        "num_heads": 2,
        "attn_head_dim": 4,
        "num_layers": 2,
        "freq_dim": 8,
        "action_horizon": 8,
    }
    torch.manual_seed(72)
    parent_policy = ParentActionDiT(**policy_kwargs)
    torch.manual_seed(72)
    copied_policy = SideModel3ActionDiT(**policy_kwargs)
    action_state = torch.randn(2, NUM_SLOTS, HIDDEN_DIM)
    noisy_action = torch.randn(2, 8, ACTION_DIM)
    timestep = torch.tensor([0.2, 0.7])
    assert torch.equal(
        copied_policy(action_state, noisy_action, timestep),
        parent_policy(action_state, noisy_action, timestep),
    )


def test_visual_fusion_is_exact_identity_and_gate_can_open() -> None:
    torch.manual_seed(3)
    fusion = VisualAnchorActionFusion(
        num_slots=NUM_SLOTS,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
    )
    control_state = torch.randn(2, NUM_SLOTS, HIDDEN_DIM)
    anchors = torch.randn(2, 3, HIDDEN_DIM)

    initialized = fusion(control_state, anchors)
    assert torch.equal(initialized, control_state)
    assert torch.count_nonzero(fusion.query_gate) == 0

    with torch.no_grad():
        fusion.query_gate.fill_(0.25)
    opened = fusion(control_state, anchors)
    assert opened.shape == control_state.shape
    assert not torch.equal(opened, control_state)


@pytest.mark.parametrize("horizon", [4, 8])
def test_transition_and_latent_head_shapes_and_gradients(horizon: int) -> None:
    torch.manual_seed(4 + horizon)
    action_encoder = MultiHorizonActionChunkEncoder(
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        max_horizon=8,
    )
    transition = LatentTransitionPredictor(
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        ffn_dim=16,
        num_blocks=2,
    )
    latent_head = MultiHorizonFutureLatentChangeHead(
        latent_channels=4,
        grid_height=2,
        grid_width=3,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        ffn_dim=16,
    )
    actions = torch.randn(2, 8, ACTION_DIM, requires_grad=True)
    control_state = torch.randn(2, NUM_SLOTS, HIDDEN_DIM, requires_grad=True)

    action_tokens = action_encoder(actions, horizon=horizon)
    predicted_state = transition(control_state, action_tokens)
    latent_delta = latent_head(predicted_state, horizon=horizon)

    assert action_tokens.shape == (2, horizon, HIDDEN_DIM)
    assert predicted_state.shape == (2, NUM_SLOTS, HIDDEN_DIM)
    assert latent_delta.shape == (2, 4, 2, 3)

    (predicted_state.square().mean() + latent_delta.square().mean()).backward()
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert control_state.grad is not None and control_state.grad.abs().sum() > 0
    assert _gradient_energy(action_encoder) > 0.0
    assert _gradient_energy(transition) > 0.0
    assert _gradient_energy(latent_head) > 0.0


def test_ema_target_is_frozen_and_moves_only_toward_online() -> None:
    torch.manual_seed(13)
    online = _tiny_ladder()
    target = EMATargetPredictiveEncoder(online, decay=0.75)
    online_parameter = next(online.parameters())
    target_parameter = next(target.encoder.parameters())
    initial_target = target_parameter.detach().clone()

    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert torch.equal(target_parameter, online_parameter)
    target.train()
    assert not target.training
    assert not target.encoder.training

    with torch.no_grad():
        online_parameter.add_(2.0)
    target.update(online)

    expected = initial_target + 0.25 * (online_parameter.detach() - initial_target)
    assert torch.allclose(target_parameter, expected)
    assert not torch.equal(target_parameter, online_parameter)

    states = [torch.randn(2, 3, VIDEO_HIDDEN_DIM) for _ in range(5)]
    output, trace = target(states, torch.randn(2, PROPRIO_DIM))
    assert not output.requires_grad
    assert not trace.requires_grad


def test_ema_target_accumulates_small_bf16_source_updates_in_fp32() -> None:
    online = _tiny_ladder().to(dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.fill_(1.0)
    target = EMATargetPredictiveEncoder(online, decay=0.996)
    online_parameter = next(online.parameters())
    target_parameter = next(target.encoder.parameters())

    assert all(parameter.dtype == torch.float32 for parameter in target.parameters())
    with torch.no_grad():
        online_parameter.fill_(1.01)
    source_value = float(online_parameter.flatten()[0])
    for _ in range(1000):
        target.update(online)

    expected = 1.0 + (source_value - 1.0) * (1.0 - 0.996**1000)
    assert float(target_parameter.flatten()[0]) == pytest.approx(expected, abs=1.0e-5)


def test_residual_adapter_identity_and_fp32_ema_target() -> None:
    online = torch.nn.ModuleDict(
        {
            str(layer): ResidualAdapter(
                hidden_dim=VIDEO_HIDDEN_DIM,
                adapter_dim=3,
                eps=1.0e-6,
                scale=1.0,
            )
            for layer in (8, 16, 24)
        }
    ).to(dtype=torch.bfloat16)
    target = EMATargetWanAdapters(online, decay=0.996)
    tokens = torch.randn(2, 5, VIDEO_HIDDEN_DIM, dtype=torch.bfloat16)

    for layer in (8, 16, 24):
        adapted, delta = online[str(layer)](tokens)
        assert torch.equal(adapted, tokens)
        assert torch.count_nonzero(delta) == 0
    assert all(parameter.dtype == torch.float32 for parameter in target.parameters())

    online_parameter = next(online.parameters())
    target_parameter = next(target.adapters.parameters())
    with torch.no_grad():
        online_parameter.add_(0.01)
    initial = target_parameter.detach().clone()
    target.update(online)
    expected = initial.lerp(online_parameter.detach().float(), 0.004)
    assert torch.allclose(target_parameter, expected)


def test_ema_adapter_target_keeps_torch_module_apply_protocol() -> None:
    target = EMATargetWanAdapters(
        torch.nn.ModuleDict(
            {
                "8": ResidualAdapter(
                    hidden_dim=VIDEO_HIDDEN_DIM,
                    adapter_dim=2,
                    eps=1.0e-6,
                    scale=1.0,
                )
            }
        )
    )
    visited: list[torch.nn.Module] = []

    target.apply(visited.append)

    assert target in visited
    assert target.adapters in visited


class _EulerScheduler:
    def build_inference_schedule(
        self,
        *,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del shift_override
        timesteps = torch.linspace(
            1.0,
            0.0,
            num_inference_steps,
            device=device,
            dtype=dtype,
        )
        deltas = torch.full_like(timesteps, -1.0 / num_inference_steps)
        return timesteps, deltas

    @staticmethod
    def step(
        velocity: torch.Tensor,
        delta: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return action + delta * velocity


def test_action_dit_sampling_runs_requested_solver_steps() -> None:
    torch.manual_seed(21)
    policy = SideModel3ActionDiT(
        action_dim=ACTION_DIM,
        context_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        ffn_dim=16,
        num_heads=2,
        attn_head_dim=4,
        num_layers=2,
        freq_dim=8,
        action_horizon=8,
    )
    action_state = torch.randn(2, NUM_SLOTS, HIDDEN_DIM)
    calls: list[torch.Size] = []
    handle = policy.register_forward_hook(
        lambda _module, _inputs, output: calls.append(output.shape)
    )
    try:
        sampled = policy.sample(
            action_state,
            action_horizon=8,
            scheduler=_EulerScheduler(),
            num_inference_steps=3,
            generator=torch.Generator(device="cpu").manual_seed(7),
            noise_device="cpu",
        )
    finally:
        handle.remove()

    assert sampled.shape == (2, 8, ACTION_DIM)
    assert torch.isfinite(sampled).all()
    assert calls == [torch.Size((2, 8, ACTION_DIM))] * 3
