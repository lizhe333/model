"""Focused orchestration contracts for a lightweight SideModel3WAM shell."""

from __future__ import annotations

from types import MethodType

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from side_model3.models.ema_target import EMATargetPredictiveEncoder
from side_model3.models.future_latent_change_head import (
    MultiHorizonFutureLatentChangeHead,
)
from side_model3.models.ladder_side_encoder import LadderSideEncoder
from side_model3.models.latent_transition import (
    LatentTransitionPredictor,
    MultiHorizonActionChunkEncoder,
)
from side_model3.models.side_model3_wam import SideModel3WAM
from side_model3.models.visual_anchor_resampler import (
    VisualAnchorActionFusion,
    VisualAnchorResampler,
)


BATCH_SIZE = 2
HIDDEN_DIM = 8
VIDEO_HIDDEN_DIM = 6
NUM_SLOTS = 4
PROPRIO_DIM = 3
ACTION_DIM = 3
LATENT_CHANNELS = 4


def _gradient_energy(module: nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


class _FrozenWanStub(nn.Module):
    """A frozen current-frame feature extractor standing in for Wan hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(LATENT_CHANNELS, VIDEO_HIDDEN_DIM)
        self.requires_grad_(False)
        self.calls = 0

    def forward(self, latents: torch.Tensor) -> list[torch.Tensor]:
        self.calls += 1
        tokens = latents.squeeze(2).flatten(2).transpose(1, 2)
        projected = self.projection(tokens)
        return [projected + 0.01 * position for position in range(5)]


class _TrainingScheduler:
    @staticmethod
    def sample_training_t(
        *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.full((batch_size,), 0.4, device=device, dtype=dtype)

    @staticmethod
    def add_noise(
        action: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        amount = timestep.view(-1, 1, 1)
        return (1.0 - amount) * action + amount * noise

    @staticmethod
    def training_target(
        action: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        return noise - action

    @staticmethod
    def training_weight(timestep: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(timestep)


class _TinyActionPolicy(nn.Module):
    """Cheap context-conditioned flow head; Action-DiT itself is tested separately."""

    def __init__(self) -> None:
        super().__init__()
        self.action_horizon = 8
        self.action_dim = ACTION_DIM
        self.context_projection = nn.Linear(HIDDEN_DIM, ACTION_DIM)
        self.action_projection = nn.Linear(ACTION_DIM, ACTION_DIM)
        self.sample_calls = 0

    def config_dict(self) -> dict[str, int]:
        return {
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "context_dim": HIDDEN_DIM,
        }

    def forward(
        self,
        action_state: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_projection(action_state.mean(dim=1)).unsqueeze(1)
        time = timestep.view(-1, 1, 1).to(dtype=noisy_action.dtype)
        return self.action_projection(noisy_action) + context + time

    @torch.no_grad()
    def sample(
        self,
        action_state: torch.Tensor,
        *,
        action_horizon: int,
        scheduler,
        num_inference_steps: int,
        generator,
        noise_device,
        sigma_shift,
    ) -> torch.Tensor:
        del scheduler, num_inference_steps, generator, noise_device, sigma_shift
        if action_horizon != self.action_horizon:
            raise ValueError("unexpected action horizon")
        self.sample_calls += 1
        context = self.context_projection(action_state.mean(dim=1)).unsqueeze(1)
        return context.expand(-1, self.action_horizon, -1)


def _prepared_inputs() -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(101)
    current = torch.randn(
        BATCH_SIZE,
        LATENT_CHANNELS,
        1,
        4,
        6,
        generator=generator,
    )
    return {
        "context": torch.randn(BATCH_SIZE, 2, 5, generator=generator),
        "context_mask": torch.ones(BATCH_SIZE, 2, dtype=torch.bool),
        "current_latents": current,
        "future_latents": {
            4: current + torch.randn(current.shape, generator=generator) * 0.1,
            8: current + torch.randn(current.shape, generator=generator) * 0.2,
        },
        "proprio": {
            0: torch.randn(BATCH_SIZE, PROPRIO_DIM, generator=generator),
            4: torch.randn(BATCH_SIZE, PROPRIO_DIM, generator=generator),
            8: torch.randn(BATCH_SIZE, PROPRIO_DIM, generator=generator),
        },
        "action": torch.randn(BATCH_SIZE, 8, ACTION_DIM, generator=generator),
        "action_is_pad": None,
        "future_valid": {4: None, 8: None},
    }


def _make_lightweight_wam(
    *, loss_weights: dict[str, float] | None = None
) -> SideModel3WAM:
    """Build the model shell without constructing LightWAM or downloading Wan."""

    torch.manual_seed(41)
    model = SideModel3WAM.__new__(SideModel3WAM)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_dim = PROPRIO_DIM

    model.online_predictive_encoder = LadderSideEncoder(
        video_hidden_dim=VIDEO_HIDDEN_DIM,
        proprio_dim=PROPRIO_DIM,
        layer_indices=SideModel3WAM.selected_wan_layers,
        num_slots=NUM_SLOTS,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        ffn_dim=16,
        residual_gate_init=0.1,
    )
    model.target_predictive_encoder = EMATargetPredictiveEncoder(
        model.online_predictive_encoder,
        decay=0.996,
    )
    model.action_policy = _TinyActionPolicy()
    model.visual_resampler = VisualAnchorResampler(
        video_hidden_dim=VIDEO_HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_anchors=3,
        num_heads=2,
        ffn_dim=16,
    )
    model.visual_fusion = VisualAnchorActionFusion(
        num_slots=NUM_SLOTS,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
    )
    # The exact-zero initialization is covered separately. Open it here so the
    # action-loss boundary reaches both visual modules in one backward pass.
    with torch.no_grad():
        model.visual_fusion.query_gate.fill_(0.2)
    model.action_chunk_encoder = MultiHorizonActionChunkEncoder(
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        max_horizon=8,
    )
    model.transition_predictor = LatentTransitionPredictor(
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        ffn_dim=16,
        num_blocks=2,
    )
    model.latent_change_head = MultiHorizonFutureLatentChangeHead(
        latent_channels=LATENT_CHANNELS,
        grid_height=2,
        grid_width=3,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        ffn_dim=16,
    )
    model.frozen_wan = _FrozenWanStub()
    model.video_expert = model.frozen_wan
    model.mot = nn.Linear(1, 1, bias=False)
    model.mot.requires_grad_(False)
    model.vae = nn.Identity()
    model.text_encoder = None
    model.train_action_scheduler = _TrainingScheduler()
    model.infer_action_scheduler = object()
    model.loss_weights = loss_weights or {
        "action": 1.0,
        "state_4": 0.25,
        "state_8": 0.50,
        "latent_4": 0.10,
        "latent_8": 0.20,
    }
    model.ema_update_count = 0
    model._ema_optimizer_hook_handle = None
    model._prepared_test_inputs = _prepared_inputs()

    def build_inputs(self, sample, tiled: bool = False):
        del sample, tiled
        return self._prepared_test_inputs

    def extract_states(
        self,
        *,
        observation_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        del context, context_mask
        with torch.no_grad():
            return [state.detach() for state in self.frozen_wan(observation_latents)]

    def action_loss_per_sample(
        self,
        *,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        del self
        loss = F.mse_loss(pred_action, target_action, reduction="none")
        if action_is_pad is not None:
            loss = loss.masked_fill(action_is_pad.unsqueeze(-1), 0.0)
        return loss.mean(dim=(1, 2))

    model.build_inputs = MethodType(build_inputs, model)
    model._extract_frozen_wan_states = MethodType(extract_states, model)
    model._compute_action_loss_per_sample = MethodType(
        action_loss_per_sample, model
    )
    return model


@pytest.mark.parametrize(
    ("active_loss", "active_modules", "inactive_modules"),
    [
        (
            "action",
            (
                "online_predictive_encoder",
                "action_policy",
                "visual_resampler",
                "visual_fusion",
            ),
            ("action_chunk_encoder", "transition_predictor", "latent_change_head"),
        ),
        (
            "state_4",
            (
                "online_predictive_encoder",
                "action_chunk_encoder",
                "transition_predictor",
            ),
            ("action_policy", "visual_resampler", "visual_fusion", "latent_change_head"),
        ),
        (
            "state_8",
            (
                "online_predictive_encoder",
                "action_chunk_encoder",
                "transition_predictor",
            ),
            ("action_policy", "visual_resampler", "visual_fusion", "latent_change_head"),
        ),
        (
            "latent_4",
            (
                "online_predictive_encoder",
                "action_chunk_encoder",
                "transition_predictor",
                "latent_change_head",
            ),
            ("action_policy", "visual_resampler", "visual_fusion"),
        ),
        (
            "latent_8",
            (
                "online_predictive_encoder",
                "action_chunk_encoder",
                "transition_predictor",
                "latent_change_head",
            ),
            ("action_policy", "visual_resampler", "visual_fusion"),
        ),
    ],
)
def test_each_loss_respects_its_gradient_boundary(
    active_loss: str,
    active_modules: tuple[str, ...],
    inactive_modules: tuple[str, ...],
) -> None:
    weights = {
        name: float(name == active_loss)
        for name in ("action", "state_4", "state_8", "latent_4", "latent_8")
    }
    model = _make_lightweight_wam(loss_weights=weights)

    total, metrics = model.training_loss({})
    total.backward()

    assert torch.isfinite(total)
    assert set(metrics) == {
        "loss_action",
        "loss_state_4",
        "loss_state_8",
        "loss_latent_4",
        "loss_latent_8",
        "loss_total",
        "ema_updates",
    }
    assert total.item() == pytest.approx(metrics[f"loss_{active_loss}"])
    for module_name in active_modules:
        assert _gradient_energy(getattr(model, module_name)) > 0.0, module_name
    for module_name in inactive_modules:
        assert _gradient_energy(getattr(model, module_name)) == 0.0, module_name
    assert all(
        parameter.grad is None for parameter in model.target_predictive_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.frozen_wan.parameters())


def test_five_losses_use_the_frozen_contract_weights() -> None:
    model = _make_lightweight_wam()

    total, metrics = model.training_loss({})

    expected = sum(
        model.loss_weights[name] * metrics[f"loss_{name}"]
        for name in ("action", "state_4", "state_8", "latent_4", "latent_8")
    )
    assert total.item() == pytest.approx(expected)
    assert metrics["loss_total"] == pytest.approx(expected)


def test_ema_optimizer_post_step_hook_updates_only_after_real_step() -> None:
    model = _make_lightweight_wam()
    optimizer = torch.optim.SGD(model.online_predictive_encoder.parameters(), lr=0.1)
    handle = model.register_ema_optimizer_hook(optimizer)
    target_parameter = next(model.target_predictive_encoder.encoder.parameters())
    online_parameter = next(model.online_predictive_encoder.parameters())
    target_before = target_parameter.detach().clone()

    total, _ = model.training_loss({})
    total.backward()
    assert model.ema_update_count == 0
    assert torch.equal(target_parameter, target_before)

    optimizer.step()
    online_after = online_parameter.detach().clone()
    expected = target_before.lerp(
        online_after,
        1.0 - model.target_predictive_encoder.decay,
    )
    assert model.ema_update_count == 1
    assert not torch.equal(online_after, target_before)
    assert torch.allclose(target_parameter, expected)

    target_after_step = target_parameter.detach().clone()
    model.training_loss({})
    assert model.ema_update_count == 1
    assert torch.equal(target_parameter, target_after_step)
    handle.remove()


def test_side_model3_checkpoint_round_trip_and_identity_guard(tmp_path) -> None:
    model = _make_lightweight_wam()
    checkpoint_path = tmp_path / "side_model3.pt"
    before = next(model.online_predictive_encoder.parameters()).detach().clone()

    model.save_checkpoint(checkpoint_path, step=17)
    with torch.no_grad():
        next(model.online_predictive_encoder.parameters()).add_(9.0)
    payload = model.load_checkpoint(checkpoint_path)

    assert payload["method_id"] == SideModel3WAM.method_id
    assert payload["model_class"] == "SideModel3WAM"
    assert payload["step"] == 17
    assert torch.equal(next(model.online_predictive_encoder.parameters()), before)

    wrong_identity_path = tmp_path / "wrong_method.pt"
    payload["method_id"] = "model3_future_dynamics_wam"
    torch.save(payload, wrong_identity_path)
    with pytest.raises(ValueError, match="method identity mismatch"):
        model.load_checkpoint(wrong_identity_path)


class _ForbiddenFutureModule(nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("future-only module executed during inference")


def test_inference_never_executes_future_only_modules() -> None:
    model = _make_lightweight_wam()
    model.target_predictive_encoder = _ForbiddenFutureModule()
    model.action_chunk_encoder = _ForbiddenFutureModule()
    model.transition_predictor = _ForbiddenFutureModule()
    model.latent_change_head = _ForbiddenFutureModule()

    def encode_image(self, *, input_image: torch.Tensor, tiled: bool):
        del tiled
        return torch.zeros(
            input_image.shape[0], LATENT_CHANNELS, 1, 4, 6, dtype=self.torch_dtype
        )

    model._encode_input_image_latents_tensor = MethodType(encode_image, model)
    before_wan_calls = model.frozen_wan.calls

    result = model.infer_action(
        prompt=None,
        input_image=torch.zeros(3, 8, 8),
        action_horizon=8,
        proprio=torch.zeros(PROPRIO_DIM),
        context=torch.zeros(2, 5),
        context_mask=torch.ones(2, dtype=torch.bool),
        num_inference_steps=3,
        seed=5,
        rand_device="cpu",
    )

    assert result["action"].shape == (8, ACTION_DIM)
    assert model.frozen_wan.calls == before_wan_calls + 1
    assert model.action_policy.sample_calls == 1
