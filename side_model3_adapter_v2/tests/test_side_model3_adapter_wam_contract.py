"""Focused orchestration contracts for a lightweight SideModel3AdapterV2WAM shell."""

from __future__ import annotations

from types import MethodType

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from side_model3_adapter_v2.models.ema_target import (
    EMATargetPredictiveEncoder,
    EMATargetWanAdapters,
)
from side_model3_adapter_v2.models.future_latent_change_head import (
    MultiHorizonFutureLatentChangeHead,
)
from side_model3_adapter_v2.models.ladder_side_encoder import LadderSideEncoder
from side_model3_adapter_v2.models.latent_transition import (
    LatentTransitionPredictor,
    MultiHorizonActionChunkEncoder,
)
from side_model3_adapter_v2.models.side_model3_adapter_v2_wam import SideModel3AdapterV2WAM
from side_model3_adapter_v2.models.visual_anchor_resampler import (
    VisualAnchorActionFusion,
    VisualAnchorResampler,
)
from side_model3_adapter_v2.runtime import create_side_model3_adapter_v2_wam
from model3.third_party.light_wam.src.lightwam.models.wan22.wan_video_dit import (
    ResidualAdapter,
)


BATCH_SIZE = 2
HIDDEN_DIM = 8
VIDEO_HIDDEN_DIM = 6
NUM_SLOTS = 4
PROPRIO_DIM = 3
ACTION_DIM = 3
LATENT_CHANNELS = 4


def _factory_kwargs() -> dict[str, object]:
    return {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "tokenizer_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "video_dit_config": {
            "text_dim": 4096,
            "video_attention_mask_mode": "first_frame_causal",
            "use_wam_adapter": True,
            "use_backbone_lora": False,
        },
        "side_encoder_config": {
            "layer_indices": [8, 16, 20, 24, 29],
            "num_slots": 64,
            "hidden_dim": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "residual_gate_init": 0.1,
        },
        "action_policy_config": {
            "hidden_dim": 512,
            "ffn_dim": 2048,
            "num_heads": 8,
            "attn_head_dim": 64,
            "num_layers": 16,
            "action_horizon": 8,
            "use_gradient_checkpointing": False,
        },
        "visual_anchor_config": {"num_anchors": 16, "num_heads": 8, "ffn_dim": 2048},
        "transition_config": {"num_blocks": 2, "num_heads": 8, "ffn_dim": 2048},
        "latent_change_config": {
            "grid_height": 14,
            "grid_width": 28,
            "num_heads": 8,
            "ffn_dim": 2048,
        },
        "action_scheduler": {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        },
        "loss": {
            "lambda_video": 0.0,
            "future_video_flow_loss": False,
            "action": 1.0,
            "state_4": 0.25,
            "state_8": 0.5,
            "latent_4": 0.1,
            "latent_8": 0.2,
        },
        "wam_adapter": {
            "use_wam_adapter": True,
            "freeze_backbone": True,
            "remove_original_action_expert": False,
            "adapter_layer_indices": [8, 16, 24],
            "adapter_dim": 256,
            "adapter_scale": 1.0,
            "use_backbone_lora": False,
        },
        "state_fusion_action_expert_config": None,
        "proprio_dim": 8,
        "model_dtype": torch.float32,
        "device": "cpu",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("use_wam_adapter", False), "requires Wan residual adapters"),
        (("use_backbone_lora", True), "cannot instantiate Wan LoRA"),
        (("adapter_layer_indices", [8, 16, 20]), "adapter layers 8,16,24"),
    ],
)
def test_factory_rejects_non_v1_adapter_routes(mutation, message) -> None:
    kwargs = _factory_kwargs()
    kwargs["wam_adapter"][mutation[0]] = mutation[1]
    with pytest.raises(ValueError, match=message):
        create_side_model3_adapter_v2_wam(**kwargs)


def test_factory_rejects_inherited_state_fusion() -> None:
    kwargs = _factory_kwargs()
    kwargs["state_fusion_action_expert_config"] = {"trunk_dim": 6144}
    with pytest.raises(ValueError, match="cannot instantiate StateFusion"):
        create_side_model3_adapter_v2_wam(**kwargs)


def _gradient_energy(module: nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


class _FrozenWanStub(nn.Module):
    """Frozen base projection plus the three trainable residual adapters."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(LATENT_CHANNELS, VIDEO_HIDDEN_DIM)
        self.projection.requires_grad_(False)
        self.use_wam_adapter = True
        self.adapter_layer_indices = (8, 16, 24)
        self.adapter_dim = 3
        self.adapter_scale = 1.0
        self.wam_adapters = nn.ModuleDict(
            {
                str(layer): ResidualAdapter(
                    hidden_dim=VIDEO_HIDDEN_DIM,
                    adapter_dim=self.adapter_dim,
                    eps=1.0e-6,
                    scale=self.adapter_scale,
                )
                for layer in self.adapter_layer_indices
            }
        )
        self.calls = 0

    @staticmethod
    def has_backbone_lora() -> bool:
        return False

    def forward(self, latents: torch.Tensor) -> list[torch.Tensor]:
        self.calls += 1
        tokens = latents.squeeze(2).flatten(2).transpose(1, 2)
        projected = self.projection(tokens)
        return [projected + 0.01 * position for position in range(5)]


class _TinyWanBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(VIDEO_HIDDEN_DIM, VIDEO_HIDDEN_DIM)
        self.requires_grad_(False)

    def forward(
        self,
        tokens,
        context,
        time_modulation,
        frequencies,
        context_mask=None,
        self_attn_mask=None,
    ):
        del context, time_modulation, frequencies, context_mask, self_attn_mask
        return tokens + 0.01 * self.projection(tokens)


class _AdapterWanBackboneStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyWanBlock() for _ in range(30)])
        self.use_wam_adapter = True
        self.adapter_layer_indices = (8, 16, 24)
        self.adapter_dim = 3
        self.adapter_scale = 1.0
        self.fuse_vae_embedding_in_latents = True
        self.wam_adapters = nn.ModuleDict(
            {
                str(layer): ResidualAdapter(
                    hidden_dim=VIDEO_HIDDEN_DIM,
                    adapter_dim=self.adapter_dim,
                    eps=1.0e-6,
                    scale=self.adapter_scale,
                )
                for layer in self.adapter_layer_indices
            }
        )

    @staticmethod
    def has_backbone_lora() -> bool:
        return False

    @staticmethod
    def build_video_to_video_mask(
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        del video_tokens_per_frame
        return torch.ones(video_seq_len, video_seq_len, dtype=torch.bool, device=device)


class _TrainingScheduler:
    num_train_timesteps = 1000

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
        amount = (timestep / 1000.0).view(-1, 1, 1)
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
) -> SideModel3AdapterV2WAM:
    """Build the model shell without constructing LightWAM or downloading Wan."""

    torch.manual_seed(41)
    model = SideModel3AdapterV2WAM.__new__(SideModel3AdapterV2WAM)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_dim = PROPRIO_DIM

    model.online_predictive_encoder = LadderSideEncoder(
        video_hidden_dim=VIDEO_HIDDEN_DIM,
        proprio_dim=PROPRIO_DIM,
        layer_indices=SideModel3AdapterV2WAM.selected_wan_layers,
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
    model.target_wan_adapters = EMATargetWanAdapters(
        model.video_expert.wam_adapters,
        decay=0.996,
    )
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
    model.v2_local_optimizer_steps = 0
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
        use_target_adapters: bool,
    ) -> list[torch.Tensor]:
        del context, context_mask
        states = self.frozen_wan(observation_latents)
        adapted_states = []
        for layer, state in zip(self.selected_wan_layers, states):
            if layer in self.adapter_layer_indices:
                if use_target_adapters:
                    state = self.target_wan_adapters.apply_adapters(layer, state)
                else:
                    state, _ = self.video_expert.wam_adapters[str(layer)](state)
            adapted_states.append(state)
        if use_target_adapters:
            return [state.detach() for state in adapted_states]
        return adapted_states

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
    model._extract_adapted_wan_states = MethodType(extract_states, model)
    model._compute_action_loss_per_sample = MethodType(
        action_loss_per_sample, model
    )
    return model


def test_manual_wan_path_uses_online_and_ema_adapter_banks() -> None:
    torch.manual_seed(91)
    model = SideModel3AdapterV2WAM.__new__(SideModel3AdapterV2WAM)
    nn.Module.__init__(model)
    model.video_expert = _AdapterWanBackboneStub()
    model.target_wan_adapters = EMATargetWanAdapters(
        model.video_expert.wam_adapters,
        decay=0.5,
    )

    def build_video_pre(
        self,
        *,
        observation_latents,
        timestep_video,
        context,
        context_mask,
        fuse_vae_embedding_in_latents,
    ):
        del self, timestep_video, fuse_vae_embedding_in_latents
        tokens = observation_latents.squeeze(2).flatten(2).transpose(1, 2)
        return {
            "tokens": tokens,
            "context": context,
            "context_mask": context_mask,
            "t_mod": torch.zeros(tokens.shape[0], 6, VIDEO_HIDDEN_DIM),
            "freqs": torch.zeros(tokens.shape[1], 1, VIDEO_HIDDEN_DIM // 2),
            "meta": {"tokens_per_frame": tokens.shape[1]},
        }

    model._build_action_observation_video_pre = MethodType(build_video_pre, model)
    observation = torch.randn(BATCH_SIZE, VIDEO_HIDDEN_DIM, 1, 2, 2)
    context = torch.zeros(BATCH_SIZE, 2, 5)
    context_mask = torch.ones(BATCH_SIZE, 2, dtype=torch.bool)

    online_states = model._extract_adapted_wan_states(
        observation_latents=observation,
        context=context,
        context_mask=context_mask,
        use_target_adapters=False,
    )
    with torch.no_grad():
        target_states = model._extract_adapted_wan_states(
            observation_latents=observation,
            context=context,
            context_mask=context_mask,
            use_target_adapters=True,
        )
    assert all(torch.equal(online, target) for online, target in zip(online_states, target_states))

    sum(state.square().mean() for state in online_states).backward()
    assert all(
        adapter.up_proj.weight.grad is not None
        and adapter.up_proj.weight.grad.abs().sum() > 0
        for adapter in model.video_expert.wam_adapters.values()
    )
    assert all(
        parameter.grad is None
        for block in model.video_expert.blocks
        for parameter in block.parameters()
    )
    assert all(parameter.grad is None for parameter in model.target_wan_adapters.parameters())

    target_before = [state.clone() for state in target_states]
    with torch.no_grad():
        model.video_expert.wam_adapters["8"].up_proj.weight.add_(0.05)
    with torch.no_grad():
        unchanged_target = model._extract_adapted_wan_states(
            observation_latents=observation,
            context=context,
            context_mask=context_mask,
            use_target_adapters=True,
        )
    assert all(torch.equal(before, after) for before, after in zip(target_before, unchanged_target))
    model.target_wan_adapters.update(model.video_expert.wam_adapters)
    with torch.no_grad():
        moved_target = model._extract_adapted_wan_states(
            observation_latents=observation,
            context=context,
            context_mask=context_mask,
            use_target_adapters=True,
        )
    assert any(not torch.equal(before, after) for before, after in zip(target_before, moved_target))


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
        "loss_dyn_action",
        "dyn_action_weight",
        "v2_local_optimizer_steps",
        "loss_total",
        "ema_updates",
    }
    assert total.item() == pytest.approx(metrics[f"loss_{active_loss}"])
    for module_name in active_modules:
        assert _gradient_energy(getattr(model, module_name)) > 0.0, module_name
    for module_name in inactive_modules:
        assert _gradient_energy(getattr(model, module_name)) == 0.0, module_name
    assert _gradient_energy(model.video_expert.wam_adapters) > 0.0
    assert all(
        parameter.grad is None for parameter in model.target_predictive_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.frozen_wan.projection.parameters())
    assert all(parameter.grad is None for parameter in model.target_wan_adapters.parameters())


def test_five_losses_use_the_frozen_contract_weights() -> None:
    model = _make_lightweight_wam()

    total, metrics = model.training_loss({})

    expected = sum(
        model.loss_weights[name] * metrics[f"loss_{name}"]
        for name in ("action", "state_4", "state_8", "latent_4", "latent_8")
    )
    assert total.item() == pytest.approx(expected)
    assert metrics["loss_total"] == pytest.approx(expected)


def test_predicted_action_branch_keeps_action_jacobian_without_et_parameter_grads() -> None:
    model = _make_lightweight_wam(
        loss_weights={
            "action": 0.0,
            "state_4": 0.0,
            "state_8": 0.0,
            "latent_4": 0.0,
            "latent_8": 0.0,
        }
    )
    model.v2_local_optimizer_steps = 1

    total, metrics = model.training_loss({})
    total.backward()

    assert metrics["dyn_action_weight"] == pytest.approx(0.0001)
    assert metrics["loss_dyn_action"] > 0.0
    assert _gradient_energy(model.action_policy) > 0.0
    assert _gradient_energy(model.action_chunk_encoder) == 0.0
    assert _gradient_energy(model.transition_predictor) == 0.0
    assert all(parameter.grad is None for parameter in model.target_predictive_encoder.parameters())


def test_clean_action_reconstruction_matches_perfect_flow_velocity() -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)
    action = torch.randn(BATCH_SIZE, 8, ACTION_DIM, generator=generator)
    noise = torch.randn(action.shape, generator=generator)
    timestep = torch.full((BATCH_SIZE,), 400.0)
    noisy = 0.6 * action + 0.4 * noise
    reconstructed = SideModel3AdapterV2WAM._reconstruct_clean_action(
        noisy,
        noise - action,
        timestep,
        num_train_timesteps=1000,
    )
    assert torch.allclose(reconstructed, action)


def test_predicted_action_weight_is_applied_before_masked_reduction() -> None:
    loss = torch.tensor([1.0, 3.0])
    timestep_weight = torch.tensor([1.0, 10.0])
    valid = torch.tensor([True, False])
    reduced = SideModel3AdapterV2WAM._weighted_masked_batch_mean(
        loss,
        timestep_weight,
        valid,
    )
    assert reduced.item() == pytest.approx(1.0)
    assert SideModel3AdapterV2WAM._weighted_masked_batch_mean(
        loss,
        timestep_weight,
        torch.ones(2, dtype=torch.bool),
    ).item() == pytest.approx(15.5)


def test_trainable_policy_enables_only_wan_adapters_inside_backbone() -> None:
    model = _make_lightweight_wam()
    model.configure_trainable_modules()

    assert all(
        not parameter.requires_grad for parameter in model.video_expert.projection.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.video_expert.wam_adapters.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in model.target_wan_adapters.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.target_predictive_encoder.parameters()
    )


def test_ema_optimizer_post_step_hook_updates_only_after_real_step() -> None:
    model = _make_lightweight_wam()
    optimizer = torch.optim.SGD(
        list(model.online_predictive_encoder.parameters())
        + list(model.video_expert.wam_adapters.parameters()),
        lr=0.1,
    )
    handle = model.register_ema_optimizer_hook(optimizer)
    target_parameter = next(model.target_predictive_encoder.encoder.parameters())
    online_parameter = next(model.online_predictive_encoder.parameters())
    target_adapter_parameter = model.target_wan_adapters.adapters["8"].up_proj.weight
    online_adapter_parameter = model.video_expert.wam_adapters["8"].up_proj.weight
    target_before = target_parameter.detach().clone()
    target_adapter_before = target_adapter_parameter.detach().clone()

    total, _ = model.training_loss({})
    total.backward()
    assert model.ema_update_count == 0
    assert torch.equal(target_parameter, target_before)
    assert torch.equal(target_adapter_parameter, target_adapter_before)

    optimizer.step()
    online_after = online_parameter.detach().clone()
    expected = target_before.lerp(
        online_after,
        1.0 - model.target_predictive_encoder.decay,
    )
    assert model.ema_update_count == 1
    assert model.v2_local_optimizer_steps == 1
    assert not torch.equal(online_after, target_before)
    assert torch.allclose(target_parameter, expected)
    expected_adapter = target_adapter_before.lerp(
        online_adapter_parameter.detach().float(),
        1.0 - model.target_wan_adapters.decay,
    )
    assert torch.allclose(target_adapter_parameter, expected_adapter)

    target_after_step = target_parameter.detach().clone()
    model.training_loss({})
    assert model.ema_update_count == 1
    assert model.v2_local_optimizer_steps == 1
    assert torch.equal(target_parameter, target_after_step)
    handle.remove()


def test_side_model3_adapter_v2_checkpoint_round_trip_and_identity_guard(tmp_path) -> None:
    model = _make_lightweight_wam()
    model.v2_local_optimizer_steps = 37
    checkpoint_path = tmp_path / "side_model3_adapter_v2.pt"
    before = next(model.online_predictive_encoder.parameters()).detach().clone()
    adapter_before = next(model.video_expert.wam_adapters.parameters()).detach().clone()

    model.save_checkpoint(checkpoint_path, step=17)
    with torch.no_grad():
        next(model.online_predictive_encoder.parameters()).add_(9.0)
    payload = model.load_checkpoint(checkpoint_path)

    assert payload["method_id"] == SideModel3AdapterV2WAM.method_id
    assert payload["model_class"] == "SideModel3AdapterV2WAM"
    assert payload["step"] == 17
    assert payload["v2_local_optimizer_steps"] == 37
    assert payload["dyn_action_weight"] == pytest.approx(0.0037)
    assert model.v2_local_optimizer_steps == 37
    assert torch.equal(next(model.online_predictive_encoder.parameters()), before)
    assert torch.equal(next(model.video_expert.wam_adapters.parameters()), adapter_before)

    wrong_identity_path = tmp_path / "wrong_method.pt"
    payload["method_id"] = "model3_future_dynamics_wam"
    torch.save(payload, wrong_identity_path)
    with pytest.raises(ValueError, match="method identity mismatch"):
        model.load_checkpoint(wrong_identity_path)


def test_v1_model_only_warmstart_ignores_optimizer_state_and_resets_v2_steps(tmp_path) -> None:
    source = _make_lightweight_wam()
    source.v2_local_optimizer_steps = 19
    checkpoint_path = tmp_path / "v1_parent.pt"
    source.save_checkpoint(checkpoint_path, step=90000)
    payload = torch.load(checkpoint_path, map_location="cpu")
    payload["method_id"] = "side_model3_adapter_wan_residual_ladder_flow_v1"
    payload["model_class"] = "SideModel3AdapterWAM"
    payload["optimizer"] = {"ignored": True}
    torch.save(payload, checkpoint_path)

    target = _make_lightweight_wam()
    target.v2_local_optimizer_steps = 7
    target.load_v1_model_only_warmstart(checkpoint_path)

    assert target.v2_local_optimizer_steps == 0
    assert torch.equal(
        next(target.online_predictive_encoder.parameters()),
        next(source.online_predictive_encoder.parameters()),
    )


class _ForbiddenFutureModule(nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("future-only module executed during inference")


def test_inference_never_executes_future_only_modules() -> None:
    model = _make_lightweight_wam()
    model.target_predictive_encoder = _ForbiddenFutureModule()
    model.target_wan_adapters = _ForbiddenFutureModule()
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
