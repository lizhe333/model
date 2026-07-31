from __future__ import annotations

import torch
import pytest
import torch.nn.functional as F

from model5.models import Model5WAM, VLAQueryDiTActionExpert


def _tiny_policy() -> VLAQueryDiTActionExpert:
    return VLAQueryDiTActionExpert(
        video_hidden_dim=8,
        action_dim=3,
        num_fusion_layers=3,
        proprio_dim=None,
        query_dim=16,
        num_action_queries=5,
        query_num_heads=2,
        query_bridge_depth=1,
        hidden_dim=16,
        ffn_dim=32,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
        freq_dim=8,
        action_horizon=4,
    )


def _minimal_feature_model(scope: str = "current_plus_noisy_future") -> Model5WAM:
    model = Model5WAM.__new__(Model5WAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.video_latent_spatial_downsample_factor = 2
    model.action_feature_temporal_scope = scope
    model.action_feature_fixed_future_timestep = 1000
    model.action_feature_num_future_latent_slots = 8
    model.action_feature_spatial_downsample_factor = 1
    model._last_action_feature_diagnostics = {}
    model.enable_timing_breakdown = False
    model.timing_breakdown_sync_cuda = False
    model._timing_breakdown = {}
    model.state_fusion_action_expert = _tiny_policy()
    return model


def test_treatment_builds_clean_current_plus_eight_seeded_future_slots() -> None:
    model = _minimal_feature_model()
    current = torch.arange(2 * 4 * 1 * 8 * 12, dtype=torch.float32).reshape(
        2, 4, 1, 8, 12
    )

    first, timestep, slot_timesteps = model._build_action_feature_latents(
        observation_latents=current,
        generator=torch.Generator(device="cpu").manual_seed(17),
        noise_device="cpu",
    )
    second, _, _ = model._build_action_feature_latents(
        observation_latents=current,
        generator=torch.Generator(device="cpu").manual_seed(17),
        noise_device="cpu",
    )

    assert first.shape == (2, 4, 9, 8, 12)
    assert torch.equal(first[:, :, :1], current)
    assert torch.equal(first, second)
    assert torch.equal(timestep, torch.tensor([1000.0, 1000.0]))
    assert torch.equal(
        slot_timesteps[0],
        torch.tensor([0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]),
    )


def test_current_only_keeps_the_model3_high_resolution_current_anchor() -> None:
    model = _minimal_feature_model(scope="current_only")
    current = torch.randn(1, 4, 1, 8, 12)

    feature_latents, timestep, slot_timesteps = model._build_action_feature_latents(
        observation_latents=current
    )

    assert torch.equal(feature_latents, current)
    assert feature_latents.shape == (1, 4, 1, 8, 12)
    assert timestep.item() == 1000.0
    assert slot_timesteps.tolist() == [[0.0]]


def test_low_resolution_efficiency_profile_downsamples_current_and_future_slots() -> None:
    model = _minimal_feature_model()
    model.action_feature_spatial_downsample_factor = 2
    current = torch.randn(1, 4, 1, 8, 12)

    feature_latents, timestep, slot_timesteps = model._build_action_feature_latents(
        observation_latents=current,
        generator=torch.Generator(device="cpu").manual_seed(17),
        noise_device="cpu",
    )

    expected_current = F.avg_pool3d(current, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    assert feature_latents.shape == (1, 4, 9, 4, 6)
    assert torch.equal(feature_latents[:, :, :1], expected_current)
    assert timestep.tolist() == [1000.0]
    assert slot_timesteps.tolist() == [[0.0] + [1000.0] * 8]


def test_layer_state_diagnostics_cover_all_temporal_tokens() -> None:
    model = _minimal_feature_model()
    captured = {}

    def fake_build_video_pre(**kwargs):
        captured["latents"] = kwargs["latents_video"]
        captured["timestep"] = kwargs["timestep_video"]
        return {"tokens": kwargs["latents_video"]}, None

    class FakeVideoExpert:
        fuse_vae_embedding_in_latents = True

        def forward_backbone(self, video_pre):
            return video_pre["tokens"]

    model.video_expert = FakeVideoExpert()
    model._build_video_pre = fake_build_video_pre
    tokens_per_slot = 7
    model._build_multilayer_action_fusion_inputs = lambda: [
        {
            "layer_idx": layer_idx,
            "adapted": torch.randn(1, 9 * tokens_per_slot, 8),
        }
        for layer_idx in (8, 16, 24)
    ]

    states = model._build_action_layer_states(
        observation_latents=torch.randn(1, 4, 1, 8, 12),
        context=torch.randn(1, 2, 4),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        fuse_vae_embedding_in_latents=True,
        feature_generator=torch.Generator(device="cpu").manual_seed(3),
        feature_noise_device="cpu",
    )
    diagnostics = model.get_last_action_feature_diagnostics()

    assert captured["latents"].shape[2] == 9
    assert captured["latents"].shape[-2:] == (8, 12)
    assert captured["timestep"].item() == 1000.0
    assert all(state["adapted"].shape[1] == 63 for state in states)
    assert diagnostics["slot_timesteps"] == (0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000)
    assert diagnostics["hidden_tokens_per_layer"] == (63, 63, 63)
