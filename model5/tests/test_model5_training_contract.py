from __future__ import annotations

import torch

from model5.models import Model5WAM, VLAQueryDiTActionExpert
from model5.third_party.light_wam.src.lightwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def _tiny_policy() -> VLAQueryDiTActionExpert:
    return VLAQueryDiTActionExpert(
        video_hidden_dim=8,
        action_dim=3,
        num_fusion_layers=3,
        proprio_dim=2,
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


def test_model5_joint_loss_uses_future_flow_and_policy_horizon() -> None:
    torch.manual_seed(11)
    model = Model5WAM.__new__(Model5WAM)
    torch.nn.Module.__init__(model)
    model.state_fusion_action_expert = _tiny_policy()
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.loss_lambda_video = 1.0
    model.loss_lambda_action = 1.0
    model.action_temporal_weighting_enabled = False
    model.action_temporal_weighting_num_prefix_steps = None
    model.action_temporal_weighting_prefix_weight = 1.0
    model.action_temporal_weighting_tail_weight = 1.0
    model.enable_timing_breakdown = False
    model.timing_breakdown_sync_cuda = False
    model._timing_breakdown = {}
    model._last_action_feature_diagnostics = {
        "latent_slots": 9,
        "future_slots": 8,
        "fixed_future_timestep": 1000,
        "hidden_tokens_per_layer": (6, 6, 6),
    }
    model.train_video_scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    model.train_action_scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )

    batch_size = 2
    inputs = {
        "input_latents": torch.randn(batch_size, 1, 3, 1, 1),
        "first_frame_latents": torch.randn(batch_size, 1, 1, 1, 1),
        "context": torch.randn(batch_size, 2, 4),
        "context_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "action": torch.randn(batch_size, 10, 3),
        "action_is_pad": torch.zeros(batch_size, 10, dtype=torch.bool),
        "image_is_pad": None,
        "fuse_vae_embedding_in_latents": True,
        "proprio": torch.randn(batch_size, 2),
    }
    model.build_inputs = lambda sample, tiled=False: inputs
    model._build_video_training_supervision_latents = lambda latents: latents
    model._prepare_video_training_targets = lambda **kwargs: {
        "latents_video": kwargs["video_supervision_latents"],
        "target_video": torch.zeros_like(kwargs["video_supervision_latents"]),
        "apply_spatial_downsample": False,
        "restore_spatial_resolution": False,
    }
    model._predict_video_only = lambda **kwargs: torch.zeros_like(kwargs["latents_video"])
    layer_states = [
        {"layer_idx": layer_idx, "adapted": torch.randn(batch_size, 6, 8)}
        for layer_idx in (8, 16, 24)
    ]
    model._build_action_layer_states = lambda **kwargs: layer_states

    loss, metrics = model.training_loss(sample={})
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["loss_video_raw"] == 0.0
    assert metrics["loss_action_raw"] > 0.0
    assert model.action_policy.action_encoder.weight.grad is not None
    assert model.action_policy.query_encoder.action_queries.grad is not None
    assert metrics["feature/latent_slots"] == 9.0
    assert metrics["feature/fixed_future_timestep"] == 1000.0
