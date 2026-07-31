from __future__ import annotations

import torch

from model3.models import Model3WAM, VLAQueryDiTActionExpert
from model3.third_party.light_wam.src.lightwam.models.wan22.schedulers.scheduler_continuous import (
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


def test_model3_joint_loss_uses_future_flow_and_policy_horizon() -> None:
    torch.manual_seed(11)
    model = Model3WAM.__new__(Model3WAM)
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


def test_model3_benchmark_infer_action_reports_deployed_stages() -> None:
    class TinyVideoExpert(torch.nn.Module):
        video_attention_mask_mode = "first_frame_causal"
        fuse_vae_embedding_in_latents = False

        def forward_backbone(self, video_pre):
            return video_pre

    torch.manual_seed(13)
    model = Model3WAM.__new__(Model3WAM)
    torch.nn.Module.__init__(model)
    model.state_fusion_action_expert = _tiny_policy()
    model.video_expert = TinyVideoExpert()
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_dim = 2
    model.infer_action_scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    layer_states = [
        {"layer_idx": layer_idx, "adapted": torch.randn(1, 6, 8)}
        for layer_idx in (8, 16, 24)
    ]
    model._encode_input_image_latents_tensor = lambda **kwargs: torch.zeros(1, 1, 1, 1, 1)
    model._prepare_model3_context = lambda **kwargs: (
        kwargs["context"],
        kwargs["context_mask"],
    )
    model._build_action_observation_video_pre = lambda **kwargs: {"tokens": torch.zeros(1, 1, 1)}
    model._build_multilayer_action_fusion_inputs = lambda: layer_states

    result = model.benchmark_infer_action(
        prompt=None,
        input_image=torch.randn(1, 3, 16, 16),
        action_horizon=4,
        proprio=torch.randn(1, 2),
        context=torch.randn(1, 2, 4),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        num_inference_steps=2,
        seed=7,
    )

    assert tuple(result["action"].shape) == (4, 3)
    assert result["method_id"] == model.method_id
    assert set(result["timings_s"]) == {
        "vae_encode",
        "condition_prepare",
        "model_prepare_observation",
        "model_action_backbone",
        "model_query_encoder",
        "model_action_flow_solver",
        "model_predict",
        "total",
    }
    assert all(value >= 0.0 for value in result["timings_s"].values())
