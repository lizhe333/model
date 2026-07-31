from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model3.third_party.light_wam.src.lightwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from model3_regression.models import (
    Model3RegressionWAM,
    VLAQueryRegressionActionExpert,
)


def _tiny_policy() -> VLAQueryRegressionActionExpert:
    return VLAQueryRegressionActionExpert(
        video_hidden_dim=8,
        action_dim=3,
        num_fusion_layers=3,
        proprio_dim=2,
        query_dim=16,
        num_action_queries=5,
        query_num_heads=2,
        query_bridge_depth=1,
        regression_hidden_dim=16,
        regression_num_heads=2,
        regression_num_layers=2,
        regression_ffn_multiplier=2,
        action_horizon=4,
    )


def _layer_states(batch_size: int) -> list[dict[str, torch.Tensor | int]]:
    return [
        {"layer_idx": layer_idx, "adapted": torch.randn(batch_size, 6, 8)}
        for layer_idx in (8, 16, 24)
    ]


def _minimal_regression_model() -> Model3RegressionWAM:
    model = Model3RegressionWAM.__new__(Model3RegressionWAM)
    torch.nn.Module.__init__(model)
    model.mot = torch.nn.Linear(3, 3)
    model.state_fusion_action_expert = _tiny_policy()
    model.proprio_encoder = torch.nn.Linear(2, 4)
    model.torch_dtype = torch.float32
    return model


def test_direct_policy_is_deterministic_and_query_dependent() -> None:
    torch.manual_seed(5)
    policy = _tiny_policy().eval()
    layer_states = _layer_states(batch_size=2)
    proprio = torch.randn(2, 2)

    first = policy(layer_states, proprio)
    second = policy.sample(
        layer_states,
        action_horizon=4,
        num_inference_steps=1,
        proprio=proprio,
        generator=torch.Generator().manual_seed(999),
    )
    changed_states = [dict(state) for state in layer_states]
    changed_states[0] = {
        **changed_states[0],
        "adapted": changed_states[0]["adapted"] + 1.0,
    }
    changed = policy(changed_states, proprio)

    assert tuple(first.shape) == (2, 4, 3)
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, changed)


def test_direct_policy_rejects_solver_semantics() -> None:
    policy = _tiny_policy().eval()
    query_memory = torch.randn(1, 5, 16)

    try:
        policy.sample_from_queries(
            query_memory,
            action_horizon=4,
            num_inference_steps=5,
        )
    except ValueError as error:
        assert "direct regression" in str(error)
    else:
        raise AssertionError("direct regression accepted iterative solver steps")


def test_masked_l1_excludes_padded_actions() -> None:
    model = Model3RegressionWAM.__new__(Model3RegressionWAM)
    torch.nn.Module.__init__(model)
    model.action_temporal_weighting_enabled = False
    model.action_temporal_weighting_num_prefix_steps = None
    model.action_temporal_weighting_prefix_weight = 1.0
    model.action_temporal_weighting_tail_weight = 1.0
    prediction = torch.tensor([[[1.0, -1.0], [100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    is_pad = torch.tensor([[False, True]])

    loss = model._compute_regression_action_loss_per_sample(
        prediction,
        target,
        is_pad,
    )

    torch.testing.assert_close(loss, torch.tensor([1.0]))


def test_joint_training_loss_reaches_queries_and_regression_decoder() -> None:
    torch.manual_seed(11)
    model = Model3RegressionWAM.__new__(Model3RegressionWAM)
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
    layer_states = _layer_states(batch_size)
    model._build_action_layer_states = lambda **kwargs: layer_states

    loss, metrics = model.training_loss(sample={})
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["loss_video_raw"] == 0.0
    assert metrics["loss_action_raw"] > 0.0
    assert model.action_policy.action_projection.weight.grad is not None
    assert model.action_policy.action_slots.grad is not None
    assert model.action_policy.query_encoder.action_queries.grad is not None


def test_regression_checkpoint_round_trip_and_legacy_nested_identity(tmp_path: Path) -> None:
    source = _minimal_regression_model()
    checkpoint = tmp_path / "regression.pt"
    source.save_checkpoint(checkpoint, step=20)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["action_policy_config"]["method_id"] == source.method_id

    legacy_payload = dict(payload)
    legacy_payload["action_policy_config"] = dict(payload["action_policy_config"])
    legacy_payload["action_policy_config"].pop("method_id")
    legacy_checkpoint = tmp_path / "regression_legacy.pt"
    torch.save(legacy_payload, legacy_checkpoint)
    restored = _minimal_regression_model()
    restored.load_checkpoint(legacy_checkpoint)

    invalid_payload = dict(payload)
    invalid_payload["action_policy_config"] = dict(payload["action_policy_config"])
    invalid_payload["action_policy_config"]["method_id"] = "wrong_method"
    invalid_checkpoint = tmp_path / "regression_invalid.pt"
    torch.save(invalid_payload, invalid_checkpoint)
    with pytest.raises(ValueError, match="mismatched method_id"):
        restored.load_checkpoint(invalid_checkpoint)
