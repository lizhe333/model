from pathlib import Path

import pytest
import torch

from model3.third_party.light_wam.src.lightwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from model3.models import Model3WAM, VLAQueryDiTActionExpert


def _make_expert() -> VLAQueryDiTActionExpert:
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


def _make_layer_states(batch_size: int = 2):
    return [
        {"layer_idx": layer_idx, "adapted": torch.randn(batch_size, 6, 8)}
        for layer_idx in (8, 16, 24)
    ]


def test_vla_queries_are_recurrent_and_flow_gradients_reach_all_components():
    torch.manual_seed(0)
    expert = _make_expert()
    layer_states = _make_layer_states()
    proprio = torch.randn(2, 2)
    noisy_action = torch.randn(2, 4, 3)
    timestep = torch.tensor([100.0, 800.0])

    query_memory, query_trace = expert.encode_queries(layer_states, proprio)
    velocity = expert(layer_states, noisy_action, timestep, proprio=proprio)

    assert query_memory.shape == (2, 5, 16)
    assert query_trace.shape == (2, 3, 5, 16)
    assert velocity.shape == (2, 4, 3)
    assert torch.isfinite(velocity).all()

    velocity.square().mean().backward()
    assert expert.query_encoder.action_queries.grad is not None
    assert all(
        layer_blocks[0].cross_attention.in_proj_weight.grad is not None
        for layer_blocks in expert.query_encoder.layer_blocks
    )
    assert expert.action_encoder.weight.grad is not None
    assert expert.query_context_projection.weight.grad is not None


def test_early_layer_changes_propagate_and_layer_order_matters():
    torch.manual_seed(1)
    expert = _make_expert().eval()
    layer_states = _make_layer_states(batch_size=1)
    proprio = torch.zeros(1, 2)

    _, baseline_trace = expert.encode_queries(layer_states, proprio)
    changed_states = [dict(state) for state in layer_states]
    changed_states[0] = dict(changed_states[0])
    changed_states[0]["adapted"] = changed_states[0]["adapted"] + 1.0
    _, changed_trace = expert.encode_queries(changed_states, proprio)
    swapped_memory, _ = expert.encode_queries(list(reversed(layer_states)), proprio)

    assert not torch.allclose(baseline_trace[:, 0], changed_trace[:, 0])
    assert not torch.allclose(baseline_trace[:, -1], changed_trace[:, -1])
    assert not torch.allclose(baseline_trace[:, -1], swapped_memory)


def test_vla_query_flow_sampling_is_seeded_and_enforces_policy_horizon():
    torch.manual_seed(2)
    expert = _make_expert().eval()
    layer_states = _make_layer_states()
    proprio = torch.randn(2, 2)
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    first = expert.sample(
        layer_states,
        action_horizon=4,
        scheduler=scheduler,
        num_inference_steps=2,
        proprio=proprio,
        generator=torch.Generator(device="cpu").manual_seed(7),
        noise_device="cpu",
    )
    second = expert.sample(
        layer_states,
        action_horizon=4,
        scheduler=scheduler,
        num_inference_steps=2,
        proprio=proprio,
        generator=torch.Generator(device="cpu").manual_seed(7),
        noise_device="cpu",
    )

    assert first.shape == (2, 4, 3)
    assert torch.allclose(first, second)
    with pytest.raises(ValueError, match="requires action_horizon=4"):
        expert.sample(
            layer_states,
            action_horizon=3,
            scheduler=scheduler,
            num_inference_steps=1,
            proprio=proprio,
        )


def _minimal_model3() -> Model3WAM:
    model = Model3WAM.__new__(Model3WAM)
    torch.nn.Module.__init__(model)
    model.mot = torch.nn.Linear(3, 3)
    model.state_fusion_action_expert = _make_expert()
    model.proprio_encoder = torch.nn.Linear(2, 4)
    model.torch_dtype = torch.float32
    return model


def test_model3_checkpoint_round_trip_and_rejects_a1_identity(tmp_path: Path):
    torch.manual_seed(3)
    source = _minimal_model3()
    checkpoint = tmp_path / "model3.pt"
    source.save_checkpoint(checkpoint, step=12)

    restored = _minimal_model3()
    payload = restored.load_checkpoint(checkpoint)
    assert payload["method_id"] == Model3WAM.method_id
    assert payload["step"] == 12
    for expected, actual in zip(source.action_policy.parameters(), restored.action_policy.parameters()):
        assert torch.equal(expected, actual)

    invalid_checkpoint = tmp_path / "a1.pt"
    torch.save(
        {
            "mot": source.mot.state_dict(),
            "state_fusion_action_decoder_type": "dit_flow",
        },
        invalid_checkpoint,
    )
    with pytest.raises(ValueError, match="method_id"):
        restored.load_checkpoint(invalid_checkpoint)


def test_production_model3_action_policy_is_smaller_than_statefusion_a1():
    with torch.device("meta"):
        expert = VLAQueryDiTActionExpert(
            video_hidden_dim=1536,
            action_dim=7,
            num_fusion_layers=3,
            proprio_dim=8,
            query_dim=512,
            num_action_queries=64,
            query_num_heads=8,
            query_bridge_depth=2,
            hidden_dim=512,
            ffn_dim=2048,
            num_heads=8,
            attn_head_dim=64,
            num_layers=16,
            freq_dim=256,
            action_horizon=8,
        )

    parameter_count = sum(parameter.numel() for parameter in expert.parameters())
    assert parameter_count < 120_000_000
