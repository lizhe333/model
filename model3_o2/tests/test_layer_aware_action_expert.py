from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from model3.models import Model3WAM, VLAQueryDiTActionExpert
from model3_o2.models import (
    LayerSeparableGatedResidualReadout,
    Model3O2WAM,
    VLAQueryLayerAwareDiTActionExpert,
)


def _expert(cls=VLAQueryLayerAwareDiTActionExpert):
    return cls(
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


def _states(batch: int = 2):
    return [
        {"layer_idx": layer, "adapted": torch.randn(batch, 6, 8)}
        for layer in (8, 16, 24)
    ]


def _minimal_wam(cls, policy):
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.mot = torch.nn.Linear(3, 3)
    model.state_fusion_action_expert = policy
    model.proprio_encoder = torch.nn.Linear(2, 4)
    model.torch_dtype = torch.float32
    return model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_zero_initialized_readout_is_exact_q3_identity() -> None:
    torch.manual_seed(0)
    expert = _expert().eval()
    routed, trace = expert.encode_queries(_states(), torch.randn(2, 2))

    assert trace.shape == (2, 3, 5, 16)
    assert routed.shape == (2, 5, 16)
    assert torch.equal(routed, trace[:, -1])
    assert torch.count_nonzero(expert.layer_readout.residual_scales) == 0


def test_enabled_readout_depends_on_earlier_layers_and_order() -> None:
    torch.manual_seed(1)
    readout = LayerSeparableGatedResidualReadout(num_layers=3, query_dim=4)
    with torch.no_grad():
        readout.residual_scales.fill_(0.75)
        for index, projection in enumerate(readout.value_projections, start=1):
            projection.weight.copy_(torch.eye(4) * index)
            projection.bias.zero_()
        for gate in readout.gate_projections:
            gate.weight.zero_()
            gate.bias.zero_()

    trace = torch.randn(2, 3, 5, 4)
    output = readout(trace)
    changed = trace.clone()
    changed[:, 0] += torch.tensor([0.5, -0.25, 0.75, 0.1])
    changed_output = readout(changed)
    swapped_output = readout(trace[:, [1, 0, 2]])

    assert not torch.allclose(output, changed_output)
    assert not torch.allclose(output, swapped_output)


def test_action_flow_uses_routed_memory_and_gradients_reach_all_modules() -> None:
    torch.manual_seed(2)
    expert = _expert()
    with torch.no_grad():
        expert.layer_readout.residual_scales.fill_(0.25)
    velocity = expert(
        _states(),
        torch.randn(2, 4, 3),
        torch.tensor([100.0, 700.0]),
        proprio=torch.randn(2, 2),
    )
    assert velocity.shape == (2, 4, 3)
    velocity.square().mean().backward()

    assert expert.query_encoder.action_queries.grad is not None
    assert expert.action_encoder.weight.grad is not None
    assert expert.blocks[0].self_attn.q.weight.grad is not None
    assert expert.layer_readout.residual_scales.grad is not None
    assert all(module.weight.grad is not None for module in expert.layer_readout.value_projections)
    assert all(module.weight.grad is not None for module in expert.layer_readout.gate_projections)


def test_o2_only_adds_layer_readout_to_the_model3_action_dit() -> None:
    torch.manual_seed(3)
    parent = _expert(VLAQueryDiTActionExpert)
    torch.manual_seed(3)
    o2 = _expert()
    parent_state = parent.state_dict()
    o2_state = o2.state_dict()
    shared = set(parent_state) & set(o2_state)
    added = set(o2_state) - set(parent_state)

    assert set(parent_state) == shared
    assert added
    assert all(key.startswith("layer_readout.") for key in added)
    assert all(parent_state[key].shape == o2_state[key].shape for key in shared)
    assert len(parent.blocks) == len(o2.blocks) == 2
    assert type(parent.blocks[0]) is type(o2.blocks[0])
    assert type(parent.head) is type(o2.head)


def test_model3_warmstart_loads_every_shared_tensor_and_only_misses_readout(
    tmp_path: Path,
) -> None:
    torch.manual_seed(4)
    parent = _minimal_wam(Model3WAM, _expert(VLAQueryDiTActionExpert))
    checkpoint = tmp_path / "model3-step20k.pt"
    parent.save_checkpoint(checkpoint, step=20_000)

    torch.manual_seed(9)
    o2 = _minimal_wam(Model3O2WAM, _expert())
    o2.load_model3_warmstart(
        checkpoint,
        expected_sha256=_sha256(checkpoint),
        expected_step=20_000,
    )

    parent_state = parent.action_policy.state_dict()
    o2_state = o2.action_policy.state_dict()
    assert all(torch.equal(value, o2_state[key]) for key, value in parent_state.items())
    assert torch.count_nonzero(o2.action_policy.layer_readout.residual_scales) == 0
    assert o2.model3_warmstart_identity["step"] == 20_000


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("method_id", "wrong", "method_id"),
        ("model_class", "WrongModel", "model_class"),
        ("step", 19_999, "step"),
    ],
)
def test_model3_warmstart_rejects_wrong_identity(
    tmp_path: Path,
    field: str,
    value,
    match: str,
) -> None:
    parent = _minimal_wam(Model3WAM, _expert(VLAQueryDiTActionExpert))
    checkpoint = tmp_path / "parent.pt"
    parent.save_checkpoint(checkpoint, step=20_000)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload[field] = value
    torch.save(payload, checkpoint)
    o2 = _minimal_wam(Model3O2WAM, _expert())

    with pytest.raises(ValueError, match=match):
        o2.load_model3_warmstart(
            checkpoint,
            expected_sha256=_sha256(checkpoint),
            expected_step=20_000,
        )


def test_model3_warmstart_rejects_wrong_hash_and_shared_shape(tmp_path: Path) -> None:
    parent = _minimal_wam(Model3WAM, _expert(VLAQueryDiTActionExpert))
    checkpoint = tmp_path / "parent.pt"
    parent.save_checkpoint(checkpoint, step=20_000)
    o2 = _minimal_wam(Model3O2WAM, _expert())
    with pytest.raises(ValueError, match="SHA mismatch"):
        o2.load_model3_warmstart(
            checkpoint, expected_sha256="0" * 64, expected_step=20_000
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = next(iter(payload["action_policy_state_dict"]))
    payload["action_policy_state_dict"][key] = torch.zeros(1)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="tensors are incompatible"):
        o2.load_model3_warmstart(
            checkpoint,
            expected_sha256=_sha256(checkpoint),
            expected_step=20_000,
        )


def test_o2_same_track_checkpoint_round_trip_is_strict(tmp_path: Path) -> None:
    torch.manual_seed(5)
    source = _minimal_wam(Model3O2WAM, _expert())
    checkpoint = tmp_path / "o2.pt"
    source.save_checkpoint(checkpoint, step=7)
    restored = _minimal_wam(Model3O2WAM, _expert())
    restored.load_checkpoint(checkpoint)

    for key, value in source.action_policy.state_dict().items():
        assert torch.equal(value, restored.action_policy.state_dict()[key])

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["method_id"] = Model3WAM.method_id
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="method_id"):
        restored.load_checkpoint(checkpoint)


def test_production_readout_adds_less_than_one_million_parameters() -> None:
    readout = LayerSeparableGatedResidualReadout(num_layers=3, query_dim=512)
    parameter_count = sum(parameter.numel() for parameter in readout.parameters())
    assert parameter_count == 530_436
    assert parameter_count < 1_000_000
