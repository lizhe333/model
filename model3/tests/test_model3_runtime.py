import pytest

from model3 import runtime
from model3.models import Model3WAM


def _factory_kwargs():
    return {
        "model_id": "test-model",
        "tokenizer_model_id": "test-tokenizer",
        "video_dit_config": {"text_dim": 16},
        "action_dit_config": {"action_dim": 3},
        "video_scheduler": {},
        "action_scheduler": {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        },
        "wam_adapter": {
            "use_wam_adapter": True,
            "remove_original_action_expert": True,
        },
        "action_query_policy_config": {
            "num_action_queries": 64,
            "action_horizon": 8,
        },
    }


def test_model3_factory_passes_public_query_config_to_model3(monkeypatch):
    captured = {}

    def fake_from_pretrained(cls, **kwargs):
        captured.update(kwargs)
        return "model3"

    monkeypatch.setattr(Model3WAM, "from_wan22_pretrained", classmethod(fake_from_pretrained))
    result = runtime.create_model3_wam(**_factory_kwargs())

    assert result == "model3"
    policy_config = captured["action_query_policy_config"]
    assert policy_config["num_action_queries"] == 64
    assert policy_config["action_horizon"] == 8


def test_model3_factory_rejects_state_fusion_config():
    kwargs = _factory_kwargs()
    kwargs["state_fusion_action_expert_config"] = {"decoder_type": "dit_flow"}
    with pytest.raises(ValueError, match="cannot consume a StateFusion"):
        runtime.create_model3_wam(**kwargs)
