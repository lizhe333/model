from __future__ import annotations

import json
import time

import torch

from model5.models import Model5WAM, VLAQueryDiTActionExpert
from model5.third_party.light_wam.src.lightwam.trainer import Wan22Trainer


class _CpuAccelerator:
    device = torch.device("cpu")
    mixed_precision = "bf16"
    is_main_process = True
    num_processes = 1

    @staticmethod
    def wait_for_everyone() -> None:
        return None


def test_benchmark_records_loss_gradient_feature_and_inference_metrics(tmp_path) -> None:
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _CpuAccelerator()
    trainer._benchmark_start_time = time.perf_counter() - 0.01
    trainer.benchmark_description = "model5 single-gpu smoke"
    trainer.benchmark_warmup_steps = 0
    trainer.benchmark_measure_steps = 1
    trainer.batch_size = 1
    trainer.gradient_accumulation_steps = 1
    trainer.total_params = 100
    trainer.trainable_params = 25
    trainer.global_step = 1
    trainer.output_dir = str(tmp_path)
    trainer.benchmark_output_filename = "model5_smoke_benchmark.json"

    metrics = {
        "loss": 1.5,
        "grad_norm": 0.75,
        "optimizer_step_was_skipped": False,
        "loss_video_raw": 0.5,
        "loss_action_raw": 1.0,
        "feature/latent_slots": 9.0,
        "gradient/query_encoder_has_grad": True,
        "gradient/wan_adapter_has_grad": True,
        "gradient/wan_lora_has_grad": True,
        "gradient/frozen_wan_base_has_grad": False,
        "inference/action_finite": True,
        "inference/action_shape_ok": True,
        "inference/reproducible": True,
    }
    payload = trainer._finalize_benchmark(final_step_metrics=metrics)

    saved = json.loads(
        (tmp_path / "model5_smoke_benchmark.json").read_text(encoding="utf-8")
    )
    assert payload["final_step_metrics"] == metrics
    assert saved["final_step_metrics"] == metrics
    assert saved["global_step"] == 1
    assert saved["effective_global_batch_size"] == 1


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


class _TinyVideoExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wam_adapters = torch.nn.Linear(2, 2)
        self.lora_proj = torch.nn.Linear(2, 2)
        self.base = torch.nn.Linear(2, 2)
        self.base.requires_grad_(False)


def test_gradient_smoke_summary_separates_peft_and_frozen_backbone() -> None:
    model = Model5WAM.__new__(Model5WAM)
    torch.nn.Module.__init__(model)
    model.video_expert = _TinyVideoExpert()
    model.state_fusion_action_expert = _tiny_policy()

    model.action_policy.query_encoder.action_queries.grad = torch.ones_like(
        model.action_policy.query_encoder.action_queries
    )
    model.video_expert.wam_adapters.weight.grad = torch.ones_like(
        model.video_expert.wam_adapters.weight
    )
    model.video_expert.lora_proj.weight.grad = torch.ones_like(
        model.video_expert.lora_proj.weight
    )

    summary = model.gradient_smoke_summary()

    assert summary["gradient/query_encoder_has_grad"]
    assert summary["gradient/wan_adapter_has_grad"]
    assert summary["gradient/wan_lora_has_grad"]
    assert not summary["gradient/frozen_wan_base_has_grad"]
    assert summary["gradient/query_encoder_norm"] > 0
    assert summary["gradient/wan_adapter_norm"] > 0
    assert summary["gradient/wan_lora_norm"] > 0
