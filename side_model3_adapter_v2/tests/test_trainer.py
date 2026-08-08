"""DeepSpeed-compatible EMA step integration for Side-Model3-Adapter."""

import torch
from torch import nn

from side_model3_adapter_v2.trainer import SideModel3AdapterV2Trainer


class _FakeOptimizer:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> str:
        self.step_count += 1
        return "stepped"


class _FakeAccelerator:
    optimizer_step_was_skipped = False


class _FakeModel:
    def __init__(self) -> None:
        self.ema_update_count = 0
        self.target_predictive_encoder = nn.Sequential(
            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
        ).to(dtype=torch.bfloat16)
        self.target_wan_adapters = nn.ModuleDict(
            {"8": nn.Linear(2, 2)}
        ).to(dtype=torch.bfloat16)

    def update_ema_after_optimizer_step(self) -> None:
        self.ema_update_count += 1


def test_adapter_ema_callback_runs_only_after_executed_optimizer_step() -> None:
    trainer = object.__new__(SideModel3AdapterV2Trainer)
    trainer.optimizer = _FakeOptimizer()
    trainer.accelerator = _FakeAccelerator()
    model = _FakeModel()

    trainer._install_ema_optimizer_step_callback(model)

    assert trainer.optimizer.step() == "stepped"
    assert trainer.optimizer.step_count == 1
    assert model.ema_update_count == 1

    trainer.accelerator.optimizer_step_was_skipped = True
    assert trainer.optimizer.step() == "stepped"
    assert trainer.optimizer.step_count == 2
    assert model.ema_update_count == 1


def test_adapter_restore_ema_targets_fp32_after_distributed_prepare() -> None:
    model = _FakeModel()

    SideModel3AdapterV2Trainer._restore_ema_targets_fp32(model)

    targets = (model.target_predictive_encoder, model.target_wan_adapters)
    assert all(
        parameter.dtype == torch.float32
        for target in targets
        for parameter in target.parameters()
    )
    assert all(
        not buffer.is_floating_point() or buffer.dtype == torch.float32
        for target in targets
        for buffer in target.buffers()
    )
