"""Trainer integration for Side-Model3's optimizer-step EMA contract."""

from __future__ import annotations

from model3.third_party.light_wam.src.lightwam.trainer import Wan22Trainer

from .contracts import validate_training_data_config


class SideModel3AdapterTrainer(Wan22Trainer):
    """Light-WAM trainer with target-encoder EMA attached to the optimizer."""

    def __init__(self, model, train_dataset, val_dataset=None, *, cfg):
        validate_training_data_config(cfg.data.train)
        if cfg.data.get("val") is not None:
            validate_training_data_config(cfg.data.val)
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            cfg=cfg,
        )
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.ema_optimizer_hook_handle = unwrapped_model.register_ema_optimizer_hook(
            self.optimizer
        )
