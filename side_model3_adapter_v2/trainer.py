"""Trainer integration for Side-Model3's optimizer-step EMA contract."""

from __future__ import annotations

from model3.third_party.light_wam.src.lightwam.trainer import Wan22Trainer

from .contracts import validate_training_data_config


class SideModel3AdapterV2Trainer(Wan22Trainer):
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
        self._restore_ema_targets_fp32(unwrapped_model)
        self._install_ema_optimizer_step_callback(unwrapped_model)

    @staticmethod
    def _restore_ema_targets_fp32(unwrapped_model) -> None:
        unwrapped_model.target_predictive_encoder.float()
        unwrapped_model.target_wan_adapters.float()

    def _install_ema_optimizer_step_callback(self, unwrapped_model) -> None:
        optimizer_step = self.optimizer.step

        def optimizer_step_with_ema(*args, **kwargs):
            result = optimizer_step(*args, **kwargs)
            if not self.accelerator.optimizer_step_was_skipped:
                unwrapped_model.update_ema_after_optimizer_step()
            return result

        self.optimizer.step = optimizer_step_with_ema
