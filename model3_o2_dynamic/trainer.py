"""Stage-2 trainer with a sealed response-adapter freeze/unfreeze transition.

This is deliberately separate from the vendored Light-WAM trainer.  The base
trainer has one parameter group and cannot represent the Dynamic contract that
the adapters exist in the optimizer from step zero, create no Adam state while
frozen, and then track PEFT LR at exactly $0.1\times$ after step 5K.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from omegaconf import DictConfig

from model3.third_party.light_wam.src.lightwam.trainer import Wan22Trainer
from model3.third_party.light_wam.src.lightwam.utils.fs import ensure_dir
from model3.third_party.light_wam.src.lightwam.utils.logging_config import get_logger
from model3.third_party.light_wam.src.lightwam.utils.pytorch_utils import set_global_seed

from .models import Model3O2DynamicWAM


logger = get_logger(__name__)


FREEZE_THROUGH_STEP = 5_000
FIRST_ADAPTER_UPDATE_STEP = 5_001
ADAPTER_LR_SCALE = 0.1
GATE_LR_SCALE = 1.0
GATE_SCHEDULE_BY_LOCAL_BUDGET = {
    10_000: (0, 1),
    35_000: (30_000, 30_001),
}


class CoupledAdapterScheduler:
    """Keep one adapter group at an exact scale of the inherited PEFT schedule."""

    def __init__(
        self,
        scheduler: Any,
        optimizer: torch.optim.Optimizer,
        *,
        base_group_index: int,
        adapter_group_index: int,
        scale: float,
        additional_group_scales: dict[int, float] | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.base_group_index = int(base_group_index)
        self.adapter_group_index = int(adapter_group_index)
        self.scale = float(scale)
        self.coupled_group_scales = {self.adapter_group_index: self.scale}
        for index, group_scale in (additional_group_scales or {}).items():
            normalized_index = int(index)
            if normalized_index == self.base_group_index:
                raise ValueError("the base optimizer group cannot couple to itself")
            if normalized_index in self.coupled_group_scales:
                raise ValueError("duplicate coupled optimizer group index")
            self.coupled_group_scales[normalized_index] = float(group_scale)
        self._enforce_ratio()

    def _enforce_ratio(self) -> None:
        base_lr = self.optimizer.param_groups[self.base_group_index]["lr"]
        for group_index, group_scale in self.coupled_group_scales.items():
            self.optimizer.param_groups[group_index]["lr"] = base_lr * group_scale

    def step(self, *args: Any, **kwargs: Any) -> Any:
        result = self.scheduler.step(*args, **kwargs)
        self._enforce_ratio()
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler.state_dict(),
            "base_group_index": self.base_group_index,
            "adapter_group_index": self.adapter_group_index,
            "scale": self.scale,
            "coupled_group_scales": self.coupled_group_scales,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if int(state_dict["base_group_index"]) != self.base_group_index:
            raise ValueError("Stage 2 scheduler base group identity changed on resume")
        if int(state_dict["adapter_group_index"]) != self.adapter_group_index:
            raise ValueError("Stage 2 scheduler adapter group identity changed on resume")
        if float(state_dict["scale"]) != self.scale:
            raise ValueError("Stage 2 scheduler adapter LR scale changed on resume")
        saved_scales = {
            int(index): float(value)
            for index, value in state_dict.get("coupled_group_scales", {}).items()
        }
        if saved_scales != self.coupled_group_scales:
            raise ValueError("Stage 2 scheduler coupled optimizer groups changed on resume")
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self._enforce_ratio()

    def get_last_lr(self) -> list[float]:
        values = list(self.scheduler.get_last_lr())
        for group_index, group_scale in self.coupled_group_scales.items():
            values[group_index] = values[self.base_group_index] * group_scale
        return values


class DynamicWan22Trainer(Wan22Trainer):
    """The original joint O2 trainer plus the predeclared adapter schedule."""

    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        if not isinstance(model, Model3O2DynamicWAM):
            raise TypeError(
                "DynamicWan22Trainer requires Model3O2DynamicWAM, "
                f"got {type(model).__name__}"
            )
        schedule_cfg = cfg.get("dynamic_response_schedule", cfg.model.get("dynamic_response_schedule", {}))
        self.freeze_through_step = int(schedule_cfg.get("freeze_through_step", FREEZE_THROUGH_STEP))
        self.first_adapter_update_step = int(
            schedule_cfg.get("first_adapter_update_step", FIRST_ADAPTER_UPDATE_STEP)
        )
        self.adapter_lr_scale = float(schedule_cfg.get("adapter_lr_scale", ADAPTER_LR_SCALE))
        self.gate_freeze_through_step = int(schedule_cfg.get("gate_freeze_through_step", -1))
        self.first_gate_update_step = int(schedule_cfg.get("first_gate_update_step", -1))
        self.gate_lr_scale = float(schedule_cfg.get("gate_lr_scale", GATE_LR_SCALE))
        configured_max_steps = int(cfg.max_steps) if cfg.max_steps is not None else None
        expected_gate_schedule = GATE_SCHEDULE_BY_LOCAL_BUDGET.get(configured_max_steps)
        if (
            self.freeze_through_step != FREEZE_THROUGH_STEP
            or self.first_adapter_update_step != FIRST_ADAPTER_UPDATE_STEP
            or self.adapter_lr_scale != ADAPTER_LR_SCALE
            or self.gate_lr_scale != GATE_LR_SCALE
            or expected_gate_schedule is None
            or (self.gate_freeze_through_step, self.first_gate_update_step)
            != expected_gate_schedule
        ):
            raise ValueError(
                "Dynamic Stage 2 schedule does not match the registered adapter/O2-gate boundaries"
            )
        self.response_adapter_group_index = 1
        self.o2_gate_group_index = 2
        self._freeze_boundary_checkpoint_recorded = False
        self._gate_boundary_checkpoint_recorded = self.gate_freeze_through_step == 0

        # This is an intentional, isolated copy of Wan22Trainer initialization.
        # It differs only at optimizer construction, where the A parameters get a
        # dedicated group before Accelerate/ZeRO prepares any optimizer state.
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        warmup_steps = cfg.get("warmup_steps")
        self.warmup_steps = int(warmup_steps) if warmup_steps is not None else None
        self.warmup_ratio = float(cfg.get("warmup_ratio", 0.05))
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError(f"`warmup_steps` must be >= 0, got {self.warmup_steps}.")
        if self.warmup_ratio < 0:
            raise ValueError(f"`warmup_ratio` must be >= 0, got {self.warmup_ratio}.")
        self.max_steps = int(cfg.max_steps) if cfg.max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.resume = cfg.resume
        self.resume_path = self._resolve_resume_path()
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(f"Unsupported mixed_precision: {cfg.mixed_precision}.")
        timing_cfg = cfg.get("timing_breakdown", {})
        self.timing_breakdown_enabled = bool(timing_cfg.get("enabled", False))
        self.timing_breakdown_sync_cuda = bool(timing_cfg.get("sync_cuda", True))
        self._timing_accumulator: dict[str, float] = {}
        train_vis_cfg = cfg.get("train_visualization", {})
        self.train_visualization_enabled = bool(train_vis_cfg.get("enabled", False))
        self.train_visualization_every = int(train_vis_cfg.get("every", 0))
        self.train_visualization_fps = int(train_vis_cfg.get("fps", 8))
        self.train_visualization_tiled = bool(train_vis_cfg.get("tiled", False))
        self.train_action_fit_enabled = bool(train_vis_cfg.get("action_fit_enabled", False))
        action_fit_steps = train_vis_cfg.get("action_fit_num_steps", None)
        self.train_action_fit_num_steps = (
            None if action_fit_steps in (None, "", "null") else int(action_fit_steps)
        )
        if self.train_action_fit_num_steps is not None and self.train_action_fit_num_steps <= 0:
            raise ValueError("train_visualization.action_fit_num_steps must be positive")
        parameter_report_cfg = cfg.get("parameter_report", {})
        self.parameter_report_enabled = bool(parameter_report_cfg.get("enabled", False))
        self.parameter_report_filename = (
            str(parameter_report_cfg.get("filename", "parameter_report.json")).strip()
            or "parameter_report.json"
        )
        benchmark_cfg = cfg.get("benchmark", {})
        self.benchmark_enabled = bool(benchmark_cfg.get("enabled", False))
        self.benchmark_warmup_steps = int(benchmark_cfg.get("warmup_steps", 10))
        self.benchmark_measure_steps = int(benchmark_cfg.get("measure_steps", 50))
        self.benchmark_output_filename = (
            str(benchmark_cfg.get("output_filename", "training_speed_benchmark.json")).strip()
            or "training_speed_benchmark.json"
        )
        benchmark_description = benchmark_cfg.get("description")
        self.benchmark_description = (
            None if benchmark_description in (None, "", "null") else str(benchmark_description)
        )
        self._benchmark_start_time = None
        self.wandb_enabled = bool(cfg.wandb.enabled)
        if self.benchmark_enabled:
            if self.benchmark_warmup_steps < 0 or self.benchmark_measure_steps <= 0:
                raise ValueError("invalid Dynamic trainer benchmark configuration")
            self.benchmark_total_steps = self.benchmark_warmup_steps + self.benchmark_measure_steps
            if self.max_steps is None or self.max_steps != self.benchmark_total_steps:
                self.max_steps = self.benchmark_total_steps
        else:
            self.benchmark_total_steps = None

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        if hasattr(self.model, "set_timing_breakdown"):
            self.model.set_timing_breakdown(
                enabled=self.timing_breakdown_enabled,
                sync_cuda=self.timing_breakdown_sync_cuda,
            )

        # Include A in a real, independent optimizer group before DDP/ZeRO sees
        # the model.  The inherited O2 ``layer_readout`` likewise has its own
        # group from step zero; both become requires_grad=False after prepare.
        self.model.set_response_adapters_trainable(True)
        self.model.set_o2_gate_trainable(True)
        self._apply_dit_only_train_mode(self.model)
        self._maybe_load_weight_checkpoint_before_prepare()
        if hasattr(self.model, "log_parameter_summary"):
            self.model.log_parameter_summary()
        self.total_params = int(sum(parameter.numel() for parameter in self.model.parameters()))
        response_parameters = list(self.model.response_adapters.parameters())
        gate_parameters = list(self.model.action_policy.layer_readout.parameters())
        response_ids = {id(parameter) for parameter in response_parameters}
        gate_ids = {id(parameter) for parameter in gate_parameters}
        if response_ids & gate_ids:
            raise RuntimeError("response-adapter and original O2-gate parameters must be disjoint")
        original_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in response_ids and id(parameter) not in gate_ids
        ]
        if not original_parameters or not response_parameters or not gate_parameters:
            raise RuntimeError(
                "Dynamic Stage 2 must have inherited O2, response-adapter, and O2-gate parameters"
            )
        self.trainable_params = int(
            sum(parameter.numel() for parameter in original_parameters + response_parameters + gate_parameters)
        )
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": original_parameters,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "group_name": "o2_peft",
                },
                {
                    "params": response_parameters,
                    "lr": self.learning_rate * self.adapter_lr_scale,
                    "weight_decay": self.weight_decay,
                    "group_name": "response_adapters",
                },
                {
                    "params": gate_parameters,
                    "lr": self.learning_rate * self.gate_lr_scale,
                    "weight_decay": self.weight_decay,
                    "group_name": "o2_layer_readout_gate",
                },
            ],
            betas=(0.9, 0.95),
        )
        self._assert_optimizer_groups(self.optimizer)
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = self.warmup_steps if self.warmup_steps is not None else int(total_train_steps * self.warmup_ratio)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self.train_vis_dir = os.path.join(self.output_dir, "train_vis")
        self.action_fit_dir = os.path.join(self.output_dir, "action_fit")
        for directory in (
            self.output_dir,
            self.checkpoint_root,
            self.weights_dir,
            self.state_dir,
            self.eval_dir,
            self.train_vis_dir,
            self.action_fit_dir,
        ):
            ensure_dir(directory)
        self._maybe_save_parameter_report(self.model)
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        prepared_model = self._dynamic_model()
        if hasattr(prepared_model, "set_timing_breakdown"):
            prepared_model.set_timing_breakdown(
                enabled=self.timing_breakdown_enabled,
                sync_cuda=self.timing_breakdown_sync_cuda,
        )
        prepared_model.set_response_adapters_trainable(False)
        prepared_model.set_o2_gate_trainable(False)
        self.wandb_run = None
        self._init_wandb()
        self._resume_after_prepare()
        if (
            self.gate_freeze_through_step == 0
            and self.global_step == 0
            and not prepared_model.o2_gate_trainable
        ):
            # The Long contract has no frozen gate interval. Keep the gate in
            # the optimizer from step zero, but activate gradients on the first
            # action-loss forward so optimizer state is still created by its
            # first real update.
            prepared_model.schedule_o2_gate_unfreeze()
        self.optimizer.zero_grad(set_to_none=True)
        logger.info(
            "Dynamic Stage 2 initialized: A frozen through=%d then step=%d at %.3fx; "
            "O2 gate frozen through=%d then step=%d at %.3fx",
            self.freeze_through_step,
            self.first_adapter_update_step,
            self.adapter_lr_scale,
            self.gate_freeze_through_step,
            self.first_gate_update_step,
            self.gate_lr_scale,
        )

    def _dynamic_model(self) -> Model3O2DynamicWAM:
        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, Model3O2DynamicWAM):
            raise RuntimeError(f"expected unwrapped Dynamic model, got {type(model).__name__}")
        return model

    def _assert_optimizer_groups(self, optimizer: torch.optim.Optimizer) -> None:
        if len(optimizer.param_groups) != 3:
            raise RuntimeError("Dynamic Stage 2 optimizer must have exactly three parameter groups")
        base, adapters, gate = optimizer.param_groups
        if (
            base.get("group_name") != "o2_peft"
            or adapters.get("group_name") != "response_adapters"
            or gate.get("group_name") != "o2_layer_readout_gate"
        ):
            raise RuntimeError("Dynamic Stage 2 optimizer group identity mismatch")
        if not math.isclose(
            float(adapters["lr"]),
            float(base["lr"]) * self.adapter_lr_scale,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise RuntimeError("Dynamic Stage 2 adapter initial LR ratio is not 0.1")
        if not math.isclose(
            float(gate["lr"]),
            float(base["lr"]) * self.gate_lr_scale,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise RuntimeError("Dynamic Stage 2 O2-gate initial LR ratio is not 1.0")

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        inherited = super()._build_scheduler(
            scheduler_type=scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        return CoupledAdapterScheduler(
            inherited,
            self.optimizer,
            base_group_index=0,
            adapter_group_index=self.response_adapter_group_index,
            scale=self.adapter_lr_scale,
            additional_group_scales={self.o2_gate_group_index: self.gate_lr_scale},
        )

    def _optimizer_state_count(self, parameters) -> int:
        parameter_ids = {id(parameter) for parameter in parameters}
        optimizer = self.optimizer
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        return sum(1 for parameter in optimizer.state if id(parameter) in parameter_ids)

    def _adapter_optimizer_state_count(self) -> int:
        return self._optimizer_state_count(self._dynamic_model().response_adapters.parameters())

    def _gate_optimizer_state_count(self) -> int:
        return self._optimizer_state_count(self._dynamic_model().action_policy.layer_readout.parameters())

    def _save_trainer_state(self, state_path: str):
        super()._save_trainer_state(state_path)
        state_file = Path(state_path) / "trainer_state.json"
        payload = json.loads(state_file.read_text(encoding="utf-8"))
        model = self._dynamic_model()
        payload["dynamic_response_schedule"] = {
            "freeze_through_step": self.freeze_through_step,
            "first_adapter_update_step": self.first_adapter_update_step,
            "adapter_lr_scale": self.adapter_lr_scale,
            "response_adapters_trainable": model.response_adapters_trainable,
            "adapter_pending_unfreeze": bool(
                getattr(model, "_response_adapters_pending_unfreeze", False)
            ),
            "adapter_optimizer_state_count": self._adapter_optimizer_state_count(),
            "base_lr": float(self.optimizer.param_groups[0]["lr"]),
            "adapter_lr": float(self.optimizer.param_groups[1]["lr"]),
            "gate_freeze_through_step": self.gate_freeze_through_step,
            "first_gate_update_step": self.first_gate_update_step,
            "gate_lr_scale": self.gate_lr_scale,
            "o2_gate_trainable": model.o2_gate_trainable,
            "gate_pending_unfreeze": bool(getattr(model, "_o2_gate_pending_unfreeze", False)),
            "gate_optimizer_state_count": self._gate_optimizer_state_count(),
            "gate_lr": float(self.optimizer.param_groups[self.o2_gate_group_index]["lr"]),
        }
        state_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def load_training_state(self, state_dir: str):
        super().load_training_state(state_dir)
        model = self._dynamic_model()
        if self.global_step < self.freeze_through_step:
            model.set_response_adapters_trainable(False)
        elif self.global_step == self.freeze_through_step:
            model.set_response_adapters_trainable(False)
            model.schedule_response_adapter_unfreeze()
        else:
            model.set_response_adapters_trainable(True)
        if self.global_step < self.gate_freeze_through_step:
            model.set_o2_gate_trainable(False)
        elif self.global_step == self.gate_freeze_through_step:
            model.set_o2_gate_trainable(False)
            model.schedule_o2_gate_unfreeze()
        else:
            model.set_o2_gate_trainable(True)
        self._freeze_boundary_checkpoint_recorded = self.global_step >= self.freeze_through_step
        self._gate_boundary_checkpoint_recorded = self.global_step >= self.gate_freeze_through_step
        self._assert_runtime_schedule()

    def _assert_runtime_schedule(self) -> None:
        base_lr = float(self.optimizer.param_groups[0]["lr"])
        adapter_lr = float(self.optimizer.param_groups[1]["lr"])
        gate_lr = float(self.optimizer.param_groups[self.o2_gate_group_index]["lr"])
        if not math.isclose(
            adapter_lr,
            base_lr * self.adapter_lr_scale,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                "Dynamic Stage 2 adapter LR lost its 0.1x coupling: "
                f"base={base_lr}, adapter={adapter_lr}"
            )
        if self.global_step <= self.freeze_through_step and self._adapter_optimizer_state_count() != 0:
            raise RuntimeError("response adapters acquired Adam state during frozen Stage 2 steps")
        if not math.isclose(
            gate_lr,
            base_lr * self.gate_lr_scale,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                "Dynamic Stage 2 O2-gate LR lost its 1.0x coupling: "
                f"base={base_lr}, gate={gate_lr}"
            )
        if self.global_step <= self.gate_freeze_through_step and self._gate_optimizer_state_count() != 0:
            raise RuntimeError("O2 layer_readout acquired Adam state during frozen Stage 2 steps")

    def save_checkpoint(self):
        result = super().save_checkpoint()
        if self.global_step == self.freeze_through_step and not self._freeze_boundary_checkpoint_recorded:
            model = self._dynamic_model()
            if model.response_adapters_trainable:
                raise RuntimeError("response adapters were trainable at the frozen 5K checkpoint")
            self._assert_runtime_schedule()
            model.schedule_response_adapter_unfreeze()
            self._freeze_boundary_checkpoint_recorded = True
            if self.accelerator.is_main_process:
                transition = {
                    "freeze_through_step": self.freeze_through_step,
                    "first_adapter_update_step": self.first_adapter_update_step,
                    "adapter_lr_scale": self.adapter_lr_scale,
                    "adapter_optimizer_state_count_at_step_5000": self._adapter_optimizer_state_count(),
                    "status": "adapter_step_005000_checkpoint_saved_pending_unfreeze",
                }
                Path(self.output_dir, "dynamic_response_transition.json").write_text(
                    json.dumps(transition, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        if self.global_step == self.gate_freeze_through_step and not self._gate_boundary_checkpoint_recorded:
            model = self._dynamic_model()
            if model.o2_gate_trainable:
                raise RuntimeError("O2 layer_readout was trainable at its frozen boundary checkpoint")
            self._assert_runtime_schedule()
            model.schedule_o2_gate_unfreeze()
            self._gate_boundary_checkpoint_recorded = True
            if self.accelerator.is_main_process:
                transition = {
                    "gate_freeze_through_step": self.gate_freeze_through_step,
                    "first_gate_update_step": self.first_gate_update_step,
                    "gate_lr_scale": self.gate_lr_scale,
                    "gate_optimizer_state_count_at_boundary": self._gate_optimizer_state_count(),
                    "status": "o2_gate_boundary_checkpoint_saved_pending_unfreeze",
                }
                Path(self.output_dir, "dynamic_o2_gate_transition.json").write_text(
                    json.dumps(transition, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        return result

    def _maybe_save_train_visualization(self, sample):
        """Force a registered gate-boundary audit checkpoint when one exists."""

        result = super()._maybe_save_train_visualization(sample)
        if (
            self.global_step == self.gate_freeze_through_step
            and not self._gate_boundary_checkpoint_recorded
        ):
            self._assert_runtime_schedule()
            checkpoint = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[dynamic_o2_gate_boundary] step=%d weights=%s state=%s",
                    self.global_step,
                    checkpoint["weights_path"],
                    checkpoint["state_path"],
                )
        return result
