"""Hydra entrypoint that preserves Light-WAM data/runtime but uses Dynamic trainer."""

from __future__ import annotations

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from model3.third_party.light_wam.src.lightwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    build_datasets,
)
from model3.third_party.light_wam.src.lightwam.utils import misc
from model3.third_party.light_wam.src.lightwam.utils.config_resolvers import register_default_resolvers
from model3.third_party.light_wam.src.lightwam.utils.logging_config import setup_logging

from model3_o2_dynamic.trainer import DynamicWan22Trainer


register_default_resolvers()


@hydra.main(
    config_path="../model3/third_party/light_wam/configs",
    config_name="train",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    setup_logging()
    misc.register_work_dir(cfg.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    # Stage 1 carriers seal the newly initialized O2 gate under seed 42.  It
    # must be recreated before Hydra instantiates the Dynamic model so the
    # adapter export's O2 identity is meaningful at Stage 2.
    initialization_seed = int(cfg.seed)
    if initialization_seed != 42:
        raise ValueError("Dynamic O2 Stage 2 requires seed=42 for exact O2 initialization")
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    OmegaConf.save(cfg, Path(cfg.output_dir) / "config.yaml", resolve=True)
    model_dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=_resolve_train_device())
    train_dataset, validation_dataset = build_datasets(cfg.data)
    trainer = DynamicWan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
