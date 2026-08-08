"""Hydra training entrypoint for Side-Model3-Adapter cached Object training."""

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
from model3.third_party.light_wam.src.lightwam.utils.config_resolvers import (
    register_default_resolvers,
)
from model3.third_party.light_wam.src.lightwam.utils.logging_config import setup_logging

from side_model3_adapter_v2.trainer import SideModel3AdapterV2Trainer


register_default_resolvers()


@hydra.main(
    config_path="../model3/third_party/light_wam/configs",
    config_name="train",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    setup_logging()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))

    initialization_seed = int(cfg.seed)
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)

    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    model_dtype = _mixed_precision_to_model_dtype(
        _normalize_mixed_precision(cfg.mixed_precision)
    )
    model = instantiate(
        cfg.model,
        model_dtype=model_dtype,
        device=_resolve_train_device(),
    )
    train_dataset, validation_dataset = build_datasets(cfg.data)
    trainer = SideModel3AdapterV2Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
