from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


VENDORED_PACKAGE = "model3.third_party.light_wam.src.lightwam"


def _rewrite_legacy_string(value: str) -> str:
    rewritten = value
    if rewritten == "fastwam":
        rewritten = "lightwam"
    if rewritten == "lightwam.runtime.create_fastwam":
        rewritten = f"{VENDORED_PACKAGE}.runtime.create_lightwam"
    elif rewritten.startswith("lightwam."):
        rewritten = VENDORED_PACKAGE + rewritten[len("lightwam") :]
    if rewritten.startswith("fastwam."):
        rewritten = VENDORED_PACKAGE + "." + rewritten[len("fastwam.") :]
    if rewritten.startswith("experiments.robotwin.fastwam_policy."):
        rewritten = "experiments.robotwin.lightwam_policy." + rewritten[
            len("experiments.robotwin.fastwam_policy.") :
        ]
    return rewritten


def _rewrite_legacy_object(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_legacy_object(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_rewrite_legacy_object(item) for item in value]
    if isinstance(value, str):
        return _rewrite_legacy_string(value)
    return value


def rewrite_legacy_fastwam_config(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig, ListConfig)):
        container = OmegaConf.to_container(cfg, resolve=False)
        return OmegaConf.create(_rewrite_legacy_object(container))
    return _rewrite_legacy_object(cfg)


def load_compatible_omegaconf(path: str | Path) -> DictConfig | ListConfig:
    cfg = OmegaConf.load(str(path))
    return rewrite_legacy_fastwam_config(cfg)
