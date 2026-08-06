"""Copied Side-Model3-Adapter v1 boundary with lightweight imports."""

from .config import SideModel3AdapterConfig, default_config, load_config
from .contracts import ContractError, validate_contract

__all__ = [
    "ContractError",
    "SideModel3AdapterConfig",
    "default_config",
    "load_config",
    "validate_contract",
]
