"""Side-Model3-Adapter-v2 package boundary with lightweight imports."""

from .config import SideModel3AdapterV2Config, default_config, load_config
from .contracts import ContractError, validate_contract

__all__ = [
    "ContractError",
    "SideModel3AdapterV2Config",
    "default_config",
    "load_config",
    "validate_contract",
]
