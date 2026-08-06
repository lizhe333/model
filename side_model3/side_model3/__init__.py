"""Side-Model3 v1 method boundary with lightweight preflight imports."""

from .config import SideModel3Config, default_config, load_config
from .contracts import ContractError, validate_contract

__all__ = [
    "ContractError",
    "SideModel3Config",
    "default_config",
    "load_config",
    "validate_contract",
]
