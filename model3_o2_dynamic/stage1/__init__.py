"""Offline action-response warmup for the Dynamic O2 treatment.

Nothing in this package is imported by the deployed policy.  It prepares and
trains the temporary $Q_l$ heads, then emits an adapter-only artifact consumed
by Stage 2.
"""

from .cache import load_response_cache, validate_response_cache
from .contracts import Stage1TrainConfig
from .train import train_stage1

__all__ = [
    "Stage1TrainConfig",
    "load_response_cache",
    "train_stage1",
    "validate_response_cache",
]
