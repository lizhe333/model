"""Legacy Fast-WAM compatibility aliases.

The canonical implementation now lives in ``lightwam.py``. This module keeps
legacy import paths working without changing training or inference behavior.
"""

from .lightwam import DisabledActionExpert, LightWAM


class FastWAM(LightWAM):
    """Backward-compatible Fast-WAM alias for the canonical Light-WAM model."""

    pass
