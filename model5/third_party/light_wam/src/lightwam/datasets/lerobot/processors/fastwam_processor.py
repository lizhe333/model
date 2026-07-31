"""Legacy Fast-WAM processor compatibility aliases.

The canonical processor implementation now lives in ``lightwam_processor.py``.
This module keeps older import paths working unchanged.
"""

from .lightwam_processor import FastWAMProcessor, LightWAMProcessor

__all__ = ["FastWAMProcessor", "LightWAMProcessor"]
