"""Model-specific adapters for training methods."""

from .base import DenoisingModelAdapter
from .real_h3 import RealMiniMaxH3Adapter
from .wan import WanTrainingAdapter

__all__ = ["DenoisingModelAdapter", "RealMiniMaxH3Adapter", "WanTrainingAdapter"]
