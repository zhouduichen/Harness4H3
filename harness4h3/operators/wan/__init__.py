"""Structural compression operators for Wan-shaped DiT models."""

from .compression import BlockPruningOperator, FFNChannelPruningOperator, WanCompressionRegistry, build_wan_compression_registry

__all__ = ["BlockPruningOperator", "FFNChannelPruningOperator", "WanCompressionRegistry", "build_wan_compression_registry"]

