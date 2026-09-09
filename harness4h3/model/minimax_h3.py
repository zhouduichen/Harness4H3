"""Compatibility imports for the ComfyUI artifact backend.

MiniMax H3 model state lives under :mod:`harness4h3.h3`; ComfyUI is only an
artifact-generation backend in the EvoGen architecture.
"""

from ..backends.comfyui import BackendError, BackendResult, MiniMaxH3Adapter

__all__ = ["BackendError", "BackendResult", "MiniMaxH3Adapter"]
