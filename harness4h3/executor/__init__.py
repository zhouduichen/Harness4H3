"""Restricted process execution for registered model operators."""

from .local import LocalProcessExecutor
from .result import ProcessResult

__all__ = ["LocalProcessExecutor", "ProcessResult"]
