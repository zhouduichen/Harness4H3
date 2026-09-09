"""Real H3/ComfyUI benchmark orchestration and hardware sampling."""

from .h3 import BenchmarkSummary, H3BenchmarkRunner
from .validation import M5ValidationResult, M5ValidationRunner, MetricStatistics, aggregate_summaries

__all__ = [
    "BenchmarkSummary",
    "H3BenchmarkRunner",
    "M5ValidationResult",
    "M5ValidationRunner",
    "MetricStatistics",
    "aggregate_summaries",
]
