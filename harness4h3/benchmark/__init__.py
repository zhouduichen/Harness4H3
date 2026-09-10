"""Real H3/ComfyUI benchmark orchestration and hardware sampling."""

from .h3 import BenchmarkSummary, H3BenchmarkRunner
from .m6 import M6ValidationResult, M6ValidationRunner
from .runtime_policy import apply_runtime_policy
from .validation import M5ValidationResult, M5ValidationRunner, MetricStatistics, aggregate_summaries

__all__ = [
    "BenchmarkSummary",
    "H3BenchmarkRunner",
    "M6ValidationResult",
    "M6ValidationRunner",
    "apply_runtime_policy",
    "M5ValidationResult",
    "M5ValidationRunner",
    "MetricStatistics",
    "aggregate_summaries",
]
