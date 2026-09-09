from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence

from .result import ProcessResult


class ProcessExecutor(Protocol):
    def execute(
        self,
        argv: Sequence[str],
        experiment_dir: Path,
        env: Optional[Mapping[str, str]] = None,
        timeout_s: Optional[float] = None,
    ) -> ProcessResult:
        ...
