from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ProcessResult:
    status: str
    argv: Tuple[str, ...]
    cwd: str
    pid: Optional[int]
    returncode: Optional[int]
    timed_out: bool
    terminated: bool
    wall_time_s: float
    stdout_path: str
    stderr_path: str
    failure_type: Optional[str] = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "success" and self.returncode == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
