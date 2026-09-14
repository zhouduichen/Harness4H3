"""Remote NVIDIA power sampling for benchmark energy estimates."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple

from .ssh import SSHClient


def integrate_power_samples(samples: Sequence[Tuple[float, float]]) -> float:
    """Integrate watts over seconds with left rectangles."""

    total = 0.0
    previous: Optional[Tuple[float, float]] = None
    for timestamp, power in samples:
        timestamp = float(timestamp)
        power = float(power)
        if not math.isfinite(timestamp) or not math.isfinite(power) or power < 0:
            continue
        if previous is not None and timestamp >= previous[0]:
            total += (timestamp - previous[0]) * previous[1]
        previous = (timestamp, power)
    return float(total)


class RemotePowerSampler:
    QUERY = ("nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits")

    def __init__(self, client: SSHClient, interval_s: float = 1.0):
        if float(interval_s) <= 0:
            raise ValueError("power sampling interval must be positive")
        self.client = client
        self.interval_s = float(interval_s)
        self.samples: List[Tuple[float, float]] = []
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _parse(stdout: Any) -> Optional[float]:
        values = []
        for line in str(stdout).splitlines():
            line = line.strip()
            if not line:
                continue
            token = line.split(",", 1)[0].strip().split()[0] if line.split(",", 1)[0].strip() else ""
            try:
                value = float(token)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value) or value < 0:
                return None
            values.append(value)
        return float(sum(values)) if values else None

    def _sample(self) -> None:
        try:
            # The configured SSH host may spend several seconds on its login
            # path. Keep the command timeout independent of the sampling
            # interval so a slow handshake does not silently erase energy.
            result = self.client.run(self.QUERY, timeout_s=max(30.0, self.interval_s * 4.0))
            power = self._parse(getattr(result, "stdout", ""))
            if power is None:
                raise ValueError("nvidia-smi returned no valid power values")
            with self._lock:
                self.samples.append((time.monotonic(), power))
        except Exception as exc:  # sampling must never turn a benchmark into fake zero energy
            with self._lock:
                self.errors.append(str(exc))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("power sampler is already running")
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="harness4h3-remote-power", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s + 1.0))
        self._sample()

    def summary(self):
        with self._lock:
            samples = tuple(self.samples)
        if len(samples) < 2:
            return {"energy_j": None, "power_w_peak": max((item[1] for item in samples), default=None), "samples": len(samples)}
        return {
            "energy_j": integrate_power_samples(samples),
            "power_w_peak": max(item[1] for item in samples),
            "samples": len(samples),
        }
