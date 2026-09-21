"""Remote NVIDIA power sampling for benchmark energy estimates."""

from __future__ import annotations

import math
import subprocess
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    GPU_QUERY = (
        "nvidia-smi",
        "--query-gpu=index,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    )

    def __init__(self, client: SSHClient, interval_s: float = 1.0):
        if float(interval_s) <= 0:
            raise ValueError("power sampling interval must be positive")
        self.client = client
        self.interval_s = float(interval_s)
        self.samples: List[Tuple[float, float]] = []
        # (monotonic timestamp, GPU index, watts, utilization percent). Keep
        # this bounded: the aggregate samples remain the canonical energy
        # trace, while these rows provide enough evidence to diagnose an idle
        # or imbalanced card.
        self.gpu_samples: List[Tuple[float, int, float, float]] = []
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None

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

    def _sample_local(self) -> None:
        """Sample per-GPU power/utilization through a local trusted client."""

        try:
            result = self.client.run(self.GPU_QUERY, timeout_s=max(30.0, self.interval_s * 4.0))
            timestamp = time.monotonic()
            total = 0.0
            count = 0
            for line in str(getattr(result, "stdout", "")).splitlines():
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 3:
                    continue
                try:
                    index = int(fields[0])
                    power = float(fields[1])
                    utilization = float(fields[2])
                except (TypeError, ValueError):
                    continue
                if not 0 <= index or not math.isfinite(power) or power < 0:
                    continue
                self._append_gpu_sample(timestamp, index, power, utilization)
                total += power
                count += 1
            if count == 0:
                raise ValueError("nvidia-smi returned no valid per-GPU power values")
            with self._lock:
                self.samples.append((timestamp, total))
        except Exception as exc:
            with self._lock:
                self.errors.append(str(exc))

    def _append_gpu_sample(self, timestamp: float, index: int, power: float, utilization: float) -> None:
        if (
            not math.isfinite(timestamp)
            or not math.isfinite(power)
            or not math.isfinite(utilization)
            or power < 0
            or utilization < 0
        ):
            return
        with self._lock:
            self.gpu_samples.append((timestamp, index, power, utilization))
            if len(self.gpu_samples) > 4096:
                del self.gpu_samples[: len(self.gpu_samples) - 4096]

    def _run(self) -> None:
        local = bool(getattr(self.client, "is_local", False))
        while not self._stop.wait(self.interval_s):
            self._sample_local() if local else self._sample()

    def _run_stream(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in iter(process.stdout.readline, ""):
                if self._stop.is_set():
                    break
                raw = str(line).strip()
                if not raw or "," not in raw:
                    continue
                fields = raw.split(",")
                if len(fields) == 5 and fields[0].strip() == "gpu":
                    try:
                        timestamp = float(fields[1])
                        index = int(fields[2])
                        power = float(fields[3])
                        utilization = float(fields[4])
                    except (TypeError, ValueError):
                        continue
                    self._append_gpu_sample(timestamp, index, power, utilization)
                    continue
                timestamp_raw, power_raw = raw.split(",", 1)
                try:
                    timestamp, power = float(timestamp_raw), float(power_raw)
                except ValueError:
                    continue
                if math.isfinite(timestamp) and math.isfinite(power) and power >= 0:
                    with self._lock:
                        self.samples.append((timestamp, power))
            returncode = process.wait()
            if returncode != 0 and not self._stop.is_set():
                stderr = process.stderr.read().strip() if process.stderr is not None else ""
                with self._lock:
                    self.errors.append("remote power stream exited with code %d%s" % (returncode, (": " + stderr) if stderr else ""))
        except (OSError, ValueError) as exc:
            with self._lock:
                self.errors.append(str(exc))

    def _lane_assignments(self) -> Tuple[Dict[int, str], Tuple[str, ...]]:
        """Read the short-lived campaign markers that explain each sample.

        Power/utilization is useful only when the next Controller turn can
        tell whether a low card was assigned to training, evaluation, or the
        Controller itself.  Markers are optional so older transports and
        offline tests keep working; malformed or unreadable metadata is
        reported as a bounded diagnostic rather than guessed.
        """

        config = getattr(self.client, "config", None)
        campaign_root = getattr(config, "resolved_campaign_root", None)
        reader = getattr(self.client, "read_json", None)
        if not campaign_root or not callable(reader):
            return {}, ()
        root = str(campaign_root).rstrip("/")
        markers = [
            ("controller", root + "/.controller-gpu-lease.json", "allocated_gpus"),
            ("worker", root + "/.h3-worker-gpu-lease.json", "allocated_gpus"),
            ("comfyui", root + "/.comfyui-gpu-lease.json", "gpu_index"),
        ]
        # The primary ComfyUI worker keeps the historical marker name.  Extra
        # workers use a suffix keyed by their GPU index; include all four
        # supported campaign GPUs so the Controller can distinguish a live
        # ComfyUI lane from an actually idle card in its power report.
        markers.extend(
            ("comfyui", root + "/.comfyui-gpu-lease-%d.json" % index, "gpu_index")
            for index in range(1, 4)
        )
        owners: Dict[int, List[str]] = {}
        diagnostics: List[str] = []
        for lane, path, field in markers:
            try:
                value = reader(path)
            except Exception:
                continue
            if not isinstance(value, Mapping):
                diagnostics.append("%s lease is not a mapping" % lane)
                continue
            raw = value.get(field)
            indices = raw if field == "allocated_gpus" else [raw]
            if not isinstance(indices, list):
                diagnostics.append("%s lease has invalid %s" % (lane, field))
                continue
            for index in indices:
                if isinstance(index, bool):
                    diagnostics.append("%s lease has invalid GPU index" % lane)
                    continue
                try:
                    normalized = int(index)
                except (TypeError, ValueError):
                    diagnostics.append("%s lease has invalid GPU index" % lane)
                    continue
                if normalized < 0:
                    diagnostics.append("%s lease has invalid GPU index" % lane)
                    continue
                owners.setdefault(normalized, []).append(lane)
        assignments = {
            index: (labels[0] if len(set(labels)) == 1 else "conflict:" + "+".join(sorted(set(labels))))
            for index, labels in owners.items()
        }
        return assignments, tuple(diagnostics[-4:])

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("power sampler is already running")
        self._stop.clear()
        if bool(getattr(self.client, "is_local", False)):
            # ``run_overnight_controller.py --on-server`` uses a
            # LocalCommandClient.  Opening SSH back to the same host can fail
            # (and made live benchmark power samples silently become zero),
            # so poll through the already-local trusted transport instead.
            self._process = None
            self._thread = threading.Thread(target=self._run, name="harness4h3-local-power", daemon=True)
            self._thread.start()
            return
        remote_loop = (
            "while :; do "
            "ts=$(date +%%s.%%N); "
            "rows=$(nvidia-smi --query-gpu=index,power.draw,utilization.gpu --format=csv,noheader,nounits); "
            "watts=$(printf '%%s\\n' \"$rows\" | awk -F, '{sum += $2} END {print sum+0}'); "
            "printf '%%s,%%s\\n' \"$ts\" \"$watts\"; "
            "printf '%%s\\n' \"$rows\" | awk -F, -v ts=\"$ts\" '{gsub(/^[[:space:]]+|[[:space:]]+$/,\"\",$1); gsub(/^[[:space:]]+|[[:space:]]+$/,\"\",$2); gsub(/^[[:space:]]+|[[:space:]]+$/,\"\",$3); if ($1 != \"\" && $2 != \"\" && $3 != \"\") printf \"gpu,%%s,%%s,%%s,%%s\\n\", ts,$1,$2,$3}'; "
            "sleep %s; "
            "done"
        ) % self.interval_s
        try:
            ssh_argv = ["ssh", "-T"]
            ssh_port = int(getattr(self.client.config, "ssh_port", 22))
            if ssh_port != 22:
                ssh_argv.extend(["-p", str(ssh_port)])
            ssh_argv.extend([self.client.config.host, remote_loop])
            self._process = subprocess.Popen(
                ssh_argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._thread = threading.Thread(target=self._run_stream, name="harness4h3-remote-power", daemon=True)
        except OSError as exc:
            with self._lock:
                self.errors.append(str(exc))
            self._process = None
            self._sample()
            self._thread = threading.Thread(target=self._run, name="harness4h3-remote-power", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s + 1.0))
        if process is not None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        else:
            self._sample_local() if bool(getattr(self.client, "is_local", False)) else self._sample()

    def summary(self):
        with self._lock:
            samples = tuple(self.samples)
            gpu_samples = tuple(self.gpu_samples)
        lane_assignments, lane_diagnostics = self._lane_assignments()
        per_gpu = {}
        for _timestamp, index, power, utilization in gpu_samples:
            entry = per_gpu.setdefault(
                str(index),
                {
                    "power_w_peak": 0.0,
                    "power_w_sum": 0.0,
                    "utilization_gpu_pct_sum": 0.0,
                    "utilization_gpu_pct_peak": 0.0,
                    "samples": 0,
                },
            )
            entry["power_w_peak"] = max(float(entry["power_w_peak"]), power)
            entry["power_w_sum"] += power
            entry["utilization_gpu_pct_sum"] += utilization
            entry["utilization_gpu_pct_peak"] = max(float(entry["utilization_gpu_pct_peak"]), utilization)
            entry["samples"] += 1
            entry["lane"] = lane_assignments.get(index, "unknown")
        for entry in per_gpu.values():
            entry["power_w_avg"] = entry.pop("power_w_sum") / max(1, entry["samples"])
            entry["utilization_gpu_pct_avg"] = entry.pop("utilization_gpu_pct_sum") / max(1, entry["samples"])
            # Keep the descriptive names used by the evidence schema in
            # addition to the original fields so old consumers remain
            # compatible.  Missing lane metadata is explicit; it is never
            # inferred as an idle or fully utilized lane.
            entry["power_mean_w"] = entry["power_w_avg"]
            entry["utilization_mean"] = entry["utilization_gpu_pct_avg"]
            entry["sample_count"] = entry["samples"]
            entry["lane_unknown"] = entry.get("lane", "unknown") == "unknown"
        gpu_summary = {"per_gpu": per_gpu, "sample_rows": len(gpu_samples)}
        if lane_diagnostics:
            gpu_summary["lane_metadata_diagnostics"] = list(lane_diagnostics)
        average_power = (sum(item[1] for item in samples) / len(samples)) if samples else None
        if len(samples) < 2:
            return {
                "energy_j": None,
                "power_w_peak": max((item[1] for item in samples), default=None),
                "power_w_avg": average_power,
                "samples": len(samples),
                **gpu_summary,
            }
        return {
            "energy_j": integrate_power_samples(samples),
            "power_w_peak": max(item[1] for item in samples),
            "power_w_avg": average_power,
            "samples": len(samples),
            **gpu_summary,
        }
