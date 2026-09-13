from __future__ import annotations

import argparse
import json
from pathlib import Path
from subprocess import PIPE, run
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import yaml


REQUIRED_MAPPINGS = ("platform", "hardware", "paths", "services", "limits", "capabilities")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MEMINFO_PATH = Path("/proc/meminfo")
NVIDIA_SMI_ARGV = (
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader,nounits",
)
HARDWARE_PROBE_TRIGGER_KEYS = (
    "gpu_name_contains",
    "min_vram_gib_per_gpu",
    "min_system_ram_gib",
    "distributed_backend",
)


def _check(name: str, passed: bool, detail: str) -> Dict[str, str]:
    return {"name": name, "status": "passed" if passed else "failed", "detail": detail}


def _skipped(name: str, detail: str) -> Dict[str, str]:
    return {"name": name, "status": "skipped", "detail": detail}


def _load_profile(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("device profile must be a YAML mapping")
    return raw


def _schema_errors(raw: Mapping[str, Any]) -> List[str]:
    errors = []
    if raw.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if not str(raw.get("id", "")).strip():
        errors.append("id is required")
    for name in REQUIRED_MAPPINGS:
        if not isinstance(raw.get(name), Mapping):
            errors.append("%s must be a mapping" % name)
    formats = raw.get("formats")
    if not isinstance(formats, list) or not formats or not all(isinstance(item, str) and item for item in formats):
        errors.append("formats must be a non-empty list of strings")
    return errors


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _gpu_observations() -> List[Dict[str, Any]]:
    completed = run(NVIDIA_SMI_ARGV, stdin=PIPE, capture_output=True, text=True, check=False, timeout=10)
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi exited with status %d: %s" % (completed.returncode, completed.stderr.strip()))
    observations = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        try:
            name, memory_mib = line.rsplit(",", 1)
            observations.append({"name": name.strip(), "memory_gib": float(memory_mib.strip()) / 1024.0})
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed nvidia-smi row: %s" % line) from exc
    if not observations:
        raise ValueError("nvidia-smi returned no GPUs")
    return observations


def _system_ram_gib() -> float:
    for line in MEMINFO_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise ValueError("MemTotal is missing from /proc/meminfo")


def _torch_observation() -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    return {
        "version": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }


def _hardware_checks(hardware: Mapping[str, Any], enabled: bool) -> List[Dict[str, str]]:
    if not any(key in hardware for key in HARDWARE_PROBE_TRIGGER_KEYS):
        return []
    names = (
        "hardware.gpu_count",
        "hardware.gpu_name",
        "hardware.gpu_vram",
        "hardware.system_ram",
        "runtime.torch",
    )
    if not enabled:
        return [_skipped(name, "required hardware probe skipped") for name in names]

    checks: List[Dict[str, str]] = []
    expected_count = int(hardware.get("gpu_count", 0))
    expected_name = str(hardware.get("gpu_name_contains", "")).strip()
    minimum_vram = float(hardware.get("min_vram_gib_per_gpu", 0.0))
    try:
        observations = _gpu_observations()
        checks.append(
            _check(
                "hardware.gpu_count",
                len(observations) == expected_count,
                "observed %d; required %d" % (len(observations), expected_count),
            )
        )
        names_ok = bool(expected_name) and all(expected_name in item["name"] for item in observations)
        checks.append(
            _check(
                "hardware.gpu_name",
                names_ok,
                "observed %s; required substring %s"
                % (", ".join(str(item["name"]) for item in observations), expected_name),
            )
        )
        vram_ok = bool(observations) and all(float(item["memory_gib"]) >= minimum_vram for item in observations)
        checks.append(
            _check(
                "hardware.gpu_vram",
                vram_ok,
                "observed GiB %s; required at least %.3f each"
                % (", ".join("%.3f" % float(item["memory_gib"]) for item in observations), minimum_vram),
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        detail = "nvidia-smi unavailable or invalid: %s" % exc
        checks.extend(_check(name, False, detail) for name in names[:3])

    minimum_ram = float(hardware.get("min_system_ram_gib", 0.0))
    try:
        observed_ram = _system_ram_gib()
        checks.append(
            _check(
                "hardware.system_ram",
                observed_ram >= minimum_ram,
                "observed %.3f GiB; required at least %.3f GiB" % (observed_ram, minimum_ram),
            )
        )
    except (OSError, TypeError, ValueError) as exc:
        checks.append(_check("hardware.system_ram", False, "unable to read system RAM: %s" % exc))

    try:
        torch = _torch_observation()
        torch_ok = bool(torch.get("available")) and int(torch.get("device_count", 0)) == expected_count
        checks.append(
            _check(
                "runtime.torch",
                torch_ok,
                "version=%s cuda=%s available=%s visible_devices=%s required=%d"
                % (torch.get("version"), torch.get("cuda"), torch.get("available"), torch.get("device_count"), expected_count),
            )
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        checks.append(_check("runtime.torch", False, str(exc)))
    return checks


def preflight(
    profile_path: Path,
    operator: Optional[str],
    check_services: bool = True,
    timeout_s: float = 3.0,
    check_hardware: bool = True,
) -> Dict[str, Any]:
    try:
        raw = _load_profile(Path(profile_path))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {
            "status": "blocked",
            "profile_id": None,
            "operator": operator,
            "checks": [_check("profile.schema", False, str(exc))],
        }

    profile_id = str(raw.get("id", "")).strip() or None
    schema_errors = _schema_errors(raw)
    checks = [_check("profile.schema", not schema_errors, "; ".join(schema_errors) or "schema valid")]
    if schema_errors:
        return {"status": "blocked", "profile_id": profile_id, "operator": operator, "checks": checks}

    capabilities = raw["capabilities"]
    if operator:
        capability = capabilities.get(operator)
        if not isinstance(capability, Mapping):
            checks.append(_check("capability.%s" % operator, False, "operator is not declared"))
        else:
            enabled = capability.get("enabled") is True
            reason = str(capability.get("reason", "")).strip() or ("enabled" if enabled else "disabled")
            checks.append(_check("capability.%s" % operator, enabled, reason))

    checks.extend(_hardware_checks(raw["hardware"], check_hardware))

    for name, entry in raw["paths"].items():
        if not isinstance(entry, Mapping) or entry.get("required") is not True:
            continue
        required_for = entry.get("required_for", [])
        if operator and required_for and operator not in required_for:
            continue
        value = str(entry.get("path", "")).strip()
        resolved = _resolve_path(value) if value else None
        exists = bool(resolved and resolved.exists())
        checks.append(_check("path.%s" % name, exists, str(resolved) if resolved else "path is required"))

    if check_services and operator:
        for name, entry in raw["services"].items():
            if not isinstance(entry, Mapping) or operator not in entry.get("required_for", []):
                continue
            base_url = str(entry.get("url", "")).strip()
            health_path = str(entry.get("health_path", "")).strip()
            url = urljoin(base_url.rstrip("/") + "/", health_path.lstrip("/"))
            try:
                request = Request(url, headers={"User-Agent": "Harness4H3-preflight/1.0"})
                with urlopen(request, timeout=timeout_s):
                    pass
                checks.append(_check("service.%s" % name, True, url))
            except Exception as exc:
                checks.append(_check("service.%s" % name, False, "%s: %s" % (url, exc)))

    status = "blocked" if any(item["status"] in {"failed", "skipped"} for item in checks) else "ready"
    return {"status": status, "profile_id": profile_id, "operator": operator, "checks": checks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one Harness4H3 device profile without running an experiment")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--operator")
    parser.add_argument("--skip-services", action="store_true")
    parser.add_argument("--skip-hardware", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = preflight(
        args.profile,
        args.operator,
        check_services=not args.skip_services,
        timeout_s=args.timeout_s,
        check_hardware=not args.skip_hardware,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print("%s: %s" % (result.get("profile_id") or args.profile, result["status"]))
        for item in result["checks"]:
            print("[%s] %s: %s" % (item["status"], item["name"], item["detail"]))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
