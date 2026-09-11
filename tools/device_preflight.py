from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import yaml


REQUIRED_MAPPINGS = ("platform", "hardware", "paths", "services", "limits", "capabilities")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, passed: bool, detail: str) -> Dict[str, str]:
    return {"name": name, "status": "passed" if passed else "failed", "detail": detail}


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


def preflight(
    profile_path: Path,
    operator: Optional[str],
    check_services: bool = True,
    timeout_s: float = 3.0,
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

    for name, entry in raw["paths"].items():
        if not isinstance(entry, Mapping) or entry.get("required") is not True:
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

    status = "blocked" if any(item["status"] == "failed" for item in checks) else "ready"
    return {"status": status, "profile_id": profile_id, "operator": operator, "checks": checks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one Harness4H3 device profile without running an experiment")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--operator")
    parser.add_argument("--skip-services", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = preflight(args.profile, args.operator, not args.skip_services, args.timeout_s)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print("%s: %s" % (result.get("profile_id") or args.profile, result["status"]))
        for item in result["checks"]:
            print("[%s] %s: %s" % (item["status"], item["name"], item["detail"]))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
