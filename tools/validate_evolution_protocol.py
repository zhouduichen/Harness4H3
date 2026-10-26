#!/usr/bin/env python3
"""Validate and optionally manifest a matched four-arm Wan experiment."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from harness4h3.evolution.protocol import (
    DEFAULT_EXPECTED_CASES,
    DEFAULT_LATENCY_REPEATS,
    DEFAULT_ROUNDS,
    ProtocolValidationError,
    validate_protocol,
    write_manifests,
)


def _parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use ARM=CONFIG")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must use non-empty ARM=CONFIG")
    return name.strip(), path.strip()


def _write_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".protocol-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run, help="ARM=CONFIG; repeat four times")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--expected-cases", type=int, default=DEFAULT_EXPECTED_CASES)
    parser.add_argument("--latency-repeats", type=int, default=DEFAULT_LATENCY_REPEATS)
    parser.add_argument("--source-revision")
    parser.add_argument("--allow-missing-assets", action="store_true", help="only for remote preflight; records unresolved assets")
    parser.add_argument("--write-manifests", action="store_true")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--json", action="store_true", help="print the complete report as JSON")
    args = parser.parse_args(argv)
    specs: dict[str, str] = {}
    for name, path in args.run:
        if name in specs:
            parser.error("duplicate arm: %s" % name)
        specs[name] = path
    try:
        report = validate_protocol(
            specs,
            rounds=args.rounds,
            expected_cases=args.expected_cases,
            latency_repeats=args.latency_repeats,
            source_revision=args.source_revision,
            allow_missing_assets=args.allow_missing_assets,
        )
        if args.write_manifests:
            write_manifests(report)
        if args.output:
            _write_json(args.output, report)
    except (ProtocolValidationError, OSError) as exc:
        print("protocol_invalid: %s" % exc)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("protocol_valid fingerprint=%s" % report["protocol_fingerprint"])
        for name in report["required_arms"]:
            print("%s output_root=%s" % (name, report["arms"][name]["output_root"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
