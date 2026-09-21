#!/usr/bin/env python3
"""Resume the isolated real-H3 campaign for a bounded overnight controller run."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from harness4h3.controller.provider import RuleBasedMockController, build_controller_from_config
from harness4h3.memory.observation import ControllerEventStore
from harness4h3.remote.config import load_remote_campaign_config
from harness4h3.remote.ssh import LocalCommandClient
from research.experiments.remote_h3_closed_loop import build_campaign_from_config


def _parse_controller_fallback_ports(raw):
    """Parse an explicit comma-separated list of loopback fallback ports."""

    if raw is None or not str(raw).strip():
        return ()
    ports = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            port = int(token)
        except ValueError as exc:
            raise ValueError("Controller fallback port must be an integer: %s" % token) from exc
        if not 1 <= port <= 65535:
            raise ValueError("Controller fallback port must be in [1, 65535]: %s" % port)
        if port not in ports:
            ports.append(port)
    return tuple(ports)


@contextmanager
def _single_flight_lock(path: Path) -> Iterator[None]:
    """Prevent two autonomous processes from mutating one campaign root."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another overnight Controller run already owns %s" % path) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}, sort_keys=True) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_supervised(args: argparse.Namespace, config: Path, output: Path):
    """Keep one campaign lease alive across recoverable top-level crashes.

    The campaign loop handles expected remote/evaluator failures itself. This
    outer boundary is for process-level faults such as an import error,
    unexpected library exception, or a transient launcher failure. Rebuilding
    the campaign object is safe because all lineage, experience, and pending
    state are persisted under ``output`` and the single-flight lock remains
    held for the whole supervisor lifetime.
    """

    restart_count = 0
    while True:
        campaign = None
        try:
            controller = (
                RuleBasedMockController()
                if args.controller == "rulebased"
                else build_controller_from_config(
                    REPOSITORY_ROOT / "configs/controller.yaml",
                    provider_name="vllm",
                    **(
                        {"remote_port": int(args.controller_remote_port)}
                        if getattr(args, "controller_remote_port", None) is not None
                        else {}
                    ),
                )
            )
            fallback_ports = _parse_controller_fallback_ports(
                getattr(args, "controller_fallback_ports", None)
            )
            if fallback_ports:
                controller.fallback_remote_ports = fallback_ports
            if getattr(args, "controller_remote_port", None) is not None:
                controller.preferred_remote_port = int(args.controller_remote_port)
            ssh = LocalCommandClient(load_remote_campaign_config(config).remote) if args.local_resources else None
            campaign = build_campaign_from_config(config, controller=controller, output_root=output, ssh=ssh)
            return campaign.run_loop(
                resume=True,
                max_iterations=args.max_iterations,
                split="heldout",
                resource_poll_interval_s=args.resource_poll_interval_s,
                stop_file=getattr(args, "stop_file", None),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            restart_count += 1
            delay_s = min(300.0, max(1.0, float(args.restart_backoff_s)) * restart_count)
            payload = {
                "restart_count": restart_count,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "retry_after_s": delay_s,
            }
            try:
                if campaign is not None:
                    campaign.events.append("launcher_exception", payload)
                else:
                    ControllerEventStore(output / "controller-events.jsonl").append("launcher_exception", payload)
            except Exception:
                # The original exception is more useful than a secondary
                # failure while recording the supervisor audit event.
                pass
            print(
                "overnight Controller crashed (%s); restart %d after %.1fs: %s"
                % (type(exc).__name__, restart_count, delay_s, str(exc)[:500]),
                file=sys.stderr,
                flush=True,
            )
            if restart_count >= args.max_restarts:
                raise
            time.sleep(delay_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/remote-l40-h3-rsi-overnight.yaml")
    parser.add_argument("--output", default="var/remote-h3-controller-20260914")
    parser.add_argument("--controller", choices=("vllm", "rulebased"), default="vllm")
    parser.add_argument(
        "--controller-remote-port",
        type=int,
        default=None,
        help="explicit loopback vLLM port override; default is configs/controller.yaml",
    )
    parser.add_argument(
        "--controller-fallback-ports",
        default=None,
        help="comma-separated explicit loopback vLLM fallback ports",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=512,
        help="bounded Controller -> worker -> benchmark cycles (default: 512; use a smaller value for a smoke run)",
    )
    parser.add_argument("--resource-poll-interval-s", type=float, default=60.0)
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="request a clean stop at the next completed iteration boundary",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=16,
        help="maximum process-level restarts before failing the detached run (default: 16)",
    )
    parser.add_argument(
        "--restart-backoff-s",
        type=float,
        default=30.0,
        help="linear backoff between process-level restarts (default: 30 seconds)",
    )
    parser.add_argument(
        "--local-resources",
        "--on-server",
        action="store_true",
        help="run trusted commands and loopback services directly on this server (no nested SSH)",
    )
    args = parser.parse_args()
    if args.max_iterations <= 0:
        raise SystemExit("--max-iterations must be positive")
    if args.max_restarts <= 0:
        raise SystemExit("--max-restarts must be positive")
    if args.restart_backoff_s < 0:
        raise SystemExit("--restart-backoff-s must be non-negative")
    config = Path(args.config).resolve()
    output = Path(args.output).resolve()
    try:
        with _single_flight_lock(output / ".overnight-controller.lock"):
            result = _run_supervised(args, config, output)
    except RuntimeError as exc:
        message = str(exc)
        label = "refused" if "already owns" in message else "failed"
        print("overnight Controller run %s: %s" % (label, message), file=sys.stderr)
        return 2
    payload = {
        "status": result.status,
        "current_model_id": result.current_model_id,
        "report": result.report,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "overnight-result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result.status, "current_model_id": result.current_model_id, "loop": result.report.get("loop")}, ensure_ascii=False))
    return 0 if result.status not in {"controller_unavailable", "remote_unavailable", "resources_unavailable"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
