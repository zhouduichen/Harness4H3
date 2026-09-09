from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .result import ProcessResult


class ExecutorValidationError(ValueError):
    pass


class LocalProcessExecutor:
    """Run a fixed argv in one experiment directory without invoking a shell."""

    DEFAULT_INHERITED_ENV = (
        "PATH",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "PYTHONIOENCODING",
    )

    def __init__(
        self,
        timeout_s: float = 3600.0,
        allowed_env: Sequence[str] = (),
        inherited_env: Sequence[str] = DEFAULT_INHERITED_ENV,
        termination_grace_s: float = 5.0,
    ):
        if timeout_s <= 0 or termination_grace_s < 0:
            raise ExecutorValidationError("executor timeouts must be positive")
        self.timeout_s = float(timeout_s)
        self.allowed_env = frozenset(str(item) for item in allowed_env)
        self.inherited_env = tuple(str(item) for item in inherited_env)
        self.termination_grace_s = float(termination_grace_s)

    def execute(
        self,
        argv: Sequence[str],
        experiment_dir: Path,
        env: Optional[Mapping[str, str]] = None,
        timeout_s: Optional[float] = None,
    ) -> ProcessResult:
        command = self._validate_argv(argv)
        supplied_env = dict(env or {})
        unexpected = sorted(set(supplied_env) - self.allowed_env)
        if unexpected:
            raise ExecutorValidationError("environment variable(s) not allowlisted: %s" % ", ".join(unexpected))
        if any(not isinstance(value, str) for value in supplied_env.values()):
            raise ExecutorValidationError("environment values must be strings")
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0:
            raise ExecutorValidationError("timeout must be positive")

        cwd = Path(experiment_dir).resolve()
        cwd.mkdir(parents=True, exist_ok=True)
        if not cwd.is_dir():
            raise ExecutorValidationError("experiment directory is not a directory")
        (cwd / "artifacts").mkdir(exist_ok=True)
        stdout_path = cwd / "stdout.log"
        stderr_path = cwd / "stderr.log"
        child_env: Dict[str, str] = {name: os.environ[name] for name in self.inherited_env if name in os.environ}
        child_env.update(supplied_env)
        child_env["HARNESS4H3_EXPERIMENT_DIR"] = str(cwd)

        started = time.monotonic()
        process: Optional[subprocess.Popen] = None
        timed_out = False
        terminated = False
        message = ""
        failure_type: Optional[str] = None
        returncode: Optional[int] = None
        popen_options = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=str(cwd),
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    **popen_options,
                )
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminated = True
                    failure_type = "timeout"
                    message = "process exceeded %.3f second timeout" % timeout
                    self._terminate(process)
                    returncode = process.returncode
        except OSError as exc:
            failure_type = "executor_error"
            message = str(exc)
        wall_time = time.monotonic() - started
        if failure_type is None and returncode != 0:
            failure_type = "process_exit"
            message = "process exited with status %s" % returncode
        status = "success" if failure_type is None and returncode == 0 else "failed"
        return ProcessResult(
            status=status,
            argv=command,
            cwd=str(cwd),
            pid=process.pid if process is not None else None,
            returncode=returncode,
            timed_out=timed_out,
            terminated=terminated,
            wall_time_s=wall_time,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            failure_type=failure_type,
            message=message,
        )

    @staticmethod
    def _validate_argv(argv: Sequence[str]) -> Tuple[str, ...]:
        if isinstance(argv, (str, bytes)) or not argv:
            raise ExecutorValidationError("argv must be a non-empty sequence of strings")
        command = tuple(argv)
        if any(not isinstance(item, str) or "\x00" in item for item in command):
            raise ExecutorValidationError("argv must contain only valid strings")
        if not command[0].strip():
            raise ExecutorValidationError("argv executable must be non-empty")
        return command

    def _terminate(self, process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=self.termination_grace_s)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=max(self.termination_grace_s, 0.1))
        except (OSError, subprocess.TimeoutExpired):
            pass
