"""Small, auditable SSH transport used by the remote H3 campaign.

The Controller never supplies commands or paths to this module. All executable
commands are fixed here or come from the trusted campaign configuration.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


_MAX_METADATA_BYTES = 8 * 1024 * 1024


class RemoteError(RuntimeError):
    """Base class for expected remote transport failures."""


class RemotePathError(RemoteError):
    pass


class RemoteLinkConflict(RemoteError):
    pass


class RemoteCommandError(RemoteError):
    def __init__(self, argv: Sequence[str], returncode: int, stderr: str = ""):
        self.argv = tuple(str(item) for item in argv)
        self.returncode = int(returncode)
        self.stderr = stderr
        super().__init__("remote command failed (%d): %s%s" % (returncode, " ".join(self.argv), (": " + stderr.strip()) if stderr else ""))


def _normalize_root(value: str, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty absolute path" % name)
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise ValueError("%s must be absolute" % name)
    return PurePosixPath(os.path.normpath(str(path)))


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    harness_root: str
    model_root: str
    comfyui_root: str
    comfyui_port: int = 8188
    python: str = "python3"
    results_root: Optional[str] = None
    deployment_dir: Optional[str] = None
    campaign_root: Optional[str] = None
    training_python: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.host).strip():
            raise ValueError("remote host must not be empty")
        if int(self.comfyui_port) <= 0 or int(self.comfyui_port) > 65535:
            raise ValueError("comfyui_port must be between 1 and 65535")
        for name in ("harness_root", "model_root", "comfyui_root"):
            _normalize_root(str(getattr(self, name)), name)
        for name in ("results_root", "deployment_dir", "campaign_root"):
            value = getattr(self, name)
            if value is not None:
                _normalize_root(str(value), name)
        if self.results_root is not None and not self._under_any_root(self.results_root, include_results=False):
            raise ValueError("results_root must be under model_root or harness_root")
        if self.deployment_dir is not None and not self._under_root(self.deployment_dir, self.comfyui_root):
            raise ValueError("deployment_dir must be under comfyui_root")
        if self.campaign_root is not None and not self._under_root(self.campaign_root, self.harness_root):
            raise ValueError("campaign_root must be under harness_root")

    def _under_root(self, value: str, root: str) -> bool:
        candidate = _normalize_root(str(value), "path")
        base = _normalize_root(str(root), "root")
        return candidate == base or base in candidate.parents

    def _under_any_root(self, value: str, include_results: bool = True) -> bool:
        roots = (self.harness_root, self.model_root, self.comfyui_root)
        if include_results and self.results_root:
            roots += (self.results_root,)
        return any(self._under_root(value, root) for root in roots)

    @property
    def resolved_deployment_dir(self) -> str:
        return self.deployment_dir or str(PurePosixPath(self.comfyui_root) / "models" / "diffusion_models")

    @property
    def resolved_campaign_root(self) -> str:
        return self.campaign_root or str(PurePosixPath(self.harness_root) / "work" / "remote-h3")


def link_action(existing_target: str, requested_target: str) -> str:
    """Return the safe action for a model link or raise on overwrite."""

    if os.path.normpath(existing_target) == os.path.normpath(requested_target):
        return "keep"
    raise RemoteLinkConflict("refusing to replace model link %s with %s" % (existing_target, requested_target))


class SSHClient:
    def __init__(self, config: RemoteConfig, runner: Optional[Callable[..., Any]] = None, command_timeout_s: float = 120.0):
        self.config = config
        self.runner = runner or subprocess.run
        self.command_timeout_s = float(command_timeout_s)

    def _path(self, value: str, *, allow_deployment: bool = False) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RemotePathError("remote path must be a non-empty absolute path")
        candidate = PurePosixPath(os.path.normpath(value))
        if not candidate.is_absolute():
            raise RemotePathError("remote path must be absolute: %s" % value)
        roots = [self.config.harness_root, self.config.model_root, self.config.comfyui_root]
        if self.config.results_root:
            roots.append(self.config.results_root)
        if allow_deployment:
            roots.append(self.config.resolved_deployment_dir)
        if not any(self.config._under_root(str(candidate), root) for root in roots):
            raise RemotePathError("remote path escapes configured roots: %s" % value)
        return str(candidate)

    @staticmethod
    def _command_string(argv: Sequence[str]) -> str:
        return " ".join(shlex.quote(str(item)) for item in argv)

    def run(self, argv: Sequence[str], *, check: bool = True, timeout_s: Optional[float] = None) -> Any:
        if not argv:
            raise ValueError("remote command must not be empty")
        remote_command = self._command_string(argv)
        local_argv = ("ssh", self.config.host, remote_command)
        result = self.runner(
            local_argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.command_timeout_s if timeout_s is None else float(timeout_s),
        )
        if result is None:
            raise RemoteCommandError(local_argv, 1, "runner returned no result")
        if check and int(getattr(result, "returncode", 0)) != 0:
            raise RemoteCommandError(local_argv, int(result.returncode), str(getattr(result, "stderr", "")))
        return result

    def _metadata_text(self, path: str) -> str:
        checked = self._path(path)
        result = self.run(("head", "-c", str(_MAX_METADATA_BYTES + 1), "--", checked))
        text = str(getattr(result, "stdout", ""))
        if len(text.encode("utf-8")) > _MAX_METADATA_BYTES:
            raise RemoteError("remote metadata exceeds %d bytes: %s" % (_MAX_METADATA_BYTES, checked))
        return text

    def read_text(self, path: str) -> str:
        return self._metadata_text(path)

    def read_json(self, path: str) -> Any:
        try:
            return json.loads(self._metadata_text(path))
        except json.JSONDecodeError as exc:
            raise RemoteError("invalid JSON at %s: %s" % (path, exc))

    def write_json(self, path: str, value: Any) -> None:
        checked = self._path(path)
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        if len(encoded) > 2 * 1024 * 1024:
            raise RemoteError("refusing oversized remote JSON write")
        payload = base64.b64encode(encoded).decode("ascii")
        script = (
            "import base64,os,pathlib,tempfile,sys;"
            "p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);"
            "fd,tmp=tempfile.mkstemp(prefix=p.name+'.tmp-',dir=str(p.parent));"
            "os.write(fd,base64.b64decode(sys.argv[2]));os.fsync(fd);os.close(fd);os.replace(tmp,p)"
        )
        self.run((self.config.python, "-c", script, checked, payload))

    def find(self, pattern: str, root: Optional[str] = None) -> List[str]:
        if not isinstance(pattern, str) or not pattern or "/" in pattern:
            raise ValueError("find pattern must be a file name without path separators")
        base = self._path(root or self.config.results_root or self.config.harness_root)
        result = self.run(("find", base, "-type", "f", "-name", pattern, "-print"))
        return [line.strip() for line in str(getattr(result, "stdout", "")).splitlines() if line.strip()]

    def sha256(self, path: str) -> str:
        checked = self._path(path)
        result = self.run(("sha256sum", "--", checked))
        value = str(getattr(result, "stdout", "")).strip().split()
        if not value or len(value[0]) != 64:
            raise RemoteError("invalid sha256sum output for %s" % checked)
        return value[0].lower()

    def ensure_model_link(self, model_path: str, model_id: str) -> str:
        source = self._path(model_path)
        if not model_id or "/" in model_id or "\\" in model_id or model_id in {".", ".."}:
            raise RemotePathError("invalid model id: %s" % model_id)
        target = self._path(str(PurePosixPath(self.config.resolved_deployment_dir) / (model_id + ".safetensors")), allow_deployment=True)
        source_real = str(getattr(self.run(("readlink", "-f", source)), "stdout", "")).strip() or source
        exists = self.run(("test", "-e", target), check=False)
        if int(getattr(exists, "returncode", 1)) == 0:
            existing = self.run(("readlink", "-f", target), check=False)
            existing_real = str(getattr(existing, "stdout", "")).strip()
            if not existing_real:
                raise RemoteLinkConflict("deployment target is not a symlink: %s" % target)
            link_action(existing_real, source_real)
            return target
        self.run(("ln", "-s", source, target))
        return target


class ComfyUITunnel:
    """Forward a local ephemeral port to the remote ComfyUI loopback port."""

    def __init__(self, client: SSHClient, connect_timeout_s: float = 10.0):
        self.client = client
        self.connect_timeout_s = float(connect_timeout_s)
        self.process: Optional[subprocess.Popen] = None
        self.local_port: Optional[int] = None

    @property
    def base_url(self) -> str:
        if self.local_port is None:
            raise RemoteError("ComfyUI tunnel is not running")
        return "http://127.0.0.1:%d" % self.local_port

    def __enter__(self) -> "ComfyUITunnel":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            self.local_port = int(sock.getsockname()[1])
        finally:
            sock.close()
        argv = [
            "ssh",
            "-N",
            "-L",
            "%d:127.0.0.1:%d" % (self.local_port, int(self.client.config.comfyui_port)),
            self.client.config.host,
        ]
        self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + self.connect_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RemoteError("ComfyUI tunnel exited with code %s" % self.process.returncode)
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.settimeout(0.2)
            try:
                if probe.connect_ex(("127.0.0.1", self.local_port)) == 0:
                    return self
            finally:
                probe.close()
            time.sleep(0.05)
        self.__exit__(None, None, None)
        raise RemoteError("timed out waiting for ComfyUI SSH tunnel")

    def __exit__(self, exc_type, exc, tb) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

