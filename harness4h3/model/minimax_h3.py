from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .. import Harness4H3Error


class BackendError(Harness4H3Error):
    def __init__(self, message: str, failure_type: str = "backend_request"):
        super().__init__(message)
        self.failure_type = failure_type


@dataclass(frozen=True)
class BackendResult:
    prompt_id: str
    artifacts: Tuple[Path, ...]
    history: Mapping[str, Any]
    wall_time_s: float


class MiniMaxH3Adapter:
    def __init__(
        self,
        base_url: str,
        request_timeout_s: float = 30,
        poll_interval_s: float = 2,
        task_timeout_s: float = 3600,
        get_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.poll_interval_s = poll_interval_s
        self.task_timeout_s = task_timeout_s
        self.get_retries = get_retries

    def _json(self, path: str, payload: Optional[Mapping[str, Any]] = None) -> Any:
        url = self.base_url + (path if path.startswith("/") else "/" + path)
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Accept": "application/json", **({"Content-Type": "application/json"} if data else {})},
            method="POST" if data is not None else "GET",
        )
        attempts = 1 if data is not None else self.get_retries + 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                    raw = response.read()
                    return json.loads(raw.decode("utf-8")) if raw else {}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                raise BackendError("HTTP %d %s: %s" % (exc.code, url, detail[:500]))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt + 1 >= attempts:
                    raise BackendError("request failed for %s: %s" % (url, exc))
                time.sleep(min(0.2 * (attempt + 1), 1.0))
            except json.JSONDecodeError as exc:
                raise BackendError("invalid JSON from %s: %s" % (url, exc))
        raise BackendError("request failed for %s" % url)

    def submit(self, workflow: Mapping[str, Any]) -> str:
        result = self._json("/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())})
        prompt_id = result.get("prompt_id") if isinstance(result, dict) else None
        if not prompt_id:
            raise BackendError("ComfyUI returned no prompt_id: %r" % result, "backend_submission")
        return str(prompt_id)

    def wait(self, prompt_id: str) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.task_timeout_s
        last = None
        path = "/history/" + urllib.parse.quote(prompt_id, safe="")
        while time.monotonic() < deadline:
            result = self._json(path)
            item = result.get(prompt_id) if isinstance(result, dict) else None
            if isinstance(item, dict):
                last = item
                status = item.get("status") or {}
                status_text = status.get("status_str")
                if status_text == "error":
                    messages = status.get("messages") or []
                    raise BackendError("ComfyUI execution failed: %r" % messages[-3:], "backend_execution")
                if status.get("completed") or status_text == "success":
                    return item
            time.sleep(self.poll_interval_s)
        raise BackendError("timed out waiting for %s; last=%r" % (prompt_id, last), "backend_timeout")

    @staticmethod
    def output_items(history: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        items: List[Mapping[str, Any]] = []
        outputs = history.get("outputs") or {}
        if not isinstance(outputs, Mapping):
            return items
        for node in outputs.values():
            if not isinstance(node, Mapping):
                continue
            for key in ("videos", "gifs", "images", "audio"):
                values = node.get(key) or []
                if isinstance(values, list):
                    items.extend(item for item in values if isinstance(item, Mapping) and item.get("filename"))
        return items

    def _download(self, item: Mapping[str, Any], output_dir: Path, prompt_id: str) -> Path:
        filename = Path(str(item["filename"])).name
        if filename in {"", ".", ".."}:
            raise BackendError("unsafe output filename", "artifact_missing")
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / filename
        if destination.exists():
            destination = output_dir / (destination.stem + "-" + prompt_id[:8] + destination.suffix)
        query = urllib.parse.urlencode(
            {
                "filename": str(item["filename"]),
                "subfolder": str(item.get("subfolder", "")),
                "type": str(item.get("type", "output")),
            }
        )
        url = self.base_url + "/view?" + query
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=self.request_timeout_s) as response:
                data = response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            raise BackendError("artifact download failed for %s: %s" % (url, exc), "artifact_download")
        if not data:
            raise BackendError("downloaded artifact is empty", "artifact_missing")
        destination.write_bytes(data)
        return destination

    def run(self, workflow: Mapping[str, Any], output_dir: Path) -> BackendResult:
        started = time.monotonic()
        prompt_id = self.submit(workflow)
        history = self.wait(prompt_id)
        items = self.output_items(history)
        if not items:
            raise BackendError("ComfyUI history contains no output artifacts", "artifact_missing")
        artifacts = tuple(self._download(item, Path(output_dir), prompt_id) for item in items)
        return BackendResult(
            prompt_id=prompt_id,
            artifacts=artifacts,
            history=history,
            wall_time_s=time.monotonic() - started,
        )

