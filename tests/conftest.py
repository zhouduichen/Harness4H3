from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def workflow():
    return {
        "prompt": {"class_type": "Prompt", "inputs": {"text": "original"}},
        "seed": {"class_type": "Noise", "inputs": {"seed": 1}},
        "sampler": {"class_type": "Sampler", "inputs": {"steps": 4, "cfg": 5.0}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["sampler", 0]}},
    }


@pytest.fixture
def fake_comfyui():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            requests.append((self.path, body))
            payload = {"prompt_id": "prompt-1", "number": 1}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_GET(self):
            requests.append((self.path, None))
            if self.path.startswith("/history/"):
                payload = {
                    "prompt-1": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {"save": {"videos": [{"filename": "result.mp4", "subfolder": "", "type": "output"}]}},
                    }
                }
                data = json.dumps(payload).encode()
                content_type = "application/json"
            elif self.path.startswith("/view?"):
                data = b"fake-video"
                content_type = "video/mp4"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class Fake:
        url = "http://127.0.0.1:%d" % server.server_port
        calls = requests

    try:
        yield Fake()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

