from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "services" / "snapshot_sync.py"
SPEC = importlib.util.spec_from_file_location("snapshot_sync", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
snapshot_sync = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = snapshot_sync
SPEC.loader.exec_module(snapshot_sync)


class _Handler(BaseHTTPRequestHandler):
    state_body = b'{"state":"open","window_n":42}'
    state_status = 200

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/state"):
            body, status = self.state_body, self.state_status
        elif self.path == "/health":
            body, status = b'{"status":"healthy"}', 200
        else:
            body, status = b'{"detail":"not_found"}', 404
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_publish_is_atomic_and_identity_matches_gzip(tmp_path):
    publisher = snapshot_sync.SnapshotPublisher(tmp_path / "snapshots", tmp_path / "metrics")
    target = snapshot_sync.Target("state-default", "/state", 0.05, 0.2, 1024)
    digest = publisher.publish(target, _Handler.state_body)

    current = tmp_path / "snapshots" / "current" / "state-default"
    assert current.is_symlink()
    assert (current / "payload.json").read_bytes() == _Handler.state_body
    assert gzip.decompress((current / "payload.json.gz").read_bytes()) == _Handler.state_body
    assert digest == snapshot_sync.hashlib.sha256(_Handler.state_body).hexdigest()


def test_target_loop_publishes_then_retires_explicit_503(tmp_path):
    server = _server()
    publisher = snapshot_sync.SnapshotPublisher(tmp_path / "snapshots", tmp_path / "metrics")
    target = snapshot_sync.Target("state-default", "/state", 0.02, 0.2, 1024)
    stop = threading.Event()
    thread = threading.Thread(
        target=snapshot_sync.target_loop,
        kwargs={
            "upstream": f"http://127.0.0.1:{server.server_port}",
            "target": target,
            "publisher": publisher,
            "stop": stop,
            "timeout_s": 0.2,
        },
    )
    try:
        _Handler.state_status = 200
        thread.start()
        current = tmp_path / "snapshots" / "current" / "state-default"
        deadline = time.monotonic() + 1.0
        while not current.is_symlink() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert current.is_symlink()

        _Handler.state_status = 503
        deadline = time.monotonic() + 1.0
        while current.is_symlink() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not current.exists()
        status = json.loads((tmp_path / "snapshots" / "status" / "state-default.json").read_text())
        assert status["upstream_status"] == 503
        assert status["published"] is False
    finally:
        stop.set()
        thread.join(timeout=1.0)
        server.shutdown()
        server.server_close()
        _Handler.state_status = 200


def test_fetch_rejects_invalid_or_oversized_json(tmp_path):
    server = _server()
    try:
        _Handler.state_body = b"not-json"
        result = snapshot_sync.fetch_json(
            f"http://127.0.0.1:{server.server_port}/state",
            timeout_s=0.2,
            max_bytes=1024,
        )
        assert result.status == 0
        assert result.error == "invalid_json"

        _Handler.state_body = b'"' + (b"x" * 128) + b'"'
        result = snapshot_sync.fetch_json(
            f"http://127.0.0.1:{server.server_port}/state",
            timeout_s=0.2,
            max_bytes=16,
        )
        assert result.status == 0
        assert result.error == "response_too_large"
    finally:
        server.shutdown()
        server.server_close()
        _Handler.state_body = b'{"state":"open","window_n":42}'
