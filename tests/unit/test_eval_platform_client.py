"""The pod's platform client, against an in-memory platform speaking the
internal evaluation contract."""

from __future__ import annotations

import hashlib
import json
import re

import httpx
import pytest

from reliquary.eval.platform_client import LeaseLost, PlatformClient, PlatformError

TOKEN = "pod-token"


class FakePlatform:
    """The platform's /api/internal/evaluations routes, in memory."""

    def __init__(self, *, tasks=(), prompts=(), part_size=64) -> None:
        self.tasks = list(tasks)
        self.prompts = list(prompts)
        self.part_size = part_size
        self.claimed: dict | None = None
        self.heartbeats = 0
        self.events: list[dict] = []
        self.uploads: dict[str, dict] = {}
        self.objects: dict[str, bytes] = {}
        self.results: list[dict] = []
        self.puts = 0
        self.fail_puts_after: int | None = None
        self.heartbeat_status = 200
        self.flaky: list[int] = []  # statuses returned once each, first
        self.calls: list[tuple[str, str]] = []
        self.status_reports_key = True

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self, **kwargs) -> PlatformClient:
        http = httpx.Client(base_url="http://platform", transport=self.transport())
        return PlatformClient("http://platform", TOKEN, executor_id="pod-1", http=http,
                              sleep=lambda s: None, **kwargs)

    def _json(self, request):
        return json.loads(request.content or b"{}")

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        self.calls.append((method, path))
        if self.flaky:
            return httpx.Response(self.flaky.pop(0))
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"detail": "bad token"})
        if path == "/api/internal/evaluations/claim" and method == "POST":
            assert self._json(request) == {"executor_id": "pod-1"}
            if not self.tasks:
                return httpx.Response(204)
            self.claimed = self.tasks.pop(0)
            return httpx.Response(200, json=self.claimed)
        match = re.fullmatch(r"/api/internal/evaluations/([^/]+)/(.+)", path)
        assert match, path
        task_id, rest = match.groups()
        if (self.claimed is None or task_id != self.claimed["task_id"]
                or request.headers.get("x-task-lease") != self.claimed["lease"]):
            return httpx.Response(409, json={"detail": "lease"})
        if rest == "prompts":
            body = "".join(json.dumps(p) + "\n" for p in self.prompts)
            return httpx.Response(200, text=body)
        if rest == "heartbeat":
            self.heartbeats += 1
            return httpx.Response(self.heartbeat_status, json={})
        if rest == "events":
            self.events.append(self._json(request))
            return httpx.Response(200, json={})
        if rest == "result":
            self.results.append(self._json(request))
            return httpx.Response(200, json={})
        if rest == "uploads":
            body = self._json(request)
            upload_id = f"u{len(self.uploads)}"
            self.uploads[upload_id] = {**body, "parts": {}}
            return httpx.Response(201, json={"upload_id": upload_id, "part_size": self.part_size})
        match = re.fullmatch(r"uploads/([^/]+)(?:/(\w+))?", rest)
        assert match, rest
        upload_id, tail = match.groups()
        upload = self.uploads.get(upload_id)
        if upload is None:
            return httpx.Response(404, json={"detail": "upload_unknown"})
        if upload.get("expired"):
            return httpx.Response(410, json={"detail": "upload_expired"})
        if tail is None and method == "GET":
            answer = {"parts": sorted(upload["parts"])}
            if self.status_reports_key and upload.get("key"):
                answer["key"] = upload["key"]
            return httpx.Response(200, json=answer)
        if tail == "complete":
            if upload.get("key"):
                return httpx.Response(409, json={"detail": "upload_complete"})
            data = b"".join(upload["parts"][n] for n in sorted(upload["parts"]))
            if len(data) != upload["size"] or hashlib.sha256(data).hexdigest() != upload["sha256"]:
                return httpx.Response(422, json={"detail": "incomplete"})
            key = f"evaluations/{task_id}/{upload['name']}"
            self.objects[key] = data
            upload["key"] = key
            return httpx.Response(200, json={"key": key})
        if method == "PUT":
            if self.fail_puts_after is not None and self.puts >= self.fail_puts_after:
                raise httpx.ConnectError("pod lost the network")
            if hashlib.sha256(request.content).hexdigest() != request.headers["x-part-sha256"]:
                return httpx.Response(422, json={"detail": "part sha"})
            self.puts += 1
            upload["parts"][int(tail)] = request.content
            return httpx.Response(200, json={})
        raise AssertionError((method, path))


def _task(task_id="ev-1"):
    return {"task_id": task_id, "lease": "lease-1",
            "model": {"repo": "org/m", "revision": "a" * 40}, "gpu_count": 1,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7},
            "max_new_tokens": 64, "thinking": False,
            "envs": [{"env": "logic", "set_id": "logic-eval-s1-n4", "samples": 2,
                      "problem_ids_count": 2}]}


def test_claim_then_every_call_carries_the_lease():
    platform = FakePlatform(tasks=[_task()], prompts=[{"problem_id": "p"}])
    client = platform.client()
    assert client.claim()["task_id"] == "ev-1"
    assert list(client.prompts()) == [{"problem_id": "p"}]
    client.heartbeat()
    client.event("progress", {"done": 1, "total": 2})
    client.result(completion_keys=["k"], vllm_version="v", gpu="g", model_sha="s", rows=1,
                  seconds=1.0)
    assert platform.heartbeats == 1
    assert platform.events == [{"kind": "progress", "detail": {"done": 1, "total": 2}}]
    assert platform.results[0]["completion_keys"] == ["k"]


def test_no_task_is_none():
    assert FakePlatform().client().claim() is None


def test_a_refused_lease_is_lost_not_retried():
    platform = FakePlatform(tasks=[_task()])
    client = platform.client()
    client.claim()
    platform.claimed["lease"] = "another"
    with pytest.raises(LeaseLost):
        client.heartbeat()
    assert sum(1 for c in platform.calls if c[1].endswith("heartbeat")) == 1


def test_server_errors_are_retried():
    platform = FakePlatform(tasks=[_task()])
    client = platform.client()
    platform.flaky = [503, 502]
    assert client.claim() is not None
    platform.flaky = [500] * 10
    with pytest.raises(PlatformError):
        client.heartbeat()


def test_upload_in_parts_and_resume_sends_only_the_missing_parts(tmp_path):
    platform = FakePlatform(tasks=[_task()], part_size=10)
    client = platform.client(retries=0)
    client.claim()
    path = tmp_path / "chunk.jsonl"
    path.write_bytes(bytes(range(256)) * 2)  # 512 bytes, 52 parts
    started = []
    platform.fail_puts_after = 20
    with pytest.raises(PlatformError):
        client.upload_file(path, name="chunk.jsonl", on_created=started.append)
    assert platform.puts == 20
    platform.fail_puts_after = None
    key = client.upload_file(path, name="chunk.jsonl", resume=started[0])
    assert platform.objects[key] == path.read_bytes()
    assert platform.puts == 52  # the 20 already there were not sent again


def test_an_unknown_resume_starts_a_new_upload(tmp_path):
    platform = FakePlatform(tasks=[_task()], part_size=10)
    client = platform.client()
    client.claim()
    path = tmp_path / "c"
    path.write_bytes(b"x" * 25)
    key = client.upload_file(path, name="c", resume={"upload_id": "gone", "part_size": 10})
    assert platform.objects[key] == b"x" * 25


def test_an_upload_already_complete_is_success_not_a_lost_lease(tmp_path):
    platform = FakePlatform(tasks=[_task()], part_size=10)
    client = platform.client()
    client.claim()
    path = tmp_path / "c"
    path.write_bytes(b"y" * 25)
    started = []
    key = client.upload_file(path, name="c", on_created=started.append)
    puts = platform.puts
    # A crash after `complete` but before the key was saved: resume finds it done.
    assert client.upload_file(path, name="c", resume=started[0]) == key
    assert platform.puts == puts
    # A platform whose status does not name the key: complete answers 409 once
    # done, and the client starts a fresh upload rather than stopping.
    platform.status_reports_key = False
    again = client.upload_file(path, name="c", resume=started[0])
    assert platform.objects[again] == b"y" * 25


def test_an_expired_upload_is_restarted(tmp_path):
    platform = FakePlatform(tasks=[_task()], part_size=10)
    client = platform.client()
    client.claim()
    path = tmp_path / "c"
    path.write_bytes(b"z" * 25)
    started = []
    platform.fail_puts_after = 1
    with pytest.raises(PlatformError):
        client.upload_file(path, name="c", on_created=started.append)
    platform.fail_puts_after = None
    platform.uploads[started[0]["upload_id"]]["expired"] = True
    key = client.upload_file(path, name="c", resume=started[0])
    assert platform.objects[key] == b"z" * 25


def test_a_refused_token_on_an_upload_still_stops(tmp_path):
    platform = FakePlatform(tasks=[_task()], part_size=10)
    client = platform.client()
    client.claim()
    path = tmp_path / "c"
    path.write_bytes(b"z" * 25)
    client._token = "revoked"
    with pytest.raises(LeaseLost):
        client.upload_file(path, name="c")
