"""The RL front: one public app routing to a controller per RL task (stub controllers over ASGI)."""
from __future__ import annotations

import asyncio
import gzip
import json

import pytest

httpx = pytest.importorskip("httpx")

from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, Response  # noqa: E402
from starlette.routing import Route  # noqa: E402

from reliquary.validator.rl_front import build_front_app  # noqa: E402


def controller(name: str, shares: dict[str, float]) -> Starlette:
    async def echo(request: Request) -> JSONResponse:
        body = await request.body()
        return JSONResponse({"prefix": request.headers.get("x-forwarded-prefix"), "task": name, "path": request.url.path, "method": request.method,
                             "query": request.url.query, "body": body.decode(),
                             "x": request.headers.get("x-test")})

    async def auction(request: Request) -> JSONResponse:
        return JSONResponse({"lag_target": 3, "environments": {e: {"share": s} for e, s in shares.items()}})

    async def zipped(request: Request) -> Response:
        return Response(gzip.compress(b'{"z":1}'), media_type="application/json",
                        headers={"Content-Encoding": "gzip"})

    return Starlette(routes=[Route("/rl/auction", auction), Route("/rl/zipped", zipped),
                             Route("/rl/{rest:path}", echo, methods=["GET", "POST"]),
                             Route("/{rest:path}", echo, methods=["GET", "POST"])])


class Stack:
    def __init__(self, tmp_path, tasks, default="a", **front_kwargs):
        self.apps = {8001: controller("a", {"math": 0.5, "code": 0.5}), 8002: controller("b", {"if": 1.0})}
        self.down = set()
        spec = {"default": default, "tasks": {t: {"port": p, "run": f"run-{t}"} for t, p in tasks.items()}}
        self.map = tmp_path / "map.json"
        self.map.write_text(json.dumps(spec))

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError

        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(inner, request):
                port = request.url.port
                if port in self.down or port not in self.apps:
                    raise httpx.ConnectError("refused", request=request)
                assert request.url.host == "127.0.0.1"
                return await httpx.ASGITransport(app=self.apps[port]).handle_async_request(request)

        self.hung = set()
        self.closed = 0

        class Hang(httpx.AsyncBaseTransport):
            async def handle_async_request(inner, request):
                await asyncio.Event().wait()

        self.transports = {}

        def factory(port):
            return httpx.AsyncClient(transport=Hang() if port in self.hung else Transport())

        self.front = build_front_app(self.map, client_factory=factory, **front_kwargs)

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.front), base_url="http://front")


@pytest.fixture
def stack(tmp_path):
    return Stack(tmp_path, {"a": 8001, "b": 8002})


async def test_a_task_path_reaches_its_controller(stack):
    async with stack.client() as c:
        r = await c.post("/rl/b/lease/request?x=1", content=b"hello", headers={"x-test": "t"})
    assert r.status_code == 200
    assert r.json() == {"prefix": "/rl/b", "task": "b", "path": "/rl/lease/request", "method": "POST", "query": "x=1",
                        "body": "hello", "x": "t"}


async def test_a_bare_path_reaches_the_default_task(stack):
    async with stack.client() as c:
        rl = await c.get("/rl/state")
        other = await c.get("/state")
    assert rl.json()["prefix"] is None
    assert rl.json()["task"] == "a" and rl.json()["path"] == "/rl/state"
    assert other.json()["task"] == "a" and other.json()["path"] == "/state"


async def test_a_down_task_answers_503_and_others_still_work(stack):
    stack.down.add(8002)
    async with stack.client() as c:
        down = await c.get("/rl/b/state")
        up = await c.get("/rl/a/state")
    assert down.status_code == 503 and down.json()["error"] == "task_down"
    assert up.status_code == 200 and up.json()["task"] == "a"


async def test_bare_paths_answer_503_while_the_default_task_is_not_mapped(tmp_path):
    s = Stack(tmp_path, {"b": 8002}, default="a")
    async with s.client() as c:
        r = await c.get("/rl/state")
    assert r.status_code == 503 and r.json()["error"] == "task_down"


async def test_an_unknown_task_answers_404_and_without_a_default_bare_paths_do(tmp_path):
    s = Stack(tmp_path, {"a": 8001}, default=None)
    async with s.client() as c:
        # "ghost" is not a task: the whole path is a bare one, and there is no default
        assert (await c.get("/rl/ghost/trainer/x")).status_code == 404
        assert (await c.get("/rl/state")).status_code == 404
        assert (await c.get("/state")).status_code == 404


async def test_tasks_lists_live_tasks_with_their_auction(stack):
    async with stack.client() as c:
        both = (await c.get("/rl/tasks")).json()
        stack.down.add(8002)
        live = (await c.get("/rl/tasks")).json()
    assert {t["task_id"] for t in both} == {"a", "b"}
    a = next(t for t in both if t["task_id"] == "a")
    assert a["environments"] == {"math": 0.5, "code": 0.5} and a["draw_floor"] == 0.9
    assert a["auction"]["lag_target"] == 3
    assert [t["task_id"] for t in live] == ["a"]


async def test_gzip_passes_through_untouched(stack):
    async with stack.client() as c:
        r = await c.get("/rl/a/zipped")
    assert r.headers["content-encoding"] == "gzip" and r.json() == {"z": 1}


async def two_chunks():
    for _ in range(2):
        yield b"x" * 6


async def test_a_body_over_the_limit_is_refused(tmp_path):
    s = Stack(tmp_path, {"a": 8001}, max_body=10)
    async with s.client() as c:
        declared = await c.post("/rl/a/x", content=b"x" * 11)
        streamed = await c.post("/rl/a/x", content=two_chunks())
        fine = await c.post("/rl/a/x", content=b"x" * 10)
    assert declared.status_code == 413 and streamed.status_code == 413 and fine.status_code == 200


async def test_trainer_paths_need_tls_unless_allowed(tmp_path):
    s = Stack(tmp_path, {"a": 8001})
    async with s.client() as c:
        plain = await c.get("/rl/a/trainer/step")
        tls = await c.get("/rl/a/trainer/step", headers={"X-Forwarded-Proto": "https"})
        http = await c.get("/rl/a/trainer/step", headers={"X-Forwarded-Proto": "http"})
        bare = await c.get("/rl/trainer/step", headers={"X-Forwarded-Proto": "https"})
    assert plain.status_code == 403 and plain.json()["error"] == "trainer_needs_tls"
    assert http.status_code == 403
    assert tls.status_code == 200 and tls.json()["path"] == "/rl/trainer/step"
    assert bare.status_code == 404   # operator trainers keep their own loopback listener

    open_ = Stack(tmp_path, {"a": 8001}, allow_plain_trainer=True)
    async with open_.client() as c:
        assert (await c.get("/rl/a/trainer/step")).status_code == 200
        assert (await c.get("/rl/trainer/step")).status_code == 404


async def test_a_hung_task_does_not_delay_the_others_or_tasks_listing(stack, monkeypatch):
    import reliquary.validator.rl_front as front

    monkeypatch.setattr(front, "AUCTION_TIMEOUT_S", 0.3)
    stack.hung.add(8002)
    async with stack.client() as c:
        stuck = asyncio.create_task(c.get("/rl/b/state"))
        await asyncio.sleep(0.05)
        ok = await asyncio.wait_for(c.get("/rl/a/state"), 1.0)
        listed = await asyncio.wait_for(c.get("/rl/tasks"), 2.0)
        stuck.cancel()
    assert ok.status_code == 200
    assert [t["task_id"] for t in listed.json()] == ["a"]


async def test_a_pool_timeout_answers_503_task_busy(tmp_path):
    s = Stack(tmp_path, {"a": 8001})

    class Busy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.PoolTimeout("busy", request=request)

    s.front = build_front_app(s.map, client_factory=lambda port: httpx.AsyncClient(transport=Busy()))
    async with s.client() as c:
        r = await c.get("/rl/a/state")
    assert r.status_code == 503 and r.json()["error"] == "task_busy"


async def test_the_upstream_is_closed_after_the_response(tmp_path):
    s = Stack(tmp_path, {"a": 8001})
    closed = []

    class Tracked(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"{}"

        async def aclose(self):
            closed.append(1)

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, stream=Tracked())

    s.front = build_front_app(s.map, client_factory=lambda port: httpx.AsyncClient(transport=T()))
    async with s.client() as c:
        assert (await c.get("/rl/a/state")).status_code == 200
    assert closed


async def test_a_live_task_without_an_auction_is_listed_with_null(tmp_path):
    s = Stack(tmp_path, {"a": 8001})

    class Plain(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(409, json={"error": "not_auctioned"})

    s.front = build_front_app(s.map, client_factory=lambda port: httpx.AsyncClient(transport=Plain()))
    async with s.client() as c:
        listed = (await c.get("/rl/tasks")).json()
    assert listed == [{"task_id": "a", "environments": {}, "auction": None, "draw_floor": 0.9, "model": None}]


async def test_empty_segments_under_rl_are_refused(stack):
    async with stack.client() as c:
        for path in ("/rl//trainer/x", "/rl/a//trainer/x", "/rl/a//state"):
            r = await c.get(path, headers={"X-Forwarded-Proto": "https"})
            assert r.status_code == 404, path


async def test_tasks_name_each_tasks_model_from_the_map(tmp_path):
    s = Stack(tmp_path, {"a": 8001, "b": 8002})
    spec = json.loads(s.map.read_text())
    spec["tasks"]["a"]["model"] = {"id": "Qwen/Qwen3.5-4B", "revision": "r1"}
    s.map.write_text(json.dumps(spec))
    async with s.client() as c:
        listed = {t["task_id"]: t for t in (await c.get("/rl/tasks")).json()}
    assert listed["a"]["model"] == {"id": "Qwen/Qwen3.5-4B", "revision": "r1"} and listed["b"]["model"] is None
