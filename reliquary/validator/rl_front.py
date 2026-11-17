"""The RL front: the one process on the validator's public port. It routes miners (and trainer pods) to the RL
controller of each task, which listens on a local port (the supervisor's map file names them).

* ``/rl/<task>/<rest>``  ->  ``http://127.0.0.1:<port>/rl/<rest>`` of that task's controller;
* ``/rl/<rest>`` and every non-``/rl`` path  ->  the default task's controller;
* ``GET /rl/tasks``  ->  the live tasks with their auction views (miners pick a task from it).

Trainer paths (``/rl/<task>/trainer/*``) cross the front only behind the host's TLS terminator
(``X-Forwarded-Proto: https``) unless ``allow_plain_trainer``. The bare ``/rl/trainer/*`` is never proxied. The front
adds no rate limit: lease and trainer traffic are the controller's to judge.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 64 * 1024 * 1024
AUCTION_TIMEOUT_S = 2.0
DRAW_FLOOR = 0.9
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "host"}


class _BodyTooLarge(Exception):
    pass


def _refuse(status: int, error: str) -> JSONResponse:
    return JSONResponse({"error": error}, status_code=status, headers={"Cache-Control": "no-store"})


class _TaskMap:
    """The supervisor's map file, re-read when it changes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._stamp: tuple[int, int] | None = None
        self._value: dict[str, Any] = {"default": None, "tasks": {}}

    def get(self) -> dict[str, Any]:
        try:
            stat = self.path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp != self._stamp:
                raw = json.loads(self.path.read_text())
                tasks = {str(t): v for t, v in (raw.get("tasks") or {}).items() if int(v["port"]) > 0}
                default = raw.get("default")
                self._value = {"default": str(default) if default is not None else None, "tasks": tasks}
                self._stamp = stamp
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            logger.warning("rl front: task map %s unreadable; keeping the last one", self.path)
        return self._value


def build_front_app(map_path: Path, *, client: httpx.AsyncClient | None = None, allow_plain_trainer: bool = False,
                    max_body: int = MAX_BODY_BYTES) -> Starlette:
    tasks_map = _TaskMap(map_path)
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))

    def upstream_url(port: int, path: str, query: str) -> str:
        return f"http://127.0.0.1:{port}{path}" + (f"?{query}" if query else "")

    async def proxy(request: Request, port: int, path: str) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_body:
            return _refuse(413, "body_too_large")
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP]
        has_body = request.method not in ("GET", "HEAD", "OPTIONS") and (
            declared not in (None, "0") or "transfer-encoding" in request.headers)

        async def body() -> AsyncIterator[bytes]:
            seen = 0
            async for chunk in request.stream():
                seen += len(chunk)
                if seen > max_body:
                    raise _BodyTooLarge
                yield chunk

        upstream_request = http.build_request(request.method, upstream_url(port, path, request.url.query),
                                              headers=headers, content=body() if has_body else None)
        try:
            upstream = await http.send(upstream_request, stream=True)
        except _BodyTooLarge:
            return _refuse(413, "body_too_large")
        except httpx.ConnectError:
            return _refuse(503, "task_down")
        except httpx.TimeoutException:
            return _refuse(504, "task_timeout")
        except httpx.HTTPError as exc:
            if isinstance(exc.__cause__, _BodyTooLarge):
                return _refuse(413, "body_too_large")
            logger.warning("rl front: upstream error %s", type(exc).__name__)
            return _refuse(502, "task_error")
        out = [(k, v) for k, v in upstream.headers.multi_items() if k.lower() not in _HOP_BY_HOP]

        async def chunks() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():   # raw: a gzip body stays gzip
                    yield chunk
            finally:
                await upstream.aclose()

        response = StreamingResponse(chunks(), status_code=upstream.status_code)
        response.raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in out]
        return response

    async def task_view(task_id: str, port: int) -> dict | None:
        try:
            r = await http.get(upstream_url(port, "/rl/auction", ""), timeout=AUCTION_TIMEOUT_S)
            if r.status_code != 200:
                return None
            view = r.json()
            envs = {env: v.get("share") for env, v in (view.get("environments") or {}).items()}
            return {"task_id": task_id, "environments": envs, "auction": view, "draw_floor": DRAW_FLOOR}
        except (httpx.HTTPError, ValueError, AttributeError):
            return None

    async def tasks(request: Request) -> Response:
        listed = tasks_map.get()["tasks"]
        views = await asyncio.gather(*(task_view(t, int(v["port"])) for t, v in listed.items()))
        return JSONResponse([v for v in views if v is not None], headers={"Cache-Control": "no-store"})

    async def route(request: Request) -> Response:
        path = request.url.path
        current = tasks_map.get()
        segments = path.split("/")   # "", "rl", <task>, ...
        if path == "/rl/tasks" and request.method == "GET":
            return await tasks(request)
        if len(segments) > 3 and segments[1] == "rl" and segments[2] in current["tasks"]:
            task, rest = segments[2], "/rl/" + "/".join(segments[3:])
            if segments[3] == "trainer":
                if not allow_plain_trainer and request.headers.get("x-forwarded-proto", "").lower() != "https":
                    return _refuse(403, "trainer_needs_tls")
        else:
            if segments[1:3] == ["rl", "trainer"]:
                return _refuse(404, "not_found")
            task, rest = current["default"], path
            if task not in current["tasks"]:
                return _refuse(404, "task_unknown")
        return await proxy(request, int(current["tasks"][task]["port"]), rest)

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    return Starlette(routes=[Route("/{rest:path}", route, methods=methods)])


__all__ = ["DRAW_FLOOR", "MAX_BODY_BYTES", "build_front_app"]
