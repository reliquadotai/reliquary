"""The reward host: the only machine that runs customer env containers.

Zero secret of ours (no wallet, no bucket credential; registry read access only). The admin host and the
validators call it over mutual TLS, as the remote CPU executor (``environment/grader/remote.py``):

* ``POST /v1/build`` (body: the archive, ``X-Env-Id``): builds and pushes the image, answers its digest;
* ``POST /v1/freeze`` ``{image, offset, limit, fresh}`` and ``POST /v1/score`` ``{image, items, fresh}``:
  through a warm worker per image, or a new sandbox with ``fresh`` (qualification's determinism check).

Infrastructure failures answer 503 with the reason; a customer's failure on an item is in its result.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import ssl
import subprocess
import threading
from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Request

from reliquary.customer_env import parse_env_id
from reliquary.customer_env.runner import RewardClient, RewardInfrastructureError, WorkerProcess, sealed_argv

logger = logging.getLogger(__name__)
MAX_WARM_IMAGES = 16
MAX_BODY_BYTES = 64 * 1024 * 1024
_IMAGE_RE = re.compile(r"\A[a-z0-9][a-z0-9./:_-]{0,254}@sha256:[0-9a-f]{64}\Z")


class WorkerPool:
    """One warm worker per image, at most ``MAX_WARM_IMAGES`` (least recently used closed first)."""

    def __init__(self, open_client: Callable[[str], RewardClient], *, allowed_prefix: str,
                 max_warm: int = MAX_WARM_IMAGES) -> None:
        self._open, self._prefix, self._max = open_client, allowed_prefix, max_warm
        self._warm: collections.OrderedDict[str, RewardClient] = collections.OrderedDict()
        self._lock = threading.Lock()

    def check(self, image: object) -> str:
        if not isinstance(image, str) or not _IMAGE_RE.match(image) or not image.startswith(self._prefix):
            raise ValueError(f"image must be {self._prefix}...@sha256:<digest>")
        return image

    def client(self, image: str) -> RewardClient:
        with self._lock:
            client = self._warm.pop(image, None) or self._open(image)
            self._warm[image] = client
            while len(self._warm) > self._max:
                _, old = self._warm.popitem(last=False)
                old.close()
            return client

    def close(self) -> None:
        with self._lock:
            for client in self._warm.values():
                client.close()
            self._warm.clear()


def docker_client(image: str, *, runtime: str = "runsc") -> RewardClient:
    pulled = subprocess.run(["docker", "pull", "--quiet", image], capture_output=True, text=True, timeout=900)
    if pulled.returncode != 0:
        raise RewardInfrastructureError(f"spawn_failed: pull failed: {pulled.stderr[-300:]}")
    return RewardClient(WorkerProcess(sealed_argv(image, runtime=runtime)))


def create_app(pool: WorkerPool, *, builder=None):
    app = FastAPI(title="Reliquary reward host", docs_url=None, redoc_url=None, openapi_url=None)

    async def body_json(request: Request) -> dict:
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            raise HTTPException(413, "request too large")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(400, "a JSON object") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "a JSON object")
        try:
            body["image"] = pool.check(body.get("image"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return body

    def through(body: dict, call):
        fresh = body.get("fresh") is True
        client = pool._open(body["image"]) if fresh else pool.client(body["image"])
        try:
            return call(client)
        except RewardInfrastructureError as exc:
            raise HTTPException(503, str(exc)[:500]) from None
        finally:
            if fresh:
                client.close()

    @app.post("/v1/score")
    async def score(request: Request) -> dict:
        body = await body_json(request)
        items = body.get("items")
        if not isinstance(items, list) or not items:
            raise HTTPException(422, "items is a non-empty list")
        from starlette.concurrency import run_in_threadpool

        results = await run_in_threadpool(through, body, lambda c: c.score(items))
        return {"results": [{"score": r.score} if r.error is None else {"error": r.error} for r in results]}

    @app.post("/v1/freeze")
    async def freeze(request: Request) -> dict:
        body = await body_json(request)
        from starlette.concurrency import run_in_threadpool

        rows, refused = await run_in_threadpool(through, body, lambda c: c.freeze_all())
        return {"rows": rows, "refused": refused}

    @app.post("/v1/build")
    async def build(request: Request) -> dict:
        from starlette.concurrency import run_in_threadpool

        from reliquary.customer_env.image import BuildFailed
        from reliquary.customer_env.package import PackageInvalid, read_package

        if builder is None:
            raise HTTPException(404, "this host does not build")
        try:
            env = parse_env_id(request.headers.get("x-env-id", ""))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        data = await request.body()
        try:
            package = read_package(data)
        except PackageInvalid as exc:
            raise HTTPException(422, f"package_invalid: {exc}") from None
        if package.sha256 != env.sha256:
            raise HTTPException(422, "package_invalid: the archive is not the version's sha256")

        try:
            return {"image": await run_in_threadpool(builder.build_archive, data, env)}
        except (BuildFailed, PackageInvalid) as exc:
            reason = "build_failed" if isinstance(exc, BuildFailed) else "package_invalid"
            raise HTTPException(422, f"{reason}: {exc}"[:2000]) from None

    return app


class RemoteRewardClient:
    """``RewardClient``'s interface over a reward host. ``fresh`` asks for a new sandbox per call."""

    def __init__(self, base_url: str, image: str, *, fresh: bool = False, cert: tuple[str, str] | None = None,
                 ca: str | None = None, timeout_s: float = 900.0, transport=None) -> None:
        import httpx

        self._image, self._fresh = image, fresh
        self._http = httpx.Client(base_url=base_url.rstrip("/"), cert=cert, verify=ca or True,
                                  timeout=timeout_s, transport=transport)
        self.provenance = None

    def _post(self, path: str, payload: dict) -> dict:
        import httpx

        try:
            response = self._http.post(path, json={"image": self._image, "fresh": self._fresh, **payload})
        except httpx.HTTPError as exc:
            raise RewardInfrastructureError(f"reward_host_unreachable: {exc}") from exc
        if response.status_code == 503:
            raise RewardInfrastructureError(response.json().get("detail", "reward host failed"))
        if response.status_code != 200:
            raise RewardInfrastructureError(f"reward_host_refused: {response.status_code} {response.text[:300]}")
        return response.json()

    def score(self, items):
        from reliquary.customer_env.runner import ScoreResult

        results = self._post("/v1/score", {"items": list(items)})["results"]
        if len(results) != len(items):
            raise RewardInfrastructureError("reward_host_garbled: one result per item expected")
        return [ScoreResult(r.get("score"), r.get("error")) for r in results]

    def freeze_all(self, **_):
        body = self._post("/v1/freeze", {})
        return body["rows"], body["refused"]

    def close(self) -> None:
        self._http.close()


class RemoteBuilder:
    """A builder host's ``/v1/build``: the admin sends the archive as bytes and never unpacks it."""

    def __init__(self, base_url: str, **http) -> None:
        import httpx

        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=2400, **http)

    def build_archive(self, data: bytes, env) -> str:
        from reliquary.customer_env.image import BuildFailed

        response = self._http.post("/v1/build", content=data, headers={"x-env-id": str(env),
                                                                       "content-type": "application/gzip"})
        if response.status_code == 422:
            detail = response.json().get("detail", "")
            raise BuildFailed(detail)
        response.raise_for_status()
        return response.json()["image"]


def main() -> None:  # pragma: no cover - a service entrypoint
    import uvicorn

    from reliquary.customer_env.image import REPOSITORY, DockerBuilder, LocalBuilder

    parser = argparse.ArgumentParser(description="Reliquary reward host")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--tls-cert", default=os.environ.get("RELIQUARY_REWARD_HOST_TLS_CERT", ""))
    parser.add_argument("--tls-key", default=os.environ.get("RELIQUARY_REWARD_HOST_TLS_KEY", ""))
    parser.add_argument("--client-ca", default=os.environ.get("RELIQUARY_REWARD_HOST_CLIENT_CA", ""))
    parser.add_argument("--build", action="store_true", help="also serve /v1/build (a builder host)")
    args = parser.parse_args()
    if not (args.tls_cert and args.tls_key and args.client_ca):
        raise SystemExit("TLS cert, key and client CA are required")
    pool = WorkerPool(docker_client, allowed_prefix=REPOSITORY + "/")
    app = create_app(pool, builder=LocalBuilder(DockerBuilder()) if args.build else None)
    uvicorn.run(app, host=args.host, port=args.port, ssl_certfile=args.tls_cert, ssl_keyfile=args.tls_key,
                ssl_ca_certs=args.client_ca, ssl_cert_reqs=ssl.CERT_REQUIRED)


__all__ = ["RemoteBuilder", "RemoteRewardClient", "WorkerPool", "create_app", "docker_client"]

if __name__ == "__main__":  # pragma: no cover
    main()
