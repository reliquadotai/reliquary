"""``Content-Encoding: gzip`` on proof requests, and its failure modes.

Starlette decodes no request encoding at all, so the worker carries its own
inflater. The bound must hold on the DECOMPRESSED size, otherwise a compressed
body could make the worker allocate far more than a plain one is allowed to.
"""
from __future__ import annotations

import gzip

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
import pytest

from reliquary.validator.remote_proof_protocol import InflateRequest

LIMIT = 64 * 1024
JSON = {"content-type": "application/json"}
GZIP = {**JSON, "content-encoding": "gzip"}


@pytest.fixture
def client():
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request):
        # Mirrors the worker: the body is consumed as a stream, which is why a
        # BaseHTTPMiddleware cannot do this job.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > LIMIT:
                raise HTTPException(413, "request too large")
        return {"length": len(body)}

    app.add_middleware(InflateRequest, limit=LIMIT)
    return TestClient(app, raise_server_exceptions=False)


def body(size=8192):
    return b'{"pad":"' + b"a" * size + b'"}'


def test_plain_bodies_are_untouched(client):
    raw = body()
    assert client.post("/echo", content=raw, headers=JSON).json() == {"length": len(raw)}


@pytest.mark.parametrize("chunked", [False, True])
def test_a_gzip_body_reaches_the_endpoint_decoded(client, chunked):
    raw = body()
    wire = gzip.compress(raw, 1)
    assert len(wire) < len(raw)
    content = ([wire[i:i + 512] for i in range(0, len(wire), 512)]
               if chunked else wire)
    response = client.post("/echo", content=iter(content) if chunked else content,
                           headers=GZIP)
    assert response.json() == {"length": len(raw)}


def test_the_bound_is_on_the_decompressed_size(client):
    # 8 MiB of zeroes is a few KiB on the wire; the limit still has to bite.
    bomb = gzip.compress(b"\0" * (8 * 1024 * 1024), 1)
    assert len(bomb) < LIMIT
    assert client.post("/echo", content=bomb, headers=GZIP).status_code == 413


def test_a_body_that_is_not_gzip_is_refused(client):
    assert client.post("/echo", content=b"not gzip at all", headers=GZIP).status_code == 400


def test_a_truncated_stream_is_refused(client):
    assert client.post("/echo", content=gzip.compress(body(), 1)[:64],
                       headers=GZIP).status_code == 400
