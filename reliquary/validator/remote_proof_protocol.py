"""JSON wire helpers of the remote RL audit host (``remote_rl_audit``): canonical bytes, bounded strict
models and gzip request bodies."""
from __future__ import annotations

import hashlib
import json
import zlib
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from reliquary.shared.strict_json import strict_json_loads

MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_TOKENS = 65536
Name = Annotated[str, Field(min_length=1, max_length=192, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+#-]*$")]
Count = Annotated[int, Field(ge=0, le=2**63-1)]
Probability = Annotated[float, Field(ge=0, le=1)]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


# Level 1: the audit RPC is a latency path and higher levels gain little for their CPU.
GZIP_LEVEL = 1
# Below this a compressed body is not smaller than the plain one.
GZIP_MIN_BYTES = 1024


async def _refuse(send, status: int, detail: str) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body",
                "body": b'{"detail":"%s"}' % detail.encode("ascii")})


class InflateRequest:
    """Decode ``Content-Encoding: gzip`` on request bodies.

    The bound is applied to the DECOMPRESSED size, so ``MAX_*_REQUEST_BYTES``
    keeps the meaning it has without compression and a compressed body cannot
    make the worker allocate more than a plain one.

    Pure ASGI on purpose: the endpoints consume ``request.stream()``, and a
    BaseHTTPMiddleware that rewrites its own Request object leaves them reading
    the still-compressed channel.
    """

    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not any(
            k == b"content-encoding" and v.lower() == b"gzip"
            for k, v in scope["headers"]
        ):
            return await self.app(scope, receive, send)
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        body = bytearray()
        more = True
        try:
            while more:
                message = await receive()
                if message["type"] != "http.request":
                    break
                body.extend(decoder.decompress(
                    message.get("body", b""), self.limit - len(body) + 1))
                if len(body) > self.limit:
                    return await _refuse(send, 413, "request too large")
                more = message.get("more_body", False)
        except zlib.error:
            return await _refuse(send, 400, "malformed gzip body")
        if not decoder.eof:
            return await _refuse(send, 400, "truncated gzip body")
        payload, delivered = bytes(body), False

        async def inflated_receive():
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}

        headers = [(k, v) for k, v in scope["headers"]
                   if k not in (b"content-encoding", b"content-length")]
        headers.append((b"content-length", str(len(payload)).encode("ascii")))
        await self.app({**scope, "headers": headers}, inflated_receive, send)


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @classmethod
    def read(cls, body: bytes, *, limit: int = MAX_REQUEST_BYTES):
        if len(body) > limit:
            raise ValueError("remote proof message is too large")
        return cls.model_validate(strict_json_loads(body))
