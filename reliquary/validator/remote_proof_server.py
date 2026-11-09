"""mTLS GPU host for the async RL validator's remote audits (``audit_source() == "remote"``).

Serves ``/v1/health`` and the ``remote_rl_audit`` routes on a dedicated listener (never public HTTP)."""
from __future__ import annotations

import logging
import os
import ssl

from fastapi import FastAPI, HTTPException, Response
from starlette.middleware.gzip import GZipMiddleware

from reliquary.validator.remote_proof_protocol import (
    GZIP_LEVEL, GZIP_MIN_BYTES, MAX_REQUEST_BYTES, InflateRequest, canonical_bytes,
)

logger = logging.getLogger(__name__)


def create_proof_app(*, rl_audit, worker_id: str) -> FastAPI:
    """App factory for the mTLS listener: ``rl_audit`` is a ``remote_rl_audit.RlAuditBackend``."""
    from reliquary.validator.remote_rl_audit import mount_rl_audit

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    async def read(request, cls, limit=MAX_REQUEST_BYTES):
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise HTTPException(415, "application/json required")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                raise HTTPException(413, "request too large")
        try:
            return cls.read(bytes(body), limit=limit)
        except (ValueError, TypeError, RecursionError) as exc:
            raise HTTPException(422, "invalid audit protocol message") from exc

    @app.get("/v1/health")
    async def health():
        return Response(canonical_bytes({"worker_id": worker_id}), media_type="application/json")

    mount_rl_audit(app, rl_audit, worker_id=worker_id, read=read)
    # Responses: httpx decodes gzip itself. Requests: starlette decodes no Content-Encoding, hence
    # InflateRequest, added last so the body is plain before any other layer reads it.
    app.add_middleware(GZipMiddleware, minimum_size=GZIP_MIN_BYTES, compresslevel=GZIP_LEVEL)
    app.add_middleware(InflateRequest, limit=MAX_REQUEST_BYTES)
    return app


def main():
    import uvicorn

    from reliquary import constants as c
    from reliquary.validator.remote_rl_audit import AUDIT_DEVICE_ENV, rl_audit_backend_from_environment

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    def required(key):
        value = os.environ.get(key, "").strip()
        if not value:
            raise RuntimeError(f"{key} is required")
        return value

    worker_id = required("RELIQUARY_PROOF_WORKER_ID")
    host = required("RELIQUARY_PROOF_HOST")
    # Validate mandatory TLS before loading the model.
    tls = {"ssl_certfile": required("RELIQUARY_PROOF_TLS_CERT"),
           "ssl_keyfile": required("RELIQUARY_PROOF_TLS_KEY"),
           "ssl_ca_certs": required("RELIQUARY_PROOF_CLIENT_CA"),
           "ssl_cert_reqs": ssl.CERT_REQUIRED}
    # The audits' verify model, synced from the signed checkpoint index the controller forwards; its weights
    # replace the base model's at the first sync.
    rl_audit = rl_audit_backend_from_environment(
        checkpoint=c.DEFAULT_BASE_MODEL, load_kwargs={"revision": c.DEFAULT_BASE_MODEL_REVISION})
    if rl_audit is None:
        raise RuntimeError(f"{AUDIT_DEVICE_ENV} is required")
    app = create_proof_app(rl_audit=rl_audit, worker_id=worker_id)
    uvicorn.run(app, host=host, port=int(os.environ.get("RELIQUARY_PROOF_PORT", "8445")),
                access_log=False, server_header=False, timeout_keep_alive=120, **tls)


if __name__ == "__main__":
    main()
