"""The async RL validator's public HTTP app: health probes plus the routers ``rl_async_service`` mounts."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from reliquary.infrastructure.process_health import collect_process_health
from reliquary.shared.checkpoint_namespace import active_checkpoint_namespace


def validator_state_dir() -> Path:
    """The validator's local state directory, scoped to the active checkpoint namespace."""
    return active_checkpoint_namespace().local_path(
        os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    )


class RlHttpApp:
    """``/livez``, ``/readyz`` (process health only) and the mounted ``/rl/*`` routers."""

    def __init__(self) -> None:
        self._process_health: dict[str, Any] = {"status": "pending"}
        self.app = FastAPI(title="Reliquary validator", docs_url=None, redoc_url=None, openapi_url=None)

        @self.app.get("/livez")
        async def livez() -> Response:
            return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

        @self.app.get("/readyz")
        async def readyz() -> Response:
            reasons = []
            if self._process_health.get("status") in {"critical", "error", "pending"}:
                reasons.append("process_health_unavailable")
            return JSONResponse({"ready": not reasons, "reasons": reasons},
                                status_code=503 if reasons else 200, headers={"Cache-Control": "no-store"})

    def mount_rl_lease_router(self, router: Any) -> None:
        """Include ``router``; a route it serves replaces the app's own at the same path and method."""
        served = {(getattr(r, "path", None), m) for r in router.routes for m in (getattr(r, "methods", None) or ())}
        self.app.router.routes[:] = [
            r for r in self.app.router.routes
            if not any((getattr(r, "path", None), m) in served for m in (getattr(r, "methods", None) or ()))]
        self.app.include_router(router)

    async def monitor_process_health(self, interval: float = 30.0) -> None:
        """Refresh /proc and cgroup telemetry off the event loop; telemetry never stops the service."""
        while True:
            try:
                self._process_health = await asyncio.to_thread(collect_process_health)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._process_health = {"status": "unknown", "collected_at": time.time(),
                                        "collector_error_type": type(exc).__name__}
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
