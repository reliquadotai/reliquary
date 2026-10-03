"""What every pull executor shares (audit, grade): one secret, its token, and
HTTPS it opens itself to a control's ``{prefix}/heartbeat``, ``{prefix}/claim``
and ``{prefix}/{lease_id}/result`` routes. Subclasses say what a lease is and
how its work is done; this module only talks to the control."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

ERROR_BACKOFF_SECONDS = 10.0
REQUEST_TIMEOUT_SECONDS = 120.0
TOKEN_ENV = "RELIQUARY_EXECUTOR_TOKEN"


class LeaseExecutor:
    kind = "executor"

    def __init__(self, *, http, executor_id: str, token: str, prefix: str,
                 heartbeat_seconds: float, idle_seconds: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not token:
            raise ValueError(f"{TOKEN_ENV} is empty")
        self._http = http
        self._executor_id = executor_id
        self._prefix = prefix
        self._headers = {"Authorization": f"Bearer {token}"}
        self._heartbeat_every = heartbeat_seconds
        self._idle = idle_seconds
        self._clock = clock
        self._last_heartbeat: float | None = None
        self.leases = 0

    async def _post(self, path: str, body: dict):
        return await self._http.post(path, json=body, headers=self._headers,
                                     timeout=REQUEST_TIMEOUT_SECONDS)

    def heartbeat_detail(self) -> dict:
        return {"leases": self.leases}

    async def heartbeat(self) -> dict:
        response = await self._post(f"{self._prefix}/heartbeat", {
            "executor_id": self._executor_id, "detail": self.heartbeat_detail()})
        response.raise_for_status()
        self._last_heartbeat = self._clock()
        return response.json()

    async def heartbeat_if_due(self) -> None:
        if self._last_heartbeat is None or self._clock() - self._last_heartbeat >= self._heartbeat_every:
            await self.heartbeat()

    async def post_result(self, lease_id: str, body: dict) -> None:
        """A late (410) or refused (422) result is the control's call, not an
        executor failure; anything else unexpected raises."""
        posted = await self._post(f"{self._prefix}/{lease_id}/result", body)
        if posted.status_code in (410, 422):
            logger.warning("%s lease %s not taken: %s", self.kind, lease_id[:8], posted.text[:200])
        else:
            posted.raise_for_status()
        self.leases += 1

    async def start(self) -> None:
        raise NotImplementedError

    async def step(self) -> bool:
        raise NotImplementedError

    async def run(self) -> None:
        await self.start()
        while True:
            try:
                worked = await self.step()
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 401:
                    # Revoked, expired or quarantined: nothing left to do here.
                    logger.critical("%s %s refused by the control: %s", self.kind,
                                    self._executor_id, exc)
                    raise
                logger.warning("%s step failed: %r; backing off", self.kind, exc)
                await asyncio.sleep(ERROR_BACKOFF_SECONDS)
                continue
            if not worked:
                await asyncio.sleep(self._idle)


def serve_executor(control_url: str, build: Callable[..., LeaseExecutor]) -> None:
    """Run ``build(http=..., token=...)`` against ``control_url`` until it
    raises, with the token from ``RELIQUARY_EXECUTOR_TOKEN``."""
    import httpx

    token = os.environ.get(TOKEN_ENV, "").strip()

    async def main() -> None:
        async with httpx.AsyncClient(base_url=control_url.rstrip("/"),
                                     follow_redirects=False) as http:
            await build(http=http, token=token).run()

    asyncio.run(main())


__all__ = ["ERROR_BACKOFF_SECONDS", "LeaseExecutor", "REQUEST_TIMEOUT_SECONDS", "TOKEN_ENV",
           "serve_executor"]
