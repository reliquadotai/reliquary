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
# A result lost on the wire (the connection dropped, a timeout, a gateway
# error) is posted again: otherwise its lease only expires on the control,
# 3.3 h later for a replay. Bounded in attempts and in time, and never past
# the lease's own expiry, when known.
RESULT_POST_ATTEMPTS = 5
RESULT_RETRY_SECONDS = 120.0
RESULT_RETRY_BACKOFF_SECONDS = (5.0, 10.0, 20.0, 40.0)
_RETRIED_STATUSES = frozenset({502, 503, 504})


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
        # The control's time, which a lease's ``expires_at`` is in.
        self._wall_clock: Callable[[], float] = time.time
        self._sleep = asyncio.sleep
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

    async def post_result(self, lease_id: str, body: dict, *,
                          expires_at: float | None = None) -> None:
        """A late (410) or refused (422) result is the control's call, not an
        executor failure; a transport error or a gateway error is retried
        (``RESULT_POST_ATTEMPTS`` within ``RESULT_RETRY_SECONDS``, never past
        ``expires_at``); anything else unexpected raises."""
        import httpx

        started = self._clock()
        attempt = 0
        while True:
            attempt += 1
            try:
                posted = await self._post(f"{self._prefix}/{lease_id}/result", body)
                if posted.status_code in _RETRIED_STATUSES:
                    posted.raise_for_status()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if isinstance(exc, httpx.HTTPStatusError) and status not in _RETRIED_STATUSES:
                    raise
                delay = RESULT_RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(RESULT_RETRY_BACKOFF_SECONDS) - 1)]
                if (attempt >= RESULT_POST_ATTEMPTS
                        or self._clock() - started + delay > RESULT_RETRY_SECONDS
                        or (expires_at is not None and self._wall_clock() + delay >= expires_at)):
                    logger.error("%s lease %s: result not delivered after %d attempt(s): %r",
                                 self.kind, lease_id[:8], attempt, exc)
                    raise
                logger.warning("%s lease %s: result post failed (%r); retrying in %.0f s "
                               "(attempt %d/%d)", self.kind, lease_id[:8], exc, delay, attempt,
                               RESULT_POST_ATTEMPTS)
                await self._sleep(delay)
                continue
            break
        if posted.status_code in (410, 422):
            if attempt > 1:
                # The earlier attempt may have landed before its connection
                # dropped: the control then no longer knows the lease. Final.
                logger.warning("%s lease %s not taken after a retry (%d attempts; an earlier "
                               "attempt may have been delivered): %s", self.kind, lease_id[:8],
                               attempt, posted.text[:200])
            else:
                logger.warning("%s lease %s not taken: %s", self.kind, lease_id[:8],
                               posted.text[:200])
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


__all__ = ["ERROR_BACKOFF_SECONDS", "LeaseExecutor", "REQUEST_TIMEOUT_SECONDS",
           "RESULT_POST_ATTEMPTS", "RESULT_RETRY_SECONDS", "TOKEN_ENV",
           "serve_executor"]
