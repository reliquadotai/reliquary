"""The subnet's registered hotkeys, as the corpus route reads them.

Same rules as the RL validator's registration gate: requests only read a
snapshot, a background loop refreshes it from the chain, and a snapshot too old
to trust answers "unavailable" (retry) rather than "not registered" (refuse).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from reliquary.constants import (
    REGISTERED_HOTKEY_CACHE_TTL_SECONDS,
    REGISTERED_HOTKEY_REFRESH_TIMEOUT_SECONDS,
    REGISTERED_HOTKEY_STALE_GRACE_SECONDS,
)

logger = logging.getLogger(__name__)

NOT_REGISTERED = "not_registered"
UNAVAILABLE = "unavailable"

# Well inside the TTL, so a miner registered minutes ago is admitted soon.
REFRESH_SECONDS = 600.0
RETRY_SECONDS = 60.0


async def load_registered_hotkeys(netuid: int) -> set[str]:
    from reliquary.infrastructure import chain

    subtensor = await chain.get_subtensor()
    try:
        neurons = await chain.get_neurons_lite(subtensor, netuid)
    finally:
        await chain.close_subtensor(subtensor)
    return {
        hotkey.strip() for neuron in neurons
        if isinstance(hotkey := getattr(neuron, "hotkey", None), str) and hotkey.strip()
    }


class RegisteredHotkeys:
    def __init__(self, *, load: Callable[[], Awaitable[set[str]]],
                 clock: Callable[[], float] = time.time,
                 ttl_seconds: float = REGISTERED_HOTKEY_CACHE_TTL_SECONDS,
                 grace_seconds: float = REGISTERED_HOTKEY_STALE_GRACE_SECONDS) -> None:
        self._load = load
        self._clock = clock
        self._ttl = ttl_seconds
        self._grace = grace_seconds
        self._hotkeys: frozenset[str] | None = None
        self._refreshed_at: float | None = None

    async def reason(self, hotkey: str) -> str | None:
        if self._hotkeys is None or self._refreshed_at is None:
            return UNAVAILABLE
        age = self._clock() - self._refreshed_at
        if age > self._grace:
            return UNAVAILABLE
        if hotkey in self._hotkeys:
            return None
        # An old snapshot may predate this miner's registration.
        return UNAVAILABLE if age > self._ttl else NOT_REGISTERED

    async def refresh(self) -> bool:
        try:
            hotkeys = await asyncio.wait_for(
                self._load(), timeout=REGISTERED_HOTKEY_REFRESH_TIMEOUT_SECONDS
            )
            if not hotkeys:
                raise RuntimeError("the chain returned no registered hotkeys")
        except Exception as exc:
            logger.warning("corpus registration refresh failed: %r", exc)
            return False
        self._hotkeys = frozenset(hotkeys)
        self._refreshed_at = self._clock()
        return True

    async def refresh_forever(self) -> None:
        while True:
            ok = await self.refresh()
            await asyncio.sleep(REFRESH_SECONDS if ok else RETRY_SECONDS)
