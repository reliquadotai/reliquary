"""The subnet's registered hotkeys, as the corpus route reads them.

Same rules as the RL validator's registration gate: requests only read a
snapshot, a background loop refreshes it from the chain, and a snapshot too old
to trust answers "unavailable" (retry) rather than "not registered" (refuse).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping

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


async def load_registered_hotkeys(netuid: int) -> dict[str, str | None]:
    """Each registered hotkey and its registration's coldkey (None if the chain gave none)."""
    from reliquary.infrastructure import chain

    subtensor = await chain.get_subtensor()
    try:
        neurons = await chain.get_neurons_lite(subtensor, netuid)
    finally:
        await chain.close_subtensor(subtensor)
    registered: dict[str, str | None] = {}
    for neuron in neurons:
        hotkey = getattr(neuron, "hotkey", None)
        if isinstance(hotkey, str) and hotkey.strip():
            coldkey = getattr(neuron, "coldkey", None)
            registered[hotkey.strip()] = coldkey.strip() if isinstance(coldkey, str) and coldkey.strip() else None
    return registered


class RegisteredHotkeys:
    def __init__(self, *, load: Callable[[], Awaitable[set[str] | Mapping[str, str | None]]],
                 clock: Callable[[], float] = time.time,
                 ttl_seconds: float = REGISTERED_HOTKEY_CACHE_TTL_SECONDS,
                 keep_owner_history: bool = False,
                 grace_seconds: float = REGISTERED_HOTKEY_STALE_GRACE_SECONDS) -> None:
        self._owner_history = deque(maxlen=8) if keep_owner_history else None
        self._load = load
        self._clock = clock
        self._ttl = ttl_seconds
        self._grace = grace_seconds
        self._hotkeys: frozenset[str] | None = None
        # Each hotkey's coldkey, from the last snapshot that gave them.
        self._coldkeys: dict[str, str] = {}
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
        if isinstance(hotkeys, Mapping):
            self._coldkeys = {hotkey: coldkey for hotkey, coldkey in hotkeys.items() if coldkey}
        self._refreshed_at = self._clock()
        if self._owner_history is not None:
            self._owner_history.append((self._refreshed_at, self._coldkeys if isinstance(hotkeys, Mapping) else None))
        return True

    def coldkey(self, hotkey: str) -> str | None:
        """The coldkey ``hotkey`` was registered under, as the last snapshot read it (None: unknown)."""
        return self._coldkeys.get(hotkey)

    def coldkeys(self) -> dict[str, str]:
        """Every hotkey's coldkey, as the last snapshot read them."""
        return dict(self._coldkeys)

    def owners_at(self, opened_at):
        """Newest chain read already completed at the opening second, within the cache TTL."""
        for observed_at, owners in reversed(self._owner_history or ()):
            if observed_at <= opened_at:
                if owners is not None and opened_at - observed_at <= self._ttl:
                    return observed_at, dict(owners)
                break
        return None

    async def refresh_forever(self) -> None:
        while True:
            ok = await self.refresh()
            await asyncio.sleep(REFRESH_SECONDS if ok else RETRY_SECONDS)
