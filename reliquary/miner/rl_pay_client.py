"""Reference-miner environment weights from the validator's ``GET /rl/pay``.

A price only steers supply if miners look at it. The reference miner draws
its next environment with probability proportional to what the validator
advertises an environment pays per generated token; an environment not priced
yet gets the median weight. The draw stays random, so miners spread instead of
all piling onto the single best environment.

Two board formats: the async validator's (``rl_ledger.RlPayLedger.board``:
per env ``token_price`` and ``recent``, the share of its recent groups trained
and the share that were paid first scans; a group pays about ``token_price``
per token when trained and a quarter of that as a first scan) and the window
validator's (alpha per million paid tokens times the middle of the paid
fraction's low/high range).

``RELIQUARY_MINER_ENV_CHOICE=mix`` keeps today's draw by the static mix.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

ENV_CHOICE_VARIABLE = "RELIQUARY_MINER_ENV_CHOICE"
_MODES = ("price", "mix")
CACHE_SECONDS = 60.0
FIRST_SCAN_FRACTION = 0.25        # a first scan pays a quarter of a trained group (rl_ledger)
UNPAID_EXPLORATION = 0.05         # weight of an env whose recent groups paid nothing, relative to the median
FETCH_TIMEOUT_SECONDS = 5.0


def env_choice_mode(env: Mapping[str, str] = os.environ) -> str:
    value = str(env.get(ENV_CHOICE_VARIABLE, "price")).strip().lower()
    if value not in _MODES:
        raise ValueError(f"{ENV_CHOICE_VARIABLE} must be one of {_MODES}, got {value!r}")
    return value


def weights_from_pay(body: Any) -> dict[str, float] | None:
    """Per-environment draw weights, or None when the answer cannot steer.

    An environment the board cannot price yet (no paid group in its window
    history: new, or starved) gets the median of the priced weights, so a
    rising price can still draw miners to it. The paid fraction is the middle
    of its low/high range: a low bound of 0 must not zero a paying env.
    """
    environments = body.get("environments") if isinstance(body, Mapping) else None
    if not isinstance(environments, Mapping):
        return None
    weights: dict[str, float] = {}
    unpriced: list[str] = []
    unpaid: list[str] = []          # async: priced, but its recent groups paid nothing
    for name, entry in environments.items():
        try:
            if isinstance(entry, Mapping) and "token_price" in entry:     # the async board
                recent = entry.get("recent") or {}
                if not recent.get("groups"):
                    raise KeyError("no recent group")
                rate = float(entry["token_price"])
                fraction = float(recent["trained_share"]) + FIRST_SCAN_FRACTION * float(recent["first_scan_share"])
            else:
                rate = float(entry["now"]["alpha_per_mtoken_paid"])
                paid = entry["market"]["paid_fraction_of_submitted"]
                fraction = (float(paid["low"]) + float(paid["high"])) / 2.0
        except (KeyError, TypeError, ValueError):
            unpriced.append(str(name))
            continue
        weight = rate * fraction
        if math.isfinite(weight) and weight > 0:
            weights[str(name)] = weight
        elif isinstance(entry, Mapping) and "token_price" in entry and math.isfinite(weight):
            unpaid.append(str(name))
        else:
            unpriced.append(str(name))
    if not weights:
        return None
    middle = statistics.median(weights.values())
    for name in unpriced:
        weights[name] = middle
    for name in unpaid:
        weights[name] = UNPAID_EXPLORATION * middle    # still sampled now and then: its pay can come back
    return weights


def _http_get_json(url: str) -> Any:
    import httpx

    response = httpx.get(url, timeout=FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


class PayWeights:
    """The validator's advertised pay as draw weights, refreshed every minute.

    Any failure (route missing, validator down, unpriced answer) yields None,
    which every caller treats as "draw by the static mix". A failure is cached
    like a success, so a down validator costs one request per minute.

    ``background=True`` (the default) never blocks the caller: a due refresh
    runs on a daemon thread and ``weights()`` returns the last value at once
    (None before the first fetch lands). The episode loop calls it on its
    event loop, where a slow validator must not stall every stream.
    """

    def __init__(self, base_url: str, *, fetch: Callable[[str], Any] = _http_get_json,
                 clock: Callable[[], float] = time.monotonic, background: bool = True) -> None:
        self._url = base_url.rstrip("/") + "/rl/pay"
        self._fetch, self._clock, self._background = fetch, clock, background
        self._value: dict[str, float] | None = None
        self._at: float | None = None
        self._thread: threading.Thread | None = None

    def _refresh(self) -> None:
        try:
            self._value = weights_from_pay(self._fetch(self._url))
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the mix
            logger.debug("rl pay unavailable (%r); drawing environments by the mix", exc)
            self._value = None

    def weights(self) -> dict[str, float] | None:
        now = self._clock()
        if self._at is not None and now - self._at < CACHE_SECONDS:
            return self._value
        if self._thread is not None and self._thread.is_alive():
            return self._value
        self._at = now
        if not self._background:
            self._refresh()
            return self._value
        self._thread = threading.Thread(target=self._refresh, name="rl-pay-weights", daemon=True)
        self._thread.start()
        return self._value

    def join(self, timeout: float | None = None) -> None:
        """Wait for a refresh in flight (tests, shutdown)."""
        if self._thread is not None:
            self._thread.join(timeout)
