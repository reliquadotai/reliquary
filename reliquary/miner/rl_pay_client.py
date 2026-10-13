"""Reference-miner environment weights from the validator's ``GET /rl/pay``.

A price only steers supply if miners look at it. The reference miner draws
its next environment with probability proportional to what the validator
advertises an environment pays per generated token: alpha per million paid
tokens times the share of generated groups that get paid (the geometric mean
of the published low/high range). The draw stays random, so miners spread
instead of all piling onto the single best environment.

``RELIQUARY_MINER_ENV_CHOICE=mix`` keeps today's draw by the static mix.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

ENV_CHOICE_VARIABLE = "RELIQUARY_MINER_ENV_CHOICE"
_MODES = ("price", "mix")
CACHE_SECONDS = 60.0
FETCH_TIMEOUT_SECONDS = 5.0


def env_choice_mode(env: Mapping[str, str] = os.environ) -> str:
    value = str(env.get(ENV_CHOICE_VARIABLE, "price")).strip().lower()
    if value not in _MODES:
        raise ValueError(f"{ENV_CHOICE_VARIABLE} must be one of {_MODES}, got {value!r}")
    return value


def weights_from_pay(body: Any) -> dict[str, float] | None:
    """Per-environment draw weights, or None when the answer cannot steer."""
    environments = body.get("environments") if isinstance(body, Mapping) else None
    if not isinstance(environments, Mapping):
        return None
    weights: dict[str, float] = {}
    for name, entry in environments.items():
        try:
            rate = float(entry["now"]["alpha_per_mtoken_paid"])
            paid = entry["market"]["paid_fraction_of_submitted"]
            fraction = math.sqrt(float(paid["low"]) * float(paid["high"]))
        except (KeyError, TypeError, ValueError):
            continue
        weight = rate * fraction
        if math.isfinite(weight) and weight > 0:
            weights[str(name)] = weight
    return weights or None


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
    """

    def __init__(self, base_url: str, *, fetch: Callable[[str], Any] = _http_get_json,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._url = base_url.rstrip("/") + "/rl/pay"
        self._fetch, self._clock = fetch, clock
        self._value: dict[str, float] | None = None
        self._at: float | None = None

    def weights(self) -> dict[str, float] | None:
        now = self._clock()
        if self._at is not None and now - self._at < CACHE_SECONDS:
            return self._value
        self._at = now
        try:
            self._value = weights_from_pay(self._fetch(self._url))
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the mix
            logger.debug("rl pay unavailable (%r); drawing environments by the mix", exc)
            self._value = None
        return self._value
