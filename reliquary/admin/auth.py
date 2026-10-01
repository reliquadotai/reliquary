"""HMAC request signing for the admin service.

The signature is the hex HMAC-SHA256, keyed by ``RELIQUARY_ADMIN_SECRET``, of
``timestamp\\nnonce\\nMETHOD\\npath\\nsha256hex(body)``. A timestamp more than
``MAX_SKEW_SECONDS`` from ours is stale, and a nonce already accepted inside
that window is a replay, so two identical requests in one second still pass
under two nonces.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections.abc import Callable

MAX_SKEW_SECONDS = 300
TIMESTAMP_HEADER = "X-Reliquary-Timestamp"
NONCE_HEADER = "X-Reliquary-Nonce"
SIGNATURE_HEADER = "X-Reliquary-Signature"

_NONCE_RE = re.compile(r"^[0-9a-fA-F]{16,64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def sign_request(secret: bytes, timestamp: str, nonce: str, method: str, path: str,
                 body: bytes) -> str:
    message = "\n".join(
        (timestamp, nonce, method.upper(), path, hashlib.sha256(body).hexdigest())
    ).encode()
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


class HmacVerifier:
    """Checks one request; ``refusal`` names why it is refused, or None."""

    def __init__(self, secret: bytes, *, clock: Callable[[], float] = time.time,
                 max_skew_seconds: int = MAX_SKEW_SECONDS) -> None:
        if not secret:
            raise ValueError("the admin secret is empty")
        self._secret = secret
        self._clock = clock
        self._skew = max_skew_seconds
        # Accepted nonce -> when; a replay can only land inside the window.
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def refusal(self, method: str, path: str, timestamp: str | None, nonce: str | None,
                signature: str | None, body: bytes) -> str | None:
        if not timestamp or not nonce or not signature:
            return "missing_signature"
        if not _NONCE_RE.fullmatch(nonce):
            return "bad_nonce"
        try:
            stamp = int(timestamp)
        except ValueError:
            return "stale_timestamp"
        now = self._clock()
        if abs(now - stamp) > self._skew:
            return "stale_timestamp"
        if not _SIGNATURE_RE.fullmatch(signature):
            return "bad_signature"
        expected = sign_request(self._secret, timestamp, nonce, method, path, body)
        if not hmac.compare_digest(expected, signature.lower()):
            return "bad_signature"
        key = nonce.lower()
        with self._lock:
            for seen, at in list(self._seen.items()):
                if at < now - 2 * self._skew:
                    del self._seen[seen]
            if key in self._seen:
                return "replayed"
            self._seen[key] = now
        return None


__all__ = [
    "HmacVerifier",
    "MAX_SKEW_SECONDS",
    "NONCE_HEADER",
    "SIGNATURE_HEADER",
    "TIMESTAMP_HEADER",
    "sign_request",
]
