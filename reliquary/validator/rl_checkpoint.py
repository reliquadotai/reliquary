"""`GET /rl/checkpoint`: the trainer's checkpoint index, signed with the
validator hotkey, plus short-lived presigned GET URLs into the private
``reliquary-public`` bucket (docs/rl-async-design.md §4.8).

The index is reread from the bucket at most every ``refresh_seconds`` and signed
once per distinct index. URLs are presigned per answer (a local HMAC, no call to
R2) and are not covered by the signature: what a miner downloads is checked
against the per-object SHA-256 in the signed index.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Callable

from reliquary.shared.ckpt_index import (
    INDEX_FORMAT,
    index_keys,
    index_signing_payload,
    sign_index,
)
from reliquary.shared.strict_json import strict_json_loads

logger = logging.getLogger(__name__)

PRESIGN_TTL_SECONDS = 900
REFRESH_SECONDS = 5.0


def presigner(client: Any, bucket: str, ttl: int = PRESIGN_TTL_SECONDS) -> Callable[[str], str]:
    def presign(key: str) -> str:
        return client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=int(ttl))
    return presign


def wallet_index_signer(wallet: Any) -> Callable[[bytes], bytes]:
    """Sign with the local wallet hotkey. The remote signer (reliquary/signer)
    only signs checkpoint payloads today; it needs an index operation first."""
    return lambda payload: bytes(wallet.hotkey.sign(payload))


class RlCheckpointBoard:
    def __init__(
        self,
        *,
        load_index: Callable[[], bytes | None],
        sign_fn: Callable[[bytes], bytes],
        validator_hotkey: str,
        presign: Callable[[str], str],
        ttl_seconds: int = PRESIGN_TTL_SECONDS,
        refresh_seconds: float = REFRESH_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._load, self._sign, self._presign = load_index, sign_fn, presign
        self.validator_hotkey = validator_hotkey
        self._ttl, self._refresh, self._clock = ttl_seconds, refresh_seconds, clock
        self._loaded_at = float("-inf")
        self._signed: tuple[str, dict, str] | None = None  # (payload digest, index, signature)
        self._lock = asyncio.Lock()

    async def _current(self) -> tuple[dict, str] | None:
        async with self._lock:
            if self._clock() - self._loaded_at >= self._refresh:
                try:
                    raw = await asyncio.to_thread(self._load)
                    self._loaded_at = self._clock()
                    if raw:
                        self._accept(strict_json_loads(raw))
                except Exception:
                    logger.warning("checkpoint index reload failed; serving the last one",
                                   exc_info=True)
            return None if self._signed is None else self._signed[1:]

    def _accept(self, index: dict) -> None:
        if index.get("format") != INDEX_FORMAT:
            raise ValueError("unknown checkpoint index format")
        if self._signed and index["latest"]["version"] < self._signed[1]["latest"]["version"]:
            raise ValueError("checkpoint index went backwards")
        digest = hashlib.sha256(index_signing_payload(index)).hexdigest()
        if self._signed is None or self._signed[0] != digest:
            self._signed = (digest, index, sign_index(index, self._sign))

    async def answer(self) -> dict | None:
        current = await self._current()
        if current is None:
            return None
        index, signature = current
        return {
            "index": index,
            "signature": signature,
            "validator_hotkey": self.validator_hotkey,
            "urls": {key: self._presign(key) for key in index_keys(index)},
            "urls_expire_at": int(self._clock()) + int(self._ttl),
        }


async def rl_checkpoint_response(board: RlCheckpointBoard | None):
    """Body of ``GET /rl/checkpoint`` (to be mounted next to ``/rl/pay``)."""
    from fastapi.responses import JSONResponse

    answer = None if board is None else await board.answer()
    if answer is None:
        return JSONResponse(status_code=503, content={"status": "unavailable"},
                            headers={"Retry-After": "5"})
    return JSONResponse(content=answer, headers={"Cache-Control": "private, no-store"})
