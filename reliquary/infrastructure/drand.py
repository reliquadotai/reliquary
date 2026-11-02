#!/usr/bin/env python3
"""
Drand distributed randomness beacon integration for Reliquary.

Features:
- v2-first HTTP API with v1 fallbacks
- Correct chain hashes (quicknet=3s, default=30s)
- Pinned chain timing, independent of relay metadata
- Robust networking (Session, retries, shuffled relays, sensible timeouts)
- Uniform schema for real & mock beacons
"""

from __future__ import annotations

import concurrent.futures
import re
import logging
import os
import threading
import time
from threading import Lock
from typing import Any

import requests
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ──────────────────────────  RELAYS / NETWORKING  ──────────────────────────

DRAND_URLS = [
    "https://api.drand.sh",
    "https://api2.drand.sh",
    "https://api3.drand.sh",
    "https://drand.cloudflare.com",
    "https://api.drand.secureweb3.com:6875",
]

# Retries disabled: _http_get_json races relays in parallel so siblings
# provide the fallback that retries used to provide on the serial path.
# Keeping retries here would let a single bad relay multiply its per-request
# timeout by 4 (connect + 3 retries), stretching worst-case all-degraded
# latency from ~2s to ~11.5s and defeating the race.
_RETRY = Retry(
    total=0,
    connect=0,
    read=0,
    backoff_factor=0,
    status_forcelist=(),
    allowed_methods={"GET"},
    raise_on_status=False,
)
_SESSION = requests.Session()
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))
_HEADERS = {"User-Agent": "Reliquary-drand/0.2"}

# Per-thread sessions for _http_get_json's parallel race — avoids urllib3
# HTTPConnectionPool contention when a straggler on a dead relay holds a slot.
_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        # No retries — race siblings provide the fallback (matches _RETRY rationale).
        s.mount("https://", HTTPAdapter(max_retries=0))
        _thread_local.session = s
    return s

# ─────────────────────────────  CHAINS / STATE  ─────────────────────────────

# NOTE: hashes are authoritative identifiers of chains (v2 uses them in the path).
# quicknet: 3s, unchained; default: 30s, chained.
DRAND_CHAINS: dict[str, dict[str, Any]] = {
    "quicknet": {
        "hash": "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971",
        "description": "Fast 3-second randomness (unchained, recommended)",
        # Published chain timing is immutable; never override it from a relay.
        "period": 3,
        "genesis_time": 1692803367,
    },
    "default": {
        "hash": "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce",
        "description": "Original 30-second chain (chained)",
        "period": 30,
        "genesis_time": 1595431050,
    },
}

DEFAULT_CHAIN = os.getenv("DRAND_CHAIN", "quicknet").strip() or "quicknet"

_current_chain = DEFAULT_CHAIN
# Cached, resolved parameters (populated lazily)
_DRAND_CHAIN_HASH: str | None = None
_DRAND_GENESIS_TIME: int | None = None
_DRAND_PERIOD: int | None = None

# Thread-safety for globals and mock counter
_LOCK = Lock()
_BEACON_COUNTER = 0

# ─────────────────────────  BEACON SIGNATURE VERIFICATION  ──────────────────

def verify_beacon_signature(
    chain_hash: str,
    round_number: int,
    randomness_hex: str,
    signature_hex: str | None,
) -> bool:
    """Verify the pinned quicknet round and SHA256(signature), offline."""
    import hashlib

    if chain_hash != DRAND_CHAINS["quicknet"]["hash"]:
        return False
    point = g1_signature(signature_hex)
    if point is None or not isinstance(randomness_hex, str):
        return False
    if randomness_hex.lower() != hashlib.sha256(point).hexdigest():
        return False
    return verify_round_signature(round_number, signature_hex)


# ───────────────────────────────  UTILITIES  ────────────────────────────────


def _get_chain_record(name: str) -> dict[str, Any]:
    if name not in DRAND_CHAINS:
        raise ValueError(f"Unknown drand chain '{name}'. Available: {list(DRAND_CHAINS.keys())}")
    return DRAND_CHAINS[name]


def _http_get_json(paths: list[str]) -> dict[str, Any] | None:
    """Race all relays in parallel for the first valid JSON response.

    For each path (in order), fire all relays simultaneously and return
    as soon as any one yields a parsable 200 JSON body. If every relay
    fails or returns a non-200 for the current path, fall through to
    the next path. Returns None if no relay×path combination produces
    a parsable 200.

    Timeouts are tight (connect=0.5s, read=2.0s) because the other
    relays in the race act as the fallback — we don't need a long
    per-relay budget. Per-relay retries are disabled (see _RETRY); the
    race siblings provide the fallback, keeping worst-case wall-clock
    cost at ~2s even when every relay is degraded.
    """
    relays = list(DRAND_URLS)
    if not relays:
        return None

    for path in paths:
        if not path:
            continue
        urls = [f"{base}{path}" for base in relays]

        def _try(url: str) -> dict[str, Any] | None:
            try:
                r = _get_thread_session().get(url, timeout=(0.5, 2.0), headers=_HEADERS)
                if r.status_code != 200:
                    logger.debug(f"[Drand] GET {url} -> HTTP {r.status_code} {r.text[:160]!r}")
                    return None
                try:
                    return r.json()  # type: ignore[no-any-return]
                except Exception as e:
                    logger.debug(f"[Drand] JSON parse error for {url}: {e}; body[:160]={r.text[:160]!r}")
                    return None
            except Exception as e:
                logger.debug(f"[Drand] GET {url} error: {e}")
                return None

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=len(urls))
        futures = {ex.submit(_try, u): u for u in urls}
        # Don't block on __exit__; all futures are already running since
        # max_workers == len(urls), so cancel_futures=False is safe.
        ex.shutdown(wait=False, cancel_futures=False)
        winner = None
        for fut in concurrent.futures.as_completed(futures):
            try:
                payload = fut.result()
            except concurrent.futures.CancelledError:
                # Pending future was cancelled by shutdown — treat as non-winner.
                continue
            if payload is not None:
                winner = payload
                break
        if winner is not None:
            return winner
    return None


def _ensure_params(refresh: bool = False) -> None:
    """Resolve immutable published timing without trusting relay metadata."""
    global _DRAND_CHAIN_HASH, _DRAND_GENESIS_TIME, _DRAND_PERIOD

    rec = _get_chain_record(_current_chain)
    _DRAND_CHAIN_HASH = rec["hash"]
    _DRAND_GENESIS_TIME = rec["genesis_time"]
    _DRAND_PERIOD = rec["period"]


# ─────────────────────────────  PUBLIC API  ─────────────────────────────


def get_current_chain() -> dict[str, Any]:
    """
    Return details of the active chain, including resolved period/genesis_time if known.
    """
    _ensure_params(refresh=False)
    rec = _get_chain_record(_current_chain).copy()
    rec.update(
        {
            "name": _current_chain,
            "hash": _DRAND_CHAIN_HASH,
            "period": _DRAND_PERIOD,
            "genesis_time": _DRAND_GENESIS_TIME,
        }
    )
    return rec


_G1_HEX = re.compile(r"\A[0-9a-fA-F]{96}\Z")


def g1_signature(signature_hex: Any) -> bytes | None:
    """The 48 bytes of a quicknet signature (one compressed G1 point), or None
    for anything else: bytes appended or missing, whitespace, a 0x prefix.
    The BLS check reads only the first 48 bytes, so anything after them would
    let a relay choose SHA256(signature), the randomness."""
    if not isinstance(signature_hex, str) or not _G1_HEX.fullmatch(signature_hex):
        return None
    return bytes.fromhex(signature_hex)


def verify_round_signature(round_number: int, signature_hex: str | None) -> bool:
    """True iff ``signature_hex`` is quicknet's BLS signature of
    ``round_number``, checked here, offline, in ~7 ms.

    A quicknet round's signature is the identity key of that round in drand's
    time-lock scheme: a message encrypted to the round (``encrypt_at_round``,
    the chain's public key built into bittensor-drand) opens with it and with
    nothing else (``decrypt_with_signature`` fails on any other bytes, or on
    another round's signature). No relay has to be trusted.
    """
    import secrets

    if type(round_number) is not int or not 0 < round_number < 2**64:
        return False
    point = g1_signature(signature_hex)
    if point is None:
        return False
    try:
        import bittensor_drand

        nonce = secrets.token_bytes(16)
        sealed = bittensor_drand.encrypt_at_round(nonce, int(round_number))
        ciphertext = sealed[0] if isinstance(sealed, (tuple, list)) else sealed
        return bittensor_drand.decrypt_with_signature(ciphertext, point.hex()) == nonce
    except Exception:
        return False


def get_verified_beacon(round_id: int) -> dict[str, Any] | None:
    """Round ``round_id`` from the first relay whose answer verifies by BLS
    (``verify_round_signature``) and whose randomness is SHA256(signature):
    the fastest relay's answer (~60 ms), not the slowest of two. None when no
    relay gives a verifiable answer."""
    import hashlib

    if type(round_id) is not int or not 0 < round_id < 2**64:
        return None
    _ensure_params(refresh=False)
    if _DRAND_CHAIN_HASH != DRAND_CHAINS["quicknet"]["hash"] or not DRAND_URLS:
        return None
    rid = str(int(round_id))
    paths = [f"/v2/chains/{_DRAND_CHAIN_HASH}/rounds/{rid}", f"/{_DRAND_CHAIN_HASH}/public/{rid}"]
    urls = [f"{base}{path}" for base in DRAND_URLS for path in paths]

    def _try(url: str) -> dict[str, Any] | None:
        try:
            r = _get_thread_session().get(url, timeout=(0.5, 2.0), headers=_HEADERS)
            data = r.json() if r.status_code == 200 else None
        except Exception as e:
            logger.debug(f"[Drand] GET {url} error: {e}")
            return None
        if not isinstance(data, dict) or type(data.get("round")) is not int or data["round"] != int(round_id):
            return None
        point = g1_signature(data.get("signature"))
        if point is None:
            return None
        sig = point.hex()
        randomness = hashlib.sha256(point).hexdigest()
        given = data.get("randomness")
        if given is not None and str(given).lower() != randomness:
            return None
        if not verify_round_signature(int(round_id), sig):
            logger.warning("[Drand] %s gave a signature for round %s that does not verify",
                           url.split("/")[2], rid)
            return None
        return {"round": int(round_id), "signature": sig, "randomness": randomness}

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=len(urls))
    futures = [ex.submit(_try, url) for url in urls]
    ex.shutdown(wait=False, cancel_futures=False)
    for fut in concurrent.futures.as_completed(futures):
        try:
            beacon = fut.result()
        except concurrent.futures.CancelledError:
            continue
        if beacon is not None:
            return beacon
    return None


def get_agreed_beacon(round_id: int, *, agree: int = 2) -> dict[str, Any] | None:
    """Round ``round_id`` as ``agree`` distinct relays sign it, or None.

    Every relay is asked at once (v2 and v1 paths); the round is returned as
    soon as ``agree`` relays gave the same signature for it and randomness =
    SHA256(signature) holds, so its cost is the second-fastest relay's answer
    (cloudflare's is ~60 ms from the corpus box). Local BLS verification is
    required as well. None: no verified agreement; the caller falls back.
    """
    import hashlib

    if type(round_id) is not int or not 0 < round_id < 2**64:
        return None
    _ensure_params(refresh=False)
    if _DRAND_CHAIN_HASH != DRAND_CHAINS["quicknet"]["hash"] or not DRAND_URLS:
        return None
    rid = str(int(round_id))
    paths = [f"/v2/chains/{_DRAND_CHAIN_HASH}/rounds/{rid}", f"/{_DRAND_CHAIN_HASH}/public/{rid}"]
    jobs = [(base, f"{base}{path}") for base in DRAND_URLS for path in paths]

    def _try(url: str) -> dict[str, Any] | None:
        try:
            r = _get_thread_session().get(url, timeout=(0.5, 2.0), headers=_HEADERS)
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            logger.debug(f"[Drand] GET {url} error: {e}")
            return None

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs))
    futures = {ex.submit(_try, url): base for base, url in jobs}
    ex.shutdown(wait=False, cancel_futures=False)
    votes: dict[str, set[str]] = {}
    for fut in concurrent.futures.as_completed(futures):
        try:
            data = fut.result()
        except concurrent.futures.CancelledError:
            continue
        if not isinstance(data, dict) or type(data.get("round")) is not int or data["round"] != int(round_id):
            continue
        point = g1_signature(data.get("signature"))
        if point is None:
            continue
        sig = point.hex()
        randomness = hashlib.sha256(point).hexdigest()
        given = data.get("randomness")
        if given is not None and str(given).lower() != randomness:
            continue
        relays = votes.setdefault(sig.lower(), set())
        relays.add(futures[fut])
        if len(relays) >= agree and verify_round_signature(int(round_id), sig):
            return {"round": int(round_id), "signature": sig.lower(), "randomness": randomness}
    return None


def get_drand_beacon(round_id: int | None = None, use_fallback: bool = False) -> dict[str, Any]:
    """
    Fetch randomness from the drand network (v2-first, v1 fallback).

    Returns:
        {
            "source": "drand",
            "chain": <name>,
            "chain_hash": <hex>,
            "period": <int seconds>,
            "round": <int>,
            "randomness": <hex str>,
            "signature": <hex str | None>,
            "previous_signature": <hex str | None>,
        }
    """
    _ensure_params(refresh=False)
    if _DRAND_CHAIN_HASH != DRAND_CHAINS["quicknet"]["hash"]:
        raise RuntimeError("Only quicknet has an offline signature verifier")

    if round_id is None:
        round_id = int((time.time() - _DRAND_GENESIS_TIME) // _DRAND_PERIOD) + 1
    if type(round_id) is not int or not 0 < round_id < 2**64:
        raise ValueError("drand round must be a positive unsigned 64-bit integer")
    rid = str(int(round_id))
    v2_path = f"/v2/chains/{_DRAND_CHAIN_HASH}/rounds/{rid}"
    v1_path = f"/{_DRAND_CHAIN_HASH}/public/{rid}"
    data = _http_get_json([v2_path, v1_path])
    if not isinstance(data, dict) or not data:
        logger.warning("[Drand] All relays/paths failed to fetch beacon")
        if use_fallback:
            return get_mock_beacon()
        raise RuntimeError("drand fetch failed (all relays/paths)")

    rnd = data.get("randomness")
    rno = data.get("round")
    sig = data.get("signature")
    if (type(rno) is not int or (round_id is not None and rno != int(round_id))
            or not verify_round_signature(rno, sig)):
        raise RuntimeError("drand response failed local round verification")
    import hashlib

    expected_randomness = hashlib.sha256(bytes.fromhex(sig)).hexdigest()
    # The v2 API omits randomness; derive it only from the verified point.
    if rnd is None:
        rnd = expected_randomness
    if not isinstance(rnd, str) or rnd.lower() != expected_randomness:
        raise RuntimeError("drand response has invalid randomness")

    logger.debug(f"[Drand-{_current_chain}] ok round={rno} rand={rnd[:8]}... (locally verified)")
    return {
        "source": "drand",
        "chain": _current_chain,
        "chain_hash": _DRAND_CHAIN_HASH,
        "period": _DRAND_PERIOD,
        "round": int(rno),
        "randomness": expected_randomness,
        "signature": sig.lower(),
        "previous_signature": data.get("previous_signature"),
    }


def get_mock_beacon() -> dict[str, Any]:
    """
    Fallback mock beacon for testing/development; uniform schema.
    """
    global _BEACON_COUNTER
    _ensure_params(refresh=False)
    with _LOCK:
        _BEACON_COUNTER += 1
        rno = _BEACON_COUNTER
    rnd = os.urandom(32).hex()
    logger.debug(f"[MockBeacon] round={rno} randomness={rnd[:8]}...")
    return {
        "source": "mock",
        "chain": _current_chain,
        "chain_hash": _DRAND_CHAIN_HASH,
        "period": _DRAND_PERIOD,
        "round": rno,
        "randomness": rnd,
        "signature": None,
        "previous_signature": None,
    }


def get_beacon(
    round_id: str = "latest", use_drand: bool = True, use_fallback: bool = False
) -> dict[str, Any]:
    """
    Convenience wrapper:
      - round_id: "latest" or round number as string/int
      - use_drand=False forces mock (testing only)
      - use_fallback=False (default) raises on network errors
    """
    if not use_drand:
        return get_mock_beacon()
    try:
        rid = None if str(round_id) == "latest" else int(round_id)
        return get_drand_beacon(rid, use_fallback=use_fallback)
    except Exception as e:
        logger.warning(f"[Drand] get_beacon error: {e}")
        if use_fallback:
            return get_mock_beacon()
        raise


# ───────────────────────────────  BOOTSTRAP  ───────────────────────────────

# Honor DRAND_CHAIN env var at import time and warm up parameters.
try:
    _ensure_params(refresh=True)
except Exception as _e:
    # Don't crash imports; we will retry on first call.
    logger.debug(f"[Drand] initial chain-info refresh failed: {_e}")
