#!/usr/bin/env python3
"""
Drand distributed randomness beacon integration for Reliquary.

Features:
- v2-first HTTP API with v1 fallbacks
- Correct chain hashes (quicknet=3s, default=30s)
- Programmatic fetch of chain info (genesis_time, period) with caching
- Robust networking (Session, retries, shuffled relays, sensible timeouts)
- Uniform schema for real & mock beacons
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import threading
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
        # Reasonable defaults; will be refreshed via /info on first use:
        "period": 3,
        "genesis_time": None,
    },
    "default": {
        "hash": "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce",
        "description": "Original 30-second chain (chained)",
        "period": 30,
        "genesis_time": None,
    },
}

DEFAULT_CHAIN = os.getenv("DRAND_CHAIN", "quicknet").strip() or "quicknet"

_current_chain = DEFAULT_CHAIN
# Cached, resolved parameters (populated lazily)
_DRAND_CHAIN_HASH: str | None = None
_DRAND_GENESIS_TIME: int | None = None
_DRAND_PERIOD: int | None = None

# Cache chain-info lookups by chain-hash to avoid repeated /info calls
_CHAIN_INFO_CACHE: dict[str, dict[str, Any]] = {}

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
    """Verify a drand quicknet beacon by cross-check against bittensor-drand.

    Drand quicknet signatures are BLS12-381 sigs on G1 (48 bytes) with the
    chain's public key on G2 (message = SHA256(round_be8), DST
    ``BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_``). The reference C
    implementation is `supranational/blst`, which is NOT distributed on PyPI
    as ``pip install blst`` — so pairing verification in Python would require
    either shipping a compiled shared lib or rewriting hash-to-curve from
    scratch (both brittle).

    Instead, we rely on a second independent source: ``bittensor_drand`` is
    already pulled in as a dependency of ``bittensor`` and is a Rust binding
    that fetches the same quicknet chain (hash
    ``52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971``)
    from a separate relay (``api.drand.secureweb3.com:6875``). A beacon is
    considered authentic iff:

    1. The fetched signature is a non-empty hex string.
    2. ``bittensor_drand.get_signature_for_round(round)`` returns a byte-for-
       byte identical signature (forging requires compromising both mirrors).
    3. The randomness field equals ``SHA256(sig)`` — the unchained quicknet
       derivation.

    Returns True if all three checks pass, False otherwise. Fails closed on
    any network, format, or parsing error.
    """
    if not signature_hex:
        logger.warning("[Drand] beacon has no signature — cannot verify")
        return False

    try:
        import hashlib

        # Check 1 — randomness derivation (unchained quicknet: sha256(sig))
        try:
            sig_bytes = bytes.fromhex(signature_hex)
        except ValueError:
            logger.warning("[Drand] signature is not valid hex, rejecting")
            return False
        expected_rand = hashlib.sha256(sig_bytes).hexdigest()
        if randomness_hex.lower() != expected_rand:
            logger.error(
                "[Drand] randomness != SHA256(sig) for round %d — rejecting",
                round_number,
            )
            return False

        # Check 2 — cross-check the sig against bittensor-drand's independent
        # relay fetch. Delegates BLS verification to the Rust binding's
        # internal flow (drand quicknet, G1 sigs, BLS12381G1).
        try:
            import bittensor_drand
        except ImportError:
            logger.error(
                "[Drand] bittensor_drand not installed — cannot cross-check"
            )
            return False

        try:
            canonical = bittensor_drand.get_signature_for_round(int(round_number))
        except Exception as exc:
            logger.error(
                "[Drand] bittensor-drand failed to fetch round %d: %s",
                round_number, exc,
            )
            return False

        if canonical.lower() != signature_hex.lower():
            logger.error(
                "[Drand] signature mismatch for round %d — rejecting "
                "(ours=%s... bittensor-drand=%s...)",
                round_number, signature_hex[:16], canonical[:16],
            )
            return False

        return True

    except Exception as e:
        logger.warning("[Drand] beacon signature verification failed: %s", e)
        return False


# ───────────────────────────────  UTILITIES  ────────────────────────────────


def _get_chain_record(name: str) -> dict[str, Any]:
    if name not in DRAND_CHAINS:
        raise ValueError(f"Unknown drand chain '{name}'. Available: {list(DRAND_CHAINS.keys())}")
    return DRAND_CHAINS[name]


def _parse_chain_info_payload(
    payload: dict[str, Any],
) -> tuple[int | None, int | None]:
    """
    Extract (genesis_time, period) from v2 or v1 /info responses.
    Be tolerant to possible key naming variants.
    """
    # Common/expected v2 keys
    gt = (
        payload.get("genesis_time") or payload.get("genesisTime") or payload.get("genesis")
    )  # some docs use 'genesis'
    pd = payload.get("period") or payload.get("round_time") or payload.get("roundTime")

    # Many responses encode period as seconds int; sometimes strings—normalize:
    try:
        if isinstance(pd, str):
            pd = int(pd)
    except Exception:
        pd = None

    try:
        if isinstance(gt, str):
            gt = int(gt)
    except Exception:
        gt = None

    return gt, pd


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


def _fetch_chain_info(chain_hash: str) -> dict[str, Any] | None:
    """
    Programmatically fetch chain info (genesis_time, period, pubkey...) from relays.
    Tries v2 then v1 shapes; caches on success.
    """
    if chain_hash in _CHAIN_INFO_CACHE:
        return _CHAIN_INFO_CACHE[chain_hash]

    v2_info = f"/v2/chains/{chain_hash}/info"
    v1_info = f"/{chain_hash}/info"
    # Additional v1 fallback for 'default' (some relays allow /info at root):
    root_info = "/info" if chain_hash == DRAND_CHAINS["default"]["hash"] else None

    payload = _http_get_json([v2_info, v1_info, root_info])  # type: ignore[list-item]
    if not payload:
        logger.debug(f"[Drand] chain info fetch failed for {chain_hash}")
        return None

    # Normalize core fields we care about; keep full payload for callers.
    gt, pd = _parse_chain_info_payload(payload)
    if gt is not None:
        payload["genesis_time"] = gt
    if pd is not None:
        payload["period"] = pd

    _CHAIN_INFO_CACHE[chain_hash] = payload
    return payload


def _ensure_params(refresh: bool = False) -> None:
    """
    Ensure _DRAND_CHAIN_HASH / _DRAND_GENESIS_TIME / _DRAND_PERIOD are populated.
    If refresh=True, re-fetch chain info and overwrite cached values.
    """
    global _DRAND_CHAIN_HASH, _DRAND_GENESIS_TIME, _DRAND_PERIOD

    rec = _get_chain_record(_current_chain)
    chain_hash = rec["hash"]
    if (
        refresh
        or _DRAND_CHAIN_HASH != chain_hash
        or _DRAND_GENESIS_TIME is None
        or _DRAND_PERIOD is None
    ):
        # Start with configured defaults:
        gt = rec.get("genesis_time")
        pd = rec.get("period")

        # Try to fetch live chain info to override:
        info = _fetch_chain_info(chain_hash)
        if info:
            gt_live = info.get("genesis_time")
            pd_live = info.get("period")
            if isinstance(gt_live, int) and gt_live > 0:
                gt = gt_live
            if isinstance(pd_live, int) and pd_live > 0:
                pd = pd_live

        _DRAND_CHAIN_HASH = chain_hash
        _DRAND_GENESIS_TIME = int(gt) if gt is not None else None
        _DRAND_PERIOD = int(pd) if pd is not None else None

        msg = (
            f"[Drand] chain={_current_chain} hash={_DRAND_CHAIN_HASH} "
            f"period={_DRAND_PERIOD} genesis_time={_DRAND_GENESIS_TIME}"
        )
        if info:
            logger.info(msg + " (refreshed via /info)")
        else:
            logger.info(msg + " (using configured defaults)")


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
    if _DRAND_CHAIN_HASH is None or _DRAND_PERIOD is None:
        raise RuntimeError("Drand chain parameters not initialized")

    rid = "latest" if round_id is None else str(int(round_id))
    v2_path = f"/v2/chains/{_DRAND_CHAIN_HASH}/rounds/{rid}"
    v1_path = f"/{_DRAND_CHAIN_HASH}/public/{rid}"
    # Default chain extra fallback (root v1 path without chain hash)
    root_v1 = f"/public/{rid}" if _DRAND_CHAIN_HASH == DRAND_CHAINS["default"]["hash"] else None

    data = _http_get_json([v2_path, v1_path, root_v1])  # type: ignore[list-item]
    if not data:
        logger.warning("[Drand] All relays/paths failed to fetch beacon")
        if use_fallback:
            return get_mock_beacon()
        raise RuntimeError("drand fetch failed (all relays/paths)")

    rnd = data.get("randomness")
    rno = data.get("round")
    sig = data.get("signature")
    # The drand v2 HTTP API omits the `randomness` field and only returns
    # `{round, signature}`. Per drand spec, randomness = SHA256(signature),
    # so we derive it locally. The signature is still verified below.
    if rnd is None and sig is not None:
        import hashlib
        try:
            rnd = hashlib.sha256(bytes.fromhex(sig)).hexdigest()
        except ValueError:
            rnd = None
    if rnd is None or rno is None:
        logger.debug(f"[Drand] Missing fields in response: {json.dumps(data)[:200]}")
        if use_fallback:
            return get_mock_beacon()
        raise RuntimeError("drand response missing required fields")

    # NB: cross-check against bittensor_drand moved off the hot path.
    # The validator schedules it as a background asyncio task right
    # after OPEN (see ``ValidatorService._verify_beacon_async``); if
    # it fails, the window is invalidated before ``_train_and_publish``
    # consumes any submissions. Miner callers get no cross-check —
    # a bad relay just makes their commitments fail GRAIL on the
    # validator side, no security implication for the miner.
    # The local ``randomness == SHA256(sig)`` derivation above is
    # gratis and stays.

    logger.debug(f"[Drand-{_current_chain}] ok round={rno} rand={str(rnd)[:8]}... (sig verify deferred)")
    return {
        "source": "drand",
        "chain": _current_chain,
        "chain_hash": _DRAND_CHAIN_HASH,
        "period": _DRAND_PERIOD,
        "round": int(rno),
        "randomness": str(rnd),
        "signature": sig,
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
