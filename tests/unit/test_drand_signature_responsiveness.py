"""Offline native verification leaves the serving loop available."""

import asyncio
import hashlib
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from reliquary.infrastructure import drand
from reliquary.validator.service import ValidationService

ROUND = 32_700_000
SIGNATURE = ("afaef99ecc0a56993fc66f032403971368456f07f3d727bdcd365af742331a2c"
             "57ffad1c5ad3052166b4beaa7ea7727c")
RANDOMNESS = hashlib.sha256(bytes.fromhex(SIGNATURE)).hexdigest()
CHAIN_HASH = drand.DRAND_CHAINS["quicknet"]["hash"]


@pytest.fixture(autouse=True)
def no_network_or_child_process(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("offline verification used a network or child process")

    monkeypatch.setattr(drand.requests.Session, "get", unexpected)
    monkeypatch.setattr(socket, "create_connection", unexpected)
    monkeypatch.setattr(subprocess, "run", unexpected)


@pytest.mark.asyncio
async def test_native_verification_keeps_the_serving_heartbeat_responsive():
    gaps = []
    done = threading.Event()

    async def heartbeat():
        previous = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.002)
            now = time.monotonic()
            gaps.append(now - previous)
            previous = now

    pulse = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.01)
    try:
        for _ in range(16):
            assert await asyncio.to_thread(drand.verify_beacon_signature, CHAIN_HASH,
                                           ROUND, RANDOMNESS, SIGNATURE)
    finally:
        done.set()
        await pulse
    assert gaps and max(gaps) < 0.15


@pytest.mark.parametrize("round_number,expected", [(ROUND, True), (ROUND + 1, False)])
def test_verification_needs_no_relay_or_helper(round_number, expected):
    assert drand.verify_beacon_signature(CHAIN_HASH, round_number, RANDOMNESS, SIGNATURE) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [False, True])
async def test_async_verification_invalidates_only_an_unverified_beacon(valid):
    svc = SimpleNamespace(_window_n=41)
    batchers = [SimpleNamespace(beacon_invalid=False), SimpleNamespace(beacon_invalid=False)]
    await ValidationService._verify_beacon_async(svc, batchers, CHAIN_HASH,
                                                ROUND if valid else ROUND + 1,
                                                RANDOMNESS, SIGNATURE)
    assert all(b.beacon_invalid is not valid for b in batchers)
