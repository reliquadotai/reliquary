"""Miner reference client for the published run observation log, and the default prompt policy."""
import asyncio
import json
import logging
import threading
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair

from reliquary.constants import M_ROLLOUTS
from reliquary.services import publication
from reliquary.services.publication import ObservationPublisher, index_key, page_key
from tests.unit.service_v2_fixtures import MATH

BASE = "https://observations.example/"
RUN = "r1"
POOL = 2 * M_ROLLOUTS


def keypair():
    return Keypair.create_from_seed("0x" + "07" * 32)


class FakeR2:
    """A bucket: the real publisher writes to it, the client reads from it through ``fetch``."""

    def __init__(self):
        self.objects, self.fetched = {}, []

    async def put(self, key, body, content_type, cache_control):
        self.objects[key] = body

    def fetch(self, url):
        assert url.startswith(BASE)
        self.fetched.append(url[len(BASE):])
        try:
            return self.objects[url[len(BASE):]]
        except KeyError:
            raise OSError("404") from None

    def count(self, key):
        return self.fetched.count(key)


def publish_all(rt, bucket, key=None, clock=lambda: 1000.7):
    pub = ObservationPublisher(rt, run_id=RUN, task_id="t", wallet=SimpleNamespace(hotkey=key or keypair()),
                               put=bucket.put, clock=clock)
    while asyncio.run(pub.flush()) is not None:
        pass
    return pub


def small_pages(monkeypatch, page=3):
    monkeypatch.setattr(publication, "PAGE_SEGMENTS", page)
    monkeypatch.setattr(publication, "SERVICE_OBSERVATION_SEGMENT_MAX_EVENTS", 1)  # one event per segment


def log_events(rt):
    return [e for _, e in rt.events(limit=10_000)]


def last_settles(events):
    return {e["id"]: e for e in events if e["type"] == "settle"}


# ------------------------------------------------------------------ reconstruction

# ------------------------------------------------------------------ refusals

# ------------------------------------------------------------------ restart, polling

# ------------------------------------------------------------------ table semantics

def obs(id_, prompt=1, *, lane="training", verdict="in-zone", status="proven", proof="proven", window=1,
        checkpoint_n=1, rewards=None, seeds=None, pool="p" * 64, uncertain=None, env="math"):
    e = {"type": "observation", "id": id_, "env": env, "prompt_idx": prompt, "window": window,
         "checkpoint_n": checkpoint_n, "lane": lane, "verdict": verdict, "status": status, "proof": proof,
         "rewards_bps": rewards or [0] * M_ROLLOUTS}
    if seeds is not None:
        e["candidate"] = {"pool_sha256": pool, "seeds": seeds}
    if uncertain:
        e["uncertain"] = uncertain
    return e


def settle(id_, status, proof, **kw):
    return {"type": "settle", "id": id_, "status": status, "proof": proof, **kw}


# ------------------------------------------------------------------ policy

# ------------------------------------------------------------------ engine

class _Rng:
    def getrandbits(self, n):
        return 7


# ------------------------------------------------------------------ fix round 1

def test_f2_a_decompression_bomb_is_refused_by_verify_segment():
    import gzip as gz
    import hashlib

    def entry_of(raw):
        return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    bomb = gz.compress(b"0" * 100_000, mtime=0)
    with pytest.raises(ValueError, match="size limit"):
        publication.verify_segment(bomb, entry_of(bomb), max_size=1000)
    small = gz.compress(b'{"a":1}\n', mtime=0)
    assert publication.verify_segment(small, entry_of(small), max_size=8) == [{"a": 1}]
    truncated = bomb[:-8]
    with pytest.raises(ValueError):
        publication.verify_segment(truncated, entry_of(truncated))


