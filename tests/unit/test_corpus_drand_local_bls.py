"""drand rounds verified here, by BLS, taken from the first relay that
answers; a judge prefetches every round its pending records need, so a pass
finds them cached (prod 2026-10-03 02:30: decide=257 s of a 329 s pass)."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time

import pytest

from reliquary.corpus.audit_policy import AuditParams
from reliquary.infrastructure import drand
from reliquary.validator import corpus_auditor, corpus_validator
from tests.unit import corpus_judge_sim as sim


# -- BLS, locally -----------------------------------------------------------------


def test_a_signature_that_is_not_the_round_s_is_refused_offline():
    start = time.monotonic()
    assert drand.verify_round_signature(1000, "ab" * 48) is False
    assert drand.verify_round_signature(1000, "not hex") is False
    assert drand.verify_round_signature(1000, "") is False
    assert time.monotonic() - start < 1.0


@pytest.mark.skipif(not os.environ.get("RELIQUARY_TEST_QUICKNET_SIG"),
                    reason="a genuine quicknet round: RELIQUARY_TEST_QUICKNET_ROUND/_SIG")
def test_a_genuine_quicknet_signature_verifies_and_only_for_its_round():
    round_number = int(os.environ["RELIQUARY_TEST_QUICKNET_ROUND"])
    sig = os.environ["RELIQUARY_TEST_QUICKNET_SIG"]
    assert drand.verify_round_signature(round_number, sig) is True
    assert drand.verify_round_signature(round_number + 1, sig) is False


def test_the_check_is_the_round_s_time_lock(monkeypatch):
    """Encrypt to the round, open with the signature: only the round's own
    BLS signature (the identity key of that round) opens it."""
    import bittensor_drand

    calls = []

    def encrypt_at_round(data, round_number):
        calls.append(("encrypt", round_number))
        return b"ct:" + str(round_number).encode() + b":" + data, round_number

    def decrypt_with_signature(ct, sig):
        calls.append(("decrypt", sig))
        _, round_number, data = ct.split(b":", 2)
        if sig != f"{int(round_number):096x}":
            raise ValueError("decryption failed")
        return data

    monkeypatch.setattr(bittensor_drand, "encrypt_at_round", encrypt_at_round)
    monkeypatch.setattr(bittensor_drand, "decrypt_with_signature", decrypt_with_signature)
    assert drand.verify_round_signature(41, f"{41:096x}") is True
    assert drand.verify_round_signature(42, f"{41:096x}") is False


# -- one relay, fastest first ------------------------------------------------------------


class _Relays:
    def __init__(self, delays, forger=None):
        self.delays, self.forger, self.asked = delays, forger, []

    def get(self, url, timeout=None, headers=None):
        base = next(b for b in self.delays if url.startswith(b))
        self.asked.append(base)
        time.sleep(self.delays[base])
        round_number = int(url.rsplit("/", 1)[1])
        sig = f"{round_number + (7 if base == self.forger else 0):096x}"

        class _R:
            status_code = 200
            text = ""

            def json(_self):
                return {"round": round_number, "signature": sig}

        return _R()


@pytest.fixture
def relays(monkeypatch):
    def install(delays, **kw):
        fake = _Relays(delays, **kw)
        monkeypatch.setattr(drand, "DRAND_URLS", list(delays))
        monkeypatch.setattr(drand, "_get_thread_session", lambda: fake)
        monkeypatch.setattr(drand, "_ensure_params", lambda refresh=False: None)
        monkeypatch.setattr(drand, "_DRAND_CHAIN_HASH", "c" * 64)
        # Genuine for its round iff it spells it (stands in for the BLS check).
        monkeypatch.setattr(drand, "verify_round_signature",
                            lambda r, sig: sig == f"{r:096x}")
        return fake

    return install


def test_the_fastest_relay_s_verified_answer_is_taken(relays):
    relays({"https://cf": 0.01, "https://slow": 1.5})
    start = time.monotonic()
    beacon = drand.get_verified_beacon(41)
    assert time.monotonic() - start < 0.5
    assert beacon["randomness"] == hashlib.sha256(bytes.fromhex(f"{41:096x}")).hexdigest()


def test_a_lying_relay_however_fast_is_ignored(relays):
    relays({"https://liar": 0.0, "https://honest": 0.05}, forger="https://liar")
    assert drand.get_verified_beacon(41)["signature"] == f"{41:096x}"
    relays({"https://liar": 0.0}, forger="https://liar")
    assert drand.get_verified_beacon(41) is None


def test_the_corpus_beacon_prefers_the_verified_relay(relays, monkeypatch):
    relays({"https://cf": 0.0})
    monkeypatch.setattr(corpus_validator, "_BEACONS", {})
    monkeypatch.delenv("RELIQUARY_CORPUS_DRAND_CACHE", raising=False)
    monkeypatch.setattr(drand, "get_agreed_beacon",
                        lambda *a, **k: pytest.fail("agreement not needed"))
    expected = hashlib.sha256(bytes.fromhex(f"{41:096x}")).hexdigest()
    assert corpus_validator.drand_beacon(41) == expected


# -- the prefetcher ------------------------------------------------------------------


def _passes(prefetch: bool, caplog):
    """140k? No: 20k pending over 15 h, rounds at 2 s each (a slow relay day)."""
    params = AuditParams(q=0.15, probation_submissions=100, hold_seconds=4320.0,
                         ban_after_failures=1000)
    population = sim.Population.prod_like(seed=0, probation=0)
    population.tokens = 4200
    kwargs = {} if prefetch else {"prefetch_rounds": False}
    with caplog.at_level(logging.INFO, logger="reliquary.validator.corpus_auditor"):
        sim.simulate(corpus_auditor, rate_per_hour=6900.0, hours=1.5, params=params,
                     population=population, store=sim.Store(seed=0, latency=(0.05, 0.10),
                                                            write_latency=(0.2, 0.4)),
                     gpu=sim.Gpu(tokens_per_second=8200.0), beacon=sim.Beacon(latency=2.0),
                     start_backlog=20_000, backlog_age=15 * 3600.0, sample_every=600.0,
                     auditor_kwargs=kwargs)
    lines = [r.getMessage() for r in caplog.records]
    decide = [float(re.search(r"decide=([0-9.]+)s", l).group(1)) for l in lines
              if l.startswith("corpus judge pass: ") and "ids=512" in l]
    hits = [l for l in lines if l.startswith("corpus judge pass started")]
    caplog.clear()
    return decide, hits


def test_prefetched_rounds_make_a_pass_decide_in_seconds(caplog):
    decide, hits = _passes(True, caplog)
    later = decide[len(decide) // 3:]
    assert later and sorted(later)[len(later) // 2] < 20.0, decide
    assert hits and all("rounds_cached=" in l for l in hits)
    assert re.search(r"rounds_cached=([0-9]+)/([0-9]+)", hits[-1])


def test_without_the_prefetcher_a_pass_waits_on_its_rounds(caplog):
    decide, _ = _passes(False, caplog)
    later = decide[len(decide) // 3:]
    assert later and sorted(later)[len(later) // 2] > 20.0, decide
