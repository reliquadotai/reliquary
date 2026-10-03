"""A math pass of 512 ids in about two minutes (prod 2026-10-03 00:32-01:21:
14-31 min each, the GPU mostly idle).

- The backward audit of a caught hotkey takes its pending records from
  memory, not from a listing of the job (670-850 s a pass at 146k pending).
- The failures of a pass are re-audited in one call, not one call each.
- A drand round is taken from two relays that agree (the fastest answer in
  ~60 ms), cached for the process and on disk; the slow cross-check fetch
  is the fallback only."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time

import pytest

from reliquary.corpus.audit_policy import AuditParams, MinerState
from reliquary.infrastructure import drand
from reliquary.validator import corpus_auditor, corpus_validator
from reliquary.validator.corpus_miner_states import MinerStates
from tests.unit import corpus_judge_sim as sim
from tests.unit import test_corpus_judge_equivalence as equivalence
from tests.unit import _main_corpus_auditor as main_module


# -- drand: two agreeing relays, cached -----------------------------------------


def _sig(round_number, forged=False):
    return hashlib.sha256(f"sig-{round_number}-{forged}".encode()).hexdigest() * 3


class _Relays:
    """Each relay answers after its own delay; ``forger`` signs differently."""

    def __init__(self, delays, forger=None, down=()):
        self.delays, self.forger, self.down = delays, forger, set(down)
        self.calls = 0
        self.lock = threading.Lock()

    def get(self, url, timeout=None, headers=None):
        base = next(b for b in self.delays if url.startswith(b))
        with self.lock:
            self.calls += 1
        time.sleep(self.delays[base])
        round_number = int(url.rsplit("/", 1)[1])

        class _R:
            status_code = 404 if base in self.down else 200
            text = ""

            def json(_self):
                sig = _sig(round_number, forged=base == self.forger)
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
        monkeypatch.setattr(drand, "_DRAND_PERIOD", 3)
        # Their signatures are fakes: only agreement can vouch for them here.
        monkeypatch.setattr(drand, "get_verified_beacon", lambda round_id: None)
        return fake

    return install


def test_two_agreeing_relays_answer_at_the_speed_of_the_second_fastest(relays):
    relays({"https://a": 0.01, "https://b": 0.03, "https://c": 1.5, "https://d": 1.5})
    start = time.monotonic()
    beacon = drand.get_agreed_beacon(41)
    assert time.monotonic() - start < 0.5
    sig = _sig(41)
    assert beacon == {"round": 41, "signature": sig,
                      "randomness": hashlib.sha256(bytes.fromhex(sig)).hexdigest()}


def test_one_lying_relay_is_outvoted_and_two_disagreeing_alone_give_nothing(relays):
    relays({"https://a": 0.01, "https://b": 0.02, "https://c": 0.03}, forger="https://a")
    assert drand.get_agreed_beacon(41)["signature"] == _sig(41)
    relays({"https://a": 0.01, "https://b": 0.02}, forger="https://a")
    assert drand.get_agreed_beacon(41) is None


def test_a_single_answer_is_not_enough(relays):
    relays({"https://a": 0.01, "https://b": 0.01}, down={"https://b"})
    assert drand.get_agreed_beacon(41) is None


def test_the_corpus_beacon_is_cached_in_memory_and_on_disk(relays, tmp_path, monkeypatch):
    fake = relays({"https://a": 0.0, "https://b": 0.0})
    cache = tmp_path / "drand.jsonl"
    monkeypatch.setenv("RELIQUARY_CORPUS_DRAND_CACHE", str(cache))
    monkeypatch.setattr(corpus_validator, "_BEACONS", {})
    first = corpus_validator.drand_beacon(41)
    calls = fake.calls
    assert corpus_validator.drand_beacon(41) == first and fake.calls == calls
    # A new process (empty memory) reads the file, not the network.
    monkeypatch.setattr(corpus_validator, "_BEACONS", {})
    monkeypatch.setattr(corpus_validator, "_DISK_LOADED", set())
    assert corpus_validator.drand_beacon(41) == first and fake.calls == calls
    assert json.loads(cache.read_text().splitlines()[0])["round"] == 41


def test_without_agreement_the_old_cross_checked_path_decides(relays, monkeypatch):
    relays({"https://a": 0.0, "https://b": 0.0}, forger="https://a")
    monkeypatch.setattr(corpus_validator, "_BEACONS", {})
    monkeypatch.delenv("RELIQUARY_CORPUS_DRAND_CACHE", raising=False)
    seen = []

    def old(round_id=None, use_fallback=False):
        seen.append(round_id)
        sig = _sig(round_id)
        return {"round": round_id, "signature": sig, "chain_hash": "c" * 64,
                "randomness": hashlib.sha256(bytes.fromhex(sig)).hexdigest()}

    monkeypatch.setattr(drand, "get_drand_beacon", old)
    monkeypatch.setattr(drand, "verify_beacon_signature", lambda *a: True)
    assert corpus_validator.drand_beacon(41) is not None and seen == [41]


# -- the pass ----------------------------------------------------------------------


class _Counting(equivalence._Store):
    pass


def _caught_run(monkeypatch, *, n=3000):
    """Two hotkeys forge every record; their failures trigger backward audits."""
    loop = sim.VirtualTimeLoop()
    forwards = []

    async def scenario():
        store = _Counting()
        clock = sim._Clock()
        hotkeys = [f"5Hk{c}" for c in "ABCDEF"]
        for hotkey in hotkeys:
            store.miners[hotkey] = MinerState(audited_passed=50).to_dict()
        params = AuditParams(q=0.15, probation_submissions=5, hold_seconds=600.0,
                             ban_after_failures=1000)
        auditor = corpus_auditor.CorpusAuditor(
            job_id="math-v1", records=store, model=None, tokenizer=None, proof=None,
            params=params, miner_states=MinerStates(store, "math-v1", clock=clock),
            beacon=sim.Beacon(latency=0.0), round_at=sim.round_at, clock=clock,
            accept_slack_seconds=30.0, rescan_every_seconds=20.0)

        async def forward(records, *, local=False):
            forwards.append((len(records), local))
            return [dict(equivalence._BAD if r["hotkey"] in ("5HkA", "5HkB") else equivalence._OK)
                    for r in records]

        auditor._forward = forward
        t0 = sim.virtual_clock()
        for i in range(n):
            sid = "%064x" % (i + 1)
            store.submissions[sid] = {"hotkey": hotkeys[i % 6],
                                      "received_at": t0 - 7200 + 7200 * i / n,
                                      "token_count": 100, "completions": [{"tokens": [1]}]}
        runner = asyncio.ensure_future(auditor.run())
        await asyncio.sleep(1800)
        runner.cancel()
        return store

    try:
        store = loop.run_until_complete(scenario())
    finally:
        loop.close()
    return store, forwards


def test_backward_audits_never_list_the_job_and_failures_share_one_call(monkeypatch):
    store, forwards = _caught_run(monkeypatch)
    # The start lists once (submissions and verdicts); nothing after it does.
    assert store.calls.get("list", 0) <= 2 + 2 * (1800 // corpus_auditor.FULL_RESCAN_SECONDS)
    caught = [v for v in store.verdicts.values() if v["hotkey"] in ("5HkA", "5HkB")]
    assert caught and all(not v["passed"] for v in caught)
    local = [size for size, is_local in forwards if is_local]
    assert local and max(local) > 1          # re-audited together
    assert len(local) <= len([1 for _, is_local in forwards if not is_local]) + 1


@pytest.mark.parametrize("seed", [7, 3])
def test_the_faster_judge_writes_main_s_verdicts(seed):
    faults = equivalence.ALL
    main, _ = equivalence._run(main_module, "poll", faults, seed)
    fast, _ = equivalence._run(corpus_auditor, "run", faults, seed)
    assert equivalence._outcome(fast) == equivalence._outcome(main)


# -- throughput on a 140k pending index -----------------------------------------------


def _catch_up_140k(beacon_seconds: float):
    """140k pending up to 15 h old, prod traffic (6.9k/h) on top; listings at
    1.4 ms a key, reads 50-100 ms, create-only writes 1.2-1.8 s (64
    connections); a GPU at 8.2k tok/s on 4.2k-token records; two of the 24
    hotkeys forge, so passes keep catching them (backward audits)."""
    params = AuditParams(q=0.15, probation_submissions=100, hold_seconds=4320.0,
                         ban_after_failures=1000)
    population = sim.Population.prod_like(seed=0, probation=0)
    population.tokens = 4200
    result = sim.simulate(
        corpus_auditor, rate_per_hour=6900.0, hours=2.0, params=params, population=population,
        store=sim.Store(seed=0, latency=(0.05, 0.10), write_latency=(1.2, 1.8),
                        in_flight_cap=64),
        gpu=sim.Gpu(tokens_per_second=8200.0, cheaters=population.hotkeys[-2:]),
        beacon=sim.Beacon(latency=beacon_seconds), start_backlog=140_000,
        backlog_age=15 * 3600.0, sample_every=600.0)
    series = result.pending_series
    (h0, _, _, v0), (h1, _, _, v1) = series[3], series[-1]
    return (v1 - v0) / (h1 - h0), series


def test_a_140k_math_backlog_is_judged_at_15k_verdicts_an_hour():
    """A round from two agreeing relays: ~0.15 s."""
    rate, series = _catch_up_140k(0.15)
    print(f"\n140k backlog: {rate:.0f} verdicts/h, series {series}")
    assert rate >= 15_000, (rate, series)
