"""A corpus task's validator: refuses what it cannot run, serves what it can."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from reliquary.protocol.profiles import PROOF_SCHEME_TOPLOC
from reliquary.validator.corpus_validator import (
    build_corpus_app,
    build_corpus_audit_wiring,
    drand_beacon,
    make_round_at,
    startup_refusal,
)
from tests.unit.test_corpus_service import (  # noqa: F401
    EOS, _Tokenizer, _faithful_prompt, _r2_client, _text_for, fake_r2, seeded_job,
)


def _entry(**kw):
    return SimpleNamespace(task_id="corpus-math", job_id="swe-v1", mechanism="corpus-generation", **kw)


def _profile(toploc=True, model_id="org/Frozen", model_revision="abc123"):
    proof = SimpleNamespace(scheme=PROOF_SCHEME_TOPLOC, mode="enforce", chunk_tokens=32, topk=128)
    return SimpleNamespace(
        model_id=model_id, model_revision=model_revision, proofs=(proof,) if toploc else ()
    )


def test_a_contract_without_toploc_refuses(seeded_job):
    assert "toploc" in startup_refusal(_entry(), seeded_job.job, _profile(toploc=False), seeded_job.job.checkpoint_sha256)


def test_a_shadow_toploc_refuses(seeded_job):
    profile = _profile()
    profile.proofs[0].mode = "shadow"
    assert "enforce" in startup_refusal(_entry(), seeded_job.job, profile, seeded_job.job.checkpoint_sha256)


def test_a_contract_on_another_model_refuses(seeded_job):
    assert "model" in startup_refusal(
        _entry(), seeded_job.job, _profile(model_id="org/Other"), seeded_job.job.checkpoint_sha256
    )


def test_a_contract_on_another_revision_refuses(seeded_job):
    # `ProtocolProfile` pins both the repo and the revision it was measured
    # on; a job declared at another revision is not the checkpoint the
    # contract's proof thresholds were set against.
    assert "model" in startup_refusal(
        _entry(), seeded_job.job, _profile(model_revision="deadbeef"), seeded_job.job.checkpoint_sha256
    )


def test_a_checkpoint_that_does_not_match_the_job_refuses(seeded_job):
    assert "fingerprint" in startup_refusal(_entry(), seeded_job.job, _profile(), "0" * 64)


def test_a_matching_setup_starts(seeded_job):
    assert startup_refusal(_entry(), seeded_job.job, _profile(), seeded_job.job.checkpoint_sha256) is None


def test_the_app_serves_the_corpus_routes_and_nothing_of_rl(seeded_job):
    enqueued = []
    auditor = SimpleNamespace(enqueue=enqueued.append)
    app = build_corpus_app(entry=_entry(), job=seeded_job.job, store=seeded_job.store, records=None,
                           tokenizer=_Tokenizer(), renderer=seeded_job.renderer,
                           verify_signature=lambda r: True, auditor=auditor, proof_chunk_tokens=None,
                           prompt_job_for=seeded_job.prompt_job_for)
    client = TestClient(app)
    assert client.get("/corpus/job").status_code == 200
    assert client.get("/state").status_code == 404


# --------------------------------------------------------------------------
# Task 7: wiring the audit into the corpus validator
# --------------------------------------------------------------------------


@pytest.fixture
def wired_records(_r2_client, monkeypatch):
    """A real ``BucketRecordStore`` over the same fake bucket ``seeded_job``
    uses, so ``MinerStates`` reads and writes actual (fake) bytes rather than
    a second, drifting store."""
    from reliquary.infrastructure import corpus_record_store as records_module
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    monkeypatch.setattr(records_module, "get_s3_client", lambda **kw: _r2_client)
    return BucketRecordStore()


@pytest.fixture
def fixed_drand_chain(monkeypatch):
    """A resolved drand chain, so the wiring never reaches the network."""
    from reliquary.infrastructure import drand

    monkeypatch.setattr(
        drand, "get_current_chain", lambda: {"genesis_time": 1_600_000_000.0, "period": 3.0}
    )


def test_audit_q_absent_gives_the_auditor_q_one(seeded_job, wired_records, fixed_drand_chain):
    from reliquary.validator.corpus_auditor import CorpusAuditor

    entry = _entry(params={"cap": 1.0})  # no audit_* keys at all
    params, miner_states, is_banned, beacon, round_at = build_corpus_audit_wiring(
        entry=entry, job=seeded_job.job, records=wired_records
    )
    assert params.q == 1.0

    auditor = CorpusAuditor(job_id="swe-v1", records=wired_records, model=None,
                            tokenizer=None, proof=None, params=params,
                            miner_states=miner_states, beacon=beacon, round_at=round_at)
    assert auditor._params.q == 1.0


def test_wiring_resolves_beacon_and_round_at_when_the_chain_is_known(
    seeded_job, wired_records, fixed_drand_chain
):
    entry = _entry(params={})
    _, _, _, beacon, round_at = build_corpus_audit_wiring(
        entry=entry, job=seeded_job.job, records=wired_records
    )
    assert beacon is drand_beacon
    # round 1 publishes exactly at genesis; round_at(genesis) is the next one.
    assert round_at(1_600_000_000.0) == 2


def test_wiring_never_touches_the_chain_at_build_time(seeded_job, wired_records, monkeypatch):
    """Resolution is lazy (fix round 1, finding 2): an `/info` fetch that
    fails while this process boots must not decide sampling for the rest of
    its life, so building the wiring must not itself resolve the chain --
    only `round_at`'s first actual use does."""
    from reliquary.infrastructure import drand

    calls = []
    monkeypatch.setattr(
        drand, "get_current_chain",
        lambda: (calls.append(1), {"genesis_time": None, "period": 3.0})[1],
    )
    entry = _entry(params={})

    params, miner_states, is_banned, beacon, round_at = build_corpus_audit_wiring(
        entry=entry, job=seeded_job.job, records=wired_records
    )

    assert calls == []
    assert beacon is drand_beacon
    assert round_at is not None and callable(round_at)


def test_lazy_round_at_raises_until_the_chain_resolves(monkeypatch):
    """Chain unknown at start: `round_at` raises rather than guessing a round
    (the auditor then audits, per test_a_raising_round_at_audits in
    test_corpus_judge.py). Chain resolves later: the SAME object starts
    returning real rounds -- no restart, nobody rebuilds it."""
    from reliquary.infrastructure import drand
    from reliquary.validator.corpus_validator import LazyRoundAt

    chain = {"genesis_time": None, "period": 3.0}
    monkeypatch.setattr(drand, "get_current_chain", lambda: dict(chain))

    # retry_seconds=0: this test is about resolution succeeding once the
    # chain is known, not about the throttle (covered separately below).
    round_at = LazyRoundAt(retry_seconds=0.0)
    with pytest.raises(Exception):
        round_at(0.0)

    chain["genesis_time"] = 1_600_000_000.0
    assert round_at(1_600_000_000.0) == 2


def test_lazy_round_at_throttles_retries_to_once_per_window(monkeypatch):
    from reliquary.infrastructure import drand
    from reliquary.validator.corpus_validator import LazyRoundAt

    calls = []
    monkeypatch.setattr(
        drand, "get_current_chain",
        lambda: (calls.append(1), {"genesis_time": None, "period": 3.0})[1],
    )

    now = [1000.0]
    round_at = LazyRoundAt(retry_seconds=60.0, clock=lambda: now[0])

    with pytest.raises(Exception):
        round_at(0.0)
    assert len(calls) == 1

    # Still inside the retry window: not retried.
    now[0] += 10.0
    with pytest.raises(Exception):
        round_at(0.0)
    assert len(calls) == 1

    # Past the retry window: retried.
    now[0] += 60.0
    with pytest.raises(Exception):
        round_at(0.0)
    assert len(calls) == 2


def test_lazy_round_at_caches_a_resolved_chain_forever(monkeypatch):
    from reliquary.infrastructure import drand
    from reliquary.validator.corpus_validator import LazyRoundAt

    calls = []
    monkeypatch.setattr(
        drand, "get_current_chain",
        lambda: (calls.append(1), {"genesis_time": 1_600_000_000.0, "period": 3.0})[1],
    )

    round_at = LazyRoundAt()
    assert round_at(1_600_000_000.0) == 2
    assert round_at(1_600_000_003.0) == 3
    assert len(calls) == 1


def test_is_banned_reflects_a_banned_hotkey(seeded_job, wired_records, fixed_drand_chain):
    entry = _entry(params={})
    _, miner_states, is_banned, _, _ = build_corpus_audit_wiring(
        entry=entry, job=seeded_job.job, records=wired_records
    )
    asyncio.run(miner_states.update("5Banned", lambda m: replace(m, banned_until=4_102_444_800.0)))

    assert asyncio.run(is_banned("5Banned")) is True
    assert asyncio.run(is_banned("5Hot")) is False


def test_the_app_refuses_a_banned_hotkey(seeded_job, wired_records, fixed_drand_chain):
    entry = _entry(params={})
    job = seeded_job.job
    _, miner_states, is_banned, _, _ = build_corpus_audit_wiring(
        entry=entry, job=job, records=wired_records
    )
    asyncio.run(miner_states.update("5Banned", lambda m: replace(m, banned_until=4_102_444_800.0)))

    auditor = SimpleNamespace(enqueue=lambda sid: None)
    app = build_corpus_app(entry=entry, job=job, store=seeded_job.store, records=wired_records,
                           tokenizer=_Tokenizer(), renderer=seeded_job.renderer,
                           verify_signature=lambda r: True, auditor=auditor, proof_chunk_tokens=None,
                           prompt_job_for=seeded_job.prompt_job_for, is_banned=is_banned)
    client = TestClient(app)

    body = client.post(
        "/corpus/submit",
        json={
            "job_id": "swe-v1",
            "miner_hotkey": "5Banned",
            "cursor": 0,
            "prompt_index": 0,
            "checkpoint_sha256": "a" * 64,
            "rendered_prompt": _faithful_prompt(0),
            "completions": [{"tokens": [7] * 16 + [EOS], "text": _text_for([7] * 16 + [EOS])}],
            "signature": "ok",
        },
    ).json()

    assert body["accepted"] is False
    assert body["reason"] == "miner_banned"


# --------------------------------------------------------------------------
# round_at: the first drand round published strictly after t
# --------------------------------------------------------------------------


def test_round_at_boundary_at_a_3s_period():
    round_at = make_round_at(genesis_time=1000.0, period=3.0)
    # Round 5 publishes at 1000 + 4*3 = 1012.
    assert round_at(1011.999) == 5
    assert round_at(1012.0) == 6  # exactly at a publication instant -> the next round
    assert round_at(1012.001) == 6


def test_round_at_boundary_at_a_30s_period():
    round_at = make_round_at(genesis_time=2000.0, period=30.0)
    # Round 4 publishes at 2000 + 3*30 = 2090.
    assert round_at(2089.999) == 4
    assert round_at(2090.0) == 5  # exactly at a publication instant -> the next round
    assert round_at(2090.001) == 5


# --------------------------------------------------------------------------
# drand_beacon: randomness, or None on anything that is not a clean fetch
# --------------------------------------------------------------------------


def test_drand_beacon_none_on_fetch_error(monkeypatch):
    from reliquary.infrastructure import drand

    def _raise(**kw):
        raise RuntimeError("no relay reachable")

    monkeypatch.setattr(drand, "get_drand_beacon", _raise)
    assert drand_beacon(42) is None


def test_drand_beacon_none_on_round_mismatch(monkeypatch):
    from reliquary.infrastructure import drand

    monkeypatch.setattr(drand, "get_drand_beacon", lambda **kw: {
        "round": 41, "randomness": "ab" * 32, "signature": "cd" * 48, "chain_hash": "x",
    })
    monkeypatch.setattr(drand, "verify_beacon_signature", lambda *a, **kw: True)
    assert drand_beacon(42) is None


def test_drand_beacon_none_on_malformed_randomness(monkeypatch):
    from reliquary.infrastructure import drand

    monkeypatch.setattr(drand, "get_drand_beacon", lambda **kw: {
        "round": 42, "randomness": "not-hex", "signature": "cd" * 48, "chain_hash": "x",
    })
    monkeypatch.setattr(drand, "verify_beacon_signature", lambda *a, **kw: True)
    assert drand_beacon(42) is None


def test_drand_beacon_none_when_the_signature_does_not_verify(monkeypatch):
    from reliquary.infrastructure import drand

    monkeypatch.setattr(drand, "get_drand_beacon", lambda **kw: {
        "round": 42, "randomness": "AB" * 32, "signature": "cd" * 48, "chain_hash": "x",
    })
    monkeypatch.setattr(drand, "verify_beacon_signature", lambda *a, **kw: False)
    assert drand_beacon(42) is None


def test_drand_beacon_lowercases_randomness_on_success(monkeypatch):
    from reliquary.infrastructure import drand

    calls = []
    monkeypatch.setattr(drand, "get_drand_beacon", lambda **kw: {
        "round": 42, "randomness": "AB" * 32, "signature": "cd" * 48, "chain_hash": "x",
    })

    def _verify(chain_hash, round_number, randomness_hex, signature_hex):
        calls.append((chain_hash, round_number, randomness_hex, signature_hex))
        return True

    monkeypatch.setattr(drand, "verify_beacon_signature", _verify)

    assert drand_beacon(42) == "ab" * 32
    assert calls == [("x", 42, "ab" * 32, "cd" * 48)]
