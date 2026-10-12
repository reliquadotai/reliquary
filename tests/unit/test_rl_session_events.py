"""The issuer reports its sessions and records their faults on the session record."""
import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.miner.rl_episode_client import RlSignedEpisodeRunner  # noqa: E402
from reliquary.miner.signed_episode import (  # noqa: E402
    SignedSweEpisodeRunner,
    open_refusal_of,
    signed_close_request,
)
from reliquary.protocol.sandbox_session import SandboxSessionCloseRequest  # noqa: E402
from reliquary.protocol.signatures import build_sandbox_close_binding  # noqa: E402
from reliquary.sandbox import sessions as sessions_module  # noqa: E402
from reliquary.sandbox.sessions import (  # noqa: E402
    Grant,
    Refusal,
    SessionRecord,
    pool_abort_fault,
    runtime_start_fault,
)
from reliquary.sandbox.tasks import ResolvedTask  # noqa: E402
from tests.unit.test_rl_engagements import HOTKEY, build, close, open_  # noqa: E402

RECEIPT = {"document": {"kind": "open_refusal", "machine_id": "m", "key_id": "k1", "session_id": "s",
                        "token_sha256": "a" * 64, "reason": "no_free_slot", "at": 1}, "signature": "sig"}


class Events:
    def __init__(self):
        self.seen_opened, self.seen_closed = [], []

    def opened(self, record):
        self.seen_opened.append(record.session_id)

    def closed(self, record, final):
        self.seen_closed.append((record.session_id, None if final is None else final["status"]))


def rl(env, *, events=None, check=None):
    env.issuer._events = events
    env.issuer._record_faults = True
    env.issuer._open_refusal_check = check
    return env


def close_failed(env, grant, receipt=None):
    return asyncio.run(env.issuer.close(hotkey=HOTKEY, session_id=grant.session_id, reason="open_failed",
                                        transcript=None, open_refusal=receipt))


def test_the_issuer_reports_each_open_and_each_close(tmp_path):
    events = Events()
    env = rl(build(tmp_path), events=events)
    grant = open_(env, 0)
    assert isinstance(grant, Grant) and events.seen_opened == [grant.session_id]
    close(env, grant, "graded")
    other = open_(env, 1)
    close(env, other, reason="open_failed")
    assert events.seen_closed == [(grant.session_id, "graded"), (other.session_id, None)]


def test_a_failing_listener_never_fails_an_open(tmp_path):
    class Broken:
        def opened(self, record):
            raise RuntimeError("listener")

        def closed(self, record, final):
            raise RuntimeError("listener")

    env = rl(build(tmp_path), events=Broken())
    grant = open_(env, 0)
    assert isinstance(grant, Grant)
    assert close(env, grant, "graded")["state"] == "closed_graded"


def test_a_record_without_a_fault_keeps_its_document_bytes(tmp_path):
    env = build(tmp_path)
    record = env.book.get(open_(env, 0).session_id)
    document = record.to_document()
    assert "fault" not in document
    assert SessionRecord.from_document(document) == record          # an old document reads back
    record.fault = "runtime_exec_failed"
    assert SessionRecord.from_document(record.to_document()).fault == "runtime_exec_failed"
    with pytest.raises(ValueError):
        SessionRecord.from_document({**record.to_document(), "fault": 7})


def test_a_signed_open_refusal_is_recorded_as_a_fault(tmp_path):
    seen = []
    env = rl(build(tmp_path), check=lambda receipt, record: seen.append(record.session_id) or True)
    grant = open_(env, 0)
    close_failed(env, grant, RECEIPT)
    assert env.book.get(grant.session_id).fault == "open_refused:no_free_slot" and seen == [grant.session_id]
    stored = env.issuer._store.documents[grant.session_id]                 # persisted with the record
    assert stored["fault"] == "open_refused:no_free_slot"


def test_an_unverifiable_receipt_or_none_is_the_miners_word(tmp_path):
    env = rl(build(tmp_path), check=lambda receipt, record: False)
    first, second = open_(env, 0), open_(env, 1)
    close_failed(env, first, RECEIPT)
    close_failed(env, second)
    assert env.book.get(first.session_id).fault is None and env.book.get(second.session_id).fault is None


def test_a_receipt_the_directory_cannot_judge_yet_is_resent_not_lost(tmp_path):
    verdicts = [None, True]
    env = rl(build(tmp_path), check=lambda receipt, record: verdicts.pop(0))
    grant = open_(env, 0)
    refused = close_failed(env, grant, RECEIPT)
    assert isinstance(refused, Refusal) and refused.reason == "directory_unavailable" and refused.retry_after
    assert env.book.get(grant.session_id).state == "live"                # not settled: the miner resends
    close_failed(env, grant, RECEIPT)
    assert env.book.get(grant.session_id).fault == "open_refused:no_free_slot"


def test_only_an_open_lost_in_the_restart_window_is_ours(tmp_path):
    env = rl(build(tmp_path))
    recent, old, silent, listed = open_(env, 0), open_(env, 1), open_(env, 2), open_(env, 3)
    store = env.issuer._store.documents
    last = max(document["issued_at"] for document in store.values())     # the last activity before the crash
    store[recent.session_id] = {**store[recent.session_id], "issued_at": last - 30}
    store[old.session_id] = {**store[old.session_id], "issued_at": last - 600}
    env.issuer._clock.now += 300                                          # the restart took five minutes
    env.issuer._tokens.clear()                                            # the restart: every token is lost
    env.issuer._fleet.session_reported = lambda session_id: session_id == listed.session_id   # reached its gateway
    asyncio.run(env.issuer.restore())
    for seed in (0, 1, 3):                                                # the miners resend their lost opens
        resent = open_(env, seed)
        assert isinstance(resent, Refusal) and resent.reason == "request_reused"
    for grant in (recent, old, silent, listed):
        close_failed(env, grant)
    faults = {grant.session_id: env.book.get(grant.session_id).fault for grant in (recent, old, silent, listed)}
    assert faults == {recent.session_id: "validator_restarted", old.session_id: None, silent.session_id: None,
                      listed.session_id: None}


def test_a_pool_abort_caused_by_others_counts_toward_no_cap_and_frees_the_engagement(tmp_path, monkeypatch):
    env = rl(build(tmp_path))
    env.book.policy = dataclasses.replace(env.book.policy, max_aborted_per_day=1)
    real = sessions_module.verify_transcript

    def pool_abort(*args, **kwargs):
        result = real(*args, **kwargs)
        final = {**result.final.to_dict(), "status": "aborted", "reason": "the grading pool stayed full"}
        return SimpleNamespace(ok=result.ok, claims=result.claims, reasons=result.reasons,
                               final=SimpleNamespace(status="aborted", to_dict=lambda: final))

    monkeypatch.setattr(sessions_module, "verify_transcript", pool_abort)
    for _ in range(2):                                                    # the same seed, aborted twice
        grant = open_(env, 0, request_id=f"{len(env.book.records()):032x}")
        assert isinstance(grant, Grant)
        close(env, grant, "graded")
        record = env.book.get(grant.session_id)
        assert record.state == "aborted" and record.fault == "grading_pool"
    assert not env.book.engagement_held(record.engagement)              # freed, whatever the once-only rule says
    refusal = env.book.open_refusal(HOTKEY, int(env.issuer._clock()))
    assert refusal is None or refusal.reason != "aborted_cap"
    own = {"status": "aborted", "reason": "the grading pool stayed full with this miner's own gradings"}
    assert pool_abort_fault(own) is None                                  # the miner's own: an ordinary abort


def test_only_the_rl_runner_forwards_a_receipt():                          # the corpus wire stays
    assert SignedSweEpisodeRunner.forwards_open_refusal is False
    assert RlSignedEpisodeRunner.forwards_open_refusal is True


def test_a_proven_runtime_start_failure_is_recorded(tmp_path, monkeypatch):
    env = rl(build(tmp_path))
    grant = open_(env, 0)
    real = sessions_module.verify_transcript

    def failing_runtime(*args, **kwargs):
        result = real(*args, **kwargs)
        final = result.final.to_dict()
        final["grading"] = {**(final.get("grading") or {}), "_grader": {"runtime_exec_failed": True}}
        return SimpleNamespace(ok=result.ok, claims=result.claims, reasons=result.reasons,
                               final=SimpleNamespace(status=result.final.status, to_dict=lambda: final))

    monkeypatch.setattr(sessions_module, "verify_transcript", failing_runtime)
    close(env, grant, "graded")
    assert env.book.get(grant.session_id).fault == "runtime_exec_failed"
    assert runtime_start_fault({"grading": {"_grader": {"runtime_exec_failed": False}}}) is None


def test_without_record_faults_nothing_is_recorded(tmp_path):
    env = build(tmp_path)                                                   # the corpus issuer's settings
    env.issuer._open_refusal_check = lambda receipt, record: True
    grant = open_(env, 0)
    close_failed(env, grant, RECEIPT)
    assert env.book.get(grant.session_id).fault is None
    assert "fault" not in env.issuer._store.documents[grant.session_id]


def test_the_close_request_carries_a_receipt_only_with_a_failed_open():
    base = {"miner_hotkey": "5Hot", "request_id": "a" * 32, "at": 1, "session_id": "s", "signature": "x"}
    assert SandboxSessionCloseRequest(**base, reason="open_failed").open_refusal is None
    assert SandboxSessionCloseRequest(**base, reason="open_failed", open_refusal=RECEIPT).open_refusal == RECEIPT
    with pytest.raises(ValueError):
        SandboxSessionCloseRequest(**base, reason="withdraw", transcript={}, open_refusal=RECEIPT)
    with pytest.raises(ValueError):
        SandboxSessionCloseRequest(**base, reason="open_failed", open_refusal={"x": "y" * 5000})


def test_the_miner_forwards_the_gateways_receipt_and_its_binding_is_unchanged():
    error = RuntimeError("open refused")
    error.open_refusal = RECEIPT
    try:
        try:
            raise error
        except RuntimeError as inner:
            raise ValueError("the runner failed") from inner
    except ValueError as outer:
        assert open_refusal_of(outer) == RECEIPT
    assert open_refusal_of(ValueError("no receipt")) is None
    kwargs = {"hotkey": "5Hot", "session_id": "s", "reason": "open_failed", "transcript": None,
              "sign_binding": lambda binding: "sig", "now": 1.0, "request_id": "a" * 32,
              "validator_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", "path": "/rl/sessions/s/close"}
    plain, carried = signed_close_request(**kwargs), signed_close_request(**kwargs, open_refusal=RECEIPT)
    assert "open_refusal" not in plain and carried["open_refusal"] == RECEIPT
    audience = {"validator_hotkey": kwargs["validator_hotkey"], "path": kwargs["path"]}
    assert build_sandbox_close_binding(plain, **audience) == build_sandbox_close_binding(carried, **audience)
    assert json.dumps(plain, sort_keys=True)          # still plain JSON


def test_a_token_outlives_its_tasks_preparation_at_the_gateway(tmp_path, monkeypatch):
    from reliquary.sandbox.sessions import token_validity_s
    from tests.unit import test_rl_engagements

    assert token_validity_s(900, 1800) == 2700                            # the default window: unchanged
    assert token_validity_s(100, 1800) == 600 + 60 + 60 + 1800            # a short window is lengthened
    assert token_validity_s(900, 1800, 1500.5) == 1501 + 60 + 60 + 1800  # a long preparation too
    monkeypatch.setattr(test_rl_engagements, "ResolvedTask",
                        lambda image, limits: ResolvedTask(image, limits, prepare_timeout_s=1500))
    env = rl(build(tmp_path))
    grant = open_(env, 0)
    claims = attest.SessionClaims.from_dict(grant.token["claims"])
    opened_at = claims.issued_at + 60                                     # the token's transit
    assert opened_at + 1500 + 60 < claims.expires_at                      # the gateway's late rule admits it
    assert claims.expires_at - claims.issued_at == token_validity_s(900, claims.budgets.wall_s, 1500)


def fleet_with(directory_ready=True):
    from reliquary.sandbox.fleet import Fleet

    async def documents():
        return []

    fleet = Fleet(read_documents=documents, clock=lambda: 1_000_000)
    if directory_ready:
        fleet._directory_at = 1_000_000
    record = SimpleNamespace(machine_id="m", session_id="s", token_sha256="a" * 64,
                             issued_at=999_000, expires_at=1_002_000)
    return fleet, record


def test_the_fleet_verifies_a_receipt_signed_while_its_token_was_valid(monkeypatch, caplog):
    seen = []
    monkeypatch.setattr(attest, "open_refusal_valid", lambda receipt, directory, **kw: seen.append(kw) or True,
                        raising=False)
    fleet, record = fleet_with()
    receipt = {**RECEIPT, "document": {**RECEIPT["document"], "at": 1_000_000}}
    assert fleet.open_refusal_valid(receipt, record) is True
    assert seen == [{"machine_id": "m", "session_id": "s", "token_sha256": "a" * 64, "now": 1_000_000}]
    for at in (999_000 - 121, 1_002_000 + 121, "1"):                    # outside the token's validity
        late = {**RECEIPT, "document": {**RECEIPT["document"], "at": at}}
        assert fleet.open_refusal_valid(late, record) is False
    assert len(seen) == 1
    stale, _ = fleet_with(directory_ready=False)
    assert stale.open_refusal_valid(receipt, record) is None             # cannot judge now
    monkeypatch.delattr(attest, "open_refusal_valid", raising=False)      # a reliquary-sandbox without receipts
    with caplog.at_level("ERROR"):
        assert fleet.open_refusal_valid(receipt, record) is False
        assert fleet.open_refusal_valid(receipt, record) is False
    assert sum("re-pin" in r.getMessage() for r in caplog.records) == 1


def test_the_box_end_is_the_first_close_with_the_machines_final_and_a_withdrawal_keeps_it(tmp_path):
    # A graded session withdrawn later (its group was not chosen): the withdrawal moves its closed_at and its
    # closed status, never its box's end, so its held seconds still count, from issue to that end. A failed open
    # carries no machine final: no box end, and nothing held unless a machine listed it.
    from reliquary.sandbox.rl_yield import held_seconds

    env = rl(build(tmp_path))
    clock = env.issuer._clock
    grant = open_(env, 0)
    record = env.book.get(grant.session_id)
    clock.now = record.issued_at + 300
    assert close(env, grant, "graded")["state"] == "closed_graded"
    assert record.ended_at == record.issued_at + 300
    clock.now += 600
    assert close(env, grant, "graded", reason="withdraw")["status"] == "withdrawn"
    assert record.closed_at == record.issued_at + 900 and record.ended_at == record.issued_at + 300
    assert held_seconds(record, clock.now + 5_000) == 300.0
    failed = open_(env, 1)
    close_failed(env, failed)
    failed = env.book.get(failed.session_id)
    assert failed.ended_at is None and held_seconds(failed, clock.now + 5_000) == 0.0
    assert held_seconds(failed, clock.now + 5_000, presented=True) == failed.closed_at - failed.issued_at
