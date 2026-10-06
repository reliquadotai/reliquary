"""Who may open a sandbox episode: the engagement, the reservation, the per-hotkey
caps, the machine; and how each session ends."""

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.corpus.job import parse_job  # noqa: E402
from reliquary.infrastructure import sandbox_store  # noqa: E402
from reliquary.infrastructure.sandbox_store import MemorySessionStore, R2SessionStore  # noqa: E402
from reliquary.sandbox.fleet import Placement  # noqa: E402
from reliquary.sandbox import sessions  # noqa: E402
from reliquary.sandbox.sessions import (  # noqa: E402
    ABORTED, CLOSED, CLOSED_GRADED, LAPSED, LIVE, SUBMITTED, VOIDED, SessionRecord, CorpusEngagements, Grant,
    RlPrecommitEngagements, SandboxPolicy, SessionBook, SessionIssuer, SignedJobView,
)
from reliquary.sandbox.tasks import ResolvedTask  # noqa: E402
from tests.unit.sandbox_fixtures import (  # noqa: E402
    ADDRESS, IMAGE, MACHINE, NOW, directory, signer, transcript,
)
from tests.unit.test_corpus_job_episode import _manifest  # noqa: E402
from tests.unit.test_corpus_job_signed_sandbox import signed_episode  # noqa: E402
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2  # noqa: E402

JOB = parse_job(_manifest(episode=signed_episode()))
CORPUS = {"kind": "corpus", "job_id": JOB.job_id, "prompt_index": 3}


@pytest.fixture(autouse=True)
def _sessions_logs_reach_caplog():
    """Importing bittensor sets every logger that exists then to CRITICAL; without this,
    a log assertion here would pass on an empty capture."""
    logger = sessions.logger
    level = logger.level
    logger.setLevel("DEBUG")
    yield
    logger.setLevel(level)


class FakeFleet:
    def __init__(self, snapshot, placement=Placement(MACHINE, ADDRESS)):
        self.snapshot, self.placement = snapshot, placement
        self.picked, self.issued = [], []
        self.ready = True

    def directory_ready(self, now=None):
        return self.ready

    def pick(self, **kw):
        self.picked.append(kw)
        return self.placement

    def note_issued(self, machine_id, at):
        self.issued.append((machine_id, at))

    def directory(self):
        return self.snapshot


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


def build(tmp_path, *, remaining=2, policy=SandboxPolicy(), store=None, limits=None, clock=None,
          jobs=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    validator, machine = signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1")
    clock = clock or Clock()
    book = SessionBook(policy)
    store = store if store is not None else MemorySessionStore()
    fleet = FakeFleet(directory(machine))
    left = {"n": remaining}

    async def resolve(index):
        return ResolvedTask(IMAGE, dict(limits or {}))

    async def slots(index):
        return left["n"]

    view = SignedJobView(job=JOB, resolve_task=resolve, slots_remaining=slots)
    jobs = jobs if jobs is not None else {JOB.job_id: view}
    ids = iter(f"s-{n}" for n in range(1000))
    issuer = SessionIssuer(
        book=book, store=store, fleet=fleet, signer=validator,
        token_verifier=attest.Ed25519TokenVerifier({"v1": validator.public_key_b64}),
        engagements={"corpus": CorpusEngagements(jobs.get, book, clock),
                     "rl_precommit": RlPrecommitEngagements()},
        policy=policy, clock=clock, new_session_id=lambda: next(ids))
    return SimpleNamespace(issuer=issuer, book=book, store=store, fleet=fleet, validator=validator,
                           machine=machine, clock=clock, left=left)


def open_(env, hotkey="5Hot", request_id="a" * 32, engagement=CORPUS):
    return asyncio.run(env.issuer.open(hotkey=hotkey, request_id=request_id, engagement=engagement))


def final_of(env, grant, status, **kw):
    session = attest.SessionClaims.from_dict(grant.token["claims"])
    return transcript(env.validator, env.machine, session, status=status, **kw)


def close(env, grant, transcript_, hotkey="5Hot", reason="final"):
    return asyncio.run(env.issuer.close(hotkey=hotkey, session_id=grant.session_id, reason=reason,
                                        transcript=transcript_))


def test_a_grant_binds_the_engagement_and_raises_budgets_to_the_tasks_limits(tmp_path):
    env = build(tmp_path, limits={"memory_bytes": 6 * 1024**3, "cpus": 4})
    grant = open_(env)
    claims = attest.verify_session_token(grant.token, {"v1": attest.public_key_from_b64(
        env.validator.public_key_b64)})
    assert (claims.hotkey, claims.engagement, claims.env, claims.split, claims.index,
            claims.checkpoint, claims.machine_id, claims.image) == (
        "5Hot", f"corpus:{JOB.job_id}:3", "reliquary-swe", "train:20", 3, JOB.checkpoint_sha256,
        MACHINE, IMAGE)
    assert claims.budgets.memory_bytes == 6 * 1024**3
    assert claims.expires_at == NOW + 900 + JOB.episode.sandbox.budgets.wall_s
    assert grant.gateway_url == ADDRESS and env.fleet.issued == [(MACHINE, NOW)]


def test_a_resent_request_gets_the_same_token_once(tmp_path):
    env = build(tmp_path)
    first, again = open_(env), open_(env)
    assert again == first and len(env.fleet.issued) == 1
    assert env.book.reserved(JOB.job_id, 3, NOW) == 1


def test_a_token_never_reaches_the_logs(tmp_path, caplog):
    env = build(tmp_path)
    caplog.set_level("DEBUG")
    grant = open_(env)
    assert f"sandbox session {grant.session_id} issued" in caplog.text   # captured for real
    assert grant.token["signature"] not in caplog.text
    assert grant.token["signature"] not in repr(grant)
    assert "signature" not in env.store.documents[grant.session_id]


def test_a_prompt_has_as_many_reservations_as_free_slots(tmp_path):
    env = build(tmp_path, remaining=1)
    grant = open_(env)
    refused = open_(env, hotkey="5Other", request_id="b" * 32)
    assert refused.reason == "prompt_unavailable"
    close(env, grant, final_of(env, grant, "expired"))
    assert isinstance(open_(env, hotkey="5Other", request_id="b" * 32), Grant)


@pytest.mark.parametrize("engagement,reason", [
    ({"kind": "corpus", "job_id": "other", "prompt_index": 3}, "job_not_served"),
    ({"kind": "corpus", "job_id": JOB.job_id, "prompt_index": 10**9}, "prompt_mismatch"),
    ({"kind": "corpus", "job_id": JOB.job_id, "prompt_index": True}, "prompt_mismatch"),
    ({"kind": "rl_precommit", "precommit": {"window": 1}}, "engagement_kind_unsupported"),
    ({"kind": "batch"}, "engagement_kind_unsupported"),
])
def test_an_engagement_this_validator_cannot_serve_is_refused(tmp_path, engagement, reason):
    assert open_(build(tmp_path), engagement=engagement).reason == reason


def test_a_replay_job_gets_no_session(tmp_path):
    replay = parse_job(_manifest())
    view = SignedJobView(job=replay, resolve_task=None, slots_remaining=None)
    assert open_(build(tmp_path, jobs={JOB.job_id: view})).reason == "job_not_signed"


def test_a_complete_job_gets_no_session(tmp_path):
    env = build(tmp_path)
    env.left["n"] = None
    assert open_(env).reason == "job_complete"


def test_no_machine_means_retry_after(tmp_path):
    env = build(tmp_path)
    env.fleet.placement = None
    refused = open_(env)
    assert refused.reason == "sandbox_capacity" and refused.retry_after == 10
    assert env.book.reserved(JOB.job_id, 3, NOW) == 0


def test_a_store_failure_issues_nothing(tmp_path):
    env = build(tmp_path)
    env.store.fail = True
    refused = open_(env)
    assert refused.reason == "store_unavailable" and env.book.reserved(JOB.job_id, 3, NOW) == 0


def test_live_sessions_per_hotkey_are_capped(tmp_path):
    env = build(tmp_path, policy=SandboxPolicy(max_live_per_hotkey=1), remaining=5)
    open_(env)
    assert open_(env, request_id="b" * 32).reason == "live_cap"


def test_opens_per_hour_are_capped_and_aborted_ones_refunded(tmp_path):
    env = build(tmp_path, policy=SandboxPolicy(max_opens_per_hour=1), remaining=5)
    grant = open_(env)
    close(env, grant, final_of(env, grant, "aborted"))
    second = open_(env, request_id="b" * 32)
    assert isinstance(second, Grant)
    close(env, second, final_of(env, second, "box_failed"))
    assert open_(env, request_id="c" * 32).reason == "open_rate_cap"
    env.clock.now = NOW + 3601
    assert isinstance(open_(env, request_id="d" * 32), Grant)


def test_aborted_episodes_per_hotkey_are_capped(tmp_path):
    env = build(tmp_path, policy=SandboxPolicy(max_aborted_per_day=1), remaining=5)
    grant = open_(env)
    assert close(env, grant, final_of(env, grant, "aborted"))["state"] == ABORTED
    assert open_(env, request_id="b" * 32).reason == "aborted_cap"


@pytest.mark.parametrize("status,reason,state", [
    ("expired", None, CLOSED), ("box_failed", None, CLOSED),
    ("budget_exhausted", "transcript_bytes", CLOSED), ("aborted", "task changed", ABORTED),
])
def test_an_unpaid_final_ends_the_reservation(tmp_path, status, reason, state):
    env = build(tmp_path)
    grant = open_(env)
    closed = close(env, grant, final_of(env, grant, status, reason=reason))
    assert closed == {"session_id": grant.session_id, "state": state, "status": status}
    assert env.book.reserved(JOB.job_id, 3, NOW) == 0
    assert env.store.documents[grant.session_id]["state"] == state


def test_a_close_must_carry_the_sessions_own_verified_transcript(tmp_path):
    env = build(tmp_path, remaining=5)
    grant = open_(env)
    other = open_(env, request_id="b" * 32, engagement={**CORPUS, "prompt_index": 4})
    assert close(env, grant, final_of(env, other, "expired")).reason == "transcript_invalid"
    assert close(env, grant, None).reason == "transcript_invalid"
    assert close(env, grant, final_of(env, grant, "expired"), hotkey="5Other").reason == \
        "session_unknown"
    forged = final_of(env, grant, "expired")
    forged["records"][-1]["body"]["status"] = "aborted"
    assert close(env, grant, forged).reason == "transcript_invalid"


def test_a_failed_open_closes_without_a_transcript(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    assert close(env, grant, None, reason="open_failed")["state"] == CLOSED


def test_a_submitted_session_releases_its_reservation_once(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    asyncio.run(env.issuer.submitted(grant.session_id))
    asyncio.run(env.issuer.submitted(grant.session_id))
    assert env.book.get(grant.session_id).state == SUBMITTED
    assert env.book.reserved(JOB.job_id, 3, NOW) == 0
    assert grant.session_id in env.book.submitted_ids()


def test_a_session_past_its_grace_lapses(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    env.clock.now = grant.expires_at + attest.GRADING_GRACE_S + 1
    asyncio.run(env.issuer.maintain())
    assert env.book.get(grant.session_id).state == LAPSED
    assert env.book.reserved(JOB.job_id, 3, env.clock.now) == 0


def test_a_drained_machines_sessions_are_voided_without_fault(tmp_path):
    env = build(tmp_path)
    grant = open_(env)

    async def drain():
        env.issuer.void_machine(MACHINE)
        await asyncio.sleep(0)

    asyncio.run(drain())
    assert env.book.get(grant.session_id).state == VOIDED
    assert env.book.reserved(JOB.job_id, 3, NOW) == 0
    assert isinstance(open_(env, request_id="b" * 32), Grant)    # not an aborted re-roll


def test_a_restart_restores_reservations_and_caps_but_no_token(tmp_path, monkeypatch):
    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(sandbox_store, "get_s3_client", lambda **kw: fake)
    policy = SandboxPolicy(max_live_per_hotkey=1)
    first = build(tmp_path / "a", store=R2SessionStore(), policy=policy, remaining=1)
    grant = open_(first)
    again = build(tmp_path / "b", store=R2SessionStore(), policy=policy, remaining=1)
    assert asyncio.run(again.issuer.restore()) == 1
    assert again.book.get(grant.session_id).state == LIVE
    assert again.book.reserved(JOB.job_id, 3, NOW) == 1
    assert open_(again).reason == "request_reused"                 # the token is not kept
    assert open_(again, request_id="b" * 32).reason == "live_cap"
    stored = fake.objects[sandbox_store.session_key(grant.session_id, grant.expires_at)]
    assert "signature" not in str(stored)


def test_the_policy_reads_the_environment():
    policy = SandboxPolicy.from_env({"RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY": "3",
                                     "RELIQUARY_SANDBOX_MAX_ABORTED_PER_DAY": "5"})
    assert (policy.max_live_per_hotkey, policy.max_aborted_per_day, policy.max_opens_per_hour) == \
        (3, 5, 120)
    with pytest.raises(ValueError):
        SandboxPolicy.from_env({"RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY": "0"})


def test_the_rl_stub_documents_its_mapping():
    stub = RlPrecommitEngagements()
    assert stub.kind == "rl_precommit" and "rl:{window}:{precommit_sha256}" in stub.__doc__
    refused = asyncio.run(stub.terms("5Hot", {"kind": "rl_precommit"}))
    assert refused.reason == "engagement_kind_unsupported"


def test_the_swe_task_resolver_reads_image_and_declared_limits():
    from reliquary.sandbox.tasks import SweTaskResolver

    seen = []

    def sandbox_task(split, index):
        seen.append((split, index))
        limits = SimpleNamespace(memory_bytes=4 * 1024**3, disk_bytes=None, pids=1024,
                                 wall_s=3600, per_call_timeout_s=600, max_calls=None)
        return SimpleNamespace(image=IMAGE, limits=limits)

    resolver = SweTaskResolver("train:20", sandbox_task=sandbox_task)
    task = asyncio.run(resolver.resolve(3))
    asyncio.run(resolver.resolve(3))
    assert task == ResolvedTask(IMAGE, {"memory_bytes": 4 * 1024**3, "pids": 1024,
                                        "wall_s": 3600, "per_call_timeout_s": 600})
    assert seen == [("train:20", 3)]


def test_a_stale_directory_refuses_an_open_retryably_before_any_placement(tmp_path):
    env = build(tmp_path)
    env.fleet.ready = False
    refused = open_(env)
    assert refused.reason == "directory_unavailable" and refused.retry_after == 10
    assert env.fleet.picked == [] and env.book.reserved(JOB.job_id, 3, NOW) == 0
    env.fleet.ready = True
    assert isinstance(open_(env), Grant)


def test_a_stale_directory_defers_a_close_instead_of_calling_it_unknown_key(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    env.fleet.ready = False
    refused = close(env, grant, final_of(env, grant, "expired"))
    assert refused.reason == "directory_unavailable" and refused.retry_after == 10
    assert env.book.get(grant.session_id).state == LIVE
    env.fleet.ready = True
    assert close(env, grant, final_of(env, grant, "expired"))["state"] == CLOSED


def test_a_refusal_never_echoes_the_token(tmp_path):
    env = build(tmp_path, remaining=5)
    grant = open_(env)
    forged = final_of(env, grant, "expired")
    forged["records"][-1]["body"]["status"] = "aborted"
    refused = close(env, grant, forged)
    assert grant.token["signature"] not in repr(refused)
    assert grant.token["signature"] not in str(env.store.documents)


def test_the_machine_is_asked_for_the_tokens_validity_and_budgets(tmp_path):
    env = build(tmp_path, limits={"memory_bytes": 6 * 1024**3})
    open_(env)
    (asked,) = env.fleet.picked
    budgets = JOB.episode.sandbox.budgets
    assert asked["validity_s"] == 900 + budgets.wall_s
    assert asked["budgets"] == {**budgets.to_contract(), "memory_bytes": 6 * 1024**3}
    assert (asked["image"], asked["env"], asked["env_package"], asked["now"]) == (
        IMAGE, "reliquary-swe", JOB.episode.sandbox.env_package, NOW)


def test_a_restart_keeps_the_aborted_cap(tmp_path, monkeypatch):
    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(sandbox_store, "get_s3_client", lambda **kw: fake)
    policy = SandboxPolicy(max_aborted_per_day=1)
    first = build(tmp_path / "a", store=R2SessionStore(), policy=policy, remaining=5)
    grant = open_(first)
    assert close(first, grant, final_of(first, grant, "aborted"))["state"] == ABORTED
    again = build(tmp_path / "b", store=R2SessionStore(), policy=policy, remaining=5)
    assert asyncio.run(again.issuer.restore()) == 1
    assert again.book.get(grant.session_id).state == ABORTED
    assert open_(again, request_id="b" * 32).reason == "aborted_cap"



# -- fix round 1 ---------------------------------------------------------------------

def test_a_hotkey_holds_one_live_session_per_prompt(tmp_path):
    env = build(tmp_path, remaining=5)
    open_(env)
    refused = open_(env, request_id="b" * 32)
    assert refused.reason == "prompt_live_cap"
    assert env.book.reserved(JOB.job_id, 3, NOW) == 1
    assert isinstance(open_(env, hotkey="5Other", request_id="b" * 32), Grant)


def test_a_hotkey_holds_a_bounded_number_of_live_sessions_per_job(tmp_path):
    env = build(tmp_path, policy=SandboxPolicy(max_live_per_hotkey_job=2), remaining=5)
    for n, index in enumerate((3, 4)):
        assert isinstance(open_(env, request_id=str(n) * 32,
                                engagement={**CORPUS, "prompt_index": index}), Grant)
    refused = open_(env, request_id="z" * 32, engagement={**CORPUS, "prompt_index": 5})
    assert refused.reason == "job_live_cap"
    assert SandboxPolicy().max_live_per_hotkey_job == 4
    assert SandboxPolicy.from_env({"RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY_JOB": "6"}) \
        .max_live_per_hotkey_job == 6


def test_a_graded_close_keeps_the_slot_until_submitted_or_lapsed(tmp_path):
    env = build(tmp_path, remaining=1)
    grant = open_(env)
    closed = close(env, grant, final_of(env, grant, "graded"))
    assert closed == {"session_id": grant.session_id, "state": CLOSED_GRADED, "status": "graded"}
    assert env.book.reserved(JOB.job_id, 3, NOW) == 1
    assert open_(env, hotkey="5Other", request_id="b" * 32).reason == "prompt_unavailable"
    asyncio.run(env.issuer.submitted(grant.session_id))
    assert env.book.get(grant.session_id).state == SUBMITTED
    assert env.store.documents[grant.session_id]["state"] == SUBMITTED
    assert env.book.reserved(JOB.job_id, 3, NOW) == 0


def test_a_graded_close_that_is_never_submitted_lapses(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    close(env, grant, final_of(env, grant, "graded"))
    env.clock.now = grant.expires_at + attest.GRADING_GRACE_S + 1
    asyncio.run(env.issuer.maintain())
    assert env.book.get(grant.session_id).state == LAPSED
    assert env.store.documents[grant.session_id]["state"] == LAPSED


def test_a_graded_close_is_not_voided_by_a_drain(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    close(env, grant, final_of(env, grant, "graded"))

    async def drain():
        env.issuer.void_machine(MACHINE)
        await asyncio.sleep(0)

    asyncio.run(drain())
    assert env.book.get(grant.session_id).state == CLOSED_GRADED


def test_a_submission_ends_a_graded_close(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    close(env, grant, final_of(env, grant, "graded"))
    asyncio.run(env.issuer.submitted(grant.session_id))
    assert env.book.get(grant.session_id).state == SUBMITTED
    assert env.store.documents[grant.session_id]["state"] == SUBMITTED
    assert env.book.reserved(JOB.job_id, 3, NOW) == 0


def test_a_lapsed_session_is_never_paid(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    env.clock.now = grant.expires_at + attest.GRADING_GRACE_S + 1
    asyncio.run(env.issuer.maintain())
    asyncio.run(env.issuer.submitted(grant.session_id))
    assert not sessions.session_submittable(LAPSED)
    assert env.book.get(grant.session_id).state == LAPSED
    assert env.store.documents[grant.session_id]["state"] == LAPSED
    assert SUBMITTED not in sandbox_store.SESSION_TRANSITIONS[LAPSED]


def test_a_voided_session_is_never_paid(tmp_path):
    """Amended ruling 3: a drained machine is our fault, so the miner keeps its open
    refund and opens again; its voided session's late graded submission is not paid."""
    env = build(tmp_path)
    grant = open_(env)

    async def drain_then_submit():
        env.issuer.void_machine(MACHINE)
        await asyncio.sleep(0)
        await env.issuer.submitted(grant.session_id)

    asyncio.run(drain_then_submit())
    assert not sessions.session_submittable(VOIDED)
    assert env.book.get(grant.session_id).state == VOIDED
    assert env.store.documents[grant.session_id]["state"] == VOIDED
    assert SUBMITTED not in sandbox_store.SESSION_TRANSITIONS[VOIDED]
    # the late close sees the voided state; the miner may open again
    assert close(env, grant, final_of(env, grant, "graded"))["state"] == VOIDED
    assert isinstance(open_(env, request_id="b" * 32), Grant)


def test_an_aborted_session_is_never_paid(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    assert close(env, grant, final_of(env, grant, "aborted"))["state"] == ABORTED
    asyncio.run(env.issuer.submitted(grant.session_id))
    assert not sessions.session_submittable(ABORTED)
    assert env.book.get(grant.session_id).state == ABORTED
    assert env.store.documents[grant.session_id]["state"] == ABORTED
    assert SUBMITTED not in sandbox_store.SESSION_TRANSITIONS[ABORTED]


def test_a_close_racing_a_submission_reports_the_submission(tmp_path, monkeypatch):
    env = build(tmp_path)
    grant = open_(env)
    real = sessions.verify_transcript

    def verify_while_submitted(*args, **kwargs):
        result = real(*args, **kwargs)
        env.book.settle(grant.session_id, SUBMITTED, now=NOW, status="graded")
        return result

    monkeypatch.setattr(sessions, "verify_transcript", verify_while_submitted)
    closed = close(env, grant, final_of(env, grant, "graded"))
    assert closed["state"] == SUBMITTED
    assert env.book.get(grant.session_id).state == SUBMITTED


def test_a_close_is_verified_outside_the_issuer_lock(tmp_path, monkeypatch):
    env = build(tmp_path)
    grant = open_(env)
    real = sessions.verify_transcript
    held = []

    def verify(*args, **kwargs):
        held.append(env.issuer._lock.locked())
        return real(*args, **kwargs)

    monkeypatch.setattr(sessions, "verify_transcript", verify)
    close(env, grant, final_of(env, grant, "expired"))
    assert held == [False]


@pytest.mark.parametrize("store_kind", ["memory", "r2"])
def test_a_stale_lapse_never_overwrites_a_submission(tmp_path, monkeypatch, store_kind):
    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(sandbox_store, "get_s3_client", lambda **kw: fake)
    store = MemorySessionStore() if store_kind == "memory" else R2SessionStore()
    env = build(tmp_path, store=store)
    grant = open_(env)
    live = SessionRecord.from_document(asyncio.run(store.list_recent(NOW))[0])
    submitted = dataclasses.replace(live, state=SUBMITTED, closed_status="graded", closed_at=NOW)
    lapsed = dataclasses.replace(live, state=LAPSED, closed_at=NOW + 9999)
    asyncio.run(store.update(submitted.to_document()))
    asyncio.run(store.update(submitted.to_document()))           # the same write again: fine
    with pytest.raises(sandbox_store.SessionStoreConflict):
        asyncio.run(store.update(lapsed.to_document()))
    (stored,) = asyncio.run(store.list_recent(NOW))
    assert stored["state"] == SUBMITTED and stored["session_id"] == grant.session_id


def test_a_failed_persist_is_retried_then_alerts(tmp_path, monkeypatch, caplog):
    slept = []

    async def no_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(sessions, "_sleep", no_sleep)
    env = build(tmp_path)
    grant = open_(env)
    failures = {"n": 2}
    real_update = env.store.update

    async def flaky(document):
        if failures["n"]:
            failures["n"] -= 1
            raise OSError("bucket down")
        await real_update(document)

    env.store.update = flaky
    asyncio.run(env.issuer.submitted(grant.session_id))
    assert env.store.documents[grant.session_id]["state"] == SUBMITTED and len(slept) == 2

    second = open_(env, request_id="b" * 32, engagement={**CORPUS, "prompt_index": 4})
    assert isinstance(second, Grant), second
    env.store.fail = True
    env.store.update = real_update
    caplog.set_level("ERROR")
    asyncio.run(env.issuer.submitted(second.session_id))
    assert "ALERT" in caplog.text and second.session_id in caplog.text
    assert len(slept) == 2 + sessions.PERSIST_ATTEMPTS - 1


def test_a_request_id_reused_for_another_engagement_conflicts(tmp_path):
    env = build(tmp_path, remaining=5)
    first = open_(env)
    assert open_(env, engagement={**CORPUS, "prompt_index": 4}).reason == "request_conflict"
    assert open_(env) == first


def test_a_malformed_session_document_is_skipped_on_restore(tmp_path, caplog):
    store = MemorySessionStore()
    env = build(tmp_path, store=store)
    grant = open_(env)
    good = dict(store.documents[grant.session_id])
    for n, change in enumerate(({"index": "3"}, {"issued_at": True}, {"state": "weird"},
                                {"job_id": 7}, {"hotkey": None})):
        store.documents[f"bad-{n}"] = {**good, "session_id": f"bad-{n}", **change}
    again = build(tmp_path / "b", store=store)
    assert asyncio.run(again.issuer.restore()) == 1
    again.issuer._new_id = lambda: "s-after-restart"          # the first run used s-0
    assert isinstance(open_(again, hotkey="5Other", request_id="b" * 32), Grant)


def test_voided_sessions_do_not_count_toward_the_open_rate(tmp_path):
    env = build(tmp_path, policy=SandboxPolicy(max_opens_per_hour=1), remaining=5)
    open_(env)

    async def drain():
        env.issuer.void_machine(MACHINE)
        await asyncio.sleep(0)

    asyncio.run(drain())
    assert isinstance(open_(env, request_id="b" * 32), Grant)


def test_a_ledger_that_hangs_is_refused_retryably(tmp_path):
    env = build(tmp_path, policy=SandboxPolicy(io_timeout_s=1))

    async def hang(index):
        await asyncio.sleep(30)

    view = SignedJobView(job=JOB, resolve_task=None, slots_remaining=hang)
    env.issuer._engagements["corpus"] = CorpusEngagements({JOB.job_id: view}.get, env.book,
                                                          env.clock)
    refused = open_(env)
    assert refused.reason == "ledger_unavailable" and refused.retry_after == 10


def test_the_swe_task_resolver_cache_is_bounded_and_not_on_the_class():
    from reliquary.sandbox.tasks import SweTaskResolver

    calls = []

    def sandbox_task(split, index):
        calls.append(index)
        return SimpleNamespace(image=IMAGE, limits=None)

    resolver = SweTaskResolver("train:20", sandbox_task=sandbox_task, cache_size=2)
    for index in (1, 2, 1, 3, 1, 2):
        asyncio.run(resolver.resolve(index))
    assert calls == [1, 2, 3, 2]



# -- fix round 2 ---------------------------------------------------------------------

def test_an_open_failed_close_can_never_be_paid_later(tmp_path):
    """Probe 7: hold a graded transcript, close `open_failed` to free the slot, let
    another miner take it, then submit. The late submission is refused, and the other
    miner's reservation stands."""
    env = build(tmp_path, remaining=1)
    grant = open_(env)
    graded = final_of(env, grant, "graded")                       # kept by the miner
    assert close(env, grant, None, reason="open_failed")["state"] == CLOSED
    other = open_(env, hotkey="5Other", request_id="b" * 32)
    assert isinstance(other, Grant)
    assert graded["records"][-1]["body"]["status"] == "graded"
    assert not sessions.session_submittable(env.book.get(grant.session_id).state)
    asyncio.run(env.issuer.submitted(grant.session_id))           # even if called anyway
    assert env.book.get(grant.session_id).state == CLOSED
    assert env.store.documents[grant.session_id]["state"] == CLOSED
    assert env.book.get(other.session_id).state == LIVE
    assert env.book.reserved(JOB.job_id, 3, NOW) == 1


@pytest.mark.parametrize("state,submittable", [
    (LIVE, True), (CLOSED_GRADED, True), (SUBMITTED, False), (CLOSED, False),
    (ABORTED, False), (VOIDED, False), (LAPSED, False), ("unknown", False), (None, False),
])
def test_only_a_live_or_graded_closed_session_is_submittable(state, submittable):
    assert sessions.session_submittable(state) is submittable


def test_only_a_live_or_graded_closed_session_becomes_submitted_in_the_store():
    assert {state for state, moves in sandbox_store.SESSION_TRANSITIONS.items()
            if SUBMITTED in moves} == {LIVE, CLOSED_GRADED}
    assert set(sandbox_store.SESSION_TRANSITIONS) == {LIVE, CLOSED_GRADED, SUBMITTED, CLOSED,
                                                       ABORTED, VOIDED, LAPSED}
