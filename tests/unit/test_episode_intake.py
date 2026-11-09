"""Admission wiring for signed-episode groups."""
import asyncio
import dataclasses
import inspect
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.constants import M_ROLLOUTS  # noqa: E402
from reliquary.protocol.submission import RejectReason  # noqa: E402
from reliquary.validator import admission
from reliquary.validator.admission import AdmissionContext, AdmissionProblemMaterials
from reliquary.validator.episode_admission import EpisodeGroupChecker  # noqa: E402
from tests.unit.episode_v2_fixtures import EPISODE, WINDOW_BEACON, FixedSource, episode_contract, episode_group, episode_signers, register_episode_env
from tests.unit.sandbox_fixtures import NOW, directory  # noqa: E402
from tests.unit.service_v2_fixtures import MATH  # noqa: E402
from tests.unit.test_trajectory_parse import FakeRenderer  # noqa: E402

CONTRACT = episode_contract()
POLICY = CONTRACT.episode_policy(EPISODE)


class Sessions:
    """The issuer's surface the intake uses; ``claim_all`` is all-or-none like ``SessionIssuer.claim_all``."""

    policy = SimpleNamespace(claim_ttl_s=300)

    def __init__(self, refuse=None):
        self.claimed, self.released, self.paid, self.held = [], [], [], set()
        self.refuse = dict(refuse or {})
        self.fail_release = False
        self.group_keys = []
        self.paid_calls = 0
        self.claim_gate = self.release_gate = None      # asyncio.Event: hold the call until set
        self.persisted, self.persist_ok, self.events = [], None, []   # persist_ok: how many records store
        self.persist_gate = None                                         # asyncio.Event: a slow store
        self.persist_raises = False

    async def claim_all(self, session_ids, *, hotkey, received=None, precommit_sha256=None):
        self.group_keys.append(precommit_sha256)
        if self.claim_gate is not None:
            await self.claim_gate.wait()
        for session_id in session_ids:
            if session_id in self.refuse:
                return session_id, SimpleNamespace(reason=self.refuse[session_id], retry_after=None, detail={})
        self.claimed.extend(session_ids)
        self.held.update(session_ids)
        return None

    async def release_claim(self, session_id):
        if self.release_gate is not None:
            await self.release_gate.wait()
        if self.fail_release:
            raise RuntimeError("store down")
        self.released.append(session_id)
        self.held.discard(session_id)
        self.events.append("release")

    async def persist_submitted(self, session_ids):
        if self.persist_gate is not None:
            await self.persist_gate.wait()
        if self.persist_raises:
            raise RuntimeError("store down")
        self.persisted.extend(session_ids)
        self.events.append("persist")
        return len(session_ids) if self.persist_ok is None else self.persist_ok

    async def submitted_all(self, session_ids):
        self.events.append("paid")
        self.paid_calls += 1
        self.paid.extend(session_ids)
        self.held.difference_update(session_ids)


class Outcomes:
    def __init__(self):
        self.calls = []

    def group_settled(self, **kwargs):
        self.calls.append(kwargs)


def admit(w, environment=EPISODE):
    return asyncio.run(w.intake.admit(environment=environment, prepared=w.prepared, received=NOW + 100,
                                      contract=CONTRACT))


def run_server_admit(server, batcher, w):
    return asyncio.run(server._admit_episode_group(batcher, SimpleNamespace(environment=EPISODE), w.prepared,
                                                   SimpleNamespace(t_body_completed=NOW + 100)))


# -- the real auction path: settlement by the batcher's answer, after the bookkeeping ------------------

async def run_auction(a):
    try:
        await a.server._process_auction_submission(a.item, asyncio.Queue())
    finally:
        a.pool.shutdown(wait=True)


RETRYABLE_CASES = [
    (RejectReason.RATE_LIMITED, "episode_group_in_flight", False),
    (RejectReason.WORKER_DROPPED, "episode_checker_busy", False),
    (RejectReason.WORKER_DROPPED, "episode_session_busy", False),
    (RejectReason.WORKER_DROPPED, "episode_directory", False),
    (RejectReason.WORKER_DROPPED, "episode_persist_failed", False),
    (RejectReason.RATE_LIMITED, "episode_checks_in_flight", False),    # retryable, never refunded
]


# -- bounds on the parent-side check ---------------------------------------------------------------------

# -- the issuer's all-or-none group claim ------------------------------------------------------------

def _issuer_with_sessions(tmp_path, count):
    from tests.unit.test_sandbox_sessions import build, open_

    env = build(tmp_path)
    grant = open_(env)
    record = env.book.get(grant.session_id)
    ids = [grant.session_id]
    for n in range(1, count):
        clone = dataclasses.replace(record, session_id=f"g-{n}", request_id=f"{n:032d}")
        env.book.add(clone)
        ids.append(clone.session_id)
    return env, ids


def test_the_issuer_claims_a_group_all_or_none(tmp_path):
    from reliquary.sandbox.sessions import SUBMITTED

    env, ids = _issuer_with_sessions(tmp_path, 3)
    assert asyncio.run(env.issuer.claim_all(ids, hotkey="5Hot", received=NOW)) is None
    assert all(env.book.is_claimed(session_id) for session_id in ids)
    for session_id in ids:
        asyncio.run(env.issuer.release_claim(session_id))
    # One session already paid in another group: nothing of this group stays claimed.
    env.book.add(dataclasses.replace(env.book.get(ids[2]), state=SUBMITTED))
    refused = asyncio.run(env.issuer.claim_all(ids, hotkey="5Hot", received=NOW))
    assert refused[0] == ids[2] and refused[1].reason == "session_submitted"
    assert not any(env.book.is_claimed(session_id) for session_id in ids)


def test_a_session_claimed_by_another_group_in_flight_refuses_the_whole_group(tmp_path):
    env, ids = _issuer_with_sessions(tmp_path, 3)
    assert asyncio.run(env.issuer.claim(ids[1], hotkey="5Hot", received=NOW)) is None
    refused = asyncio.run(env.issuer.claim_all(ids, hotkey="5Hot", received=NOW))
    assert refused[0] == ids[1] and refused[1].reason == "session_claimed"
    assert not env.book.is_claimed(ids[0]) and not env.book.is_claimed(ids[2])
    assert env.book.is_claimed(ids[1])           # the other group's claim is untouched
    twice = asyncio.run(env.issuer.claim_all([ids[0], ids[0]], hotkey="5Hot", received=NOW))
    assert twice[1].reason == "session_claimed" and not env.book.is_claimed(ids[0])


def test_submitted_records_are_stored_before_the_batcher_takes_the_group(tmp_path, monkeypatch):
    from reliquary.protocol.service_episode import rl_engagement
    from reliquary.sandbox import sessions as sessions_module
    from reliquary.sandbox.sessions import LIVE, SUBMITTED

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(sessions_module, "_sleep", no_sleep)
    env, ids = _issuer_with_sessions(tmp_path, 4)
    sha = "9" * 64
    for n, session_id in enumerate(ids):
        record = dataclasses.replace(env.book.get(session_id), engagement=rl_engagement(1, sha, n),
                                     kind="rl_precommit")
        env.book.add(record)
        env.store.documents[session_id] = record.to_document()
    group, unclaimed = ids[:3], ids[3]
    assert asyncio.run(env.issuer.claim_all(group, hotkey="5Hot", received=NOW, precommit_sha256=sha)) is None
    env.store.fail = True
    assert asyncio.run(env.issuer.persist_submitted(group + [unclaimed])) == 0
    assert all(env.store.documents[i]["state"] == LIVE for i in ids)
    env.store.fail = False
    assert asyncio.run(env.issuer.persist_submitted(group + [unclaimed])) == 3
    assert all(env.store.documents[i]["state"] == SUBMITTED for i in group)
    assert env.store.documents[unclaimed]["state"] == LIVE          # never claimed: never written
    assert all(env.book.get(i).state == LIVE and env.book.is_claimed(i) for i in group)   # book unchanged
    # The batcher refused: the claims are released, the store keeps ``submitted``; a retry counts them.
    for session_id in group:
        asyncio.run(env.issuer.release_claim(session_id))
    assert asyncio.run(env.issuer.claim_all(group, hotkey="5Hot", received=NOW, precommit_sha256=sha)) is None
    writes = []
    real_update = env.store.update

    async def counted(document):
        writes.append(document["session_id"])
        return await real_update(document)

    monkeypatch.setattr(env.store, "update", counted)
    assert asyncio.run(env.issuer.persist_submitted(group)) == 3
    asyncio.run(env.issuer.submitted_all(group))
    assert writes == [] and all(env.book.get(i).state == SUBMITTED for i in group)


def test_one_paid_group_per_precommit_even_on_disjoint_seeds(tmp_path):
    from reliquary.protocol.service_episode import rl_engagement
    from reliquary.sandbox.sessions import SUBMITTED

    env, ids = _issuer_with_sessions(tmp_path, 4)
    sha = "a" * 64
    for n, session_id in enumerate(ids):
        env.book.add(dataclasses.replace(env.book.get(session_id), engagement=rl_engagement(1, sha, n),
                                         kind="rl_precommit"))
    first, second = ids[:2], ids[2:]
    assert asyncio.run(env.issuer.claim_all(first, hotkey="5Hot", received=NOW, precommit_sha256=sha)) is None
    busy = asyncio.run(env.issuer.claim_all(second, hotkey="5Hot", received=NOW, precommit_sha256=sha))
    assert busy[1].reason == "precommit_claimed" and busy[1].retry_after
    assert not any(env.book.is_claimed(session_id) for session_id in second)
    env.book.add(dataclasses.replace(env.book.get(first[0]), state=SUBMITTED))
    for session_id in first:
        asyncio.run(env.issuer.release_claim(session_id))
    used = asyncio.run(env.issuer.claim_all(second, hotkey="5Hot", received=NOW, precommit_sha256=sha))
    assert used[1].reason == "precommit_submitted"
    assert not any(env.book.is_claimed(session_id) for session_id in second)
    # Another precommit is not affected.
    assert asyncio.run(env.issuer.claim_all(second, hotkey="5Hot", received=NOW, precommit_sha256="b" * 64)) is None


def test_a_paid_group_keeps_its_precommit_taken_across_a_restart(tmp_path):
    from reliquary.protocol.service_episode import rl_engagement
    from reliquary.sandbox.sessions import SUBMITTED
    from tests.unit.test_sandbox_sessions import build

    env, ids = _issuer_with_sessions(tmp_path, 4)
    sha = "c" * 64
    for n, session_id in enumerate(ids):
        record = dataclasses.replace(env.book.get(session_id), engagement=rl_engagement(1, sha, n),
                                     kind="rl_precommit")
        env.book.add(record)
        env.store.documents[session_id] = record.to_document()
    first, second = ids[:2], ids[2:]
    assert asyncio.run(env.issuer.claim_all(first, hotkey="5Hot", received=NOW, precommit_sha256=sha)) is None
    asyncio.run(env.issuer.submitted_all(first))
    # Every session of the paid group is written before submitted_all returns.
    assert all(env.store.documents[session_id]["state"] == SUBMITTED for session_id in first)
    restarted = build(tmp_path / "restart", store=env.store)
    asyncio.run(restarted.issuer.restore())
    used = asyncio.run(restarted.issuer.claim_all(second, hotkey="5Hot", received=NOW, precommit_sha256=sha))
    assert used is not None and used[1].reason == "precommit_submitted"
    assert not any(restarted.book.is_claimed(session_id) for session_id in second)


def test_only_the_submitters_own_sessions_hold_its_precommit(tmp_path):
    from reliquary.protocol.service_episode import rl_engagement
    from reliquary.sandbox.sessions import SUBMITTED

    env, ids = _issuer_with_sessions(tmp_path, 3)
    sha = "d" * 64
    for n, session_id in enumerate(ids):
        env.book.add(dataclasses.replace(env.book.get(session_id), engagement=rl_engagement(1, sha, n),
                                         kind="rl_precommit"))
    env.book.add(dataclasses.replace(env.book.get(ids[2]), hotkey="5Other", state=SUBMITTED))
    assert asyncio.run(env.issuer.claim_all(ids[:2], hotkey="5Hot", received=NOW, precommit_sha256=sha)) is None


def test_the_book_indexes_sessions_by_precommit(tmp_path, monkeypatch):
    from reliquary.protocol.service_episode import rl_engagement

    env, ids = _issuer_with_sessions(tmp_path, 3)
    sha, other = "e" * 64, "f" * 64
    for n, session_id in enumerate(ids):
        env.book.add(dataclasses.replace(env.book.get(session_id), engagement=rl_engagement(1, sha, n),
                                         kind="rl_precommit"))
    assert [r.session_id for r in env.book.of_precommit(sha)] == sorted(ids)
    env.book.add(dataclasses.replace(env.book.get(ids[0]), engagement=rl_engagement(1, other, 0)))
    assert {r.session_id for r in env.book.of_precommit(sha)} == set(ids[1:])
    assert [r.session_id for r in env.book.of_precommit(other)] == [ids[0]]
    assert env.book.of_precommit("0" * 64) == ()
    # The group claim reads the index, never the whole book.
    monkeypatch.setattr(env.book, "records", lambda: (_ for _ in ()).throw(AssertionError("full scan")))
    assert asyncio.run(env.issuer.claim_all(ids[1:], hotkey="5Hot", received=NOW, precommit_sha256=sha)) is None


# -- in-flight caps --------------------------------------------------------------------------------

def stages_of(a):
    stages, real = [], a.server._record_raw_terminal

    def spy(*args, **kwargs):
        stages.append(kwargs.get("stage"))
        return real(*args, **kwargs)

    a.server._record_raw_terminal = spy
    return stages


def test_a_cut_persist_still_notes_what_it_stored(tmp_path, monkeypatch):
    from reliquary.protocol.service_episode import rl_engagement
    from reliquary.sandbox.sessions import SUBMITTED

    env, ids = _issuer_with_sessions(tmp_path, 2)
    sha = "8" * 64
    for n, session_id in enumerate(ids):
        record = dataclasses.replace(env.book.get(session_id), engagement=rl_engagement(1, sha, n),
                                     kind="rl_precommit")
        env.book.add(record)
        env.store.documents[session_id] = record.to_document()

    async def scenario():
        assert await env.issuer.claim_all(ids, hotkey="5Hot", received=NOW, precommit_sha256=sha) is None
        gate = asyncio.Event()
        real_update = env.store.update

        async def slow(document):
            await gate.wait()
            return await real_update(document)

        monkeypatch.setattr(env.store, "update", slow)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(env.issuer.persist_submitted(ids), 0.1)
        gate.set()
        await asyncio.gather(*list(env.issuer._tasks))
        assert env.issuer._stored_submitted == set(ids)
        assert all(env.store.documents[i]["state"] == SUBMITTED for i in ids)
        assert await env.issuer.persist_submitted(ids) == len(ids)       # counted, not rewritten

    asyncio.run(scenario())


def test_the_stored_submitted_set_is_pruned_with_the_book(tmp_path):
    from reliquary.sandbox.sessions import SUBMITTED

    env, ids = _issuer_with_sessions(tmp_path, 2)
    old, kept = ids
    env.book.add(dataclasses.replace(env.book.get(old), state=SUBMITTED, closed_at=NOW - 10))
    env.issuer._stored_submitted.update(ids)
    env.clock.now = NOW + 26 * 3600
    asyncio.run(env.issuer.maintain())
    assert env.book.get(old) is None and env.book.get(kept) is not None
    assert old not in env.issuer._stored_submitted         # kept lapsed too: no longer stored as submitted
    assert kept not in env.issuer._stored_submitted


def test_a_lapsed_session_is_no_longer_counted_as_stored_submitted(tmp_path):
    from reliquary.protocol.service_episode import rl_engagement
    from reliquary.sandbox.sessions import LAPSED, SUBMITTED

    env, ids = _issuer_with_sessions(tmp_path, 1)
    sha = "9" * 64
    record = dataclasses.replace(env.book.get(ids[0]), engagement=rl_engagement(1, sha, 0),
                                 kind="rl_precommit")
    env.book.add(record)
    env.store.documents[ids[0]] = record.to_document()

    async def scenario():
        assert await env.issuer.claim_all(ids, hotkey="5Hot", received=NOW, precommit_sha256=sha) is None
        assert await env.issuer.persist_submitted(ids) == 1
        assert env.store.documents[ids[0]]["state"] == SUBMITTED
        env.clock.now = NOW + 26 * 3600
        await env.issuer.maintain()
        assert env.book.get(ids[0]).state == LAPSED
        assert ids[0] not in env.issuer._stored_submitted      # the stored claim no longer stands
        calls = []
        real_update = env.store.update

        async def counting(document):
            calls.append(document["state"])
            return await real_update(document)

        env.store.update = counting
        await env.issuer.persist_submitted(ids)
        assert calls == [SUBMITTED]                             # rewritten, not counted as stored

    asyncio.run(scenario())
