"""RL sandbox sessions bound to an episode precommit."""
import asyncio
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.infrastructure import sandbox_store  # noqa: E402
from reliquary.infrastructure.sandbox_store import (  # noqa: E402
    RL_SESSION_PREFIX, SESSION_PREFIX, MemorySessionStore, R2SessionStore,
)
from reliquary.protocol.service_episode import rl_engagement  # noqa: E402
from reliquary.sandbox.rl_engagements import EpisodeEnvironmentView, RlEpisodeEngagements  # noqa: E402
from reliquary.sandbox.sessions import (  # noqa: E402
    ABORTED, CLOSED, VOIDED, Grant, Refusal, SandboxPolicy, SessionBook, SessionIssuer,
)
from reliquary.sandbox.tasks import ResolvedTask  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    BUDGETS, ENV_PACKAGE, EPISODE, REVISION, SANDBOX_ENV, SPLIT, TASK, episode_contract, episode_precommit,
)
from tests.unit.sandbox_fixtures import IMAGE, MACHINE, NOW, directory, signer, transcript  # noqa: E402
from tests.unit.test_sandbox_sessions import Clock, FakeFleet  # noqa: E402

HOTKEY = "5Hot"
CONTRACT = episode_contract()
POLICY = CONTRACT.episode_policy(EPISODE)


class Quota:
    def __init__(self, refusal=None):
        self.calls, self.refusal = [], refusal

    async def admit_open(self, hotkey, *, environment, precommit_sha256):
        self.calls.append((hotkey, environment, precommit_sha256))
        return self.refusal


def build(tmp_path, *, store=None, quota=None, limits=None, precommits=None, on_resolve=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    validator, machine = signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1")
    policy = SandboxPolicy(max_live_per_hotkey=POLICY.pool_seeds + 1)
    book = SessionBook(policy)
    precommit = episode_precommit(CONTRACT, hotkey=HOTKEY)
    precommits_ = {precommit.sha256: precommit}
    current = {"window": 1}
    recorded = {precommit.sha256: float(NOW)}
    wall = {"now": float(NOW)}

    async def resolve(index):
        if on_resolve is not None:
            on_resolve(current)
        return ResolvedTask(IMAGE, dict(limits or {}))

    views = {EPISODE: EpisodeEnvironmentView(policy=POLICY, resolve_task=resolve)}
    ids = iter(f"s-{tmp_path.name}-{n}" for n in range(10_000))
    engagements = RlEpisodeEngagements(precommits=precommits or precommits_.get, environments=views.get,
                                       current_window=lambda: current["window"], book=book, quota=quota,
                                       recorded_at=recorded.get, clock=lambda: wall["now"])
    issuer = SessionIssuer(
        book=book, store=store if store is not None else MemorySessionStore(),
        fleet=FakeFleet(directory(machine)), signer=validator,
        token_verifier=attest.Ed25519TokenVerifier({"v1": validator.public_key_b64}),
        engagements={"rl_precommit": engagements}, policy=policy, clock=Clock(),
        new_session_id=lambda: next(ids))
    return SimpleNamespace(issuer=issuer, book=book, precommit=precommit, current=current, recorded=recorded, wall=wall,
                           validator=validator, machine=machine)


def engagement(precommit, seed):
    return {"kind": "rl_precommit", "precommit": {"precommit_sha256": precommit.sha256, "seed_index": seed}}


def open_(env, seed, *, request_id=None, hotkey=HOTKEY, precommit=None):
    return asyncio.run(env.issuer.open(hotkey=hotkey, request_id=request_id or f"{seed:032x}",
                                       engagement=engagement(precommit or env.precommit, seed)))


def close(env, grant, status=None, reason="final"):
    session = attest.SessionClaims.from_dict(grant.token["claims"])
    signed = None if reason == "open_failed" else transcript(env.validator, env.machine, session,
                                                             status=status, env_package=ENV_PACKAGE)
    return asyncio.run(env.issuer.close(hotkey=HOTKEY, session_id=grant.session_id, reason=reason,
                                        transcript=signed))


def test_a_grant_binds_precommit_seed_task_checkpoint_and_raises_budgets(tmp_path):
    env = build(tmp_path, limits={"wall_s": 7200})
    grant = open_(env, 5)
    assert isinstance(grant, Grant)
    claims = attest.SessionClaims.from_dict(grant.token["claims"])
    assert claims.engagement == rl_engagement(1, env.precommit.sha256, 5)
    assert (claims.hotkey, claims.env, claims.split, claims.index, claims.checkpoint) == (
        HOTKEY, SANDBOX_ENV, SPLIT, TASK, REVISION)
    assert claims.image == IMAGE
    assert claims.budgets.wall_s == 7200 and claims.budgets.max_calls == BUDGETS["max_calls"]
    record = env.book.get(grant.session_id)
    assert record.kind == "rl_precommit" and record.job_id is None and record.prompt_index is None


def test_every_seed_of_the_pool_opens_once_and_no_seed_outside_it(tmp_path):
    env = build(tmp_path)
    assert all(isinstance(open_(env, seed), Grant) for seed in range(POLICY.pool_seeds))
    assert open_(env, POLICY.pool_seeds).reason == "seed_out_of_pool"
    assert open_(env, 0, request_id="f" * 32).reason == "engagement_taken"


def test_two_concurrent_opens_of_one_seed_get_one_grant(tmp_path):
    env = build(tmp_path)

    async def both():
        return await asyncio.gather(
            env.issuer.open(hotkey=HOTKEY, request_id="a" * 32, engagement=engagement(env.precommit, 3)),
            env.issuer.open(hotkey=HOTKEY, request_id="b" * 32, engagement=engagement(env.precommit, 3)))

    outcomes = asyncio.run(both())
    assert sum(isinstance(o, Grant) for o in outcomes) == 1
    assert [o.reason for o in outcomes if isinstance(o, Refusal)] == ["engagement_taken"]


def test_only_our_faults_free_a_seed(tmp_path):
    env = build(tmp_path)
    assert close(env, open_(env, 1), "aborted")["state"] == ABORTED
    assert isinstance(open_(env, 1, request_id="1" * 32), Grant)                  # our fault: reopened
    assert close(env, open_(env, 2), reason="open_failed")["state"] == CLOSED
    assert open_(env, 2, request_id="2" * 32).reason == "engagement_taken"        # no re-roll
    assert close(env, open_(env, 3), "expired")["state"] == CLOSED
    assert open_(env, 3, request_id="3" * 32).reason == "engagement_taken"
    drained = open_(env, 4)

    async def drain():
        env.issuer.void_machine(MACHINE)
        await asyncio.sleep(0)

    asyncio.run(drain())
    assert env.book.get(drained.session_id).state == VOIDED
    assert open_(env, 4, request_id="4" * 32).reason == "engagement_taken"        # a drain frees nothing (M1)


@pytest.mark.parametrize("window", [2, None])
def test_a_precommit_of_another_window_opens_nothing(tmp_path, window):
    env = build(tmp_path)
    env.current["window"] = window
    assert open_(env, 0).reason == "precommit_stale"


def test_another_hotkey_or_an_unrecorded_precommit_opens_nothing(tmp_path):
    env = build(tmp_path)
    assert open_(env, 0, hotkey="5Other").reason == "precommit_unknown"
    stranger = episode_precommit(CONTRACT, hotkey=HOTKEY, task=TASK + 1)
    assert open_(env, 0, precommit=stranger).reason == "precommit_unknown"


def test_a_malformed_engagement_is_refused(tmp_path):
    env = build(tmp_path)
    for value in ({"precommit_sha256": "x", "seed_index": 0},
                  {"precommit_sha256": env.precommit.sha256, "seed_index": True},
                  {"precommit_sha256": env.precommit.sha256}):
        outcome = asyncio.run(env.issuer.open(hotkey=HOTKEY, request_id="9" * 32,
                                              engagement={"kind": "rl_precommit", "precommit": value}))
        assert outcome.reason == "engagement_kind_unsupported"


def test_the_quota_hook_is_asked_and_can_refuse(tmp_path):
    quota = Quota()
    env = build(tmp_path, quota=quota)
    assert isinstance(open_(env, 0), Grant)
    assert quota.calls == [(HOTKEY, EPISODE, env.precommit.sha256)]
    quota.refusal = Refusal("quota_exhausted", {"live": 1}, retry_after=30)
    refused = open_(env, 1)
    assert (refused.reason, refused.retry_after) == ("quota_exhausted", 30)


def test_a_restart_keeps_every_seed_taken(tmp_path):
    store = MemorySessionStore()
    first = build(tmp_path / "a", store=store)
    grant = open_(first, 6)
    again = build(tmp_path / "b", store=store)
    assert asyncio.run(again.issuer.restore()) == 1
    assert again.book.get(grant.session_id) is not None
    assert open_(again, 6, request_id="6" * 32).reason == "engagement_taken"


def test_rl_sessions_live_under_their_own_prefix(tmp_path, monkeypatch):
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(sandbox_store, "get_s3_client", lambda **kw: fake)
    env = build(tmp_path, store=R2SessionStore(prefix=RL_SESSION_PREFIX))
    grant = open_(env, 0)
    assert list(fake.objects) == [sandbox_store.session_key(grant.session_id, grant.expires_at, RL_SESSION_PREFIX)]
    assert not list(fake.objects)[0].startswith(SESSION_PREFIX)
    assert sandbox_store.session_key("s-1", NOW) == sandbox_store.session_key("s-1", NOW, SESSION_PREFIX)
    assert asyncio.run(R2SessionStore().list_recent(grant.expires_at)) == []
    assert len(asyncio.run(R2SessionStore(prefix=RL_SESSION_PREFIX).list_recent(grant.expires_at))) == 1


def test_a_window_turning_while_the_task_resolves_opens_nothing(tmp_path):
    env = build(tmp_path, on_resolve=lambda current: current.update(window=2))
    assert open_(env, 0).reason == "precommit_stale"
    assert not env.book._sessions and not env.book.engagement_held(
        rl_engagement(1, env.precommit.sha256, 0))


def test_an_engagement_with_extra_fields_is_refused(tmp_path):
    env = build(tmp_path)
    for extra in ({"job_id": "swe"}, {"prompt_index": 0}):
        outcome = asyncio.run(env.issuer.open(hotkey=HOTKEY, request_id="8" * 32,
                                              engagement={**engagement(env.precommit, 0), **extra}))
        assert outcome.reason == "engagement_kind_unsupported"
    assert isinstance(open_(env, 0), Grant)


def test_a_graded_withdrawn_or_lapsed_session_keeps_its_seed(tmp_path):
    env = build(tmp_path)
    assert close(env, open_(env, 1), "graded")["state"] == "closed_graded"
    assert open_(env, 1, request_id="1" * 32).reason == "engagement_taken"
    assert close(env, open_(env, 2), "graded", reason="withdraw")["state"] == CLOSED
    assert open_(env, 2, request_id="2" * 32).reason == "engagement_taken"
    lapsing = open_(env, 3)
    env.issuer._clock.now = lapsing.expires_at + 10_000
    asyncio.run(env.issuer.maintain())
    assert env.book.get(lapsing.session_id).state == "lapsed"
    assert open_(env, 3, request_id="3" * 32).reason == "engagement_taken"


def test_a_restart_keeps_a_failed_open_taken_and_an_abort_free(tmp_path):
    store = MemorySessionStore()
    first = build(tmp_path / "a", store=store)
    close(first, open_(first, 1), reason="open_failed")
    close(first, open_(first, 2), "aborted")
    again = build(tmp_path / "b", store=store)
    assert asyncio.run(again.issuer.restore()) == 2
    assert open_(again, 1, request_id="1" * 32).reason == "engagement_taken"
    assert isinstance(open_(again, 2, request_id="2" * 32), Grant)


def test_an_unreadable_precommit_or_a_failing_quota_refuses_retryably(tmp_path):
    def broken(sha):
        raise OSError("db locked")

    env = build(tmp_path, precommits=broken)
    refused = open_(env, 0)
    assert refused.reason == "ledger_unavailable" and refused.retry_after

    class Failing:
        async def admit_open(self, hotkey, *, environment, precommit_sha256):
            raise RuntimeError("quota down")

    env = build(tmp_path / "q", quota=Failing())
    refused = open_(env, 0)
    assert refused.reason == "ledger_unavailable" and refused.retry_after
    assert not env.book._sessions


def test_the_corpus_stub_is_untouched():
    from reliquary.sandbox.sessions import RlPrecommitEngagements

    refused = asyncio.run(RlPrecommitEngagements().terms(HOTKEY, {"kind": "rl_precommit"}))
    assert refused.reason == "engagement_kind_unsupported"


def test_a_drain_frees_no_seed_whatever_the_miner_left_open(tmp_path):
    env = build(tmp_path)
    good, left = open_(env, 1), open_(env, 2)
    assert close(env, good, "graded")["state"] == "closed_graded"

    async def drain():
        env.issuer.void_machine(MACHINE)
        await asyncio.sleep(0)

    asyncio.run(drain())
    assert env.book.get(left.session_id).state == VOIDED
    assert open_(env, 1, request_id="1" * 32).reason == "engagement_taken"
    assert open_(env, 2, request_id="2" * 32).reason == "engagement_taken"


def test_a_seed_is_freed_by_an_abort_only_once(tmp_path):
    env = build(tmp_path)
    assert close(env, open_(env, 1), "aborted")["state"] == ABORTED
    second = open_(env, 1, request_id="1" * 32)
    assert isinstance(second, Grant)
    assert close(env, second, "aborted")["state"] == ABORTED
    assert open_(env, 1, request_id="2" * 32).reason == "engagement_taken"
    assert isinstance(open_(env, 2), Grant)                       # another seed keeps its own count


def test_a_precommit_recorded_more_than_24h_ago_opens_nothing(tmp_path):
    env = build(tmp_path)
    env.wall["now"] = NOW + 86_400
    assert isinstance(open_(env, 0), Grant)
    env.wall["now"] = NOW + 86_401
    assert open_(env, 1).reason == "precommit_stale"
    env.wall["now"] = NOW
    env.recorded.clear()                                          # no recorded age: fail closed
    assert open_(env, 2).reason == "precommit_stale"


def test_a_session_longer_than_the_restore_horizon_is_refused_at_open(tmp_path):
    over = 86_400 - SandboxPolicy().open_window_s
    env = build(tmp_path, limits={"wall_s": over + 1})
    refused = open_(env, 0)
    assert refused.reason == "session_too_long" and not env.book._sessions
    env = build(tmp_path / "ok", limits={"wall_s": over})
    assert isinstance(open_(env, 0), Grant)
