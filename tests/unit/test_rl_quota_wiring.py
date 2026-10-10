"""The yield quota behind the RL engagements' quota hook: checked again under the issuer lock, sized by the
fleet, its index fed by the issuer's events and rebuilt at start."""
import asyncio
import dataclasses
import time

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.fleet import HEARTBEAT_MAX_AGE_S  # noqa: E402
from reliquary.sandbox.rl_quota import YieldQuota  # noqa: E402
from reliquary.sandbox.routes import REFUSAL_STATUS  # noqa: E402
from reliquary.sandbox.sessions import LIVE, FanOutEvents, Grant, Refusal  # noqa: E402
from reliquary.validator.rl_sandbox_wiring import NO_PER_KEY_CAP  # noqa: E402
from tests.unit.episode_v2_fixtures import EPISODE, SANDBOX_ENV  # noqa: E402
from tests.unit.sandbox_fixtures import NOW  # noqa: E402
from tests.unit.test_rl_engagements import HOTKEY, POLICY, build, close, open_  # noqa: E402
from tests.unit.test_rl_episode_wiring import _record, _services  # noqa: E402


class OneLive:
    """A quota of one live session, counted from the book: what matters is WHERE it is asked."""

    def __init__(self, book):
        self.book, self.locked = book, []

    async def admit_open(self, hotkey, *, environment, precommit_sha256, seed_index=None):
        return None                                              # the pre-check lets both through

    def admit_locked(self, hotkey, now, *, precommit_sha256=None, seed_index=None, environment=None):
        self.locked.append((precommit_sha256, environment))
        live = sum(1 for r in self.book.records() if r.state == LIVE)
        return Refusal("quota_live", {"live": live, "max": 1}) if live >= 1 else None


class Ledger:
    def snapshot(self):
        return {}


def test_the_quota_is_checked_again_under_the_issuer_lock_with_the_precommit(tmp_path):
    env = build(tmp_path)
    quota = OneLive(env.book)
    env.issuer._engagements["rl_precommit"]._quota = quota

    async def two():
        engagement = lambda seed: {"kind": "rl_precommit",  # noqa: E731
                                   "precommit": {"precommit_sha256": env.precommit.sha256, "seed_index": seed}}
        return await asyncio.gather(
            env.issuer.open(hotkey=HOTKEY, request_id="a" * 32, engagement=engagement(0)),
            env.issuer.open(hotkey=HOTKEY, request_id="b" * 32, engagement=engagement(1)))

    outcomes = asyncio.run(two())
    assert sum(isinstance(o, Grant) for o in outcomes) == 1
    assert [o.reason for o in outcomes if isinstance(o, Refusal)] == ["quota_live"]
    assert set(quota.locked) == {(env.precommit.sha256, EPISODE)}     # the env a new group's M is read from


def test_the_books_own_caps_read_the_hotkeys_index_not_the_whole_book(tmp_path):
    env = build(tmp_path)
    for seed in range(3):
        open_(env, seed)

    class NoScan(dict):
        def values(self):
            raise AssertionError("scanned the whole book")

    env.book._sessions = NoScan(env.book._sessions)
    assert env.book.open_refusal(HOTKEY, int(time.time())) is None
    assert env.book.open_refusal("5Nobody", int(time.time())) is None
    assert len(env.book._by_hotkey[HOTKEY]) == 3


def test_the_hotkeys_index_follows_the_prune(tmp_path):
    env = build(tmp_path)
    grant = open_(env, 0)
    close(env, grant, "aborted")
    env.book.prune(NOW + 3 * 86_400)
    assert env.book.get(grant.session_id) is None and HOTKEY not in env.book._by_hotkey


def test_a_used_seed_asked_again_is_no_demand(tmp_path):
    env = build(tmp_path)
    quota = YieldQuota(ledger=Ledger(), pool_seeds=POLICY.pool_seeds, group_size=POLICY.pool_seeds // 2,
                       capacity=lambda: 2_160, refresh_s=0.0, clock=lambda: NOW)
    env.issuer._engagements["rl_precommit"]._quota = quota
    env.issuer._events = FanOutEvents(quota)
    for seed in range(POLICY.pool_seeds):                  # the miner asks for every seed of its pool at once
        quota.refusal(HOTKEY, NOW, precommit_sha256=env.precommit.sha256, seed_index=seed, environment=EPISODE)
    grant = open_(env, 0)
    assert isinstance(grant, Grant)
    close(env, grant)
    demand = quota._demand(HOTKEY, NOW)
    env.issuer._policy = dataclasses.replace(env.issuer._policy, max_refused_opens_per_minute=1_000)
    for n in range(100):
        assert open_(env, 0, request_id=f"{n + 1:032x}").reason == "engagement_taken"
    assert quota._demand(HOTKEY, NOW) == demand == POLICY.pool_seeds
    assert (env.precommit.sha256, 0) not in quota._wants[HOTKEY]


def test_the_books_caps_refuse_before_the_quota_hears_of_the_request(tmp_path):
    env = build(tmp_path)
    quota = YieldQuota(ledger=Ledger(), pool_seeds=POLICY.pool_seeds, group_size=POLICY.pool_seeds // 2,
                       capacity=lambda: 2_160, refresh_s=0.0, clock=lambda: NOW)
    env.issuer._engagements["rl_precommit"]._quota = quota
    engagements = env.issuer._engagements["rl_precommit"]
    env.book.policy = dataclasses.replace(env.book.policy, max_live_per_hotkey=0)
    refused = asyncio.run(engagements.terms(HOTKEY, {"kind": "rl_precommit", "precommit": {
        "precommit_sha256": env.precommit.sha256, "seed_index": 0}}))
    assert refused.reason == "live_cap" and not quota._wants.get(HOTKEY)


def test_the_cap_scale_multiplies_the_abort_cap_a_hotkey_meets(tmp_path):
    env = build(tmp_path)
    env.book.policy = dataclasses.replace(env.book.policy, max_aborted_per_day=1)
    close(env, open_(env, 1), "aborted")
    refused = open_(env, 2)
    assert refused.reason == "aborted_cap" and refused.detail["max"] == 1
    env.book.cap_scale = lambda hotkey: 3.0
    for seed in (2, 3):
        close(env, open_(env, seed, request_id=f"{seed + 100:032x}"), "aborted")
    refused = open_(env, 4)
    assert refused.reason == "aborted_cap" and refused.detail["max"] == 3


def test_the_cap_scale_multiplies_the_refused_open_rate(tmp_path):
    env = build(tmp_path)
    env.issuer._policy = dataclasses.replace(env.issuer._policy, max_refused_opens_per_minute=2)
    for n in range(2):
        assert open_(env, 10_000, request_id=f"{n:032x}").reason == "seed_out_of_pool"
    assert open_(env, 10_000, request_id=f"{9:032x}").reason == "open_refused_rate"
    env.book.cap_scale = lambda hotkey: 2.0
    assert open_(env, 10_000, request_id=f"{10:032x}").reason == "seed_out_of_pool"


def test_the_refusals_are_throttles_for_the_miner():
    assert {REFUSAL_STATUS[r] for r in ("quota_live", "quota_opens", "quota_pool", "quota_pool_opens")} == {429}


def test_fan_out_events_tell_every_listener():
    seen = []

    class Listener:
        def opened(self, record):
            seen.append(record)

        def closed(self, record, final):
            raise RuntimeError("one listener fails")

    class Refusing(Listener):
        def refused(self, hotkey, engagement, refusal):
            seen.append(("refused", hotkey))

    events = FanOutEvents(Listener(), None, Refusing())
    events.opened("r")
    events.closed("r", None)
    events.refused("5Hot", {}, Refusal("x", {}))                 # a listener without ``refused`` is skipped
    assert seen == ["r", "r", ("refused", "5Hot")] and len(events.listeners) == 2


def test_the_fleet_counts_the_session_slots_of_its_fresh_machines(tmp_path):
    from tests.unit.sandbox_fixtures import ADDRESS, capacity_report
    from tests.unit.test_sandbox_fleet import make

    fleet, machine, served, _clock = make(tmp_path)
    assert fleet.session_capacity() == 0                                    # no report yet
    served[ADDRESS] = capacity_report(machine, at=NOW)
    asyncio.run(fleet.poll_once())
    reported = sum(document["capacity"] for document in fleet._reports.values())
    assert reported > 0 and fleet.session_capacity() == reported
    assert fleet.directory_ready(NOW + HEARTBEAT_MAX_AGE_S + 1)
    assert fleet.session_capacity(now=NOW + HEARTBEAT_MAX_AGE_S + 1) == 0   # a stale report counts nothing
    fleet.drained.add(next(iter(fleet._reports)))
    assert fleet.session_capacity() == 0                                    # nor a drained machine


def test_the_wiring_installs_the_yield_quota_and_lifts_the_per_key_caps_but_not_the_abort_cap(tmp_path):
    services, _rt = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    assert isinstance(services.quota, YieldQuota)
    assert services.issuer._engagements["rl_precommit"]._quota is services.quota
    assert services.quota in services.issuer._events.listeners
    assert services.quota._capacity_of == services.fleet.session_capacity
    assert services.ledger._capacity_allowance == services.quota.groups_allowed
    assert services.book.cap_scale == services.quota.cap_scale
    assert services.book.policy.max_live_per_hotkey == services.book.policy.max_opens_per_hour == NO_PER_KEY_CAP
    plain, _ = _services(tmp_path / "plain")
    assert services.book.policy.max_aborted_per_day == plain.book.policy.max_aborted_per_day
    assert plain.quota is None and plain.book.policy.max_live_per_hotkey < NO_PER_KEY_CAP
    # M by both names an env goes by: the env an open names and the sandbox env a session record stores
    m = services.policies[EPISODE].pool_seeds // 2
    assert services.quota._group_sizes[EPISODE] == services.quota._group_sizes[SANDBOX_ENV] == m


def test_an_opened_ledger_gets_the_yield_quota_too(tmp_path):
    from reliquary.sandbox.rl_yield import SessionYieldLedger

    ledger = SessionYieldLedger(tmp_path / "rl-yield.sqlite3")
    try:
        services, _ = _services(tmp_path, ledger=ledger)
        assert isinstance(services.quota, YieldQuota) and services.quota._ledger is ledger
        assert services.book.policy.max_live_per_hotkey == NO_PER_KEY_CAP
    finally:
        ledger.close()


async def test_start_rebuilds_the_quotas_index_from_the_restored_book(tmp_path):
    from reliquary.infrastructure.sandbox_store import MemorySessionStore

    store = MemorySessionStore()
    record = _record("s-live", sha="a" * 64, state=LIVE)
    store.documents[record.session_id] = record.to_document()
    services, _ = _services(tmp_path, store=store, yield_path=tmp_path / "rl-yield.sqlite3")
    assert not services.quota._index
    await services.start()
    assert set(services.quota._index[record.hotkey]) == {"s-live"}
    await services.stop()
