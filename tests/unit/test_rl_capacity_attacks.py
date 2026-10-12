"""Attacks on the RL session quota, the open queue, the capacity attribution, the allocation route and envs added
to the running run: each runs against the real ledger, quota, queue and issuer (fakes only for the fleet and the
machines)."""
import asyncio
import time
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.infrastructure.sandbox_store import MemorySessionStore  # noqa: E402
from reliquary.sandbox.fleet import Placement  # noqa: E402
from reliquary.sandbox.open_queue import OpenQueue  # noqa: E402
from reliquary.sandbox.rl_capacity import (  # noqa: E402
    LOW_YIELD, MAIN, NEWCOMER, YieldStats, entity_sizes, sessions_of,
)
from reliquary.sandbox.rl_quota import YieldQuota  # noqa: E402
from reliquary.sandbox.rl_yield import LedgerEvents, SessionYieldLedger, held_seconds  # noqa: E402
from reliquary.sandbox.sessions import (  # noqa: E402
    LAPSED, LIVE, SUBMITTED, FanOutEvents, Grant, SandboxPolicy, SessionBook, SessionRecord,
)
from tests.unit.sandbox_fixtures import ADDRESS, MACHINE, NOW  # noqa: E402
from tests.unit.test_open_queue import queue as fair_queue  # noqa: E402
from tests.unit.test_rl_engagements import HOTKEY, POLICY, build, close, engagement, open_  # noqa: E402
from tests.unit.test_rl_episode_wiring import _record  # noqa: E402

M = POLICY.pool_seeds // 2
POOL = POLICY.pool_seeds
WASTED_ROUNDS = -(-200 // POOL)                      # enough wasted precommits to fill a 200-session yield
# The split-invariance stack has its own pool of 32 seeds, independent of the fixture's:
P32 = 32
CAPACITY = 2_160                                     # N = 216, L = 64 (the production constants)


def sha(n):
    return f"{n:064x}"


def stack(tmp_path, *, pool_seeds=None, capacity=CAPACITY):
    pool_seeds = pool_seeds or POOL
    ledger = SessionYieldLedger(tmp_path / "y.sqlite3")
    book = SessionBook(SandboxPolicy())
    quota = YieldQuota(ledger=ledger, pool_seeds=pool_seeds, group_size=pool_seeds // 2, capacity=lambda: capacity,
                       refresh_s=0.0)
    return ledger, book, quota


def settle(ledger, book, quota, precommit, records, *, now, used=False, aborted=False):
    for record in records:
        book.add(record)
        ledger.note_opened(record)
        quota.opened(record)
    rows = [(r.session_id, r.state, r.closed_status, getattr(r, "fault", None), held_seconds(r, now))
            for r in book.of_precommit(precommit)]
    ledger.decide(precommit, rows, used=used, aborted=aborted, now=now)
    ledger.flush(now)


def abandoned(hotkey, precommit, n, *, now):
    return [_record(f"{hotkey}-{precommit[-6:]}-{seed}", sha=precommit, hotkey=hotkey, state=LAPSED, seed=seed,
                    issued_at=now - 7 * 3600, expires_at=now - 6 * 3600) for seed in range(n)]


def live(hotkey, precommit, n, *, now):
    return [_record(f"{hotkey}-live-{seed}", sha=precommit, hotkey=hotkey, state=LIVE, seed=seed,
                    issued_at=now - 60, expires_at=now + 600) for seed in range(n)]


def trained(ledger, book, quota, hotkey, rounds, *, first, now, pool_seeds):
    """``rounds`` precommits of ``hotkey`` whose groups were trained: every session useful."""
    for n in range(first, first + rounds):
        records = [_record(f"{hotkey}-t{n}-{seed}", sha=sha(n), hotkey=hotkey, state=SUBMITTED, seed=seed,
                           issued_at=now - 3600, expires_at=now - 3000) for seed in range(pool_seeds)]
        for record in records:
            record.closed_at = now - 30                     # held 3 570 s each: the weight is exact
        settle(ledger, book, quota, sha(n), records, now=now, used=True)


def want(quota, hotkey, n, first, *, now, pool_seeds):
    """``hotkey`` asks for ``n`` sessions now: its current demand (the allocation is recomputed after)."""
    for i in range(n):
        quota.refusal(hotkey, now, precommit_sha256=sha(first + i // pool_seeds), seed_index=i % pool_seeds)
    quota._alloc_key = None


def take(quota, hotkey, first, *, now, pool_seeds):
    """Open sessions one at a time (``pool_seeds`` per precommit) until the quota refuses; how many it holds."""
    held = 0
    quota._refresh_s = float("inf")                  # the allocation held still while it opens
    for n in range(first, first + 10_000):
        for seed in range(pool_seeds):
            if quota.admit_locked(hotkey, now, precommit_sha256=sha(n), seed_index=seed) is not None:
                return held
            quota.opened(_record(f"{hotkey}-l{n}-{seed}", sha=sha(n), hotkey=hotkey, state=LIVE, seed=seed,
                                 issued_at=now - 30, expires_at=now + 600))
            held += 1
    return held


def test_one_hotkey_and_ten_hotkeys_with_a_tenth_of_its_volume_get_the_same_total(tmp_path):   # split-invariant
    now = int(time.time())
    totals = []
    for name, hotkeys, rounds in (("one", ["5One"], 30), ("ten", [f"5S{n}" for n in range(10)], 3)):
        (tmp_path / name).mkdir()
        ledger, book, quota = stack(tmp_path / name, pool_seeds=P32)
        trained(ledger, book, quota, "5Other", 64, first=0, now=now, pool_seeds=P32)    # 2 048 useful sessions
        for k, hotkey in enumerate(hotkeys):
            trained(ledger, book, quota, hotkey, rounds, first=1_000 + 100 * k, now=now, pool_seeds=P32)
            assert quota.lane(hotkey) == MAIN
        want(quota, "5Other", CAPACITY, 900_000, now=now, pool_seeds=P32)
        for k, hotkey in enumerate(hotkeys):          # the same demand, split like the volume
            want(quota, hotkey, CAPACITY // len(hotkeys), 50_000 + 1_000 * k, now=now, pool_seeds=P32)
        totals.append(sum(quota.allocation(now).main[hotkey] for hotkey in hotkeys))
    assert totals[0] == pytest.approx(totals[1]) and totals[0] > 0     # through the real ledger, production regime


def test_a_hundred_fresh_hotkeys_get_no_more_than_the_newcomer_entitys_rule(tmp_path):        # split-invariant
    now = int(time.time())
    ledger, book, quota = stack(tmp_path, pool_seeds=P32)
    trained(ledger, book, quota, "5Main", 64, first=0, now=now, pool_seeds=P32)
    want(quota, "5Main", CAPACITY, 900_000, now=now, pool_seeds=P32)                 # a high main load
    for n in range(100):
        want(quota, f"5F{n:03d}", 2 * P32, 10_000 + 100 * n, now=now, pool_seeds=P32)
    rule = quota.allocation(now).newcomer
    newcomer_n, _ = entity_sizes(CAPACITY, pool_seeds=P32)
    share = 32.0 / (32.0 + quota.rank("5Main")[1])                                    # NEWCOMER_WEIGHT vs 5Main's
    assert rule <= newcomer_n + share * (CAPACITY - newcomer_n) + 1e-6
    got = [take(quota, f"5F{n:03d}", 10_000 + 100 * n, now=now, pool_seeds=P32) for n in range(100)]
    assert quota.lane("5F000") == NEWCOMER and sum(got) <= sessions_of(rule)


def test_ten_wasting_hotkeys_get_no_more_than_the_low_yield_pool(tmp_path):                    # split-invariant
    now = int(time.time())
    ledger, book, quota = stack(tmp_path, pool_seeds=P32)
    trained(ledger, book, quota, "5Main", 4, first=0, now=now, pool_seeds=P32)
    for k in range(10):
        for n in range(4):                                         # 128 sessions abandoned: the newcomer exits
            first = 1_000 + 10 * k + n
            settle(ledger, book, quota, sha(first), abandoned(f"5W{k}", sha(first), P32, now=now), now=now)
    assert {quota.lane(f"5W{k}") for k in range(10)} == {LOW_YIELD}
    for k in range(10):
        want(quota, f"5W{k}", P32, 20_000 + 100 * k, now=now, pool_seeds=P32)
    got = sum(take(quota, f"5W{k}", 20_000 + 100 * k, now=now, pool_seeds=P32) for k in range(10))
    assert got <= entity_sizes(CAPACITY, pool_seeds=P32)[1]


def test_failures_of_ours_and_of_the_machines_never_lower_the_yield(tmp_path):
    ledger, book, quota = stack(tmp_path)
    now = int(time.time())
    faults = [_record(f"f-{s}", sha=sha(1), hotkey="5A", state="closed", seed=s) for s in range(5 * 4)]
    for n, record in enumerate(faults):
        kind = n % 5
        if kind == 0:
            record.state, record.closed_status = "aborted", "aborted"
        elif kind == 1:
            record.state, record.closed_status = "voided", "machine_drained"
        elif kind == 2:
            record.state, record.closed_status, record.fault = "closed_graded", "graded", "runtime_exec_failed"
        elif kind == 3:
            record.closed_status, record.fault = "open_failed", "open_refused:no_free_resources"
        else:
            record.closed_status, record.fault = "open_failed", "validator_restarted"
    settle(ledger, book, quota, sha(1), faults, now=now)
    ours = abandoned("5A", sha(2), POOL, now=now)
    ledger.note_opened(ours[0])
    ledger.group_settled(hotkey="5A", precommit_sha256=sha(2), session_ids=(), accepted=False,
                         stage="episode_persist_failed")           # a verified group we failed to persist
    settle(ledger, book, quota, sha(2), ours, now=now)
    settle(ledger, book, quota, sha(3), abandoned("5A", sha(3), POOL, now=now), now=now, aborted=True)
    assert ledger.stats("5A") == YieldStats()


def test_box_failed_is_the_agents_doing(tmp_path):                     # a box the agent killed is waste
    ledger, book, quota = stack(tmp_path)
    now = int(time.time())
    bombs = [_record(f"b-{s}", sha=sha(4), hotkey="5A", state="closed", seed=s) for s in range(POOL)]
    for record in bombs:
        record.closed_status = "box_failed"                        # e.g. `kill -9 -1` as the first tool call
    settle(ledger, book, quota, sha(4), bombs, now=now)
    assert (ledger.stats("5A").useful, ledger.stats("5A").decided) == (0, POOL)


def test_a_forged_verdict_cannot_spoil_or_launder_another_miners_precommit(tmp_path):
    ledger, book, quota = stack(tmp_path)
    now = int(time.time())
    victim = [_record(f"v-{s}", sha=sha(5), hotkey="5Victim", state="closed", seed=s) for s in range(POOL)]
    for record in victim:
        ledger.note_opened(record)
    ledger.group_handed_back(hotkey="5Attacker", precommit_sha256=sha(5), session_ids=(), stage="validator_lost")
    settle(ledger, book, quota, sha(5), victim, now=now, used=True)
    assert ledger.stats("5Victim").useful == POOL                  # the attacker's hand-back was ignored
    loser = [_record(f"l-{s}", sha=sha(6), hotkey="5Loser", state="closed", seed=s) for s in range(POOL)]
    for record in loser:
        ledger.note_opened(record)
    ledger.group_settled(hotkey="5Friend", precommit_sha256=sha(6), session_ids=(), accepted=False,
                         stage="episode_persist_failed")           # a "refusal of ours" sent for someone else
    settle(ledger, book, quota, sha(6), loser, now=now)
    assert ledger.stats("5Loser").useful == 0 and ledger.stats("5Loser").decided == POOL


@pytest.mark.parametrize("junk_stage", ["episode_directory", "episode_checker_busy", "episode_window_resumed",
                                        "episode_session_busy"])
def test_a_junk_group_at_a_stale_directory_does_not_clear_a_refused_precommit(tmp_path, junk_stage):
    ledger, book, quota = stack(tmp_path)
    now = int(time.time())
    refused = abandoned("5A", sha(11), POOL, now=now)
    for record in refused:
        ledger.note_opened(record)
    ledger.group_settled(hotkey="5A", precommit_sha256=sha(11), session_ids=(), accepted=False,
                         stage="episode_transcript")               # the real refusal
    ledger.group_settled(hotkey="5A", precommit_sha256=sha(11), session_ids=(), accepted=False,
                         stage=junk_stage)                         # then a junk group naming it, refused before the check
    settle(ledger, book, quota, sha(11), refused, now=now)
    quiet = abandoned("5A", sha(12), POOL, now=now)                # an abandoned precommit, never refused
    for record in quiet:
        ledger.note_opened(record)
    ledger.group_settled(hotkey="5A", precommit_sha256=sha(12), session_ids=(), accepted=False, stage=junk_stage)
    settle(ledger, book, quota, sha(12), quiet, now=now)
    assert (ledger.stats("5A").useful, ledger.stats("5A").decided) == (0, 2 * POOL)   # both wasted


async def test_a_restart_forgets_nothing_the_quota_counts(tmp_path):
    ledger, book, quota = stack(tmp_path)
    now = int(time.time())
    for n in range(WASTED_ROUNDS):
        settle(ledger, book, quota, sha(10 + n), abandoned("5A", sha(10 + n), POOL, now=now), now=now)
    store = MemorySessionStore()
    for record in live("5A", sha(99), M, now=now):
        store.documents[record.session_id] = record.to_document()
    before = quota.lane("5A")
    assert before == LOW_YIELD
    ledger.close()
    again = SessionYieldLedger(tmp_path / "y.sqlite3")
    restored = SessionBook(SandboxPolicy())
    restored.restore([SessionRecord.from_document(d) for d in await store.list_recent(now)])
    quota_again = YieldQuota(ledger=again, pool_seeds=POOL, group_size=M, capacity=lambda: CAPACITY)
    quota_again.rebuild(restored.records())
    assert quota_again.lane("5A") == before
    # its live group counts again after the restart: a second precommit waits (its maximum is one group)
    assert quota_again.admit_locked("5A", now, precommit_sha256=sha(100), seed_index=0).reason == "quota_live"


def test_open_and_abandon_through_the_issuer_is_waste(tmp_path):
    env = build(tmp_path)
    ledger = SessionYieldLedger(tmp_path / "y.sqlite3")
    quota = YieldQuota(ledger=ledger, pool_seeds=POOL, group_size=M, capacity=lambda: CAPACITY,
                       clock=env.issuer._clock, refresh_s=0.0)
    env.issuer._engagements["rl_precommit"]._quota = quota
    env.issuer._events = FanOutEvents(LedgerEvents(ledger), quota)
    env.issuer._record_faults = True
    for seed in range(POOL):
        grant = open_(env, seed)
        assert isinstance(grant, Grant)
        close(env, grant, "expired")                               # opened, then abandoned to expiry
    precommit = env.precommit.sha256
    assert ledger.precommits_of(env.precommit.window) == (precommit,)
    ledger.decide(precommit, [(r.session_id, r.state, r.closed_status, r.fault) for r in env.book.of_precommit(precommit)],
                  used=False, aborted=False, now=time.time())
    ledger.flush()
    assert ledger.stats(HOTKEY).decided == POOL and ledger.stats(HOTKEY).useful == 0
    # one wasted pool: still a newcomer (it leaves for the low-yield pool after NEWCOMER_EXIT_GROUPS pools)
    assert quota.lane(HOTKEY) == NEWCOMER


def test_an_unsigned_open_failed_is_waste_and_a_signed_one_is_not(tmp_path):
    env = build(tmp_path)
    ledger = SessionYieldLedger(tmp_path / "y.sqlite3")
    env.issuer._events = LedgerEvents(ledger)
    env.issuer._record_faults = True
    env.issuer._open_refusal_check = lambda receipt, record: receipt.get("signature") == "machine"
    receipt = {"document": {"reason": "open_failed"}, "signature": "machine"}
    signed, unsigned, forged = open_(env, 0), open_(env, 1), open_(env, 2)
    for grant, carried in ((signed, receipt), (unsigned, None), (forged, {**receipt, "signature": "miner"})):
        asyncio.run(env.issuer.close(hotkey=HOTKEY, session_id=grant.session_id, reason="open_failed",
                                     transcript=None, open_refusal=carried))
    precommit = env.precommit.sha256
    ledger.decide(precommit, [(r.session_id, r.state, r.closed_status, r.fault) for r in env.book.of_precommit(precommit)],
                  used=False, aborted=False, now=time.time())
    ledger.flush()
    assert (ledger.stats(HOTKEY).decided, ledger.stats(HOTKEY).useful) == (2, 0)   # the signed one is neither


def test_a_real_gateway_receipt_is_verified_by_the_fleet(tmp_path):
    if not hasattr(attest, "open_refusal_valid"):
        pytest.skip("this reliquary-sandbox does not sign open refusals yet")
    from reliquary.sandbox.fleet import Fleet  # noqa: F401
    from tests.unit.test_sandbox_fleet import make

    fleet, machine, _served, _clock = make(tmp_path)
    document = {"kind": "open_refusal", "machine_id": MACHINE, "key_id": machine.key_id, "session_id": "s-1",
                "token_sha256": "a" * 64, "reason": "no_free_resources", "at": int(fleet._clock())}
    receipt = {"document": document, "signature": machine.sign(attest.OPEN_REFUSAL_DOMAIN, document)}
    record = _record("s-1", sha=sha(7))
    record.machine_id, record.token_sha256 = MACHINE, "a" * 64
    assert fleet.open_refusal_valid(receipt, record)
    record.token_sha256 = "b" * 64
    assert not fleet.open_refusal_valid(receipt, record)


async def test_queue_flooding_is_bounded_and_does_not_starve_others(tmp_path):
    ledger, _, quota = stack(tmp_path)
    ledger._snapshot = {"5A": YieldStats(useful=0, decided=200),                # the flooder wasted everything
                        "5Main": YieldStats(useful=180, decided=200, useful_24h=500, useful_s_24h=500 * 1_800.0)}
    ledger.version += 1
    queue = OpenQueue(rank=quota.rank, max_waiting=6, max_per_key=3, fair_every=0)
    cls = ("env", None, None)
    holder = queue.join("5Z0", cls=cls)                            # holds the turn
    flood = [queue.join("5A", cls=cls, engagement=f"rl:1:{sha(8)}:{n}") for n in range(3)]
    assert queue.join("5A", cls=cls).reason == "open_queue_key_full"
    assert queue.join("5A", cls=cls, engagement=f"rl:1:{sha(8)}:0").reason in ("open_queued", "open_queue_key_full")
    established = queue.join("5Main", cls=cls)
    queue.done(holder, placed=True)
    assert established.granted and not any(t.granted for t in flood)   # weight 500 beats a pooled ticket's 0


async def test_a_flooder_of_an_unplaceable_class_never_blocks_a_placeable_open(tmp_path):
    env = build(tmp_path)
    q = OpenQueue(rank=lambda hotkey: (hotkey, 1000.0 if hotkey == "5Rich" else 1.0, False), fair_every=0)
    env.issuer._queue = q
    for n in range(3):                                             # a high-score key waiting in a class nobody serves now
        q.join("5Rich", cls=("reliquary-swe", 4000, 4), engagement=f"rl:1:{sha(9)}:{n}")
    grant = await env.issuer.open(hotkey=HOTKEY, request_id="a" * 32, engagement=engagement(env.precommit, 0))
    assert isinstance(grant, Grant)


# --- a capacity verdict needs others' saturation, sessions that ran and a group that could not form

def saturating(tmp_path):
    env = build(tmp_path)
    ledger = SessionYieldLedger(tmp_path / "y.sqlite3")
    env.issuer._queue = fair_queue(wait_s=0.05)
    env.issuer._events = LedgerEvents(ledger)
    env.issuer._record_faults = True
    return env, ledger


def others_fill_the_machine(env, n=8, *, hotkey="5Other", precommit="f" * 64):
    for k in range(n):
        record = _record(f"{hotkey}-fill-{k}", sha=precommit, hotkey=hotkey, state=LIVE, seed=k, issued_at=NOW - 60,
                         expires_at=NOW + 600)
        record.machine_id = MACHINE
        env.book.add(record)


def refused_open(env, seed):
    env.issuer._fleet.placement = None
    outcome = open_(env, seed)
    assert outcome.reason == "sandbox_saturated"


def decided_of(env, ledger):
    precommit = env.precommit.sha256
    ledger.decide(precommit, [(r.session_id, r.state, r.closed_status, r.fault) for r in env.book.of_precommit(precommit)],
                  used=False, aborted=False, now=time.time(), group_size=M)
    ledger.flush()
    return ledger.stats(HOTKEY).decided


def test_a_self_made_saturation_leaves_an_abandoned_precommit_wasted(tmp_path):
    # The refused open's own group never makes a saturation its own; the miner's OTHER load does (here
    # another of its precommits filling the machine, with no allocation entitling it to that load).
    env, ledger = saturating(tmp_path)
    for seed in range(3):
        close(env, open_(env, seed), "graded")                     # three graded, fewer than M; the episodes look bad
    others_fill_the_machine(env, hotkey=HOTKEY, precommit="e" * 64)   # its own other group fills the machine
    refused_open(env, 3)                                           # one more open: refused, its own doing
    assert env.precommit.sha256 not in ledger._verdicts
    assert decided_of(env, ledger) == 3                            # abandoned: waste


def test_squatted_tokens_stay_waste_under_others_saturation(tmp_path):
    env, ledger = saturating(tmp_path)
    grants = [open_(env, seed) for seed in range(2)]               # tokens taken, never presented
    others_fill_the_machine(env)
    refused_open(env, 2)                                           # a real saturation: a capacity verdict
    assert ledger._verdicts[env.precommit.sha256][0] == "capacity"
    for grant in grants:
        env.book.get(grant.session_id).state = "lapsed"            # they lapsed live: no box ever ran
    assert decided_of(env, ledger) == 2


def test_an_honest_half_group_is_neutral_under_others_saturation(tmp_path):
    env, ledger = saturating(tmp_path)
    for seed in range(3):
        close(env, open_(env, seed), "graded")                     # three graded, fewer than M
    others_fill_the_machine(env)
    refused_open(env, 3)                                           # the rest of its group could not be opened
    assert decided_of(env, ledger) == 0


# --- the reference miner's rule against the real quota, production regime, an idle fleet -----------------------

def closed_loop(capacity, keys, life_s, *, hours=1.0, warm_s=1_200, poll_s=5, retry_s=10):
    """Each key {hotkey: (weight, ceiling)} runs ``groups_to_run(ceiling, told, current)`` groups of P32 seeds,
    every seed asked at once, a refused seed asked again every ``retry_s``, each group ending ``life_s`` after it
    started; the quota is the real one (refresh 1 s, production constants). Returns {hotkey: [held per poll]} and
    {hotkey: [groups run per poll]} after the warm-up."""
    from reliquary.miner.episode_mining import groups_to_run
    from reliquary.protocol.service_episode import rl_engagement

    clock = SimpleNamespace(now=float(NOW))

    class Ledger:
        version = 0

        def snapshot(self):
            return {h: YieldStats(useful=180, decided=200, useful_24h=int(w * 48), useful_s_24h=w * 86_400.0)
                    for h, (w, _ceiling) in keys.items()}

    quota = YieldQuota(ledger=Ledger(), pool_seeds=P32, group_size=P32 // 2, capacity=lambda: capacity,
                       clock=lambda: clock.now)
    state = {h: {"run": ceiling, "groups": [], "made": 0} for h, (_w, ceiling) in keys.items()}
    held, run, made = {h: [] for h in keys}, {h: [] for h in keys}, [0]
    start = clock.now
    while clock.now < start + hours * 3_600:
        now = int(clock.now)
        for h, st in state.items():
            told = max(1, -(-quota.sessions_allowed(h) // P32))       # the route's `groups`
            st["run"] = groups_to_run(keys[h][1], told, st["run"])
            for group in [g for g in st["groups"] if now >= g["start"] + life_s]:
                for record in group["records"]:
                    record.state, record.closed_status, record.closed_at = "closed", "expired", now
                quota.closed(None, None)
                st["groups"].remove(group)
            while len(st["groups"]) < st["run"]:
                st["made"] += 1
                st["groups"].append({"sha": sha(10_000 * (list(keys).index(h) + 1) + st["made"]), "start": now,
                                     "seeds": list(range(P32)), "records": [], "tried": -10**9})
            for group in st["groups"]:
                if now - group["tried"] < retry_s:
                    continue
                group["tried"] = now
                for seed in list(group["seeds"]):
                    if quota.admit_locked(h, now, precommit_sha256=group["sha"], seed_index=seed) is None:
                        made[0] += 1
                        record = SimpleNamespace(session_id=f"c{made[0]}", hotkey=h, env="reliquary-swe",
                                                 engagement=rl_engagement(1, group["sha"], seed), issued_at=now,
                                                 expires_at=now + life_s + 600, state=LIVE, closed_status=None,
                                                 closed_at=None)
                        quota.opened(record)
                        group["records"].append(record)
                        group["seeds"].remove(seed)
        if clock.now - start >= warm_s:                                   # every poll: no aliasing with group ends
            for h, st in state.items():
                held[h].append(len(quota._counted(h, now)))
                run[h].append(st["run"])
        clock.now += poll_s
    return held, run


@pytest.mark.parametrize("life_s", [1_800, 600])
def test_the_reference_miner_reaches_the_same_total_with_one_hotkey_or_ten(life_s):
    one_held, one_run = closed_loop(CAPACITY, {"5One": (300.0, 10)}, life_s)
    ten = {f"5S{n}": (30.0, 1) for n in range(10)}
    ten_held, ten_run = closed_loop(CAPACITY, ten, life_s)
    one_total = sum(one_held["5One"]) / len(one_held["5One"])
    ten_total = sum(sum(minute) for minute in zip(*ten_held.values())) / len(ten_held["5S0"])
    assert one_total > 0.9 * 10 * P32                                  # an idle fleet: its ceiling, not one group
    assert abs(one_total - ten_total) <= 0.1 * ten_total               # the same total, one hotkey or ten
    assert set(one_run["5One"]) == {10}                                 # no oscillation: its ceiling throughout
    assert all(set(runs) == {1} for runs in ten_run.values())


# --- the reference miner's rule against the real quota, under CONTENTION ---------------------------------------
# A fleet smaller than what the miners would run: every MAIN key's ceiling is above its weighted share. The quota and
# the miner's rule (``groups_to_run``) are the real ones; the ledger is a fixed snapshot (the standings do not move in
# an hour). A group that has not started sends one seed at a time until its first grant, then the rest at once
# (the reference miner); a refused seed is asked again every ``retry_s``.

def contended_loop(capacity, keys, life_s, *, hours=1.0, poll_s=5, retry_s=10, jitter=0.0, window_s=None):
    """``keys``: {hotkey: (weight or None for a newcomer, ceiling in groups)}. A group lives ``life_s`` from its
    first session, times a fixed factor in ``1 +- jitter`` drawn from its precommit, and ends at the latest when
    its window closes (``window_s``: every group of every key turns over at once there). Returns, per poll from the
    start, {hotkey: [(told, run, held)]} and the fleet's [held total]."""
    from reliquary.miner.episode_mining import groups_to_run
    from reliquary.protocol.service_episode import rl_engagement

    clock = SimpleNamespace(now=float(NOW))

    class Ledger:
        version = 0

        def snapshot(self):
            return {h: YieldStats(useful=180, decided=200, useful_24h=int(w * 48), useful_s_24h=w * 86_400.0)
                    for h, (w, _ceiling) in keys.items() if w is not None}

    quota = YieldQuota(ledger=Ledger(), pool_seeds=P32, group_size=P32 // 2, capacity=lambda: capacity,
                       clock=lambda: clock.now)
    order = list(keys)
    state = {h: {"run": 1, "groups": [], "made": 0} for h in keys}
    trace, fleet = {h: [] for h in keys}, []
    made = 0
    start = clock.now
    while clock.now < start + hours * 3_600:
        now = int(clock.now)
        for h in order:
            st = state[h]
            told = max(1, -(-quota.sessions_allowed(h) // P32))       # the route's `groups`
            st["run"] = groups_to_run(keys[h][1], told, st["run"])
            st["told"] = told
            closing = window_s is not None and (now - start) % window_s == 0 and now > start
            for group in [g for g in st["groups"]
                          if closing or (g["records"] and now >= g["start"] + g["life"])]:
                for record in group["records"]:
                    record.state, record.closed_status, record.closed_at = "closed", "expired", now
                quota.closed(None, None)
                st["groups"].remove(group)
            # the miner drops the groups above its number that have not started (a started one is run to its end)
            unstarted = [g for g in st["groups"] if not g["records"]]
            while len(st["groups"]) > st["run"] and unstarted:
                st["groups"].remove(unstarted.pop())
            while len(st["groups"]) < st["run"]:
                st["made"] += 1
                n = 10_000 * (order.index(h) + 1) + st["made"]
                spread = 1.0 + jitter * (2.0 * ((n * 2_654_435_761) % 1_000) / 1_000 - 1.0)
                st["groups"].append({"sha": sha(n), "start": now, "life": life_s * spread,
                                     "seeds": list(range(P32)), "records": [], "tried": -10**9})
            for group in st["groups"]:
                if now - group["tried"] < retry_s:
                    continue
                group["tried"] = now
                for seed in list(group["seeds"]):
                    if quota.admit_locked(h, now, precommit_sha256=group["sha"], seed_index=seed) is not None:
                        if not group["records"]:
                            break                                     # not started: one seed at a time
                        continue
                    made += 1
                    if not group["records"]:
                        group["start"] = now                          # its life runs from its first session
                    record = SimpleNamespace(session_id=f"c{made}", hotkey=h, env="reliquary-swe",
                                             engagement=rl_engagement(1, group["sha"], seed), issued_at=now,
                                             expires_at=now + life_s + 600, state=LIVE, closed_status=None,
                                             closed_at=None)
                    quota.opened(record)
                    group["records"].append(record)
                    group["seeds"].remove(seed)
        total = 0
        for h in order:
            held = len(quota._counted(h, now))
            total += held
            trace[h].append((state[h]["told"], state[h]["run"], held))
        fleet.append(total)
        clock.now += poll_s
    return trace, fleet


CONTENDED = {"5B": (300.0, 40), "5C": (150.0, 40)}         # ceilings of 1 280 sessions each, far above any share
NEWCOMERS = {f"5N{n}": (None, 2) for n in range(3)}        # fresh keys at their member maximum (2 groups)


def weighted_shares(keys):
    """The MAIN keys' weighted shares of what the newcomers leave (their 3 x 2 groups fit their guarantee N)."""
    main = {h: w for h, (w, _c) in keys.items() if w is not None}
    left = CAPACITY - sum(2 * P32 for w, _c in keys.values() if w is None)
    return {h: left * w / sum(main.values()) for h, w in main.items()}


def tail_mean(rows, column=2):
    return sum(row[column] for row in rows) / len(rows)


@pytest.mark.parametrize("life_s, window_s, hours", [(600, None, 1.5), (1_800, 1_800, 2.0)])
def test_under_contention_the_allocations_shrink_to_the_shares_and_hold_still(life_s, window_s, hours):
    """Three MAIN keys whose ceilings (40 groups each: 3 840 sessions) are far above a 2 160-session fleet, and three
    newcomers. Group lives vary by +-30 % (episodes differ); with ``window_s`` every group also ends at the window's
    close. Measured from three lives (or one window) on."""
    keys = {"5A": (600.0, 40), **CONTENDED, **NEWCOMERS}
    trace, fleet = contended_loop(CAPACITY, keys, life_s, hours=hours, jitter=0.3, window_s=window_s)
    warm = (3 * life_s if window_s is None else window_s) // 5
    shares = weighted_shares(keys)
    for hotkey, share in shares.items():
        rows = trace[hotkey][warm:]
        told = [row[0] for row in rows]
        assert trace[hotkey][0][0] > max(told) + 10               # told the idle fleet first, then it shrank
        assert max(told) - min(told) <= 2                         # the oscillation bound: two groups, never more
        assert max(told) <= -(-share // P32) + 2                  # within two groups of its share, rounded up
        assert [row[1] for row in rows] == told                   # the miner runs what it is told (all below 40)
        # it holds its share within one group or 5 % (the groups a window turnover started are never cut)
        assert abs(tail_mean(rows) - share) <= max(P32, 0.05 * share)
    for hotkey in NEWCOMERS:
        assert tail_mean(trace[hotkey][warm:]) >= 2 * P32 - 4     # newcomers still hold their two groups
    # the fleet: on average within 4 % of C; at worst 4 groups above it (the never-cut groups of a turnover)
    assert sum(fleet[warm:]) / len(fleet[warm:]) <= 1.04 * CAPACITY
    assert max(fleet[warm:]) <= CAPACITY + 4 * P32


@pytest.mark.parametrize("life_s, window_s, hours", [(600, None, 1.5), (1_800, 1_800, 2.0)])
def test_under_contention_splitting_a_hotkey_gains_nothing(life_s, window_s, hours):            # split-invariant
    one = {"5A": (600.0, 40), **CONTENDED, **NEWCOMERS}
    ten = {**{f"5A{n}": (60.0, 4) for n in range(10)}, **CONTENDED, **NEWCOMERS}     # the same volume, ten keys
    warm = (3 * life_s if window_s is None else window_s) // 5
    totals, others = [], []
    for keys in (one, ten):
        trace, _fleet = contended_loop(CAPACITY, keys, life_s, hours=hours, jitter=0.3, window_s=window_s)
        parts = [h for h in keys if h.startswith("5A")]
        totals.append(sum(tail_mean(trace[h][warm:]) for h in parts))
        others.append(tail_mean(trace["5B"][warm:]) + tail_mean(trace["5C"][warm:]))
        for hotkey in NEWCOMERS:
            assert tail_mean(trace[hotkey][warm:]) >= 2 * P32 - 4
    assert totals[1] <= totals[0] + P32 / 2                       # ten keys never hold more (half a group: noise)
    assert abs(totals[1] - totals[0]) <= 0.02 * totals[0]
    assert others[1] >= others[0] - P32 / 2                       # and take nothing from the others


@pytest.mark.xfail(strict=True, reason="known: when every group lives through its window, each window's "
                   "turnover is a race: a key's potential is the idle fleet, it starts its ceiling's groups at M each "
                   "and none is cut, so the ramp, not the weights, sets the shares and the fleet is overcommitted")
def test_when_every_group_lives_through_its_window_the_ramp_does_not_set_the_shares():
    keys = {"5A": (600.0, 40), **CONTENDED, **NEWCOMERS}
    trace, fleet = contended_loop(CAPACITY, keys, 3_600, hours=1.5, jitter=0.0, window_s=1_800)
    shares = weighted_shares(keys)
    for hotkey in CONTENDED:                                      # measured: 5B ~960 (share 562), 5C 640 (281)
        assert tail_mean(trace[hotkey][360:]) <= shares[hotkey] + P32
    assert max(fleet[360:]) <= CAPACITY + 4 * P32                 # measured: ~2 900 on 2 160


# --- the signed allocation route: replay, a foreign hotkey, the rate ---------------------------------------------

def _allocation_route():
    pytest.importorskip("bittensor")
    from tests.unit.test_miner_allocation import MINER, allocation_client, body_of

    asked = []

    def allocation_of(hotkey):
        asked.append(hotkey)
        return {"hotkey": hotkey, "groups": {"e": 3}}

    return allocation_client(allocation_of), body_of, MINER, asked


@pytest.mark.parametrize("respell", [str.upper, lambda s: "0x" + s, lambda s: " " + s + " ",
                                     lambda s: s[:10] + "0x" + s[10:]])
def test_a_replayed_allocation_request_with_its_signature_respelt_is_refused(respell):
    client, body_of, miner, asked = _allocation_route()
    first = body_of()
    assert client.post("/rl/sessions/allocation", json=first).status_code == 200
    again = client.post("/rl/sessions/allocation", json={**first, "signature": respell(first["signature"])})
    assert again.status_code == 409 and again.json()["reason"] == "request_reused"   # the same signature bytes
    assert asked == [miner.ss58_address]                          # the eavesdropper learnt nothing


def test_a_foreign_hotkeys_signature_reads_nothing_and_spends_nothing_of_the_victims_minute():
    pytest.importorskip("bittensor")
    import bittensor as bt

    from reliquary.sandbox.rl_routes import ALLOCATION_MAX_PER_MINUTE

    client, body_of, miner, asked = _allocation_route()
    eve = bt.Keypair.create_from_uri("//Eve")
    for n in range(ALLOCATION_MAX_PER_MINUTE + 5):                # Eve floods requests naming Alice
        answer = client.post("/rl/sessions/allocation", json=body_of(eve, hotkey=miner.ss58_address, at=NOW - n % 5))
        assert answer.status_code == 403
    assert client.post("/rl/sessions/allocation", json=body_of()).status_code == 200   # Alice's minute untouched
    assert asked == [miner.ss58_address]


# --- the miner's manifest-digest refusal, on an env added to the running run -------------------------------------

def test_an_added_env_whose_install_is_not_its_rows_manifest_is_never_mined(tmp_path):
    pytest.importorskip("reliquary_sandbox_verifiers")
    pytest.importorskip("bittensor")
    from reliquary.miner import episode_mining as em
    from tests.unit.episode_v2_fixtures import (
        ADDED, ADDED_IDENTITY, ENV_PACKAGE, EPISODE, WINDOW_BEACON, added_block, bridged_version, episode_contract,
        episode_runtime,
    )
    from tests.unit.test_miner_catalogue import CATALOGUE, ROWS, build_source, forbidden, pinned_added_block, with_added

    contract = episode_contract(catalogue=CATALOGUE, rows=ROWS, episode_version=bridged_version(CATALOGUE, ENV_PACKAGE,
                                                                                                ROWS))
    rt = episode_runtime(tmp_path, contract)
    plain = rt.announcement(window=1, randomness=WINDOW_BEACON)
    pinned = pinned_added_block()
    moved = added_block(version="0" * 64, dataset=pinned["environment"]["dataset"])   # same rows, another manifest
    for block, refused in ((moved, True), (pinned, False)):
        prompts = em.EpisodeTaskPrompts([EPISODE, ADDED], load=forbidden,
                                        build_source=build_source({EPISODE: ENV_PACKAGE, ADDED: ADDED_IDENTITY}))
        prompts.bind(plain)
        value = dict(plain, schedule=with_added(contract, rt.schedule, block=block).to_dict())
        if refused:
            with pytest.raises(ValueError, match="not the order's"):
                prompts.bind(value)
            assert ADDED not in prompts.sources and prompts.size(ADDED) == 0   # nothing of it is ever mined
        else:
            prompts.bind(value)
            assert prompts.size(ADDED) == ROWS


# --- add-request integrity: a refused add leaves every byte of the run as it was ----------------------

def _announcement_after(tmp_path, tamper):
    """A runtime given (or not, ``tamper`` None) an add request ``tamper`` spoiled, then window 2's announcement."""
    import json

    from reliquary.services.runtime import protocol_slot_geometry
    from reliquary.services.schedule import ScheduleRequestStore
    from tests.unit.episode_v2_fixtures import EPISODE, WINDOW_BEACON, episode_runtime
    from tests.unit.service_v2_fixtures import MATH
    from tests.unit.test_schedule_add import SHARES, folder_with_attestation, request

    tmp_path.mkdir()
    rt = episode_runtime(tmp_path)
    if tamper is not None:
        folder_with_attestation(tmp_path)
        store = ScheduleRequestStore(tmp_path)
        req = request(rt.contract, rt.schedule, active=list(SHARES), shares=SHARES)
        check, why = tamper(tmp_path, req)
        store.write(req)
        rt.apply_pending_schedule_request(store, window=1, catalogue_check=check)
        assert store.status()["status"] == "refused" and why in store.status()["detail"], store.status()
    picks, slots = protocol_slot_geometry()
    rt.open_window(2, pools={MATH: 0.5, EPISODE: 0.5}, picks_target=picks, batch_slots=slots)
    value = rt.announcement(window=2, randomness=WINDOW_BEACON)
    return json.dumps(value, sort_keys=True), rt.contract.sha256, rt.schedule.sha256


def _swapped_after_writing(folder, req):
    import os

    os.chmod(folder / "conformance.json", 0o600)
    (folder / "conformance.json").write_bytes(b'{"gate": true, "passed": true}')   # swapped after the CLI checked it
    return (lambda name, added, attestation: None,                                 # even an accepting check
            "the attestation file is not the one the entry names")


def _attesting_another_env(folder, req):
    import hashlib
    import json

    from reliquary.services.schedule import raise_activation_problem
    from reliquary.validator.catalogue_activation import attestation_problem
    from tests.unit.test_schedule_add import ATTESTATION

    pytest.importorskip("reliquary_sandbox_verifiers")
    forged = json.loads(ATTESTATION)
    forged["env_package"] = "reliquary-other==9.9.9"                 # a passing gate, of another env
    raw = json.dumps(forged, sort_keys=True).encode()
    (folder / "conformance.json").write_bytes(raw)
    for entry in req["add"].values():
        entry["catalogue"]["conformance_sha256"] = hashlib.sha256(raw).hexdigest()   # its digest matches
    return (lambda name, added, attestation: raise_activation_problem(name, attestation_problem(attestation, added)),
            "it attests another env package")


def _named_like_a_launch_env(folder, req):
    from tests.unit.episode_v2_fixtures import ADDED, EPISODE

    req["add"] = {EPISODE: req["add"][ADDED]}                         # the launch episode env's name
    req["active"] = [name for name in req["active"] if name != ADDED]
    req["shares"] = {name: bps for name, bps in req["shares"].items() if name != ADDED}
    return lambda name, added, attestation: None, f"{EPISODE} is already in the schedule"


@pytest.mark.parametrize("tamper", [_swapped_after_writing, _attesting_another_env, _named_like_a_launch_env])
def test_a_refused_add_request_leaves_the_announcement_the_order_and_the_schedule_byte_identical(tmp_path, tamper):
    assert _announcement_after(tmp_path / "attacked", tamper) == _announcement_after(tmp_path / "plain", None)


# --- the automatic switch-off of an added env: its revision can never be pre-empted -------------------

def test_an_operator_request_cannot_take_the_request_id_of_a_later_automatic_switch_off(tmp_path):
    from reliquary.protocol.service_schedule import next_schedule
    from reliquary.services.schedule import ScheduleRequestStore, build_request
    from tests.unit.episode_v2_fixtures import ADDED, EPISODE
    from tests.unit.service_v2_fixtures import MATH
    from tests.unit.test_live_env_addition import BLOCK, SHARES, validator

    me, rt = validator(tmp_path)
    rt.apply_schedule(next_schedule(rt.order_contract, rt.schedule, add={ADDED: BLOCK}, active=[MATH, EPISODE, ADDED],
                                    shares=SHARES), request_id="add-1", window=1)
    future = rt.schedule.revision + 2                              # the revision a switch-off would take later
    store = ScheduleRequestStore(tmp_path)
    squat = build_request(order_sha256=rt.contract.sha256, revision=rt.schedule.revision + 1, cooldowns={MATH: 10})
    squat["request_id"] = f"auto-deactivate-r{future}"
    store.write(squat)
    rt.apply_pending_schedule_request(store, window=2)
    assert store.status()["status"] == "refused" and "auto-" in store.status()["detail"]   # the validator's own ids
    me._deactivate_added_envs({ADDED: "broken"}, 3)                # never raises: the window opens without it
    assert ADDED not in rt.schedule.active_environments()


@pytest.mark.parametrize("broken", ["build", "checks"])
def test_an_added_env_broken_for_good_costs_a_bounded_number_of_builds(tmp_path, monkeypatch, broken):
    """24 h, prestage tick every 30 s, boundary every 300 s: an env whose build always raises, or whose build
    always fails its boundary checks (another rows count), is built at most once per backoff (60 s doubling to
    1 h), never twice at once, never at the boundary, and stays out of the run."""
    from reliquary.environment import bridged_episode
    from reliquary.validator import service
    from tests.unit.episode_v2_fixtures import ADDED
    from tests.unit.test_live_env_addition import WAIT, add_to_schedule, built, validator

    clock = SimpleNamespace(now=0.0)
    starts, inflight = [], {"now": 0, "max": 0}

    def build(config):
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        starts.append(clock.now)
        try:
            if broken == "build":
                raise RuntimeError("the package's import fails")
            return built(rows=1)                                   # not the entry's taskset: refused at the boundary
        finally:
            inflight["now"] -= 1

    monkeypatch.setattr(bridged_episode, "build_environment", build)
    me, rt = validator(tmp_path)
    me._added_clock = lambda: clock.now
    schedule = add_to_schedule(rt, active=False)
    me._added_unavailable(ADDED, "the boot could not build it")
    day = 24 * 3600.0
    while clock.now <= day:
        me._retry_added_builds(schedule)                           # the prestage tick
        assert me._join_builds(WAIT)
        if int(clock.now) % 300 == 0:                              # a boundary
            boundary_starts = len(starts)
            me._stage_added_envs(schedule, window=int(clock.now // 300) + 2)
            assert len(starts) == boundary_starts                  # the boundary never builds
            assert me._join_builds(WAIT)
            assert me._staged_envs == {}
        clock.now += 30.0
    # 60 + 120 + ... + 3 600 then one an hour: 7 builds in the first 2 h, about 22 more in the rest of the day
    assert 20 <= len(starts) <= 7 + 24 and inflight["max"] == 1          # retried all day, never more
    assert all(b - a >= service.ADDED_RETRY_MIN_S for a, b in zip(starts, starts[1:]))
    assert ADDED not in me.envs and ADDED not in rt.schedule.active_environments()


# --- snapshots that may lack an added env: the optional set opens no other hole -------------------------

def test_a_forged_snapshot_cannot_hide_behind_an_added_env():
    from reliquary.constants import TRAINING_RUN_ID
    from reliquary.shared.checkpoint_namespace import active_checkpoint_namespace
    from reliquary.validator.service import ValidationService
    from tests.unit.episode_v2_fixtures import ADDED
    from tests.unit.service_v2_fixtures import MATH

    base = {**active_checkpoint_namespace().identity, "run_id": TRAINING_RUN_ID, "complete": True, "snapshot_window": 5}
    prompt = {**base, "schema_version": 2}
    content = {**base, "schema_version": 1}
    optional = frozenset({ADDED})
    digest = "a" * 64
    for validate, snapshot, future in ((ValidationService._validate_cooldown_snapshot, prompt, {"3": 9}),
                                       (ValidationService._validate_content_snapshot, content, {digest: 9})):
        # an added env this process keeps a map for is validated like any other: no selection after the snapshot
        with pytest.raises(ValueError, match="outside its history"):
            validate({**snapshot, "envs": {MATH: {}, ADDED: future}}, {MATH, ADDED}, 7, optional)
        # an added env being optional never makes a launch env optional
        with pytest.raises(ValueError, match="incomplete"):
            validate({**snapshot, "envs": {ADDED: {}}}, {MATH, ADDED}, 7, optional)
    # a content snapshot naming an env the run never had is refused (a cooldown snapshot's extra env is never read)
    with pytest.raises(ValueError, match="incomplete"):
        ValidationService._validate_content_snapshot({**content, "envs": {MATH: {}, "ghost": {}}}, {MATH}, 7, optional)


# --- set_active_envs: a restart never counts an inactive env's machines in the fleet ----------------

@pytest.mark.parametrize("resumed", [False, True])
def test_a_resumed_window_tells_the_services_its_envs_as_a_fresh_one_does(resumed):
    import types

    from reliquary.validator.service import ValidationService
    from tests.unit.episode_v2_fixtures import EPISODE
    from tests.unit.service_v2_fixtures import MATH

    told, noted = [], []
    me = SimpleNamespace(
        _service_runtime=SimpleNamespace(open_window=lambda *a, **k: None,
                                         announcement=lambda **k: {"pool_randomness": "r"}),
        _candidate_service_window={"window": 2, "pools": {MATH: 0.4, EPISODE: 0.3}, "env_mix": [(MATH, 8)],
                                   "picks_target": 1, "batch_slots": 8},
        _episode_services=SimpleNamespace(set_active_envs=told.append),
        _service_window_frozen=lambda runtime, window: resumed,
        _note_episode_window=lambda window, resumed: noted.append(resumed), _refresh_service_active=lambda: None)
    ValidationService._open_service_window(me, 2, "r")
    assert noted == [resumed] and told == [sorted([EPISODE, MATH])]   # an added env left inactive is not counted


# --- capacity attribution by operator: another hotkey of the same operator is the same miner ------------

@pytest.mark.parametrize("same_operator", [True, False])
def test_a_second_hotkey_of_the_same_operator_cannot_make_its_saturation_someone_elses(tmp_path, same_operator):
    env, ledger = saturating(tmp_path)
    operators = {HOTKEY: "5Coldkey", "5Alt": "5Coldkey" if same_operator else "5Stranger"}
    env.issuer._operator_of = operators.get
    for seed in range(3):
        close(env, open_(env, seed), "graded")                     # three graded, fewer than M
    others_fill_the_machine(env, hotkey="5Alt")                    # its sybil (or a stranger) fills the machine
    refused_open(env, 3)
    assert (env.precommit.sha256 in ledger._verdicts) is not same_operator
    assert decided_of(env, ledger) == (3 if same_operator else 0)  # its own doing: waste; a stranger's: neutral
