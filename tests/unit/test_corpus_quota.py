"""The corpus signed-sandbox quota: a hotkey holds as many sessions as it asks for, bounded by the fleet and the
other miners' fair shares (``sandbox.corpus_quota``), and the miner grows to the allowance it is told."""

import asyncio

import pytest

pytest.importorskip("reliquary_sandbox")

from reliquary.sandbox.corpus_quota import (
    LOW_YIELD, MAIN, NEWCOMER, CorpusQuota, stats_of, standing,
)
from reliquary.sandbox.sessions import (
    CLOSED, LIVE, SUBMITTED, SandboxPolicy, SessionBook, SessionRecord,
)

NOW = 1_800_000_000
JOB = "swe-signed"


def rec(sid, hotkey, prompt, *, state=LIVE, issued=NOW - 100, closed=None, status=None, job=JOB):
    return SessionRecord(session_id=sid, hotkey=hotkey, request_id=sid, engagement_sha256="e", kind="corpus",
                         engagement=f"corpus:{job}:{prompt}", env="reliquary-swe", split="train", index=prompt,
                         checkpoint="c", job_id=job, prompt_index=prompt, machine_id="m", issued_at=issued,
                         expires_at=issued + 5_000, token_sha256="t", state=state, closed_status=status,
                         closed_at=closed)


def history(book, hotkey, n, *, useful=True, issued=NOW - 72_000, held_s=3_600):
    """``n`` decided sessions of the hotkey: submitted (useful) or closed unpaid (wasted)."""
    for i in range(n):
        if useful:
            book.add(rec(f"{hotkey}-h{i}", hotkey, 10_000 + i, state=SUBMITTED, issued=issued,
                         closed=issued + held_s, status="graded"))
        else:
            book.add(rec(f"{hotkey}-h{i}", hotkey, 10_000 + i, state=CLOSED, issued=issued,
                         closed=issued + 60, status="open_failed"))


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def quota_of(book, capacity=37, **kw):
    clock = Clock()
    quota = CorpusQuota(book=book, capacity=lambda: capacity, clock=clock, refresh_s=0.0, **kw)
    return quota, clock


class Miner:
    def __init__(self, hotkey, batch):
        self.hotkey, self.batch, self.next, self.granted = hotkey, batch, 0, 0

    def ask(self, quota, book):
        """One admissible open of a fresh prompt; True when granted (the session then holds)."""
        held = sum(1 for r in book.of_hotkey(self.hotkey) if r.state == LIVE)
        if held >= self.batch:
            return False
        prompt = self.next
        self.next += 1
        refused = quota.admit(self.hotkey, NOW, engagement=f"corpus:{JOB}:{prompt}")
        if refused is not None:
            return False
        book.add(rec(f"{self.hotkey}-{prompt}", self.hotkey, prompt))
        self.granted += 1
        return True


def run(quota, book, miners, rounds=200):
    for _ in range(rounds):
        if not any([miner.ask(quota, book) for miner in miners]):
            # one more round: every miner's refused asks now count as demand
            if not any([miner.ask(quota, book) for miner in miners]):
                break
    return {m.hotkey: sum(1 for r in book.of_hotkey(m.hotkey) if r.state == LIVE) for m in miners}


def test_a_lone_newcomer_grows_to_its_batch_then_to_the_fleet():
    book = SessionBook(SandboxPolicy())
    quota, _ = quota_of(book, capacity=37)
    held = run(quota, book, [Miner("5A", batch=20)])
    assert held == {"5A": 20}                      # its own batch: no fixed 4 any more
    assert quota.allowance("5A") == 37             # what it could hold: the whole fleet
    held = run(quota, book, [Miner("5B", batch=64)])
    assert held["5B"] == 17                        # the rest of the fleet, never more
    refused = quota.admit("5B", NOW, engagement=f"corpus:{JOB}:999")
    assert refused.reason == "quota_live" and refused.retry_after == 10
    assert "allowance" in refused.detail and refused.detail["standing"] == NEWCOMER


def test_two_newcomers_share_the_fleet_evenly():
    book = SessionBook(SandboxPolicy())
    quota, _ = quota_of(book, capacity=37)
    held = run(quota, book, [Miner("5A", batch=64), Miner("5B", batch=64)])
    assert sum(held.values()) == 37 and abs(held["5A"] - held["5B"]) <= 1


def test_the_fleet_bounds_every_hotkey_together():
    book = SessionBook(SandboxPolicy())
    quota, _ = quota_of(book, capacity=10)
    held = run(quota, book, [Miner(f"5K{i}", batch=64) for i in range(3)])
    assert sum(held.values()) == 10 and max(held.values()) - min(held.values()) <= 1


def test_no_capacity_known_refuses_retryably():
    book = SessionBook(SandboxPolicy())
    quota, _ = quota_of(book, capacity=0)
    refused = quota.admit("5A", NOW, engagement=f"corpus:{JOB}:1")
    assert refused.reason == "quota_live" and refused.retry_after == 10 and refused.detail["allowance"] == 0


def test_standing_reads_the_sessions_in_the_book():
    book = SessionBook(SandboxPolicy())
    history(book, "5Main", 30)
    history(book, "5Waste", 30, useful=False)
    history(book, "5New", 5)
    assert standing(stats_of(book.of_hotkey("5Main"), NOW)[0]) == MAIN
    assert standing(stats_of(book.of_hotkey("5Waste"), NOW)[0]) == LOW_YIELD
    assert standing(stats_of(book.of_hotkey("5New"), NOW)[0]) == NEWCOMER
    stats, horizon = stats_of(book.of_hotkey("5Main"), NOW)
    assert (stats.useful, stats.decided, stats.useful_s_24h, horizon) == (30, 30, 30 * 3_600.0, 72_000.0)


def test_a_low_yield_hotkey_is_hard_capped():
    book = SessionBook(SandboxPolicy())
    history(book, "5Waste", 30, useful=False)
    quota, _ = quota_of(book, capacity=37)
    held = run(quota, book, [Miner("5Waste", batch=64)])
    assert held == {"5Waste": 1}                   # L = max(1, 3 % of 37)
    assert quota.lane("5Waste") == LOW_YIELD and quota.allowance("5Waste") == 1
    refused = quota.admit("5Waste", NOW, engagement=f"corpus:{JOB}:777")
    assert refused.reason == "quota_live"


def test_a_main_miner_keeps_its_weighted_share_against_a_newcomer():
    book = SessionBook(SandboxPolicy())
    history(book, "5Main", 200, held_s=3_600)      # w = 200 x 3600 / 72000 = 10 concurrent useful sessions
    quota, _ = quota_of(book, capacity=37)
    _demand(quota, "5Main", 64)                    # both ask for more than the fleet from the start
    _demand(quota, "5New", 64)
    alloc = quota.allocation(NOW, force=True)
    # N = 3 guaranteed to the newcomers, the 34 others by weight 10 : 4
    assert alloc.main["5Main"] == pytest.approx(34 * 10 / 14)
    assert alloc.newcomer == pytest.approx(3 + 34 * 4 / 14)
    held = run(quota, book, [Miner("5Main", batch=64), Miner("5New", batch=64)])
    assert held == {"5Main": 24, "5New": 12}       # each its floor
    assert quota.lane("5Main") == MAIN and quota.lane("5New") == NEWCOMER
    assert quota.allowance("5Main") == 24


def test_a_greedy_newcomer_keeps_what_it_opened_but_the_turnover_restores_the_shares():
    """No session is ever cut: a newcomer that filled an idle fleet keeps its sessions, and the main miner's share
    comes back as they end."""
    book = SessionBook(SandboxPolicy())
    history(book, "5Main", 200, held_s=3_600)
    quota, _ = quota_of(book, capacity=37)
    newcomer, main = Miner("5New", batch=64), Miner("5Main", batch=64)
    assert run(quota, book, [newcomer]) == {"5New": 37}
    assert run(quota, book, [main]) == {"5Main": 0}
    for record in book.of_hotkey("5New"):            # the newcomer's episodes end
        book.settle(record.session_id, CLOSED, now=NOW, status="expired")
    held = run(quota, book, [main, newcomer])
    assert held["5Main"] >= 24 and sum(held.values()) <= 37


def _demand(quota, hotkey, n):
    for i in range(n):
        quota.admit(hotkey, NOW, engagement=f"corpus:{JOB}:{hotkey}-{i}")


def test_splitting_a_hotkey_buys_nothing():
    book = SessionBook(SandboxPolicy())
    history(book, "5Whole", 60)
    history(book, "5PartA", 30)
    history(book, "5PartB", 30)
    quota, _ = quota_of(book, capacity=37)
    _demand(quota, "5Whole", 64)
    _demand(quota, "5PartA", 32)
    _demand(quota, "5PartB", 32)
    alloc = quota.allocation(NOW, force=True)
    assert alloc.main["5Whole"] == pytest.approx(alloc.main["5PartA"] + alloc.main["5PartB"])
    assert alloc.main["5Whole"] == pytest.approx(18.5)


def test_the_book_caps_follow_the_quota_once_installed():
    policy = SandboxPolicy()
    book = SessionBook(policy)
    for i in range(6):
        book.add(rec(f"s{i}", "5A", i))
    assert book.open_refusal("5A", NOW) is None
    assert book.engagement_refusal("5A", JOB, 99, NOW).reason == "job_live_cap"     # fixed: 4
    quota, _ = quota_of(book, capacity=37)
    quota.install()
    assert book.engagement_refusal("5A", JOB, 99, NOW) is None                        # ceiling: the fleet
    assert book.engagement_refusal("5A", JOB, 3, NOW).reason == "prompt_live_cap"     # one per prompt stays
    assert quota.ceilings("5A") == (37, 37)
    assert quota.opens_per_hour("5A") == max(policy.max_opens_per_hour, 12 * quota.allowance("5A"))
    assert book.cap_scale("5A") == quota.allowance("5A") / 4
    capped, _ = quota_of(SessionBook(policy), capacity=37, max_live_per_hotkey=10, max_live_per_hotkey_job=5)
    assert capped.ceilings("5A") == (10, 5)
    tiny, _ = quota_of(SessionBook(policy), capacity=2)
    assert tiny.ceilings("5A") == (policy.max_live_per_hotkey, policy.max_live_per_hotkey_job)


def test_a_restart_rebuilds_the_quota_from_the_restored_book():
    book = SessionBook(SandboxPolicy())
    history(book, "5Main", 120)
    history(book, "5Waste", 30, useful=False)
    for i in range(12):
        book.add(rec(f"live-{i}", "5Main", i))
    book.add(rec("live-w", "5Waste", 500))
    quota, _ = quota_of(book, capacity=37)
    before = {k: (quota.lane(k), quota.allowance(k)) for k in ("5Main", "5Waste", "5New")}
    restored = SessionBook(SandboxPolicy())
    restored.restore([SessionRecord.from_document(r.to_document()) for r in book.records()])
    again, _ = quota_of(restored, capacity=37)
    assert {k: (again.lane(k), again.allowance(k)) for k in ("5Main", "5Waste", "5New")} == before
    assert again.admit("5Waste", NOW, engagement=f"corpus:{JOB}:501").reason == "quota_live"   # its 1 is held
    assert again.admit("5Main", NOW, engagement=f"corpus:{JOB}:502") is None


def test_an_ask_that_opened_is_not_counted_twice():
    book = SessionBook(SandboxPolicy())
    quota, _ = quota_of(book, capacity=37)
    assert quota.admit("5A", NOW, engagement=f"corpus:{JOB}:1") is None
    book.add(rec("s1", "5A", 1))
    quota.allocation(NOW, force=True)
    assert quota._demands["5A"] == 1


def test_demand_is_remembered_after_a_turnover_then_forgotten(monkeypatch):
    book = SessionBook(SandboxPolicy())
    quota, clock = quota_of(book, capacity=37)
    for i in range(10):
        book.add(rec(f"s{i}", "5A", i))
    quota.allocation(NOW, force=True)
    for i in range(10):
        book.settle(f"s{i}", CLOSED, now=NOW, status="expired")
    quota.allocation(NOW + 60, force=True)
    assert quota._demands["5A"] == pytest.approx(10 * (1 - 60 / 600))
    quota.allocation(NOW + 700, force=True)
    assert "5A" not in quota._demands


# --- the issuer and the route ---

attest = pytest.importorskip("reliquary_sandbox.attest")

from tests.unit import test_sandbox_sessions as ts  # noqa: E402


def _issuer_env(tmp_path, quota=True, capacity=37):
    env = ts.build(tmp_path, remaining=5)
    if quota:
        q = CorpusQuota(book=env.book, capacity=lambda: capacity, clock=env.clock, refresh_s=0.0)
        q.install()
        env.issuer._quota = q
        env.quota = q
    return env


def _open(env, prompt, hotkey="5Hot"):
    return ts.open_(env, hotkey=hotkey, request_id=f"{prompt:032d}",
                    engagement={**ts.CORPUS, "prompt_index": prompt})


def test_the_issuer_grants_past_the_fixed_cap_and_names_the_allowance(tmp_path):
    env = _issuer_env(tmp_path)
    grants = [_open(env, prompt) for prompt in range(8)]
    assert all(isinstance(g, ts.Grant) for g in grants), grants
    assert {g.allowance for g in grants} == {37}


def test_without_the_quota_the_fixed_caps_bind(tmp_path):
    env = _issuer_env(tmp_path, quota=False)
    outcomes = [_open(env, prompt) for prompt in range(5)]
    assert all(isinstance(g, ts.Grant) and g.allowance is None for g in outcomes[:4])
    assert outcomes[4].reason == "job_live_cap"


def test_the_issuer_refuses_beyond_the_fleet_with_the_allowance(tmp_path):
    env = _issuer_env(tmp_path, capacity=3)
    outcomes = [_open(env, prompt) for prompt in range(4)]
    assert all(isinstance(g, ts.Grant) for g in outcomes[:3])
    refused = outcomes[3]
    assert refused.reason == "quota_live" and refused.detail["allowance"] == 3 and refused.retry_after == 10
    # Counted for its job: the order control's `quota_refusals_fleet_full`.
    assert env.quota.fleet_refusals(ts.CORPUS["job_id"]) == 1


def test_a_restored_issuer_keeps_counting_the_held_sessions(tmp_path):
    env = _issuer_env(tmp_path, capacity=3)
    for prompt in range(3):
        assert isinstance(_open(env, prompt), ts.Grant)
    fresh = ts.build(tmp_path / "again", remaining=5, store=env.store)
    asyncio.run(fresh.issuer.restore())
    quota = CorpusQuota(book=fresh.book, capacity=lambda: 3, clock=fresh.clock, refresh_s=0.0)
    quota.install()
    fresh.issuer._quota = quota
    refused = _open(fresh, 9)
    assert refused.reason == "quota_live" and refused.detail["live"] == 3


def test_the_route_answers_the_allowance_only_when_named():
    from tests.unit import test_sandbox_routes as tr

    grant = ts.Grant("s-1", tr.TOKEN, "http://10.0.0.5:8080", tr.NOW + 4500, allowance=12)
    answer = tr.client(tr.FakeIssuer(grant)).post("/corpus/sandbox/sessions", json=tr.open_body())
    assert answer.status_code == 200 and answer.json()["allowance"] == 12
    plain = ts.Grant("s-1", tr.TOKEN, "http://10.0.0.5:8080", tr.NOW + 4500)
    answer = tr.client(tr.FakeIssuer(plain)).post("/corpus/sandbox/sessions", json=tr.open_body())
    assert "allowance" not in answer.json()
    refused = ts.sessions.Refusal("quota_live", {"allowance": 2, "max": 2}, retry_after=10)
    answer = tr.client(tr.FakeIssuer(refused)).post("/corpus/sandbox/sessions", json=tr.open_body())
    assert answer.status_code == 429 and answer.headers["Retry-After"] == "10"
    assert answer.json()["detail"]["allowance"] == 2


# --- the wiring ---

def test_the_quota_is_on_by_default_and_can_be_turned_off(tmp_path):
    from reliquary.validator.sandbox_wiring import SandboxValidatorConfig
    from tests.unit.test_signed_grading_and_wiring import _services

    keys = {"RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE": str(tmp_path / "v.pem"),
            "RELIQUARY_SANDBOX_VALIDATOR_KEY_ID": "v1"}
    config = SandboxValidatorConfig.from_env(keys)
    assert config.quota and config.quota_max_live_per_hotkey == config.quota_max_live_per_hotkey_job == 0
    off = SandboxValidatorConfig.from_env({**keys, "RELIQUARY_SANDBOX_CORPUS_QUOTA": "0",
                                           "RELIQUARY_SANDBOX_QUOTA_MAX_LIVE_PER_HOTKEY": "40"})
    assert not off.quota and off.quota_max_live_per_hotkey == 40
    with pytest.raises(ValueError):
        SandboxValidatorConfig.from_env({**keys, "RELIQUARY_SANDBOX_CORPUS_QUOTA": "maybe"})
    with pytest.raises(ValueError):
        SandboxValidatorConfig.from_env({**keys, "RELIQUARY_SANDBOX_QUOTA_MAX_LIVE_PER_HOTKEY_JOB": "-1"})
    _, services = _services(tmp_path)
    assert services.quota is not None and services.book.live_caps == services.quota.ceilings
    assert services.issuer._quota is services.quota
    assert services.quota._capacity() == 0               # no served job, no machine yet
    (tmp_path / "off").mkdir()
    _, plain = _services(tmp_path / "off", quota=False)
    assert plain.quota is None and plain.book.live_caps is None and plain.issuer._quota is None


def test_the_fleet_counts_the_slots_of_machines_serving_the_signed_envs():
    from reliquary.sandbox.fleet import Fleet

    fleet = Fleet(read_documents=None, fetch_report=None, clock=lambda: NOW)
    documents = [("m1", {"capacity": 10, "env_packages": {"reliquary-swe": "pkg1"}}),
                 ("m2", {"capacity": 7, "env_packages": {"reliquary-swe": "pkg1", "other": "p"}}),
                 ("m3", {"capacity": 20, "env_packages": {"other": "p"}}),
                 ("m4", {"capacity": 5, "env_packages": {"reliquary-swe": "old"}})]
    fleet._fresh_documents = lambda now: iter(documents)
    assert fleet.slots_serving({("reliquary-swe", "pkg1")}) == 17
    assert fleet.slots_serving({("reliquary-swe", "pkg1"), ("other", "p")}) == 37
    assert fleet.slots_serving(set()) == 0


# --- the miner ---

from tests.unit import test_signed_miner as tm  # noqa: E402


def _miner_run(tmp_path, *, grant_extra, max_live, episodes=24, concurrency=16):
    signed = tm._transcript(tmp_path, status="expired")
    sessions = tm.FakeSessions(grant={**tm._grant(signed), **grant_extra})
    signed_runner = tm.runner(sessions, tm.FakeSandboxRunner(signed, status="expired", delay=0.01),
                              max_live=max_live)
    tm._mine_signed(signed_runner, episodes=episodes, concurrency=concurrency)
    return sessions, signed_runner


def test_the_miner_grows_to_the_announced_allowance(tmp_path):
    sessions, signed_runner = _miner_run(tmp_path, grant_extra={"allowance": 10}, max_live=64)
    assert signed_runner.allowance == 10 and signed_runner.live_limit == 10
    assert sessions.peak == 10 and len(sessions.closed) == 24


def test_the_miner_never_exceeds_its_own_bound(tmp_path):
    sessions, signed_runner = _miner_run(tmp_path, grant_extra={"allowance": 50}, max_live=6)
    assert signed_runner.live_limit == 6 and sessions.peak == 6


def test_an_old_validator_keeps_the_miner_at_four(tmp_path):
    sessions, signed_runner = _miner_run(tmp_path, grant_extra={}, max_live=64)
    assert signed_runner.allowance is None and signed_runner.live_limit == 4 and sessions.peak == 4


def test_a_quota_refusal_lowers_the_limit_and_holds_the_gate(tmp_path):
    signed = tm._transcript(tmp_path)
    signed_runner = tm.runner(tm.FakeSessions(grant=tm._grant(signed)), tm.FakeSandboxRunner(signed), max_live=64)
    signed_runner._note_allowance(20)
    assert signed_runner.live_limit == 20
    refused = tm.SessionRefused("quota_live", {"allowance": 2, "max": 2}, retry_after=10.0, status=429)
    assert signed_runner._note_refusal(refused) is refused
    assert signed_runner.live_limit == 2 and signed_runner._not_before >= 10.0
    signed_runner._note_allowance(0)
    assert signed_runner.live_limit == 1                  # always at least one try
    signed_runner._note_allowance(True)                   # not a count: ignored
    assert signed_runner.live_limit == 1


def test_the_live_limit_moves_both_ways():
    from reliquary.miner.signed_episode import _LiveLimit

    async def main():
        limit = _LiveLimit(1)
        await limit.acquire()
        waiters = [asyncio.ensure_future(limit.acquire()) for _ in range(3)]
        await asyncio.sleep(0)
        assert limit.held == 1
        limit.set_limit(3)                                  # grows: two waiters enter at once
        await asyncio.sleep(0)
        assert limit.held == 3 and sum(w.done() for w in waiters) == 2
        limit.set_limit(1)                                  # shrinks: holders finish, nobody new enters
        limit.release()
        await asyncio.sleep(0)
        assert limit.held == 2 and sum(w.done() for w in waiters) == 2
        pending = next(w for w in waiters if not w.done())
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        limit.release()
        limit.release()
        await limit.acquire()
        assert limit.held == 1

    asyncio.run(main())


def test_the_rl_runner_ignores_allowances():
    from reliquary.miner.rl_episode_client import RlSignedEpisodeRunner

    assert RlSignedEpisodeRunner.follows_allowance is False


def test_fleet_full_refusals_are_counted_per_job_for_ten_minutes():
    book = SessionBook(SandboxPolicy())
    quota, clock = quota_of(book, capacity=2)
    for prompt in range(2):
        assert quota.admit("5A", NOW, engagement=f"corpus:{JOB}:{prompt}", job_id=JOB) is None
        book.add(rec(f"5A-{prompt}", "5A", prompt))
    # The fleet is fully allocated: more capacity would have admitted these (whoever asks, whatever the check).
    assert quota.admit("5A", NOW, engagement=f"corpus:{JOB}:2", job_id=JOB).reason == "quota_live"
    assert quota.admit("5B", NOW + 30, engagement=f"corpus:{JOB}:3", job_id=JOB) is not None
    assert quota.admit("5B", NOW + 30, engagement="corpus:other:3", job_id="other") is not None
    assert quota.admit("5B", NOW + 30, engagement=f"corpus:{JOB}:4") is not None     # no job named: not counted
    assert (quota.fleet_refusals(JOB, NOW + 60), quota.fleet_refusals("other", NOW + 60)) == (2, 1)
    assert quota.fleet_refusals(JOB, NOW + 60, window_s=40) == 1
    assert quota.fleet_refusals(JOB, NOW + 601) == 1                                # the first one is 10 min old
    clock.now = NOW + 700
    assert quota.fleet_refusals(JOB) == 0 and quota.fleet_refusals("never") == 0


def test_no_capacity_known_is_fleet_full():
    book = SessionBook(SandboxPolicy())
    quota, _ = quota_of(book, capacity=0)
    assert quota.admit("5A", NOW, engagement=f"corpus:{JOB}:1", job_id=JOB) is not None
    assert quota.fleet_refusals(JOB, NOW) == 1


def test_a_refusal_with_free_capacity_is_the_hotkeys_own():
    book = SessionBook(SandboxPolicy())
    history(book, "5Waste", 30, useful=False)                 # low-yield: hard-capped at L = 1
    quota, _ = quota_of(book, capacity=37)
    assert quota.admit("5Waste", NOW, engagement=f"corpus:{JOB}:1", job_id=JOB) is None
    book.add(rec("5Waste-1", "5Waste", 1))
    refused = quota.admit("5Waste", NOW, engagement=f"corpus:{JOB}:2", job_id=JOB)
    assert refused.reason == "quota_live" and refused.detail["standing"] == LOW_YIELD
    assert quota.fleet_refusals(JOB, NOW) == 0


# --- declared wants (design 2026-11-13-sandbox-sizing-miner-demand) ---

from reliquary.protocol.sandbox_session import (  # noqa: E402
    MAX_SESSION_WANT, WANT_ACCEPTED, SandboxSessionOpenRequest,
)
from reliquary.sandbox.corpus_quota import FORECAST_WINDOW_S  # noqa: E402


def test_the_quota_keeps_the_last_want_per_job_and_hotkey_for_the_forecast_window():
    book = SessionBook(SandboxPolicy())
    quota, clock = quota_of(book)
    quota.declare("5A", JOB, 64, NOW - 100)
    quota.declare("5A", JOB, 48, NOW)                         # the last one wins
    quota.declare("5B", "other", MAX_SESSION_WANT + 100, NOW)  # bounded
    for junk in (None, 0, -3, True, "64", 2.5):
        quota.declare("5C", JOB, junk, NOW)
    quota.declare("5C", None, 8, NOW)
    assert quota.declarations(NOW) == {JOB: {"5A": (48, NOW)}, "other": {"5B": (MAX_SESSION_WANT, NOW)}}
    assert quota.declarations(NOW + FORECAST_WINDOW_S + 1) == {}


def test_a_declared_want_changes_neither_the_allocation_nor_the_allowance():
    book = SessionBook(SandboxPolicy())
    plain, _ = quota_of(book)
    declared, _ = quota_of(book)
    declared.declare("5A", JOB, 512, NOW)
    for quota in (plain, declared):
        assert quota.admit("5A", NOW, engagement=f"corpus:{JOB}:1", job_id=JOB) is None
    assert plain.allowance("5A") == declared.allowance("5A") == 37
    assert plain.view() == declared.view()


def test_a_quota_refusal_says_the_validator_reads_wants():
    book = SessionBook(SandboxPolicy())
    quota, _ = quota_of(book, capacity=0)
    assert quota.admit("5A", NOW, engagement=f"corpus:{JOB}:1").detail[WANT_ACCEPTED] is True
    quota, _ = quota_of(book, capacity=1)
    book.add(rec("5A-1", "5A", 1))
    assert quota.admit("5A", NOW, engagement=f"corpus:{JOB}:2").detail[WANT_ACCEPTED] is True


def test_the_issuer_hands_a_served_jobs_declared_want_to_the_quota(tmp_path):
    env = _issuer_env(tmp_path, capacity=3)
    grant = ts.open_(env, request_id="1" * 32, engagement={**ts.CORPUS, "want": 48})
    assert isinstance(grant, ts.Grant)
    job_id = ts.CORPUS["job_id"]
    assert env.quota.declarations()[job_id]["5Hot"][0] == 48
    # Refused for the fleet: still what the miner wants.
    for prompt in range(3):
        _open(env, 10 + prompt)
    refused = ts.open_(env, hotkey="5Other", request_id="2" * 32, engagement={**ts.CORPUS, "want": 16})
    assert refused.reason in ("quota_live", "prompt_unavailable")
    assert env.quota.declarations()[job_id]["5Other"][0] == 16
    # A job it cannot mine declares nothing.
    ts.open_(env, hotkey="5Far", request_id="3" * 32, engagement={**ts.CORPUS, "job_id": "nope", "want": 16})
    assert "nope" not in env.quota.declarations()
    # An open without a want (an older miner) declares nothing.
    _open(env, 20, hotkey="5Old")
    assert "5Old" not in env.quota.declarations().get(job_id, {})


def test_the_route_says_it_reads_wants_with_the_allowance():
    from tests.unit import test_sandbox_routes as tr

    grant = ts.Grant("s-1", tr.TOKEN, "http://10.0.0.5:8080", tr.NOW + 4500, allowance=12)
    issuer = tr.FakeIssuer(grant)
    body = tr.open_body(engagement={"kind": "corpus", "job_id": "swe-agentic-v1", "prompt_index": 3, "want": 40})
    answer = tr.client(issuer).post("/corpus/sandbox/sessions", json=body)
    assert answer.status_code == 200 and answer.json()[WANT_ACCEPTED] is True
    assert issuer.calls[0][1]["engagement"]["want"] == 40
    plain = ts.Grant("s-1", tr.TOKEN, "http://10.0.0.5:8080", tr.NOW + 4500)
    assert WANT_ACCEPTED not in tr.client(tr.FakeIssuer(plain)).post("/corpus/sandbox/sessions",
                                                                     json=tr.open_body()).json()


@pytest.mark.parametrize("want,ok", [(None, True), (1, True), (MAX_SESSION_WANT, True), (0, False),
                                     (MAX_SESSION_WANT + 1, False), (True, False), ("8", False), (8.0, False)])
def test_the_open_request_bounds_the_want(want, ok):
    from pydantic import ValidationError

    from tests.unit import test_sandbox_routes as tr

    engagement = {"kind": "corpus", "job_id": "swe-agentic-v1", "prompt_index": 3}
    if want is not None:
        engagement["want"] = want
    if ok:
        SandboxSessionOpenRequest.model_validate(tr.open_body(engagement=engagement))
    else:
        with pytest.raises(ValidationError):
            SandboxSessionOpenRequest.model_validate(tr.open_body(engagement=engagement))


def test_the_want_is_signed_with_the_engagement():
    from reliquary.miner.signed_episode import signed_open_request
    from reliquary.protocol.signatures import build_sandbox_open_binding

    kw = dict(hotkey="5Hot", job_id=JOB, prompt_index=3, sign_binding=lambda b: b.hex(), now=NOW,
              request_id="a" * 32, validator_hotkey="5Val", path="/corpus/sandbox/sessions")
    plain, declared = signed_open_request(**kw), signed_open_request(**kw, want=40)
    assert "want" not in plain["engagement"] and declared["engagement"]["want"] == 40
    assert plain["signature"] != declared["signature"]
    bound = build_sandbox_open_binding(declared, validator_hotkey="5Val", path="/corpus/sandbox/sessions")
    assert bound.hex() == declared["signature"]
    assert signed_open_request(**kw, want=10_000)["engagement"]["want"] == MAX_SESSION_WANT


def test_the_miner_declares_its_want_only_once_the_validator_reads_it(tmp_path):
    sessions, signed_runner = _miner_run(tmp_path, grant_extra={"allowance": 10, WANT_ACCEPTED: True}, max_live=64)
    wants = [body["engagement"].get("want") for body in sessions.opened]
    assert wants[0] is None and set(wants[1:]) == {64}      # the first open learns it
    assert signed_runner.declares_want and signed_runner.want == 64
    (tmp_path / "old").mkdir()
    old, old_runner = _miner_run(tmp_path / "old", grant_extra={"allowance": 10}, max_live=64)
    assert not old_runner.declares_want
    assert all("want" not in body["engagement"] for body in old.opened)


def test_the_miners_want_is_its_batch_within_its_bound(tmp_path):
    signed = tm._transcript(tmp_path)
    signed_runner = tm.runner(tm.FakeSessions(grant=tm._grant(signed)), tm.FakeSandboxRunner(signed), max_live=64)
    assert signed_runner.want == 64
    from reliquary.miner.signed_episode import SignedSweEpisodeRunner

    smaller = SignedSweEpisodeRunner(**{**_runner_kwargs(signed_runner), "max_live": 64, "want": 24})
    assert smaller.want == 24
    capped = SignedSweEpisodeRunner(**{**_runner_kwargs(signed_runner), "max_live": 6, "want": 24})
    assert capped.want == 6


def _runner_kwargs(signed_runner):
    return dict(job=tm.JOB, hotkey=tm.HOT, sign_binding=lambda binding: "s", sessions=signed_runner._sessions,
                model_name="m", renderer_model_dir="/ck", generate_url="http://127.0.0.1:1",
                sampling=tm.JOB.sampling, source=signed_runner._source)


def test_a_quota_refusal_also_turns_the_declaration_on(tmp_path):
    signed = tm._transcript(tmp_path)
    signed_runner = tm.runner(tm.FakeSessions(grant=tm._grant(signed)), tm.FakeSandboxRunner(signed), max_live=32)
    refused = tm.SessionRefused("quota_live", {"allowance": 2, WANT_ACCEPTED: True}, retry_after=10.0, status=429)
    signed_runner._note_refusal(refused)
    assert signed_runner.declares_want and signed_runner._declared_want() == 32


def test_the_rl_runner_never_declares():
    from reliquary.miner.rl_episode_client import RlSignedEpisodeRunner

    runner = object.__new__(RlSignedEpisodeRunner)
    runner.declares_want, runner.want = True, 16
    assert runner._declared_want() is None


def test_each_hotkey_declares_its_share_of_the_process_batch():
    from reliquary.miner.agentic_miner import declared_want

    assert declared_want(64, 8, 1) == 8                  # the default --concurrency
    assert declared_want(64, 128, 1) == 64               # bounded by --max-live-per-job
    assert declared_want(64, 100, 3) == 34               # shared by its identities, rounded up
    assert declared_want(64, 1, 4) == 1
