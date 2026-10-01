"""`GET /corpus/jobs/{job_id}/miners/{hotkey}`: one miner's audit state, verdict
counts, recent failures and pay, from in-memory state, cached, never a listing
per request."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from reliquary.corpus.audit_policy import AuditParams
from reliquary.validator.corpus_hot_jobs import CorpusJobSet
from reliquary.validator.corpus_miner_states import MinerStates
from reliquary.validator.corpus_miner_status import (
    MINER_STATUS_CACHE_SECONDS,
    RECENT_FAILURES,
    MinerBook,
)
from reliquary.validator.corpus_service import CorpusJobRoutes

HK = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
OTHER = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"
NOW = 1_000_000.0
PARAMS = AuditParams(probation_submissions=100)
THRESHOLDS = {"exp_mismatch": 8, "mant_mean": 0.02, "mant_median": 0.01}


def _sid(n: int) -> str:
    return f"{n:064x}"


def _verdict(hotkey=HK, *, passed=True, audited=True, tokens=10, reason=None, at=NOW, **extra):
    return {"schema": "reliquary/corpus-verdict/v1", "hotkey": hotkey, "token_count": tokens,
            "passed": passed, "audited": audited, "reason": reason, "audited_at": at,
            "worst_exp": 0, "worst_mant_mean": 0.0, "worst_mant_median": 0.0, **extra}


def _toploc_failure(at=NOW, hotkey=HK):
    return _verdict(hotkey, passed=False, reason="mant_err_mean", at=at, worst_exp=3,
                    worst_mant_mean=0.031, worst_mant_median=0.009, tokens=7000,
                    draw={"round": 5, "q": 1.0, "drawn": True}, scored_by=["exec-1"])


class _Store:
    """A counting fake of the record store: every call is recorded."""

    def __init__(self, verdicts=None, *, settlement=None, miners=None, voided=()):
        self.verdicts = dict(verdicts or {})
        self.settlement = dict(settlement or {})
        self.miners = dict(miners or {})
        self.voided = list(voided)
        self.calls: list[str] = []

    async def list_verdict_ids(self, job_id):
        self.calls.append("list_verdict_ids")
        return sorted(self.verdicts)

    async def list_submission_ids(self, job_id):
        self.calls.append("list_submission_ids")
        return []

    async def list_voided_ids(self, job_id):
        self.calls.append("list_voided_ids")
        return list(self.voided)

    async def read_verdict(self, job_id, sid):
        self.calls.append("read_verdict")
        return self.verdicts.get(sid)

    async def read_settlement(self, job_id):
        self.calls.append("read_settlement")
        return dict(self.settlement), "e1"

    async def read_miners(self, job_id):
        self.calls.append("read_miners")
        return dict(self.miners), "m1"

    async def write_miners(self, job_id, document, etag):
        self.calls.append("write_miners")
        self.miners = dict(document)
        return "m2"

    def listings(self):
        return [c for c in self.calls if c.startswith("list_")]


def _book(store=None, *, now=None, read_windows=None):
    clock = (lambda: now[0]) if now is not None else (lambda: NOW)
    return MinerBook(job_id="job-a", task_id="corpus-a", records=store or _Store(),
                     read_windows=read_windows, thresholds=THRESHOLDS, clock=clock)


class _Auditor:
    def __init__(self, pending=None):
        self.pending = dict(pending or {})

    def pending_count(self, hotkey):
        return self.pending.get(hotkey, 0)


def _wiring(job_id="job-a", *, store, book=None, pending=None, cap=0.1, now):
    states = MinerStates(store, job_id, clock=lambda: now[0])
    return SimpleNamespace(
        entry=SimpleNamespace(task_id=f"corpus-{job_id}", job_id=job_id), cap=cap,
        auditor=_Auditor(pending), miner_states=states, audit_params=PARAMS,
        miners=book or MinerBook(job_id=job_id, task_id=f"corpus-{job_id}", records=store,
                                 thresholds=THRESHOLDS, clock=lambda: now[0]))


def _set(*wirings, now):
    routes = CorpusJobRoutes()
    for w in wirings:
        routes.add(w.entry.job_id, SimpleNamespace())
    job_set = CorpusJobSet(routes=routes, router_for=None, wire=None, jobs_of=lambda w: [],
                           clock=lambda: now[0])
    for w in wirings:
        job_set.served[w.entry.job_id] = w
    return job_set


def _status(job_set, job_id, hotkey):
    async def go():
        status = await job_set.miner_status(job_id, hotkey)
        # Let a backfill the request started run to its end.
        for w in job_set.served.values():
            await w.miners.wait_backfill()
        return status

    return asyncio.run(go())


# --------------------------------------------------------------------------
# The book: counts, failures, pay
# --------------------------------------------------------------------------


def test_a_failed_toploc_verdict_surfaces_its_reason_and_measures():
    book = _book()
    book.observe(_sid(1), _toploc_failure())
    failure, = book.counts(HK)["recent_failures"]
    assert failure == {
        "submission_id": _sid(1), "audited_at": NOW, "reason": "mant_err_mean",
        "token_count": 7000, "worst_exp": 3, "worst_mant_mean": 0.031,
        "worst_mant_median": 0.009,
    }
    # What the verdict holds about the validator (the executor, the draw) stays private.
    assert "exec-1" not in json.dumps(book.counts(HK))


def test_every_verdict_kind_is_counted_once_and_only_for_its_hotkey():
    book = _book()
    book.observe(_sid(1), _verdict(tokens=10))
    book.observe(_sid(1), _verdict(tokens=10))  # reported twice: counted once
    book.observe(_sid(2), _verdict(audited=False, tokens=5))
    book.observe(_sid(3), _toploc_failure())
    book.observe(_sid(4), _verdict(passed=False, audited=False, reason="banned"))
    book.observe(_sid(5), _verdict(OTHER, passed=False, reason="exp_mismatch"))
    book.observe(_sid(6), None)
    counts = book.counts(HK)
    assert {k: counts[k] for k in ("verdicts", "audited", "passed", "passed_unaudited",
                                   "failed", "voided")} == {
        "verdicts": 4, "audited": 2, "passed": 2, "passed_unaudited": 1, "failed": 1,
        "voided": 1}
    assert [f["submission_id"] for f in counts["recent_failures"]] == [_sid(3)]
    assert book.counts(OTHER)["failed"] == 1


def test_recent_failures_are_the_newest_twenty_newest_first():
    book = _book()
    for n in range(RECENT_FAILURES + 5):
        book.observe(_sid(n), _toploc_failure(at=NOW + n))
    failures = book.counts(HK)["recent_failures"]
    assert len(failures) == RECENT_FAILURES
    assert failures[0]["audited_at"] == NOW + RECENT_FAILURES + 4
    assert failures[-1]["audited_at"] == NOW + 5
    # An older failure learnt late (the backfill) never displaces a newer one.
    book.observe(_sid(999), _toploc_failure(at=NOW - 50))
    assert book.counts(HK)["recent_failures"] == failures


def test_verified_tokens_count_once_settled_and_a_voided_pass_is_never_paid():
    book = _book()
    book.observe(_sid(1), _verdict(tokens=10))
    book.observe(_sid(2), _verdict(tokens=20))
    book.observe(_sid(3), _toploc_failure())
    assert book.counts(HK)["verified_tokens_settled"] == 0
    book.voided(_sid(2), {"hotkey": HK, "reason": "executor_quarantined", "voided_at": NOW,
                          "passed": False, "worst_exp": 9})
    book.settled([_sid(1), _sid(2), _sid(3)])
    counts = book.counts(HK)
    assert counts["verified_tokens_settled"] == 10
    assert counts["passed"] == 1 and counts["voided"] == 1
    assert counts["recent_failures"][0]["reason"] == "executor_quarantined"
    # Settled before it was seen (written before a restart): paid when seen.
    book.settled([_sid(4)])
    book.observe(_sid(4), _verdict(tokens=7))
    assert book.counts(HK)["verified_tokens_settled"] == 17


def test_the_share_is_this_hotkeys_part_of_the_last_settled_windows():
    book = _book()
    book.window(100, {HK: 0.025, OTHER: 0.075})
    book.window(101, {OTHER: 0.1})
    share = book.share(HK)
    assert share["windows"] == 2 and share["first_window"] == 100 and share["last_window"] == 101
    assert share["reward"] == pytest.approx(0.025)
    assert share["share"] == pytest.approx(0.125)
    assert book.share("5Nobody")["share"] == 0.0


def test_only_the_last_k_windows_are_kept():
    from reliquary.validator.corpus_miner_status import SHARE_WINDOWS

    book = _book()
    for w in range(SHARE_WINDOWS + 3):
        book.window(w, {HK: 1.0} if w < 3 else {OTHER: 1.0})
    share = book.share(HK)
    assert share["windows"] == SHARE_WINDOWS and share["first_window"] == 3
    assert share["share"] == 0.0


# --------------------------------------------------------------------------
# The backfill: once, lazily, in the background
# --------------------------------------------------------------------------


def test_the_backfill_counts_what_was_written_before_this_process():
    store = _Store(
        {_sid(1): _verdict(tokens=10), _sid(2): _toploc_failure(), _sid(3): _verdict(tokens=4),
         _sid(4): _verdict(tokens=99)},
        settlement={"settled": [_sid(1), _sid(2), _sid(4)], "last_window": 41},
        voided=[_sid(4)])
    asked = []

    async def read_windows(task_id, last_window, k):
        asked.append((task_id, last_window))
        return [{"window_start": 40, "rewards_by_hotkey": {HK: 0.05, OTHER: 0.05}},
                {"window_start": 41, "rewards_by_hotkey": {HK: 0.1}}]

    book = _book(store, read_windows=read_windows)
    assert book.counts(HK)["counts_complete"] is False
    book.observe(_sid(3), _verdict(tokens=4))  # live, before the backfill reads it
    asyncio.run(book.backfill())
    counts = book.counts(HK)
    assert counts["counts_complete"] is True
    assert counts["verdicts"] == 4 and counts["passed"] == 2 and counts["failed"] == 1
    assert counts["voided"] == 1
    assert counts["verified_tokens_settled"] == 10
    assert asked == [("corpus-a", 41)]
    assert book.share(HK)["share"] == pytest.approx(0.75)
    assert store.listings() == ["list_voided_ids", "list_verdict_ids"]


def test_a_backfill_that_fails_says_so_and_is_retried_later_never_raised():
    store = _Store({_sid(1): _verdict()})

    async def broken(job_id):
        raise OSError("r2 down")

    store.list_verdict_ids = broken
    now = [NOW]
    book = _book(store, now=now)
    asyncio.run(book.backfill())
    assert book.counts(HK)["counts_complete"] is False

    async def again():
        book.start_backfill()
        await book.wait_backfill()

    asyncio.run(again())
    assert store.calls.count("read_settlement") == 1  # not retried at once
    del store.list_verdict_ids
    now[0] += 3600
    asyncio.run(again())
    assert book.counts(HK)["counts_complete"] is True


# --------------------------------------------------------------------------
# The status: states, cache, cost
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry,state,extra", [
    ({}, "probation", {"probation_remaining": 100, "suspect_until": None, "banned_until": None}),
    ({"audited_passed": 40}, "probation", {"probation_remaining": 60}),
    ({"audited_passed": 100}, "sampled", {"probation_remaining": None}),
    ({"audited_passed": 0, "suspect_until": NOW + 50}, "suspect",
     {"suspect_until": NOW + 50, "banned_until": None}),
    ({"banned_until": NOW + 99, "audited_passed": 70}, "banned",
     {"banned_until": NOW + 99, "probation_remaining": None}),
    # A ban that ended: a fresh probation, whatever was counted before.
    ({"banned_until": NOW - 1, "audited_passed": 70}, "probation",
     {"probation_remaining": 100, "banned_until": None}),
    ({"suspect_until": NOW - 1, "audited_passed": 3}, "probation",
     {"probation_remaining": 97, "suspect_until": None}),
])
def test_the_audit_state_and_its_deadlines(entry, state, extra):
    now = [NOW]
    store = _Store(miners={HK: entry})
    job_set = _set(_wiring(store=store, now=now), now=now)
    status = _status(job_set, "job-a", HK)
    assert status["audit_state"] == state
    for key, value in extra.items():
        assert status[key] == value, key


def test_every_field_of_one_miners_status():
    now = [NOW]
    store = _Store(miners={HK: {"audited_passed": 12}})
    w = _wiring(store=store, now=now, pending={HK: 2}, cap=0.04)
    w.miners.observe(_sid(1), _verdict(tokens=10))
    w.miners.observe(_sid(2), _toploc_failure())
    w.miners.settled([_sid(1), _sid(2)])
    w.miners.window(7, {HK: 0.01, OTHER: 0.03})
    asyncio.run(w.miners.backfill())  # a first request starts it, and answers before it ends
    status = _status(_set(w, now=now), "job-a", HK)
    assert status == {
        "job_id": "job-a", "hotkey": HK, "as_of": NOW,
        "audit_state": "probation", "probation_remaining": 88,
        "suspect_until": None, "banned_until": None,
        "submissions_accepted": 4, "audited": 2, "passed": 1, "passed_unaudited": 0,
        "failed": 1, "pending_audit": 2, "voided": 0, "counts_complete": True,
        "recent_failures": [{
            "submission_id": _sid(2), "audited_at": NOW, "reason": "mant_err_mean",
            "token_count": 7000, "worst_exp": 3, "worst_mant_mean": 0.031,
            "worst_mant_median": 0.009}],
        "toploc_thresholds": THRESHOLDS,
        "verified_tokens_settled": 10,
        "share_last_windows": {"windows": 1, "first_window": 7, "last_window": 7,
                               "reward": 0.01, "share": 0.25},
        "cap": 0.04,
    }


def test_an_unknown_hotkey_is_a_probation_miner_with_nothing_counted():
    now = [NOW]
    store = _Store(miners={HK: {"audited_passed": 5}})
    status = _status(_set(_wiring(store=store, now=now), now=now), "job-a", OTHER)
    assert status["audit_state"] == "probation" and status["probation_remaining"] == 100
    assert all(status[k] == 0 for k in ("submissions_accepted", "audited", "passed", "failed",
                                        "pending_audit", "voided", "verified_tokens_settled"))
    assert status["recent_failures"] == [] and status["share_last_windows"]["windows"] == 0


def test_a_job_not_served_has_no_miner_status():
    now = [NOW]
    job_set = _set(_wiring(store=_Store(), now=now), now=now)
    assert _status(job_set, "nope", HK) is None


def test_requests_cost_no_listing_and_one_miners_read_per_period():
    now = [NOW]
    store = _Store({_sid(1): _verdict()}, miners={HK: {}})
    job_set = _set(_wiring(store=store, now=now), now=now)
    for _ in range(20):
        _status(job_set, "job-a", HK)
        _status(job_set, "job-a", OTHER)
    # The one backfill: two listings, ever; one miners.json read for both hotkeys.
    assert store.listings() == ["list_voided_ids", "list_verdict_ids"]
    assert store.calls.count("read_miners") == 1
    now[0] += MINER_STATUS_CACHE_SECONDS + 1
    for _ in range(20):
        _status(job_set, "job-a", HK)
    assert store.listings() == ["list_voided_ids", "list_verdict_ids"]
    assert store.calls.count("read_miners") == 2


def test_the_auditors_own_reads_keep_the_miners_document_fresh():
    now = [NOW]
    store = _Store(miners={HK: {"banned_until": NOW + 10}})
    w = _wiring(store=store, now=now)
    asyncio.run(w.miner_states.get(HK))  # what every submission's ban check reads
    job_set = _set(w, now=now)
    assert _status(job_set, "job-a", HK)["audit_state"] == "banned"
    assert store.calls.count("read_miners") == 1
    asyncio.run(w.miner_states.update(HK, lambda m: replace(m, banned_until=None,
                                                              audited_passed=100)))
    reads = store.calls.count("read_miners")
    now[0] += 10
    assert _status(job_set, "job-a", HK)["audit_state"] == "banned"  # this hotkey's cache
    assert _status(_set(w, now=now), "job-a", HK)["audit_state"] == "sampled"
    assert store.calls.count("read_miners") == reads  # the write's document, no read


def test_one_hotkeys_status_is_cached_for_the_period():
    now = [NOW]
    store = _Store(miners={})
    w = _wiring(store=store, now=now)
    job_set = _set(w, now=now)
    assert _status(job_set, "job-a", HK)["failed"] == 0
    w.miners.observe(_sid(1), _toploc_failure())
    assert _status(job_set, "job-a", HK)["failed"] == 0
    now[0] += MINER_STATUS_CACHE_SECONDS + 1
    assert _status(job_set, "job-a", HK)["failed"] == 1


def test_the_cache_holds_a_bounded_number_of_hotkeys(monkeypatch):
    from reliquary.validator import corpus_hot_jobs

    monkeypatch.setattr(corpus_hot_jobs, "MINER_STATUS_ENTRIES", 3)
    now = [NOW]
    job_set = _set(_wiring(store=_Store(), now=now), now=now)
    for n in range(10):
        _status(job_set, "job-a", f"5Hotkey{n:040d}")
    assert len(job_set._miner_cache) == 3


def test_a_miners_document_that_cannot_be_read_is_unavailable():
    now = [NOW]
    store = _Store()

    async def broken(job_id):
        raise OSError("r2 down")

    store.read_miners = broken
    job_set = _set(_wiring(store=store, now=now), now=now)
    with pytest.raises(OSError):
        _status(job_set, "job-a", HK)


# --------------------------------------------------------------------------
# The hooks that feed the book
# --------------------------------------------------------------------------


def test_the_auditor_reports_a_void_it_writes():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    seen = []
    auditor = CorpusAuditor(job_id="j", records=None, model=None, tokenizer=None, proof=None,
                            on_voided=lambda sid, doc: seen.append((sid, doc["reason"])))
    auditor._report_voided(_sid(1), {"reason": "executor_quarantined"})
    assert seen == [(_sid(1), "executor_quarantined")]

    def broken(sid, doc):
        raise RuntimeError("bug")

    auditor._on_voided = broken
    auditor._report_voided(_sid(1), {})  # a subscriber's bug never reaches the auditor


def test_the_auditor_counts_each_hotkeys_records_awaiting_a_verdict():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    auditor = CorpusAuditor(job_id="j", records=None, model=None, tokenizer=None, proof=None)
    auditor._unjudged = {HK: {_sid(1), _sid(2)}}
    assert auditor.pending_count(HK) == 2 and auditor.pending_count(OTHER) == 0


def test_the_settler_reports_each_window_it_pays_and_survives_a_bad_subscriber():
    from tests.unit.test_corpus_settlement import _Archives, _Records, _settler, _v

    records = _Records({"1" * 64: _v("A", 10), "2" * 64: _v("B", 30)})
    settler = _settler(records, _Archives(46000))
    seen = []
    settler.on_window = lambda window, rewards: seen.append((window, rewards))
    asyncio.run(settler.settle_once())
    assert seen == [(46000, pytest.approx({"A": 0.025, "B": 0.075}))]

    def broken(window, rewards):
        raise RuntimeError("bug")

    records.verdicts["3" * 64] = _v("A", 5)
    settler = _settler(records, _Archives(46001))
    settler.on_window = broken
    assert asyncio.run(settler.settle_once()) == 46001


def test_the_miners_document_is_mirrored_from_every_read_and_write():
    store = _Store(miners={HK: {"audited_passed": 3}})
    now = [NOW]
    states = MinerStates(store, "job-a", clock=lambda: now[0])
    assert states.mirror() is None
    asyncio.run(states.get(HK))
    document, at = states.mirror()
    assert document[HK]["audited_passed"] == 3 and at == NOW
    now[0] += 5
    asyncio.run(states.update(OTHER, lambda m: replace(m, audited_passed=1)))
    document, at = states.mirror()
    assert document[OTHER]["audited_passed"] == 1 and at == NOW + 5


def test_wiring_feeds_the_book_beside_the_status_counts():
    from reliquary.validator.corpus_job_status import JobStats
    from reliquary.validator.corpus_miner_status import feed

    stats = JobStats()
    book = _book()
    on_verdict, on_settled = feed(stats, book)
    on_verdict(_sid(1), _verdict(tokens=10))
    on_settled([_sid(1)])
    assert stats.unsettled() == (0, 0, 0)
    assert book.counts(HK)["verified_tokens_settled"] == 10


# --------------------------------------------------------------------------
# The routes
# --------------------------------------------------------------------------


from tests.unit.test_corpus_service import _r2_client, fake_r2, seeded_job  # noqa: E402,F401


def test_the_routes_answer_per_job_and_the_legacy_path_for_the_default(seeded_job):  # noqa: F811
    from tests.unit.test_corpus_route_skip import _app

    client = _app(seeded_job, ("job-a", "job-b"))
    now = [NOW]
    a = _wiring("job-a", store=_Store(miners={HK: {"banned_until": NOW + 9}}), now=now)
    b = _wiring("job-b", store=_Store(miners={HK: {"audited_passed": 100}}), now=now)
    b.miners.observe(_sid(1), _toploc_failure())
    job_set = CorpusJobSet(routes=client.app.state.corpus_routes, router_for=None, wire=None,
                           jobs_of=lambda w: [], clock=lambda: now[0])
    job_set.served.update({"job-a": a, "job-b": b})
    client.app.state.corpus_jobs = job_set
    first = client.get(f"/corpus/jobs/job-a/miners/{HK}")
    assert first.status_code == 200, first.text
    assert first.json()["audit_state"] == "banned" and first.json()["failed"] == 0
    second = client.get(f"/corpus/jobs/job-b/miners/{HK}").json()
    assert second["audit_state"] == "sampled" and second["failed"] == 1
    assert second["recent_failures"][0]["reason"] == "mant_err_mean"
    assert client.get(f"/corpus/miners/{HK}").json()["job_id"] == "job-a"
    assert client.get(f"/corpus/jobs/nope/miners/{HK}").status_code == 404
    assert client.get("/corpus/jobs/job-a/miners/not%20a%20hotkey!").status_code == 400
    assert client.get(f"/corpus/jobs/job-a/miners/{'5' * 200}").status_code == 400


def test_the_route_answers_503_while_the_state_cannot_be_read(seeded_job):  # noqa: F811
    from tests.unit.test_corpus_route_skip import _app

    client = _app(seeded_job, ("job-a",))
    now = [NOW]
    store = _Store()

    async def broken(job_id):
        raise OSError("r2 down")

    store.read_miners = broken
    job_set = CorpusJobSet(routes=client.app.state.corpus_routes, router_for=None, wire=None,
                           jobs_of=lambda w: [], clock=lambda: now[0])
    job_set.served["job-a"] = _wiring("job-a", store=store, now=now)
    client.app.state.corpus_jobs = job_set
    response = client.get(f"/corpus/jobs/job-a/miners/{HK}")
    assert response.status_code == 503
    assert response.json()["detail"] == "corpus_miner_status_unavailable"


def test_before_the_job_set_is_attached_no_job_is_served(seeded_job):  # noqa: F811
    from tests.unit.test_corpus_route_skip import _app

    client = _app(seeded_job, ("job-a",))
    assert client.get(f"/corpus/jobs/job-a/miners/{HK}").status_code == 404
    assert client.get(f"/corpus/miners/{HK}").status_code == 404


# --------------------------------------------------------------------------
# The miner's CLI
# --------------------------------------------------------------------------


def _example_status():
    return {
        "job_id": "job-a", "hotkey": HK, "as_of": NOW, "audit_state": "suspect",
        "probation_remaining": None, "suspect_until": NOW + 3600, "banned_until": None,
        "submissions_accepted": 12, "audited": 10, "passed": 9, "passed_unaudited": 0,
        "failed": 1, "pending_audit": 2, "voided": 0, "counts_complete": True,
        "recent_failures": [{"submission_id": _sid(2), "audited_at": NOW,
                             "reason": "mant_err_mean", "token_count": 7000, "worst_exp": 3,
                             "worst_mant_mean": 0.031, "worst_mant_median": 0.009}],
        "toploc_thresholds": THRESHOLDS, "verified_tokens_settled": 4200,
        "share_last_windows": {"windows": 3, "first_window": 10, "last_window": 12,
                               "reward": 0.012, "share": 0.1},
        "cap": 0.04,
    }


def test_the_summary_names_the_state_counts_failures_and_pay():
    from reliquary.miner.corpus_status import format_miner_status

    text = format_miner_status(_example_status())
    for fragment in ("job-a", HK, "suspect", "until", "accepted 12", "failed 1",
                     "pending 2", "mant_err_mean", "0.031", "4200", "10.0%", "0.04"):
        assert fragment in text, fragment


def test_the_cli_prints_the_summary_or_the_json(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.miner import corpus_status

    asked = []

    def fetch(validator_url, hotkey, job_id=None):
        asked.append((validator_url, hotkey, job_id))
        return _example_status()

    monkeypatch.setattr(corpus_status, "fetch_miner_status", fetch)
    runner = CliRunner()
    result = runner.invoke(app, ["corpus", "status", "--validator-url", "http://v",
                                 "--job-id", "job-a", "--hotkey", HK])
    assert result.exit_code == 0, result.output
    assert "mant_err_mean" in result.output
    result = runner.invoke(app, ["corpus", "status", "--validator-url", "http://v",
                                 "--hotkey", HK, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == _example_status()
    assert asked == [("http://v", HK, "job-a"), ("http://v", HK, None)]


def test_the_fetch_uses_the_job_scoped_or_the_legacy_path():
    import httpx

    from reliquary.miner.corpus_status import MinerStatusError, fetch_miner_status

    paths = []

    def handler(request):
        paths.append(request.url.path)
        if "nope" in request.url.path:
            return httpx.Response(404, json={"detail": "corpus_job_not_served"})
        return httpx.Response(200, json={"job_id": "x"})

    transport = httpx.MockTransport(handler)
    assert fetch_miner_status("http://v", HK, "job-a", transport=transport) == {"job_id": "x"}
    fetch_miner_status("http://v/", HK, transport=transport)
    assert paths == [f"/corpus/jobs/job-a/miners/{HK}", f"/corpus/miners/{HK}"]
    with pytest.raises(MinerStatusError, match="corpus_job_not_served"):
        fetch_miner_status("http://v", HK, "nope", transport=transport)
