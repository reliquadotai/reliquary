"""`GET /corpus/pay`: what each active corpus job is paying per verified token,
estimated from memory, the period archives and the chain, cached."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_periods as cp
from reliquary.validator import corpus_pay as pay
from tests.unit.test_corpus_period_settlement import (
    GENESIS, P, Archives, Records, at, settler as period_settler, verdict,
)

EMISSION = 147.6  # miner alpha per period
PRICED = cp.price_params(0.9, EMISSION, 1.0)  # 0.9 alpha per million tokens
PRICE = PRICED[cp.MAX_PRICE_PARAM]


# -- the projection ---------------------------------------------------------------


def test_a_priced_period_with_low_supply_pays_the_price_per_token():
    view = pay.projection(cap=0.15, price=PRICE, tokens=2_000_000, emission=EMISSION)
    assert view["budget_pool_share"] == pytest.approx(2_000_000 * PRICE)
    assert view["projected_alpha_per_mtoken"] == pytest.approx(0.9)
    assert view["saturated"] is False


def test_a_priced_period_with_high_supply_shares_the_cap():
    tokens = 100_000_000  # past cap / price = 24.6 M tokens
    view = pay.projection(cap=0.15, price=PRICE, tokens=tokens, emission=EMISSION)
    assert view["budget_pool_share"] == pytest.approx(0.15)
    assert view["projected_alpha_per_mtoken"] == pytest.approx(0.15 / tokens * 1e6 * EMISSION)
    assert view["projected_alpha_per_mtoken"] < 0.9 and view["saturated"] is True


def test_an_unpriced_period_shares_the_cap_whatever_the_supply():
    view = pay.projection(cap=0.05, price=None, tokens=1_000_000, emission=EMISSION)
    assert view["budget_pool_share"] == 0.05
    assert view["projected_pool_share_per_mtoken"] == pytest.approx(0.05)
    assert view["projected_alpha_per_mtoken"] == pytest.approx(0.05 * EMISSION)
    assert view["saturated"] is True


def test_an_empty_period_projects_nothing_and_an_unknown_emission_no_alpha():
    empty = pay.projection(cap=0.05, price=PRICE, tokens=0, emission=EMISSION)
    assert empty["budget_pool_share"] == 0.0
    assert empty["projected_alpha_per_mtoken"] is None
    assert empty["projected_pool_share_per_mtoken"] is None
    unknown = pay.projection(cap=0.05, price=None, tokens=None, emission=EMISSION)
    assert unknown["verified_tokens"] is None and unknown["budget_pool_share"] is None
    no_chain = pay.projection(cap=0.05, price=None, tokens=10, emission=None)
    assert no_chain["projected_pool_share_per_mtoken"] == pytest.approx(0.05 / 10 * 1e6)
    assert no_chain["projected_alpha_per_mtoken"] is None and no_chain["budget_alpha"] is None


def test_tokens_to_saturate_is_the_cap_over_the_price():
    assert pay.tokens_to_saturate(0.15, PRICE) == pytest.approx(0.15 / PRICE)
    assert pay.tokens_to_saturate(0.15, None) is None


# -- what the settler counts of the periods still open -----------------------------


def test_the_settler_counts_unsettled_verified_tokens_by_period_received():
    records, archives = Records(), Archives()
    records.verdicts = {
        "a": verdict("a", "a", 300, at(3)),          # period 3: closed, settled this pass
        "b": verdict("b", "b", 100, at(5)),          # the current period
        "c": verdict("c", "c", 50, at(5)),
        "d": verdict("d", "d", 70, at(5), passed=False),
        "e": verdict("e", "e", 40, at(5)),           # voided: never paid
    }
    records.voided = {"e"}
    s = period_settler(records, archives, at(5, 1000))
    assert s.open_periods is None  # nothing counted before a pass
    asyncio.run(s.settle_once())
    assert s.open_periods == {"5": {"verdicts": 4, "passed": 2, "verified_tokens": 150}}
    assert s.open_periods_at == at(5, 1000)


def test_the_settler_says_how_old_the_oldest_undecided_submission_is():
    s = period_settler(Records(), Archives(), at(5), oldest=at(4, 7))
    assert s.undecided_since() == {"known": True, "at": at(4, 7)}

    def unknown():
        raise LookupError("not seeded")

    s._oldest_pending = unknown
    assert s.undecided_since() == {"known": False, "at": None}
    s._oldest_pending = lambda: float("-inf")
    assert s.undecided_since() == {"known": False, "at": None}
    s._oldest_pending = lambda: None
    assert s.undecided_since() == {"known": True, "at": None}


def test_an_empty_job_counts_no_open_period():
    s = period_settler(Records(), Archives(), at(5))
    asyncio.run(s.settle_once())
    assert s.open_periods == {}


# -- the history, from the period archives -----------------------------------------


def _archive(work, entry, tokens, paid, *, cap=0.15, budget=None):
    doc = {"work_period": work, "entry_period": entry, "tokens": tokens, "cap": cap,
           "rewards_by_hotkey": {"h1": paid * 0.75, "h2": paid * 0.25}, "verdicts": 3}
    if budget is not None:
        doc.update(max_price_per_token=PRICE, budget=budget)
    return doc


def test_history_rows_join_a_periods_archives_and_keep_the_last_ones():
    docs = [_archive(w, w + 1, 1_000_000, 0.01) for w in range(20)]
    docs.append(_archive(19, 21, 500_000, 0.005))  # a late archive of period 19
    docs.append(_archive(20, 21, 2_000_000, 2_000_000 * PRICE, budget=2_000_000 * PRICE))
    rows = pay.history_rows(docs, emission=EMISSION, limit=12)
    assert [r["work_period"] for r in rows] == list(range(20, 8, -1))
    assert rows[0]["budget"] == pytest.approx(2_000_000 * PRICE)
    assert rows[0]["paid_alpha_per_mtoken"] == pytest.approx(0.9)
    assert rows[1] == {"work_period": 19, "verified_tokens": 1_500_000,
                       "paid_pool_share": pytest.approx(0.015), "budget": 0.15,
                       "paid_pool_share_per_mtoken": pytest.approx(0.01),
                       "paid_alpha_per_mtoken": pytest.approx(0.01 * EMISSION)}
    assert pay.history_rows(docs, emission=None, limit=1)[0]["paid_alpha_per_mtoken"] is None


class CountingArchives(Archives):
    def __init__(self):
        super().__init__()
        self.lists, self.reads = 0, []

    async def list(self, task_id):
        self.lists += 1
        return await super().list(task_id)

    async def read(self, task_id, work, entry):
        self.reads.append((work, entry))
        return await super().read(task_id, work, entry)


def test_the_history_lists_once_per_refresh_and_reads_each_archive_once():
    async def go():
        now = [0.0]
        archives = CountingArchives()
        for w in range(15):
            archives.docs[(w, w + 1)] = _archive(w, w + 1, 1000, 0.01)
        history = pay.PeriodHistory(archives, clock=lambda: now[0], limit=12)
        assert history.docs("corpus-x") is None  # loading, behind the request
        await history.settle()
        first = history.docs("corpus-x")
        assert [d["work_period"] for d in first] == list(range(3, 15))
        assert archives.lists == 1 and len(archives.reads) == 12
        for _ in range(50):
            history.docs("corpus-x")
        await history.settle()
        assert archives.lists == 1
        archives.docs[(15, 16)] = _archive(15, 16, 1000, 0.01)
        now[0] += pay.HISTORY_REFRESH_SECONDS + 1
        history.docs("corpus-x")
        await history.settle()
        assert archives.lists == 2 and archives.reads[12:] == [(15, 16)]
        assert history.docs("corpus-x")[-1]["work_period"] == 15

    asyncio.run(go())


def test_an_unreadable_bucket_keeps_the_last_history():
    async def go():
        now = [0.0]
        archives = CountingArchives()
        archives.docs[(1, 2)] = _archive(1, 2, 1000, 0.01)
        history = pay.PeriodHistory(archives, clock=lambda: now[0])
        history.docs("t")
        await history.settle()

        async def broken(task_id):
            raise ConnectionError("r2 down")

        archives.list = broken
        now[0] += pay.HISTORY_REFRESH_SECONDS + 1
        history.docs("t")
        await history.settle()
        assert [d["work_period"] for d in history.docs("t")] == [1]

    asyncio.run(go())


# -- the chain ----------------------------------------------------------------------


def test_the_subnet_pool_price_is_tao_in_over_alpha_in(monkeypatch):
    from reliquary.infrastructure import chain

    class Subtensor:
        async def subnet(self, netuid):
            return SimpleNamespace(alpha_out_emission=1.0, tao_in=2_000.0, alpha_in=400_000.0)

        async def query_subtensor(self, name):
            return SimpleNamespace(value=11796)  # 18 %

    read = asyncio.run(chain.subnet_pool(Subtensor()))
    assert read["alpha_price_tao"] == pytest.approx(0.005)
    assert read["miner_alpha_per_block"] == pytest.approx(0.41, rel=1e-3)


def test_an_unreadable_chain_gives_nulls_never_an_error():
    async def go():
        async def broken():
            raise TimeoutError("finney wedged")

        chain = pay.ChainView(broken, clock=lambda: 0.0)
        assert chain.current() is None
        await chain.settle()
        assert chain.current() is None

    asyncio.run(go())


# -- the board ----------------------------------------------------------------------


class Link:
    """A judge process's stats, as the front reads them."""

    def __init__(self, stats):
        self.stats_doc, self.calls = stats, 0

    async def stats(self, job_id):
        self.calls += 1
        return self.stats_doc


def _board_set(now, *, params, settler, link=None, archives=None, chain=None):
    from tests.unit.test_corpus_job_status import _set

    job_set, router, _ = _set({0: 2, 1: 1}, now=now)
    wiring = job_set.served["job-b"]
    wiring.entry.params = params
    wiring.cap = params["cap"]
    wiring.settler = settler
    if link is not None:
        wiring.judge_link = link
    board = pay.PayBoard(job_set, archives=archives or Archives(),
                         read_chain=chain or _chain(), clock=lambda: now[0], genesis=GENESIS)
    return board, router


def _chain(price=0.005, miner_per_block=0.41):
    async def read():
        return {"alpha_price_tao": price, "miner_alpha_per_block": miner_per_block,
                "netuid": 81}
    return read


def _settler_view(open_periods, oldest):
    return SimpleNamespace(settled_count=4, totals=None, last_period=None,
                           open_periods=open_periods, open_periods_at=1.0,
                           undecided_since=lambda: {"known": True, "at": oldest})


def test_the_board_reports_each_active_job_and_its_current_period():
    async def go():
        now = [at(5, 600)]
        archives = Archives()
        archives.docs[(4, 5)] = _archive(4, 5, 3_000_000, 3_000_000 * PRICE,
                                         budget=3_000_000 * PRICE)
        settler = _settler_view({"4": {"verdicts": 1, "passed": 1, "verified_tokens": 9},
                                 "5": {"verdicts": 5, "passed": 4,
                                       "verified_tokens": 2_000_000}}, at(5, 0))
        params = {"cap": 0.15, "settlement": cp.SETTLEMENT_PERIOD_EMA, **PRICED}
        board, _ = _board_set(now, params=params, settler=settler, archives=archives)
        await board.pay()          # starts the chain and history reads
        await board.settle()
        now[0] += pay.PAY_CACHE_SECONDS
        doc = await board.pay()
        return doc

    doc = asyncio.run(go())
    emission = 0.41 * P / 12
    assert doc["current_period"] == 5 and doc["period_seconds"] == P
    assert doc["alpha_price_tao"] == 0.005
    assert doc["miner_alpha_per_period"] == pytest.approx(emission)
    assert doc["miner_alpha_per_period_source"] == "chain"
    assert "estimate" in doc["note"].lower()
    [job] = doc["jobs"]
    assert job["job_id"] == "job-b" and job["task_id"] == "corpus-b"
    assert job["cap"] == 0.15 and job["max_price_alpha_per_mtoken"] == 0.9
    assert job["tokens_to_saturate"] == pytest.approx(0.15 / PRICE)
    assert job["prompts_total"] == 3 and job["prompts_open"] == 2
    current = job["current_period"]
    assert current["work_period"] == 5 and current["verified_tokens"] == 2_000_000
    assert current["projected_alpha_per_mtoken"] == pytest.approx(
        2_000_000 * PRICE / 2_000_000 * 1e6 * emission)
    assert current["ends_at"] == GENESIS + 6 * P
    assert job["unsettled_periods"] == [{"work_period": 4, "verified_tokens": 9}]
    assert [h["work_period"] for h in job["history"]] == [4]
    assert job["settlement_lag_seconds"] == pytest.approx(600 + pay.PAY_CACHE_SECONDS)
    json.dumps(doc)  # plain JSON, no -inf


def test_the_board_with_an_empty_period_and_no_chain():
    async def go():
        now = [at(5, 600)]

        async def broken():
            raise ConnectionError("no chain")

        params = {"cap": 0.05, "settlement": cp.SETTLEMENT_PERIOD_EMA, **PRICED}
        board, _ = _board_set(now, params=params, settler=_settler_view({}, None),
                              chain=broken)
        await board.pay()
        await board.settle()
        now[0] += pay.PAY_CACHE_SECONDS
        return await board.pay()

    doc = asyncio.run(go())
    assert doc["alpha_price_tao"] is None
    # The emission frozen with the price converts it.
    assert doc["miner_alpha_per_period"] == EMISSION
    assert doc["miner_alpha_per_period_source"] == "price_ceiling"
    [job] = doc["jobs"]
    assert job["current_period"]["verified_tokens"] == 0
    assert job["current_period"]["projected_alpha_per_mtoken"] is None
    assert job["settlement_lag_seconds"] == 0.0 and job["history"] == []


def test_the_board_is_computed_once_for_many_requests():
    async def go():
        now = [at(5, 600)]
        calls = []
        settler = _settler_view({"5": {"verdicts": 1, "passed": 1, "verified_tokens": 10}},
                                None)
        real = settler.undecided_since

        def counted():
            calls.append(1)
            return real()

        settler.undecided_since = counted
        params = {"cap": 0.05, "settlement": cp.SETTLEMENT_PERIOD_EMA}
        board, router = _board_set(now, params=params, settler=settler)
        await board.pay()
        await board.settle()
        now[0] += pay.PAY_CACHE_SECONDS
        calls.clear()
        docs = await asyncio.gather(*(board.pay() for _ in range(50)))
        assert len(calls) == 1 and all(d is docs[0] for d in docs)
        now[0] += pay.PAY_CACHE_SECONDS - 1
        await board.pay()
        assert len(calls) == 1
        now[0] += 2
        await board.pay()
        assert len(calls) == 2

    asyncio.run(go())


def test_a_failed_recompute_serves_the_last_board():
    async def go():
        now = [at(5, 600)]
        settler = _settler_view({}, None)
        params = {"cap": 0.05, "settlement": cp.SETTLEMENT_PERIOD_EMA}
        board, _ = _board_set(now, params=params, settler=settler)
        first = await board.pay()

        async def broken():
            raise RuntimeError("boom")

        board._compute = broken
        now[0] += pay.PAY_CACHE_SECONDS + 1
        assert (await board.pay())["as_of"] == first["as_of"]
        await board.close()

    asyncio.run(go())


def test_retired_and_window_settled_jobs_are_left_out():
    async def go():
        now = [at(5, 600)]
        board, _ = _board_set(now, params={"cap": 0.05},
                              settler=_settler_view({}, None))
        assert (await board.pay())["jobs"] == []
        board2, _ = _board_set(now, params={"cap": 0.05,
                                            "settlement": cp.SETTLEMENT_PERIOD_EMA},
                               settler=_settler_view({}, None))
        board2._job_set._routes.retire("job-b")
        assert (await board2.pay())["jobs"] == []
        await board.close()
        await board2.close()

    asyncio.run(go())


def test_split_mode_reads_the_judges_stats():
    async def go():
        now = [at(5, 600)]
        link = Link({"unsettled": [2, 2, 30], "settled_count": 1, "totals": None,
                     "last_period": None, "open_periods_at": 1.0,
                     "open_periods": {"5": {"verdicts": 2, "passed": 2,
                                            "verified_tokens": 30}},
                     "undecided": {"known": True, "at": at(5, 300)}})
        from reliquary.validator.corpus_validator import JudgedElsewhere

        params = {"cap": 0.05, "settlement": cp.SETTLEMENT_PERIOD_EMA}
        board, _ = _board_set(now, params=params, settler=JudgedElsewhere(None, "job-b"),
                              link=link)
        doc = await board.pay()
        await board.close()
        return doc

    doc = asyncio.run(go())
    [job] = doc["jobs"]
    assert job["current_period"]["verified_tokens"] == 30
    assert job["settlement_lag_seconds"] == pytest.approx(300)


def test_split_mode_with_an_older_judge_reports_unknowns():
    async def go():
        now = [at(5, 600)]
        link = Link({"unsettled": [0, 0, 0], "settled_count": 1, "totals": None})
        from reliquary.validator.corpus_validator import JudgedElsewhere

        params = {"cap": 0.05, "settlement": cp.SETTLEMENT_PERIOD_EMA}
        board, _ = _board_set(now, params=params, settler=JudgedElsewhere(None, "job-b"),
                              link=link)
        doc = await board.pay()
        await board.close()
        return doc

    [job] = asyncio.run(go())["jobs"]
    assert job["current_period"]["verified_tokens"] is None
    assert job["settlement_lag_seconds"] is None


def test_the_judge_stats_carry_the_open_periods_and_the_undecided():
    from fastapi.testclient import TestClient

    from reliquary.validator.corpus_judge_process import build_judge_app

    settler = _settler_view({"5": {"verdicts": 1, "passed": 1, "verified_tokens": 3}}, 12.0)
    w = SimpleNamespace(stats=SimpleNamespace(unsettled=lambda: (1, 1, 3)), settler=settler)
    body = TestClient(build_judge_app({"job-b": w}, feed=None)).get("/jobs/job-b/stats").json()
    assert body["open_periods"] == {"5": {"verdicts": 1, "passed": 1, "verified_tokens": 3}}
    assert body["undecided"] == {"known": True, "at": 12.0}


# -- the route ----------------------------------------------------------------------


def _app():
    from reliquary.validator.corpus_validator import build_corpus_jobs_app

    return build_corpus_jobs_app(jobs=[], store=None, records=None, tokenizer=None,
                                 verify_signature=None, proof_chunk_tokens=32)


def test_the_route_serves_the_board_and_503s_before_the_job_set():
    from fastapi.testclient import TestClient

    app = _app()
    client = TestClient(app)
    assert client.get("/corpus/pay").status_code == 503

    class Board:
        async def pay(self):
            return {"jobs": [], "note": "estimate"}

        async def close(self):
            return None

    app.state.corpus_jobs = object()
    app.state.corpus_pay = Board()
    assert client.get("/corpus/pay").json() == {"jobs": [], "note": "estimate"}

    class Broken(Board):
        async def pay(self):
            raise RuntimeError("down")

    app.state.corpus_pay = Broken()
    assert client.get("/corpus/pay").status_code == 503
