"""The auditor's two seams for running in a judge process: a scorer (the GPU
process) instead of a local model, and a gate that holds unaudited passes
while the arrival feed from the front cannot be trusted complete."""

from __future__ import annotations

import asyncio

from reliquary.corpus.audit_policy import AuditParams, MinerState
from reliquary.validator.corpus_auditor import CorpusAuditor
from reliquary.validator.corpus_miner_states import MinerStates
from tests.unit import corpus_judge_sim as sim
from tests.unit import corpus_split_fakes as fakes


def _auditor(*, scorer=None, model=None, vocab_size=None, **kw):
    return CorpusAuditor(job_id="math-v1", records=None, model=model,
                         tokenizer=fakes.Tokenizer(), proof=fakes.PROOF, scorer=scorer,
                         vocab_size=vocab_size, **kw)


def test_a_scorer_judges_as_the_local_forward_does(monkeypatch):
    from reliquary.validator import corpus_auditor

    monkeypatch.setattr(corpus_auditor, "score_sequences", fakes.score_sequences)
    records = [fakes.record("5A", 1.0), fakes.record("5B", 2.0, forged=True),
               {**fakes.record("5C", 3.0), "completions": []},
               fakes.record("5D", 4.0, tokens=(5, fakes.VOCAB + 1))]
    local = _auditor(model=fakes.Model())._judge_many([dict(r) for r in records])

    seen = []

    async def scorer(rows):
        seen.append(rows)
        return fakes.score_rows(rows), 0.5, 0.25

    remote = asyncio.run(_auditor(scorer=scorer, vocab_size=fakes.VOCAB)._forward(records))
    assert remote == local
    assert [r["passed"] for r in remote] == [True, False, False, False]
    assert remote[2]["reason"] == "no_completions"
    assert "scored_by" not in remote[0]
    # Only the records that need the GPU cross: prompt ids then completion.
    assert len(seen) == 1 and len(seen[0]) == 2
    tokens, prompt_len, proofs = seen[0][0]
    assert prompt_len == 1 and tokens[1:] == [5, 6, 7, 8] and proofs == ["A" * 8]


def test_a_failure_is_re_audited_through_the_scorer_too():
    calls = []

    async def scorer(rows):
        calls.append(len(rows))
        return fakes.score_rows(rows), 0.0, 0.0

    auditor = _auditor(scorer=scorer, vocab_size=fakes.VOCAB)
    outcomes = asyncio.run(auditor._audit_outcomes([fakes.record("5B", 1.0, forged=True)],
                                                   local=True))
    assert outcomes[0]["passed"] is False and calls == [1]


def test_a_scorer_error_is_a_validator_side_error_retried_one_by_one():
    calls = []

    async def scorer(rows):
        calls.append(len(rows))
        if len(rows) > 1:
            raise RuntimeError("CUDA out of memory")
        return fakes.score_rows(rows), 0.0, 0.0

    auditor = _auditor(scorer=scorer, vocab_size=fakes.VOCAB)
    outcomes = asyncio.run(auditor._audit_outcomes([fakes.record("5A", 1.0),
                                                    fakes.record("5B", 2.0)]))
    assert [o["passed"] for o in outcomes] == [True, True]
    assert calls == [2, 1, 1]


def test_nothing_to_score_never_calls_the_scorer():
    async def scorer(rows):
        raise AssertionError("called")

    auditor = _auditor(scorer=scorer, vocab_size=fakes.VOCAB)
    out = asyncio.run(auditor._forward([{**fakes.record("5C", 1.0), "completions": []}]))
    assert out[0]["reason"] == "no_completions"


class _Store(sim.Store):
    async def read_settlement(self, job_id):
        return {}, None


def _gated_run(complete):
    """Two hotkeys well past probation, q=0.5, one beacon: some records are
    drawn (audited), the rest pass unaudited once their hold is over --
    unless the feed is incomplete."""
    loop = sim.VirtualTimeLoop()

    async def scenario():
        store = _Store(latency=(0.0, 0.0))
        clock = sim._Clock()
        for hotkey in ("5A", "5B"):
            store.miners[hotkey] = MinerState(audited_passed=50).to_dict()
        params = AuditParams(q=0.5, probation_submissions=5, hold_seconds=30.0,
                             ban_after_failures=1000)
        auditor = CorpusAuditor(
            job_id="math-v1", records=store, model=None, tokenizer=None, proof=None,
            params=params, miner_states=MinerStates(store, "math-v1", clock=clock),
            beacon=sim.Beacon(latency=0.0), round_at=sim.round_at, clock=clock,
            accept_slack_seconds=5.0, rescan_every_seconds=10.0,
            arrivals_complete=lambda: complete["now"])

        async def forward(records, *, local=False):
            return [{"passed": True, "reason": None, "worst_exp": 0,
                     "worst_mant_mean": 0.01, "worst_mant_median": 0.01} for _ in records]

        auditor._forward = forward
        async def arrive():
            # Steady traffic, so neither hotkey falls under the slow-hotkey rule.
            for i in range(10_000):
                sid = "%064x" % (i + 1)
                store.submissions[sid] = {"hotkey": ("5A", "5B")[i % 2],
                                          "received_at": sim.virtual_clock(),
                                          "token_count": 10, "completions": [{"tokens": [1]}]}
                auditor.enqueue(sid)
                await asyncio.sleep(0.5)

        feeder = asyncio.ensure_future(arrive())
        runner = asyncio.ensure_future(auditor.run())
        await asyncio.sleep(120.0)
        before = dict(store.verdicts)
        complete["now"] = True
        await asyncio.sleep(120.0)
        feeder.cancel()
        runner.cancel()
        return before, dict(store.verdicts)

    try:
        return loop.run_until_complete(scenario())
    finally:
        loop.close()


def test_an_incomplete_feed_holds_unaudited_passes_but_not_audits():
    before, after = _gated_run({"now": False})
    assert len(before) > 50 and all(v["audited"] for v in before.values())
    # Once complete, the held records pass at once (the next rescan).
    assert sum(not v["audited"] for v in after.values()) > 100


def test_a_complete_feed_changes_nothing():
    before, _ = _gated_run({"now": True})
    assert sum(not v["audited"] for v in before.values()) > 50
