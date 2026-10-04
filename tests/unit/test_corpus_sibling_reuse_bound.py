"""Review B2: a sibling is paid in the pass that decides it only when every
record inside its own hold was decided in that pass too. When the oldest
payable record's window holds more than PASS_SIBLINGS siblings, only the
first ones are decided: the horizon is the first sibling left undecided."""

from __future__ import annotations

import asyncio

from reliquary.corpus.audit_policy import AuditParams, MinerState, drawn
from reliquary.validator import corpus_auditor
from reliquary.validator.corpus_miner_states import MinerStates
from tests.unit import test_corpus_judge_equivalence as equivalence

RANDOMNESS = "5a" * 32
Q, HOLD = 0.5, 100.0
NOW = 10_000.0


def _ids(want_drawn: bool, n: int, start: int = 0) -> list[str]:
    out, i = [], start
    while len(out) < n:
        sid = "%064x" % (i + 1)
        if drawn(RANDOMNESS, sid, Q) == want_drawn:
            out.append(sid)
        i += 1
    return out


def _round_at(t):
    return int(t // 3) + 1


def test_a_sibling_whose_hold_reaches_past_the_cut_is_not_paid(monkeypatch):
    monkeypatch.setattr(corpus_auditor, "PASS_SIBLINGS", 3)
    undrawn = _ids(False, 6)
    (cheat,) = _ids(True, 1, start=1000)
    s_old, x, a, b, fresh1, fresh2 = undrawn
    store = equivalence._Store()
    store.miners["5HkH"] = MinerState(audited_passed=50).to_dict()

    def rec(t, tokens=10):
        return {"hotkey": "5HkH", "received_at": t, "token_count": tokens,
                "completions": [{"tokens": [1]}]}

    # S oldest; X the pass's payable record; A, B, then a drawn D beyond the
    # cut of 3 but inside S's hold; two fresh records keep the hotkey sampled.
    store.submissions.update({s_old: rec(0.0), x: rec(1.0), a: rec(2.0), b: rec(3.0),
                              cheat: rec(4.0), fresh1: rec(NOW - 10), fresh2: rec(NOW - 5)})
    clock = lambda: NOW  # noqa: E731
    auditor = corpus_auditor.CorpusAuditor(
        job_id="math-v1", records=store, model=None, tokenizer=None, proof=None,
        params=AuditParams(q=Q, probation_submissions=5, hold_seconds=HOLD,
                           ban_after_failures=1000),
        miner_states=MinerStates(store, "math-v1", clock=clock),
        beacon=lambda r: RANDOMNESS, round_at=_round_at, clock=clock,
        accept_slack_seconds=5.0, prefetch_rounds=False)

    async def forward(records, *, local=False):
        return [dict(equivalence._BAD if r["received_at"] == 4.0 else equivalence._OK)
                for r in records]

    auditor._forward = forward

    async def scenario():
        await auditor._seed(await auditor.pending_ids())
        await auditor.judge_many([x])

    asyncio.run(scenario())
    # D (drawn, beyond the cut) is not audited yet: nothing within its reach is paid.
    assert cheat not in store.verdicts
    assert s_old not in store.verdicts and x not in store.verdicts, store.verdicts.keys()
