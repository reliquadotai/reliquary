"""An edit is several tokens each far from the draw; honest drift is near-ties, or one far token or a cut end
spread over every hotkey. Only the edit (or a sample far off the draw) is charged."""

from __future__ import annotations

import math

import pytest

from reliquary.corpus import eval_draw
from reliquary.corpus.draw_audit import diagnostics_agree, diagnostics_from_wire, diagnostics_to_wire
from reliquary.environment.forced_sampling import SeedConsistencyDiagnostics as D


def _miss(gap):
    return 1 - math.exp(-gap)


BASE = dict(n_positions=1000, n_stochastic=500, n_drawn_exact_match=997)


def test_two_far_tokens_before_the_end_are_an_edit():
    ok, detail = eval_draw.judge([D(**BASE, max_cdf_miss=_miss(9.0), n_far_miss=2)])
    assert not ok and detail["penalty"] is True and detail["far_misses"] == 2


@pytest.mark.parametrize("d", [
    D(**BASE, max_cdf_miss=_miss(12.0), n_far_miss=1),                           # one far token
    D(**BASE, max_cdf_miss=_miss(9.0), last_miss=_miss(9.0), n_far_miss=2),      # one far token and a cut end
    D(**BASE, max_cdf_miss=_miss(12.0)),                                         # an executor without the count
])
def test_one_far_token_or_a_cut_end_is_refused_uncharged(d):
    ok, detail = eval_draw.judge([d])
    assert not ok and detail["penalty"] is False


def test_the_far_count_crosses_the_wire_and_an_older_executor_still_reads():
    d = D(**BASE, max_cdf_miss=_miss(9.0), last_miss=0.0, n_far_miss=3)
    wire = diagnostics_to_wire(d)
    assert diagnostics_from_wire(wire).n_far_miss == 3
    older = {k: v for k, v in wire.items() if k != "far_misses"}
    assert diagnostics_from_wire(older).n_far_miss == 0
    assert diagnostics_agree(wire, older) and not diagnostics_agree(wire, {**wire, "far_misses": 0})


@pytest.mark.parametrize("far", [-1, 1001, True, 2.0, "2"])
def test_a_malformed_far_count_is_refused(far):
    wire = {**diagnostics_to_wire(D(**BASE)), "far_misses": far}
    with pytest.raises(ValueError):
        diagnostics_from_wire(wire)


def test_a_hotkey_refused_far_more_often_than_the_others_is_charged():
    from reliquary.validator.corpus_auditor import frequent_refusals

    # The order's honest rate (~0.5 % at T 0.6, 2026-11-07): a few refusals stay in it.
    assert not frequent_refusals(5, 300, 105, 19_000)
    assert not frequent_refusals(2, 300, 0, 19_000)
    # One edit at a time, every tenth sample.
    assert frequent_refusals(30, 300, 105, 19_000)
    # Nothing to compare with yet: nothing charged.
    assert not frequent_refusals(10, 20, 0, 0)


def test_the_auditor_charges_frequent_refusals_once_per_submission():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    auditor = CorpusAuditor.__new__(CorpusAuditor)
    auditor._draw_job, auditor._job_id = object(), "order-gen-x"
    auditor._draw_counts, auditor._draw_totals, auditor._draw_counted = {}, [0, 0], set()
    ok = {"passed": True}
    drift = {"passed": False, "reason": "draw_mismatch", "penalty": False}
    for i in range(2000):
        auditor._count_refusal(f"h{i}", f"5Honest{i % 50}", dict(ok))
    charged = [auditor._count_refusal(f"c{i}", "5Edits", dict(drift if i % 5 == 0 else ok))
               for i in range(100)]
    refusals = [o for o in charged if not o["passed"]]
    assert refusals[0]["penalty"] is False and refusals[-1]["penalty"] is True
    assert refusals[-1]["charged_for"] == "frequent_refusals"
    # A re-audit of a counted submission counts nothing again.
    assert auditor._count_refusal("c0", "5Edits", dict(drift))["penalty"] is False
    assert auditor._draw_counts["5Edits"] == [100, 20]
