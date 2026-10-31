"""A forced-draw job's judge thresholds are job parameters (``judge`` in the manifest), with today's values as
defaults: a job that does not set them stores and hashes exactly as before."""
from __future__ import annotations

import math

import pytest

from reliquary.corpus import eval_draw
from reliquary.corpus.job import DEFAULT_DRAW_JUDGE, DrawJudge, JobError, parse_job
from tests.unit.test_eval_draw import _diag, _raw


def test_a_job_without_a_judge_uses_the_defaults_and_writes_nothing():
    job = parse_job(_raw())
    assert job.judge == DEFAULT_DRAW_JUDGE
    assert "judge" not in job.to_contract()
    assert (DEFAULT_DRAW_JUDGE.floor_rate, DEFAULT_DRAW_JUDGE.grace, DEFAULT_DRAW_JUDGE.gap_tolerance,
            DEFAULT_DRAW_JUDGE.terminal_gap) == (eval_draw.EVAL_MATCH_FLOOR, eval_draw.EVAL_MATCH_GRACE,
                                                 eval_draw.EVAL_GAP_TOLERANCE, eval_draw.EVAL_TERMINAL_GAP)


def test_a_judge_round_trips_through_the_contract():
    judge = {"floor_rate": 0.95, "grace": 4, "gap_tolerance": 5.0, "terminal_gap": 2.0}
    job = parse_job(_raw(judge=judge))
    assert job.judge == DrawJudge(0.95, 4, 5.0, 2.0)
    assert job.to_contract()["judge"] == judge
    assert parse_job(job.to_contract()) == job


def test_a_judge_equal_to_the_defaults_hashes_like_no_judge():
    defaults = {"floor_rate": 0.98, "grace": 8, "gap_tolerance": 4.0, "terminal_gap": 3.0}
    assert parse_job(_raw(judge=defaults)).to_contract() == parse_job(_raw()).to_contract()


@pytest.mark.parametrize("judge", [
    {"floor_rate": 0.95, "grace": 4, "gap_tolerance": 5.0},                                    # a field missing
    {"floor_rate": 0.95, "grace": 4, "gap_tolerance": 5.0, "terminal_gap": 2.0, "extra": 1},     # unknown field
    {"floor_rate": 0.4, "grace": 4, "gap_tolerance": 5.0, "terminal_gap": 2.0},                  # floor too low
    {"floor_rate": 1.01, "grace": 4, "gap_tolerance": 5.0, "terminal_gap": 2.0},
    {"floor_rate": 0.95, "grace": -1, "gap_tolerance": 5.0, "terminal_gap": 2.0},
    {"floor_rate": 0.95, "grace": 65, "gap_tolerance": 5.0, "terminal_gap": 2.0},
    {"floor_rate": 0.95, "grace": True, "gap_tolerance": 5.0, "terminal_gap": 2.0},
    {"floor_rate": 0.95, "grace": 4, "gap_tolerance": 0.0, "terminal_gap": 2.0},
    {"floor_rate": 0.95, "grace": 4, "gap_tolerance": 5.0, "terminal_gap": math.inf},
    "loose",
])
def test_a_bad_judge_is_refused(judge):
    with pytest.raises(JobError):
        parse_job(_raw(judge=judge))


def test_a_judge_needs_a_draw():
    raw = _raw(judge={"floor_rate": 0.95, "grace": 4, "gap_tolerance": 5.0, "terminal_gap": 2.0})
    raw["sampling"] = {**raw["sampling"]}
    del raw["sampling"]["draw"]
    with pytest.raises(JobError):
        parse_job(raw)


def test_the_judge_reads_the_jobs_thresholds():
    """3 % of drawn positions missed: refused at the default 2 %, accepted at a 0.95 floor (an fp8 tier)."""
    d = _diag(n_positions=1000, n_stochastic=500, n_drawn_exact_match=970)
    assert not eval_draw.judge([d])[0]
    assert not eval_draw.judge([d], DEFAULT_DRAW_JUDGE)[0]
    assert eval_draw.judge([d], DrawJudge(0.95, 8, 4.0, 3.0))[0]
    edit = _diag(n_positions=1000, n_stochastic=500, n_drawn_exact_match=999, max_cdf_miss=1 - math.exp(-5.0))
    assert not eval_draw.judge([edit])[0]
    assert eval_draw.judge([edit], DrawJudge(0.98, 8, 6.0, 3.0))[0]


def test_the_auditor_judges_with_the_jobs_thresholds():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    d = _diag(n_positions=1000, n_stochastic=500, n_drawn_exact_match=970)

    def verdict(job):
        auditor = CorpusAuditor.__new__(CorpusAuditor)
        auditor._draw_job = job
        results = [{"passed": True, "reason": None}]
        auditor._judge_draws(results, [(0, 0)], {0: d})
        return results[0]["passed"]

    assert not verdict(parse_job(_raw()))
    assert verdict(parse_job(_raw(judge={"floor_rate": 0.95, "grace": 8, "gap_tolerance": 4.0,
                                         "terminal_gap": 3.0})))


def test_a_partial_judge_fills_in_the_defaults():
    from reliquary.corpus.job import draw_judge_from

    assert draw_judge_from({"floor_rate": 0.95}) == DrawJudge(0.95, 8, 4.0, 3.0)
    assert draw_judge_from({}) == DEFAULT_DRAW_JUDGE
    with pytest.raises(JobError):
        draw_judge_from({"floor_rate": 0.95, "unknown": 1})
    with pytest.raises(JobError):
        draw_judge_from({"floor_rate": 0.1})


def test_a_suite_set_passes_its_judge_to_the_job(tmp_path):
    import json

    from reliquary.eval.rl_schedule import load_suite
    from tests.unit.test_eval_rl_schedule import World, scheduler

    path = tmp_path / "suite.json"
    path.write_text(json.dumps([
        {"set_id": "s", "short": "aime", "slots_per_prompt": 8, "cap": 0.02, "max_new_tokens": 32768},
        {"set_id": "g", "short": "gpqa", "slots_per_prompt": 8, "cap": 0.02, "max_new_tokens": 32768,
         "judge": {"floor_rate": 0.95}},
    ]))
    plain, strict = load_suite(path)
    world = World()
    scheduler(tmp_path, world, suite=[plain, strict]).tick(execute=True)
    first, second = world.commands
    assert "--draw-judge" not in first
    assert json.loads(second[second.index("--draw-judge") + 1]) == {"floor_rate": 0.95}


def test_a_manifest_carries_a_judge_only_when_it_differs_from_the_defaults():
    from reliquary.corpus.job import judge_field

    assert judge_field({"floor_rate": 0.95}) == {"floor_rate": 0.95, "grace": 8, "gap_tolerance": 4.0,
                                                 "terminal_gap": 3.0}
    assert judge_field({"floor_rate": 0.98}) is None
    assert judge_field({}) is None
