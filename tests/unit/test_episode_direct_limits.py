"""`episode.limit_cut_fails`: an eval episode that states its limits directly (turns, context, the
sandbox's call, CPU and wall budgets) instead of naming a profile (design 2026-11-14). It reaches
the wire's ceilings and is scored as a profile is: a model limit scores 0, an infra end is redone."""

from __future__ import annotations

import pytest

from reliquary.corpus.job import (
    EPISODE_TOKENS_CEILING, EPISODE_TURNS_CEILING, JobError, episode_contract, limit_cuts_fail,
    model_closed, parse_job, scores_model_closes,
)
from reliquary.eval.job_grading import episode_protocol
from tests.unit.test_corpus_job_episode import _manifest
from tests.unit.test_corpus_job_signed_sandbox import sandbox_spec
from tests.unit.test_signed_bridged_eval import ENV_PACKAGE, PACKAGE, terminal_episode

EVAL_SOURCE = "eval-set:tmax:3:" + "e" * 64


def _direct(turns=250, tokens=131072, **budgets):
    budgets = {**sandbox_spec()["budgets"], "max_calls": 512, "per_call_timeout_s": 600,
               "wall_s": 14400, "cpu_s": 14400, **budgets}
    return terminal_episode(limit_cut_fails=True, max_turns=turns, max_total_tokens=tokens,
                            max_tokens_per_turn=16384,
                            sandbox=sandbox_spec(env=PACKAGE, env_package=ENV_PACKAGE, tools=["bash"],
                                                 budgets=budgets))


def _job(episode, prompt_source=EVAL_SOURCE):
    return parse_job(_manifest(prompt_source=prompt_source, episode=episode,
                               sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                                         "max_new_tokens": episode["max_tokens_per_turn"], "n": 1}))


def test_stated_limits_reach_the_wire_ceilings_and_round_trip():
    assert (EPISODE_TURNS_CEILING, EPISODE_TOKENS_CEILING) == (250, 131072)
    job = _job(_direct())
    assert (job.episode.max_turns, job.episode.max_total_tokens, job.episode.profile) == (250, 131072, None)
    assert job.episode.limit_cut_fails and limit_cuts_fail(job.episode)
    assert job.to_contract()["episode"]["limit_cut_fails"] is True
    assert parse_job(job.to_contract()) == job


def test_any_limits_within_the_ceilings_are_taken():
    job = _job(_direct(turns=100, tokens=65536, max_calls=100, wall_s=3600, cpu_s=3600))
    assert (job.episode.max_turns, job.episode.max_total_tokens) == (100, 65536)


def test_without_the_flag_nothing_changes():
    plain = terminal_episode()
    job = _job(plain)
    assert not job.episode.limit_cut_fails and not limit_cuts_fail(job.episode)
    assert "limit_cut_fails" not in job.to_contract()["episode"]
    with pytest.raises(JobError, match=r"max_turns must be a whole number in \[1, 64\]"):
        _job({**plain, "max_turns": 100})


@pytest.mark.parametrize("change,match", [
    ({"limit_cut_fails": False}, "written only as true"),
    ({"max_turns": 251}, r"max_turns must be a whole number in \[1, 250\]"),
    ({"max_total_tokens": 131073}, r"\[1, 131072\]"),
])
def test_the_flag_is_bounded(change, match):
    with pytest.raises(JobError, match=match):
        _job({**_direct(), **change})


def test_a_profile_already_scores_limit_cuts():
    from tests.unit.test_episode_profile import _leaderboard

    with pytest.raises(JobError, match="already scores limit cuts"):
        _job({**_leaderboard(), "limit_cut_fails": True})


@pytest.mark.parametrize("budgets,match", [
    ({"max_calls": 249}, "call budget never cuts before the 250-turn cap"),
    ({"cpu_s": 3600}, "cpu_s must be at least wall_s"),
    ({"wall_s": 14401}, r"wall_s must be a whole number in \[1, 14400\]"),
])
def test_the_budgets_let_the_model_limits_cut_first(budgets, match):
    with pytest.raises(JobError, match=match):
        _job(_direct(**budgets))


def test_the_flag_is_for_eval_jobs_only():
    from reliquary.corpus.job import _parse_episode

    with pytest.raises(JobError, match="eval jobs only"):
        _parse_episode({**_direct(), "env": {**_direct()["env"], "split": "train"}}, eval_job=False)
    assert episode_contract(_direct(), eval_job=True)["limit_cut_fails"] is True


def test_stated_limits_score_model_cuts_and_redo_infra_ends():
    episode = _job(_direct()).episode
    assert scores_model_closes(episode)
    assert model_closed(episode, "budget_exhausted", "max_calls")
    assert not model_closed(episode, "budget_exhausted", "cpu_s")
    assert not model_closed(episode, "expired", "deadline passed")


def test_the_report_records_the_limits_played():
    protocol = episode_protocol(_job(_direct(turns=100, tokens=65536, max_calls=200)).episode)
    assert protocol["max_turns"] == 100 and protocol["max_total_tokens"] == 65536
    assert protocol["limit_cut_fails"] is True and "profile" not in protocol
    assert (protocol["per_call_timeout_s"], protocol["max_calls"], protocol["wall_s"]) == (600, 200, 14400)
    plain = episode_protocol(_job(terminal_episode()).episode)
    assert "limit_cut_fails" not in plain and "wall_s" not in plain


def test_a_limit_cut_scores_zero_under_stated_limits(tmp_path, monkeypatch):
    from tests.unit.test_eval_episode_grading import _GradeRecords, _grade_job, _publish, _report, _sid
    from tests.unit.test_episode_profile import _leaderboard

    plain, rows, subnet = _publish(tmp_path, monkeypatch, "swe")
    episode = {k: v for k, v in _leaderboard().items() if k != "profile"}
    episode.update(limit_cut_fails=True, harness_spec=None, max_turns=120)
    episode = {k: v for k, v in episode.items() if v is not None}
    episode["env"]["split"] = plain.episode.env.split
    job = parse_job(_manifest(prompt_source=plain.prompt_source, prompt_count=plain.prompt_count,
                              episode=episode,
                              sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                                        "max_new_tokens": episode["max_tokens_per_turn"], "n": 1}))
    records = _GradeRecords(job, submissions_readable=False)
    for (prompt, k), stop in {(0, 0): "agent_completed", (0, 1): "max_turns", (1, 0): "context_length",
                              (1, 1): "agent_completed"}.items():
        records.submit(_sid(prompt, k), prompt, reward=1.0)
        records.grades[_sid(prompt, k)]["stop"] = stop
    _grade_job(tmp_path, job, records, subnet)
    report = _report(tmp_path)
    env = report["envs"]["reliquary-swe"]
    assert env["pass@1"]["value"] == pytest.approx(0.5)
    assert env["limit_cuts"]["by_stop"] == {"max_turns": 1, "context_length": 1}
    assert report["provenance"]["episode"]["limit_cut_fails"] is True
    assert report["provenance"]["episode"]["max_turns"] == 120
