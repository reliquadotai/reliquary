"""`episode.profile`: a named eval run configuration (the SWE-bench bash-only leaderboard's)
that lifts the turn and context ceilings for one harness, and how its limit cuts are scored."""

from __future__ import annotations

import pytest

from reliquary.corpus.job import (
    EPISODE_PROFILES, MAX_EPISODE_TOTAL_TOKENS, MAX_EPISODE_TURNS, SWEBENCH_BASH_ONLY, JobError,
    episode_contract, parse_job,
)
from tests.unit.test_corpus_job_episode import EVAL_SOURCE, _manifest, _signed
from tests.unit.test_corpus_job_signed_sandbox import sandbox_spec

MINI_SWE_2 = {"id": "mini-swe", "version": "2",
              "digest": "cecffb2631c01e2dd415a4e9cb76d79717d1a0b67bd0187422a658ef9173f15d"}


def _leaderboard(**overrides):
    budgets = {**sandbox_spec()["budgets"], "max_calls": 512, "per_call_timeout_s": 60, "wall_s": 14400,
               "cpu_s": 14400}
    fields = {"profile": SWEBENCH_BASH_ONLY, "harness_spec": MINI_SWE_2, "max_turns": 250,
              "max_tokens_per_turn": 16384, "max_total_tokens": 131072,
              "sandbox": sandbox_spec(budgets=budgets)}
    fields.update(overrides)
    return _signed(env={"split": "eval", "num_images": 500}, **fields)


def _job(episode, **overrides):
    overrides.setdefault("prompt_source", EVAL_SOURCE)
    return parse_job(_manifest(episode=episode,
                               sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                                         "min_new_tokens": 2,
                                         "max_new_tokens": episode["max_tokens_per_turn"], "n": 1},
                               **overrides))


def test_the_leaderboard_profile_lifts_the_ceilings_and_round_trips():
    job = _job(_leaderboard())
    assert (job.episode.profile, job.episode.max_turns, job.episode.max_total_tokens) == (
        SWEBENCH_BASH_ONLY, 250, 131072)
    assert job.to_contract()["episode"]["profile"] == SWEBENCH_BASH_ONLY
    assert parse_job(job.to_contract()) == job


def test_without_a_profile_the_default_ceilings_hold_and_nothing_is_written():
    assert (MAX_EPISODE_TURNS, MAX_EPISODE_TOTAL_TOKENS) == (64, 60000)
    plain = {k: v for k, v in _leaderboard().items() if k != "profile"}
    # A CPU budget past the gateways' default cap is the profile's only.
    with pytest.raises(JobError, match=r"cpu_s must be a whole number in \[1, 3600\]"):
        _job({**plain, "max_turns": 48, "max_total_tokens": 60000, "max_tokens_per_turn": 8192})
    plain["sandbox"] = {**plain["sandbox"], "budgets": {**plain["sandbox"]["budgets"], "cpu_s": 3600}}
    with pytest.raises(JobError, match=r"max_total_tokens must be a whole number in \[1, 60000\]"):
        _job(plain)
    with pytest.raises(JobError, match=r"max_turns must be a whole number in \[1, 64\]"):
        _job({**plain, "max_total_tokens": 60000, "max_tokens_per_turn": 8192})
    contract = episode_contract({**plain, "max_turns": 48, "max_total_tokens": 60000,
                                 "max_tokens_per_turn": 8192}, eval_job=True)
    assert "profile" not in contract


@pytest.mark.parametrize("episode,match", [
    (_leaderboard(profile="swebench-full"), "profile must be one of"),
    (_leaderboard(max_turns=251), r"max_turns must be a whole number in \[1, 250\]"),
    (_leaderboard(max_total_tokens=131073), r"\[1, 131072\]"),
    ({k: v for k, v in _leaderboard().items() if k != "harness_spec"}, "plays harness mini-swe@2"),
    (_leaderboard(harness_spec={**MINI_SWE_2, "id": "claude-code", "version": "1"}),
     "plays harness mini-swe@2"),
    (_leaderboard(sandbox=sandbox_spec(budgets={**sandbox_spec()["budgets"], "max_calls": 512})),
     "per_call_timeout_s must be 60"),
    (_leaderboard(sandbox=sandbox_spec(budgets={**sandbox_spec()["budgets"], "max_calls": 512,
                                                "per_call_timeout_s": 60, "wall_s": 7200})),
     "wall_s must be at least 14400"),
])
def test_a_profile_pins_its_harness_and_bounds(episode, match):
    with pytest.raises(JobError, match=match):
        _job(episode)


def test_a_profile_is_for_eval_jobs_only():
    with pytest.raises(JobError, match="eval jobs only"):
        episode = _leaderboard()
        episode_contract({**episode, "env": {**episode["env"], "split": "train"}}, eval_job=False)


def test_the_profile_matches_the_reference_configuration():
    profile = EPISODE_PROFILES[SWEBENCH_BASH_ONLY]
    # mini-swe-agent config/benchmarks/swebench.yaml: step_limit 250, environment.timeout 60.
    assert (profile.harness, profile.max_turns, profile.per_call_timeout_s) == (("mini-swe", "2"), 250, 60)
    assert profile.limit_cut_fails and profile.package == "reliquary-swe"


def test_a_limit_cut_scores_zero_and_is_counted_apart(tmp_path, monkeypatch):
    import json

    from tests.unit.test_eval_episode_grading import (
        _GradeRecords, _grade_job, _publish, _report, _sid,
    )

    plain, rows, subnet = _publish(tmp_path, monkeypatch, "swe")
    episode = _leaderboard()
    episode["env"]["split"] = plain.episode.env.split
    job = _job(episode, prompt_source=plain.prompt_source, prompt_count=plain.prompt_count)
    records = _GradeRecords(job, submissions_readable=False)
    stops = {(0, 0): "agent_completed", (0, 1): "max_turns", (1, 0): "context_length",
             (1, 1): "agent_completed"}
    for (prompt, k), stop in stops.items():
        records.submit(_sid(prompt, k), prompt, reward=1.0)
        records.grades[_sid(prompt, k)]["stop"] = stop
    _grade_job(tmp_path, job, records, subnet)
    report = _report(tmp_path)
    env = report["envs"]["reliquary-swe"]
    # Both cut episodes left a passing box, and still fail, as the leaderboard counts them.
    assert env["pass@1"]["value"] == pytest.approx(0.5)
    assert env["limit_cuts"] == {"rows": 2, "by_stop": {"max_turns": 1, "context_length": 1},
                                 "box_state_passed": 2}
    assert env["stop_reasons"] == {"agent_completed": 2, "max_turns": 1, "context_length": 1}
    assert report["provenance"]["episode"]["profile"] == SWEBENCH_BASH_ONLY
    assert report["provenance"]["episode"]["per_call_timeout_s"] == 60
    import pyarrow.parquet as pq

    cut = [json.loads(r["grader_detail"]) for r in pq.read_table(tmp_path / "out" / "graded.parquet").to_pylist()
           if r["score"] == 0.0]
    assert [d["limit_cut"] for d in cut] == ["max_turns", "context_length"]
    assert all(d["box_reward"] == 1.0 and d["graded_success"] is False for d in cut)


def test_a_whole_set_with_exclusions_is_restated_over_the_benchmark():
    from reliquary.eval.grading import _excluded_bracket

    card = {"count": 493, "selection": {"exclude": {f"t-{i}": "needs the internet" for i in range(7)}}}
    bracket = _excluded_bracket(card, 493, {"pass@1": {"value": 0.5}})
    assert bracket["benchmark_tasks"] == 500 and len(bracket["excluded"]) == 7
    assert bracket["pass@1_excluded_failed"] == pytest.approx(246.5 / 500)
    assert bracket["pass@1_excluded_passed"] == pytest.approx(253.5 / 500)
    assert _excluded_bracket(card, 100, {"pass@1": {"value": 0.5}}) is None
    assert _excluded_bracket({"count": 10}, 10, {"pass@1": {"value": 0.5}}) is None


def test_the_call_budget_never_cuts_before_the_step_cap():
    budgets = {**sandbox_spec()["budgets"], "max_calls": 249, "per_call_timeout_s": 60, "wall_s": 14400,
               "cpu_s": 14400}
    with pytest.raises(JobError, match="max_calls"):
        _job(_leaderboard(sandbox=sandbox_spec(budgets=budgets)))
    with pytest.raises(JobError, match="cpu_s must be at least wall_s"):
        _job(_leaderboard(sandbox=sandbox_spec(budgets={**budgets, "max_calls": 512, "cpu_s": 3600})))


def test_model_budgets_close_as_a_scored_zero_and_infra_ends_do_not():
    from reliquary.corpus.job import model_closed, scores_model_closes

    profiled = _job(_leaderboard()).episode
    plain = _job(_signed(env={"split": "eval", "num_images": 500})).episode
    assert scores_model_closes(profiled) and not scores_model_closes(plain)
    for reason in ("max_calls", "transcript_bytes"):
        assert model_closed(profiled, "budget_exhausted", reason)
        assert not model_closed(plain, "budget_exhausted", reason)
    # CPU time depends on the gateway's hardware: infra, redone.
    assert not model_closed(profiled, "budget_exhausted", "cpu_s")
    # Wall time is the gateway's and the miner's speed, never the model's: redone, not scored.
    assert not model_closed(profiled, "expired", "deadline passed")
    assert not model_closed(profiled, "box_failed", None)
    assert not model_closed(profiled, "graded", None)


def test_a_budget_close_scores_zero_and_redone_slots_are_counted(tmp_path, monkeypatch):
    from reliquary.eval.grading import redone_slots
    from tests.unit.test_eval_episode_grading import (
        _GradeRecords, _grade_job, _publish, _report, _sid,
    )

    plain, rows, subnet = _publish(tmp_path, monkeypatch, "swe")
    episode = _leaderboard()
    episode["env"]["split"] = plain.episode.env.split
    job = _job(episode, prompt_source=plain.prompt_source, prompt_count=plain.prompt_count)
    records = _GradeRecords(job, submissions_readable=False)
    for prompt, k in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        records.submit(_sid(prompt, k), prompt, reward=1.0)
    closed = records.grades[_sid(1, 1)]
    closed.update(stop="episode_closed", graded_success=False,
                  closed={"status": "budget_exhausted", "reason": "max_calls"})
    closed["grade"]["reward"] = 0.0
    _grade_job(tmp_path, job, records, subnet)
    env = _report(tmp_path)["envs"]["reliquary-swe"]
    assert env["pass@1"]["value"] == pytest.approx(0.75)
    assert env["limit_cuts"] == {"rows": 1, "by_stop": {"budget:max_calls": 1}, "box_state_passed": 0}

    sessions = [
        {"job_id": job.job_id, "prompt_index": 0, "state": "closed", "closed_status": "expired"},
        {"job_id": job.job_id, "prompt_index": 0, "state": "lapsed", "closed_status": None},
        {"job_id": job.job_id, "prompt_index": 1, "state": "closed", "closed_status": "box_failed"},
        {"job_id": job.job_id, "prompt_index": 1, "state": "submitted", "closed_status": "graded"},
        {"job_id": job.job_id, "prompt_index": 1, "state": "live", "closed_status": None},
        {"job_id": "another", "prompt_index": 0, "state": "closed", "closed_status": "expired"},
    ]
    summary = redone_slots(sessions, job.job_id, lambda i: rows[i]["problem_id"])
    assert summary == {"redone": 3, "by_reason": {"expired": 1, "lapsed": 1, "box_failed": 1},
                       "by_problem": {rows[0]["problem_id"]: {"expired": 1, "lapsed": 1},
                                      rows[1]["problem_id"]: {"box_failed": 1}}}


def test_only_a_scored_close_builds_an_episode_closed_trajectory():
    from reliquary.corpus.trajectory import TrajectoryUnbuildable, build_trajectory

    with pytest.raises(TrajectoryUnbuildable, match="stop 'episode_closed'"):
        build_trajectory([], final_diff="", stop="episode_closed")
    with pytest.raises(TrajectoryUnbuildable, match="no turn"):
        build_trajectory([], final_diff="", stop="episode_closed", closed=True)
