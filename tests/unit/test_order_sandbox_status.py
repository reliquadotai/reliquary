"""The `sandbox` object of the order control's status, per signed episode order (design 2026-11-12)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("reliquary_sandbox")

from reliquary.sandbox.sessions import CLOSED, HANDED_BACK, LIVE, SUBMITTED, SandboxPolicy, SessionBook, SessionRecord
from reliquary.validator.order_sandbox_status import (
    EpisodeTokens,
    audit_episodes_per_second,
    book_sessions,
    episode_counts,
    episode_seconds,
    sandbox_status,
)

NOW = 1_800_000_000
JOB = "order-eval-swe1"


def rec(sid, *, job=JOB, state=LIVE, issued=NOW - 100, closed=None, status=None, expires=None):
    return SessionRecord(session_id=sid, hotkey="5Hot", request_id=sid, engagement_sha256="e", kind="corpus",
                         engagement=f"corpus:{job}:0", env="reliquary-swe", split="train", index=0, checkpoint="c",
                         job_id=job, prompt_index=0, machine_id="m", issued_at=issued,
                         expires_at=issued + 5_000 if expires is None else expires, token_sha256="t", state=state,
                         closed_status=status, closed_at=closed)


def job(prompts=100, samples=4):
    return SimpleNamespace(prompt_count=prompts, slots_per_prompt=samples)


def test_episode_tokens_mean_audited_prompt_and_completion_once_per_episode():
    tokens = EpisodeTokens(kept=3)
    assert tokens.mean() is None                                  # null before any
    tokens.observe("a", {"token_count": 30_000, "prompt_token_count": 10_000})
    tokens.observe("a", {"token_count": 30_000, "prompt_token_count": 10_000})   # the settler re-reads it
    tokens.observe("b", {"token_count": 20_000})                  # no prompt count recorded
    tokens.observe("c", None)
    tokens.observe("d", {"token_count": 0})
    assert tokens.mean() == pytest.approx(30_000)
    for sid in "efg":                                             # only the last `kept` count
        tokens.observe(sid, {"token_count": 1_000})
    assert tokens.mean() == pytest.approx(1_000)


def test_the_book_gives_each_jobs_held_sessions_and_finished_walls():
    book = SessionBook(SandboxPolicy())
    book.add(rec("live"))
    book.add(rec("expired", issued=NOW - 90_000, expires=NOW - 80_000))           # lapsed in all but name
    book.add(rec("done", state=SUBMITTED, issued=NOW - 1_000, closed=NOW - 160, status="graded"))
    book.add(rec("back", state=SUBMITTED, issued=NOW - 1_000, closed=NOW - 10, status=HANDED_BACK))
    book.add(rec("failed", state=CLOSED, issued=NOW - 1_000, closed=NOW - 900, status="open_failed"))
    book.add(rec("other", job="order-eval-other"))
    sessions = book_sessions(book, NOW)
    assert sessions[JOB] == (1, [(NOW - 160, 840)])
    assert sessions["order-eval-other"] == (1, [])


def test_episode_seconds_is_the_median_of_the_last_fifty():
    assert episode_seconds([]) is None
    old = [(NOW - 10_000 + i, 5_000) for i in range(30)]
    recent = [(NOW + i, 600 + 10 * i) for i in range(50)]
    assert episode_seconds(old + recent) == pytest.approx(600 + 10 * 24.5)
    assert episode_seconds([(NOW, 840), (NOW + 1, 900), (NOW + 2, 100)]) == 840.0


def test_episode_counts_from_the_jobs_status():
    status = {"state": "open", "submissions_accepted": 130, "audited": 100, "passed": 90}
    assert episode_counts(job(), status, reopens=False) == (400, 130, 270)
    # An eval order reopens a failed audit's slot: that episode is played again.
    assert episode_counts(job(), status, reopens=True) == (400, 120, 280)
    assert episode_counts(job(), {**status, "state": "full"}, reopens=True) == (400, 120, 0)
    assert episode_counts(job(), {**status, "complete": True}, reopens=True)[2] == 0
    assert episode_counts(job(), None, reopens=False) == (400, None, None)


def test_audit_episodes_per_second():
    # 600 completion + 200 prompt tokens/s per executor, 2 executors wanted, 40k tokens per episode.
    assert audit_episodes_per_second((600.0, 200.0), 2, 40_000.0) == pytest.approx(0.04)
    assert audit_episodes_per_second((600.0, 200.0), 0, 40_000.0) == 0.0
    assert audit_episodes_per_second(None, 2, 40_000.0) is None          # throughput unmeasured
    assert audit_episodes_per_second((600.0, 200.0), None, 40_000.0) is None
    assert audit_episodes_per_second((600.0, 200.0), 2, None) is None


def test_the_sandbox_object():
    status = {"state": "open", "submissions_accepted": 120, "audited": 0, "passed": 0}
    assert sandbox_status(job=job(), status=status, reopens=True, held=61, finished=[(NOW, 840)],
                          tokens_per_episode=41_000.0, tokens_per_second=(2_000.0, 870.0), executors_wanted=1,
                          fleet_refusals=37) == {
        "episodes_total": 400, "episodes_finished": 120, "episodes_left": 280, "sessions_held": 61,
        "episode_seconds": 840.0, "tokens_per_episode": 41_000.0,
        "audit_episodes_per_second": pytest.approx(0.07), "quota_refusals_fleet_full": 37}
    assert sandbox_status(job=job(), status=None, reopens=False, held=0, finished=[], tokens_per_episode=None,
                          tokens_per_second=None, executors_wanted=None, fleet_refusals=0) == {
        "episodes_total": 400, "episodes_finished": None, "episodes_left": None, "sessions_held": 0,
        "episode_seconds": None, "tokens_per_episode": None, "audit_episodes_per_second": None,
        "quota_refusals_fleet_full": 0}
