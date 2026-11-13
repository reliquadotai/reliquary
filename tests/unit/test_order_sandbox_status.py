"""The `sandbox` object of the order control's status, per signed episode order (designs 2026-11-12 and
2026-11-13-sandbox-sizing-miner-demand)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("reliquary_sandbox")

from reliquary.sandbox.sessions import CLOSED, HANDED_BACK, LIVE, SUBMITTED, SandboxPolicy, SessionBook, SessionRecord
from reliquary.validator.order_sandbox_status import (
    WANT_TTL_S,
    EpisodeTokens,
    audit_capacity_sessions,
    audit_episodes_per_second,
    book_sessions,
    episode_counts,
    episode_seconds,
    held_by_hotkey,
    job_demand,
    sandbox_status,
    sessions_target,
)

NOW = 1_800_000_000
JOB = "order-eval-swe1"


def rec(sid, *, job=JOB, state=LIVE, issued=NOW - 100, closed=None, status=None, expires=None, hotkey="5Hot"):
    return SessionRecord(session_id=sid, hotkey=hotkey, request_id=sid, engagement_sha256="e", kind="corpus",
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


NEW = {"demand_sessions": None, "demand_declared_sessions": None, "demand_forecast_sessions": None,
       "demand_hotkeys": None, "audit_share": None, "executors_allowed": None, "audit_capacity_sessions": None,
       "sessions_target": None, "binding": None}


def test_the_sandbox_object():
    status = {"state": "open", "submissions_accepted": 120, "audited": 0, "passed": 0}
    assert sandbox_status(job=job(), status=status, reopens=True, held=61, finished=[(NOW, 840)],
                          tokens_per_episode=41_000.0, tokens_per_second=(2_000.0, 870.0), executors_wanted=1,
                          fleet_refusals=37) == {
        "episodes_total": 400, "episodes_finished": 120, "episodes_left": 280, "sessions_held": 61,
        "episode_seconds": 840.0, "tokens_per_episode": 41_000.0,
        "audit_episodes_per_second": pytest.approx(0.07), "quota_refusals_fleet_full": 37, **NEW}
    assert sandbox_status(job=job(), status=None, reopens=False, held=0, finished=[], tokens_per_episode=None,
                          tokens_per_second=None, executors_wanted=None, fleet_refusals=0) == {
        "episodes_total": 400, "episodes_finished": None, "episodes_left": None, "sessions_held": 0,
        "episode_seconds": None, "tokens_per_episode": None, "audit_episodes_per_second": None,
        "quota_refusals_fleet_full": 0, **NEW}


def test_the_sandbox_object_names_the_target_and_its_binding_term():
    status = {"state": "open", "submissions_accepted": 120, "audited": 0, "passed": 0}
    demand = {"sessions": 192, "declared": 128, "forecast": 64, "held_undeclared": 0, "hotkeys": 3}
    common = dict(job=job(), status=status, reopens=False, held=40, finished=[(NOW, 840)],
                  tokens_per_episode=41_000.0, tokens_per_second=(2_000.0, 870.0), executors_wanted=1,
                  fleet_refusals=0, demand=demand)
    # 5 % audited, 2 executors allowed: 2 870 x 2 / (41 000 x 0.05) x 840 s = 2 352 sessions: the demand binds.
    out = sandbox_status(**common, executors_allowed=2, audit_share=0.05)
    assert out["audit_capacity_sessions"] == 2352
    assert (out["demand_sessions"], out["demand_declared_sessions"], out["demand_forecast_sessions"],
            out["demand_hotkeys"]) == (192, 128, 64, 3)
    assert (out["sessions_target"], out["binding"]) == (192, "demand")
    assert out["audit_share"] == 0.05 and out["executors_allowed"] == 2
    # Every episode audited, one executor: 2 870 / 41 000 x 840 = 58 sessions: the audit binds.
    out = sandbox_status(**common, executors_allowed=1, audit_share=1.0)
    assert (out["audit_capacity_sessions"], out["sessions_target"], out["binding"]) == (58, 58, "audit")
    # Fewer episodes left than either.
    nearly = {"state": "open", "submissions_accepted": 390, "audited": 0, "passed": 0}
    out = sandbox_status(**{**common, "status": nearly}, executors_allowed=2, audit_share=0.05)
    assert (out["sessions_target"], out["binding"]) == (10, "episodes")


def test_sessions_target_is_the_least_known_term():
    assert sessions_target(None, 280, 50) == (None, None)            # nothing declared: the platform's own rule
    assert sessions_target(64, 280, None) == (64, "demand")          # audit unmeasured: never binds
    assert sessions_target(64, None, None) == (64, "demand")
    assert sessions_target(64, 280, 30) == (30, "audit")
    assert sessions_target(64, 64, 64) == (64, "demand")             # a tie names the demand
    assert sessions_target(64, 0, 30) == (0, "episodes")


def test_audit_capacity_sessions():
    # 800 audited tokens/s per executor x 4 / (40 000 tokens x 0.1 audited) x 900 s = 720 sessions.
    assert audit_capacity_sessions((600.0, 200.0), 4, 40_000.0, 0.1, 900.0) == 720
    assert audit_capacity_sessions((600.0, 200.0), 4, 40_000.0, 1.0, 900.0) == 72
    for missing in ({"tps": None}, {"executors": 0}, {"tokens": None}, {"share": None}, {"seconds": None}):
        args = {"tps": (600.0, 200.0), "executors": 4, "tokens": 40_000.0, "share": 0.1, "seconds": 900.0, **missing}
        assert audit_capacity_sessions(args["tps"], args["executors"], args["tokens"], args["share"],
                                       args["seconds"]) is None


def test_held_by_hotkey():
    book = SessionBook(SandboxPolicy())
    book.add(rec("a1", hotkey="5A"))
    book.add(rec("a2", hotkey="5A"))
    book.add(rec("b1", hotkey="5B"))
    book.add(rec("b2", hotkey="5B", job="order-eval-other"))
    book.add(rec("done", hotkey="5A", state=SUBMITTED, issued=NOW - 1_000, closed=NOW - 10, status="graded"))
    assert held_by_hotkey(book, NOW) == {JOB: {"5A": 2, "5B": 1}, "order-eval-other": {"5B": 1}}


MODEL, OTHER_MODEL = ("Qwen/Qwen3-8B", "r1"), ("Qwen/Qwen3-32B", "r1")


def demand(job_id=JOB, *, declarations, held=None, model_of=None, served=None, excluded=lambda hotkey: False):
    return job_demand(job_id, now=NOW, declarations=declarations, held=held or {},
                      model_of=model_of or {JOB: MODEL}, served=served if served is not None else {JOB},
                      excluded=excluded)


def test_the_demand_is_none_until_someone_declares():
    out = demand(declarations={}, held={JOB: {"5Old": 4}})
    assert out == {"sessions": None, "declared": 0, "forecast": 0, "held_undeclared": 4, "hotkeys": 1}


def test_the_demand_sums_declared_wants_and_what_older_miners_hold():
    out = demand(declarations={JOB: {"5A": (64, NOW - 60), "5B": (16, NOW - 10)}},
                 held={JOB: {"5A": 10, "5B": 20, "5Old": 4}})
    # 5B holds more than it declared: what it holds counts.
    assert out == {"sessions": 64 + 20 + 4, "declared": 84, "forecast": 0, "held_undeclared": 4, "hotkeys": 3}


def test_a_stale_want_counts_only_while_its_hotkey_still_holds_sessions():
    stale = NOW - WANT_TTL_S - 1
    assert demand(declarations={JOB: {"5A": (64, stale)}})["sessions"] is None
    assert demand(declarations={JOB: {"5A": (64, stale)}}, held={JOB: {"5A": 1}})["sessions"] == 64


def test_a_low_yield_hotkey_counts_nothing():
    out = demand(declarations={JOB: {"5A": (64, NOW), "5Bad": (512, NOW)}}, excluded=lambda hotkey: hotkey == "5Bad")
    assert out["sessions"] == 64 and out["hotkeys"] == 1


def test_a_new_order_is_forecast_from_idle_miners_of_its_model():
    old, other, running = "order-eval-old", "order-eval-32b", "order-eval-running"
    declarations = {old: {"5A": (64, NOW - 3_600), "5B": (32, NOW - 7_200), "5C": (16, NOW - 600)},
                    other: {"5D": (128, NOW - 600)},
                    running: {"5C": (16, NOW - 60)}}
    model_of = {JOB: MODEL, old: MODEL, other: OTHER_MODEL, running: MODEL}
    out = demand(declarations=declarations, model_of=model_of, served={JOB, running})
    # 5A and 5B mined the retired order of the same model; 5C is busy on a running one; 5D has another model.
    assert out == {"sessions": 96, "declared": 0, "forecast": 96, "held_undeclared": 0, "hotkeys": 2}
    # Once 5A opens on the new order its own declaration replaces the forecast (never counted twice).
    declarations[JOB] = {"5A": (48, NOW)}
    out = demand(declarations=declarations, model_of=model_of, served={JOB, running})
    assert (out["sessions"], out["declared"], out["forecast"]) == (80, 48, 32)


def test_the_latest_forecast_want_of_a_hotkey_wins():
    a, b = "order-eval-a", "order-eval-b"
    declarations = {a: {"5A": (64, NOW - 3_600)}, b: {"5A": (8, NOW - 600)}}
    out = demand(declarations=declarations, model_of={JOB: MODEL, a: MODEL, b: MODEL})
    assert out["forecast"] == 8
