"""The split's switch (which jobs leave the front), what crosses to a spawned
child, and the status routes' answers for a job judged in another process:
the same public JSON as when the front judged it."""

from __future__ import annotations

import asyncio
import pickle
import threading
from types import SimpleNamespace

import pytest

from reliquary.validator.corpus_split import SplitSpec, plan_groups
from tests.unit import corpus_split_fakes as fakes

JOBS = [("corpus-code-v1", "code-qwen38-27b-v1"), ("corpus-math-omi-v1", "math-omi-qwen38-27b-v1"),
        ("corpus-if-v1", "if-qwen38-27b-v1"), ("corpus-logic-v1", "logic-qwen38-27b-v1")]


def test_unset_or_star_gives_every_job_its_own_process():
    every = [[j] for _, j in JOBS]
    assert plan_groups(None, JOBS) == every
    assert plan_groups("", JOBS) == every
    assert plan_groups("*", JOBS) == every


def test_math_alone_leaves_the_others_in_the_front():
    assert plan_groups("math-omi-qwen38-27b-v1", JOBS) == [["math-omi-qwen38-27b-v1"]]
    # By task id too.
    assert plan_groups("corpus-math-omi-v1", JOBS) == [["math-omi-qwen38-27b-v1"]]


def test_math_alone_and_the_rest_together():
    groups = plan_groups("corpus-math-omi-v1; code-qwen38-27b-v1,corpus-if-v1,logic-qwen38-27b-v1",
                         JOBS)
    assert groups == [["math-omi-qwen38-27b-v1"],
                      ["code-qwen38-27b-v1", "if-qwen38-27b-v1", "logic-qwen38-27b-v1"]]
    assert plan_groups("corpus-math-omi-v1;*", JOBS) == [
        ["math-omi-qwen38-27b-v1"], ["code-qwen38-27b-v1"], ["if-qwen38-27b-v1"],
        ["logic-qwen38-27b-v1"]]


@pytest.mark.parametrize("value,why", [
    ("math-v9", "does not serve"),
    ("corpus-math-omi-v1;math-omi-qwen38-27b-v1", "two groups"),
    ("corpus-math-omi-v1,*", "group of its own"),
])
def test_a_bad_plan_is_refused(value, why):
    with pytest.raises(ValueError, match=why):
        plan_groups(value, JOBS)


def test_the_spec_crosses_to_a_spawned_child():
    from reliquary.shared.task_registry import TaskEntry

    entry = TaskEntry(task_id="corpus-math-omi-v1", profile_id="p", profile_sha256="a" * 64,
                      mechanism="corpus-generation", params={"cap": 0.1, "audit_q": 0.15},
                      status="active", retired_at=None, contract={"environments": {}},
                      job_id="math-omi-qwen38-27b-v1")
    spec = SplitSpec(served=[(entry, 0.1)], directory="/hf/x", fingerprint="f" * 64,
                     proof=fakes.PROOF, run_dir="/tmp/x", groups=[["math-omi-qwen38-27b-v1"]])
    back = pickle.loads(pickle.dumps(spec))
    assert back.served[0][0] == entry and back.proof == fakes.PROOF
    assert back.group_of("math-omi-qwen38-27b-v1") == 0 and back.group_of("code") is None


# -- status of a job judged elsewhere ------------------------------------------


def _wiring():
    """A job's books after some verdicts and a settlement, as a judge holds them."""
    from reliquary.corpus.audit_policy import AuditParams
    from reliquary.validator.corpus_job_status import JobStats
    from reliquary.validator.corpus_miner_status import MinerBook, proof_thresholds

    stats = JobStats()
    book = MinerBook(job_id="math", task_id="corpus-math", records=None,
                     thresholds=proof_thresholds(fakes.PROOF))
    book.complete = True
    for i, (hotkey, passed) in enumerate([("5A", True), ("5A", False), ("5B", True),
                                          ("5A", True)]):
        verdict = {"hotkey": hotkey, "passed": passed, "audited": i != 3, "token_count": 100 + i,
                   "audited_at": 1000.0 + i, "reason": None if passed else "exp_mismatch",
                   "worst_exp": 0, "worst_mant_mean": 0.5, "worst_mant_median": 0.25}
        stats.observe(f"{i:064x}", verdict)
        book.observe(f"{i:064x}", verdict)
    stats.settled([f"{0:064x}"])
    book.settled([f"{0:064x}"])
    book.window(7, {"5A": 0.06, "5B": 0.04})
    for _ in range(3):
        stats.accepted()
    settler = SimpleNamespace(settled_count=1, totals={"verdicts": 1, "passed": 1,
                                                       "verified_tokens": 100, "complete": True})
    auditor = SimpleNamespace(pending_count=lambda hotkey: 2 if hotkey == "5A" else 0)
    return SimpleNamespace(stats=stats, miners=book, settler=settler, auditor=auditor,
                           audit_params=AuditParams(q=0.15), cap=0.1, job=None, entry=None)


class _Routes:
    retired: set = set()
    paused: set = set()

    def __init__(self):
        self.routers = {"math": SimpleNamespace(ledger_state=self._ledger)}

    async def _ledger(self, job):
        slots = SimpleNamespace(snapshot=lambda: {}, remaining=lambda i: 1, filled=5)
        return SimpleNamespace(prompt_count=10, prompt_source="openmathinstruct"), \
            SimpleNamespace(slots=slots)


class _MinerStates:
    async def state_at_most(self, hotkey, seconds):
        from reliquary.corpus.audit_policy import MinerState

        return MinerState(audited_passed=3), None


def _serve_judge(tmp_path, wiring):
    from reliquary.validator.corpus_feed import ArrivalFeed
    from reliquary.validator.corpus_gpu import serve_unix
    from reliquary.validator.corpus_judge_process import build_judge_app

    path = tmp_path / "judge-0.sock"
    loop = asyncio.new_event_loop()
    task = loop.create_task(serve_unix(build_judge_app({"math": wiring}, ArrivalFeed()), path))
    def run():
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    for _ in range(100):
        if path.exists():
            break
        threading.Event().wait(0.05)

    def stop():
        loop.call_soon_threadsafe(task.cancel)
        thread.join(5)

    return path, stop


def test_status_and_miner_status_of_a_job_judged_elsewhere_are_the_same(tmp_path):
    from reliquary.validator.corpus_feed import JudgeLink
    from reliquary.validator.corpus_hot_jobs import CorpusJobSet

    judged_here = _wiring()
    path, stop = _serve_judge(tmp_path, judged_here)
    front_side = SimpleNamespace(stats=judged_here.stats, judge_link=JudgeLink(path, ["math"]),
                                 miners=None, settler=None, audit_params=judged_here.audit_params,
                                 cap=0.1, job=None, miner_states=_MinerStates())
    local = SimpleNamespace(**{**vars(judged_here), "miner_states": _MinerStates()})

    def job_set(wiring):
        js = CorpusJobSet(routes=_Routes(), router_for=None, wire=None, jobs_of=lambda w: [],
                          clock=lambda: 5000.0)
        js.served["math"] = wiring
        return js

    async def answers(wiring):
        js = job_set(wiring)
        return (await js.status("math"), await js.miner_status("math", "5A"),
                await js.miner_status("math", "5B"))

    try:
        in_process = asyncio.run(answers(local))
        split = asyncio.run(answers(front_side))
    finally:
        stop()
    assert split == in_process
    status, miner_a, _ = split
    assert status["audited"] == 4 and status["passed"] == 3 and status["accepted_last_hour"] == 3
    assert miner_a["failed"] == 1 and miner_a["pending_audit"] == 2
    assert miner_a["share_last_windows"]["share"] == 0.6


def test_a_judge_that_does_not_answer_is_a_status_failure(tmp_path):
    from reliquary.validator.corpus_feed import JudgeLink
    from reliquary.validator.corpus_hot_jobs import CorpusJobSet

    wiring = SimpleNamespace(stats=_wiring().stats, cap=0.1, miners=None, settler=None,
                             judge_link=JudgeLink(tmp_path / "nobody.sock", ["math"]),
                             miner_states=_MinerStates(), audit_params=None)
    js = CorpusJobSet(routes=_Routes(), router_for=None, wire=None, jobs_of=lambda w: [])
    js.served["math"] = wiring
    with pytest.raises(Exception):
        asyncio.run(js.status("math"))
    with pytest.raises(Exception):
        asyncio.run(js.miner_status("math", "5A"))


def test_every_child_exit_is_logged_for_alerting(caplog):
    import logging

    from reliquary.validator.corpus_split import Supervisor

    spec = SplitSpec(served=[], directory="/x", fingerprint="f", proof=fakes.PROOF,
                     run_dir="/tmp/x", groups=[["math"]])
    sup = Supervisor(spec, clock=lambda: 100.0)
    started = []
    sup._start = lambda child: started.append(child.name)

    class _Dead:
        exitcode = -9
        pid = 1

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    for child in sup.children.values():
        child.process, child.started_at = _Dead(), 40.0
    with caplog.at_level(logging.ERROR, logger="reliquary.validator.corpus_split"):
        sup.check()
    lines = [r.getMessage() for r in caplog.records]
    for name in ("gpu", "judge-0", "front"):
        assert any(line.startswith(f"corpus split: {name} exited (code -9)") for line in lines), lines
