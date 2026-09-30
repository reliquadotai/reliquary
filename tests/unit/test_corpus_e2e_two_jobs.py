"""The two-job mode of the rehearsal (scripts/corpus_e2e.py): its pure checks
must fail when one job's ban or pay leaks into the other."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import corpus_e2e
from scripts.corpus_e2e import two_job_checks

JOBS = {"a": {"cap": 0.1}, "b": {"cap": 0.2}}
MINERS = {"honest_a": {"hotkey": "5ha"}, "honest_b": {"hotkey": "5hb"},
          "dishonest_a": {"hotkey": "5cheat"}, "dishonest_b": {"hotkey": "5cheat"}}


def _verdict(hotkey, passed):
    return {"hotkey": hotkey, "passed": passed}


def _run():
    verdicts = {"a": [_verdict("5ha", True), _verdict("5cheat", False), _verdict("5cheat", False)],
                "b": [_verdict("5hb", True), _verdict("5cheat", True)]}
    archives = {"a": {"rewards_by_hotkey": {"5ha": 0.1}, "second_settle": None},
                "b": {"rewards_by_hotkey": {"5hb": 0.12, "5cheat": 0.08}, "second_settle": None}}
    states = {"a": {"5cheat": {"effective_state": "banned"}, "5ha": {"effective_state": "probation"}},
              "b": {"5cheat": {"effective_state": "probation"}}}
    return verdicts, archives, states


def test_a_clean_two_job_run_passes_every_check():
    verdicts, archives, states = _run()
    checks = two_job_checks(JOBS, MINERS, verdicts, archives, states)
    assert all(checks.values()), checks


def test_a_ban_that_follows_the_hotkey_to_the_other_job_fails():
    verdicts, archives, states = _run()
    states["b"]["5cheat"] = {"effective_state": "banned"}
    assert not two_job_checks(JOBS, MINERS, verdicts, archives, states)["dishonest_not_banned_on_b"]


def test_a_cheater_that_is_not_banned_on_its_job_fails():
    verdicts, archives, states = _run()
    states["a"]["5cheat"] = {"effective_state": "suspect"}
    assert not two_job_checks(JOBS, MINERS, verdicts, archives, states)["dishonest_banned_on_a"]


def test_an_archive_paying_the_other_jobs_miner_fails():
    verdicts, archives, states = _run()
    archives["a"]["rewards_by_hotkey"] = {"5ha": 0.05, "5hb": 0.05}
    checks = two_job_checks(JOBS, MINERS, verdicts, archives, states)
    assert not checks["a_archive_pays_only_its_own_miners"]


def test_an_archive_paying_past_its_own_cap_fails():
    verdicts, archives, states = _run()
    archives["a"]["rewards_by_hotkey"] = {"5ha": 0.2}
    assert not two_job_checks(JOBS, MINERS, verdicts, archives, states)["a_archive_sums_to_its_cap"]


def test_a_cheater_paid_on_its_own_job_fails():
    verdicts, archives, states = _run()
    verdicts["a"][1] = _verdict("5cheat", True)
    assert not two_job_checks(JOBS, MINERS, verdicts, archives, states)["dishonest_failed_on_a"]


def _validator_args(tmp_path, n):
    paths = []
    for i in range(n):
        path = tmp_path / f"entry-{i}.json"
        # The fields of the entry `declare_task` writes that the child reads.
        path.write_text(json.dumps({"params": {"cap": 0.1 * (i + 1), "audit_q": 1.0},
                                    "profile_id": f"corpus-{i}",
                                    "contract": {"profile_id": f"corpus-{i}"}}))
        paths.append(str(path))
    return SimpleNamespace(task_id=",".join(f"corpus-{i}" for i in range(n)),
                           job_id=",".join(f"job-{i}" for i in range(n)), port=1,
                           cap=",".join(str(0.1 * (i + 1)) for i in range(n)),
                           entry=",".join(paths))


def _capture_run(monkeypatch):
    from reliquary.validator import corpus_validator

    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(corpus_validator, "run_corpus_validator", fake)
    return calls


def test_the_validator_child_with_one_task_calls_the_single_job_path(tmp_path, monkeypatch):
    calls = _capture_run(monkeypatch)
    corpus_e2e.run_validator(_validator_args(tmp_path, 1))
    assert calls[0]["entry"].task_id == "corpus-0" and calls[0]["cap"] == 0.1
    assert "jobs" not in calls[0]


def test_the_validator_child_with_two_tasks_serves_both_in_one_process(tmp_path, monkeypatch):
    calls = _capture_run(monkeypatch)
    corpus_e2e.run_validator(_validator_args(tmp_path, 2))
    assert len(calls) == 1
    assert [(e.task_id, e.job_id, cap) for e, cap in calls[0]["jobs"]] == [
        ("corpus-0", "job-0", 0.1), ("corpus-1", "job-1", 0.2)]
    assert [e.contract["profile_id"] for e, _ in calls[0]["jobs"]] == ["corpus-0", "corpus-1"]
