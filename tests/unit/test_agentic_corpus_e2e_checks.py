"""The end-to-end run's verdict, as a pure function of what the bucket holds."""

import copy
import json
from types import SimpleNamespace

import pytest

from scripts.agentic_corpus_e2e import (
    FORGED_HUNK,
    JOB_ID,
    build_parser,
    evaluate,
    forgeable_bash_observations,
    forged_diff,
    public_r2_settings,
    wait_complete,
)


def _sub(**kw):
    sub = {"sid": "s", "verdict_passed": True, "grade_status": "ok", "graded_success": True,
           "replay_certified": True, "voided": False, "void_reason": None,
           "bash_observations": 20,
           "replay": {"drawn": True, "status": "ok", "certified": True, "failed": False,
                      "replay_diff_equal": True, "observations_compared": 20,
                      "observations_mismatched": [], "allowed": 5, "graded_by": ["grade-a"],
                      "providers": ["hetzner"]}}
    sub.update(kw)
    return sub


def _forged(**replay):
    base = {"drawn": True, "status": "ok", "certified": False, "failed": True,
            "replay_diff_equal": True, "observations_compared": 20,
            "observations_mismatched": list(range(20)), "allowed": 5,
            "graded_by": ["grade-a", "grade-b"], "providers": ["hetzner", "digitalocean"]}
    base.update(replay)
    return _sub(replay_certified=False, voided=True, void_reason="replay_failed", replay=base)


OK = {
    "roles": {
        "honest": {"hotkey": "5H", "submissions": [_sub(), _sub(sid="t", graded_success=False,
                                                                  replay_certified=False)],
                   "state": {"confirmed_failures": 0}},
        "forge_diff": {"hotkey": "5D", "submissions": [_forged(replay_diff_equal=False,
                                                               observations_mismatched=[])],
                       "state": {"confirmed_failures": 1}},
        "forge_obs": {"hotkey": "5O", "submissions": [_forged()], "state": {"confirmed_failures": 1}},
    },
    "export": {"sft_rows": 1, "sft_hotkeys": ["5H"]},
}


def test_the_expected_outcome_passes():
    assert evaluate(OK) == ([], [])


def test_a_forger_that_was_paid_fails_the_run():
    summary = copy.deepcopy(OK)
    summary["roles"]["forge_diff"]["submissions"][0].update(voided=False, void_reason=None)
    failures, _ = evaluate(summary)
    assert any("forge_diff" in f for f in failures)


def test_a_failure_confirmed_by_one_executor_fails_the_run():
    summary = copy.deepcopy(OK)
    summary["roles"]["forge_obs"]["submissions"][0]["replay"]["graded_by"] = ["grade-a"]
    failures, _ = evaluate(summary)
    assert any("two executors" in f for f in failures)


def test_a_failure_confirmed_by_one_provider_fails_the_run():
    # Ruling P17: two executors of one provider are one vote.
    summary = copy.deepcopy(OK)
    summary["roles"]["forge_obs"]["submissions"][0]["replay"]["providers"] = ["hetzner"]
    failures, _ = evaluate(summary)
    assert any("two providers" in f for f in failures)


def test_a_disputed_forgery_fails_the_run():
    # Ruling P16: no distinct-provider second executor -> disputed, nobody sanctioned.
    summary = copy.deepcopy(OK)
    sub = summary["roles"]["forge_obs"]["submissions"][0]
    sub.update(voided=False, void_reason=None)
    sub["replay"].update(status="disputed", failed=False, graded_by=["grade-a"],
                         providers=["hetzner"])
    summary["roles"]["forge_obs"]["state"]["confirmed_failures"] = 0
    failures, _ = evaluate(summary)
    assert any("forge_obs" in f and "disputed" in f for f in failures)


def test_a_forgery_voided_for_another_reason_fails_the_run():
    summary = copy.deepcopy(OK)
    summary["roles"]["forge_diff"]["submissions"][0]["void_reason"] = "executor_quarantined"
    failures, _ = evaluate(summary)
    assert any("forge_diff" in f and "replay_failed" in f for f in failures)


def test_an_honest_miner_voided_fails_the_run():
    summary = copy.deepcopy(OK)
    summary["roles"]["honest"]["state"]["confirmed_failures"] = 1
    assert evaluate(summary)[0]


def test_an_honest_miner_without_a_certified_success_fails_the_run():
    summary = copy.deepcopy(OK)
    summary["roles"]["honest"]["submissions"][0].update(replay_certified=False)
    assert any("certified success" in f for f in evaluate(summary)[0])


def test_a_forged_diff_that_did_not_grade_is_inconclusive_not_a_pass():
    summary = copy.deepcopy(OK)
    summary["roles"]["forge_diff"]["submissions"][0].update(graded_success=False)
    failures, inconclusive = evaluate(summary)
    assert inconclusive and not any("forge_diff" in f for f in failures)


def test_a_short_forged_observation_episode_is_inconclusive():
    summary = copy.deepcopy(OK)
    sub = summary["roles"]["forge_obs"]["submissions"][0]
    sub["bash_observations"] = 5
    sub["replay"]["observations_compared"] = 5
    failures, inconclusive = evaluate(summary)
    assert inconclusive and not any("forge_obs" in f for f in failures)


def test_only_bash_observations_count_toward_the_forgery_budget():
    # Ruling P7: edit observations are not forged by BASH_ENV, so 30 compared
    # observations of which 4 are bash cannot exceed a budget of 5.
    summary = copy.deepcopy(OK)
    sub = summary["roles"]["forge_obs"]["submissions"][0]
    sub["bash_observations"] = 4
    sub["replay"].update(observations_compared=30, allowed=5, failed=False, certified=True)
    sub.update(voided=False, void_reason=None)
    summary["roles"]["forge_obs"]["state"]["confirmed_failures"] = 0
    failures, inconclusive = evaluate(summary)
    assert any("forge_obs" in i for i in inconclusive)
    assert not any("forge_obs" in f for f in failures)


def test_an_unknown_bash_count_is_inconclusive():
    summary = copy.deepcopy(OK)
    summary["roles"]["forge_obs"]["submissions"][0]["bash_observations"] = None
    assert any("forge_obs" in i for i in evaluate(summary)[1])


def test_a_forger_caught_by_toploc_is_inconclusive():
    summary = copy.deepcopy(OK)
    summary["roles"]["forge_obs"]["submissions"][0]["verdict_passed"] = False
    failures, inconclusive = evaluate(summary)
    assert any("TOPLOC" in i for i in inconclusive)
    assert not any("forge_obs" in f for f in failures)


def test_a_replayed_diff_equal_to_the_forged_one_fails_the_run():
    summary = copy.deepcopy(OK)
    summary["roles"]["forge_diff"]["submissions"][0]["replay"]["replay_diff_equal"] = True
    assert any("forge_diff" in f for f in evaluate(summary)[0])


def test_no_certified_row_fails_the_run():
    summary = copy.deepcopy(OK)
    summary["export"] = {"sft_rows": 0, "sft_hotkeys": []}
    assert evaluate(summary)[0]


def test_a_forgers_row_in_the_sft_set_fails_the_run():
    summary = copy.deepcopy(OK)
    summary["export"]["sft_hotkeys"] = ["5H", "5O"]
    assert any("SFT" in f for f in evaluate(summary)[0])


def test_a_forger_without_a_submission_fails_the_run():
    summary = copy.deepcopy(OK)
    summary["roles"]["forge_diff"]["submissions"] = []
    assert any("forge_diff" in f for f in evaluate(summary)[0])


# -- the other pure pieces --------------------------------------------------

def _action(tool, arguments, observation="out"):
    return SimpleNamespace(tool=tool, arguments=arguments, observation=observation)


def test_forgeable_bash_observations_counts_answered_bash_calls_only():
    actions = [
        _action("bash", json.dumps({"command": "ls"})),
        _action("bash", json.dumps({"command": ""})),            # still runs bash -c
        _action("edit", json.dumps({"path": "a", "old_str": "x", "new_str": "y"})),
        _action("bash", json.dumps({"command": "ls"}), None),    # cut by context_length
        _action("bash", "{not json"),                             # answered by the harness, no bash
        _action("bash", json.dumps(["ls"])),                      # not an object: no bash
        _action("search", json.dumps({"q": "x"})),
    ]
    assert forgeable_bash_observations(actions) == 2


def test_the_forged_diff_keeps_the_gold_patch_and_adds_a_file():
    gold = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    for patch in (gold, gold.rstrip("\n"), gold + "\n\n"):
        forged = forged_diff(patch)
        assert forged.startswith(gold.rstrip("\n") + "\n")
        assert forged.endswith(FORGED_HUNK)
        assert "\n\ndiff --git a/FORGED.txt" not in forged


def test_wait_complete_needs_every_submission():
    assert not wait_complete(set(), set())
    assert not wait_complete({"a", "b"}, {"a"})
    assert wait_complete({"a", "b"}, {"a", "b", "c"})


def test_public_r2_settings_never_keep_a_credential():
    env = {"R2_BUCKET_ID": "reliquary-agentic-e2e", "R2_ENDPOINT_URL": "http://127.0.0.1:9000",
           "R2_REGION": "us-east-1", "R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s",
           "R2_SESSION_TOKEN": "t", "R2_WRITE_ACCESS_KEY_ID": "w", "HF_TOKEN": "h", "PATH": "/bin"}
    assert public_r2_settings(env) == {"R2_BUCKET_ID": "reliquary-agentic-e2e",
                                       "R2_ENDPOINT_URL": "http://127.0.0.1:9000",
                                       "R2_REGION": "us-east-1"}


def test_the_subcommands_all_take_a_state_directory():
    parser = build_parser()
    for argv in (["prepare", "--env-commit", "a" * 40], ["validator"], ["mine"], ["wait-grades"],
                 ["wait-verdicts"], ["check", "--out", "s.json"]):
        args = parser.parse_args([argv[0], "--state", "/tmp/x", *argv[1:]])
        assert args.state == "/tmp/x"
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    assert JOB_ID == "agentic-e2e"


# -- MinIO, shared with the single-turn e2e ----------------------------------

def _fake_docker(monkeypatch):
    from scripts import corpus_e2e

    calls = []
    monkeypatch.setattr(corpus_e2e.subprocess, "run",
                        lambda argv, **kw: calls.append((list(argv), kw.get("env"))))
    monkeypatch.setattr(corpus_e2e, "_wait_minio_live", lambda port: None)
    for name in ("R2_ENDPOINT_URL", "R2_REGION", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    return corpus_e2e, calls


def test_minio_takes_its_credentials_from_the_environment_never_argv(monkeypatch):
    corpus_e2e, calls = _fake_docker(monkeypatch)
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "e2euser")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "s3cr3t-password")
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://production.example")
    corpus_e2e.start_minio(container="agentic-e2e-minio", credentials_from_env=True)
    (rm, _), (run, env) = calls
    assert rm == ["docker", "rm", "-f", "agentic-e2e-minio"]
    assert "agentic-e2e-minio" in run and "corpus-e2e-minio" not in run
    assert not any("s3cr3t" in a or "e2euser" in a for a in run)
    assert env["MINIO_ROOT_USER"] == "e2euser" and env["MINIO_ROOT_PASSWORD"] == "s3cr3t-password"
    import os
    assert os.environ["R2_ENDPOINT_URL"] == "http://127.0.0.1:9000"   # forced, never defaulted
    assert os.environ["R2_SECRET_ACCESS_KEY"] == "s3cr3t-password"


def test_minio_from_env_refuses_missing_credentials(monkeypatch):
    corpus_e2e, calls = _fake_docker(monkeypatch)
    with pytest.raises(RuntimeError):
        corpus_e2e.start_minio(container="agentic-e2e-minio", credentials_from_env=True)
    assert calls == []


def test_the_single_turn_default_generates_credentials_out_of_argv(monkeypatch):
    corpus_e2e, calls = _fake_docker(monkeypatch)
    corpus_e2e.start_minio()
    (rm, _), (run, env) = calls
    assert rm == ["docker", "rm", "-f", "corpus-e2e-minio"]
    import os
    assert os.environ["R2_SECRET_ACCESS_KEY"] == env["MINIO_ROOT_PASSWORD"]
    assert not any(env["MINIO_ROOT_PASSWORD"] in a for a in run)


def test_the_agentic_run_never_names_the_single_turn_container():
    from scripts import agentic_corpus_e2e, corpus_e2e

    assert agentic_corpus_e2e.MINIO_CONTAINER != corpus_e2e.MINIO_CONTAINER


def test_the_forged_bash_env_names_a_file_every_box_has():
    # bash ignores a BASH_ENV file that does not exist (measured 2026-10-03:
    # the forged episode's observations equalled the replay's), so the forgery
    # sources a file present in every container whose content is no command.
    from scripts.agentic_corpus_e2e import FORGED_BASH_ENV

    assert FORGED_BASH_ENV == {"BASH_ENV": "/etc/hostname"}


def test_forger_keys_land_on_prompts_nobody_else_visits():
    from scripts.agentic_corpus_e2e import pick_forger_keys
    from reliquary.corpus.walk import job_walk_index

    job = SimpleNamespace(job_id=JOB_ID, prompt_start=8, prompt_count=8)
    honest = "5FjxQw33tLKUwqeoJ8oDtgmejcRUw25eKAaApKoawiV7CWur"
    counter = iter(range(10_000))
    keys = pick_forger_keys(job, honest, honest_episodes=6,
                            make_key=lambda: SimpleNamespace(ss58_address=f"key-{next(counter)}"))
    honest_prompts = {job_walk_index(job, honest, c) for c in range(6)}
    forger_prompts = [job_walk_index(job, keys[r].ss58_address, 0) for r in ("forge_diff", "forge_obs")]
    assert len(set(forger_prompts)) == 2
    assert not set(forger_prompts) & honest_prompts


def test_forger_keys_refuse_when_no_prompt_is_free():
    from scripts.agentic_corpus_e2e import pick_forger_keys

    job = SimpleNamespace(job_id=JOB_ID, prompt_start=0, prompt_count=1)
    counter = iter(range(10_000))
    with pytest.raises(SystemExit):
        pick_forger_keys(job, "honest", honest_episodes=1,
                         make_key=lambda: SimpleNamespace(ss58_address=f"key-{next(counter)}"))


def test_audit_attention_falls_back_to_sdpa_only_without_flash_attn():
    from scripts.agentic_corpus_e2e import audit_attention

    assert audit_attention(False, {}) == "sdpa"
    assert audit_attention(True, {}) is None
    assert audit_attention(False, {"GRAIL_ATTN_IMPL": "eager"}) is None
