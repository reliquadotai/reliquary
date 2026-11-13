"""Terminal-Bench 2.1 as a bridged eval env (`reliquary-terminal-bench`): the same signed,
eval-only path as TMax (`test_signed_bridged_eval`), with the bash tool and a pinned
harness spec. The bridged env is faked; no package, image or gateway is needed."""

from __future__ import annotations

import json

import pytest

from reliquary.corpus.job import BRIDGED_EVAL_PACKAGES, JobError, parse_job
from reliquary.eval import prompt_source as ps
from tests.unit.test_corpus_job_episode import COMMIT, _manifest
from tests.unit.test_corpus_job_signed_sandbox import sandbox_spec, signed_episode
from tests.unit.test_signed_bridged_eval import EVAL_SOURCE, FakeBridged

PACKAGE = "reliquary-terminal-bench"
SPLIT = "eval"
ENV_PACKAGE = "reliquary-terminal-bench==0.1.0+g0123456789abcdef.vfb2"
OPTIONS = {"tools": ["bash"], "splits": {SPLIT: {"split": SPLIT}}}
BRIDGE = {"options": OPTIONS, "image_lock_sha256": "1" * 64}
HARNESS = {"id": "terminus-2", "version": "1", "digest": "c" * 64}


def tb_episode(**overrides):
    episode = signed_episode(sandbox=sandbox_spec(env=PACKAGE, env_package=ENV_PACKAGE, tools=["bash"]),
                             bridge=BRIDGE, harness_spec=HARNESS)
    episode["env"] = {"package": PACKAGE, "version": COMMIT, "split": SPLIT, "num_images": 79}
    episode.update(overrides)
    return episode


def test_terminal_bench_is_a_bridged_eval_package():
    assert {"reliquary-terminal", "reliquary-deepswe", "reliquary-terminal-bench"} <= BRIDGED_EVAL_PACKAGES
    assert "reliquary-swe" not in BRIDGED_EVAL_PACKAGES


def test_a_terminal_bench_eval_job_parses_with_bash_and_a_harness():
    job = parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=tb_episode()))
    assert job.episode.is_bridged and job.episode.bridge == BRIDGE
    assert job.episode.sandbox.tools == ("bash",)
    assert job.episode.harness_spec == HARNESS
    assert parse_job(job.to_contract()) == job


@pytest.mark.parametrize("change, why", [
    (dict(prompt_source="reliquary_agentic_swe_v1"), "split must be 'train'"),
    (dict(episode={k: v for k, v in tb_episode().items() if k != "bridge"}), "needs episode.bridge"),
    (dict(episode=tb_episode(harness="suite")), "harness"),
])
def test_a_terminal_bench_job_is_signed_eval_and_bridged_only(change, why):
    raw = _manifest(prompt_source=EVAL_SOURCE, episode=tb_episode())
    raw.update(change)
    with pytest.raises(JobError, match=why):
        parse_job(raw)


def test_a_set_freezes_references_to_terminal_bench_tasks(tmp_path):
    from reliquary.eval import sets

    fake = FakeBridged(rows=79)
    fake.identity = ENV_PACKAGE
    card = sets.build_source_set(PACKAGE, out=tmp_path / "set", split=SPLIT, start=0, count=79,
                                 bridge=BRIDGE, open_bridged=lambda *a: fake)
    rows = [json.loads(line) for line in (tmp_path / "set" / "prompts.jsonl").read_text().splitlines()]
    assert card["interaction"] == "signed_bridged" and card["count"] == 79
    assert card["bridge"]["env_package"] == ENV_PACKAGE
    assert ps.set_interaction(rows) == "signed_bridged"
    assert card["set_id"].startswith("reliquary-terminal-bench-eval-")


def test_jobs_create_holds_a_terminal_bench_episode_to_its_set():
    from reliquary.eval.episode_jobs import refuse_episode_disagreeing_with_set

    card = {"interaction": "signed_bridged", "split": SPLIT, "source": PACKAGE,
            "bridge": {**BRIDGE, "options_sha256": "o" * 64, "env_package": ENV_PACKAGE}}
    episode = tb_episode()
    refuse_episode_disagreeing_with_set(card, "s", episode, episode["renderer"])
    with pytest.raises(ValueError, match="bridge"):
        refuse_episode_disagreeing_with_set(
            card, "s", tb_episode(bridge={**BRIDGE, "options": {"tools": ["bash", "edit"]}}),
            episode["renderer"])


def test_the_terminus_2_leaderboard_profile():
    from reliquary.corpus.job import EPISODE_PROFILES, EPISODE_TURNS_CEILING, TERMINAL_BENCH_TERMINUS_2

    profile = EPISODE_PROFILES[TERMINAL_BENCH_TERMINUS_2]
    assert (profile.harness, profile.package) == (("terminus-2", "1"), PACKAGE)
    assert EPISODE_TURNS_CEILING >= profile.max_turns == 250
    sandbox = sandbox_spec(env=PACKAGE, env_package=ENV_PACKAGE, tools=["bash"])
    sandbox["budgets"] = {**sandbox["budgets"], "max_calls": 512, "per_call_timeout_s": 600,
                          "wall_s": 12000, "cpu_s": 12000}
    episode = tb_episode(sandbox=sandbox, profile=TERMINAL_BENCH_TERMINUS_2, max_turns=250,
                         max_total_tokens=131072)
    job = parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=episode))
    assert job.episode.profile == TERMINAL_BENCH_TERMINUS_2 and job.episode.max_turns == 250
    with pytest.raises(JobError, match="terminus-2@1"):
        parse_job(_manifest(prompt_source=EVAL_SOURCE,
                            episode={**episode, "harness_spec": {**HARNESS, "id": "mini-swe", "version": "2"}}))
    with pytest.raises(JobError, match="wall_s"):
        short = {**sandbox, "budgets": {**sandbox["budgets"], "wall_s": 7200, "cpu_s": 7200}}
        parse_job(_manifest(prompt_source=EVAL_SOURCE, episode={**episode, "sandbox": short}))
