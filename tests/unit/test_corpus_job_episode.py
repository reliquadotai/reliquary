"""The optional `episode` field: parsed strictly, written only when set."""

import copy
import hashlib
import json

import pytest

from reliquary.corpus.job import JobError, parse_job

COMMIT = "a" * 40
VERIFIERS = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"


def _episode(**overrides):
    episode = {
        "env": {"package": "reliquary-swe", "version": COMMIT, "split": "train", "num_images": 20},
        "harness": "bash",
        "renderer": "renderers:qwen38@0.1.11",
        "verifiers": VERIFIERS,
        "max_turns": 40,
        "max_tokens_per_turn": 8192,
        "max_total_tokens": 60000,
        "replay_fraction_failed": 0.1,
    }
    episode.update(overrides)
    return episode


def _manifest(with_episode=True, **overrides):
    raw = {
        "schema": "reliquary/corpus-job/v1", "job_id": "swe-agentic-v1",
        "checkpoint_repo": "Qwen/Qwen3.8-27B", "checkpoint_revision": "rev1",
        "checkpoint_sha256": "c" * 64, "prompt_source": "reliquary_agentic_swe_v1",
        "prompt_count": 18546, "renderer_id": "renderers:qwen38@0.1.11", "eos_token_id": 248046,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                     "max_new_tokens": 8192, "n": 1},
        "slots_per_prompt": 2, "filter": None, "prompt_order": "free", "deadline_round": None,
    }
    if with_episode:
        raw["episode"] = _episode()
    raw.update(overrides)
    return raw


def _sha(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def test_an_episode_job_parses_and_round_trips():
    job = parse_job(_manifest())
    assert job.episode.env.num_images == 20 and job.episode.max_total_tokens == 60000
    assert job.to_contract()["episode"] == _episode()
    assert parse_job(job.to_contract()) == job


def test_a_job_without_episode_hashes_as_before():
    raw = _manifest(with_episode=False, prompt_source="openmathinstruct", renderer_id="x-v1",
                    sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                              "min_new_tokens": 2, "max_new_tokens": 4096, "n": 1})
    contract = parse_job(raw).to_contract()
    assert "episode" not in contract
    assert _sha(contract) == _sha(raw)


@pytest.mark.parametrize("field,value", [
    ("harness", "codex"), ("renderer", "qwen38"), ("verifiers", "main"),
    ("max_turns", 0), ("max_turns", 65), ("max_tokens_per_turn", 0),
    ("max_total_tokens", 60001), ("max_total_tokens", 4096),
    ("replay_fraction_failed", -0.1), ("replay_fraction_failed", 1.5),
    ("replay_fraction_failed", True), ("extra", 1),
])
def test_bad_episode_fields_are_refused(field, value):
    with pytest.raises(JobError):
        parse_job(_manifest(episode=_episode(**{field: value})))


@pytest.mark.parametrize("env", [
    {"package": "reliquary-code", "version": COMMIT, "split": "train", "num_images": 20},
    {"package": "reliquary-swe", "version": "v1", "split": "train", "num_images": 20},
    {"package": "reliquary-swe", "version": COMMIT, "split": "eval", "num_images": 20},
    {"package": "reliquary-swe", "version": COMMIT, "split": "train", "num_images": 0},
    {"package": "reliquary-swe", "version": COMMIT, "split": "train"},
])
def test_bad_episode_env_is_refused(env):
    with pytest.raises(JobError):
        parse_job(_manifest(episode=_episode(env=env)))


@pytest.mark.parametrize("overrides", [
    {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                  "max_new_tokens": 8192, "n": 2}},
    {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                  "max_new_tokens": 4096, "n": 1}},
    {"slots_per_prompt": 1},
    {"prompt_order": "miner_walk"},
    {"filter": {"grader_id": "x", "threshold": 1.0}},
    {"renderer_id": "reliquary-jsonl-tools-v1"},
])
def test_job_level_rules_bind_an_episode_job(overrides):
    with pytest.raises(JobError):
        parse_job(_manifest(**overrides))


def test_episode_dict_is_not_aliased():
    raw = _manifest()
    job = parse_job(copy.deepcopy(raw))
    raw["episode"]["max_turns"] = 1
    assert job.episode.max_turns == 40
