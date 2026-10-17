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


# -- eval jobs: an episode whose prompt_source names a frozen eval set ------------

EVAL_SOURCE = "eval-set:telecom-bench:50:" + "e" * 64
SUITE_ENV = "reliquary_stateful_tools_v1"


def _suite_manifest(split="train", **overrides):
    raw = _manifest(with_episode=False, job_id="suite-eval-v1", prompt_source=SUITE_ENV,
                    prompt_count=50, renderer_id="chatml-v1",
                    sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                              "min_new_tokens": 2, "max_new_tokens": 3072, "n": 1})
    raw["episode"] = {"env": {"package": "reliquary-suite", "name": SUITE_ENV,
                              "version": "f" * 64, "split": split},
                      "harness": "suite", "renderer": "chatml-v1", "max_turns": 8,
                      "max_tokens_per_turn": 3072, "max_total_tokens": 60000}
    raw.update(overrides)
    return raw


def _signed(env=None, **overrides):
    from tests.unit.test_corpus_job_signed_sandbox import signed_episode

    episode = signed_episode(**overrides)
    if env is not None:
        episode["env"] = {**episode["env"], **env}
    return episode


def test_suite_eval_job_may_name_any_split():
    job = parse_job(_suite_manifest(split="benchmark", prompt_source=EVAL_SOURCE))
    assert job.episode.env.split == "benchmark" and job.prompt_source == EVAL_SOURCE
    assert parse_job(job.to_contract()) == job


def test_suite_non_eval_job_still_train_only():
    with pytest.raises(JobError, match="split must be 'train'"):
        parse_job(_suite_manifest(split="eval"))
    with pytest.raises(JobError, match="prompt_source"):
        parse_job(_suite_manifest(prompt_source="reliquary_retrieval_tools_v1"))


def test_agentic_eval_job_refused():
    with pytest.raises(JobError, match="an eval job on SWE runs in the signed sandbox"):
        parse_job(_manifest(prompt_source=EVAL_SOURCE))
    env = {"package": "reliquary-swe", "version": COMMIT, "split": "benchmark", "num_images": 20}
    with pytest.raises(JobError, match="an eval job on SWE runs in the signed sandbox"):
        parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=_episode(env=env)))


def test_agentic_non_eval_job_still_train_only():
    env = {"package": "reliquary-swe", "version": COMMIT, "split": "benchmark", "num_images": 20}
    with pytest.raises(JobError, match="split must be 'train'"):
        parse_job(_manifest(episode=_signed(env=env)))


def test_signed_eval_job_accepts_eval_source():
    job = parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=_signed()))
    assert job.prompt_source == EVAL_SOURCE and job.episode.env.split == "train"


def test_signed_eval_job_any_image_count():
    job = parse_job(_manifest(prompt_source=EVAL_SOURCE,
                              episode=_signed(env={"split": "eval", "num_images": 500})))
    assert job.episode.env.split == "eval" and job.episode.env.num_images == 500
    assert parse_job(job.to_contract()) == job


def test_signed_train_job_keeps_twenty_images():
    with pytest.raises(JobError, match="num_images must be 20"):
        parse_job(_manifest(episode=_signed(env={"num_images": 19})))
    with pytest.raises(JobError, match="num_images must be 20"):
        parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=_signed(env={"num_images": 19})))
