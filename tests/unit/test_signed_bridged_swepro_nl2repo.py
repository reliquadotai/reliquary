"""SWE-bench Pro V2 (`reliquary-swebench-pro`) and NL2Repo-Bench (`reliquary-nl2repo`) as
bridged eval envs: the same signed, eval-only path as TMax (`test_signed_bridged_eval`),
with the SWE tools (bash and edit) and a pinned harness spec. The bridged envs are faked;
no package, image or gateway is needed."""

from __future__ import annotations

import json

import pytest

from reliquary.corpus.job import BRIDGED_EVAL_PACKAGES, JobError, parse_job
from reliquary.eval import prompt_source as ps
from tests.unit.test_corpus_job_episode import COMMIT, _manifest
from tests.unit.test_corpus_job_signed_sandbox import sandbox_spec, signed_episode
from tests.unit.test_signed_bridged_eval import EVAL_SOURCE, FakeBridged

SPLIT = "eval"
PACKAGES = {"reliquary-swebench-pro": 642, "reliquary-nl2repo": 104}
OPTIONS = {"tools": ["bash", "edit"], "splits": {SPLIT: {"split": SPLIT}}}
BRIDGE = {"options": OPTIONS, "image_lock_sha256": "1" * 64}
HARNESS = {"id": "mini-swe", "version": "2", "digest": "c" * 64}


def env_package(package):
    return f"{package}==0.1.0+g0123456789abcdef.vfb2"


def bridged_episode(package, **overrides):
    episode = signed_episode(sandbox=sandbox_spec(env=package, env_package=env_package(package)),
                             bridge=BRIDGE, harness_spec=HARNESS)
    episode["env"] = {"package": package, "version": COMMIT, "split": SPLIT,
                      "num_images": PACKAGES[package]}
    episode.update(overrides)
    return episode


def test_both_are_bridged_eval_packages():
    assert {"reliquary-terminal", *PACKAGES} <= BRIDGED_EVAL_PACKAGES
    assert "reliquary-swe" not in BRIDGED_EVAL_PACKAGES      # SWE keeps its own path


@pytest.mark.parametrize("package", PACKAGES)
def test_an_eval_job_parses_with_bash_and_edit_and_a_harness(package):
    job = parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=bridged_episode(package)))
    assert job.episode.is_bridged and job.episode.bridge == BRIDGE
    assert job.episode.sandbox.tools == ("bash", "edit")
    assert job.episode.harness_spec == HARNESS
    assert parse_job(job.to_contract()) == job


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("change, why", [
    (lambda p: dict(prompt_source="reliquary_agentic_swe_v1"), "split must be 'train'"),
    (lambda p: dict(episode={k: v for k, v in bridged_episode(p).items() if k != "bridge"}),
     "needs episode.bridge"),
    (lambda p: dict(episode=bridged_episode(p, harness="suite")), "harness"),
])
def test_a_job_is_signed_eval_and_bridged_only(package, change, why):
    raw = _manifest(prompt_source=EVAL_SOURCE, episode=bridged_episode(package))
    raw.update(change(package))
    with pytest.raises(JobError, match=why):
        parse_job(raw)


@pytest.mark.parametrize("package, rows", PACKAGES.items())
def test_a_set_freezes_references_to_the_tasks(package, rows, tmp_path):
    from reliquary.eval import sets

    fake = FakeBridged(rows=rows)
    fake.identity = env_package(package)
    card = sets.build_source_set(package, out=tmp_path / "set", split=SPLIT, start=0,
                                 count=rows, bridge=BRIDGE, open_bridged=lambda *a: fake)
    lines = (tmp_path / "set" / "prompts.jsonl").read_text().splitlines()
    assert card["interaction"] == "signed_bridged" and card["count"] == rows == len(lines)
    assert card["bridge"]["env_package"] == env_package(package)
    assert card["env_manifest"] == env_package(package)
    assert ps.set_interaction([json.loads(line) for line in lines]) == "signed_bridged"
    assert card["set_id"].startswith(f"{package}-eval-")


@pytest.mark.parametrize("package", PACKAGES)
def test_jobs_create_holds_the_episode_to_its_set(package):
    from reliquary.eval.episode_jobs import refuse_episode_disagreeing_with_set

    card = {"interaction": "signed_bridged", "split": SPLIT, "source": package,
            "bridge": {**BRIDGE, "options_sha256": "o" * 64, "env_package": env_package(package)}}
    episode = bridged_episode(package)
    refuse_episode_disagreeing_with_set(card, "s", episode, episode["renderer"])
    with pytest.raises(ValueError, match="bridge"):
        refuse_episode_disagreeing_with_set(
            card, "s", bridged_episode(package, bridge={**BRIDGE, "options": {"tools": ["bash"]}}),
            episode["renderer"])
