"""DeepSWE v1.1 as a bridged eval env (`reliquary-deepswe`): the same signed, eval-only path
as TMax (`test_signed_bridged_eval`), with the SWE tools (bash and edit) and a pinned
harness spec. The bridged env is faked; no package, image or gateway is needed."""

from __future__ import annotations

import json

import pytest

from reliquary.corpus.job import BRIDGED_EVAL_PACKAGES, JobError, parse_job
from reliquary.eval import prompt_source as ps
from tests.unit.test_corpus_job_episode import COMMIT, _manifest
from tests.unit.test_corpus_job_signed_sandbox import sandbox_spec, signed_episode
from tests.unit.test_signed_bridged_eval import EVAL_SOURCE, FakeBridged

PACKAGE = "reliquary-deepswe"
SPLIT = "eval"
ENV_PACKAGE = "reliquary-deepswe==0.1.0+g0123456789abcdef.vfb2"
OPTIONS = {"tools": ["bash", "edit"], "splits": {SPLIT: {"split": SPLIT}}}
BRIDGE = {"options": OPTIONS, "image_lock_sha256": "1" * 64}
HARNESS = {"id": "mini-swe", "version": "2", "digest": "c" * 64}


def deepswe_episode(**overrides):
    episode = signed_episode(sandbox=sandbox_spec(env=PACKAGE, env_package=ENV_PACKAGE),
                             bridge=BRIDGE, harness_spec=HARNESS)
    episode["env"] = {"package": PACKAGE, "version": COMMIT, "split": SPLIT, "num_images": 113}
    episode.update(overrides)
    return episode


def test_deepswe_is_a_bridged_eval_package():
    assert {"reliquary-terminal", "reliquary-deepswe"} <= BRIDGED_EVAL_PACKAGES
    assert "reliquary-swe" not in BRIDGED_EVAL_PACKAGES      # SWE keeps its own path


def test_a_deepswe_eval_job_parses_with_bash_and_edit_and_a_harness():
    job = parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=deepswe_episode()))
    assert job.episode.is_bridged and job.episode.bridge == BRIDGE
    assert job.episode.sandbox.tools == ("bash", "edit")
    assert job.episode.harness_spec == HARNESS
    assert parse_job(job.to_contract()) == job


@pytest.mark.parametrize("change, why", [
    (dict(prompt_source="reliquary_agentic_swe_v1"), "split must be 'train'"),
    (dict(episode={k: v for k, v in deepswe_episode().items() if k != "bridge"}), "needs episode.bridge"),
    (dict(episode=deepswe_episode(harness="suite")), "harness"),
])
def test_a_deepswe_job_is_signed_eval_and_bridged_only(change, why):
    raw = _manifest(prompt_source=EVAL_SOURCE, episode=deepswe_episode())
    raw.update(change)
    with pytest.raises(JobError, match=why):
        parse_job(raw)


def test_an_unlisted_package_stays_refused():
    episode = deepswe_episode(sandbox=sandbox_spec(env="reliquary-other",
                                                   env_package="reliquary-other==0.1.0+g0123456789abcdef.vfb2"))
    episode["env"] = {**episode["env"], "package": "reliquary-other"}
    with pytest.raises(JobError):
        parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=episode))


def test_a_set_freezes_references_to_deepswe_tasks(tmp_path):
    from reliquary.eval import sets

    fake = FakeBridged(rows=113)
    fake.identity = ENV_PACKAGE
    card = sets.build_source_set(PACKAGE, out=tmp_path / "set", split=SPLIT, start=0, count=113,
                                 bridge=BRIDGE, open_bridged=lambda *a: fake)
    rows = [json.loads(line) for line in (tmp_path / "set" / "prompts.jsonl").read_text().splitlines()]
    assert card["interaction"] == "signed_bridged" and card["count"] == 113
    assert card["bridge"]["env_package"] == ENV_PACKAGE and card["env_manifest"] == ENV_PACKAGE
    assert ps.set_interaction(rows) == "signed_bridged"
    assert card["set_id"].startswith("reliquary-deepswe-eval-")


def test_jobs_create_holds_a_deepswe_episode_to_its_set():
    from reliquary.eval.episode_jobs import refuse_episode_disagreeing_with_set

    card = {"interaction": "signed_bridged", "split": SPLIT, "source": PACKAGE,
            "bridge": {**BRIDGE, "options_sha256": "o" * 64, "env_package": ENV_PACKAGE}}
    episode = deepswe_episode()
    refuse_episode_disagreeing_with_set(card, "s", episode, episode["renderer"])
    with pytest.raises(ValueError, match="bridge"):
        refuse_episode_disagreeing_with_set(
            card, "s", deepswe_episode(bridge={**BRIDGE, "options": {"tools": ["bash"]}}), episode["renderer"])
