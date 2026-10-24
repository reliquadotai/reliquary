"""`publish-set`: prompts to the platform bucket, grading to the subnet bucket only."""

from __future__ import annotations

import asyncio
import json

import pytest

from reliquary.corpus.delivery import LocalDirectorySink
from reliquary.eval.sets import build_set
from reliquary.eval.storage import SetConflict, SubnetEvalStore, publish_set
from tests.unit.test_eval_sets import opener


def _built(tmp_path):
    build_set("logic", count=4, seed=1, out=tmp_path / "set", open_environment=opener(),
              clock=lambda: 1.0)
    return tmp_path / "set"


def test_publish_splits_the_set_between_the_buckets(tmp_path):
    platform, subnet = LocalDirectorySink(tmp_path / "p"), LocalDirectorySink(tmp_path / "s")
    answer = asyncio.run(publish_set(_built(tmp_path), platform=platform, subnet=subnet))
    set_id = "logic-eval-s1-n4"
    assert answer["set_id"] == set_id and len(answer["written"]) == 5
    assert (tmp_path / "p" / "eval-sets" / set_id / "prompts.jsonl").exists()
    assert (tmp_path / "p" / "eval-sets" / set_id / "set.json").exists()
    assert not (tmp_path / "p" / "eval-sets" / set_id / "grading.jsonl").exists()
    assert (tmp_path / "s" / "reliquary" / "eval-sets" / set_id / "grading.jsonl").exists()
    assert (tmp_path / "s" / "reliquary" / "eval-sets" / set_id / "prompts.jsonl").exists()
    assert (tmp_path / "s" / "reliquary" / "eval-sets" / set_id / "set.json").exists()
    # Again: nothing to write.
    again = asyncio.run(publish_set(_built_again(tmp_path), platform=platform, subnet=subnet))
    assert again["written"] == []


def _built_again(tmp_path):
    build_set("logic", count=4, seed=1, out=tmp_path / "set2", open_environment=opener(),
              clock=lambda: 1.0)
    return tmp_path / "set2"


def test_a_published_set_is_frozen(tmp_path):
    platform, subnet = LocalDirectorySink(tmp_path / "p"), LocalDirectorySink(tmp_path / "s")
    asyncio.run(publish_set(_built(tmp_path), platform=platform, subnet=subnet))
    build_set("logic", count=4, seed=1, out=tmp_path / "other", open_environment=opener(),
              clock=lambda: 2.0)  # same id, another created_at
    with pytest.raises(SetConflict):
        asyncio.run(publish_set(tmp_path / "other", platform=platform, subnet=subnet))


def test_publish_refuses_files_that_do_not_match_the_card(tmp_path):
    directory = _built(tmp_path)
    (directory / "grading.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="grading"):
        asyncio.run(publish_set(directory, platform=LocalDirectorySink(tmp_path / "p"),
                                subnet=LocalDirectorySink(tmp_path / "s")))


def test_the_subnet_store_is_create_only(monkeypatch):
    from reliquary.infrastructure import corpus_job_store
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(corpus_job_store, "get_s3_client", lambda **kw: fake)
    store = SubnetEvalStore()
    asyncio.run(store.put_bytes("reliquary/eval-sets/x/grading.jsonl", b"a"))
    asyncio.run(store.put_bytes("reliquary/eval-sets/x/grading.jsonl", b"a"))
    assert asyncio.run(store.get_bytes("reliquary/eval-sets/x/grading.jsonl")) == b"a"
    assert asyncio.run(store.get_bytes("reliquary/eval-sets/x/none")) is None
    with pytest.raises(SetConflict):
        asyncio.run(store.put_bytes("reliquary/eval-sets/x/grading.jsonl", b"b"))


def test_explicit_reviewed_publication_keeps_private_args_out_and_registration_withdrawn(tmp_path):
    from reliquary.eval.protocol import seal_set
    directory = _built(tmp_path)
    card = json.loads((directory / "set.json").read_bytes())
    card.update(source_kind="verifiers", taskset={"id": "fixture", "args": {"private": "sensitive-option"},
        "package": "fixture", "package_version": "1", "verifiers_version": "1",
        "artifact_sha256": "a" * 64})
    card = seal_set(card, reviewed={"data": {"fixture": "1" * 64},
        "licenses": [{"id": "MIT", "sha256": "2" * 64}],
        "disjointness": {"disjoint": False, "evidence_sha256": "3" * 64}})
    (directory / "set.json").write_text(json.dumps(card))
    platform, subnet = LocalDirectorySink(tmp_path / "p"), LocalDirectorySink(tmp_path / "s")
    private_only = asyncio.run(publish_set(directory, platform=None, subnet=subnet))
    assert len(private_only["written"]) == 3
    answer = asyncio.run(publish_set(directory, platform=platform, subnet=subnet, publish_reviewed=True))
    assert answer["registration"]["availability"] == "withdrawn"
    assert answer["registration"]["avg_tokens"] is None
    public = (tmp_path / "p" / "eval-sets" / card["set_id"] / "set.json").read_text()
    assert "sensitive-option" not in public and '"args"' not in public
    assert json.loads(public)["protocol_sha256"] == card["protocol_sha256"]
    assert json.loads(public)["order"] == [json.loads(line)["problem_id"]
        for line in (directory / "prompts.jsonl").read_text().splitlines()]
    assert "sensitive-option" in (tmp_path / "s/reliquary/eval-sets" / card["set_id"] / "set.json").read_text()
