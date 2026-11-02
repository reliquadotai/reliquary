"""Eval sets of row references: episode and SWE sources hold `(split, source_index,
instance_id)`, not rendered messages."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from reliquary.environment import bridged_swe
from reliquary.eval import prompt_source, sets
from reliquary.eval.prompt_source import (
    eval_source_for,
    reference_of,
    register_eval_prompts,
    set_interaction,
)
from reliquary.eval.sets import build_source_set


def jsonl(rows):
    return b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows)


def ref_row(i, *, split="benchmark", interaction="episode", **over):
    row = {"problem_id": f"p-{i}", "set_id": "s", "env": "e", "interaction": interaction,
           "split": split, "source_index": i, "instance_id": f"t-{i}"}
    row.update(over)
    return row


def single_row(i):
    return {"problem_id": f"p-{i}", "set_id": "s", "env": "e",
            "messages": [{"role": "user", "content": "hi"}]}


def register(rows):
    body = jsonl(rows)
    return register_eval_prompts(eval_source_for("s", body, len(rows)), body)


def test_register_accepts_reference_rows():
    rows = register([ref_row(3), ref_row(4)])
    assert set_interaction(rows) == "episode"
    assert reference_of(rows[0]) == ("benchmark", 3, "t-3")


def test_register_refuses_mixed_splits():
    with pytest.raises(ValueError, match="one split"):
        register([ref_row(1), ref_row(2, split="eval")])


def test_register_refuses_incomplete_reference_row():
    row = ref_row(5)
    del row["instance_id"]
    with pytest.raises(ValueError, match="p-5"):
        register([ref_row(1), row])


def test_register_refuses_mixed_interactions():
    with pytest.raises(ValueError, match="mix"):
        register([single_row(1), ref_row(2)])
    with pytest.raises(ValueError, match="mix"):
        register([ref_row(1), ref_row(2, interaction="swe")])


def test_single_turn_rows_are_unchanged():
    rows = register([single_row(1), single_row(2)])
    assert set_interaction(rows) == "single_turn"
    assert reference_of(rows[0]) is None


def test_reference_of_rejects_malformed_rows():
    assert reference_of(single_row(1)) is None
    assert reference_of(ref_row(1, source_index=True)) is None
    assert reference_of(ref_row(1, source_index=-1)) is None
    assert reference_of(ref_row(1, instance_id="")) is None
    assert reference_of(ref_row(1, split="")) is None
    assert reference_of(ref_row(1, interaction="single_turn")) is None


def test_set_interaction_empty_or_unknown():
    with pytest.raises(ValueError):
        set_interaction([])
    with pytest.raises(ValueError, match="agentic"):
        set_interaction([ref_row(1, interaction="agentic")])


class FakeEpisodeEnv:
    def __len__(self):
        return 10

    def get_task(self, index):
        return SimpleNamespace(id=f"t-{index}")


def rows_of(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


def test_build_source_set_episode(tmp_path, monkeypatch):
    from reliquary.environment import registry

    name = "reliquary_telecom_solo_v2"
    real = registry.ENVIRONMENT_SPECS[name]
    spec = SimpleNamespace(interaction_mode="episode", renderer_id="reliquary-chatml-tools-v1",
                           environment_manifest_sha256=real.environment_manifest_sha256)
    monkeypatch.setattr(registry, "ENVIRONMENT_SPECS", {**registry.ENVIRONMENT_SPECS, name: spec})
    seen = []

    def opener(source, split):
        seen.append((source, split))
        return FakeEpisodeEnv()

    card = build_source_set(name, split="benchmark", sample=4, seed=7, out=tmp_path / "s",
                            open_environment=opener)
    assert seen == [(name, "benchmark")]
    assert card["interaction"] == "episode"
    assert card["renderer_id"] == "reliquary-chatml-tools-v1"
    assert card["env_manifest"] == real.environment_manifest_sha256
    assert card["disjointness"]["disjoint"] is True
    prompts = rows_of(tmp_path / "s", "prompts.jsonl")
    grading = rows_of(tmp_path / "s", "grading.jsonl")
    assert len(prompts) == 4
    for row in prompts:
        assert row["split"] == "benchmark" and row["interaction"] == "episode"
        assert row["instance_id"] == f"t-{row['source_index']}"
        assert "messages" not in row
    assert all("env" not in g and "set_id" not in g for g in grading)
    assert grading == [{k: v for k, v in p.items() if k not in ("env", "set_id")}
                       for p in prompts]
    assert card["prompts_sha256"] == hashlib.sha256(
        (tmp_path / "s" / "prompts.jsonl").read_bytes()).hexdigest()
    body = (tmp_path / "s" / "prompts.jsonl").read_bytes()
    rows = register_eval_prompts(eval_source_for(card["set_id"], body, 4), body)
    assert set_interaction(rows) == "episode"


def test_build_source_set_still_refuses_unknown_interaction(tmp_path, monkeypatch):
    from reliquary.environment import registry

    name = "reliquary_telecom_solo_v2"
    monkeypatch.setattr(registry, "ENVIRONMENT_SPECS", {
        **registry.ENVIRONMENT_SPECS, name: SimpleNamespace(interaction_mode="agentic")})
    with pytest.raises(ValueError, match="agentic"):
        build_source_set(name, out=tmp_path / "s", open_environment=lambda s, p: FakeEpisodeEnv())


def test_build_source_set_swe(tmp_path, monkeypatch):
    monkeypatch.setattr(bridged_swe, "split_length", lambda split: 50)
    monkeypatch.setattr(bridged_swe, "instance_id_of", lambda split, i: f"repo__x-{i}")
    monkeypatch.setattr(bridged_swe, "package_version", lambda: "9.9.9")
    card = build_source_set("reliquary-swe", split="train", start=10, count=5,
                            out=tmp_path / "s")
    assert card["interaction"] == "swe" and card["renderer_id"] is None
    assert card["env_manifest"] == "9.9.9"
    assert card["source"] == "reliquary-swe" and card["count"] == 5
    prompts = rows_of(tmp_path / "s", "prompts.jsonl")
    assert [p["instance_id"] for p in prompts] == [f"repo__x-{i}" for i in range(10, 15)]
    assert all(p["interaction"] == "swe" and p["split"] == "train" for p in prompts)


def test_swe_source_is_exactly_reliquary_swe(tmp_path):
    with pytest.raises(ValueError, match="not a catalog environment"):
        build_source_set("reliquary_swe", out=tmp_path / "s")
    assert sets.lineage("reliquary-swe") == (
        "princeton-nlp/SWE-bench_Verified+SWE-smith", "reliquary-swe")


def test_instance_id_of_reads_the_task_data(monkeypatch):
    env = SimpleNamespace(task=lambda split, i: SimpleNamespace(
        data=SimpleNamespace(instance_id=f"{split}-{i}")))
    monkeypatch.setattr(bridged_swe, "bridged_env", lambda: env)
    assert bridged_swe.instance_id_of("train", 3) == "train-3"
