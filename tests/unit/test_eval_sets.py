"""Frozen evaluation sets: the held-out table, its disjointness, and build-set."""

from __future__ import annotations

import hashlib
import inspect
import json
import random

import pytest

from reliquary.eval import sets
from reliquary.eval.sets import (
    CORPUS_RANGES,
    HELD_OUT,
    RL_OVERLAP_RULED,
    build_set,
    overlaps,
    refuse_held_out_overlap,
    rl_ranges,
)


class FakeEnvironment:
    def __init__(self, name: str, length: int = 1000) -> None:
        self.name, self._length = name, length

    def __len__(self) -> int:
        return self._length

    def get_problem(self, index: int) -> dict:
        return {"prompt": f"{self.name} problem {index}", "ground_truth": str(index),
                "generator_index": index}


def opener(length=1000):
    opened = []

    def open_environment(source, split):
        opened.append((source, split))
        return FakeEnvironment(source, length)

    open_environment.opened = opened
    return open_environment


def test_the_v1_environments_are_held_out():
    assert set(HELD_OUT) == {"math", "code", "logic", "instruction_following"}
    for held in HELD_OUT.values():
        assert held.justification.strip()
        assert held.source in __import__(
            "reliquary.protocol.environment_catalog", fromlist=["x"]).ENVIRONMENT_CATALOG


def test_every_held_out_region_is_disjoint_from_every_corpus_job_and_rl():
    used = list(CORPUS_RANGES) + rl_ranges()
    for env, held in HELD_OUT.items():
        for other in used:
            if not overlaps(held.region, other):
                continue
            # The one admitted overlap: RL eligibility, ruled per environment.
            assert other.kind == "rl" and env in RL_OVERLAP_RULED, (env, other)
    # A new exception is a decision, not a drift: it must change this line.
    assert set(RL_OVERLAP_RULED) == {"code"}


def test_the_known_prod_corpus_jobs_are_listed():
    named = {(r.source, r.start, r.count) for r in CORPUS_RANGES}
    assert ("reliquary_code_v1", 0, 100_000) in named
    assert ("openmathinstruct", 10_000_000, 400_000) in named
    assert ("reliquary_instruction_following_v1", 0, 20_000) in named
    assert ("reliquary_logic_v2", 100_000_000, 50_000) in named
    assert all(r.split == "train" for r in CORPUS_RANGES)


def test_rl_samples_the_whole_train_split():
    """What `rl_ranges` assumes, pinned: the window slice stays inside
    [0, len(env)) and every runtime factory builds the train split."""
    from reliquary.environment.agentic.external import load_external_backend
    from reliquary.shared.prompt_range import window_prompt_range

    rng = random.Random(3)
    for _ in range(200):
        lo, hi = window_prompt_range(f"{rng.random()}", "x", 2_481_806, 5000)
        assert 0 <= lo < hi <= 2_481_806
    assert inspect.signature(load_external_backend).parameters["split"].default == "train"
    for held in HELD_OUT.values():
        assert any(r.source == held.source and r.split == "train" and r.count is None
                   for r in rl_ranges())


def test_overlap_is_per_source_and_split():
    a = sets.UsedRange("corpus", "s", "train", 0, 10, "a")
    assert overlaps(a, sets.UsedRange("corpus", "s", "train", 9, 5, "b"))
    assert not overlaps(a, sets.UsedRange("corpus", "s", "train", 10, 5, "b"))
    assert not overlaps(a, sets.UsedRange("corpus", "s", "eval", 0, 10, "b"))
    assert not overlaps(a, sets.UsedRange("corpus", "t", "train", 0, 10, "b"))
    assert overlaps(a, sets.UsedRange("rl", "s", "train", 0, None, "rl"))


def test_a_corpus_job_cannot_take_the_held_out_code_rows():
    held = HELD_OUT["code"]
    with pytest.raises(ValueError, match="held out"):
        refuse_held_out_overlap("reliquary_code_v1", held.start + 5, 10)
    with pytest.raises(ValueError, match="held out"):
        refuse_held_out_overlap("reliquary_code_v1", held.start - 5, 10)
    refuse_held_out_overlap("reliquary_code_v1", 0, 100_000)
    # A train-split job never reaches a non-train held-out split.
    refuse_held_out_overlap("reliquary_logic_v2", 0, 10**9)


def test_jobs_create_refuses_the_held_out_rows():
    from reliquary.cli.main import build_job_manifest

    with pytest.raises(ValueError, match="held out"):
        build_job_manifest(
            job_id="order-x", checkpoint_repo="org/m", checkpoint_revision="a" * 40,
            checkpoint_sha256="b" * 64, prompt_source="reliquary_code_v1", prompt_count=10,
            prompt_start=HELD_OUT["code"].start, renderer_id="chat-template-v1",
            eos_token_id=1, slots_per_prompt=1, temperature=1.0, top_p=1.0, top_k=0,
            min_new_tokens=1, max_new_tokens=16, n=1, grader_id=None, threshold=None,
            prompt_order="free", deadline_round=None)


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_build_set_freezes_prompts_grading_and_its_card(tmp_path):
    open_environment = opener()
    card = build_set("logic", count=20, seed=5, out=tmp_path / "s",
                     open_environment=open_environment, clock=lambda: 1234.0)
    assert open_environment.opened == [("reliquary_logic_v2", "eval")]
    prompts = _lines(tmp_path / "s" / "prompts.jsonl")
    grading = _lines(tmp_path / "s" / "grading.jsonl")
    stored = json.loads((tmp_path / "s" / "set.json").read_text())
    assert stored == card
    assert card["set_id"] == "logic-eval-s5-n20"
    assert card["count"] == len(prompts) == len(grading) == 20
    assert card["created_at"] == 1234.0
    for name in ("prompts", "grading"):
        digest = hashlib.sha256((tmp_path / "s" / f"{name}.jsonl").read_bytes()).hexdigest()
        assert card[f"{name}_sha256"] == digest
    assert [p["problem_id"] for p in prompts] == [g["problem_id"] for g in grading]
    assert len({p["problem_id"] for p in prompts}) == 20
    first = prompts[0]
    assert set(first) == {"problem_id", "env", "set_id", "messages"}
    assert first["messages"][0]["role"] == "user"
    # The pod's file carries nothing the grader uses.
    assert all("source_index" not in p and "ground_truth" not in json.dumps(p) for p in prompts)
    indices = [g["source_index"] for g in grading]
    assert indices == random.Random(5).sample(range(0, 1000), 20)
    assert grading[0]["prompt_sha256"] == hashlib.sha256(
        first["messages"][0]["content"].encode()).hexdigest()
    assert card["disjointness"]["justification"] == HELD_OUT["logic"].justification
    assert card["disjointness"]["rl"].startswith("disjoint")


def test_build_set_is_reproducible(tmp_path):
    a = build_set("math", count=5, seed=1, out=tmp_path / "a", open_environment=opener(),
                  clock=lambda: 1.0)
    b = build_set("math", count=5, seed=1, out=tmp_path / "b", open_environment=opener(),
                  clock=lambda: 1.0)
    assert a["prompts_sha256"] == b["prompts_sha256"]
    assert a["grading_sha256"] == b["grading_sha256"]


def test_code_set_stays_in_its_tail_and_says_rl_was_eligible(tmp_path):
    held = HELD_OUT["code"]
    card = build_set("code", count=10, seed=2, out=tmp_path / "c",
                     open_environment=opener(held.source_length), clock=lambda: 1.0)
    grading = _lines(tmp_path / "c" / "grading.jsonl")
    assert all(held.start <= g["source_index"] < held.start + held.count for g in grading)
    assert card["index_range"] == [held.start, held.start + held.count]
    assert card["disjointness"]["rl"].startswith("ruled")


def test_build_set_refuses_a_source_of_another_length(tmp_path):
    with pytest.raises(ValueError, match="length"):
        build_set("code", count=10, seed=2, out=tmp_path / "c", open_environment=opener(5000))


def test_build_set_refuses_more_problems_than_the_region(tmp_path):
    with pytest.raises(ValueError, match="holds"):
        build_set("logic", count=2000, seed=2, out=tmp_path / "c", open_environment=opener())


def test_build_set_refuses_an_existing_directory_with_files(tmp_path):
    build_set("logic", count=2, seed=2, out=tmp_path / "c", open_environment=opener())
    with pytest.raises(FileExistsError):
        build_set("logic", count=2, seed=2, out=tmp_path / "c", open_environment=opener())


def test_disjointness_follows_source_lineage():
    held = HELD_OUT["code"]
    # opencodeinstruct is the same curation, row for row.
    with pytest.raises(ValueError, match="held out"):
        refuse_held_out_overlap("opencodeinstruct", held.start, 10)
    refuse_held_out_overlap("opencodeinstruct", 0, 100_000)
    rl_oci = sets.UsedRange("rl", "opencodeinstruct", "train", 0, None, "rl")
    assert overlaps(held.region, rl_oci)
    # Sources of one corpus with different index spaces are taken to overlap.
    a = sets.UsedRange("corpus", "x", "train", 0, 1, "a")
    b = sets.UsedRange("corpus", "y", "eval", 99, 1, "b")
    original = dict(sets.SOURCE_LINEAGE)
    try:
        sets.SOURCE_LINEAGE.update({"x": ("C", "x"), "y": ("C", "y")})
        assert overlaps(a, b)
    finally:
        sets.SOURCE_LINEAGE.clear()
        sets.SOURCE_LINEAGE.update(original)


def test_the_code_card_carries_its_contamination_note(tmp_path):
    held = HELD_OUT["code"]
    card = build_set("code", count=2, seed=2, out=tmp_path / "c",
                     open_environment=opener(held.source_length), clock=lambda: 1.0)
    assert "inflated" in card["disjointness"]["contamination_note"]
    assert "opencodeinstruct" in card["disjointness"]["rl"]
    logic = build_set("logic", count=2, seed=2, out=tmp_path / "l",
                      open_environment=opener(), clock=lambda: 1.0)
    assert logic["disjointness"]["contamination_note"] is None
