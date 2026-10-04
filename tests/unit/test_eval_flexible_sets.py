"""Sets built from any catalog source and range, their training overlap recorded."""

from __future__ import annotations

import json
import random

import pytest

from reliquary.eval import sets
from reliquary.eval.sets import build_set, build_source_set, range_overlaps, select_indices


class FakeEnvironment:
    def __init__(self, name: str, length: int) -> None:
        self.name, self._length = name, length

    def __len__(self) -> int:
        return self._length

    def get_problem(self, index: int) -> dict:
        return {"prompt": f"{self.name} problem {index}"}


def opener(length=1000):
    opened = []

    def open_environment(source, split):
        opened.append((source, split))
        return FakeEnvironment(source, length)

    open_environment.opened = opened
    return open_environment


def rows(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


def test_select_the_whole_range_in_order():
    assert select_indices(5, 9, sample=None, seed=None) == [5, 6, 7, 8]


def test_select_a_seeded_sample_of_the_range():
    picked = select_indices(100, 200, sample=10, seed=7)
    assert picked == random.Random(7).sample(range(100, 200), 10)
    assert picked == select_indices(100, 200, sample=10, seed=7)


def test_a_sample_needs_a_seed_and_fits_the_range():
    with pytest.raises(ValueError, match="seed"):
        select_indices(0, 10, sample=3, seed=None)
    with pytest.raises(ValueError, match="holds 10 rows, fewer than 11"):
        select_indices(0, 10, sample=11, seed=0)


def test_build_a_catalog_range(tmp_path):
    out = tmp_path / "set"
    open_environment = opener(length=50)
    card = build_source_set("reliquary_dapo_math_v1", split="eval", start=10, count=5,
                            out=out, open_environment=open_environment, clock=lambda: 1.0)
    assert open_environment.opened == [("reliquary_dapo_math_v1", "eval")]
    assert card["set_id"] == "reliquary_dapo_math_v1-eval-r10-n5"
    assert card["source_kind"] == "catalog"
    assert card["interaction"] == "single_turn"
    assert card["selection"] == {"start": 10, "count": 5, "sample": None, "seed": None}
    assert card["index_range"] == [10, 15] and card["count"] == 5
    assert card["env"] == "reliquary_dapo_math_v1" and card["split"] == "eval"
    prompts, grading = rows(out, "prompts.jsonl"), rows(out, "grading.jsonl")
    assert [g["source_index"] for g in grading] == [10, 11, 12, 13, 14]
    assert prompts[0]["messages"] == [
        {"role": "user", "content": "reliquary_dapo_math_v1 problem 10"}]
    assert prompts[0]["interaction"] == "single_turn"
    assert prompts[0]["problem_id"] == grading[0]["problem_id"]


def test_the_range_defaults_to_the_whole_split(tmp_path):
    card = build_source_set("reliquary_logic_v2", split="eval", out=tmp_path / "s",
                            open_environment=opener(length=7))
    assert card["index_range"] == [0, 7] and card["count"] == 7


def test_a_sampled_set_names_its_seed(tmp_path):
    card = build_source_set("reliquary_dapo_math_v1", split="eval", start=0, count=100,
                            sample=8, seed=3, out=tmp_path / "s",
                            open_environment=opener(length=100))
    assert card["set_id"] == "reliquary_dapo_math_v1-eval-r0-n100-k8-s3"
    assert card["count"] == 8
    indices = [g["source_index"] for g in rows(tmp_path / "s", "grading.jsonl")]
    assert indices == random.Random(3).sample(range(0, 100), 8)


def test_a_range_past_the_source_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"\[40, 60\) runs past .* \(50 rows\)"):
        build_source_set("reliquary_dapo_math_v1", split="eval", start=40, count=20,
                         out=tmp_path / "s", open_environment=opener(length=50))


def test_an_unknown_catalog_source_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a catalog environment"):
        build_source_set("no_such_env", out=tmp_path / "s", open_environment=opener())


def test_a_set_is_frozen_once(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "x").write_text("x")
    with pytest.raises(FileExistsError):
        build_source_set("reliquary_dapo_math_v1", split="eval", out=tmp_path / "s",
                         open_environment=opener())


def test_overlap_with_rl_is_recorded_not_refused(tmp_path):
    card = build_source_set("reliquary_dapo_math_v1", split="train", start=0, count=4,
                            out=tmp_path / "s", open_environment=opener(length=10))
    overlap = card["disjointness"]
    assert overlap["disjoint"] is False
    assert [r["what"] for r in overlap["rl"]] == ["RL prompt sampling"]
    assert overlap["note"].startswith("rows of this set were eligible for training")


def test_an_eval_split_is_disjoint_from_rl():
    overlap = range_overlaps("reliquary_dapo_math_v1", "eval", 0, 10)
    assert overlap["rl"] == [] and overlap["corpus"] == []
    assert overlap["disjoint"] is True


def test_overlap_names_the_corpus_job():
    overlap = range_overlaps("reliquary_code_v1", "train", 50_000, 50_010)
    assert [r["what"] for r in overlap["corpus"]] == ["code-qwen38-27b-v1"]
    # The code corpus's lineage covers opencodeinstruct too.
    assert range_overlaps("opencodeinstruct", "train", 50_000, 50_010)["corpus"]


def test_overlap_marks_a_held_out_region():
    overlap = range_overlaps("reliquary_logic_v2", "eval", 0, 5)
    assert [r["what"] for r in overlap["held_out"]] == ["logic"]


def test_presets_still_build_the_v2_set(tmp_path):
    """The platform's sets: the preset path is the old function, untouched."""
    card = build_set("logic", count=3, seed=1, out=tmp_path / "s",
                     open_environment=opener(), clock=lambda: 1.0)
    assert card["set_id"] == "logic-eval-s1-n3"
    assert "source_kind" not in card
    prompts = rows(tmp_path / "s", "prompts.jsonl")
    assert set(prompts[0]) == {"problem_id", "env", "set_id", "messages"}


def test_card_hashes_cover_the_written_files(tmp_path):
    import hashlib

    card = build_source_set("reliquary_dapo_math_v1", split="eval", count=3,
                            out=tmp_path / "s", open_environment=opener(length=3))
    for name, key in (("prompts.jsonl", "prompts_sha256"), ("grading.jsonl", "grading_sha256")):
        assert hashlib.sha256((tmp_path / "s" / name).read_bytes()).hexdigest() == card[key]
    assert json.loads((tmp_path / "s" / "set.json").read_text()) == card


def test_set_ids_stay_names():
    assert sets.validated_set_id("reliquary_dapo_math_v1-eval-r10-n5")
