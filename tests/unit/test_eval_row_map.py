"""`RowMap`: the one place a job row (`prompt_index`) becomes a task."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reliquary.eval import prompt_source as ps
from reliquary.eval import rows
from reliquary.eval.prompt_source import eval_source_for
from reliquary.eval.rows import RowMap, TaskRef, job_row_map, row_map


def _rows(split="benchmark", count=3):
    return [{"problem_id": f"p{i}", "set_id": "swe-verified", "env": "reliquary_agentic_swe_v1",
             "interaction": "swe", "split": split, "source_index": 41 + i,
             "instance_id": f"t-{41 + i}"} for i in range(count)]


def _body(rows_):
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows_)


@pytest.fixture
def served(monkeypatch):
    """A fetcher serving one set's body, counting its calls; no row or map kept."""
    calls = []
    monkeypatch.setattr(ps, "_loaded", {})
    rows._row_map.cache_clear()

    def serve(body):
        def fetch(source):
            calls.append(source)
            return body
        monkeypatch.setattr(ps, "FETCHERS", [fetch])
        return calls

    yield serve
    rows._row_map.cache_clear()


def _source(rows_):
    return eval_source_for("swe-verified", _body(rows_), len(rows_)).name


def test_identity_map_is_prompt_index():
    m = RowMap.identity("train", 100, 5)
    assert m.index(102) == 102 and m.ref(102).instance_id is None
    assert m.ref(100) == TaskRef("train", 100, None) and len(m) == 5 and m.split == "train"
    assert m.is_identity


def test_identity_map_refuses_outside():
    with pytest.raises(IndexError):
        RowMap.identity("train", 100, 5).ref(105)
    with pytest.raises(IndexError):
        RowMap.identity("train", 100, 5).ref(99)


def test_from_rows_maps_row_to_reference():
    m = RowMap.from_rows(_rows())
    assert m.ref(1) == TaskRef("benchmark", 42, "t-42") and m.index(2) == 43
    assert len(m) == 3 and m.split == "benchmark" and not m.is_identity
    with pytest.raises(IndexError):
        m.ref(3)
    with pytest.raises(IndexError):
        m.ref(-1)
    with pytest.raises(IndexError):
        m.ref("1")
    with pytest.raises(IndexError):
        m.ref(True)
    with pytest.raises(AttributeError):
        m.split = "train"


def test_from_rows_refuses_mixed_splits():
    mixed = _rows()
    mixed[2] = {**mixed[2], "split": "train"}
    with pytest.raises(ValueError, match="one split"):
        RowMap.from_rows(mixed)


def test_row_map_of_a_catalog_source_is_the_identity():
    m = row_map("reliquary_stateful_tools_v1", split="train", start=10, count=4)
    assert m.index(13) == 13 and len(m) == 4
    with pytest.raises(IndexError):
        m.ref(14)


def test_row_map_reads_an_eval_source(served):
    served(_body(_rows()))
    m = row_map(_source(_rows()), split="benchmark", start=0, count=3)
    assert m.ref(0) == TaskRef("benchmark", 41, "t-41")


def test_row_map_refuses_split_disagreement(served):
    served(_body(_rows(split="eval")))
    with pytest.raises(ValueError, match="'eval'.*'benchmark'|'benchmark'.*'eval'"):
        row_map(_source(_rows(split="eval")), split="benchmark", start=0, count=3)


def test_row_map_refuses_a_count_other_than_the_sources(served):
    served(_body(_rows()))
    with pytest.raises(ValueError, match="count"):
        row_map(_source(_rows()), split="benchmark", start=0, count=2)


def test_check_instances_names_first_mismatch():
    m = RowMap.from_rows(_rows())
    m.check_instances(lambda split, index: f"t-{index}")
    wrong = {42: "other", 43: "other"}
    with pytest.raises(ValueError, match="row 1"):
        m.check_instances(lambda split, index: wrong.get(index, f"t-{index}"))
    RowMap.identity("train", 0, 3).check_instances(lambda split, index: "never asked")


def test_job_row_map_memoised(served):
    calls = served(_body(_rows()))
    job = SimpleNamespace(job_id="order-swe-eval", prompt_source=_source(_rows()),
                          prompt_start=0, prompt_count=3,
                          episode=SimpleNamespace(env=SimpleNamespace(split="benchmark")))
    first = job_row_map(job)
    assert job_row_map(job) is first and len(calls) == 1
    assert first.ref(2) == TaskRef("benchmark", 43, "t-43")


def test_job_row_map_without_episode_is_train_identity():
    job = SimpleNamespace(job_id="math", prompt_source="openmathinstruct", prompt_start=7,
                          prompt_count=2, episode=None)
    m = job_row_map(job)
    assert m.split == "train" and m.index(8) == 8 and len(m) == 2


def test_row_map_of_an_eval_source_starts_at_row_zero(served):
    served(_body(_rows()))
    with pytest.raises(ValueError, match="start at 0"):
        row_map(_source(_rows()), split="benchmark", start=1, count=3)
