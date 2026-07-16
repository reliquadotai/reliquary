from __future__ import annotations

import json

import pytest

from scripts.screen_recovery_checkpoints import (
    _source_revision,
    _token_repetition,
    resolve_model_source,
    select_code_tasks,
    select_tasks,
    summarize,
)


def test_source_revision_reads_mounted_checkout_with_explicit_safe_directory(
    tmp_path, monkeypatch
):
    calls = []

    class Completed:
        stdout = "a" * 40 + "\n"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(
        "scripts.screen_recovery_checkpoints.subprocess.run", fake_run
    )

    assert _source_revision(tmp_path) == "a" * 40
    assert calls[0][0][:3] == [
        "git",
        "-c",
        f"safe.directory={tmp_path.resolve()}",
    ]
    assert calls[0][1]["cwd"] == tmp_path.resolve()


def test_select_tasks_is_order_independent_and_revision_bound(tmp_path):
    rows = [
        {
            "unique_id": f"task-{index}",
            "problem": f"problem {index}",
            "answer": str(index),
            "subject": "Algebra",
            "level": 1,
        }
        for index in range(8)
    ]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(json.dumps(row) for row in reversed(rows)) + "\n",
        encoding="utf-8",
    )

    selected_a = select_tasks(first, n_prompts=4, dataset_revision="a" * 40)
    selected_b = select_tasks(second, n_prompts=4, dataset_revision="a" * 40)
    selected_c = select_tasks(first, n_prompts=4, dataset_revision="b" * 40)
    selected_offset = select_tasks(
        first,
        n_prompts=4,
        dataset_revision="a" * 40,
        prompt_offset=4,
    )

    assert selected_a == selected_b
    assert [row["task_id"] for row in selected_a] != [
        row["task_id"] for row in selected_c
    ]
    assert not {
        row["task_id"] for row in selected_a
    } & {row["task_id"] for row in selected_offset}


def test_select_code_tasks_is_revision_bound_and_materializes_only_holdout():
    class FakeEnvironment:
        def __init__(self):
            self.requested = []

        def __len__(self):
            return 20

        def get_problem(self, index):
            self.requested.append(index)
            return {
                "id": f"id-{index}",
                "prompt": f"prompt {index}",
                "ground_truth": f"cases-{index}",
            }

    first = FakeEnvironment()
    second = FakeEnvironment()
    offset_environment = FakeEnvironment()
    selected_a = select_code_tasks(
        first, n_prompts=4, dataset_revision="a" * 40
    )
    selected_b = select_code_tasks(
        second, n_prompts=4, dataset_revision="b" * 40
    )
    selected_offset = select_code_tasks(
        offset_environment,
        n_prompts=4,
        dataset_revision="a" * 40,
        prompt_offset=4,
    )

    assert len(selected_a) == 4
    assert len(first.requested) == 4
    assert len(offset_environment.requested) == 4
    assert [row["task_id"] for row in selected_a] != [
        row["task_id"] for row in selected_b
    ]
    assert not {
        row["task_id"] for row in selected_a
    } & {row["task_id"] for row in selected_offset}


def test_holdout_selection_rejects_out_of_bounds_ranges(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "unique_id": "only",
                "problem": "p",
                "answer": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset bounds"):
        select_tasks(
            path,
            n_prompts=1,
            dataset_revision="a" * 40,
            prompt_offset=1,
        )


def test_summarize_uses_stable_sample_zero_and_best_of_k():
    samples = [
        {
            "task_id": "a",
            "sample_index": 1,
            "reward": 1.0,
            "terminated": True,
            "forced": False,
            "boxed": True,
            "rambling_proxy": False,
            "completion_length": 20,
        },
        {
            "task_id": "a",
            "sample_index": 0,
            "reward": 0.0,
            "terminated": True,
            "forced": True,
            "boxed": False,
            "rambling_proxy": False,
            "completion_length": 30,
        },
        {
            "task_id": "b",
            "sample_index": 0,
            "reward": 1.0,
            "terminated": False,
            "forced": False,
            "boxed": True,
            "rambling_proxy": True,
            "completion_length": 40,
        },
        {
            "task_id": "b",
            "sample_index": 1,
            "reward": 0.0,
            "terminated": True,
            "forced": False,
            "boxed": True,
            "rambling_proxy": False,
            "completion_length": 10,
        },
    ]

    report = summarize(samples, 2)

    assert report["pass_at_1"] == 0.5
    assert report["pass_at_k"] == 1.0
    assert report["pass_average"] == 0.5
    assert report["termination_rate"] == 0.75
    assert report["forced_rate"] == 0.25
    assert report["rambling_proxy_rate"] == 0.25
    assert report["p50_completion_length"] == 25
    assert report["p95_completion_length"] == pytest.approx(38.5)


def test_token_repetition_detects_runs_and_repeated_ngrams():
    repeated_ratio, max_run = _token_repetition([1] * 12)
    assert repeated_ratio > 0.8
    assert max_run == 12

    diverse_ratio, diverse_run = _token_repetition(list(range(12)))
    assert diverse_ratio == 0.0
    assert diverse_run == 1


def test_resolve_model_source_accepts_local_candidate(tmp_path):
    source, kwargs, identity = resolve_model_source(
        model_repo=None,
        model_revision=None,
        model_path=tmp_path,
    )

    assert source == str(tmp_path.resolve())
    assert kwargs == {}
    assert identity == {
        "kind": "local",
        "repo": None,
        "revision": None,
        "path": str(tmp_path.resolve()),
    }


def test_resolve_model_source_requires_pinned_hub_revision():
    with pytest.raises(ValueError, match="require --model-repo"):
        resolve_model_source(
            model_repo="owner/model",
            model_revision=None,
            model_path=None,
        )
