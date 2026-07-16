from __future__ import annotations

import copy

import pytest

from scripts.compare_recovery_screens import (
    compare_screens,
    render_markdown,
    validate_paired_contract,
)


def _screen(label: str) -> dict:
    samples = []
    for task_index in range(2):
        for sample_index in range(2):
            samples.append({
                "task_id": f"task-{task_index}",
                "sample_index": sample_index,
                "reward": float(task_index == 0),
                "terminated": task_index == 0,
                "forced": task_index == 1,
                "rambling_proxy": False,
                "completion_length": 100 + task_index,
                "completion_sha256": f"hash-{task_index}-{sample_index}",
            })
    return {
        "checkpoint_label": label,
        "environment": "openmathinstruct",
        "tokenizer_repo": "owner/tokenizer",
        "tokenizer_revision": "a" * 40,
        "dataset_path": "/data/math.jsonl",
        "dataset_sha256": "b" * 64,
        "dataset_repo": None,
        "dataset_revision": "c" * 40,
        "n_prompts": 2,
        "prompt_offset": 0,
        "samples_per_prompt": 2,
        "thinking_budget": 2048,
        "answer_budget": 512,
        "seed_domain": "test",
        "attention_implementation": "flash_attention_2",
        "reliquary_revision": "d" * 40,
        "screen_script_sha256": "f" * 64,
        "runtime": {
            "python_version": "3.11.0",
            "gpu_name": "Test GPU",
            "gpu_compute_capability": [9, 0],
            "torch_version": "2.7.0",
            "cuda_version": "12.8",
            "cudnn_version": 90701,
            "transformers_version": "5.9.0",
            "flash_linear_attention_version": "0.5.0",
            "flash_attn_version": "2.8.3",
            "causal_conv1d_version": None,
            "bitsandbytes_version": "0.46.0",
        },
        "samples": samples,
    }


def test_paired_contract_rejects_runtime_or_sample_mismatch():
    baseline = _screen("base")
    candidate = _screen("candidate")
    candidate["thinking_budget"] = 1024

    with pytest.raises(ValueError, match="thinking_budget"):
        validate_paired_contract(baseline, candidate)

    candidate = _screen("candidate")
    candidate["samples"][0]["task_id"] = "different"
    with pytest.raises(ValueError, match="task/sample keys"):
        validate_paired_contract(baseline, candidate)

    candidate = _screen("candidate")
    candidate["runtime"]["torch_version"] = "different"
    with pytest.raises(ValueError, match="torch_version"):
        validate_paired_contract(baseline, candidate)

    candidate = _screen("candidate")
    candidate["runtime"]["flash_linear_attention_version"] = "different"
    with pytest.raises(ValueError, match="flash_linear_attention_version"):
        validate_paired_contract(baseline, candidate)

    legacy = _screen("legacy")
    legacy.pop("reliquary_revision")
    validate_paired_contract(baseline, legacy)

    candidate = _screen("candidate")
    candidate["reliquary_revision"] = "e" * 40
    with pytest.raises(ValueError, match="reliquary_revision"):
        validate_paired_contract(baseline, candidate)


def test_compare_screens_clusters_by_task_and_is_deterministic():
    baseline = _screen("base")
    candidate = copy.deepcopy(baseline)
    candidate["checkpoint_label"] = "candidate"
    for row in candidate["samples"]:
        row["reward"] = 1.0
        row["forced"] = False
        row["completion_length"] -= 10
    candidate["samples"][0]["completion_sha256"] = "changed"

    first = compare_screens(baseline, candidate, iterations=1_000, seed=7)
    second = compare_screens(baseline, candidate, iterations=1_000, seed=7)

    assert first == second
    assert first["task_clusters"] == 2
    assert first["metrics"]["pass_average"]["delta"] == 0.5
    assert first["metrics"]["forced_rate"]["delta"] == -0.5
    assert first["metrics"]["forced_rate"]["probability_favorable"] > 0.7
    assert first["metrics"]["mean_completion_length"]["delta"] == -10.0
    assert first["metrics"]["mean_completion_length"]["direction"] == "context_only"
    assert first["metrics"]["mean_completion_length"]["probability_favorable"] is None
    assert first["paired_sample_transitions"] == {
        "sample_count": 4,
        "reward_improved": 2,
        "reward_regressed": 0,
        "reward_unchanged": 2,
        "completion_hash_available": 4,
        "completion_hash_matches": 3,
        "completion_hash_match_rate": 0.75,
    }
    assert "candidate vs base" in render_markdown(first)
    assert "exact completion hash matches: 3/4 (0.750)" in render_markdown(
        first
    )
