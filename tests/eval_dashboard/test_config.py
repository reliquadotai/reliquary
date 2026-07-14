import json

import pytest

from reliquary.eval_dashboard.config import (
    build_effective_config,
    canonical_json_bytes,
    config_hash,
    sha256_bytes,
    sha256_file,
)
from reliquary.eval_dashboard.holdout import (
    load_locked_holdout,
    task_ids_sha256,
    write_canonical_jsonl,
)
from reliquary.eval_dashboard.models import (
    ContaminationReview,
    EvalConfig,
    HoldoutSpec,
    MathTask,
)


REV_A = "a" * 40
REV_B = "b" * 40
HASH_0 = "0" * 64


def _config(**generation_overrides) -> EvalConfig:
    reviews = _reviews()
    generation = {
        "samples_per_prompt": 4,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "batch_size": 1,
        "seed_salt": "reliquary-eval-v2-seed",
        "math_max_new_tokens": 32768,
        "math_bft_enabled": True,
        "math_thinking_budget": 2048,
        "math_answer_budget": 512,
        "math_force_template": "</think>\n\nFinal Answer: \\boxed{",
        "code_max_new_tokens": 32768,
    }
    generation.update(generation_overrides)
    return EvalConfig.model_validate(
        {
            "lineage": {
                "lineage_id": "qwen35-2b-v2",
                "base": {
                    "repo_id": "Qwen/Qwen3.5-2B",
                    "revision": REV_A,
                    "checkpoint_n": 0,
                },
                "checkpoint_repo_id": "ReliquaryForge/qwen3.5-2b-reliquary-v2",
            },
            "tokenizer_source": {
                "repo_id": "Qwen/Qwen3.5-2B",
                "revision": REV_A,
            },
            "math_holdout": {
                "domain": "math",
                "dataset_repo_id": "ReliquaryForge/eval-v2",
                "dataset_revision": REV_A,
                "split": "math",
                "artifact_sha256": HASH_0,
                "task_ids_sha256": HASH_0,
                "contamination_review_sha256": sha256_bytes(
                    canonical_json_bytes(reviews["math"].model_dump(mode="json"))
                ),
                "n_prompts": 2,
                "grader_id": "reliquary.openmath",
                "grader_revision": REV_B,
            },
            "code_holdout": {
                "domain": "code",
                "dataset_repo_id": "ReliquaryForge/eval-v2",
                "dataset_revision": REV_A,
                "split": "code",
                "artifact_sha256": HASH_0,
                "task_ids_sha256": HASH_0,
                "contamination_review_sha256": sha256_bytes(
                    canonical_json_bytes(reviews["code"].model_dump(mode="json"))
                ),
                "n_prompts": 2,
                "grader_id": "reliquary.opencode",
                "grader_revision": REV_B,
            },
            "generation": generation,
            "publish_interval_windows": 10,
            "schedule": {
                "owner": "reliquary-validator-ops",
                "cadence_seconds": 3600,
                "overdue_seconds": 21600,
            },
        }
    )


class _Tokenizer:
    chat_template = "{% if enable_thinking %}thinking{% endif %}"
    eos_token_id = 1
    pad_token_id = 1

    def __len__(self):
        return 151936


HARDWARE = {
    "device": "cuda:0",
    "cuda_available": True,
    "torch_cuda_version": "12.8",
    "gpu_name": "Test GPU",
    "gpu_total_memory_bytes": 1,
    "gpu_compute_capability": [9, 0],
}


def _reviews():
    source = {
        "repo_id": "nvidia/OpenMathInstruct-2",
        "revision": REV_A,
        "prompt_field": "problem",
        "artifacts": [{"sha256": HASH_0, "n_prompts": 100}],
    }
    return {
        domain: ContaminationReview(
            domain=domain,
            holdout_sha256=HASH_0,
            task_ids_sha256=HASH_0,
            reviewed_at="2026-07-14T00:00:00Z",
            reviewer="test-reviewer",
            method="normalized exact hash and token shingle review",
            training_sources=[source],
            exact_overlap_count=0,
            near_duplicate_count=0,
            decision="approved",
        )
        for domain in ("math", "code")
    }


def test_effective_config_hash_is_canonical(monkeypatch):
    monkeypatch.setattr(
        "reliquary.eval_dashboard.config._producer_revision",
        lambda repo_root=None: REV_B,
    )
    first = build_effective_config(
        _config(),
        tokenizer=_Tokenizer(),
        attention_implementation="sdpa",
        hardware=HARDWARE,
        contamination_reviews=_reviews(),
    )
    second = json.loads(canonical_json_bytes(first))
    assert config_hash(first) == config_hash(second)
    assert len(config_hash(first)) == 64
    assert first["tokenizer"]["chat_template_sha256"] != HASH_0
    assert first["runtime"]["producer_revision"] == REV_B
    different_gpu = build_effective_config(
        _config(),
        tokenizer=_Tokenizer(),
        attention_implementation="sdpa",
        hardware={**HARDWARE, "gpu_name": "Different GPU"},
        contamination_reviews=_reviews(),
    )
    assert config_hash(first) != config_hash(different_gpu)


def test_presence_penalty_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="presence_penalty"):
        _config(presence_penalty=1.5)


def test_protocol_parity_rejects_sampling_drift(monkeypatch):
    monkeypatch.setattr(
        "reliquary.eval_dashboard.config._producer_revision",
        lambda repo_root=None: REV_B,
    )
    with pytest.raises(ValueError, match="protocol-parity generation mismatch"):
        build_effective_config(
            _config(temperature=0.7),
            tokenizer=_Tokenizer(),
            attention_implementation="sdpa",
            hardware=HARDWARE,
            contamination_reviews=_reviews(),
        )


def test_protocol_parity_rejects_repetition_penalty(monkeypatch):
    monkeypatch.setattr(
        "reliquary.eval_dashboard.config._producer_revision",
        lambda repo_root=None: REV_B,
    )
    with pytest.raises(ValueError, match="repetition_penalty"):
        build_effective_config(
            _config(repetition_penalty=1.1),
            tokenizer=_Tokenizer(),
            attention_implementation="sdpa",
            hardware=HARDWARE,
            contamination_reviews=_reviews(),
        )


def test_effective_config_rejects_review_for_another_holdout(monkeypatch):
    monkeypatch.setattr(
        "reliquary.eval_dashboard.config._producer_revision",
        lambda repo_root=None: REV_B,
    )
    reviews = _reviews()
    reviews["math"] = reviews["math"].model_copy(update={"task_ids_sha256": "1" * 64})
    with pytest.raises(ValueError, match="different task-id list"):
        build_effective_config(
            _config(),
            tokenizer=_Tokenizer(),
            attention_implementation="sdpa",
            hardware=HARDWARE,
            contamination_reviews=reviews,
        )


def test_locked_holdout_requires_matching_approved_review(tmp_path):
    tasks = [
        MathTask(task_id="m1", prompt="1+1?", ground_truth="2"),
        MathTask(task_id="m2", prompt="2+2?", ground_truth="4"),
    ]
    holdout = tmp_path / "math.jsonl"
    write_canonical_jsonl(holdout, tasks)
    holdout_sha = sha256_file(holdout)
    ids_sha = task_ids_sha256(task.task_id for task in tasks)

    review = ContaminationReview(
        domain="math",
        holdout_sha256=holdout_sha,
        task_ids_sha256=ids_sha,
        reviewed_at="2026-07-14T00:00:00Z",
        reviewer="reliquary-validator-ops",
        method="normalized exact hash and MinHash review",
        training_sources=[
            {
                "repo_id": "nvidia/OpenMathInstruct-2",
                "revision": REV_A,
                "prompt_field": "problem",
                "artifacts": [{"sha256": HASH_0, "n_prompts": 100}],
            }
        ],
        exact_overlap_count=0,
        near_duplicate_count=0,
        decision="approved",
    )
    review_path = tmp_path / "math-review.json"
    review_path.write_bytes(canonical_json_bytes(review.model_dump(mode="json")))

    spec = HoldoutSpec(
        domain="math",
        dataset_repo_id="ReliquaryForge/eval-v2",
        dataset_revision=REV_A,
        split="math",
        artifact_sha256=holdout_sha,
        task_ids_sha256=ids_sha,
        contamination_review_sha256=sha256_file(review_path),
        n_prompts=2,
        grader_id="reliquary.openmath",
        grader_revision=REV_B,
    )
    loaded, loaded_review = load_locked_holdout(holdout, review_path, spec)
    assert [task.task_id for task in loaded] == ["m1", "m2"]
    assert loaded_review.decision == "approved"

    bad = review.model_copy(update={"near_duplicate_count": 1})
    review_path.write_bytes(canonical_json_bytes(bad.model_dump(mode="json")))
    bad_spec = spec.model_copy(
        update={"contamination_review_sha256": sha256_file(review_path)}
    )
    with pytest.raises(ValueError, match="overlap"):
        load_locked_holdout(holdout, review_path, bad_spec)


def test_locked_holdout_rejects_duplicate_prompts(tmp_path):
    tasks = [
        MathTask(task_id="m1", prompt="same prompt", ground_truth="1"),
        MathTask(task_id="m2", prompt="same prompt", ground_truth="2"),
    ]
    holdout = tmp_path / "math.jsonl"
    write_canonical_jsonl(holdout, tasks)
    ids_sha = task_ids_sha256(task.task_id for task in tasks)
    review = ContaminationReview(
        domain="math",
        holdout_sha256=sha256_file(holdout),
        task_ids_sha256=ids_sha,
        reviewed_at="2026-07-14T00:00:00Z",
        reviewer="reliquary-validator-ops",
        method="normalized exact hash and token shingle review",
        training_sources=[
            {
                "repo_id": "nvidia/OpenMathInstruct-2",
                "revision": REV_A,
                "prompt_field": "problem",
                "artifacts": [{"sha256": HASH_0, "n_prompts": 100}],
            }
        ],
        exact_overlap_count=0,
        near_duplicate_count=0,
        decision="approved",
    )
    review_path = tmp_path / "review.json"
    review_path.write_bytes(canonical_json_bytes(review.model_dump(mode="json")))
    spec = HoldoutSpec(
        domain="math",
        dataset_repo_id="ReliquaryForge/eval-v2",
        dataset_revision=REV_A,
        split="math",
        artifact_sha256=sha256_file(holdout),
        task_ids_sha256=ids_sha,
        contamination_review_sha256=sha256_file(review_path),
        n_prompts=2,
        grader_id="reliquary.openmath",
        grader_revision=REV_B,
    )
    with pytest.raises(ValueError, match="duplicate prompts"):
        load_locked_holdout(holdout, review_path, spec)
