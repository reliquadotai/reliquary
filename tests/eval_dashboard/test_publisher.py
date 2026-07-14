import json
from datetime import datetime, timezone

import pytest

from reliquary.eval_dashboard.config import (
    canonical_json_bytes,
    config_hash,
    sha256_bytes,
)
from reliquary.eval_dashboard.holdout import task_ids_sha256
from reliquary.eval_dashboard.metrics import summarize_domain
from reliquary.eval_dashboard.models import (
    CheckpointResult,
    EvalConfig,
    EvalPublicationManifest,
    SampleResult,
    TaskResult,
)
from reliquary.eval_dashboard.publisher import (
    EvalPublisher,
    ImmutableArtifactConflict,
    INDEX_KEY,
)
from reliquary.eval_dashboard.runner import derive_sample_seed
from reliquary.eval_dashboard.store import ObjectNotFound
from reliquary.eval_dashboard.worker import check_freshness


REV_A = "a" * 40
REV_B = "b" * 40
REV_C = "c" * 40
ZERO = "0" * 64
TASK_IDS_SHA = task_ids_sha256(["t"])


def _review(domain):
    return {
        "schema_version": "1",
        "domain": domain,
        "holdout_sha256": ZERO,
        "task_ids_sha256": TASK_IDS_SHA,
        "reviewed_at": "2026-07-14T00:00:00Z",
        "reviewer": "test-reviewer",
        "method": "normalized exact hash and token shingle review",
        "training_sources": [
            {
                "repo_id": "training/source",
                "revision": REV_A,
                "prompt_field": "prompt",
                "artifacts": [{"sha256": ZERO, "n_prompts": 100}],
            }
        ],
        "exact_overlap_count": 0,
        "near_duplicate_count": 0,
        "decision": "approved",
        "notes": "",
    }


def _review_sha(domain):
    return sha256_bytes(canonical_json_bytes(_review(domain)))


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.fail_key = None

    def get(self, key):
        if key not in self.objects:
            raise ObjectNotFound(key)
        return self.objects[key]

    def put(self, key, payload, *, content_type, metadata=None, if_absent=False):
        del content_type, metadata
        if key == self.fail_key:
            raise IOError("injected put failure")
        if if_absent and key in self.objects:
            from reliquary.eval_dashboard.store import ObjectAlreadyExists

            raise ObjectAlreadyExists(key)
        self.objects[key] = payload

    def list(self, prefix):
        return sorted(key for key in self.objects if key.startswith(prefix))


def _config():
    return EvalConfig.model_validate(
        {
            "lineage": {
                "lineage_id": "qwen35-2b-v2",
                "base": {
                    "repo_id": "Qwen/Qwen3.5-2B",
                    "revision": REV_A,
                    "checkpoint_n": 0,
                },
                "checkpoint_repo_id": "ReliquaryForge/model-v2",
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
                "artifact_sha256": ZERO,
                "task_ids_sha256": TASK_IDS_SHA,
                "contamination_review_sha256": _review_sha("math"),
                "n_prompts": 1,
                "grader_id": "reliquary.openmath",
                "grader_revision": REV_C,
            },
            "code_holdout": {
                "domain": "code",
                "dataset_repo_id": "ReliquaryForge/eval-v2",
                "dataset_revision": REV_A,
                "split": "code",
                "artifact_sha256": ZERO,
                "task_ids_sha256": TASK_IDS_SHA,
                "contamination_review_sha256": _review_sha("code"),
                "n_prompts": 1,
                "grader_id": "reliquary.opencode",
                "grader_revision": REV_C,
            },
            "generation": {
                "samples_per_prompt": 1,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 0,
                "repetition_penalty": 1,
                "batch_size": 1,
                "seed_salt": "reliquary-eval-v2-seed",
                "math_max_new_tokens": 32768,
                "math_bft_enabled": True,
                "math_thinking_budget": 2048,
                "math_answer_budget": 512,
                "math_force_template": "</think>\n\nFinal Answer: \\boxed{",
                "code_max_new_tokens": 32768,
            },
            "publish_interval_windows": 10,
            "schedule": {
                "owner": "reliquary-validator-ops",
                "cadence_seconds": 3600,
                "overdue_seconds": 21600,
            },
        }
    )


def _checkpoint(effective_hash, n, revision, completed_at):
    sample = SampleResult(
        sample_index=0,
        seed=derive_sample_seed("reliquary-eval-v2-seed", effective_hash, "t", 0),
        reward=1,
        completion_length=10,
        terminated=True,
        forced=False,
        completion_sha256="1" * 64,
        duration_seconds=1,
    )
    task = TaskResult(task_id="t", prompt_sha256="2" * 64, samples=[sample])
    return CheckpointResult(
        config_hash=effective_hash,
        config_sha256=effective_hash,
        lineage_id="qwen35-2b-v2",
        model_repo_id=("Qwen/Qwen3.5-2B" if n == 0 else "ReliquaryForge/model-v2"),
        model_revision=revision,
        checkpoint_n=n,
        observed_window=200 + n,
        started_at=completed_at - 10,
        completed_at=completed_at,
        completed_at_iso=datetime.fromtimestamp(completed_at, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        duration_seconds=10,
        hardware={"device": "test", "gpu_name": "Test GPU"},
        runtime={
            "producer_revision": REV_C,
            "tokenizer": {"chat_template_sha256": "f" * 64},
        },
        domains={
            "math": summarize_domain("math", [task]),
            "code": summarize_domain("code", [task]),
        },
    )


def _effective(config=None):
    config = config or _config()
    return {
        "schema_version": "1",
        "declared": config.model_dump(mode="json"),
        "tokenizer": {"chat_template_sha256": "f" * 64},
        "runtime": {"producer_revision": REV_C},
        "hardware": {"device": "test", "gpu_name": "Test GPU"},
        "contamination_reviews": {
            "math": _review("math"),
            "code": _review("code"),
        },
        "contract": "test",
    }


def test_publish_is_index_last_and_preserves_history():
    store = MemoryStore()
    store.objects[INDEX_KEY] = json.dumps(
        {
            "schema_version": "1",
            "current_config_hash": "legacy",
            "runs": [{"config_hash": "legacy", "updated_at": 10}],
        }
    ).encode()
    effective = _effective()
    effective_hash = config_hash(effective)
    publisher = EvalPublisher(store)
    publisher.publish(
        result=_checkpoint(effective_hash, 0, REV_A, 1000),
        config=_config(),
        effective_config=effective,
        now=1100,
    )
    index = json.loads(store.objects[INDEX_KEY])
    assert index["current_config_hash"] == effective_hash
    assert index["updated_at_iso"].endswith("Z")
    assert {run["config_hash"] for run in index["runs"]} == {"legacy", effective_hash}
    current = next(run for run in index["runs"] if run["config_hash"] == effective_hash)
    assert store.get(current["dashboard_key"])
    dashboard = json.loads(store.get(current["dashboard_key"]))
    assert dashboard["generated_at"] == 1100
    assert dashboard["evidence_completed_at"] == 1000
    assert current["manifest_key"].endswith("/manifest.json")
    assert current["config_manifest_key"].endswith("/manifest.json")


def test_partial_publication_does_not_advance_index():
    store = MemoryStore()
    effective = _effective()
    effective_hash = config_hash(effective)
    publisher = EvalPublisher(store)
    publisher.publish(
        result=_checkpoint(effective_hash, 0, REV_A, 1000),
        config=_config(),
        effective_config=effective,
        now=1100,
    )
    previous_index = store.objects[INDEX_KEY]
    # The snapshot key is content-addressed and written before compatibility
    # dashboard/index. Injecting this failure must leave discovery unchanged.
    result = _checkpoint(effective_hash, 1, REV_B, 2000)
    store.fail_key = (
        f"eval_dashboard/runs/{effective_hash}/publications/"
        # Unknown content hash: intercept by replacing put for publication keys.
        "never-matches"
    )
    original_put = store.put

    def fail_publication(key, payload, *, content_type, metadata=None, if_absent=False):
        if "/publications/" in key:
            raise IOError("injected snapshot failure")
        return original_put(
            key,
            payload,
            content_type=content_type,
            metadata=metadata,
            if_absent=if_absent,
        )

    store.put = fail_publication
    with pytest.raises(IOError, match="snapshot failure"):
        publisher.publish(
            result=result,
            config=_config(),
            effective_config=effective,
            now=2100,
        )
    assert store.objects[INDEX_KEY] == previous_index


def test_freshness_uses_evidence_time_and_checks_lineage():
    store = MemoryStore()
    effective = _effective()
    effective_hash = config_hash(effective)
    EvalPublisher(store).publish(
        result=_checkpoint(effective_hash, 0, REV_A, 1000),
        config=_config(),
        effective_config=effective,
        now=5000,
    )
    EvalPublisher(store).publish(
        result=_checkpoint(effective_hash, 1, REV_B, 2000),
        config=_config(),
        effective_config=effective,
        now=5001,
    )
    status = check_freshness(
        store,
        expected_repo_id="ReliquaryForge/model-v2",
        max_age_seconds=600,
        now=6002,
    )
    assert status["status"] == "overdue"
    assert status["age_seconds"] == 1001
    assert status["evidence_age_seconds"] == 4002
    with pytest.raises(RuntimeError, match="wrong model lineage"):
        check_freshness(
            store,
            expected_repo_id="other/model",
            max_age_seconds=600,
            now=1001,
        )
    with pytest.raises(RuntimeError, match="current checkpoint revision"):
        check_freshness(
            store,
            expected_repo_id="ReliquaryForge/model-v2",
            expected_revision=REV_C,
            max_age_seconds=600,
            now=5002,
        )


def test_publisher_rejects_partial_holdout_results():
    store = MemoryStore()
    config = _config().model_copy(
        update={
            "math_holdout": _config().math_holdout.model_copy(update={"n_prompts": 2})
        }
    )
    effective = _effective(config)
    effective_hash = config_hash(effective)
    with pytest.raises(ValueError, match="partial math result"):
        EvalPublisher(store).publish(
            result=_checkpoint(effective_hash, 0, REV_A, 1000),
            config=config,
            effective_config=effective,
            now=1100,
        )
    assert INDEX_KEY not in store.objects


def test_publication_manifest_records_artifact_runtime_and_coverage():
    store = MemoryStore()
    effective = _effective()
    effective_hash = config_hash(effective)
    index = EvalPublisher(store).publish(
        result=_checkpoint(effective_hash, 0, REV_A, 1000),
        config=_config(),
        effective_config=effective,
        now=1100,
    )
    current = next(run for run in index["runs"] if run["config_hash"] == effective_hash)
    manifest = EvalPublicationManifest.model_validate_json(
        store.get(current["manifest_key"])
    )
    assert manifest.dashboard.key == current["dashboard_key"]
    assert manifest.config_manifest.key == current["config_manifest_key"]
    assert manifest.checkpoints[0].hardware["gpu_name"] == "Test GPU"
    assert manifest.checkpoints[0].domains["math"].n_prompts == 1


def test_seed_mismatch_is_rejected_before_publication():
    store = MemoryStore()
    effective = _effective()
    effective_hash = config_hash(effective)
    result = _checkpoint(effective_hash, 0, REV_A, 1000)
    bad_sample = (
        result.domains["math"].tasks[0].samples[0].model_copy(update={"seed": 99})
    )
    bad_task = (
        result.domains["math"].tasks[0].model_copy(update={"samples": [bad_sample]})
    )
    bad_math = result.domains["math"].model_copy(update={"tasks": [bad_task]})
    result = result.model_copy(update={"domains": {**result.domains, "math": bad_math}})
    with pytest.raises(ValueError, match="seed does not match"):
        EvalPublisher(store).publish(
            result=result,
            config=_config(),
            effective_config=effective,
            now=1100,
        )
    assert INDEX_KEY not in store.objects


def test_publisher_rejects_unbound_contamination_review():
    store = MemoryStore()
    effective = _effective()
    effective["contamination_reviews"]["code"]["holdout_sha256"] = "9" * 64
    effective_hash = config_hash(effective)
    with pytest.raises(ValueError, match="different holdout artifact"):
        EvalPublisher(store).publish(
            result=_checkpoint(effective_hash, 0, REV_A, 1000),
            config=_config(),
            effective_config=effective,
            now=1100,
        )
    assert INDEX_KEY not in store.objects


def test_index_write_failure_keeps_previous_pointer():
    store = MemoryStore()
    effective = _effective()
    effective_hash = config_hash(effective)
    publisher = EvalPublisher(store)
    publisher.publish(
        result=_checkpoint(effective_hash, 0, REV_A, 1000),
        config=_config(),
        effective_config=effective,
        now=1100,
    )
    previous_index = store.objects[INDEX_KEY]
    store.fail_key = INDEX_KEY
    with pytest.raises(IOError, match="injected put failure"):
        publisher.publish(
            result=_checkpoint(effective_hash, 1, REV_B, 2000),
            config=_config(),
            effective_config=effective,
            now=2100,
        )
    assert store.objects[INDEX_KEY] == previous_index


def test_two_revisions_for_one_checkpoint_are_rejected():
    store = MemoryStore()
    effective = _effective()
    effective_hash = config_hash(effective)
    publisher = EvalPublisher(store)
    publisher.publish(
        result=_checkpoint(effective_hash, 0, REV_A, 1000),
        config=_config(),
        effective_config=effective,
        now=1100,
    )
    publisher.publish(
        result=_checkpoint(effective_hash, 1, REV_B, 2000),
        config=_config(),
        effective_config=effective,
        now=2100,
    )
    with pytest.raises(ImmutableArtifactConflict, match="two immutable revisions"):
        publisher.publish(
            result=_checkpoint(effective_hash, 1, REV_C, 2001),
            config=_config(),
            effective_config=effective,
            now=2101,
        )
