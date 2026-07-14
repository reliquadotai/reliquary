"""Verified, index-last publication of evaluation evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from reliquary.eval_dashboard.config import (
    canonical_json_bytes,
    config_hash as hash_config,
    sha256_bytes,
    validate_contamination_reviews,
)
from reliquary.eval_dashboard.metrics import build_dashboard
from reliquary.eval_dashboard.holdout import task_ids_sha256
from reliquary.eval_dashboard.models import (
    ArtifactReference,
    CheckpointResult,
    DomainCoverage,
    EvalConfig,
    EvalConfigManifest,
    EvalPublicationManifest,
    PublishedCheckpointArtifact,
)
from reliquary.eval_dashboard.runner import derive_sample_seed
from reliquary.eval_dashboard.store import (
    ObjectAlreadyExists,
    ObjectNotFound,
    ObjectStore,
)


INDEX_KEY = "eval_dashboard/index.json"
STATUS_KEY = "eval_dashboard/status.json"


class ImmutableArtifactConflict(RuntimeError):
    pass


def timestamp_pair(now: float) -> tuple[float, str]:
    return now, datetime.fromtimestamp(now, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _read_json(store: ObjectStore, key: str) -> Any:
    return json.loads(store.get(key))


def _write_verified(
    store: ObjectStore,
    key: str,
    payload: bytes,
    *,
    immutable: bool,
) -> str:
    digest = sha256_bytes(payload)
    if immutable:
        try:
            current = store.get(key)
        except ObjectNotFound:
            current = None
        if current is not None:
            if current != payload:
                raise ImmutableArtifactConflict(
                    f"immutable artifact already exists with different bytes: {key}"
                )
            return digest
    try:
        store.put(
            key,
            payload,
            content_type="application/json",
            metadata={"sha256": digest},
            if_absent=immutable,
        )
    except ObjectAlreadyExists:
        current = store.get(key)
        if current != payload:
            raise ImmutableArtifactConflict(
                f"immutable artifact won a concurrent write with different bytes: {key}"
            )
        return digest
    readback = store.get(key)
    if readback != payload or sha256_bytes(readback) != digest:
        raise IOError(f"object readback verification failed: {key}")
    return digest


def _result_key(result: CheckpointResult) -> str:
    return (
        f"eval_dashboard/runs/{result.config_hash}/checkpoints/"
        f"checkpoint-{result.checkpoint_n:06d}-{result.model_revision}.json"
    )


def _manifest_key(config_hash: str) -> str:
    return f"eval_dashboard/runs/{config_hash}/manifest.json"


def _dashboard_compat_key(config_hash: str) -> str:
    return f"eval_dashboard/runs/{config_hash}/dashboard.json"


def _publication_manifest_key(config_hash: str, publication_id: str) -> str:
    return (
        f"eval_dashboard/runs/{config_hash}/publications/{publication_id}/manifest.json"
    )


def _load_index(store: ObjectStore) -> dict[str, Any]:
    try:
        value = _read_json(store, INDEX_KEY)
    except ObjectNotFound:
        return {"schema_version": "2", "runs": []}
    if not isinstance(value, dict):
        raise ValueError("eval dashboard index is not an object")
    runs = value.get("runs", [])
    if not isinstance(runs, list) or any(not isinstance(item, dict) for item in runs):
        raise ValueError("eval dashboard index runs must be an object array")
    return dict(value)


def load_checkpoint_results(
    store: ObjectStore, config_hash: str
) -> list[CheckpointResult]:
    prefix = f"eval_dashboard/runs/{config_hash}/checkpoints/"
    results: list[CheckpointResult] = []
    for key in store.list(prefix):
        if not key.endswith(".json"):
            continue
        result = CheckpointResult.model_validate_json(store.get(key))
        if result.config_hash != config_hash:
            raise ValueError(f"checkpoint artifact crossed config lineage: {key}")
        results.append(result)

    by_checkpoint: dict[int, CheckpointResult] = {}
    for result in sorted(
        results, key=lambda item: (item.checkpoint_n, item.model_revision)
    ):
        previous = by_checkpoint.get(result.checkpoint_n)
        if previous is not None and previous.model_revision != result.model_revision:
            raise ImmutableArtifactConflict(
                f"checkpoint {result.checkpoint_n} has two immutable revisions"
            )
        by_checkpoint[result.checkpoint_n] = result
    return list(by_checkpoint.values())


def _validate_result_contract(
    result: CheckpointResult,
    config: EvalConfig,
    effective_config: dict[str, Any],
    effective_sha: str,
) -> None:
    if result.config_hash != effective_sha or result.config_sha256 != effective_sha:
        raise ValueError("checkpoint result does not match effective config")
    if result.lineage_id != config.lineage.lineage_id:
        raise ValueError("checkpoint result does not match configured lineage")
    if result.hardware != effective_config.get("hardware"):
        raise ValueError("checkpoint hardware does not match effective config")
    expected_runtime = effective_config.get("runtime")
    if not isinstance(expected_runtime, dict) or any(
        result.runtime.get(key) != value for key, value in expected_runtime.items()
    ):
        raise ValueError("checkpoint runtime does not match effective config")
    if result.runtime.get("tokenizer") != effective_config.get("tokenizer"):
        raise ValueError("checkpoint tokenizer does not match effective config")
    holdout_specs = {
        "math": config.math_holdout,
        "code": config.code_holdout,
    }
    for domain, holdout in holdout_specs.items():
        domain_result = result.domains[domain]
        if domain_result.n_prompts != holdout.n_prompts:
            raise ValueError(
                f"partial {domain} result cannot be published: expected "
                f"{holdout.n_prompts}, got {domain_result.n_prompts}"
            )
        if domain_result.samples_per_prompt != config.generation.samples_per_prompt:
            raise ValueError(f"{domain} sample count does not match config")
        task_ids = [task.task_id for task in domain_result.tasks]
        if task_ids_sha256(task_ids) != holdout.task_ids_sha256:
            raise ValueError(f"{domain} result does not match the locked task ids")
        for task in domain_result.tasks:
            for sample in task.samples:
                expected_seed = derive_sample_seed(
                    config.generation.seed_salt,
                    effective_sha,
                    task.task_id,
                    sample.sample_index,
                )
                if sample.seed != expected_seed:
                    raise ValueError(
                        f"{domain} result seed does not match the locked contract"
                    )

    allowed_repo = (
        config.lineage.base.repo_id
        if result.checkpoint_n == config.lineage.base.checkpoint_n
        else config.lineage.checkpoint_repo_id
    )
    if result.model_repo_id != allowed_repo:
        raise ValueError("checkpoint result model repository is outside the lineage")
    if (
        result.checkpoint_n == config.lineage.base.checkpoint_n
        and result.model_revision != config.lineage.base.revision
    ):
        raise ValueError("base checkpoint revision does not match the lineage lock")


class EvalPublisher:
    def __init__(self, store: ObjectStore) -> None:
        self.store = store

    def publish(
        self,
        *,
        result: CheckpointResult,
        config: EvalConfig,
        effective_config: dict[str, Any],
        now: float,
    ) -> dict[str, Any]:
        """Publish a complete checkpoint and move the discovery pointer last."""
        effective_sha = hash_config(effective_config)
        if effective_config.get("declared") != config.model_dump(mode="json"):
            raise ValueError(
                "effective config does not contain the declared eval config"
            )
        reviews = effective_config.get("contamination_reviews")
        if not isinstance(reviews, dict):
            raise ValueError("effective config has no contamination reviews")
        validate_contamination_reviews(config, reviews)
        _validate_result_contract(result, config, effective_config, effective_sha)

        config_manifest = EvalConfigManifest(
            config_hash=effective_sha,
            config_sha256=effective_sha,
            lineage_id=config.lineage.lineage_id,
            effective_config=effective_config,
        )
        config_manifest_payload = canonical_json_bytes(
            config_manifest.model_dump(mode="json")
        )
        config_manifest_key = _manifest_key(effective_sha)
        config_manifest_sha = _write_verified(
            self.store,
            config_manifest_key,
            config_manifest_payload,
            immutable=True,
        )

        result_payload = canonical_json_bytes(result.model_dump(mode="json"))
        result_key = _result_key(result)
        _write_verified(
            self.store,
            result_key,
            result_payload,
            immutable=True,
        )

        results = load_checkpoint_results(self.store, effective_sha)
        for stored in results:
            _validate_result_contract(stored, config, effective_config, effective_sha)
        base_result = next(
            (
                stored
                for stored in results
                if stored.checkpoint_n == config.lineage.base.checkpoint_n
                and stored.model_repo_id == config.lineage.base.repo_id
                and stored.model_revision == config.lineage.base.revision
            ),
            None,
        )
        if base_result is None:
            raise ValueError(
                "the exact locked base must be evaluated before publication"
            )
        latest = max(results, key=lambda item: item.checkpoint_n)
        evidence_timestamp = latest.completed_at
        evidence_timestamp_iso = latest.completed_at_iso
        timestamp, timestamp_iso = timestamp_pair(now)
        dashboard = build_dashboard(
            results,
            generated_at=timestamp,
            generated_at_iso=timestamp_iso,
            evidence_completed_at=evidence_timestamp,
            evidence_completed_at_iso=evidence_timestamp_iso,
            config_hash=effective_sha,
            config_sha256=effective_sha,
            lineage_id=config.lineage.lineage_id,
            checkpoint_repo_id=config.lineage.checkpoint_repo_id,
            base_checkpoint_n=config.lineage.base.checkpoint_n,
            latest_checkpoint_n=latest.checkpoint_n,
            latest_checkpoint_revision=latest.model_revision,
            latest_model_repo_id=latest.model_repo_id,
            publish_interval_windows=config.publish_interval_windows,
        )
        dashboard_payload = canonical_json_bytes(dashboard)
        dashboard_sha = sha256_bytes(dashboard_payload)
        publication_id = dashboard_sha
        snapshot_key = (
            f"eval_dashboard/runs/{effective_sha}/publications/"
            f"{publication_id}/dashboard.json"
        )
        _write_verified(
            self.store,
            snapshot_key,
            dashboard_payload,
            immutable=True,
        )

        checkpoint_artifacts: list[PublishedCheckpointArtifact] = []
        for stored in sorted(results, key=lambda item: item.checkpoint_n):
            stored_key = _result_key(stored)
            stored_payload = canonical_json_bytes(stored.model_dump(mode="json"))
            stored_sha = sha256_bytes(stored_payload)
            if self.store.get(stored_key) != stored_payload:
                raise IOError(
                    f"checkpoint artifact changed before publication: {stored_key}"
                )
            checkpoint_artifacts.append(
                PublishedCheckpointArtifact(
                    result=ArtifactReference(key=stored_key, sha256=stored_sha),
                    checkpoint_n=stored.checkpoint_n,
                    model_repo_id=stored.model_repo_id,
                    model_revision=stored.model_revision,
                    observed_window=stored.observed_window,
                    started_at=stored.started_at,
                    completed_at=stored.completed_at,
                    completed_at_iso=stored.completed_at_iso,
                    duration_seconds=stored.duration_seconds,
                    hardware=stored.hardware,
                    runtime=stored.runtime,
                    domains={
                        domain: DomainCoverage(
                            n_prompts=value.n_prompts,
                            samples_per_prompt=value.samples_per_prompt,
                            pass_at_1=value.pass_at_1,
                            pass_at_k=value.pass_at_k,
                            pass_avg=value.pass_avg,
                            trunc_pct=value.trunc_pct,
                            forced_pct=value.forced_pct,
                        )
                        for domain, value in stored.domains.items()
                    },
                )
            )

        publication_manifest = EvalPublicationManifest(
            publication_id=publication_id,
            config_hash=effective_sha,
            config_sha256=effective_sha,
            lineage_id=config.lineage.lineage_id,
            generated_at=timestamp,
            generated_at_iso=timestamp_iso,
            evidence_completed_at=evidence_timestamp,
            evidence_completed_at_iso=evidence_timestamp_iso,
            base_checkpoint_n=config.lineage.base.checkpoint_n,
            latest_checkpoint_n=latest.checkpoint_n,
            latest_checkpoint_revision=latest.model_revision,
            latest_model_repo_id=latest.model_repo_id,
            config_manifest=ArtifactReference(
                key=config_manifest_key,
                sha256=config_manifest_sha,
            ),
            dashboard=ArtifactReference(key=snapshot_key, sha256=dashboard_sha),
            checkpoints=checkpoint_artifacts,
        )
        publication_manifest_payload = canonical_json_bytes(
            publication_manifest.model_dump(mode="json")
        )
        publication_manifest_key = _publication_manifest_key(
            effective_sha, publication_id
        )
        publication_manifest_sha = _write_verified(
            self.store,
            publication_manifest_key,
            publication_manifest_payload,
            immutable=True,
        )

        # Compatibility for the existing consumer. The immutable snapshot key
        # in index.json is authoritative; the web consumer can switch to it
        # without changing the historical run layout.
        compatibility_key = _dashboard_compat_key(effective_sha)
        _write_verified(
            self.store,
            compatibility_key,
            dashboard_payload,
            immutable=False,
        )

        # Verify every object referenced by the pointer immediately before the
        # index update. A failure above leaves index.json unchanged.
        if sha256_bytes(self.store.get(config_manifest_key)) != config_manifest_sha:
            raise IOError("config manifest changed before publication")
        if (
            sha256_bytes(self.store.get(publication_manifest_key))
            != publication_manifest_sha
        ):
            raise IOError("publication manifest changed before publication")
        if sha256_bytes(self.store.get(snapshot_key)) != dashboard_sha:
            raise IOError("dashboard snapshot changed before publication")

        index = _load_index(self.store)
        old_runs = list(index.get("runs", []))
        entry = {
            "config_hash": effective_sha,
            "config_sha256": effective_sha,
            "lineage_id": config.lineage.lineage_id,
            "base_model_repo_id": config.lineage.base.repo_id,
            "base_model_revision": config.lineage.base.revision,
            "checkpoint_repo_id": config.lineage.checkpoint_repo_id,
            "latest_checkpoint_n": latest.checkpoint_n,
            "latest_checkpoint_revision": latest.model_revision,
            "latest_model_repo_id": latest.model_repo_id,
            "manifest_key": publication_manifest_key,
            "manifest_sha256": publication_manifest_sha,
            "config_manifest_key": config_manifest_key,
            "config_manifest_sha256": config_manifest_sha,
            "dashboard_key": snapshot_key,
            "dashboard_sha256": dashboard_sha,
            "compatibility_dashboard_key": compatibility_key,
            "publication_id": publication_id,
            "updated_at": timestamp,
            "updated_at_iso": timestamp_iso,
        }
        runs = [item for item in old_runs if item.get("config_hash") != effective_sha]
        runs.append(entry)
        runs.sort(key=lambda item: float(item.get("updated_at", 0.0)), reverse=True)
        new_index = dict(index)
        new_index.update(
            {
                "schema_version": "2",
                "current_config_hash": effective_sha,
                "current_dashboard_key": snapshot_key,
                "current_manifest_key": publication_manifest_key,
                "updated_at": timestamp,
                "updated_at_iso": timestamp_iso,
                "runs": runs,
            }
        )
        index_payload = canonical_json_bytes(new_index)
        _write_verified(
            self.store,
            INDEX_KEY,
            index_payload,
            immutable=False,
        )
        return new_index

    def write_status(self, status: dict[str, Any]) -> None:
        _write_verified(
            self.store,
            STATUS_KEY,
            canonical_json_bytes(status),
            immutable=False,
        )
