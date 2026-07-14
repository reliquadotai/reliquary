"""Checkpoint discovery, retries, status telemetry, and worker orchestration."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urljoin

import requests

from reliquary.eval_dashboard.config import (
    build_effective_config,
    canonical_json_bytes,
    config_hash,
    sha256_bytes,
)
from reliquary.eval_dashboard.holdout import load_locked_holdout
from reliquary.eval_dashboard.metrics import summarize_domain
from reliquary.eval_dashboard.models import (
    CheckpointResult,
    CodeTask,
    DomainResult,
    EvalConfig,
    EvalConfigManifest,
    EvalPublicationManifest,
    MathTask,
)
from reliquary.eval_dashboard.publisher import (
    EvalPublisher,
    INDEX_KEY,
    load_checkpoint_results,
    timestamp_pair,
)
from reliquary.eval_dashboard.runner import (
    hardware_metadata,
    load_tokenizer,
    run_checkpoint,
)
from reliquary.eval_dashboard.store import ObjectNotFound, ObjectStore


logger = logging.getLogger(__name__)


def compare_replay_summaries(
    published: DomainResult,
    replay: DomainResult,
    *,
    tolerance: float,
) -> dict[str, Any]:
    score_deltas = {
        "pass_at_1": replay.pass_at_1 - published.pass_at_1,
        "pass_at_k": replay.pass_at_k - published.pass_at_k,
        "pass_avg": replay.pass_avg - published.pass_avg,
    }
    percentage_deltas = {
        "trunc_pct": replay.trunc_pct - published.trunc_pct,
    }
    if published.forced_pct is not None and replay.forced_pct is not None:
        percentage_deltas["forced_pct"] = replay.forced_pct - published.forced_pct
    passed = all(abs(delta) <= tolerance for delta in score_deltas.values()) and all(
        abs(delta) <= tolerance * 100.0 for delta in percentage_deltas.values()
    )
    return {
        "published": {
            "pass_at_1": published.pass_at_1,
            "pass_at_k": published.pass_at_k,
            "pass_avg": published.pass_avg,
            "trunc_pct": published.trunc_pct,
            "forced_pct": published.forced_pct,
        },
        "replay": {
            "pass_at_1": replay.pass_at_1,
            "pass_at_k": replay.pass_at_k,
            "pass_avg": replay.pass_avg,
            "trunc_pct": replay.trunc_pct,
            "forced_pct": replay.forced_pct,
        },
        "score_deltas": score_deltas,
        "percentage_point_deltas": percentage_deltas,
        "score_tolerance": tolerance,
        "percentage_point_tolerance": tolerance * 100.0,
        "passed": passed,
    }


@dataclass(frozen=True)
class CheckpointTarget:
    repo_id: str
    revision: str
    checkpoint_n: int
    observed_window: int


def discover_checkpoint(
    validator_url: str, timeout_seconds: float = 20.0
) -> CheckpointTarget:
    base = validator_url.rstrip("/") + "/"
    checkpoint_response = requests.get(
        urljoin(base, "checkpoint"),
        timeout=timeout_seconds,
    )
    checkpoint_response.raise_for_status()
    checkpoint = checkpoint_response.json()

    observed_window = 0
    try:
        state_response = requests.get(urljoin(base, "state"), timeout=timeout_seconds)
        state_response.raise_for_status()
        observed_window = int(state_response.json().get("window_n", 0))
    except Exception as exc:
        logger.warning("could not resolve current validator window: %s", exc)

    target = CheckpointTarget(
        repo_id=str(checkpoint["repo_id"]),
        revision=str(checkpoint["revision"]),
        checkpoint_n=int(checkpoint["checkpoint_n"]),
        observed_window=observed_window,
    )
    if len(target.revision) < 40 or any(
        c not in "0123456789abcdef" for c in target.revision
    ):
        raise ValueError(
            "validator checkpoint is not pinned to an immutable hex revision"
        )
    return target


def send_alert(payload: dict[str, Any]) -> None:
    webhook = os.getenv("RELIQUARY_EVAL_ALERT_WEBHOOK", "").strip()
    if not webhook:
        return
    try:
        response = requests.post(webhook, json=payload, timeout=10.0)
        response.raise_for_status()
    except Exception:
        logger.exception("failed to send eval-dashboard alert")


class EvalWorker:
    def __init__(
        self,
        *,
        config: EvalConfig,
        store: ObjectStore,
        math_holdout_path: str | Path,
        math_review_path: str | Path,
        code_holdout_path: str | Path,
        code_review_path: str | Path,
        state_dir: str | Path,
        device: str,
        attention_implementation: str,
    ) -> None:
        self.config = config
        self.store = store
        self.publisher = EvalPublisher(store)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.attention_implementation = attention_implementation

        math_tasks, math_review = load_locked_holdout(
            math_holdout_path,
            math_review_path,
            config.math_holdout,
        )
        code_tasks, code_review = load_locked_holdout(
            code_holdout_path,
            code_review_path,
            config.code_holdout,
        )
        self.math_tasks: Sequence[MathTask] = math_tasks  # type: ignore[assignment]
        self.code_tasks: Sequence[CodeTask] = code_tasks  # type: ignore[assignment]
        self.contamination_reviews = {
            "math": math_review,
            "code": code_review,
        }

    def _effective_contract(self, target: CheckpointTarget):
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {self.device}, but CUDA is unavailable")
        tokenizer = load_tokenizer(
            self.config.tokenizer_source.repo_id,
            self.config.tokenizer_source.revision,
        )
        effective = build_effective_config(
            self.config,
            tokenizer=tokenizer,
            attention_implementation=self.attention_implementation,
            hardware=hardware_metadata(torch, self.device),
            contamination_reviews=self.contamination_reviews,
        )
        return tokenizer, effective, config_hash(effective)

    def _cache_path(self, target: CheckpointTarget, effective_hash: str) -> Path:
        directory = self.state_dir / "results" / effective_hash
        directory.mkdir(parents=True, exist_ok=True)
        return (
            directory / f"checkpoint-{target.checkpoint_n:06d}-{target.revision}.json"
        )

    def _load_cached_result(
        self,
        target: CheckpointTarget,
        effective_hash: str,
    ) -> CheckpointResult | None:
        path = self._cache_path(target, effective_hash)
        if not path.exists():
            return None
        result = CheckpointResult.model_validate_json(path.read_bytes())
        if (
            result.config_hash != effective_hash
            or result.model_revision != target.revision
            or result.model_repo_id != target.repo_id
            or result.checkpoint_n != target.checkpoint_n
        ):
            raise ValueError(f"cached result provenance mismatch: {path}")
        return result

    def _save_cached_result(self, result: CheckpointResult) -> None:
        path = self._cache_path(
            CheckpointTarget(
                repo_id=result.model_repo_id,
                revision=result.model_revision,
                checkpoint_n=result.checkpoint_n,
                observed_window=result.observed_window,
            ),
            result.config_hash,
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(canonical_json_bytes(result.model_dump(mode="json")))
        temporary.replace(path)

    def _existing_result(
        self,
        target: CheckpointTarget,
        effective_hash: str,
    ) -> CheckpointResult | None:
        for result in load_checkpoint_results(self.store, effective_hash):
            if (
                result.checkpoint_n == target.checkpoint_n
                and result.model_revision == target.revision
                and result.model_repo_id == target.repo_id
            ):
                return result
        return None

    def _status(
        self,
        state: str,
        *,
        target: CheckpointTarget,
        attempt: int,
        message: str = "",
    ) -> dict[str, Any]:
        timestamp, timestamp_iso = timestamp_pair(time.time())
        payload = {
            "schema_version": "1",
            "state": state,
            "owner": self.config.schedule.owner,
            "lineage_id": self.config.lineage.lineage_id,
            "checkpoint_repo_id": target.repo_id,
            "checkpoint_revision": target.revision,
            "checkpoint_n": target.checkpoint_n,
            "observed_window": target.observed_window,
            "attempt": attempt,
            "updated_at": timestamp,
            "updated_at_iso": timestamp_iso,
            "message": message[:2000],
        }
        try:
            self.publisher.write_status(payload)
        except Exception:
            logger.exception("failed to write eval-dashboard status")
        return payload

    def evaluate_target(
        self,
        target: CheckpointTarget,
        *,
        required_effective_hash: str | None = None,
        activate_existing: bool = True,
    ) -> tuple[CheckpointResult, dict[str, Any]]:
        if target.checkpoint_n == self.config.lineage.base.checkpoint_n:
            expected_repo = self.config.lineage.base.repo_id
            expected_revision = self.config.lineage.base.revision
            if target.repo_id != expected_repo or target.revision != expected_revision:
                raise ValueError("base target does not match configured base revision")
        elif target.repo_id != self.config.lineage.checkpoint_repo_id:
            raise ValueError(
                "target checkpoint repository does not match configured lineage"
            )

        tokenizer, effective, effective_hash = self._effective_contract(target)
        if (
            required_effective_hash is not None
            and effective_hash != required_effective_hash
        ):
            raise ValueError(
                "tokenizer/runtime contract changed inside one lineage; lock a new eval config"
            )

        existing = self._existing_result(target, effective_hash)
        if existing is not None and not activate_existing:
            logger.info(
                "checkpoint already evaluated checkpoint=%d revision=%s",
                target.checkpoint_n,
                target.revision,
            )
            return existing, effective

        attempts = self.config.schedule.retry_attempts
        candidate = existing
        for attempt in range(1, attempts + 1):
            self._status("running", target=target, attempt=attempt)
            try:
                if candidate is None:
                    candidate = self._load_cached_result(target, effective_hash)
                if candidate is None:
                    candidate, run_effective = run_checkpoint(
                        config=self.config,
                        model_repo_id=target.repo_id,
                        model_revision=target.revision,
                        checkpoint_n=target.checkpoint_n,
                        observed_window=target.observed_window,
                        math_tasks=self.math_tasks,
                        code_tasks=self.code_tasks,
                        tokenizer=tokenizer,
                        device=self.device,
                        attention_implementation=self.attention_implementation,
                        contamination_reviews=self.contamination_reviews,
                    )
                    if config_hash(run_effective) != effective_hash:
                        raise RuntimeError("effective config changed during evaluation")
                    self._save_cached_result(candidate)
                self.publisher.publish(
                    result=candidate,
                    config=self.config,
                    effective_config=effective,
                    now=time.time(),
                )
                self._status("succeeded", target=target, attempt=attempt)
                return candidate, effective
            except Exception as exc:
                payload = self._status(
                    "failed",
                    target=target,
                    attempt=attempt,
                    message=f"{type(exc).__name__}: {exc}",
                )
                logger.exception(
                    "checkpoint evaluation failed attempt=%d/%d", attempt, attempts
                )
                if attempt >= attempts:
                    send_alert({"event": "eval_dashboard_failed", **payload})
                    raise
                delay = min(
                    900,
                    self.config.schedule.retry_base_seconds * (2 ** (attempt - 1)),
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def run_once(self, target: CheckpointTarget) -> CheckpointResult:
        base = CheckpointTarget(
            repo_id=self.config.lineage.base.repo_id,
            revision=self.config.lineage.base.revision,
            checkpoint_n=self.config.lineage.base.checkpoint_n,
            observed_window=target.observed_window,
        )
        base_result, base_effective = self.evaluate_target(
            base,
            activate_existing=False,
        )
        del base_result
        result, _ = self.evaluate_target(
            target,
            required_effective_hash=config_hash(base_effective),
        )
        return result

    def replay_target(
        self,
        target: CheckpointTarget,
        *,
        n_prompts: int,
        tolerance: float,
    ) -> dict[str, Any]:
        """Re-run a bounded deterministic slice without publishing it."""
        if n_prompts <= 0:
            raise ValueError("n_prompts must be positive")
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError("tolerance must be in [0, 1]")
        tokenizer, effective, effective_hash = self._effective_contract(target)
        published = self._existing_result(target, effective_hash)
        if published is None:
            raise RuntimeError("checkpoint must be fully published before replay")
        math_tasks = self.math_tasks[:n_prompts]
        code_tasks = self.code_tasks[:n_prompts]
        if len(math_tasks) != n_prompts or len(code_tasks) != n_prompts:
            raise ValueError("replay slice exceeds the locked holdout")
        replay, replay_effective = run_checkpoint(
            config=self.config,
            model_repo_id=target.repo_id,
            model_revision=target.revision,
            checkpoint_n=target.checkpoint_n,
            observed_window=target.observed_window,
            math_tasks=math_tasks,
            code_tasks=code_tasks,
            tokenizer=tokenizer,
            device=self.device,
            attention_implementation=self.attention_implementation,
            contamination_reviews=self.contamination_reviews,
        )
        if config_hash(replay_effective) != effective_hash:
            raise RuntimeError("effective config changed during replay")

        domains: dict[str, Any] = {}
        all_passed = True
        for domain in ("math", "code"):
            published_tasks = published.domains[domain].tasks[:n_prompts]
            replay_tasks = replay.domains[domain].tasks
            if [task.task_id for task in published_tasks] != [
                task.task_id for task in replay_tasks
            ]:
                raise RuntimeError("published and replay task order differ")
            published_summary = summarize_domain(domain, published_tasks)
            replay_summary = summarize_domain(domain, replay_tasks)
            total_samples = n_prompts * self.config.generation.samples_per_prompt
            exact_matches = sum(
                left.completion_sha256 == right.completion_sha256
                for old_task, new_task in zip(published_tasks, replay_tasks)
                for left, right in zip(old_task.samples, new_task.samples)
            )
            comparison = compare_replay_summaries(
                published_summary,
                replay_summary,
                tolerance=tolerance,
            )
            all_passed = all_passed and comparison["passed"]
            domains[domain] = {
                "n_prompts": n_prompts,
                "samples_per_prompt": self.config.generation.samples_per_prompt,
                "exact_completion_match_rate": exact_matches / total_samples,
                **comparison,
            }
        return {
            "status": "passed" if all_passed else "failed",
            "config_hash": effective_hash,
            "checkpoint_n": target.checkpoint_n,
            "checkpoint_revision": target.revision,
            "domains": domains,
        }


def check_freshness(
    store: ObjectStore,
    *,
    expected_repo_id: str,
    expected_revision: str | None = None,
    expected_checkpoint_n: int | None = None,
    max_age_seconds: int,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    try:
        index = json.loads(store.get(INDEX_KEY))
    except ObjectNotFound as exc:
        raise RuntimeError("eval dashboard index is missing") from exc
    if not isinstance(index, dict):
        raise RuntimeError("eval dashboard index is invalid")
    current_hash = index.get("current_config_hash")
    runs = index.get("runs", [])
    current = next(
        (
            item
            for item in runs
            if isinstance(item, dict) and item.get("config_hash") == current_hash
        ),
        None,
    )
    if current is None:
        raise RuntimeError("eval dashboard current run is missing")
    if current.get("checkpoint_repo_id") != expected_repo_id:
        raise RuntimeError("eval dashboard points to the wrong model lineage")
    manifest_key = current.get("manifest_key")
    manifest_sha = current.get("manifest_sha256")
    if not isinstance(manifest_key, str) or not isinstance(manifest_sha, str):
        raise RuntimeError("eval dashboard current run has no publication manifest")
    manifest_payload = store.get(manifest_key)
    if sha256_bytes(manifest_payload) != manifest_sha:
        raise RuntimeError("eval publication manifest hash does not match the index")
    manifest = EvalPublicationManifest.model_validate_json(manifest_payload)
    config_manifest_payload = store.get(manifest.config_manifest.key)
    if sha256_bytes(config_manifest_payload) != manifest.config_manifest.sha256:
        raise RuntimeError("eval config manifest hash does not match the publication")
    config_manifest = EvalConfigManifest.model_validate_json(config_manifest_payload)
    if (
        manifest.config_hash != current_hash
        or config_manifest.config_hash != current_hash
        or manifest.config_sha256 != config_manifest.config_sha256
        or manifest.lineage_id != config_manifest.lineage_id
    ):
        raise RuntimeError("eval manifests disagree on run provenance")
    dashboard_key = current.get("dashboard_key")
    if not isinstance(dashboard_key, str):
        raise RuntimeError("eval dashboard current run has no immutable dashboard key")
    dashboard = json.loads(store.get(dashboard_key))
    expected_dashboard_sha = current.get("dashboard_sha256")
    if (
        not isinstance(expected_dashboard_sha, str)
        or sha256_bytes(store.get(dashboard_key)) != expected_dashboard_sha
    ):
        raise RuntimeError("eval dashboard snapshot hash does not match the index")
    if dashboard.get("latest_model_repo_id") != expected_repo_id:
        raise RuntimeError(
            "latest eval evidence is not from the expected model repository"
        )
    if (
        expected_revision is not None
        and dashboard.get("latest_checkpoint_revision") != expected_revision
    ):
        raise RuntimeError("eval evidence is not from the current checkpoint revision")
    if (
        expected_checkpoint_n is not None
        and dashboard.get("latest_checkpoint_n") != expected_checkpoint_n
    ):
        raise RuntimeError("eval evidence is not from the current checkpoint number")
    if (
        manifest.dashboard.key != dashboard_key
        or manifest.dashboard.sha256 != expected_dashboard_sha
        or manifest.latest_checkpoint_revision
        != dashboard.get("latest_checkpoint_revision")
        or manifest.latest_checkpoint_n != dashboard.get("latest_checkpoint_n")
    ):
        raise RuntimeError("eval publication manifest and dashboard disagree")
    generated_at = float(dashboard.get("generated_at", 0.0))
    publication_age = now - generated_at
    if publication_age < -300:
        raise RuntimeError("eval dashboard publication timestamp is in the future")
    evidence_completed_at = float(dashboard.get("evidence_completed_at", 0.0))
    evidence_age = now - evidence_completed_at
    status = "fresh" if publication_age <= max_age_seconds else "overdue"
    return {
        "status": status,
        "age_seconds": publication_age,
        "publication_age_seconds": publication_age,
        "evidence_age_seconds": evidence_age,
        "max_age_seconds": max_age_seconds,
        "checkpoint_n": dashboard.get("latest_checkpoint_n"),
        "checkpoint_revision": dashboard.get("latest_checkpoint_revision"),
        "generated_at": generated_at,
        "generated_at_iso": dashboard.get("generated_at_iso"),
        "evidence_completed_at": evidence_completed_at,
        "evidence_completed_at_iso": dashboard.get("evidence_completed_at_iso"),
        "config_hash": current_hash,
    }
