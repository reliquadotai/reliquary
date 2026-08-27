"""Trainer-side checkpoint publication: HF (miners' source) + R2 mirror
(validator's fast path) + the candidate manifest the validator polls.

Signing stays with the validator: it signs (checkpoint_n || revision) at
swap time, after downloading and profile-validating the snapshot. The
candidate manifest is written LAST — it is the commit point.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Callable

from reliquary.trainer.retention import (
    CheckpointRetentionPolicy,
    HfCompactionPlan,
    HfHistoryManager,
)

logger = logging.getLogger(__name__)

CANDIDATE_MANIFEST_KEY = "reliquary/training/candidate-manifest.json"
R2_CHECKPOINT_PREFIX = "reliquary/checkpoints"


def checkpoint_key(revision: str, filename: str) -> str:
    return f"{R2_CHECKPOINT_PREFIX}/{revision}/{filename}"


def _multipart_transfer_config():
    """Mandatory for the 8 GB hop: single-stream R2 is ~20 MB/s on the
    prod boxes (per-connection window), multipart x16 measured 120+ MB/s."""
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=32 * 1024 * 1024,
        multipart_chunksize=32 * 1024 * 1024,
        max_concurrency=16,
    )


class TrainerPublisher:
    def __init__(
        self,
        *,
        repo_id: str,
        staging_dir: str,
        tokenizer: Any,
        r2_client: Any,
        bucket: str,
        save_fn: Callable[[Any, Any, Path], None] | None = None,
        hf_upload_fn: Callable[..., Any] | None = None,
        retention_policy: CheckpointRetentionPolicy | None = None,
        hf_history_manager: HfHistoryManager | None = None,
    ) -> None:
        from reliquary.validator.checkpoint import (
            _default_save_hf_format,
            _default_upload,
        )

        self.repo_id = repo_id
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self._r2 = r2_client
        self._bucket = bucket
        self._save = save_fn or _default_save_hf_format
        self._hf_upload = hf_upload_fn or _default_upload
        self._previous_mirror_revision: str | None = None
        self._first_history_rooted = False
        self.retention = (
            retention_policy or CheckpointRetentionPolicy.from_env()
        )
        self._hf_history = hf_history_manager
        if self.retention.enabled and self._hf_history is None:
            self._hf_history = HfHistoryManager()

    async def publish(
        self,
        model: Any,
        *,
        checkpoint_n: int,
        lr_schedule_step: int | None,
        trained_window_cursor: int,
        reason: str,
        publication_seq: int | None = None,
        expected_parent_revision: str | None = None,
    ) -> str:
        from reliquary.validator.checkpoint_profile import (
            write_checkpoint_profile,
        )

        if self.retention.enabled:
            publication_seq = self.retention.validate_publication_seq(
                publication_seq
            )
        elif publication_seq is not None and int(publication_seq) <= 0:
            raise ValueError("publication_seq must be positive when provided")

        snapshot_dir = self.staging_dir / f"ckpt_{checkpoint_n}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        compaction_plan: HfCompactionPlan | None = None
        run_id: str | None = None
        retained_prefix: str | None = None
        retention_class: str | None = None
        try:
            await asyncio.to_thread(
                self._save, model, self.tokenizer, snapshot_dir,
            )
            extra: dict[str, Any] = {
                "trained_window_cursor": int(trained_window_cursor),
            }
            if lr_schedule_step is not None:
                extra["lr_schedule_step"] = int(lr_schedule_step)
            if publication_seq is not None:
                extra["publication_seq"] = int(publication_seq)
            write_checkpoint_profile(snapshot_dir, extra=extra)

            if self.retention.enabled:
                assert self._hf_history is not None
                await asyncio.to_thread(
                    self._hf_history.assert_storage_budget,
                    repo_id=self.repo_id,
                    freeze_bytes=self.retention.storage_freeze_bytes,
                )
                if self.retention.should_compact_before(publication_seq):
                    if not expected_parent_revision:
                        raise RuntimeError(
                            "retention compaction requires the expected parent "
                            "HF revision"
                        )
                    from reliquary.shared.training_payload import (
                        active_training_identity,
                    )

                    run_id = str(active_training_identity()["training_run_id"])
                    if publication_seq == self.retention.keep_initial + 1:
                        await asyncio.to_thread(
                            self._assert_run_start_archive_complete,
                            run_id=run_id,
                        )
                    compaction_plan = await asyncio.to_thread(
                        self._hf_history.prepare_compaction,
                        repo_id=self.repo_id,
                        run_id=run_id,
                        publication_seq=publication_seq,
                        expected_parent_revision=expected_parent_revision,
                        policy=self.retention,
                    )

            uploaded_revision = await self._hf_upload(
                folder_path=str(snapshot_dir),
                repo_id=self.repo_id,
                commit_message=f"checkpoint {checkpoint_n} ({reason})",
            )
            revision = str(uploaded_revision)
            if compaction_plan is not None:
                assert self._hf_history is not None
                revision = await asyncio.to_thread(
                    self._hf_history.compact_uploaded_head,
                    repo_id=self.repo_id,
                    uploaded_revision=revision,
                    checkpoint_n=checkpoint_n,
                    publication_seq=publication_seq,
                )

            # R2 mirror: every snapshot file under a revision-scoped
            # prefix, multipart-parallel (the validator pulls from here).
            await self._upload_snapshot(
                snapshot_dir,
                prefix=f"{R2_CHECKPOINT_PREFIX}/{revision}",
            )

            # Commit point: the candidate manifest the validator polls.
            from reliquary.shared.training_payload import (
                active_training_identity,
            )

            identity = active_training_identity()
            snapshot_files: list[dict[str, Any]] = []
            if self.retention.enabled:
                assert publication_seq is not None
                run_id = str(identity["training_run_id"])
                retention_class = self.retention.retention_class(
                    publication_seq
                )
                retained_prefix = self.retention.snapshot_prefix(
                    run_id, publication_seq
                )
                snapshot_files = await asyncio.to_thread(
                    _snapshot_inventory,
                    snapshot_dir,
                    include_sha256=retained_prefix is not None,
                )

                ledger = {
                    "schema_version": 1,
                    **identity,
                    "checkpoint_n": int(checkpoint_n),
                    "publication_seq": int(publication_seq),
                    "repo_id": self.repo_id,
                    "revision": str(revision),
                    "trained_window_cursor": int(trained_window_cursor),
                    "lr_schedule_step": (
                        int(lr_schedule_step)
                        if lr_schedule_step is not None else None
                    ),
                    "reason": str(reason),
                    "retention_class": retention_class,
                    "snapshot_prefix": retained_prefix,
                    "files": snapshot_files,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                if retained_prefix is not None:
                    await self._upload_snapshot(
                        snapshot_dir,
                        prefix=retained_prefix,
                    )
                    await asyncio.to_thread(
                        self._r2.put_object,
                        Bucket=self._bucket,
                        Key=f"{retained_prefix}/manifest.json",
                        Body=(
                            json.dumps(
                                ledger, sort_keys=True, separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8"),
                        ContentType="application/json",
                    )
                await asyncio.to_thread(
                    self._r2.put_object,
                    Bucket=self._bucket,
                    Key=self.retention.ledger_key(run_id, publication_seq),
                    Body=(
                        json.dumps(
                            ledger, sort_keys=True, separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8"),
                    ContentType="application/json",
                )

            manifest = {
                **identity,
                "checkpoint_n": int(checkpoint_n),
                "repo_id": self.repo_id,
                "revision": str(revision),
                "trained_window_cursor": int(trained_window_cursor),
                "reason": str(reason),
            }
            if publication_seq is not None:
                manifest["publication_seq"] = int(publication_seq)
            if retention_class is not None:
                manifest["retention_class"] = retention_class
                manifest["evaluation_snapshot_prefix"] = retained_prefix
            await asyncio.to_thread(
                self._r2.put_object,
                Bucket=self._bucket,
                Key=CANDIDATE_MANIFEST_KEY,
                Body=json.dumps(manifest).encode("utf-8"),
            )
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)

        if (
            self.retention.enabled
            and publication_seq is not None
            and publication_seq > self.retention.keep_initial
        ):
            # The candidate manifest is already committed, so retention
            # maintenance can never make an otherwise healthy checkpoint
            # unavailable.  It is retried on every later publication until the
            # first-history marker is rooted; a failure only delays reclamation.
            try:
                assert self._hf_history is not None
                assert run_id is not None
                if not self._first_history_rooted:
                    if compaction_plan is not None and compaction_plan.permanent:
                        expected_first_revision = expected_parent_revision
                    else:
                        expected_first_revision = await asyncio.to_thread(
                            self._run_start_revision,
                            run_id=run_id,
                        )
                    if expected_first_revision:
                        rooted = await asyncio.to_thread(
                            self._hf_history.compact_protected_branch,
                            repo_id=self.repo_id,
                            branch=self.retention.first_history_branch(run_id),
                            expected_revision=expected_first_revision,
                            publication_seq=self.retention.keep_initial,
                        )
                        self._first_history_rooted = True
                        logger.info(
                            "rooted R2-archived HF run-start branch at %s",
                            rooted[:12],
                        )
                if compaction_plan is not None:
                    deleted = await asyncio.to_thread(
                        self._hf_history.cleanup_grace_branches,
                        repo_id=self.repo_id,
                        run_id=run_id,
                        policy=self.retention,
                    )
                    if deleted:
                        logger.info("pruned HF grace branches: %s", deleted)
            except Exception:
                logger.exception("HF retention cleanup failed (non-fatal)")

        if (
            self.retention.enabled
            and retained_prefix is not None
            and retention_class == "evaluation_candidate"
        ):
            # Evaluation snapshots are intentionally a rolling queue.  The
            # milestone namespace is never touched by this janitor.
            assert run_id is not None
            await asyncio.to_thread(
                self._prune_evaluation_candidates,
                run_id=run_id,
            )

        # Bound the mirror: keep the current + previous revision only
        # (previous eliminates any race with a validator mid-download).
        # Protected HF blocks and sparse R2 evaluation/milestone snapshots
        # provide durable history; this serving mirror stays deliberately
        # small so it cannot grow ~8 GB per publication forever.
        await asyncio.to_thread(
            self._prune_mirror,
            keep={
                str(revision),
                self._previous_mirror_revision,
                expected_parent_revision,
            },
        )
        self._previous_mirror_revision = str(revision)

        logger.info(
            "Published checkpoint %d to %s@%s (cursor=%d, reason=%s)",
            checkpoint_n, self.repo_id, str(revision)[:12],
            trained_window_cursor, reason,
        )
        return str(revision)

    async def _upload_snapshot(self, directory: Path, *, prefix: str) -> None:
        config = None
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if config is None:
                config = _multipart_transfer_config()
            await asyncio.to_thread(
                self._r2.upload_file,
                str(path),
                self._bucket,
                f"{prefix}/{path.name}",
                Config=config,
            )

    def _prune_mirror(self, *, keep: set[str | None]) -> None:
        """Best-effort deletion of mirror revisions outside ``keep`` —
        including strays from before a restart. Never fails a publish."""
        try:
            listed = self._r2.list_objects_v2(
                Bucket=self._bucket,
                Prefix=f"{R2_CHECKPOINT_PREFIX}/",
                Delimiter="/",
            )
            for entry in listed.get("CommonPrefixes", []):
                prefix = entry["Prefix"]
                rev = prefix[len(R2_CHECKPOINT_PREFIX) + 1:].rstrip("/")
                if rev in keep:
                    continue
                objs = self._r2.list_objects_v2(
                    Bucket=self._bucket, Prefix=prefix,
                )
                for obj in objs.get("Contents", []):
                    self._r2.delete_object(
                        Bucket=self._bucket, Key=obj["Key"],
                    )
                logger.info("pruned mirror revision %s", rev[:12])
        except Exception:
            logger.exception("mirror prune failed (non-fatal)")

    def _run_start_revision(self, *, run_id: str) -> str | None:
        prefix = self.retention.snapshot_prefix(
            run_id,
            self.retention.keep_initial,
        )
        assert prefix is not None
        try:
            response = self._r2.get_object(
                Bucket=self._bucket,
                Key=f"{prefix}/manifest.json",
            )
            value = json.loads(response["Body"].read().decode("utf-8"))
            if (
                not isinstance(value, dict)
                or value.get("training_run_id") != run_id
                or int(value.get("publication_seq", -1))
                != self.retention.keep_initial
                or not value.get("revision")
            ):
                raise ValueError("run-start manifest identity mismatch")
            return str(value["revision"])
        except Exception:
            logger.debug(
                "R2 run-start manifest is not ready for HF branch rooting",
                exc_info=True,
            )
            return None

    def _assert_run_start_archive_complete(self, *, run_id: str) -> None:
        root = self.retention.run_start_prefix(run_id)
        listed = self._r2.list_objects_v2(
            Bucket=self._bucket,
            Prefix=root,
            Delimiter="/",
        )
        observed = {
            str(entry["Prefix"])
            for entry in listed.get("CommonPrefixes", [])
        }
        expected = {
            f"{root}publication-{seq:06d}/"
            for seq in range(1, self.retention.keep_initial + 1)
        }
        missing = set(expected - observed)
        for prefix in sorted(expected & observed):
            contents = self._r2.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
            ).get("Contents", [])
            if not any(
                str(item.get("Key")) == f"{prefix}manifest.json"
                for item in contents
            ):
                missing.add(prefix)
        if missing:
            raise RuntimeError(
                "refusing to compact HF before the dense R2 run-start "
                f"archive is complete; missing {len(missing)} publication(s)"
            )

    def _prune_evaluation_candidates(self, *, run_id: str) -> None:
        """Keep the newest bounded candidate snapshots for one training run.

        This only deletes objects below the dedicated candidate prefix.  The
        permanent milestone namespace and the lightweight JSON ledger are
        outside that prefix and cannot be selected accidentally.
        """

        try:
            root = self.retention.candidate_run_prefix(run_id)
            listed = self._r2.list_objects_v2(
                Bucket=self._bucket,
                Prefix=root,
                Delimiter="/",
            )
            prefixes = sorted(
                str(entry["Prefix"])
                for entry in listed.get("CommonPrefixes", [])
            )
            stale = prefixes[
                :-self.retention.evaluation_candidates_to_keep
            ]
            for prefix in stale:
                objects = self._r2.list_objects_v2(
                    Bucket=self._bucket,
                    Prefix=prefix,
                )
                for obj in objects.get("Contents", []):
                    self._r2.delete_object(
                        Bucket=self._bucket,
                        Key=obj["Key"],
                    )
                logger.info("pruned R2 evaluation candidate %s", prefix)
        except Exception:
            logger.exception(
                "R2 evaluation-candidate prune failed (non-fatal)"
            )


def _snapshot_inventory(
    directory: Path, *, include_sha256: bool,
) -> list[dict[str, Any]]:
    files = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        item: dict[str, Any] = {
            "path": path.name,
            "size": path.stat().st_size,
        }
        if include_sha256:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            item["sha256"] = digest.hexdigest()
        files.append(item)
    return files
