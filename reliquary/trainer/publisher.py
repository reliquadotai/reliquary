"""Trainer-side checkpoint publication: HF (miners' source) + R2 mirror
(validator's fast path) + the candidate manifest the validator polls.

Signing stays with the validator: it signs (checkpoint_n || revision) at
swap time, after downloading and profile-validating the snapshot. The
candidate manifest is written LAST — it is the commit point.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Callable

from reliquary.shared.checkpoint_identity import (
    require_checkpoint_number,
    require_checkpoint_repository,
    require_immutable_checkpoint_revision,
)

logger = logging.getLogger(__name__)

CANDIDATE_MANIFEST_KEY = "reliquary/training/candidate-manifest.json"
R2_CHECKPOINT_PREFIX = "reliquary/checkpoints"


def checkpoint_key(revision: str, filename: str) -> str:
    revision = require_immutable_checkpoint_revision(revision)
    if (
        not isinstance(filename, str)
        or not filename
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise ValueError("checkpoint filename must be a single path component")
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
        checkpoint_number_floor: int | None = None,
        save_fn: Callable[[Any, Any, Path], None] | None = None,
        hf_upload_fn: Callable[..., Any] | None = None,
    ) -> None:
        from reliquary.validator.checkpoint import (
            _default_save_hf_format,
            _default_upload,
        )

        self.repo_id = require_checkpoint_repository(repo_id)
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self._r2 = r2_client
        self._bucket = bucket
        self._save = save_fn or _default_save_hf_format
        self._hf_upload = hf_upload_fn or _default_upload
        self._checkpoint_number_floor = (
            require_checkpoint_number(
                checkpoint_number_floor,
                field="trainer publisher checkpoint number floor",
            )
            if checkpoint_number_floor is not None
            else None
        )
        self._previous_mirror_revision: str | None = None

    async def publish(
        self,
        model: Any,
        *,
        checkpoint_n: int,
        lr_schedule_step: int | None,
        trained_window_cursor: int,
        reason: str,
    ) -> str:
        from reliquary.trainer.journal import active_journal_key_space
        from reliquary.validator.checkpoint_profile import (
            write_checkpoint_profile,
        )

        checkpoint_n = require_checkpoint_number(
            checkpoint_n,
            field="trainer published checkpoint number",
        )
        if (
            self._checkpoint_number_floor is not None
            and checkpoint_n <= self._checkpoint_number_floor
        ):
            raise ValueError(
                "trainer published checkpoint number must advance monotonically"
            )

        # C3/R25: the key space the cursor below is expressed in, stored
        # beside it so a resume can detect a cutover and migrate exactly
        # once instead of relying on a manual runbook step.
        key_space = active_journal_key_space()

        snapshot_dir = self.staging_dir / f"ckpt_{checkpoint_n}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(
                self._save, model, self.tokenizer, snapshot_dir,
            )
            extra: dict[str, Any] = {
                "trained_window_cursor": int(trained_window_cursor),
                "journal_key_space": key_space,
            }
            if lr_schedule_step is not None:
                extra["lr_schedule_step"] = int(lr_schedule_step)
            write_checkpoint_profile(snapshot_dir, extra=extra)

            revision = await self._hf_upload(
                folder_path=str(snapshot_dir),
                repo_id=self.repo_id,
                commit_message=f"checkpoint {checkpoint_n} ({reason})",
            )
            revision = require_immutable_checkpoint_revision(
                revision,
                field="trainer checkpoint publisher revision",
            )

            # R2 mirror: every snapshot file under a revision-scoped
            # prefix, multipart-parallel (the validator pulls from here).
            config = None
            for path in sorted(snapshot_dir.iterdir()):
                if not path.is_file():
                    continue
                if config is None:
                    config = _multipart_transfer_config()
                await asyncio.to_thread(
                    self._r2.upload_file,
                    str(path), self._bucket,
                    checkpoint_key(revision, path.name),
                    Config=config,
                )

            # Commit point: the candidate manifest the validator polls.
            from reliquary.shared.training_payload import (
                active_training_identity,
            )

            manifest = {
                **active_training_identity(),
                "checkpoint_n": checkpoint_n,
                "repo_id": self.repo_id,
                "revision": revision,
                "trained_window_cursor": int(trained_window_cursor),
                "reason": str(reason),
                "journal_key_space": key_space,
            }
            await asyncio.to_thread(
                self._r2.put_object,
                Bucket=self._bucket,
                Key=CANDIDATE_MANIFEST_KEY,
                Body=json.dumps(manifest).encode("utf-8"),
            )
            self._checkpoint_number_floor = checkpoint_n
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)

        # Bound the mirror: keep the current + previous revision only
        # (previous eliminates any race with a validator mid-download).
        # HF is the durable archive of every revision; without cleanup
        # the mirror grows ~8 GB per publish forever.
        await asyncio.to_thread(
            self._prune_mirror,
            keep={revision, self._previous_mirror_revision},
        )
        self._previous_mirror_revision = revision

        logger.info(
            "Published checkpoint %d to %s@%s (cursor=%d, reason=%s)",
            checkpoint_n, self.repo_id, revision[:12],
            trained_window_cursor, reason,
        )
        return revision

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
