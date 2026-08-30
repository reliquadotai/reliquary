"""Runtime wiring for ``reliquary train-worker``.

Everything testable lives in journal/worker/train_runner/publisher/resume;
this module only assembles them from the environment and runs the loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def _r2_client():
    import boto3
    from botocore.config import Config

    account_id = os.getenv("R2_ACCOUNT_ID", "")
    endpoint = (
        os.getenv("R2_ENDPOINT_URL")
        or f"https://{account_id}.r2.cloudflarestorage.com"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        region_name=os.getenv("R2_REGION", "us-east-1"),
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            read_timeout=120,
            max_pool_connections=32,
        ),
    )


def _download_checkpoint(client, bucket: str, revision: str, dest: Path) -> bool:
    """Pull the R2-mirrored snapshot (multipart parallel). Returns False
    when the mirror lacks this revision (bootstrap → HF fallback)."""
    from boto3.s3.transfer import TransferConfig

    from reliquary.trainer.publisher import R2_CHECKPOINT_PREFIX

    prefix = f"{R2_CHECKPOINT_PREFIX}/{revision}/"
    listed = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = listed.get("Contents", [])
    if not contents:
        return False
    config = TransferConfig(
        multipart_threshold=32 * 1024 * 1024,
        multipart_chunksize=32 * 1024 * 1024,
        max_concurrency=16,
    )
    dest.mkdir(parents=True, exist_ok=True)
    for obj in contents:
        key = obj["Key"]
        filename = key[len(prefix):]
        if not filename or "/" in filename:
            continue
        client.download_file(bucket, key, str(dest / filename), Config=config)
    return True


def _hf_download(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, revision=revision)


def run_train_worker(*, shadow: bool = False) -> None:
    import torch

    from reliquary.constants import (
        ATTN_IMPLEMENTATION,
        CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
        DEFAULT_BASE_MODEL,
        DEFAULT_BASE_MODEL_REVISION,
        ENVIRONMENT_MIX,
        PROTOCOL_VERSION,
    )
    from reliquary.shared.modeling import (
        load_text_generation_model,
        load_tokenizer,
    )
    from reliquary.trainer.journal import (
        WindowJournal,
        migrate_journal_cursor,
        r2_fetch_fn,
    )
    from reliquary.trainer.publisher import TrainerPublisher
    from reliquary.trainer.resume import resolve_resume_point
    from reliquary.trainer.train_runner import TrainRunner
    from reliquary.trainer.worker import TrainerLockLost, TrainerWorker
    from reliquary.shared.training_payload import active_training_identity
    from reliquary.validator.checkpoint_profile import (
        validate_checkpoint_profile,
    )
    from reliquary.validator.training import reset_training_state

    repo_id = os.environ.get("RELIQUARY_HF_REPO_ID", "").strip()
    if not repo_id:
        raise SystemExit("RELIQUARY_HF_REPO_ID is required")
    bucket = os.getenv("R2_BUCKET_ID", "reliquary")
    state_dir = Path(os.environ.get(
        "RELIQUARY_TRAINER_STATE_DIR", "/root/reliquary/trainer",
    ))
    client = _r2_client()
    fetch = r2_fetch_fn(client, bucket)

    expected_identity = (
        active_training_identity() if PROTOCOL_VERSION >= 5 else None
    )
    revision, cursor, checkpoint_n = resolve_resume_point(
        fetch,
        env=os.environ,
        expected_identity=expected_identity,
    )
    lr_schedule_step: int | None = None

    if revision is None:
        logger.info(
            "bootstrap: loading base model %s@%s, cursor=%d",
            DEFAULT_BASE_MODEL, DEFAULT_BASE_MODEL_REVISION, cursor,
        )
        model_path = DEFAULT_BASE_MODEL
        load_kwargs = {"revision": DEFAULT_BASE_MODEL_REVISION}
        tokenizer = load_tokenizer(model_path, **load_kwargs)
    else:
        snapshot_dir = state_dir / "resume" / revision
        if not _download_checkpoint(client, bucket, revision, snapshot_dir):
            logger.info("R2 mirror lacks %s; falling back to HF", revision)
            snapshot_dir = Path(_hf_download(repo_id, revision))
        profile = validate_checkpoint_profile(snapshot_dir, required=True)
        # The PROFILE is authoritative for run state; the manifest cursor
        # was only a hint for which snapshot to fetch.
        cursor = int(profile.get("trained_window_cursor", cursor))
        # C3/R25: a checkpoint published before the v6 cutover carries a
        # cursor in RAW window space; this journal reads the encoded space
        # (window * FILL_CLOSED_EMISSIONS_PER_WINDOW + batch). Migrate here,
        # exactly once, detected by the marker the publisher stamps beside
        # the cursor -- never as a manual runbook step. The next publish
        # writes the migrated marker, so a later resume is a no-op.
        migrated, key_space = migrate_journal_cursor(
            cursor, profile.get("journal_key_space"),
        )
        if migrated != cursor:
            logger.warning(
                "journal key space changed (%s -> %s): cursor %d -> %d",
                profile.get("journal_key_space") or "raw",
                key_space, cursor, migrated,
            )
        cursor = migrated
        raw_step = profile.get("lr_schedule_step")
        lr_schedule_step = int(raw_step) if raw_step is not None else None
        model_path = str(snapshot_dir)
        load_kwargs = {}
        tokenizer = load_tokenizer(model_path)

    # Telemetry: train_step's emit_metrics is a silent no-op unless
    # telemetry.init ran. The trainer has no wallet; the run identity
    # comes from env so the shadow run is distinguishable from prod
    # (set RELIQUARY_WANDB_VERSION, e.g. "detached-shadow").
    from reliquary import constants as _C
    from reliquary.validator import telemetry as _telemetry

    _telemetry.init(
        hotkey_ss58=os.environ.get(
            "RELIQUARY_TRAINER_WANDB_IDENTITY", "trainer0",
        ),
        config={
            "role": "train-worker",
            "shadow": bool(shadow),
            "learning_rate": _C.LEARNING_RATE,
            "kl_beta": _C.KL_BETA,
            "publish_interval": CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
        },
    )

    logger.info("loading model from %s", model_path)
    model = load_text_generation_model(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        **load_kwargs,
    ).to("cuda:0")
    try:
        model.gradient_checkpointing_enable()
    except (AttributeError, NotImplementedError):
        logger.warning("model does not support gradient checkpointing")
    reset_training_state()

    runner = TrainRunner(
        model,
        env_targets=dict(ENVIRONMENT_MIX),
        env_order=[name for name, _ in ENVIRONMENT_MIX],
        global_step_hint=lr_schedule_step,
    )
    publisher = TrainerPublisher(
        repo_id=repo_id,
        staging_dir=str(state_dir / "staging"),
        tokenizer=tokenizer,
        r2_client=client,
        bucket=bucket,
    )

    publish_state = {"checkpoint_n": checkpoint_n}

    def publish_fn(reason: str) -> str:
        from reliquary.validator.training import current_lr_schedule_step

        publish_state["checkpoint_n"] += 1
        return asyncio.run(publisher.publish(
            runner.model,
            checkpoint_n=publish_state["checkpoint_n"],
            lr_schedule_step=current_lr_schedule_step(),
            trained_window_cursor=worker.cursor,
            reason=reason,
        ))

    def head_revision_fn() -> str | None:
        from huggingface_hub import HfApi

        try:
            return HfApi().model_info(repo_id).sha
        except Exception:
            logger.exception("HF HEAD lookup failed; skipping guard")
            return None

    # Startup reconciliation: a crash between the HF upload and the
    # manifest PUT leaves an orphaned HEAD that would trip the
    # single-writer guard on every publish forever. Adopt the observed
    # HEAD as "ours" at startup (loudly): our weights/cursor still come
    # from the manifest, and the next publish supersedes the orphan. A
    # REAL foreign publisher keeps moving HEAD and still trips the guard
    # on the next in-run publish.
    startup_head = head_revision_fn()
    last_revision = revision
    if (
        startup_head is not None
        and revision is not None
        and startup_head != revision
    ):
        logger.warning(
            "HF HEAD %s != manifest revision %s at startup — adopting "
            "HEAD (orphaned half-publish or foreign publisher; the "
            "single-writer guard stays armed for the rest of the run)",
            startup_head[:12], revision[:12],
        )
        last_revision = startup_head

    def freeze_fn() -> str | None:
        from reliquary.constants import TRAIN_UNTIL_CHECKPOINT_N

        if os.environ.get("RELIQUARY_DISABLE_TRAIN", "").lower() in {
            "1", "true", "yes", "on",
        }:
            return "emergency_training_freeze"
        if (
            TRAIN_UNTIL_CHECKPOINT_N > 0
            and publish_state["checkpoint_n"] >= TRAIN_UNTIL_CHECKPOINT_N
        ):
            return "training_checkpoint_ceiling"
        return None

    worker = TrainerWorker(
        journal=WindowJournal(
            fetch_fn=fetch,
            expected_identity=expected_identity,
        ),
        train_fn=runner.step,
        publish_fn=publish_fn,
        head_revision_fn=head_revision_fn,
        cursor=cursor,
        stride=int(os.environ.get("RELIQUARY_TRAINER_WINDOW_STRIDE", "1")),
        publish_every=CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
        last_published_revision=last_revision,
        shadow=shadow,
        freeze_fn=freeze_fn,
        abort_epoch_fn=runner.abort_epoch,
    )

    logger.info(
        "train-worker starting: cursor=%d, revision=%s, shadow=%s",
        worker.cursor, revision, shadow,
    )
    transient_failures = 0
    while True:
        try:
            outcome = worker.run_once()
        except TrainerLockLost:
            logger.critical("trainer lock lost; exiting", exc_info=True)
            raise SystemExit(3)
        except Exception:
            # Transient R2/HF error mid-poll or mid-publish: the cursor
            # never advanced, so retrying is always safe. Backoff, don't
            # die — a process exit costs the optimizer moments.
            transient_failures += 1
            delay = min(60.0, 5.0 * transient_failures)
            logger.exception(
                "worker iteration failed (attempt %d); retrying in %.0fs",
                transient_failures, delay,
            )
            time.sleep(delay)
            continue
        transient_failures = 0
        if outcome == "waited":
            time.sleep(5.0)
        elif outcome == "frozen":
            time.sleep(30.0)
        elif outcome in {"tombstone", "quarantined", "published"}:
            logger.info("%s (cursor=%d)", outcome, worker.cursor)
