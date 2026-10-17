"""Runtime wiring for ``reliquary train-async``: the detached async trainer (rl-async-design.md §4.7-4.8).

Same topology as ``train-worker``: its own GPU box, a single card or torchrun FSDP2 ranks
(``RELIQUARY_TRAINER_BACKEND=fsdp2``: rank 0 drives, the others follow ``LOAD`` / ``TRAIN`` / ``GATHER``). The
buffer, the submissions, the lease book and the prices are the validator's, reached through the trainer link
(``TrainerLink``, ``RELIQUARY_RL_TRAINER_URL`` + ``RELIQUARY_RL_TRAINER_TOKEN``); the target mix and the checkpoint
run come from the validator's task (``/rl/trainer/config``).

Start: a fresh lease book (no checkpoint adopted) starts from the base model (``DEFAULT_BASE_MODEL``, or
``RELIQUARY_RL_BASE_MODEL``), published and adopted as checkpoint 0. Otherwise the trainer resumes from the weights
of the latest published checkpoint, rebuilt from the bucket through the validator's signed index (``GET
/rl/trainer/checkpoint``, ``CheckpointSync``); a version published but never adopted (a crash between the two) is adopted
first. The optimizer starts fresh (the bf16 checkpoint is all the bucket holds).

Publication: ``DeltaPublisher`` into ``reliquary-public`` (``R2_PUBLIC_*``), the anchors carrying the model's config
and tokenizer; Hugging Face backups every ``RELIQUARY_RL_HF_BACKUP_EVERY`` steps under
``ReliquaryForge/<RELIQUARY_RL_HF_BACKUP_PREFIX>-step-<n>`` (``keep_last``, milestones), off the training path.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

HF_BACKUP_NAMESPACE = "ReliquaryForge"


def _eos_ids(model_path, load_kwargs) -> tuple[int, ...]:
    """The model's stop tokens, as the validator's grader resolves them (``rl_async_service._grader_for``)."""
    from types import SimpleNamespace

    from transformers import AutoConfig, GenerationConfig

    from reliquary.shared.modeling import load_tokenizer, resolve_eos_token_ids

    try:
        generation = GenerationConfig.from_pretrained(model_path, **load_kwargs)
    except OSError:
        generation = None
    model = SimpleNamespace(config=AutoConfig.from_pretrained(model_path, **load_kwargs), generation_config=generation)
    return tuple(sorted(resolve_eos_token_ids(model, load_tokenizer(model_path, **load_kwargs))))


def _int(env: Mapping[str, str], name: str, default: int | None, *, minimum: int = 1) -> int | None:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return default
    value = int(raw, 10)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def async_trainer_settings(env: Mapping[str, str] = os.environ) -> dict[str, Any]:
    """Everything the entry reads from the environment, checked before any GPU work."""
    url = str(env.get("RELIQUARY_RL_TRAINER_URL", "")).strip()
    token = str(env.get("RELIQUARY_RL_TRAINER_TOKEN", "")).strip()
    hotkey = str(env.get("RELIQUARY_RL_VALIDATOR_HOTKEY", "")).strip()
    if not (url and token and hotkey):
        raise ValueError("RELIQUARY_RL_TRAINER_URL, RELIQUARY_RL_TRAINER_TOKEN and RELIQUARY_RL_VALIDATOR_HOTKEY "
                         "are required")
    prefix = str(env.get("RELIQUARY_RL_HF_BACKUP_PREFIX", "")).strip()
    every = _int(env, "RELIQUARY_RL_HF_BACKUP_EVERY", None)
    if bool(prefix) != bool(every):
        raise ValueError("RELIQUARY_RL_HF_BACKUP_PREFIX and RELIQUARY_RL_HF_BACKUP_EVERY go together")
    return {
        "url": url, "token": token, "validator_hotkey": hotkey,
        "state_dir": Path(env.get("RELIQUARY_TRAINER_STATE_DIR", "/root/reliquary/trainer")) / "async",
        "base_model": str(env.get("RELIQUARY_RL_BASE_MODEL", "")).strip() or None,
        "batch_groups": _int(env, "RELIQUARY_RL_BATCH_GROUPS", 64),
        "steps_per_checkpoint": _int(env, "RELIQUARY_RL_STEPS_PER_CHECKPOINT", 1),
        "anchor_interval": _int(env, "RELIQUARY_RL_ANCHOR_INTERVAL", 50),
        "hf_prefix": prefix or None, "hf_every": every,
        "hf_keep_last": _int(env, "RELIQUARY_RL_HF_BACKUP_KEEP", 3),
        "hf_milestone_every": _int(env, "RELIQUARY_RL_HF_BACKUP_MILESTONE_EVERY", None),
    }


def resume_weights(link, *, validator_hotkey: str, work_dir: Path, sync=None, fetch=None):
    """``(state, version, model_files_dir)`` of the latest published checkpoint, or None on a fresh book. A published
    version ahead of the book (published, then the trainer died before adopting it) is adopted here."""
    latest = link.latest_checkpoint
    if latest is None:
        return None
    if fetch is None:
        fetch = link.checkpoint
    if sync is None:
        from reliquary.miner.ckpt_sync import CheckpointSync

        sync = CheckpointSync(validator_hotkey=validator_hotkey, work_dir=work_dir / "download",
                              model_dir=work_dir / "model-files")
    state, version, _ = sync.sync(fetch(), None, None)
    if version < latest:
        raise RuntimeError(f"the bucket's latest checkpoint {version} is behind the adopted {latest}")
    if version > latest:
        logger.warning("checkpoint %d was published but not adopted; adopting it", version)
        link.adopt_checkpoint(version)
    return state, version, sync.model_dir


def run_async_train_worker() -> None:
    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION, DEFAULT_BASE_MODEL, DEFAULT_BASE_MODEL_REVISION
    from reliquary.infrastructure.storage import public_checkpoint_bucket, public_checkpoint_client
    from reliquary.miner.rl_async import write_hf_dir
    from reliquary.shared.modeling import load_text_generation_model
    from reliquary.trainer.async_trainer import AsyncTrainer, AsyncTrainerConfig
    from reliquary.trainer.ckpt_publisher import DeltaPublisher, R2ObjectStore
    from reliquary.trainer.cli import _check_trainer_launch, _run_train_follower
    from reliquary.trainer.distributed import (
        CommandChannel, broadcasting_train_step, gather_full_state_dict, init_distributed, load_sharded_model,
        trainer_backend,
    )
    from reliquary.trainer.rl_link import TrainerLink
    from reliquary.validator import telemetry
    from reliquary.validator.training import configure_distributed, reset_training_state, train_step

    dist_ctx = channel = None
    backend = trainer_backend()
    _check_trainer_launch(backend, os.environ)
    if backend == "fsdp2":
        from reliquary.constants import KL_BETA

        if float(KL_BETA) != 0.0:
            raise SystemExit("fsdp2 trainer requires KL_BETA == 0")
        dist_ctx = init_distributed()
        channel = CommandChannel(dist_ctx)
        configure_distributed(dist_ctx)
        if not dist_ctx.is_main:
            _run_train_follower(dist_ctx, channel)
            return

    try:
        settings = async_trainer_settings()
        link = TrainerLink(settings["url"], settings["token"])
        task = link.config()
        telemetry.init(settings["validator_hotkey"], {
            "task_id": task["task_id"], "run": task["run"], "targets": task["targets"], "backend": backend,
            "batch_groups": settings["batch_groups"], "steps_per_checkpoint": settings["steps_per_checkpoint"],
            "gpus": dist_ctx.world_size if dist_ctx is not None else 1}, role="trainer", version=str(task["run"]))
        state_dir: Path = settings["state_dir"]
        state_dir.mkdir(parents=True, exist_ok=True)
        resumed = resume_weights(link, validator_hotkey=settings["validator_hotkey"], work_dir=state_dir / "resume")
        if resumed is None:
            from huggingface_hub import snapshot_download

            base = settings["base_model"] or DEFAULT_BASE_MODEL
            load_kwargs = {} if settings["base_model"] else {"revision": DEFAULT_BASE_MODEL_REVISION}
            model_path = base if Path(base).is_dir() else snapshot_download(base, **load_kwargs)
            model_dir, version = Path(model_path), None
            logger.info("fresh lease book: starting from %s", base)
        else:
            state, version, model_dir = resumed
            model_path = str(write_hf_dir(state, model_dir, state_dir / "resume" / "model"))
            del state
            logger.info("resuming from checkpoint %d", version)
        load_kwargs = {}

        started = time.monotonic()
        live: dict = {}                         # the trainer, once built (train_step may hand back a new model)

        def current():
            return live["trainer"].model if "trainer" in live else model

        if dist_ctx is None:
            model = load_text_generation_model(model_path, torch_dtype=torch.bfloat16,
                                               attn_implementation=ATTN_IMPLEMENTATION).to("cuda:0")
            try:
                model.gradient_checkpointing_enable()
            except (AttributeError, NotImplementedError):
                logger.warning("model does not support gradient checkpointing")
            state_fn = lambda: current().state_dict()  # noqa: E731
            step_fn = train_step
        else:
            channel.send("LOAD", {"model_path": model_path, "load_kwargs": load_kwargs})
            model = load_sharded_model(model_path, load_kwargs, dist_ctx, attn_implementation=ATTN_IMPLEMENTATION)

            def state_fn():
                channel.send("GATHER")
                return gather_full_state_dict(current(), dist_ctx)

            step_fn = broadcasting_train_step(channel, train_step)
        logger.info("startup_stage=trainer_model_load elapsed_seconds=%.3f", time.monotonic() - started)
        reset_training_state()

        publisher = DeltaPublisher(run=str(task["run"]),
                                   store=R2ObjectStore(public_checkpoint_client(), public_checkpoint_bucket()),
                                   staging_dir=state_dir / "staging", anchor_interval=settings["anchor_interval"],
                                   model_dir=model_dir)
        backup = None
        if settings["hf_prefix"]:
            from reliquary.trainer.hf_backup import BackgroundHfBackup, HfRepoPerBackup

            repos = HfRepoPerBackup(HF_BACKUP_NAMESPACE, settings["hf_prefix"])
            backup = BackgroundHfBackup(upload=repos.upload, list_backups=repos.list_backups,
                                        delete_backup=repos.delete_backup, keep_last=settings["hf_keep_last"],
                                        milestone_every=settings["hf_milestone_every"],
                                        staging_dir=state_dir / "hf-staging")
        config = AsyncTrainerConfig(batch_groups=settings["batch_groups"],
                                    steps_per_checkpoint=settings["steps_per_checkpoint"],
                                    hf_backup_every=settings["hf_every"],
                                    eos_ids=_eos_ids(model_path, load_kwargs))
        link.requeue()                          # this trainer holds nothing yet
        trainer = AsyncTrainer(
            model, config=config, buffer=link, submissions=link, book=link, publisher=publisher,
            state_fn=state_fn, backup=backup, backup_save=lambda s, folder: write_hf_dir(s, model_dir, folder),
            train_step_fn=step_fn,
            global_step_hint=None if version is None else version * settings["steps_per_checkpoint"])
        live["trainer"] = trainer
        logger.info("train-async starting: task %s, run %s, target mix %s (priced, not enforced), checkpoint %d",
                    task["task_id"], task["run"], task["targets"], trainer.version)
        try:
            run_forever(trainer)
        finally:
            trainer.close()
            if backup is not None:
                backup.close(timeout=60)
            telemetry.finish()
    finally:
        # Followers block on the next command; release them on any exit.
        if channel is not None:
            channel.send("STOP")


def run_forever(trainer, *, steps: int | None = None) -> None:
    """Step until stopped (each step waits for a full batch). A step that fails is fatal (the process restarts from
    the last adopted checkpoint, and the validator requeues what it held): the link already retried the validator,
    and a half-reported step must not be retried."""
    done = 0
    while steps is None or done < steps:
        if trainer.run_once() == "trained":
            logger.info("async step %s", trainer.last_step)
        done += 1


__all__ = ["async_trainer_settings", "resume_weights", "run_async_train_worker", "run_forever"]
