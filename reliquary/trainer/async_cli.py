"""Runtime wiring for ``reliquary train-async``: the detached async trainer (rl-async-design.md §4.7-4.8).

Same topology as ``train-worker``: its own GPU box, a single card or torchrun FSDP2 ranks
(``RELIQUARY_TRAINER_BACKEND=fsdp2``: rank 0 drives, the others follow ``LOAD`` / ``TRAIN`` / ``GATHER``). The
buffer, the submissions, the lease book and the prices are the validator's, reached through the trainer link
(``TrainerLink``, ``RELIQUARY_RL_TRAINER_URL`` + ``RELIQUARY_RL_TRAINER_TOKEN``); the target mix and the checkpoint
run come from the validator's task (``/rl/trainer/config``).

Start: a fresh lease book (no checkpoint adopted) starts from the base model (``DEFAULT_BASE_MODEL``, or
``RELIQUARY_RL_BASE_MODEL``, at ``RELIQUARY_RL_BASE_MODEL_REVISION`` when set), published and adopted as checkpoint 0. Otherwise the trainer resumes from the weights
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


def _download_base_model(settings: Mapping[str, Any]) -> str:
    """The base model's local path; a pinned ``base_model_revision`` wins over the default model's revision.

    Everything after the download (EOS ids, FSDP2 load, Miles conversion) reads this local snapshot."""
    from huggingface_hub import snapshot_download

    from reliquary.constants import DEFAULT_BASE_MODEL, DEFAULT_BASE_MODEL_REVISION

    base = settings["base_model"] or DEFAULT_BASE_MODEL
    revision = settings.get("base_model_revision") or (None if settings["base_model"] else DEFAULT_BASE_MODEL_REVISION)
    if Path(base).is_dir():
        return base
    return snapshot_download(base, **({"revision": revision} if revision else {}))


def _heartbeat_s(env: Mapping[str, str]) -> float:
    value = float(env.get("RELIQUARY_RL_TRAINER_HEARTBEAT_S", "60"))
    if not 5 <= value <= 300:                   # well under the validator's outage grace (600 s)
        raise ValueError("RELIQUARY_RL_TRAINER_HEARTBEAT_S must be in [5, 300]")
    return value


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
        "url": url, "token": token, "validator_hotkey": hotkey, "link_prefix": _link_prefix(env),
        "state_dir": Path(env.get("RELIQUARY_TRAINER_STATE_DIR", "/root/reliquary/trainer")) / "async",
        "base_model": str(env.get("RELIQUARY_RL_BASE_MODEL", "")).strip() or None,
        "base_model_revision": str(env.get("RELIQUARY_RL_BASE_MODEL_REVISION", "")).strip() or None,
        "batch_groups": _int(env, "RELIQUARY_RL_BATCH_GROUPS", 512),
        "steps_per_checkpoint": _int(env, "RELIQUARY_RL_STEPS_PER_CHECKPOINT", 1),
        "anchor_interval": _int(env, "RELIQUARY_RL_ANCHOR_INTERVAL", 50),
        "hf_prefix": prefix or None, "hf_every": every,
        "hf_keep_last": _int(env, "RELIQUARY_RL_HF_BACKUP_KEEP", 3),
        "hf_milestone_every": _int(env, "RELIQUARY_RL_HF_BACKUP_MILESTONE_EVERY", None),
        # the out-of-zone audits on this trainer's weights (the validator's default audit source)
        "audits": str(env.get("RELIQUARY_RL_TRAINER_AUDITS", "1")).strip() != "0",
        "audits_per_step": _int(env, "RELIQUARY_RL_TRAINER_AUDITS_PER_STEP", 128),
        "audit_budget_s": float(env.get("RELIQUARY_RL_TRAINER_AUDIT_BUDGET_S", "420")),
        # the next batch is downloaded while this one trains; the step's reports go out in parallel
        "prefetch": str(env.get("RELIQUARY_RL_TRAINER_PREFETCH", "1")).strip() != "0",
        "report_workers": _int(env, "RELIQUARY_RL_TRAINER_REPORT_WORKERS", 16),
        # liveness for the validator's outage pay: always on (a silent trainer reads as down)
        "heartbeat_s": _heartbeat_s(env),
    }


EXECUTOR_ID_ENV = "RELIQUARY_RL_TRAINER_EXECUTOR_ID"


def _link_prefix(env: Mapping[str, str]) -> str:
    """A rented pod (an executor id) reaches its task's trainer routes through the front (``/rl/<task>/trainer``,
    ``RELIQUARY_TASK_ID``); an operator's trainer, the controller's own listener (``/rl/trainer``)."""
    task = str(env.get("RELIQUARY_TASK_ID", "")).strip()
    if not (str(env.get(EXECUTOR_ID_ENV, "")).strip() and task):
        return "/rl/trainer"
    from reliquary.miner.rl_task_choice import task_prefix

    return f"{task_prefix(task)}/trainer"


def trainer_link(settings: Mapping[str, Any], *, client=None):
    from reliquary.trainer.rl_link import TrainerLink

    return TrainerLink(settings["url"], settings["token"], client=client, prefix=settings["link_prefix"])


def checkpoint_store(link, env: Mapping[str, str] = os.environ):
    """The publisher's store: presigned uploads through the link on a rented pod (an executor id and no
    ``R2_PUBLIC_*`` key), else the bucket directly with the operator's keys."""
    from reliquary.trainer.ckpt_publisher import R2ObjectStore

    keyed = any(str(env.get(f"R2_PUBLIC_{n}", "")).strip() for n in ("ACCESS_KEY_ID", "SECRET_ACCESS_KEY"))
    if str(env.get(EXECUTOR_ID_ENV, "")).strip() and not keyed:
        from reliquary.trainer.presigned_store import PresignedObjectStore

        return PresignedObjectStore(link)
    from reliquary.infrastructure.storage import public_checkpoint_bucket, public_checkpoint_client

    return R2ObjectStore(public_checkpoint_client(), public_checkpoint_bucket())


def publisher_options(store) -> dict[str, int]:
    """Extra ``DeltaPublisher`` arguments a store asks for (the presigned one: smaller anchor shards)."""
    size = getattr(store, "anchor_shard_bytes", None)
    return {} if size is None else {"anchor_shard_bytes": int(size)}


def drop_hf_backup_on_pod(store, settings: dict) -> None:
    """A pod on presigned uploads holds no bucket or hub credential: the HF backup is off there."""
    if getattr(store, "anchor_shard_bytes", None) is not None and settings["hf_prefix"]:
        logger.warning("presigned checkpoint store: the Hugging Face backup is off on this pod")
        settings["hf_prefix"] = settings["hf_every"] = None


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


def disable_cudnn_attention(torch) -> None:
    """The trainer's sdpa never takes the cuDNN backend: its fused attention graph fails on long sequences
    (``mha_graph.execute``) and kills the rank. Flash and memory-efficient kernels serve the same calls."""
    torch.backends.cuda.enable_cudnn_sdp(False)


def _close_async_trainer(trainer, backup) -> None:
    """Always stop the optional telemetry worker, including after a failed publication."""
    from reliquary.validator import telemetry

    try:
        trainer.close()
    finally:
        try:
            if backup is not None:
                backup.close(timeout=60)
        finally:
            telemetry.finish()


def run_async_train_worker() -> None:
    import torch

    disable_cudnn_attention(torch)              # every rank, before any model is built

    from reliquary.constants import ATTN_IMPLEMENTATION, DEFAULT_BASE_MODEL
    from reliquary.miner.rl_async import write_hf_dir
    from reliquary.shared.modeling import load_text_generation_model
    from reliquary.trainer.async_trainer import AsyncTrainer, AsyncTrainerConfig
    from reliquary.trainer.ckpt_publisher import DeltaPublisher
    from reliquary.trainer.cli import _check_trainer_launch, _run_train_follower
    from reliquary.trainer.distributed import (
        CommandChannel, broadcasting_train_step, gather_full_state_dict, init_distributed, load_sharded_model,
        trainer_backend,
    )
    from reliquary.validator import telemetry
    from reliquary.validator.training import configure_distributed, reset_training_state, train_step

    dist_ctx = channel = gateway = None
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

    heartbeat = None
    try:
        from reliquary.trainer.rl_link import TrainerHeartbeat

        settings = async_trainer_settings()
        link = trainer_link(settings)
        # Beat from here on, not once the model is loaded: resuming, downloading and starting the engine can take
        # longer than the platform's staleness window, and it replaces a pod it sees silent.
        heartbeat = TrainerHeartbeat(link, settings["heartbeat_s"]).start()
        task = link.config()
        telemetry.init(settings["validator_hotkey"], {
            "task_id": task["task_id"], "run": task["run"], "targets": task["targets"], "backend": backend,
            "order_id": task.get("order_id"), "wandb": task.get("wandb"),
            "batch_groups": settings["batch_groups"], "steps_per_checkpoint": settings["steps_per_checkpoint"],
            "gpus": dist_ctx.world_size if dist_ctx is not None else 1}, role="trainer", version=str(task["run"]))
        state_dir: Path = settings["state_dir"]
        state_dir.mkdir(parents=True, exist_ok=True)
        resumed = resume_weights(link, validator_hotkey=settings["validator_hotkey"], work_dir=state_dir / "resume")
        if resumed is None:
            base = settings["base_model"] or DEFAULT_BASE_MODEL
            model_path = _download_base_model(settings)
            model_dir, version = Path(model_path), None
            logger.info("fresh lease book: starting from %s", base)
        else:
            state, version, model_dir = resumed
            model_path = str(write_hf_dir(state, model_dir, state_dir / "resume" / "model"))
            del state
            logger.info("resuming from checkpoint %d", version)
        load_kwargs = {}
        eos_ids = _eos_ids(model_path, load_kwargs)

        started = time.monotonic()
        live: dict = {}                         # the trainer, once built (train_step may hand back a new model)

        def current():
            return live["trainer"].model if "trainer" in live else model

        if backend == "miles":
            gateway, model, step_fn, state_fn, miles_audit = _start_miles(model_path, version, eos_ids)
        elif dist_ctx is None:
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
        audits = None
        if settings["audits"]:
            from reliquary.trainer.audits import TrainerAudits, audit_group, broadcasting_audit

            if backend == "miles":
                audit_fn = miles_audit
            else:
                audit_fn = audit_group if dist_ctx is None else broadcasting_audit(channel)
            audits = TrainerAudits(link, audit_fn=audit_fn)
        logger.info("startup_stage=trainer_model_load elapsed_seconds=%.3f", time.monotonic() - started)
        reset_training_state()

        store = checkpoint_store(link)
        drop_hf_backup_on_pod(store, settings)
        publisher = DeltaPublisher(run=str(task["run"]), **publisher_options(store),
                                   store=store,
                                   staging_dir=state_dir / "staging", anchor_interval=settings["anchor_interval"],
                                   model_dir=model_dir)
        backup = None
        if settings["hf_prefix"]:
            from reliquary.trainer.hf_backup import BackgroundHfBackup, HfRepoPerBackup

            repos = HfRepoPerBackup(HF_BACKUP_NAMESPACE, settings["hf_prefix"],
                                   receipt_dir=state_dir / "hf-publications")
            backup = BackgroundHfBackup(upload=repos.upload, list_backups=repos.list_backups,
                                        delete_backup=repos.delete_backup, keep_last=settings["hf_keep_last"],
                                        milestone_every=settings["hf_milestone_every"],
                                        staging_dir=state_dir / "hf-staging")
        config = AsyncTrainerConfig(batch_groups=settings["batch_groups"],
                                    steps_per_checkpoint=settings["steps_per_checkpoint"],
                                    hf_backup_every=settings["hf_every"],
                                    eos_ids=eos_ids,
                                    audits_per_step=settings["audits_per_step"],
                                    audit_budget_s=settings["audit_budget_s"],
                                    prefetch=settings["prefetch"], report_workers=settings["report_workers"])
        link.requeue()                          # this trainer holds nothing yet

        def backup_save(state, folder, *, lr_schedule_step):
            from reliquary.trainer.hf_backup import write_backup_profile

            write_hf_dir(state, model_dir, folder)
            write_backup_profile(folder, lr_schedule_step=lr_schedule_step,
                                 checkpoint_run=str(task["run"]), task_id=str(task["task_id"]))

        trainer = AsyncTrainer(
            model, config=config, buffer=link, submissions=link, book=link, publisher=publisher,
            state_fn=state_fn, backup=backup, backup_save=backup_save,
            train_step_fn=step_fn, audits=audits, final_checkpoint=task.get("max_steps"),
            global_step_hint=None if version is None else version * settings["steps_per_checkpoint"])
        live["trainer"] = trainer
        logger.info("train-async starting: task %s, run %s, target mix %s (priced, not enforced), checkpoint %d",
                    task["task_id"], task["run"], task["targets"], trainer.version)
        try:
            run_forever(trainer)
        finally:
            _close_async_trainer(trainer, backup)
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        # Followers block on the next command; release them on any exit.
        if channel is not None:
            channel.send("STOP")
        if gateway is not None:
            gateway.stop()


def _start_miles(model_path, version: int | None, eos_ids):
    """``RELIQUARY_TRAINER_BACKEND=miles``: the gateway started on this box's GPUs from ``model_path``, then the
    trainer's callables on it. This process keeps no GPU (the gateway holds them all)."""
    from reliquary.constants import GRAD_NORM_SKIP_THRESHOLD, PROTOCOL_MODEL_ID
    from reliquary.trainer import miles_gateway
    from reliquary.trainer.audits import gateway_audit
    from reliquary.trainer.miles_engine import MilesState, MilesTrainStep, expected_weight_specs

    import signal

    settings = miles_gateway.miles_settings()
    miles_gateway.check_gateway_constants(settings["python"])

    def _terminate(signum, frame):           # a plain SIGTERM must still run the finally that stops the gateway
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate)
    gateway = miles_gateway.launch_gateway(model_path, settings, base_model=PROTOCOL_MODEL_ID,
                                           grad_norm_skip=GRAD_NORM_SKIP_THRESHOLD, env=dict(os.environ))
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # before any CUDA call here: the gateway's GPUs stay whole
        client = miles_gateway.WatchedClient(
            miles_gateway.create_full_training_client(gateway.base_url, PROTOCOL_MODEL_ID),
            alive=lambda: gateway.process.poll() is None)
        step_fn = MilesTrainStep(client, eos_ids=eos_ids, global_step_hint=version or 0)
        state_fn = MilesState(client, checkpoint_root=settings["output_dir"],
                              expected=expected_weight_specs(model_path), base_dir=model_path,
                              delete=miles_gateway.checkpoint_deleter(gateway.base_url))
        return gateway, None, step_fn, state_fn, gateway_audit(client, eos_ids=eos_ids)
    except BaseException:
        gateway.stop()
        raise


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
