"""Validator main loop — v2.1 batch-driven state machine (OPEN→TRAINING→PUBLISHING→READY)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    COOLDOWN_REBUILD_LOOKBACK,
    B_BATCH,
    BOOTSTRAP_WINDOWS,
    CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
    CHECKPOINT_STAGING_DIR_DEFAULT,
    DEFAULT_HF_REPO_ID,
    GRAD_CLIP_NORM,
    HASH_DEDUP_RETENTION_WINDOWS,
    KL_BETA,
    LEARNING_RATE,
    LR_COSINE_MAX_WINDOWS,
    LR_WARMUP_WINDOWS,
    M_ROLLOUTS,
    POLL_INTERVAL_SECONDS,
    PPO_CLIP_EPSILON,
    SUBNET_START_BLOCK,
    VALIDATOR_HTTP_PORT,
    WANDB_TRAINING_VERSION,
    WINDOW_LENGTH,
    WINDOW_TIMEOUT_SECONDS,
)
from reliquary.environment.base import Environment
from reliquary.infrastructure import chain, storage
from reliquary.protocol.submission import RolloutSubmission, WindowState
from reliquary.validator import telemetry
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.checkpoint import CheckpointStore
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.dedup import RolloutHashSet
from reliquary.validator.server import ValidatorServer
from reliquary.validator.training import train_step

logger = logging.getLogger(__name__)


def _try_empty_cuda_cache() -> None:
    """Best-effort `torch.cuda.empty_cache()` after a forward pass.

    Releases CUDA cached memory that's no longer referenced — typically
    activations from a forward pass that have gone out of scope. Active
    tensors (e.g. the model's weights) stay allocated, so this is safe
    to call after every accept_submission / train_step.

    Why we need this in the validator:

    The GRAIL verifier runs ``model.forward(...)`` on every accepted
    submission. PyTorch's CUDA caching allocator holds onto activation
    buffers between calls in a pool to avoid the cost of ``cudaMalloc``
    on every call. Under sustained traffic this is normally fine — the
    pool reuses freed slots. But when ``train_step`` is configured to
    OOM-fast (as in this validator) it leaves the pool partially
    allocated. Successive train_step calls fragment the pool over time
    and eventually verify_commitment's ``cublasCreate`` can't find a
    contiguous chunk → ``CUBLAS_STATUS_ALLOC_FAILED``.

    Calling ``empty_cache()`` after each forward pass / train_step
    returns the freed slots to the OS, preventing fragmentation
    accumulation. Cost: a few ms of cudaFree calls. Negligible against
    the ~5-25s GRAIL verification time.

    Imports lazily so non-CUDA test environments (CPU-only CI) don't
    try to import torch at module load.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Never let a cache-cleanup failure escape — it's a best-effort
        # optimization, not load-bearing logic.
        logger.debug("torch.cuda.empty_cache failed (non-fatal)", exc_info=True)


def is_bootstrap_window(window_start: int, subnet_start: int) -> bool:
    """True iff *window_start* is within ``BOOTSTRAP_WINDOWS`` of ``subnet_start``.

    Bootstrap windows use the relaxed zone / cooldown / M values so the
    batch can fill while miner population and env coverage are thin.
    """
    if window_start < subnet_start:
        return False
    return window_start - subnet_start < BOOTSTRAP_WINDOWS


def open_grpo_window(
    window_start: int,
    env,
    model,
    *,
    cooldown_map: CooldownMap,
    hash_set: RolloutHashSet | None,
    tokenizer,
    bootstrap: bool = False,
    queue_drained_predicate=None,
) -> GrpoWindowBatcher:
    """Instantiate a GrpoWindowBatcher for this window.

    ``cooldown_map`` is the validator's long-lived CooldownMap, shared
    across windows. Each window's sealed batch updates it via
    ``GrpoWindowBatcher.seal_batch``.

    ``queue_drained_predicate`` is wired by ``Service.run`` to the
    server's submit-queue ``empty()`` check so the v2.3 seal extension
    can wait for every queued trigger-round submission to be GRAIL-
    validated before firing the seal. See
    ``GrpoWindowBatcher._delayed_seal_at_drand_boundary``.
    """
    def _completion_text(rollout: RolloutSubmission) -> str:
        prompt_len = rollout.commit.get("rollout", {}).get("prompt_length", 0)
        return tokenizer.decode(rollout.tokens[prompt_len:])

    def _canonical_prompt_tokens(prompt_idx: int) -> list[int]:
        problem = env.get_problem(prompt_idx)
        return list(tokenizer.encode(problem["prompt"], add_special_tokens=False))

    return GrpoWindowBatcher(
        window_start=window_start,
        env=env,
        model=model,
        tokenizer=tokenizer,
        cooldown_map=cooldown_map,
        hash_set=hash_set,
        bootstrap=bootstrap,
        completion_text_fn=_completion_text,
        canonical_prompt_tokens_fn=_canonical_prompt_tokens,
        queue_drained_predicate=queue_drained_predicate,
    )



def _default_load_model(local_path: str):
    """Default: load a HF checkpoint onto cuda:0 in bfloat16 with the
    configured attention implementation."""
    import torch
    from transformers import AutoModelForCausalLM
    from reliquary.constants import ATTN_IMPLEMENTATION
    return AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda:0").eval()


class ValidationService:
    def __init__(
        self,
        wallet,
        model,
        tokenizer,
        env: Environment,
        netuid: int,
        *,
        use_drand: bool = True,
        http_host: str = "0.0.0.0",
        http_port: int = VALIDATOR_HTTP_PORT,
        external_ip: str | None = None,
        external_port: int | None = None,
        hf_repo_id: str | None = None,
        resume_from: str | None = None,
        load_model_fn: Any | None = None,
    ) -> None:
        self.wallet = wallet
        import importlib.metadata as _im
        try:
            reliquary_version = _im.version("reliquary")
        except _im.PackageNotFoundError:
            reliquary_version = "dev"
        telemetry.init(
            hotkey_ss58=wallet.hotkey.ss58_address,
            config={
                "learning_rate": LEARNING_RATE,
                "kl_beta": KL_BETA,
                "ppo_clip_epsilon": PPO_CLIP_EPSILON,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "lr_warmup_windows": LR_WARMUP_WINDOWS,
                "lr_cosine_max_windows": LR_COSINE_MAX_WINDOWS,
                "b_batch": B_BATCH,
                "m_rollouts_per_prompt": M_ROLLOUTS,
                "window_length": WINDOW_LENGTH,
                "wandb_training_version": WANDB_TRAINING_VERSION,
                "reliquary_version": reliquary_version,
            },
        )
        import copy
        # Two-model architecture (see docs/superpowers/plans/2026-05-13-...).
        # train_model: trainable, mutated by train_step every window.
        # verify_model: frozen snapshot of the last published checkpoint —
        # used by batcher.verify_commitment_proofs and as the KL reference
        # inside train_step. Refreshed in-place after every successful
        # publish via load_state_dict.
        self.train_model = model
        if model is not None:
            try:
                self.verify_model = copy.deepcopy(model)
                self.verify_model.eval()
                for p in self.verify_model.parameters():
                    p.requires_grad = False
            except (AttributeError, TypeError):
                # Test fixtures (e.g. MagicMock) — fall back to sharing the
                # same object. Tests don't exercise the train/verify split
                # in this case.
                self.verify_model = model
        else:
            self.verify_model = None

        # Enable gradient checkpointing on the train model only.
        try:
            self.train_model.gradient_checkpointing_enable()
        except (AttributeError, NotImplementedError):
            logger.warning(
                "train_model does not support gradient_checkpointing_enable"
            )
        self.tokenizer = tokenizer
        self.env = env
        self.netuid = netuid
        self.use_drand = use_drand
        self.external_ip = external_ip
        self.external_port = external_port

        self._last_processed_window: int = -1
        self._windows_in_interval: int = 0
        self._cooldown_map = CooldownMap(
            cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS
        )
        self._hash_set = RolloutHashSet(
            retention_windows=HASH_DEDUP_RETENTION_WINDOWS,
        )
        self._late_drops: dict[str, dict[str, int]] = {}

        self.server = ValidatorServer(host=http_host, port=http_port)
        self.server.set_late_drop_callback(self.record_late_drop)

        # v2.1 state machine infrastructure — in-memory only, bootstrapped at
        # startup from R2 + HF (no local JSON state file).
        self._window_n: int = 0
        self._checkpoint_n: int = 0
        self._publish_every = CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
        self._checkpoint_store = CheckpointStore(
            validator_hotkey=wallet.hotkey.ss58_address,
            wallet=wallet,
            repo_id=hf_repo_id or DEFAULT_HF_REPO_ID,
            staging_dir_path=CHECKPOINT_STAGING_DIR_DEFAULT,
            tokenizer=tokenizer,
        )
        self._active_batcher = None
        # Stashed by ``_set_window_randomness`` after the drand fetch
        # succeeds; consumed by the background verify task (Task 5).
        # ``None`` on the mock-only path.
        self._last_beacon: dict | None = None
        # asyncio.Task wrapping _verify_beacon_async; held so the GC
        # doesn't collect a live task between OPEN and TRAINING.
        self._verify_task: asyncio.Task | None = None
        self._current_window_state: WindowState = WindowState.READY

        self._resume_from = resume_from
        self._load_model_fn = load_model_fn or _default_load_model

    def _set_state(self, s: WindowState) -> None:
        self._current_window_state = s
        # Also notify the server so /state returns the right value.
        self.server.set_current_state(s)

    def record_late_drop(self, hotkey: str, reason: str) -> None:
        """Bump the (hotkey, reason) counter. Both call sites run on the
        asyncio event loop so no lock is needed. Reset in _archive_window.
        """
        bucket = self._late_drops.setdefault(hotkey, {})
        bucket[reason] = bucket.get(reason, 0) + 1

    async def _apply_resume_from(self) -> None:
        """If --resume-from was set, load the model from that source and
        install a manifest. No-op if unset."""
        if not self._resume_from:
            return
        from reliquary.validator.resume import (
            parse_resume_source,
            resolve_resume_source,
        )
        from reliquary.validator.checkpoint import ManifestEntry

        def _commit_title(repo_id, revision):
            from huggingface_hub import HfApi
            api = HfApi()
            commits = api.list_repo_commits(repo_id=repo_id)
            for c in commits:
                if c.commit_id == revision:
                    return c.title
            return ""

        def _download(repo_id, revision):
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=repo_id, revision=revision)

        source = parse_resume_source(self._resume_from)
        local_path, checkpoint_n = resolve_resume_source(
            source,
            hf_repo_id=self._checkpoint_store.repo_id,
            download_fn=_download,
            commit_title_fn=_commit_title,
        )
        # Load weights — this replaces both models loaded at __init__.
        # verify_model gets the resumed weights too (so the batcher
        # verifies miners against the resumed checkpoint, which is what
        # they have access to via HF).
        self.train_model = self._load_model_fn(local_path)
        try:
            self.train_model.gradient_checkpointing_enable()
        except (AttributeError, NotImplementedError):
            pass
        if self.verify_model is not None:
            self.verify_model.load_state_dict(self.train_model.state_dict())
        else:
            import copy
            self.verify_model = copy.deepcopy(self.train_model)
            self.verify_model.eval()
            for p in self.verify_model.parameters():
                p.requires_grad = False
        # Extract the canonical revision string to publish to miners.
        # IMPORTANT: strip the scheme prefix — miners call HF with this value
        # as the ``revision=`` kwarg, and HF rejects ``sha:<hex>`` / ``path:<dir>``
        # strings outright. They must see a bare 40-char hex (for sha) or a
        # bare local path identifier (for path, though that's a test-only mode
        # and miners won't successfully pull it anyway).
        from reliquary.validator.resume import ShaSource
        if isinstance(source, ShaSource):
            revision_str = source.sha
        else:
            revision_str = source.path
        # Reconstruct manifest so miners see the resumed checkpoint via /state.
        sig_payload = f"{checkpoint_n}|{revision_str}".encode()
        sig_bytes = self.wallet.hotkey.sign(sig_payload)
        entry = ManifestEntry(
            checkpoint_n=checkpoint_n,
            repo_id=self._checkpoint_store.repo_id,
            revision=revision_str,
            signature="ed25519:" + sig_bytes.hex(),
        )
        self._checkpoint_store._current = entry
        self._checkpoint_n = checkpoint_n
        self.server.set_current_checkpoint(entry)
        logger.info(
            "Resumed from %s: checkpoint_n=%d",
            self._resume_from, checkpoint_n,
        )

    def _open_window(self) -> None:
        """Create a new GrpoWindowBatcher in a non-active state.

        Builds the batcher and wires the active checkpoint hash, but does
        NOT expose it to the HTTP server yet — call ``_activate_window``
        after ``_set_window_randomness`` succeeds. This two-phase open
        prevents miner submissions from reaching a batcher whose
        ``randomness`` is still the default ``""``, which crashes commitment
        verification in ``indices_from_root`` if the chain call that fills
        randomness fails (e.g. finney WebSocket returns 503).
        """
        self._window_n += 1
        bootstrap = is_bootstrap_window(
            window_start=self._window_n,
            subnet_start=SUBNET_START_BLOCK,
        )
        self._active_batcher = open_grpo_window(
            window_start=self._window_n,
            env=self.env, model=self.verify_model,
            cooldown_map=self._cooldown_map,
            hash_set=self._hash_set,
            tokenizer=self.tokenizer,
            bootstrap=bootstrap,
            # Seal extension waits on this to confirm every queued
            # trigger-round submission has finished GRAIL before firing.
            queue_drained_predicate=lambda: self.server._submit_queue.empty(),
        )
        cp = self._checkpoint_store.current_manifest()
        self._active_batcher.current_checkpoint_hash = (
            cp.revision if cp else ""
        )

    def _activate_window(self) -> None:
        """Expose the prepared batcher to the HTTP server and mark OPEN.

        Must be called only after ``_set_window_randomness`` has populated
        ``self._active_batcher.randomness``; otherwise miner submissions
        arriving in the window between OPEN and a later randomness set
        would fail verification with ``Empty randomness hex string``.
        """
        if self._active_batcher is None:
            return
        self.server.set_active_batcher(self._active_batcher)
        self._set_state(WindowState.OPEN)

    async def _set_window_randomness(self, subtensor) -> None:
        """Populate the active batcher's per-window randomness seed.

        GRAIL sketch verification re-derives challenge indices from this
        seed; miner and validator must agree. The miner derives it from
        the same block hash + drand round, so the values match bit-for-bit.

        Retries on transient substrate failures (finney returning HTTP 503
        or WebSocket handshake errors) before bubbling. Without retries,
        any blip costs us the full window — the new two-phase open keeps
        the failure clean (no zombie accepts) but still leaves the window
        empty. A small in-loop retry recovers transparently from the
        sub-second blips that dominate the failure mode in practice.
        """
        if self._active_batcher is None:
            return
        # 3 attempts total: original + 2 retries. Backoff is 0.5s then 1.0s,
        # so worst-case added latency is 1.5s — well inside the 60s window
        # budget. Sustained outages still bubble after attempt 3.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                randomness, beacon = await self._derive_randomness(
                    subtensor, self._window_n,
                )
                self._active_batcher.randomness = randomness
                self._last_beacon = beacon
                # Schedule background bittensor_drand cross-check. Only
                # in real-drand mode (mock path returns beacon=None).
                # Task reference stored so it's not GC'd mid-run.
                if beacon is not None and beacon.get("signature"):
                    from reliquary.infrastructure.drand import get_current_chain
                    chain_info = get_current_chain()
                    self._verify_task = asyncio.create_task(
                        self._verify_beacon_async(
                            self._active_batcher,
                            chain_info["hash"],
                            int(beacon["round"]),
                            str(beacon["randomness"]),
                            beacon["signature"],
                        )
                    )
                if attempt > 0:
                    logger.info(
                        "Window %d: randomness derived on attempt %d",
                        self._window_n, attempt + 1,
                    )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    logger.warning(
                        "Window %d: _derive_randomness attempt %d failed (%s: %s); retrying",
                        self._window_n, attempt + 1,
                        type(exc).__name__, str(exc)[:120],
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    async def _verify_beacon_async(
        self,
        batcher,
        chain_hash: str,
        round_number: int,
        randomness: str,
        signature: str | None,
    ) -> None:
        """Background bittensor_drand cross-check for the just-fetched beacon.

        Runs ``verify_beacon_signature`` in a worker thread (it's blocking
        I/O — fetches an independent signature from a second drand relay
        and byte-compares). On any failure (mismatch, network error, library
        crash) flips ``batcher.beacon_invalid`` so ``_train_and_publish``
        drops the window before sealing.
        """
        from reliquary.infrastructure.drand import verify_beacon_signature
        try:
            ok = await asyncio.to_thread(
                verify_beacon_signature, chain_hash, round_number, randomness, signature,
            )
        except Exception:
            logger.exception(
                "Beacon verification crashed for round %d (window %d); invalidating window",
                round_number, self._window_n,
            )
            batcher.beacon_invalid = True
            return
        if not ok:
            logger.error(
                "Beacon verification FAILED post-OPEN for round %d; invalidating window %d",
                round_number, self._window_n,
            )
            batcher.beacon_invalid = True

    async def _train_and_publish(self) -> None:
        """TRAINING + PUBLISHING + READY phases."""
        if self._active_batcher is None:
            logger.warning("_train_and_publish called with no active batcher")
            return

        # Background drand cross-check flips beacon_invalid if the beacon
        # was forged or the verify crashed. Await up to 2s for its verdict
        # before checking — by seal-time (~3s after OPEN) it's almost always
        # done. Plain wait_for (no shield): if it times out, cancel the task
        # and check the flag below with whatever state it reached.
        if self._verify_task is not None and not self._verify_task.done():
            try:
                await asyncio.wait_for(self._verify_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Window %d: drand verify still running at train-time; "
                    "proceeding without final verdict (will check flag below)",
                    self._window_n,
                )
        if self._active_batcher.beacon_invalid:
            logger.error(
                "Window %d: dropping seal+train+archive — beacon invalid",
                self._window_n,
            )
            self.server.set_active_batcher(None)
            self._active_batcher = None
            self._set_state(WindowState.READY)
            return

        self._set_state(WindowState.TRAINING)
        # v2.3: seal_batch orders by the per-submission drand_round attached
        # by miners (see design A'). The validator does no post-close drand
        # fetch — all timing info is already attached to the submissions.
        batch, rewards = self._active_batcher.seal_batch()
        self._active_batcher.rewards_by_hotkey = rewards
        # Note: miners earn slots regardless of training. Their contribution
        # is reflected in the next ``_submit_weights`` call, which replays
        # the EMA from R2 archives written by ``_archive_window`` below.

        # Only train on a full batch. A partial seal (timeout) means miner
        # population + cadence didn't produce enough groups to train on;
        # stepping on a smaller-than-target batch gives a noisier gradient
        # and a different effective LR than full-batch steps, so we skip.
        # The miners who did submit are still credited via _update_ema.
        trained = len(batch) >= B_BATCH
        # Env-controlled skip: ``RELIQUARY_DISABLE_TRAIN=1`` bypasses the
        # train_step call entirely. Useful when the validator is configured
        # in inference-only mode (e.g. a frozen policy phase) or when the
        # train_step has a known OOM/leak pattern that's poisoning the
        # GPU pool across windows. With this flag set we proceed straight
        # to archive + skip-publish, exactly like a partial-seal path.
        skip_train = os.environ.get("RELIQUARY_DISABLE_TRAIN", "").lower() in {"1", "true", "yes", "on"}
        if trained and skip_train:
            logger.info(
                "Window %d: RELIQUARY_DISABLE_TRAIN set — skipping train_step + publish",
                self._window_n,
            )
            trained = False
        elif trained:
            try:
                self.train_model = train_step(
                    self.train_model, batch,
                    ref_model=self.verify_model,
                    window_index=self._window_n,
                )
            except Exception:
                # Don't let a training failure (e.g. CUDA OOM) skip
                # _archive_window — miners still need their R2 contribution
                # recorded so the EMA / on-chain weights reflect this window.
                logger.exception(
                    "train_step failed for window %d; archiving anyway and "
                    "skipping publish", self._window_n,
                )
                trained = False
            finally:
                # Reclaim any GPU memory the failed/successful train_step
                # held in its activation cache. This is critical when
                # train_step OOMs intermittently — without explicit cleanup
                # the partial allocations fragment the CUDA pool over
                # successive windows and eventually starve verify_commitment.
                _try_empty_cuda_cache()
        else:
            logger.info(
                "Window %d sealed with %d/%d submissions — skipping train_step + publish",
                self._window_n, len(batch), B_BATCH,
            )

        self._set_state(WindowState.PUBLISHING)
        if trained:
            # checkpoint_n only advances on publish; use window_n for cadence.
            next_n = self._checkpoint_n + 1
            # Push to HF every N windows, or immediately if no checkpoint exists yet.
            should_publish = (
                (self._window_n % self._publish_every == 0)
                or self._checkpoint_store.current_manifest() is None
            )
            if should_publish:
                try:
                    entry = await self._checkpoint_store.publish(
                        checkpoint_n=next_n, model=self.train_model,
                    )
                    self._checkpoint_n = next_n
                    self.server.set_current_checkpoint(entry)
                    # Refresh verify_model in-place so the next window's
                    # batcher verifies miners against the just-published
                    # checkpoint. In-place copy: no new allocation.
                    try:
                        self.verify_model.load_state_dict(
                            self.train_model.state_dict()
                        )
                    except (AttributeError, RuntimeError):
                        logger.exception(
                            "verify_model refresh failed; verify_model now "
                            "stale wrt checkpoint %d", entry.checkpoint_n,
                        )
                    logger.info(
                        "Published checkpoint %d to %s@%s and refreshed verify_model",
                        entry.checkpoint_n, entry.repo_id, entry.revision[:12],
                    )
                except Exception:
                    logger.exception("HF publish failed; staying on previous checkpoint")
            else:
                logger.info(
                    "Skipping HF publish for window_n=%d (publishing every %d)",
                    self._window_n, self._publish_every,
                )

        try:
            await self._archive_window(self._active_batcher, batch)
        except Exception:
            logger.exception("window archive failed")

        self.server.set_active_batcher(None)
        self._active_batcher = None
        self._set_state(WindowState.READY)

    async def _archive_window(self, batcher, batch) -> None:
        window_opened_at = getattr(batcher, "window_opened_at", None)
        eos_id = (
            getattr(self.tokenizer, "eos_token_id", None)
            if self.tokenizer is not None else None
        )

        def _resp_time(arrived_at: float) -> float | None:
            if window_opened_at is None or not arrived_at:
                return None
            return arrived_at - window_opened_at

        def _rollout_payload(s, with_text: bool):
            out = []
            texts = s.completion_texts if with_text else [None] * len(s.rollouts)
            # rollout_hashes is populated at accept-time; for legacy paths
            # (e.g. test fixtures bypassing _accept_locked) it may be empty,
            # in which case we omit the `hash` field rather than guessing.
            hashes = s.rollout_hashes if s.rollout_hashes else [None] * len(s.rollouts)
            for r, text, h in zip(s.rollouts, texts, hashes):
                tokens = list(r.tokens)
                rollout_dict = (r.commit or {}).get("rollout", {}) or {}
                prompt_length = int(rollout_dict.get("prompt_length", 0))
                completion_length = int(rollout_dict.get(
                    "completion_length", max(0, len(tokens) - prompt_length),
                ))
                eos_terminated = (
                    bool(tokens) and eos_id is not None and tokens[-1] == eos_id
                )
                entry = {
                    "tokens": tokens,
                    "reward": r.reward,
                    "completion_length": completion_length,
                    "eos_terminated": eos_terminated,
                }
                if h is not None:
                    entry["hash"] = h.hex()
                if with_text:
                    entry["completion_text"] = text
                out.append(entry)
            return out

        batched_keys = {(s.hotkey, s.prompt_idx) for s in batch}

        batch_entries = []
        for s in batch:
            problem = self.env.get_problem(s.prompt_idx)
            batch_entries.append({
                "hotkey": s.hotkey,
                "prompt_idx": s.prompt_idx,
                "sigma": s.sigma,
                "prompt": problem.get("prompt", ""),
                "ground_truth": problem.get("ground_truth", ""),
                "rollouts": _rollout_payload(s, with_text=True),
                "response_time": _resp_time(s.arrived_at),
                "merkle_root": s.merkle_root_bytes.hex(),
                "claimed_checkpoint_hash": s.claimed_checkpoint_hash,
                "sketch_diff_max": s.sketch_diff_max,
                "lp_dev_max": s.lp_dev_max,
                "dist_q10_min": s.dist_q10_min,
            })

        # All validated submissions that didn't make the final batch — metadata
        # only (no rollouts/text, no prompt) so miners can see their participation
        # without ballooning the dataset size.
        runners_up = []
        for s in batcher.valid_submissions():
            key = (s.hotkey, s.prompt_idx)
            if key in batched_keys:
                continue
            runners_up.append({
                "hotkey": s.hotkey,
                "prompt_idx": s.prompt_idx,
                "sigma": s.sigma,
                "response_time": _resp_time(s.arrived_at),
                "merkle_root": s.merkle_root_bytes.hex(),
                "sketch_diff_max": s.sketch_diff_max,
                "lp_dev_max": s.lp_dev_max,
                "dist_q10_min": s.dist_q10_min,
            })

        rejected_entries = [
            {
                "hotkey": r.hotkey,
                "prompt_idx": r.prompt_idx,
                "reason": r.reason,
                "sketch_diff_max": r.sketch_diff_max,
                "lp_dev_max": r.lp_dev_max,
                "dist_q10_min": r.dist_q10_min,
            }
            for r in getattr(batcher, "rejected_submissions", [])
        ]

        archive = {
            "window_start": batcher.window_start,
            "validator_hotkey": self.wallet.hotkey.ss58_address,  # provenance
            "randomness": batcher.randomness,
            "environment": self.env.name,
            "batch": batch_entries,
            "runners_up": runners_up,
            "reject_summary": dict(getattr(batcher, "reject_counts", {})),
            "rejected": rejected_entries,
            # v2.3: per-hotkey emission share from select_batch_and_distribute.
            # All miners whose prompt landed in the winning set appear here,
            # even if their specific submission wasn't picked for training.
            "rewards_by_hotkey": dict(getattr(batcher, "rewards_by_hotkey", {})),
            "late_drops": {
                hk: dict(counts) for hk, counts in self._late_drops.items()
            },
        }
        # Reset the in-memory counter for the next window. New events
        # arriving while this window's payload is uploading land in the
        # fresh dict and will appear in the next archive.
        self._late_drops.clear()
        # Non-blocking archive: enqueue to disk and return immediately.
        # The background ``ArchiveQueue`` worker (started in run()) picks
        # this up and uploads via the same sync-boto3 path used in
        # storage.upload_window_dataset, with persistent retry-on-failure.
        # Main-loop window iteration is unblocked even if R2 is down for
        # hours, and queued payloads survive process restarts.
        from reliquary.infrastructure.archive_queue import get_archive_queue
        get_archive_queue().enqueue(batcher.window_start, archive)

    async def run(self, subtensor) -> None:
        await self.server.start()
        await self._serve_axon_on_chain(subtensor)
        await self._apply_resume_from()                  # ← resume before bootstrap
        await self._bootstrap_state_from_external()
        await self._rebuild_cooldown_from_history()
        await self._rebuild_hashes_from_history()

        # Start the background archive-upload worker. It scans the queue
        # directory for any pending payloads (from before this restart
        # or accumulated during R2 downtime) and uploads them via sync
        # boto3 with exponential backoff. Cancelled cleanly on shutdown.
        from reliquary.infrastructure.archive_queue import get_archive_queue
        self._archive_worker_task = asyncio.create_task(
            get_archive_queue().run_forever(),
            name="archive_queue_worker",
        )

        logger.info(
            "Validator started (v2.1): env=%s, netuid=%d, http=%s:%d",
            self.env.name, self.netuid, self.server.host, self.server.port,
        )
        # Build marker — uniquely identifies the deployed code version in
        # logs after an auto-deploy (watchtower). Bump on every commit
        # that ships new behavior; greppable via:
        #   docker logs reliquary-trainer | grep "Reliquary build:"
        logger.info("Reliquary build: r2-reliability-suite (Layers 1+2+3)")
        try:
            while True:
                try:
                    self._open_window()
                    await self._wait_for_next_drand_boundary()
                    await self._set_window_randomness(subtensor)
                    self._activate_window()
                    try:
                        await asyncio.wait_for(
                            self._active_batcher.seal_event.wait(),
                            timeout=WINDOW_TIMEOUT_SECONDS,
                        )
                        logger.info(
                            "Window %d sealed (B valid received)",
                            self._window_n,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Window %d timed out at %ds — sealing partial",
                            self._window_n, WINDOW_TIMEOUT_SECONDS,
                        )

                    await self._train_and_publish()

                    # set_weights is owned by a concurrent WeightOnlyValidator
                    # task running off the same R2 archives; no need to do it
                    # here. The trainer is purely about training + uploads.
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Window iteration failed")
                    # Reset to READY so the next iteration doesn't spin on error state.
                    self.server.set_active_batcher(None)
                    self._active_batcher = None
                    self._set_state(WindowState.READY)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
        finally:
            # Cancel the archive worker and let it drain in-flight uploads
            # before we tear down the server. The worker survives many
            # window cycles so we shut it down deliberately rather than
            # waiting for process exit to GC it.
            task = getattr(self, "_archive_worker_task", None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            await self.server.stop()
            telemetry.finish()

    async def _serve_axon_on_chain(self, subtensor) -> None:
        """Publish this validator's axon (ip:port) to the chain metagraph.

        Miners read `metagraph.axons[uid].ip/port` via `discover_validator_url`
        to route their submissions. Skipped with a warning when no external
        address is configured — miners then need `--validator-url` overrides
        to find this validator.
        """
        if not self.external_ip or not self.external_port:
            logger.warning(
                "serve_axon skipped: no external_ip/external_port provided. "
                "Miners won't discover this validator via metagraph; use "
                "--validator-url on the miner side."
            )
            return
        try:
            import bittensor as bt

            axon = bt.Axon(
                wallet=self.wallet,
                ip=self.external_ip,
                port=self.external_port,
                external_ip=self.external_ip,
                external_port=self.external_port,
            )
            response = await subtensor.serve_axon(
                netuid=self.netuid,
                axon=axon,
                wait_for_inclusion=True,
                wait_for_finalization=False,
                raise_error=False,
            )
            success = getattr(response, "is_success", None)
            if success is False:
                logger.error(
                    "serve_axon failed: %s:%d not published (response=%s). "
                    "Likely: hotkey not registered on netuid %d, or chain rejected.",
                    self.external_ip, self.external_port, response, self.netuid,
                )
                return
            logger.info(
                "serve_axon published: %s:%d announced on netuid %d",
                self.external_ip, self.external_port, self.netuid,
            )
        except Exception:
            logger.exception(
                "serve_axon threw — miners will have to use --validator-url"
            )

    async def _bootstrap_state_from_external(self) -> None:
        """Derive window_n and checkpoint_n from R2 + HF.

        Called once at startup before the main loop. Miner scoring (EMA) is
        no longer bootstrapped here — ``_submit_weights`` recomputes it from
        R2 archives at every submit, which keeps the trainer in lock-step
        with weight-only validators replaying the same archives.
        """
        # 1. window_n from R2 archive keys
        try:
            windows = await storage.list_all_window_keys()
            if windows:
                self._window_n = max(windows)
                logger.info("Bootstrapped window_n=%d from R2", self._window_n)
            else:
                logger.info("No archives in R2 — starting from window_n=0")
        except Exception:
            logger.exception("Failed to bootstrap window_n from R2; starting at 0")

        # 2. checkpoint_n + revision from HF commit history.
        #
        # Auto-resume to the latest published "checkpoint N" commit. This
        # replaces the previous count-only logic, which left
        # ``_checkpoint_store._current`` populated by whatever
        # ``RELIQUARY_RESUME_FROM`` was baked into the container env.
        # A stale env var (e.g. set to an early checkpoint when the
        # validator was first deployed) caused the validator to regress
        # 19 published checkpoints (ckpt 45 → ckpt 26) on the PR #23
        # redeploy, throwing away hours of training progress that was
        # still safely on HF. HF is the durable source of truth — read
        # it on every startup.
        #
        # Operator override semantics:
        #   * No env var set: pick the latest HF checkpoint
        #   * env var set, ENV ckpt >= HF latest: keep the env (operator
        #     pinned to something they want, possibly under test)
        #   * env var set, ENV ckpt <  HF latest: warn and override with
        #     HF latest (the env is stale; HF has progressed past it)
        try:
            import re as _re
            from huggingface_hub import HfApi
            repo_id = self._checkpoint_store.repo_id
            api = HfApi()
            commits = api.list_repo_commits(repo_id=repo_id)
            ckpt_title = _re.compile(r"^checkpoint\s+(\d+)\s*$", _re.IGNORECASE)
            latest_n = -1
            latest_sha: str | None = None
            count = 0
            for c in commits:
                m = ckpt_title.match(c.title or "")
                if not m:
                    continue
                count += 1
                n = int(m.group(1))
                if n > latest_n:
                    latest_n = n
                    latest_sha = c.commit_id
            if latest_n < 0:
                logger.info(
                    "Bootstrap: no 'checkpoint N' commits on %s; keeping base",
                    repo_id,
                )
                return
            # When ``_apply_resume_from`` already installed a manifest from
            # ``RELIQUARY_RESUME_FROM``, ``self._checkpoint_n`` carries that
            # ckpt number (set on line 334 of _apply_resume_from). Treat env
            # >= HF as "operator-pinned, leave it".
            resumed_from_env = self._checkpoint_store.current_manifest() is not None
            if resumed_from_env and self._checkpoint_n >= latest_n:
                logger.info(
                    "Bootstrap: env-resumed at ckpt=%d ≥ HF latest=%d; "
                    "trusting operator pin",
                    self._checkpoint_n, latest_n,
                )
                return
            # HF has a newer checkpoint than env (or env was unset).
            # Override _resume_from and re-run _apply_resume_from to load
            # the right weights into both train_model and verify_model.
            if resumed_from_env:
                logger.warning(
                    "Bootstrap: env-resumed at ckpt=%d but HF has ckpt=%d "
                    "(sha=%s) — overriding env to avoid regression. Set "
                    "RELIQUARY_RESUME_FROM=sha:%s to silence this warning, "
                    "or unset it to always track HF latest.",
                    self._checkpoint_n, latest_n,
                    latest_sha[:12] if latest_sha else "?",
                    latest_sha,
                )
            else:
                logger.info(
                    "Bootstrap: no env resume; auto-resuming from latest HF "
                    "ckpt=%d (sha=%s, %d total ckpt commits)",
                    latest_n, latest_sha[:12] if latest_sha else "?", count,
                )
            self._resume_from = f"sha:{latest_sha}"
            await self._apply_resume_from()
        except Exception:
            logger.exception(
                "Failed to auto-discover latest HF checkpoint; "
                "validator stays on whatever --resume-from gave us"
            )

    async def _rebuild_cooldown_from_history(self) -> None:
        """At startup, reconstruct CooldownMap from the last
        COOLDOWN_REBUILD_LOOKBACK archived windows on R2 (bounded cap; not BATCH_PROMPT_COOLDOWN_WINDOWS which is now astronomical for one-shot semantics).

        R2 is the durable source of truth — each window's sealed batch is
        uploaded by ``_archive_window``. Rebuilding from that history means:
          * local disk state isn't needed (no JSON file to manage)
          * multi-validator consistency: any validator rebuilding from the
            same R2 prefix converges to the same cooldown map
          * a fresh validator joining an active subnet picks up the
            current cooldown state without coordination
        """
        try:
            current_window = self._window_n
            archives = await storage.list_recent_datasets(
                current_window=current_window + 1,
                n=COOLDOWN_REBUILD_LOOKBACK,
            )
            self._cooldown_map.rebuild_from_history(
                archives, current_window=current_window,
            )
            logger.info(
                "Rebuilt cooldown from %d archive windows (current=%d, map size=%d)",
                len(archives), current_window, len(self._cooldown_map),
            )
        except Exception:
            logger.exception(
                "Failed to rebuild cooldown from history; starting with empty state"
            )

    async def _rebuild_hashes_from_history(self) -> None:
        """Rebuild ``self._hash_set`` from the last HASH_DEDUP_RETENTION_WINDOWS
        archives. Horizon is independent of cooldown — see constants docstring.
        Compat path covers pre-feature archives (no ``hash`` field) by
        recomputing from ``tokens``.
        """
        try:
            current_window = self._window_n
            archives = await storage.list_recent_datasets(
                current_window=current_window + 1,
                n=HASH_DEDUP_RETENTION_WINDOWS,
            )
            self._hash_set.rebuild_from_history(
                archives, current_window=current_window,
            )
            logger.info(
                "Rebuilt hash set from %d archive windows (current=%d, size=%d)",
                len(archives), current_window, len(self._hash_set),
            )
        except Exception:
            logger.exception(
                "Failed to rebuild hash set from history; starting empty"
            )

    async def _wait_for_next_drand_boundary(self) -> None:
        """Align window OPEN to the next drand round boundary.

        Called between ``_open_window`` (which prepares the batcher) and
        ``_set_window_randomness`` (which fetches σ_R for the round that
        publishes at — or just after — the boundary). Aligning here means
        ``randomness_grail`` is bound to a round that didn't exist when
        miners might have tried to pre-generate. Closes the v30-style
        pre-spam exploit.
        """
        if not self.use_drand:
            return
        import time
        from reliquary.infrastructure.drand import get_current_chain
        ci = get_current_chain()
        delay = chain.seconds_until_next_drand_boundary(
            time.time(), ci["genesis_time"], ci["period"],
        )
        if delay > 0:
            logger.info(
                "Window %d: waiting %.2fs for next drand boundary before OPEN",
                self._window_n, delay,
            )
            await asyncio.sleep(delay)

    async def _derive_randomness(
        self, subtensor, target_window: int,
    ) -> tuple[str, dict | None]:
        """v2.3+: drand-only seed bound to the round publishing AT window OPEN.

        Returns ``(window_randomness, beacon_or_None)``. ``beacon`` is the
        raw drand beacon dict (``{round, randomness, signature, ...}``)
        when the drand path is active, so the caller can schedule a
        background bittensor_drand cross-check. ``None`` on the legacy
        mock path (no cross-check possible).

        Called after ``_wait_for_next_drand_boundary`` so the wall-clock-
        current drand round corresponds to the one whose σ just became
        publicly available. Miners cannot pre-fetch this σ because it
        didn't exist a few seconds ago.
        """
        if self.use_drand:
            import time
            from reliquary.infrastructure.drand import get_beacon, get_current_chain
            chain_info = get_current_chain()
            drand_round = chain.compute_current_drand_round(
                time.time(), chain_info["genesis_time"], chain_info["period"],
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            randomness = chain.compute_window_randomness(
                None, beacon["randomness"], drand_round=beacon["round"],
            )
            return randomness, beacon
        # Legacy mock-only path: still uses block_hash so tests that
        # disable drand keep working without a live drand fetch.
        block_hash = await chain.get_block_hash(subtensor, target_window)
        return chain.compute_window_randomness(block_hash), None


