"""Validator main loop — v2.1 batch-driven state machine (OPEN→TRAINING→PUBLISHING→READY)."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    B_BATCH,
    BOOTSTRAP_WINDOWS,
    CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
    CHECKPOINT_STAGING_DIR_DEFAULT,
    DEFAULT_HF_REPO_ID,
    GRAD_CLIP_NORM,
    KL_BETA,
    LEARNING_RATE,
    LR_COSINE_MAX_WINDOWS,
    LR_WARMUP_WINDOWS,
    M_ROLLOUTS,
    POLL_INTERVAL_SECONDS,
    PPO_CLIP_EPSILON,
    SUBNET_START_BLOCK,
    UID_BURN,
    VALIDATOR_HTTP_PORT,
    WANDB_TRAINING_VERSION,
    WEIGHT_SUBMISSION_INTERVAL,
    WINDOW_LENGTH,
    WINDOW_TIMEOUT_SECONDS,
)
from reliquary.environment.base import Environment
from reliquary.infrastructure import chain, storage
from reliquary.protocol.submission import RolloutSubmission, WindowState
from reliquary.validator import telemetry
from reliquary.validator.batcher import GrpoWindowBatcher, ValidSubmission
from reliquary.validator.checkpoint import CheckpointStore
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.server import ValidatorServer
from reliquary.validator.training import train_step

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH


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
    tokenizer,
    bootstrap: bool = False,
) -> GrpoWindowBatcher:
    """Instantiate a GrpoWindowBatcher for this window.

    ``cooldown_map`` is the validator's long-lived CooldownMap, shared
    across windows. Each window's sealed batch updates it via
    ``GrpoWindowBatcher.seal_batch``.
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
        bootstrap=bootstrap,
        completion_text_fn=_completion_text,
        canonical_prompt_tokens_fn=_canonical_prompt_tokens,
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
        self.model = model
        # Enable gradient checkpointing to reduce activation memory.
        # Harmless if already enabled or unsupported by the model.
        try:
            self.model.gradient_checkpointing_enable()
        except (AttributeError, NotImplementedError):
            logger.warning(
                "model does not support gradient_checkpointing_enable"
            )
        self.tokenizer = tokenizer
        self.env = env
        self.netuid = netuid
        self.use_drand = use_drand
        self.external_ip = external_ip
        self.external_port = external_port

        self._last_processed_window: int = -1
        self._windows_in_interval: int = 0
        # Last on-chain block at which we successfully submitted weights.
        # The next submit fires when current_block - this >= WEIGHT_SUBMISSION_INTERVAL.
        # Bootstrapped to the current block at startup so we don't submit on boot.
        self._last_weight_block: int = 0
        self._cooldown_map = CooldownMap(
            cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS
        )

        self.server = ValidatorServer(host=http_host, port=http_port)

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
        self._current_window_state: WindowState = WindowState.READY

        self._resume_from = resume_from
        self._load_model_fn = load_model_fn or _default_load_model

    def _set_state(self, s: WindowState) -> None:
        self._current_window_state = s
        # Also notify the server so /state returns the right value.
        self.server.set_current_state(s)

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
        # Load weights — this replaces the base model loaded at __init__.
        self.model = self._load_model_fn(local_path)
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
        """Create a new GrpoWindowBatcher and mark state OPEN.

        Increments ``self._window_n`` and wires the active checkpoint hash
        into the batcher so miners on stale checkpoints get WRONG_CHECKPOINT
        rejected before GRAIL compute. Per-window randomness is set
        separately by the async caller via ``_set_window_randomness`` —
        it needs the subtensor, which this sync path doesn't carry.
        """
        self._window_n += 1
        bootstrap = is_bootstrap_window(
            window_start=self._window_n,
            subnet_start=SUBNET_START_BLOCK,
        )
        self._active_batcher = open_grpo_window(
            window_start=self._window_n,
            env=self.env, model=self.model,
            cooldown_map=self._cooldown_map, tokenizer=self.tokenizer,
            bootstrap=bootstrap,
        )
        cp = self._checkpoint_store.current_manifest()
        self._active_batcher.current_checkpoint_hash = (
            cp.revision if cp else ""
        )
        self.server.set_active_batcher(self._active_batcher)
        self._set_state(WindowState.OPEN)

    async def _set_window_randomness(self, subtensor) -> None:
        """Populate the active batcher's per-window randomness seed.

        GRAIL sketch verification re-derives challenge indices from this
        seed; miner and validator must agree. The miner derives it from
        the same block hash + drand round, so the values match bit-for-bit.
        """
        if self._active_batcher is None:
            return
        self._active_batcher.randomness = await self._derive_randomness(
            subtensor, self._window_n,
        )

    async def _train_and_publish(self) -> None:
        """TRAINING + PUBLISHING + READY phases."""
        if self._active_batcher is None:
            logger.warning("_train_and_publish called with no active batcher")
            return

        self._set_state(WindowState.TRAINING)
        batch = self._active_batcher.seal_batch()
        # Note: miners earn slots regardless of training. Their contribution
        # is reflected in the next ``_submit_weights`` call, which replays
        # the EMA from R2 archives written by ``_archive_window`` below.

        # Only train on a full batch. A partial seal (timeout) means miner
        # population + cadence didn't produce enough groups to train on;
        # stepping on a smaller-than-target batch gives a noisier gradient
        # and a different effective LR than full-batch steps, so we skip.
        # The miners who did submit are still credited via _update_ema.
        trained = len(batch) >= B_BATCH
        if trained:
            self.model = train_step(self.model, batch, window_index=self._window_n)
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
                        checkpoint_n=next_n, model=self.model,
                    )
                    self._checkpoint_n = next_n
                    self.server.set_current_checkpoint(entry)
                    logger.info(
                        "Published checkpoint %d to %s@%s",
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
            for r, text in zip(s.rollouts, texts):
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
        }
        await storage.upload_window_dataset(batcher.window_start, archive)

    async def run(self, subtensor) -> None:
        await self.server.start()
        await self._serve_axon_on_chain(subtensor)
        await self._apply_resume_from()                  # ← resume before bootstrap
        await self._bootstrap_state_from_external()
        await self._rebuild_cooldown_from_history()
        # Anchor the weight-submission cadence at the current chain block so
        # a restart doesn't trigger a submit on the first iteration.
        try:
            self._last_weight_block = await chain.get_current_block(subtensor)
        except Exception:
            logger.exception(
                "Could not read current block at boot; _last_weight_block stays 0 "
                "(will submit on the first iteration)"
            )
        logger.info(
            "Validator started (v2.1): env=%s, netuid=%d, http=%s:%d",
            self.env.name, self.netuid, self.server.host, self.server.port,
        )
        try:
            while True:
                try:
                    self._open_window()
                    await self._set_window_randomness(subtensor)
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

                    # Weight cadence: every WEIGHT_SUBMISSION_INTERVAL on-chain
                    # blocks. We cannot key off window count because v2.1 windows
                    # are event-driven — a full window can take 20+ min = 100+
                    # blocks, so "every 72 windows" would fire every ~24 h instead
                    # of the protocol-intended ~72 min.
                    current_block = await chain.get_current_block(subtensor)
                    if current_block - self._last_weight_block >= WEIGHT_SUBMISSION_INTERVAL:
                        submitted = await self._submit_weights(subtensor)
                        if submitted:
                            self._last_weight_block = current_block
                        else:
                            logger.warning(
                                "set_weights did not succeed — EMA unchanged for next retry"
                            )
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

        # 2. checkpoint_n from HF commit history
        try:
            from huggingface_hub import HfApi
            repo_id = self._checkpoint_store.repo_id
            api = HfApi()
            commits = api.list_repo_commits(repo_id=repo_id)
            ckpt_count = sum(1 for c in commits if c.title.startswith("checkpoint "))
            self._checkpoint_n = ckpt_count
            logger.info("Bootstrapped checkpoint_n=%d from HF", self._checkpoint_n)
        except Exception:
            logger.exception("Failed to bootstrap checkpoint_n from HF; starting at 0")

    async def _rebuild_cooldown_from_history(self) -> None:
        """At startup, reconstruct CooldownMap from the last
        BATCH_PROMPT_COOLDOWN_WINDOWS archived windows on R2.

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
                n=BATCH_PROMPT_COOLDOWN_WINDOWS,
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

    async def _derive_randomness(self, subtensor, target_window: int) -> str:
        block_hash = await chain.get_block_hash(subtensor, target_window)
        if self.use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain
            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                target_window, chain_info["genesis_time"], chain_info["period"],
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                block_hash, beacon["randomness"], drand_round=beacon["round"],
            )
        return chain.compute_window_randomness(block_hash)

    async def _submit_weights(self, subtensor) -> bool:
        """Submit weights derived from R2 archives.

        Reads the same archives that any weight-only validator would replay
        (``WeightOnlyValidator``) so a trainer and a weight-only validator
        running side-by-side converge to identical weights for the same
        on-chain state. The trainer's in-memory EMA is no longer authoritative;
        R2 is the single source of truth for scoring.
        """
        from reliquary.validator.weight_only import WeightOnlyValidator

        windows = await storage.list_all_window_keys()
        if not windows:
            logger.info("No archives yet; nothing to submit")
            return False
        archives = await storage.list_recent_datasets(
            current_window=max(windows) + 1,
            n=ROLLING_WINDOWS * 3,
        )
        miner_weights = dict(WeightOnlyValidator._replay_ema(archives))
        total = sum(miner_weights.values())
        burn_weight = max(0.0, 1.0 - total)

        logger.info(
            "Submitting weights: %d miners (ema_total=%.4f), burn=%.4f",
            len(miner_weights), total, burn_weight,
        )
        for hk, w in sorted(miner_weights.items(), key=lambda x: -x[1])[:10]:
            logger.info("  %s: %.6f", hk[:8], w)

        meta = await chain.get_metagraph(subtensor, self.netuid)
        hotkey_to_uid = dict(zip(meta.hotkeys, meta.uids))
        uids: list[int] = []
        weight_vals: list[float] = []
        for hk, w in miner_weights.items():
            if hk in hotkey_to_uid and w > 0:
                uids.append(int(hotkey_to_uid[hk]))
                weight_vals.append(w)
        if burn_weight > 0:
            uids.append(UID_BURN)
            weight_vals.append(burn_weight)
        if not uids:
            logger.info("No non-zero weights to submit; nothing to do.")
            return True
        return await chain.set_weights(
            subtensor, self.wallet, self.netuid, uids, weight_vals,
        )

