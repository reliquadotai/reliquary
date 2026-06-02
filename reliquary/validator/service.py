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
    BOOTSTRAP_SIGMA_MIN,
    CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
    CHECKPOINT_STAGING_DIR_DEFAULT,
    DEFAULT_HF_REPO_ID,
    DRAND_ROUND_BACKWARD_TOLERANCE,
    ENVIRONMENT_MIX,
    GRAD_CLIP_NORM,
    HASH_DEDUP_RETENTION_WINDOWS,
    KL_BETA,
    LEARNING_RATE,
    LOGPROB_IS_EPS,
    LR_COSINE_MAX_WINDOWS,
    LR_WARMUP_WINDOWS,
    M_ROLLOUTS,
    MIN_EOS_PROBABILITY,
    POLL_INTERVAL_SECONDS,
    PPO_CLIP_EPSILON,
    MAX_PROOF_CANDIDATES_PER_WINDOW,
    MAX_SEAL_QUEUE_DRAIN_SECONDS,
    PROOF_ADMISSION_STALL_POLL_SECONDS,
    SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS,
    SPARSE_VALID_IDLE_SEAL_SECONDS,
    SPARSE_VALID_MAX_WINDOW_SECONDS,
    SIGMA_MIN,
    SUBNET_START_BLOCK,
    VALIDATOR_HTTP_PORT,
    WANDB_TRAINING_VERSION,
    WINDOW_LENGTH,
    WINDOW_TIMEOUT_SECONDS,
)
from reliquary.environment import load_environments
from reliquary.environment.base import Environment
from reliquary.infrastructure import chain, storage
from reliquary.protocol.submission import RolloutSubmission, WindowState
from reliquary.validator import telemetry
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.checkpoint import CheckpointStore
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.dedup import RolloutHashSet
from reliquary.validator.observability import log_structured, runtime_revision
from reliquary.validator.quarantine import assess_training_batch
from reliquary.validator.server import ValidatorServer
from reliquary.validator.training import train_step

logger = logging.getLogger(__name__)


def _filter_archives_for_env(archives: list[dict], env_name: str) -> list[dict]:
    """Return a filtered view of archives containing only entries for ``env_name``.

    Handles both old (pre-multi-env) and new archive shapes:
      * Old: top-level ``"environment"`` (singular), no per-entry ``env_name``.
            All batch entries belong to that env.
      * New: per-entry ``"env_name"`` field. Filter to matching entries.
    """
    out = []
    for archive in archives:
        # Determine the archive's env(s). New shape has "environments" list;
        # old shape has "environment" singular. Both may be present together.
        archive_envs: list[str] = archive.get("environments") or []
        if not archive_envs:
            singular = archive.get("environment", "")
            if singular:
                archive_envs = [singular]

        # If env info is absent entirely, include all entries (defensive).
        env_unknown = not archive_envs

        # Filter batch entries to this env.
        if env_unknown or env_name in archive_envs:
            filtered_batch = []
            for entry in archive.get("batch", []):
                entry_env = entry.get("env_name", "")
                # Include if: entry has no env_name (old archive) or matches.
                if not entry_env or entry_env == env_name:
                    filtered_batch.append(entry)
            if filtered_batch:
                out.append({
                    "window_start": archive["window_start"],
                    "batch": filtered_batch,
                })
    return out


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
        tokens = rollout.commit["tokens"]
        return tokenizer.decode(tokens[prompt_len:])

    def _canonical_prompt_tokens(prompt_idx: int) -> list[int]:
        from reliquary.protocol.tokens import encode_prompt

        problem = env.get_problem(prompt_idx)
        return encode_prompt(tokenizer, problem["prompt"])

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
    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared.modeling import load_text_generation_model

    return load_text_generation_model(
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
        env: Environment | None = None,
        netuid: int = 0,
        *,
        use_drand: bool = True,
        http_host: str = "0.0.0.0",
        http_port: int = VALIDATOR_HTTP_PORT,
        external_ip: str | None = None,
        external_port: int | None = None,
        hf_repo_id: str | None = None,
        resume_from: str | None = None,
        load_model_fn: Any | None = None,
        env_mix: list[tuple[str, int]] | None = None,
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
        self.netuid = netuid
        self.use_drand = use_drand
        self.external_ip = external_ip
        self.external_port = external_port
        self.hf_repo_id = hf_repo_id or DEFAULT_HF_REPO_ID

        # Multi-env setup. ``env_mix`` defaults to ENVIRONMENT_MIX from
        # constants; callers (CLI, tests) may pass a single-entry mix or a
        # custom one. When a legacy ``env`` is supplied it overrides the mix
        # with a single-env config so existing call sites keep working.
        if env is not None:
            # Legacy single-env path: wrap the provided env in a 1-entry mix.
            _env_name = getattr(env, "name", "unknown")
            self.env_mix: list[tuple[str, int]] = [(_env_name, B_BATCH)]
            self.envs: dict[str, Environment] = {_env_name: env}
        else:
            self.env_mix = env_mix if env_mix is not None else list(ENVIRONMENT_MIX)
            env_names = [n for n, _ in self.env_mix]
            self.envs = load_environments(env_names)

        # Legacy accessor — archive code and tests grew up around single-env.
        # Points to the first env in the mix; consumers needing all envs
        # iterate ``self.envs``.
        first_env_name = self.env_mix[0][0]
        self.env: Environment = self.envs[first_env_name]

        self._last_processed_window: int = -1
        self._windows_in_interval: int = 0
        # One CooldownMap per env so prompt-cooldown is independent across
        # environments (a math prompt cooling down doesn't block code prompts).
        self._cooldown_per_env: dict[str, CooldownMap] = {
            name: CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
            for name in self.envs
        }
        # Legacy accessor pointing to the first env's map.  Kept so
        # ``_rebuild_cooldown_from_history`` and tests that read ``_cooldown_map``
        # still work without change.
        self._cooldown_map = self._cooldown_per_env[first_env_name]
        self._hash_set = RolloutHashSet(
            retention_windows=HASH_DEDUP_RETENTION_WINDOWS,
        )
        self._late_drops: dict[str, dict[str, int]] = {}
        self._grader_failures: dict[str, int] = {}

        self.server = ValidatorServer(host=http_host, port=http_port)
        self.server.set_late_drop_callback(self.record_late_drop)

        # v2.1 state machine infrastructure — in-memory only, bootstrapped at
        # startup from R2 + HF (no local JSON state file).
        self._window_n: int = 0
        self._checkpoint_n: int = 0
        self._publish_every = CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
        self._trained_windows_since_publish = 0
        self._checkpoint_store = CheckpointStore(
            validator_hotkey=wallet.hotkey.ss58_address,
            wallet=wallet,
            repo_id=self.hf_repo_id,
            staging_dir_path=CHECKPOINT_STAGING_DIR_DEFAULT,
            tokenizer=tokenizer,
        )
        # Multi-batcher: one GrpoWindowBatcher per active env.
        self._active_batchers: dict[str, GrpoWindowBatcher] = {}
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

    @property
    def _active_batcher(self):
        """Legacy scalar accessor: first active batcher, or None.

        Kept for test backward-compatibility. Production code iterates
        ``self._active_batchers`` directly.
        """
        d = self.__dict__.get("_active_batchers", {})
        return next(iter(d.values()), None)

    @_active_batcher.setter
    def _active_batcher(self, value) -> None:
        """Legacy setter: syncs a single batcher into ``_active_batchers``.

        Setting to None clears the dict; setting to a batcher wraps it in
        a single-entry dict keyed by the batcher's env name (or "unknown").
        """
        if value is None:
            self.__dict__.setdefault("_active_batchers", {}).clear()
        else:
            env_name = getattr(getattr(value, "env", None), "name", "unknown")
            self.__dict__["_active_batchers"] = {env_name: value}

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
        """Create GrpoWindowBatchers (one per env) in a non-active state.

        Builds all batchers and wires the active checkpoint hash, but does
        NOT expose them to the HTTP server yet — call ``_activate_window``
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
        cp = self._checkpoint_store.current_manifest()
        cp_hash = cp.revision if cp else ""
        self._active_batchers = {}
        for env_name, env in self.envs.items():
            batcher = open_grpo_window(
                window_start=self._window_n,
                env=env, model=self.verify_model,
                cooldown_map=self._cooldown_per_env[env_name],
                hash_set=self._hash_set,
                tokenizer=self.tokenizer,
                bootstrap=bootstrap,
                # Seal extension waits on this to confirm every queued
                # trigger-round submission has finished GRAIL before firing.
                queue_drained_predicate=lambda: self.server._submit_queue.empty(),
            )
            batcher.current_checkpoint_hash = cp_hash
            self._active_batchers[env_name] = batcher

    def _activate_window(self) -> None:
        """Expose all prepared batchers to the HTTP server and mark OPEN.

        Must be called only after ``_set_window_randomness`` has populated
        randomness on every batcher; otherwise miner submissions arriving
        between OPEN and a later randomness set would fail verification.
        """
        if not self._active_batchers:
            return
        self.server.set_active_batchers(self._active_batchers)
        self._set_state(WindowState.OPEN)

    def _proof_admission_exhausted_and_drained(self, batcher) -> bool:
        """True when bounded proof admission cannot fill this window anymore."""
        if batcher is None or batcher.is_sealed():
            return False
        distinct_valid = self._distinct_valid_prompt_count(batcher)
        if distinct_valid >= B_BATCH:
            return False
        if (
            getattr(batcher, "proof_admission_count", 0)
            < MAX_PROOF_CANDIDATES_PER_WINDOW
        ):
            return False
        queue_depth = int(getattr(self.server, "submit_queue_depth", 0) or 0)
        inflight = int(getattr(self.server, "proof_verification_inflight", 0) or 0)
        return queue_depth == 0 and inflight == 0

    def _distinct_valid_prompt_count(self, batcher) -> int:
        """Best-effort distinct prompt count for liveness decisions."""
        counter = getattr(batcher, "distinct_valid_prompt_count", None)
        if callable(counter):
            return int(counter())
        return int(getattr(batcher, "valid_count", 0) or 0)

    def _duplicate_prompt_shortfall_drained(self, batcher) -> bool:
        """True when duplicates filled raw submissions but not trainable slots."""
        if batcher is None or batcher.is_sealed():
            return False
        if getattr(batcher, "_seal_trigger_round", None) is not None:
            return False
        valid_count = int(getattr(batcher, "valid_count", 0) or 0)
        distinct_valid = self._distinct_valid_prompt_count(batcher)
        if valid_count < B_BATCH or distinct_valid >= B_BATCH:
            return False
        queue_depth = int(getattr(self.server, "submit_queue_depth", 0) or 0)
        inflight = int(getattr(self.server, "proof_verification_inflight", 0) or 0)
        return queue_depth == 0 and inflight == 0

    def _queue_and_proofs_drained(self) -> bool:
        queue_depth = int(getattr(self.server, "submit_queue_depth", 0) or 0)
        inflight = int(getattr(self.server, "proof_verification_inflight", 0) or 0)
        return queue_depth == 0 and inflight == 0

    def _seconds_since_last_valid_submission(self, batcher) -> float | None:
        counter = getattr(batcher, "seconds_since_last_valid_submission", None)
        if callable(counter):
            return counter()
        return None

    def _window_open_age_seconds(self, batcher) -> float | None:
        opened_at = getattr(batcher, "window_opened_at", None)
        time_fn = getattr(batcher, "_time_fn", None)
        if opened_at is None or not callable(time_fn):
            return None
        return max(0.0, float(time_fn()) - float(opened_at))

    def _sparse_valid_liveness_reason(self, batcher) -> str | None:
        """Return force-seal reason for sparse valid windows, if any.

        This is a cadence guard, not a quality gate. It only fires when the
        validator has fewer than B distinct trainable prompts, no queued or
        in-flight proof work, and either no valid progress for the sparse idle
        threshold or an overlong sparse window. Zero-valid windows are included
        only for the max-age path so a hard reset with stale miners cannot
        freeze checkpoint progress indefinitely.
        """
        if batcher is None or batcher.is_sealed():
            return None
        if getattr(batcher, "_seal_trigger_round", None) is not None:
            return None
        valid_count = int(getattr(batcher, "valid_count", 0) or 0)
        distinct_valid = self._distinct_valid_prompt_count(batcher)
        if distinct_valid >= B_BATCH:
            return None
        if not self._queue_and_proofs_drained():
            return None

        idle_s = self._seconds_since_last_valid_submission(batcher)
        age_s = self._window_open_age_seconds(batcher)
        if valid_count <= 0:
            if age_s is not None and age_s >= SPARSE_VALID_MAX_WINDOW_SECONDS:
                return "zero_valid_window_timeout"
            return None
        if (
            distinct_valid >= SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS
            and idle_s is not None
            and idle_s >= SPARSE_VALID_IDLE_SEAL_SECONDS
        ):
            return "sparse_valid_idle_timeout"
        if age_s is not None and age_s >= SPARSE_VALID_MAX_WINDOW_SECONDS:
            return "sparse_valid_window_timeout"
        return None

    async def _wait_for_window_seal(self) -> str:
        """Wait for a normal seal, timeout, or bounded-admission dead end.

        The hard proof cap protects validator speed, but it creates a liveness
        edge when all admitted proofs have drained and fewer than B submissions
        survived validation. In that state no future candidate can enter the
        expensive path, so waiting the full window timeout only freezes
        checkpoint progress. We seal partial immediately; training already
        skips partial batches, while archive/EMA can still account for the
        window.
        """
        batcher = self._active_batcher
        if batcher is None:
            return "no_active_batcher"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + WINDOW_TIMEOUT_SECONDS
        duplicate_shortfall_since: float | None = None
        while True:
            if self._proof_admission_exhausted_and_drained(batcher):
                reason = "proof_admission_exhausted_drained"
                logger.warning(
                    "Window %d force-sealing partial: reason=%s "
                    "valid=%d/%d distinct_valid=%d/%d "
                    "proof_admission=%d/%d queue_depth=%d "
                    "inflight_proofs=%d",
                    self._window_n,
                    reason,
                    getattr(batcher, "valid_count", 0),
                    B_BATCH,
                    self._distinct_valid_prompt_count(batcher),
                    B_BATCH,
                    getattr(batcher, "proof_admission_count", 0),
                    MAX_PROOF_CANDIDATES_PER_WINDOW,
                    getattr(self.server, "submit_queue_depth", 0),
                    getattr(self.server, "proof_verification_inflight", 0),
                )
                batcher.force_seal(reason)
                return reason

            if self._duplicate_prompt_shortfall_drained(batcher):
                now = loop.time()
                if duplicate_shortfall_since is None:
                    duplicate_shortfall_since = now
                waited_s = now - duplicate_shortfall_since
                if waited_s >= MAX_SEAL_QUEUE_DRAIN_SECONDS:
                    reason = "duplicate_prompt_distinct_shortfall_drained"
                    logger.warning(
                        "Window %d force-sealing partial: reason=%s "
                        "valid=%d/%d distinct_valid=%d/%d "
                        "proof_admission=%d/%d queue_depth=%d "
                        "inflight_proofs=%d waited_s=%.2f",
                        self._window_n,
                        reason,
                        getattr(batcher, "valid_count", 0),
                        B_BATCH,
                        self._distinct_valid_prompt_count(batcher),
                        B_BATCH,
                        getattr(batcher, "proof_admission_count", 0),
                        MAX_PROOF_CANDIDATES_PER_WINDOW,
                        getattr(self.server, "submit_queue_depth", 0),
                        getattr(self.server, "proof_verification_inflight", 0),
                        waited_s,
                    )
                    batcher.force_seal(reason)
                    return reason
            else:
                duplicate_shortfall_since = None

            sparse_reason = self._sparse_valid_liveness_reason(batcher)
            if sparse_reason is not None:
                logger.warning(
                    "Window %d force-sealing partial: reason=%s "
                    "valid=%d/%d distinct_valid=%d/%d "
                    "proof_admission=%d/%d queue_depth=%d "
                    "inflight_proofs=%d idle_s=%s age_s=%s",
                    self._window_n,
                    sparse_reason,
                    getattr(batcher, "valid_count", 0),
                    B_BATCH,
                    self._distinct_valid_prompt_count(batcher),
                    B_BATCH,
                    getattr(batcher, "proof_admission_count", 0),
                    MAX_PROOF_CANDIDATES_PER_WINDOW,
                    getattr(self.server, "submit_queue_depth", 0),
                    getattr(self.server, "proof_verification_inflight", 0),
                    self._seconds_since_last_valid_submission(batcher),
                    self._window_open_age_seconds(batcher),
                )
                batcher.force_seal(sparse_reason)
                return sparse_reason

            remaining = deadline - loop.time()
            if remaining <= 0:
                return "timeout"

            try:
                await asyncio.wait_for(
                    batcher.seal_event.wait(),
                    timeout=min(PROOF_ADMISSION_STALL_POLL_SECONDS, remaining),
                )
                return "sealed"
            except asyncio.TimeoutError:
                continue

    async def _set_window_randomness(self, subtensor) -> None:
        """Populate all active batchers' per-window randomness seed.

        GRAIL sketch verification re-derives challenge indices from this
        seed; miner and validator must agree. The miner derives it from
        the same block hash + drand round, so the values match bit-for-bit.
        All batchers share the same randomness for a given window.

        Retries on transient substrate failures (finney returning HTTP 503
        or WebSocket handshake errors) before bubbling. Without retries,
        any blip costs us the full window — the new two-phase open keeps
        the failure clean (no zombie accepts) but still leaves the window
        empty. A small in-loop retry recovers transparently from the
        sub-second blips that dominate the failure mode in practice.
        """
        if not self._active_batchers:
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
                for batcher in self._active_batchers.values():
                    batcher.randomness = randomness
                self._last_beacon = beacon
                if beacon is not None and beacon.get("round") is not None:
                    self._active_batcher.window_open_drand_round = int(
                        beacon["round"]
                    )
                # Schedule background bittensor_drand cross-check. Only
                # in real-drand mode (mock path returns beacon=None).
                # Task reference stored so it's not GC'd mid-run.
                # Pass all batchers so the cross-check can invalidate all.
                if beacon is not None and beacon.get("signature"):
                    from reliquary.infrastructure.drand import get_current_chain
                    chain_info = get_current_chain()
                    self._active_batcher._drand_chain_info = chain_info
                    self._verify_task = asyncio.create_task(
                        self._verify_beacon_async(
                            list(self._active_batchers.values()),
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
        batchers,
        chain_hash: str,
        round_number: int,
        randomness: str,
        signature: str | None,
    ) -> None:
        """Background bittensor_drand cross-check for the just-fetched beacon.

        Runs ``verify_beacon_signature`` in a worker thread (it's blocking
        I/O — fetches an independent signature from a second drand relay
        and byte-compares). On any failure (mismatch, network error, library
        crash) flips ``beacon_invalid`` on ALL active batchers so
        ``_train_and_publish`` drops the window before sealing.

        ``batchers`` may be a single batcher or a list of batchers.
        """
        # Normalise to a list so we always iterate.
        batcher_list = batchers if isinstance(batchers, list) else [batchers]
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
            for b in batcher_list:
                b.beacon_invalid = True
            return
        if not ok:
            logger.error(
                "Beacon verification FAILED post-OPEN for round %d; invalidating window %d",
                round_number, self._window_n,
            )
            for b in batcher_list:
                b.beacon_invalid = True

    async def _train_and_publish(self) -> None:
        """TRAINING + PUBLISHING + READY phases."""
        if not self._active_batchers:
            logger.warning("_train_and_publish called with no active batchers")
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
        # Check if any batcher has been invalidated (beacon_invalid propagated
        # to all batchers by _verify_beacon_async, so checking any one suffices).
        if any(b.beacon_invalid for b in self._active_batchers.values()):
            logger.error(
                "Window %d: dropping seal+train+archive — beacon invalid",
                self._window_n,
            )
            self.server.set_active_batchers({})
            self._active_batchers = {}
            self._set_state(WindowState.READY)
            return

        self._set_state(WindowState.TRAINING)
        # v2.3: seal_batch orders by the per-submission drand_round attached
        # by miners (see design A'). The validator does no post-close drand
        # fetch — all timing info is already attached to the submissions.
        # Seal all batchers and collect results.
        per_env_targets = dict(self.env_mix)
        sealed: dict[str, tuple] = {
            name: b.seal_batch()
            for name, b in self._active_batchers.items()
        }
        for name, (batch, rewards) in sealed.items():
            self._active_batchers[name].rewards_by_hotkey = rewards

        # Emit per-submission lifecycle telemetry for every env's accepted
        # pool. Carried over from PR #40 (validator observability) and
        # extended with env_name so downstream consumers can split by env.
        for env_name, batcher in self._active_batchers.items():
            selection_meta = getattr(batcher, "selection_metadata_by_id", {})
            for sub in batcher.valid_submissions():
                meta = selection_meta.get(id(sub), {})
                selected = bool(meta.get("selected_for_batch", False))
                rewarded = bool(meta.get("rewarded", False))
                base_fields = {
                    "window_n": batcher.window_start,
                    "env_name": env_name,
                    "prompt_idx": sub.prompt_idx,
                    "hotkey": sub.hotkey,
                    "arrival_ts": sub.arrival_ts,
                    "decision_ts": sub.decision_ts,
                    "submitted_drand_round": sub.submitted_drand_round or sub.drand_round,
                    "arrival_drand_round": sub.arrival_drand_round,
                    "drand_delta": sub.drand_delta,
                    "seal_trigger_round": getattr(
                        batcher, "_seal_trigger_round", None
                    ),
                    "prompt_hash_lead": sub.prompt_hash_lead,
                    "canonical_rank": meta.get("canonical_rank"),
                    "accepted_into_pool": True,
                    "selected_for_batch": selected,
                    "rewarded": rewarded,
                    "reward_amount": meta.get("reward_amount"),
                    "selection_reason": meta.get("selection_reason"),
                    "batch_filled_reason": (
                        meta.get("selection_reason") if not selected else None
                    ),
                    "reject_stage": "none",
                    "reject_reason": "none",
                }
                log_structured(
                    logger,
                    logging.INFO,
                    "validator_submit_lifecycle",
                    {"stage": "final_batch_selected", **base_fields},
                )
                if rewarded:
                    log_structured(
                        logger,
                        logging.INFO,
                        "validator_submit_lifecycle",
                        {"stage": "reward_assigned", **base_fields},
                    )

        # Merge rewards across envs for unified weight scoring.
        all_rewards: dict[str, float] = {}
        for _batch, rewards in sealed.values():
            for hk, r in rewards.items():
                all_rewards[hk] = all_rewards.get(hk, 0.0) + r

        # Collect batches in env_mix order for grad-accumulation consistency.
        batches = [sealed[name][0] for name, _ in self.env_mix if name in sealed]

        # Only train when EVERY env has a full batch. A partial seal in any
        # env means the gradient signal is incomplete; we skip and archive.
        # Note: miners earn slots regardless of training. Their contribution
        # is reflected in the next ``_submit_weights`` call, which replays
        # the EMA from R2 archives written by ``_archive_window`` below.
        trained = all(
            len(sealed[name][0]) >= per_env_targets[name]
            for name in sealed
        )

        # Anti-exploit training quarantine (ported to multi-env): assess
        # the combined batch across all envs. A poisoned window is still
        # archived and credited; we only skip GRPO + publish. reject_counts
        # are summed over every active batcher.
        combined_reject_counts: dict[str, int] = {}
        for _b in self._active_batchers.values():
            for _k, _v in dict(getattr(_b, "reject_counts", {})).items():
                combined_reject_counts[_k] = combined_reject_counts.get(_k, 0) + _v
        flat_batch = [group for env_batch in batches for group in env_batch]
        quarantine = assess_training_batch(
            flat_batch,
            reject_counts=combined_reject_counts,
        )
        _quarantine_archive = quarantine.to_archive()
        for _b in self._active_batchers.values():
            _b.training_quarantine = _quarantine_archive
        if trained and quarantine.quarantined:
            logger.warning(
                "Window %d quarantined from training: reasons=%s metrics=%s",
                self._window_n, quarantine.reasons, quarantine.metrics,
            )
            trained = False
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
                    self.train_model, batches,
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
            total_subs = sum(len(b) for b in batches)
            total_target = sum(per_env_targets.values())
            logger.info(
                "Window %d sealed with %d/%d submissions — skipping train_step + publish",
                self._window_n, total_subs, total_target,
            )

        self._set_state(WindowState.PUBLISHING)
        if trained:
            self._trained_windows_since_publish += 1
            # checkpoint_n only advances on publish. Publish cadence is based
            # on successful trained windows rather than exact window number so
            # a quarantined boundary window cannot freeze the public checkpoint.
            next_n = self._checkpoint_n + 1
            # Push to HF every N trained windows, or immediately if no
            # checkpoint exists yet.
            should_publish = (
                self._trained_windows_since_publish >= self._publish_every
                or self._checkpoint_store.current_manifest() is None
            )
            if should_publish:
                try:
                    entry = await self._checkpoint_store.publish(
                        checkpoint_n=next_n, model=self.train_model,
                    )
                    self._checkpoint_n = next_n
                    self._trained_windows_since_publish = 0
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
                    "Skipping HF publish for window_n=%d "
                    "(%d/%d trained windows since last publish)",
                    self._window_n,
                    self._trained_windows_since_publish,
                    self._publish_every,
                )

        try:
            await self._archive_window(self._active_batchers, sealed)
        except Exception:
            logger.exception("window archive failed")

        self.server.set_active_batchers({})
        self._active_batchers = {}
        self._set_state(WindowState.READY)

    async def _archive_window(self, batchers, sealed) -> None:
        """Assemble and enqueue the per-window archive payload.

        ``batchers`` is either:
          * a dict {env_name: GrpoWindowBatcher} (multi-env, called from
            _train_and_publish), or
          * a single GrpoWindowBatcher (legacy / test call sites).

        ``sealed`` is either:
          * a dict {env_name: (batch_list, rewards_dict)} matching the
            multi-env form, or
          * a plain list of ValidSubmission (legacy / test call sites).

        Both forms produce a unified archive with ``"environments"`` (list
        of active env names) and per-submission ``"env_name"`` fields.
        Older consumers reading ``"environment"`` (singular) get the first
        env name for backward compat.
        """
        # Normalise inputs to multi-env form.
        if isinstance(batchers, dict):
            # Multi-env path: batchers is {env_name: batcher}
            batcher_dict: dict = batchers
            sealed_dict: dict = sealed  # {env_name: (batch, rewards)}
        else:
            # Legacy single-env path: batchers is one batcher, sealed is a list.
            single_batcher = batchers
            single_batch = sealed
            # Pull env.name off the batcher if it's a real string; fall back
            # to self.env.name otherwise. MagicMock-shaped attrs in tests
            # auto-generate truthy children for any access, so a plain
            # getattr fallback would never fire — explicit isinstance check.
            env_obj = getattr(single_batcher, "env", None)
            candidate = getattr(env_obj, "name", None) if env_obj is not None else None
            env_name_single = candidate if isinstance(candidate, str) else self.env.name
            batcher_dict = {env_name_single: single_batcher}
            sealed_dict = {env_name_single: (single_batch, {})}

        # Use the first batcher for window-level fields (they're shared).
        first_batcher = next(iter(batcher_dict.values()))
        window_opened_at = getattr(first_batcher, "window_opened_at", None)
        from reliquary.shared.modeling import resolve_eos_token_ids

        eos_ids = resolve_eos_token_ids(self.verify_model, self.tokenizer)

        def _resp_time(arrived_at: float) -> float | None:
            if window_opened_at is None or not arrived_at:
                return None
            return arrived_at - window_opened_at

        def _submission_obs_payload(s, batcher, *, rejected: bool = False):
            selection_meta = getattr(batcher, "selection_metadata_by_id", {})
            meta = selection_meta.get(id(s), {})
            return {
                "arrival_ts": getattr(s, "arrival_ts", None),
                "decision_ts": getattr(s, "decision_ts", None),
                "submitted_drand_round": getattr(
                    s, "submitted_drand_round", getattr(s, "drand_round", None)
                ),
                "arrival_drand_round": getattr(s, "arrival_drand_round", None),
                "drand_delta": getattr(s, "drand_delta", None),
                "seal_trigger_round": getattr(
                    s,
                    "seal_trigger_round",
                    getattr(batcher, "_seal_trigger_round", None),
                ),
                "prompt_hash_lead": getattr(s, "prompt_hash_lead", None),
                "canonical_rank": meta.get("canonical_rank"),
                "accepted_into_pool": not rejected,
                "selected_for_batch": bool(meta.get("selected_for_batch", False)),
                "rewarded": bool(meta.get("rewarded", False)),
                "batch_filled_reason": (
                    meta.get("selection_reason")
                    if not meta.get("selected_for_batch", False)
                    else None
                ),
                "reject_stage": getattr(s, "reject_stage", None),
                "reject_reason": getattr(s, "reason", None) if rejected else None,
                "reward_vector": getattr(s, "reward_vector", None),
                "truncated_count": getattr(s, "truncated_count", None),
                "reward_shape": getattr(s, "reward_shape", None),
            }

        def _rollout_payload(s, with_text: bool):
            out = []
            texts = s.completion_texts if with_text else [None] * len(s.rollouts)
            # rollout_hashes is populated at accept-time; for legacy paths
            # (e.g. test fixtures bypassing _accept_locked) it may be empty,
            # in which case we omit the `hash` field rather than guessing.
            hashes = s.rollout_hashes if s.rollout_hashes else [None] * len(s.rollouts)
            for r, text, h in zip(s.rollouts, texts, hashes):
                tokens = list(r.commit["tokens"])
                rollout_dict = (r.commit or {}).get("rollout", {}) or {}
                prompt_length = int(rollout_dict.get("prompt_length", 0))
                completion_length = int(rollout_dict.get(
                    "completion_length", max(0, len(tokens) - prompt_length),
                ))
                eos_terminated = bool(tokens) and int(tokens[-1]) in eos_ids
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

        # Build the combined batch entries and runners_up from all envs.
        batch_entries = []
        runners_up = []
        rejected_entries = []
        combined_rewards: dict[str, float] = {}
        combined_reject_counts: dict[str, int] = {}
        combined_rewarded_not_selected: dict[str, float] = {}

        for env_name, batcher in batcher_dict.items():
            env_obj = self.envs.get(env_name, self.env)
            env_batch, env_rewards = sealed_dict.get(env_name, ([], {}))

            batched_keys = {(s.hotkey, s.prompt_idx) for s in env_batch}

            for s in env_batch:
                problem = env_obj.get_problem(s.prompt_idx)
                batch_entries.append({
                    "hotkey": s.hotkey,
                    "prompt_idx": s.prompt_idx,
                    "env_name": env_name,
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
                    **_submission_obs_payload(s, batcher),
                })

            # All validated submissions that didn't make the final batch —
            # metadata only (no rollouts/text, no prompt).
            for s in batcher.valid_submissions():
                key = (s.hotkey, s.prompt_idx)
                if key in batched_keys:
                    continue
                obs = _submission_obs_payload(s, batcher)
                runner_entry = {
                    "hotkey": s.hotkey,
                    "prompt_idx": s.prompt_idx,
                    "env_name": env_name,
                    "sigma": s.sigma,
                    "response_time": _resp_time(s.arrived_at),
                    "merkle_root": s.merkle_root_bytes.hex(),
                    "sketch_diff_max": s.sketch_diff_max,
                    "lp_dev_max": s.lp_dev_max,
                    "dist_q10_min": s.dist_q10_min,
                    **obs,
                }
                # Rewarded runners-up carry rollout_hashes (ported from main):
                # cooldown/EMA rebuild keys off these to credit miners whose
                # prompt landed in the winning set but weren't selected for training.
                if obs.get("rewarded") and s.rollout_hashes:
                    runner_entry["rollout_hashes"] = [h.hex() for h in s.rollout_hashes]
                runners_up.append(runner_entry)

            for r in getattr(batcher, "rejected_submissions", []):
                rejected_entries.append({
                    "hotkey": r.hotkey,
                    "prompt_idx": r.prompt_idx,
                    "env_name": env_name,
                    "reason": r.reason,
                    "sketch_diff_max": r.sketch_diff_max,
                    "lp_dev_max": r.lp_dev_max,
                    "dist_q10_min": r.dist_q10_min,
                    **_submission_obs_payload(r, batcher, rejected=True),
                })

            for hk, r in env_rewards.items():
                combined_rewards[hk] = combined_rewards.get(hk, 0.0) + r

            for hk, r in dict(
                getattr(batcher, "rewarded_but_not_selected_by_hotkey", {})
            ).items():
                combined_rewarded_not_selected[hk] = (
                    combined_rewarded_not_selected.get(hk, 0.0) + r
                )

            for k, v in dict(getattr(batcher, "reject_counts", {})).items():
                combined_reject_counts[k] = combined_reject_counts.get(k, 0) + v

        env_names_list = list(batcher_dict.keys())
        # Backward-compat: keep "environment" (singular) pointing to the first
        # env so older readers that pre-date multi-env don't silently break.
        archive = {
            "window_start": first_batcher.window_start,
            "validator_hotkey": self.wallet.hotkey.ss58_address,  # provenance
            "randomness": first_batcher.randomness,
            "environment": env_names_list[0],   # legacy singular, kept for compat
            "environments": env_names_list,      # multi-env canonical field
            "force_seal_reason": getattr(first_batcher, "force_seal_reason", None),
            "batch": batch_entries,
            "runners_up": runners_up,
            "reject_summary": combined_reject_counts,
            "grader_failures": dict(getattr(self, "_grader_failures", {})),
            "rejected": rejected_entries,
            "training_quarantine": getattr(
                batcher,
                "training_quarantine",
                {"quarantined": False, "reasons": [], "metrics": {}},
            ),
            # v2.3: per-hotkey emission share from select_batch_and_distribute.
            # All miners whose prompt landed in the winning set appear here,
            # even if their specific submission wasn't picked for training.
            "rewards_by_hotkey": combined_rewards,
            "rewarded_but_not_selected_by_hotkey": combined_rewarded_not_selected,
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
        get_archive_queue().enqueue(first_batcher.window_start, archive)

    def _log_startup_config_banner(self) -> None:
        cp = self._checkpoint_store.current_manifest()
        drand_chain_info = None
        drand_chain_name = os.getenv("DRAND_CHAIN", "quicknet").strip() or "quicknet"
        if self.use_drand:
            try:
                from reliquary.infrastructure.drand import get_current_chain
                drand_chain_info = get_current_chain()
            except Exception:
                drand_chain_info = None
        log_structured(
            logger,
            logging.INFO,
            "validator_startup_config",
            {
                "image_revision": runtime_revision(),
                "use_drand": self.use_drand,
                "drand_chain": drand_chain_name,
                "drand_period": (
                    drand_chain_info.get("period") if drand_chain_info else None
                ),
                "drand_genesis_time": (
                    drand_chain_info.get("genesis_time") if drand_chain_info else None
                ),
                "drand_round_backward_tolerance": DRAND_ROUND_BACKWARD_TOLERANCE,
                "checkpoint_repo_id": cp.repo_id if cp else self.hf_repo_id,
                "checkpoint_revision": cp.revision if cp else None,
                "checkpoint_n": cp.checkpoint_n if cp else self._checkpoint_n,
                "batch_size": B_BATCH,
                "m_rollouts_per_prompt": M_ROLLOUTS,
                "environment": self.env.name,
                "netuid": self.netuid,
                "sigma_min": SIGMA_MIN,
                "bootstrap_sigma_min": BOOTSTRAP_SIGMA_MIN,
                "min_eos_probability": MIN_EOS_PROBABILITY,
                "logprob_is_eps": LOGPROB_IS_EPS,
                "r2_bucket": os.getenv("R2_BUCKET_ID", "reliquary"),
                "http_host": self.server.host,
                "http_port": self.server.port,
                "external_ip_configured": bool(self.external_ip),
                "external_port": self.external_port,
            },
        )

    async def run(self, subtensor) -> None:
        await self.server.start()
        await self._serve_axon_on_chain(subtensor)
        await self._apply_resume_from()                  # ← resume before bootstrap
        await self._bootstrap_state_from_external()
        await self._rebuild_cooldown_from_history()
        await self._rebuild_hashes_from_history()
        self._log_startup_config_banner()

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
            "Validator started (v2.1): envs=%s, netuid=%d, http=%s:%d",
            list(self.envs.keys()), self.netuid, self.server.host, self.server.port,
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
                    seal_reason = await self._wait_for_window_seal()
                    if seal_reason == "sealed":
                        logger.info(
                            "Window %d: all %d batcher(s) sealed",
                            self._window_n, len(self._active_batchers),
                        )
                    elif seal_reason == "timeout":
                        logger.warning(
                            "Window %d timed out at %ds — sealing partial",
                            self._window_n, WINDOW_TIMEOUT_SECONDS,
                        )
                    else:
                        logger.warning(
                            "Window %d sealed by liveness breaker: %s",
                            self._window_n, seal_reason,
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
                    self.server.set_active_batchers({})
                    self._active_batchers = {}
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
        """At startup, reconstruct per-env CooldownMaps from the last
        COOLDOWN_REBUILD_LOOKBACK archived windows on R2.

        R2 is the durable source of truth — each window's sealed batch is
        uploaded by ``_archive_window``. Rebuilding from that history means:
          * local disk state isn't needed (no JSON file to manage)
          * multi-validator consistency: any validator rebuilding from the
            same R2 prefix converges to the same cooldown map
          * a fresh validator joining an active subnet picks up the
            current cooldown state without coordination

        Backward-compat: archives created before multi-env support have no
        per-entry ``env_name``. Those archives carry a top-level ``environment``
        (singular) field that we use to assign all batch entries to that env.
        Newer archives have ``"env_name"`` on each batch entry.
        """
        try:
            current_window = self._window_n
            archives = await storage.list_recent_datasets(
                current_window=current_window + 1,
                n=COOLDOWN_REBUILD_LOOKBACK,
            )
            # Build per-env filtered views of the archives for rebuild.
            for env_name, cooldown_map in self._cooldown_per_env.items():
                env_archives = _filter_archives_for_env(archives, env_name)
                cooldown_map.rebuild_from_history(
                    env_archives, current_window=current_window,
                )
            logger.info(
                "Rebuilt cooldown from %d archive windows (current=%d, "
                "map sizes=%s)",
                len(archives), current_window,
                {n: len(m) for n, m in self._cooldown_per_env.items()},
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
                "Rebuilt hash set from %d/%d archive windows "
                "(current=%d, size=%d)",
                len(archives), HASH_DEDUP_RETENTION_WINDOWS,
                current_window, len(self._hash_set),
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
