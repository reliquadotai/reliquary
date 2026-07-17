"""Validator main loop — v2.1 batch-driven state machine (OPEN→TRAINING→PUBLISHING→READY)."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    COOLDOWN_REBUILD_LOOKBACK,
    COOLDOWN_SNAPSHOT_INTERVAL_WINDOWS,
    TRAINING_RUN_ID,
    B_BATCH,
    BOOTSTRAP_WINDOWS,
    BOOTSTRAP_SIGMA_MIN,
    CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
    CHECKPOINT_STAGING_DIR_DEFAULT,
    DEFAULT_HF_REPO_ID,
    DRAND_ROUND_BACKWARD_TOLERANCE,
    DIFFICULTY_AUCTION_DELTA,
    DIFFICULTY_AUCTION_ENFORCE,
    DIFFICULTY_AUCTION_ENVIRONMENTS,
    DIFFICULTY_AUCTION_SHADOW_ENABLED,
    DIFFICULTY_AUCTION_SHADOW_ENVIRONMENTS,
    DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES,
    DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR,
    ENVIRONMENT_MIX,
    FORCED_SEED_CDF_BOUNDARY_EPSILON,
    FORCED_SEED_CDF_ENFORCE,
    FORCED_SEED_CONSISTENCY_FLOOR,
    FORCED_SEED_ENFORCE,
    FORCED_SEED_PROTOCOL_VERSION,
    FORCED_SEED_ROLLOUT_FLOOR,
    GRAD_CLIP_NORM,
    GRAD_NORM_SKIP_THRESHOLD,
    HASH_DEDUP_RETENTION_WINDOWS,
    KL_BASE_MODEL,
    KL_BETA,
    KL_BETA_EXPLICIT,
    LEARNING_RATE,
    LOGPROB_IS_EPS,
    LEGACY_MERKLE_ROOT_ENFORCE,
    LR_COSINE_MAX_WINDOWS,
    LR_WARMUP_WINDOWS,
    M_ROLLOUTS,
    MAX_AUCTION_SLOTS_PER_OPERATOR,
    MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW,
    MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    MAX_PROOF_WALL_SECONDS,
    MAX_SEAL_QUEUE_DRAIN_SECONDS,
    MIN_EOS_PROBABILITY,
    POLL_INTERVAL_SECONDS,
    PPO_CLIP_EPSILON,
    PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD,
    PROOF_ADMISSION_STALL_POLL_SECONDS,
    REGISTERED_HOTKEY_CACHE_TTL_SECONDS,
    REGISTERED_HOTKEY_REFRESH_MIN_INTERVAL_SECONDS,
    REGISTERED_HOTKEY_REFRESH_TIMEOUT_SECONDS,
    RECOMPUTE_PI_OLD_FROM_VERIFY,
    SHAPE_LEN_FRAC,
    SHAPE_PENALTY,
    SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS,
    SPARSE_VALID_IDLE_SEAL_SECONDS,
    SPARSE_VALID_MAX_WINDOW_SECONDS,
    SIGMA_MIN,
    SUBNET_START_BLOCK,
    TRAIN_UNTIL_CHECKPOINT_N,
    VALIDATOR_HTTP_PORT,
    WANDB_TRAINING_VERSION,
    WINDOW_LENGTH,
    WINDOW_COLLECTION_SECONDS,
    WINDOW_TIMEOUT_SECONDS,
)
from reliquary.environment import load_environments
from reliquary.environment.base import Environment
from reliquary.infrastructure import chain, storage
from reliquary.protocol.submission import RejectReason, RolloutSubmission, WindowState
from reliquary.validator import telemetry
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.checkpoint import CheckpointStore
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.dedup import RolloutHashSet
from reliquary.validator.observability import log_structured, runtime_revision
from reliquary.validator.quarantine import assess_training_batch
from reliquary.validator.server import ValidatorServer
from reliquary.validator.training import TrainingStepSkipped, train_step
from reliquary.validator.training_accumulator import BalancedTrainingAccumulator

logger = logging.getLogger(__name__)

_HF_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _cooldown_snapshot_key(run_id: str) -> str:
    """R2 key for the run-keyed cooldown snapshot."""
    return f"cooldown_snapshots/{run_id}.json"


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
    operator_by_hotkey: dict[str, str] | None = None,
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
        operator_by_hotkey=operator_by_hotkey,
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


def _parse_pinned_kl_reference(spec: str) -> tuple[str, str]:
    """Parse ``repo@revision`` and require an immutable full HF commit SHA."""
    repo_id, separator, revision = spec.rpartition("@")
    if not separator or not repo_id or not revision:
        raise ValueError(
            "RELIQUARY_KL_BASE_MODEL must be repo@<full 40-character commit SHA>"
        )
    if _HF_COMMIT_RE.fullmatch(revision) is None:
        raise ValueError(
            "RELIQUARY_KL_BASE_MODEL revision must be a full 40-character "
            "Hugging Face commit SHA"
        )
    return repo_id, revision.lower()


def _model_storage_bytes(model: Any) -> int | None:
    """Best-effort parameter+buffer storage size for capacity telemetry."""
    try:
        tensors = list(model.parameters()) + list(model.buffers())
        return sum(t.numel() * t.element_size() for t in tensors)
    except (AttributeError, TypeError):
        return None


def _model_parameter_count(model: Any) -> int | None:
    try:
        return sum(parameter.numel() for parameter in model.parameters())
    except (AttributeError, TypeError):
        return None


def _model_device(model: Any) -> str | None:
    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return None


def _model_dtype(model: Any) -> str | None:
    try:
        return str(next(model.parameters()).dtype)
    except (AttributeError, StopIteration, TypeError):
        return None


def _model_config_value(model: Any, name: str) -> Any | None:
    value = getattr(getattr(model, "config", None), name, None)
    return value if isinstance(value, (str, int)) else None


def _validate_fixed_kl_reference(train_model: Any, ref_model: Any) -> None:
    """Fail at startup when a fixed reference cannot share the train inputs."""
    if ref_model is train_model:
        raise ValueError("fixed KL reference must be a distinct model instance")
    checks = {
        "device": (_model_device(train_model), _model_device(ref_model)),
        "dtype": (_model_dtype(train_model), _model_dtype(ref_model)),
        "parameter_count": (
            _model_parameter_count(train_model),
            _model_parameter_count(ref_model),
        ),
        "model_type": (
            _model_config_value(train_model, "model_type"),
            _model_config_value(ref_model, "model_type"),
        ),
        "vocab_size": (
            _model_config_value(train_model, "vocab_size"),
            _model_config_value(ref_model, "vocab_size"),
        ),
    }
    for label, (train_value, ref_value) in checks.items():
        if (
            train_value is not None
            and ref_value is not None
            and train_value != ref_value
        ):
            raise ValueError(
                f"fixed KL reference {label} mismatch: "
                f"train={train_value!r} reference={ref_value!r}"
            )


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
                "kl_base_model": KL_BASE_MODEL,
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
        # verify_model: frozen snapshot of the last published checkpoint. It
        # verifies commitment proofs and can independently supply PPO's old
        # policy. In rolling mode it is also the KL reference; fixed mode uses
        # a separately pinned base model. Refreshed only after publication.
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

        self.server = ValidatorServer(host=http_host, port=http_port)
        self.server.set_late_drop_callback(self.record_late_drop)
        self.server.configure_prompt_source_health(
            self._prompt_source_health_snapshot
        )

        # v2.1 state machine infrastructure — in-memory only, bootstrapped at
        # startup from R2 + HF (no local JSON state file).
        self._window_n: int = 0
        self._candidate_window_n: int | None = None
        self._window_preparation_stage: str | None = None
        self._checkpoint_n: int = 0
        self._publish_every = CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
        self._trained_windows_since_publish = 0
        self.server.set_training_publish_state({
            "trained_windows_since_publish": 0,
            "publish_interval": self._publish_every,
            "publication_pending": False,
        })
        self._training_accumulator = BalancedTrainingAccumulator(
            dict(self.env_mix)
        )
        self.server.set_training_accumulator_state(
            self._training_accumulator.snapshot()
        )
        self._windows_since_cooldown_snapshot = 0
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
        # Serializes proactive refreshes with the on-demand refresh invoked by
        # the HTTP registration gate after a cache miss.
        self._registration_refresh_lock = asyncio.Lock()
        self._registration_refresh_task: asyncio.Task | None = None
        self._current_window_state: WindowState = WindowState.READY

        self._resume_from = resume_from
        self._load_model_fn = load_model_fn or _default_load_model

        # Fixed mode is opt-in. An explicit fixed reference is a load-bearing
        # training control, so it is immutable and fail-closed. Empty config keeps
        # the legacy rolling reference (verify_model) exactly as before.
        self.base_ref_model = None
        self.kl_reference_state: dict[str, Any] = {
            "schema_version": 1,
            "mode": "rolling",
            "beta": KL_BETA,
            "requested_model": None,
            "repo_id": None,
            "requested_revision": None,
            "resolved_revision": None,
            "loaded": self.verify_model is not None,
            "device": _model_device(self.verify_model),
            "dtype": _model_dtype(self.verify_model),
            "parameter_count": _model_parameter_count(self.verify_model),
            "storage_bytes": _model_storage_bytes(self.verify_model),
            "beta_explicit": KL_BETA_EXPLICIT,
            "behavior_logprobs": (
                "verify_model"
                if RECOMPUTE_PI_OLD_FROM_VERIFY
                else "miner_claim"
            ),
            "learning_rate": LEARNING_RATE,
            "grad_norm_skip_threshold": GRAD_NORM_SKIP_THRESHOLD,
            "ppo_ratio_outside_clip_skip_threshold": (
                PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD
            ),
            "shape_penalty": SHAPE_PENALTY,
            "shape_len_frac": SHAPE_LEN_FRAC,
            "train_until_checkpoint_n": TRAIN_UNTIL_CHECKPOINT_N,
        }
        if KL_BASE_MODEL:
            if model is None:
                raise RuntimeError(
                    "fixed KL reference requested but no train model was loaded"
                )
            repo, rev = _parse_pinned_kl_reference(KL_BASE_MODEL)
            if not KL_BETA_EXPLICIT:
                raise ValueError(
                    "fixed KL reference requires an explicit RELIQUARY_KL_BETA; "
                    "do not inherit the rolling-reference default"
                )
            if not RECOMPUTE_PI_OLD_FROM_VERIFY:
                raise ValueError(
                    "fixed KL reference requires "
                    "RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true; the fixed "
                    "anchor and PPO behavior policy are separate contracts"
                )
            try:
                from huggingface_hub import snapshot_download
                from reliquary.shared.modeling import (
                    MODEL_SNAPSHOT_ALLOW_PATTERNS,
                )

                base_path = snapshot_download(
                    repo_id=repo,
                    revision=rev,
                    allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
                )
                path_revision = Path(base_path).resolve().name.lower()
                if (
                    _HF_COMMIT_RE.fullmatch(path_revision) is not None
                    and path_revision != rev
                ):
                    raise RuntimeError(
                        "fixed KL snapshot resolved to an unexpected revision: "
                        f"requested={rev} resolved={path_revision}"
                    )
                self.base_ref_model = self._load_model_fn(base_path)
                _validate_fixed_kl_reference(
                    self.train_model, self.base_ref_model
                )
                self.base_ref_model.eval()
                for _p in self.base_ref_model.parameters():
                    _p.requires_grad = False
            except Exception as exc:
                logger.exception(
                    "failed to load required fixed KL reference %s",
                    KL_BASE_MODEL,
                )
                raise RuntimeError(
                    f"failed to load required fixed KL reference {KL_BASE_MODEL}"
                ) from exc

            resolved_revision = path_revision
            if _HF_COMMIT_RE.fullmatch(resolved_revision) is None:
                # Some injected/custom downloaders return a non-cache path. The
                # requested revision is already a full immutable SHA, so retain it
                # rather than inventing a mutable identity from the path.
                resolved_revision = rev
            self.kl_reference_state = {
                "schema_version": 1,
                "mode": "fixed",
                "beta": KL_BETA,
                "requested_model": KL_BASE_MODEL,
                "repo_id": repo,
                "requested_revision": rev,
                "resolved_revision": resolved_revision,
                "loaded": True,
                "device": _model_device(self.base_ref_model),
                "dtype": _model_dtype(self.base_ref_model),
                "parameter_count": _model_parameter_count(
                    self.base_ref_model
                ),
                "storage_bytes": _model_storage_bytes(self.base_ref_model),
                "beta_explicit": KL_BETA_EXPLICIT,
                "behavior_logprobs": (
                    "verify_model"
                    if RECOMPUTE_PI_OLD_FROM_VERIFY
                    else "miner_claim"
                ),
                "learning_rate": LEARNING_RATE,
                "grad_norm_skip_threshold": GRAD_NORM_SKIP_THRESHOLD,
                "ppo_ratio_outside_clip_skip_threshold": (
                    PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD
                ),
                "shape_penalty": SHAPE_PENALTY,
                "shape_len_frac": SHAPE_LEN_FRAC,
                "train_until_checkpoint_n": TRAIN_UNTIL_CHECKPOINT_N,
            }
            logger.info(
                "GRPO KL reference=fixed repo=%s revision=%s beta=%.6g "
                "device=%s storage_bytes=%s",
                repo,
                resolved_revision,
                KL_BETA,
                self.kl_reference_state["device"],
                self.kl_reference_state["storage_bytes"],
            )

        self.server.set_training_kl_reference_state(self.kl_reference_state)
        telemetry.update_config({
            "kl_reference_mode": self.kl_reference_state["mode"],
            "kl_reference_repo_id": self.kl_reference_state["repo_id"],
            "kl_reference_revision": self.kl_reference_state[
                "resolved_revision"
            ],
            "kl_reference_storage_bytes": self.kl_reference_state[
                "storage_bytes"
            ],
            "pi_old_source": self.kl_reference_state["behavior_logprobs"],
            "learning_rate": LEARNING_RATE,
            "grad_norm_skip_threshold": GRAD_NORM_SKIP_THRESHOLD,
            "ppo_ratio_outside_clip_skip_threshold": (
                PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD
            ),
            "shape_penalty": SHAPE_PENALTY,
            "shape_len_frac": SHAPE_LEN_FRAC,
            "train_until_checkpoint_n": TRAIN_UNTIL_CHECKPOINT_N,
        })

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

    def _prompt_source_health_snapshot(self) -> dict[str, dict[str, Any]]:
        snapshots: dict[str, dict[str, Any]] = {}
        for env_name, env in self.envs.items():
            snapshot_fn = getattr(env, "source_health", None)
            if not callable(snapshot_fn):
                snapshots[env_name] = {"status": "unreported"}
                continue
            try:
                snapshots[env_name] = dict(snapshot_fn())
            except Exception as exc:
                snapshots[env_name] = {
                    "status": "degraded",
                    "last_error_type": type(exc).__name__,
                }
        return snapshots

    def _publish_window_preparation_state(self) -> None:
        self.server.set_window_preparation_state(
            last_committed_window_n=self._window_n,
            candidate_window_n=getattr(self, "_candidate_window_n", None),
            stage=getattr(self, "_window_preparation_stage", None),
        )

    def _set_window_preparation_stage(self, stage: str) -> None:
        self._window_preparation_stage = stage
        self._publish_window_preparation_state()

    def _rollback_preopen_window(self, exc: BaseException) -> None:
        """Keep a failed candidate reusable instead of consuming its ID."""
        if self._candidate_window_n is None:
            return
        failure = {
            "candidate_window_n": self._candidate_window_n,
            "stage": self._window_preparation_stage or "unknown",
            "error_type": type(exc).__name__,
            "ts": time.time(),
        }
        self.server.record_window_preparation_failure(failure)
        self._window_preparation_stage = None
        self._active_batchers = {}
        self.server.set_active_batchers({})
        self._publish_window_preparation_state()

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
        if self._candidate_window_n is None:
            self._candidate_window_n = self._window_n + 1
        target_window = self._candidate_window_n
        self._set_window_preparation_stage("batcher_construction")
        bootstrap = is_bootstrap_window(
            window_start=target_window,
            subnet_start=SUBNET_START_BLOCK,
        )
        cp = self._checkpoint_store.current_manifest()
        cp_hash = cp.revision if cp else ""
        operator_by_hotkey = self.server.operator_by_hotkey_snapshot()
        self._active_batchers = {}
        for env_name, env in self.envs.items():
            batcher = open_grpo_window(
                window_start=target_window,
                env=env, model=self.verify_model,
                cooldown_map=self._cooldown_per_env[env_name],
                hash_set=self._hash_set,
                tokenizer=self.tokenizer,
                bootstrap=bootstrap,
                # Seal extension waits until the submit queue AND in-flight
                # GRAIL proofs have both drained — concurrent verification
                # empties the queue while proofs are still running.
                queue_drained_predicate=self._queue_and_proofs_drained,
                operator_by_hotkey=operator_by_hotkey,
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
        target_windows = {
            int(batcher.window_start) for batcher in self._active_batchers.values()
        }
        if target_windows != {self._candidate_window_n}:
            raise RuntimeError(
                "prepared batchers do not share the candidate window"
            )
        self._set_window_preparation_stage("activation")
        # Bind the main loop into each batcher BEFORE exposing them to the
        # server, so the delayed drand-boundary seal scheduled from the
        # worker thread targets this loop. No running loop (sync tests) →
        # leave _loop None and fall back to the immediate-seal path.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        for batcher in self._active_batchers.values():
            batcher.mark_window_opened()
            if loop is not None:
                batcher.bind_event_loop(loop)
        self.server.set_active_batchers(self._active_batchers)
        self._window_n = int(self._candidate_window_n)
        self._candidate_window_n = None
        self._window_preparation_stage = None
        self.server.clear_window_preparation_failure()
        self._publish_window_preparation_state()
        self._set_state(WindowState.OPEN)

    def _registration_refresh_poll_seconds(self) -> float:
        """Poll often enough to refresh the registration snapshot before TTL.

        The public health endpoint intentionally degrades when the snapshot is
        stale. Auction windows can exceed the five-minute cache TTL, so relying
        only on the next window boundary leaves healthy validators degraded and
        makes the first late submission pay the chain-refresh latency.
        """
        return min(
            30.0,
            max(
                float(REGISTERED_HOTKEY_REFRESH_MIN_INTERVAL_SECONDS),
                float(REGISTERED_HOTKEY_CACHE_TTL_SECONDS) / 4.0,
            ),
        )

    async def _maintain_registration_cache(self) -> None:
        """Refresh the registered-hotkey snapshot before it becomes stale.

        A chain RPC can legitimately consume most of the 20-second refresh
        timeout. Starting at half the TTL leaves several bounded attempts
        before admission health degrades, while a failed attempt retries at
        the existing minimum refresh interval instead of waiting a full poll.
        """
        poll_seconds = self._registration_refresh_poll_seconds()
        retry_seconds = min(
            poll_seconds,
            float(REGISTERED_HOTKEY_REFRESH_MIN_INTERVAL_SECONDS),
        )
        refresh_threshold = max(
            0.0,
            float(REGISTERED_HOTKEY_CACHE_TTL_SECONDS) / 2.0,
        )
        while True:
            age = self.server.registration_cache_age()
            if age is None or age >= refresh_threshold:
                refreshed = await self._refresh_registered_hotkeys(
                    max_cache_age_seconds=refresh_threshold,
                    reason="proactive",
                )
                if not refreshed:
                    logger.warning(
                        "Proactive registered-hotkey cache refresh failed; "
                        "age=%.1fs",
                        age if age is not None else -1.0,
                    )
                    await asyncio.sleep(retry_seconds)
                    continue
                await asyncio.sleep(poll_seconds)
                continue

            # Startup performs the initial chain read before model and prompt
            # bootstrap. The worker may therefore begin with a nonzero cache
            # age; sleep to the actual deadline so that bootstrap time cannot
            # consume the retry runway in whole poll-sized chunks.
            await asyncio.sleep(
                min(poll_seconds, refresh_threshold - age)
            )

    async def _refresh_registered_hotkeys(
        self,
        *,
        force: bool = False,
        max_cache_age_seconds: float | None = None,
        reason: str = "unspecified",
    ) -> bool:
        """Refresh registered subnet identities without concurrent chain reads."""
        async with self._registration_refresh_lock:
            return await self._refresh_registered_hotkeys_locked(
                force=force,
                max_cache_age_seconds=max_cache_age_seconds,
                reason=reason,
            )

    async def _refresh_registered_hotkeys_locked(
        self,
        *,
        force: bool = False,
        max_cache_age_seconds: float | None = None,
        reason: str = "unspecified",
    ) -> bool:
        """Refresh registered subnet identities from a fresh chain session."""
        age = self.server.registration_cache_age()
        cache_age_limit = (
            float(REGISTERED_HOTKEY_CACHE_TTL_SECONDS)
            if max_cache_age_seconds is None
            else max(0.0, float(max_cache_age_seconds))
        )
        if (
            not force
            and age is not None
            and age < cache_age_limit
        ):
            return True

        subtensor = None

        async def _load() -> tuple[set[str], dict[str, str]]:
            nonlocal subtensor
            subtensor = await chain.get_subtensor()
            neurons = await chain.get_neurons_lite(subtensor, self.netuid)
            hotkeys: set[str] = set()
            operators: dict[str, str] = {}
            ambiguous_hotkeys: set[str] = set()
            for neuron in neurons:
                raw_hotkey = getattr(neuron, "hotkey", None)
                if not isinstance(raw_hotkey, str) or not (
                    hotkey := raw_hotkey.strip()
                ):
                    continue
                hotkeys.add(hotkey)
                raw_operator = getattr(neuron, "coldkey", None)
                if not isinstance(raw_operator, str):
                    continue
                operator = raw_operator.strip()
                if not operator or hotkey in ambiguous_hotkeys:
                    continue
                previous = operators.get(hotkey)
                if previous is not None and previous != operator:
                    operators.pop(hotkey, None)
                    ambiguous_hotkeys.add(hotkey)
                    continue
                operators[hotkey] = operator
            return hotkeys, operators

        try:
            hotkeys, operator_by_hotkey = await asyncio.wait_for(
                _load(),
                timeout=REGISTERED_HOTKEY_REFRESH_TIMEOUT_SECONDS,
            )
            if not hotkeys:
                raise RuntimeError(
                    "lite neuron refresh returned no registered hotkeys"
                )
            self.server.set_registered_hotkeys(
                hotkeys,
                operator_by_hotkey=operator_by_hotkey,
            )
            self.server.record_registration_cache_refresh(
                success=True,
                reason=reason,
            )
            logger.info(
                "Registered-hotkey cache refreshed: netuid=%d hotkeys=%d "
                "operator_mappings=%d complete=%s",
                self.netuid,
                len(hotkeys),
                len(operator_by_hotkey),
                len(operator_by_hotkey) == len(hotkeys),
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.server.record_registration_cache_refresh(
                success=False,
                reason=reason,
                failure_type=type(exc).__name__,
            )
            logger.exception(
                "Registered-hotkey cache refresh failed for netuid=%d",
                self.netuid,
            )
            return False
        finally:
            await chain.close_subtensor(subtensor)

    def _proof_admission_exhausted_and_drained(self, batcher) -> bool:
        """True when bounded proof admission cannot fill this window anymore.

        Gated on the grading-attempts ceiling, not the GRAIL candidate budget:
        out_of_zone rejects refund the latter, so for a degenerate-reward env
        it never reaches its cap — the real "can't fill anymore" signal is the
        never-refunded grading ceiling.
        """
        if batcher is None or batcher.is_sealed():
            return False
        distinct_valid = self._distinct_valid_prompt_count(batcher)
        if distinct_valid >= B_BATCH:
            return False
        if (
            getattr(batcher, "proof_grading_attempts", 0)
            < MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
        ):
            return False
        queue_depth = int(getattr(self.server, "submit_queue_depth", 0) or 0)
        inflight = int(getattr(self.server, "proof_verification_inflight", 0) or 0)
        return queue_depth == 0 and inflight == 0

    def _distinct_valid_prompt_count(self, batcher) -> int:
        """Best-effort distinct trainable prompt count for liveness decisions.

        Auction environments use the graded pending pool; legacy environments
        use the proven valid pool.
        """
        counter_name = (
            "distinct_pending_prompt_count"
            if getattr(batcher, "difficulty_auction_enabled", False)
            else "distinct_valid_prompt_count"
        )
        counter = getattr(batcher, counter_name, None)
        if callable(counter):
            return int(counter())
        count_name = (
            "pending_count"
            if getattr(batcher, "difficulty_auction_enabled", False)
            else "valid_count"
        )
        return int(getattr(batcher, count_name, 0) or 0)

    @staticmethod
    def _admitted_count(batcher) -> int:
        count_name = (
            "pending_count"
            if getattr(batcher, "difficulty_auction_enabled", False)
            else "valid_count"
        )
        return int(getattr(batcher, count_name, 0) or 0)

    def _duplicate_prompt_shortfall_drained(self, batcher) -> bool:
        """True when duplicates filled raw submissions but not trainable slots."""
        if batcher is None or batcher.is_sealed():
            return False
        if getattr(batcher, "_seal_trigger_round", None) is not None:
            return False
        valid_count = self._admitted_count(batcher)
        distinct_valid = self._distinct_valid_prompt_count(batcher)
        if valid_count < B_BATCH or distinct_valid >= B_BATCH:
            return False
        queue_depth = int(getattr(self.server, "submit_queue_depth", 0) or 0)
        inflight = int(getattr(self.server, "proof_verification_inflight", 0) or 0)
        return queue_depth == 0 and inflight == 0

    def _queue_and_proofs_drained(self) -> bool:
        queue_depth = int(getattr(self.server, "submit_queue_depth", 0) or 0)
        inflight = int(getattr(self.server, "proof_verification_inflight", 0) or 0)
        if queue_depth != 0 or inflight != 0:
            return False
        # Close the dequeue race where the worker has removed an item from the
        # asyncio queue but has not yet incremented ``_inflight_proofs``. The
        # batcher reservation spans that gap and is the authoritative signal.
        for batcher in self._active_batchers.values():
            if int(getattr(batcher, "pending_proof_reservations", 0) or 0):
                return False
            if int(getattr(batcher, "inflight_proof_reservations", 0) or 0):
                return False
        return True

    async def _freeze_auction_populations(self, batchers: list[Any]) -> bool:
        """Drain pre-deadline work, then freeze each auction pending pool.

        Returns ``True`` when the normal drain budget was exhausted. Pending
        queue entries are then rejected by ``start_proof_admission``; already
        started grading is allowed one final bounded interval to finish so seal
        can never race a mutation of the ranked population.
        """
        auction_batchers = [
            batcher for batcher in batchers
            if getattr(batcher, "difficulty_auction_enabled", False)
        ]
        if not auction_batchers:
            return False

        loop = asyncio.get_running_loop()
        drain_deadline = loop.time() + MAX_SEAL_QUEUE_DRAIN_SECONDS
        while not self._queue_and_proofs_drained():
            if loop.time() >= drain_deadline:
                break
            await asyncio.sleep(PROOF_ADMISSION_STALL_POLL_SECONDS)

        timed_out = not self._queue_and_proofs_drained()
        for batcher in auction_batchers:
            begin_snapshot = getattr(batcher, "begin_seal_snapshot", None)
            if callable(begin_snapshot):
                begin_snapshot()

        # Once frozen, pending items are cheap drops. A request that already
        # started may still hold the batcher lock while grading; do not race it.
        quiesce_deadline = loop.time() + MAX_SEAL_QUEUE_DRAIN_SECONDS
        while any(
            int(getattr(batcher, "inflight_proof_reservations", 0) or 0)
            for batcher in auction_batchers
        ):
            if loop.time() >= quiesce_deadline:
                raise RuntimeError(
                    "auction admission failed to quiesce before seal"
                )
            await asyncio.sleep(PROOF_ADMISSION_STALL_POLL_SECONDS)

        if timed_out:
            logger.warning(
                "Window %d auction queue drain reached %.1fs; froze pending "
                "populations and dropped remaining queued submissions",
                self._window_n,
                MAX_SEAL_QUEUE_DRAIN_SECONDS,
            )
        return timed_out

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
        valid_count = self._admitted_count(batcher)
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

    def _force_seal_dead_batcher(self, batcher, dup_since: dict) -> str | None:
        """Force-seal one batcher if its own liveness breaker fires; else None.

        Per-env so a fast env never seals a slower one short.
        """
        env = getattr(getattr(batcher, "env", None), "name", "?")
        if getattr(batcher, "difficulty_auction_enabled", False):
            # The full population is the auction's input. Duplicate/sparse idle
            # breakers would let an early burst truncate the fixed collection
            # period and recreate the speed race. Only an exhausted, fully
            # drained grading budget is terminal before the deadline.
            if not self._proof_admission_exhausted_and_drained(batcher):
                return None
            reason = "proof_admission_exhausted_drained"
            logger.warning(
                "Window %d env=%s force-sealing auction: reason=%s "
                "admitted=%d/%d distinct=%d/%d",
                self._window_n,
                env,
                reason,
                self._admitted_count(batcher),
                B_BATCH,
                self._distinct_valid_prompt_count(batcher),
                B_BATCH,
            )
            batcher.force_seal(reason)
            return reason
        if self._proof_admission_exhausted_and_drained(batcher):
            reason = "proof_admission_exhausted_drained"
        elif self._duplicate_prompt_shortfall_drained(batcher):
            now = asyncio.get_running_loop().time()
            if now - dup_since.setdefault(env, now) < MAX_SEAL_QUEUE_DRAIN_SECONDS:
                return None
            reason = "duplicate_prompt_distinct_shortfall_drained"
        else:
            dup_since.pop(env, None)
            reason = self._sparse_valid_liveness_reason(batcher)
        if reason is None:
            return None
        logger.warning(
            "Window %d env=%s force-sealing partial: reason=%s valid=%d/%d "
            "distinct=%d/%d idle_s=%s age_s=%s",
            self._window_n, env, reason,
            getattr(batcher, "valid_count", 0), B_BATCH,
            self._distinct_valid_prompt_count(batcher), B_BATCH,
            self._seconds_since_last_valid_submission(batcher),
            self._window_open_age_seconds(batcher),
        )
        batcher.force_seal(reason)
        return reason

    async def _wait_for_window_seal(self) -> str:
        """Wait until every active env's batcher seals.

        Auction batchers seal on their fixed collection deadline; legacy
        batchers retain their B-distinct/drand-boundary seal. Per-environment
        liveness guards cannot let a fast environment cut a slower one short.
        The window advances only once all are sealed (or the global timeout).
        """
        batchers = list(self._active_batchers.values())
        if not batchers:
            return "no_active_batcher"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + WINDOW_TIMEOUT_SECONDS
        dup_since: dict[str, float] = {}
        reasons: dict[str, str] = {}
        while True:
            for b in batchers:
                # Normal path: seal on the fixed collection deadline.
                poll = getattr(b, "poll_deadline", None)
                if callable(poll):
                    poll()
                if b.is_sealed():
                    continue
                r = self._force_seal_dead_batcher(b, dup_since)
                if r is not None:
                    reasons[getattr(getattr(b, "env", None), "name", "?")] = r

            if all(b.is_sealed() for b in batchers):
                break

            remaining = deadline - loop.time()
            if remaining <= 0:
                for b in batchers:
                    if not b.is_sealed():
                        b.force_seal("timeout")
                return "timeout"

            await asyncio.sleep(min(PROOF_ADMISSION_STALL_POLL_SECONDS, remaining))

        if await self._freeze_auction_populations(batchers):
            reasons["auction"] = "auction_queue_drain_timeout"

        if not reasons:
            return "sealed"
        if len(reasons) == 1:
            return next(iter(reasons.values()))
        return ",".join(f"{e}={r}" for e, r in reasons.items())

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
        first_batcher = next(iter(self._active_batchers.values()))
        target_window = getattr(first_batcher, "window_start", None)
        if not isinstance(target_window, int) or isinstance(target_window, bool):
            candidate_window = getattr(self, "_candidate_window_n", None)
            target_window = (
                candidate_window
                if candidate_window is not None
                else self._window_n
            )
        self._set_window_preparation_stage("randomness")
        # 3 attempts total: original + 2 retries. Backoff is 0.5s then 1.0s,
        # so worst-case added latency is 1.5s — well inside the 60s window
        # budget. Sustained outages still bubble after attempt 3.
        last_exc: Exception | None = None
        randomness: str | None = None
        beacon: dict | None = None
        for attempt in range(3):
            try:
                randomness, beacon = await self._derive_randomness(
                    subtensor, target_window,
                )
                if attempt > 0:
                    logger.info(
                        "Window %d: randomness derived on attempt %d",
                        target_window, attempt + 1,
                    )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    logger.warning(
                        "Window %d: _derive_randomness attempt %d failed (%s: %s); retrying",
                        target_window, attempt + 1,
                        type(exc).__name__, str(exc)[:120],
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
        if randomness is None:
            assert last_exc is not None
            raise last_exc

        for batcher in self._active_batchers.values():
            batcher.randomness = randomness

        self._set_window_preparation_stage("prompt_manifest")
        try:
            for batcher in self._active_batchers.values():
                batcher.set_prompt_range()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Window %d: prompt preparation failed (%s)",
                target_window,
                type(exc).__name__,
            )
            raise

        self._last_beacon = beacon
        if beacon is not None and beacon.get("round") is not None:
            self._active_batcher.window_open_drand_round = int(beacon["round"])
        # Schedule background bittensor_drand cross-check. Only in real-drand
        # mode (mock path returns beacon=None). Pass all batchers so the
        # cross-check can invalidate the whole multi-environment window.
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

    def _record_auction_final_verdicts(self, batcher: GrpoWindowBatcher) -> None:
        """Publish the final lifecycle state of every auction candidate.

        Admission and final selection are deliberately separate in auction mode:
        the worker first records that a cheap-validated candidate entered the
        pending pool, then ``seal_batch`` proves only the ranked candidates that
        can still win. The second record added here is identifiable by the
        non-null ``selected_for_batch`` and ``rewarded`` fields.

        Non-winners remain accepted candidates with no reward. A candidate that
        was sampled or selected for deferred proof and failed receives the real
        proof rejection. Telemetry publication is best-effort and never changes
        protocol state or rewards.
        """
        if not getattr(batcher, "difficulty_auction_enabled", False):
            return
        if getattr(batcher, "_auction_final_verdicts_published", False):
            return

        metadata = getattr(batcher, "difficulty_auction_metadata_by_id", {})
        for pending in batcher.pending_submissions():
            row = metadata.get(id(pending), {}) if isinstance(metadata, dict) else {}
            selected = bool(row.get("selected", False))
            proof_reject = pending.reject_response
            accepted = proof_reject is None
            reason = (
                RejectReason.ACCEPTED
                if proof_reject is None
                else proof_reject.reason
            )
            canonical_rank = row.get("rank")
            if not isinstance(canonical_rank, int) or isinstance(
                canonical_rank, bool
            ):
                canonical_rank = None

            try:
                self.server.record_verdict(
                    pending.hotkey,
                    pending.request.merkle_root,
                    accepted,
                    reason,
                    window_n=batcher.window_start,
                    telemetry=pending.telemetry,
                    reject_stage=None if accepted else "auction_seal",
                    canonical_rank=canonical_rank,
                    accepted_into_pool=True,
                    selected_for_batch=selected,
                    rewarded=selected,
                )
                log_structured(
                    logger,
                    logging.INFO if accepted else logging.WARNING,
                    "validator_submit_lifecycle",
                    {
                        "stage": "auction_finalized",
                        "window_n": batcher.window_start,
                        "env_name": str(getattr(batcher.env, "name", "")),
                        "prompt_idx": pending.prompt_idx,
                        "hotkey": pending.hotkey,
                        "accepted": accepted,
                        "reason": reason.value,
                        "canonical_rank": canonical_rank,
                        "accepted_into_pool": True,
                        "selected_for_batch": selected,
                        "rewarded": selected,
                        "auction_status": row.get("status"),
                    },
                )
            except Exception:
                logger.exception(
                    "auction final verdict publication failed window=%d prompt=%d",
                    batcher.window_start,
                    pending.prompt_idx,
                )

        batcher._auction_final_verdicts_published = True

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
        # Seal every environment after its collection deadline. Auction mode
        # ranks the frozen pending population, proves candidates top-down, and
        # selects at most B_BATCH winners independently for Math and Code.
        per_env_targets = dict(self.env_mix)
        # Split the window's emission budget (1.0) equally across the active
        # envs so the merged ``combined_rewards`` stays <= 1.0 and the weight
        # setter's ``burn = max(0, 1 - total)`` keeps working. Without this each
        # env distributed a full pool of 1.0, so two envs summed to ~2.0 and the
        # burn was permanently zeroed (it inherits the per-window total via the
        # EMA). Single-validator assumption: divide by the count THIS validator
        # runs. If multiple validators ever run different env subsets, switch
        # the denominator to ``len(ENVIRONMENT_MIX)`` (the canonical protocol
        # count, as GRAD_ACCUM_STEPS already does) so every validator uses the
        # same pool and an env a validator does not run burns its share.
        pool_per_env = 1.0 / len(self.env_mix)
        # Fetch a fresh drand beacon now — AFTER the collection deadline — to key
        # each batcher's forensic sample and equal-score operator tie-break. Its
        # randomness did not exist when miners submitted, so neither surface can be
        # ground in advance. If the fetch fails, the window randomness is the
        # deterministic ranking fallback and the forensic sample is disabled.
        seal_randomness = await self._fetch_seal_randomness()
        for b in self._active_batchers.values():
            b.seal_randomness = seal_randomness
        # seal_batch now runs the GRAIL GPU proofs (``_prove_ranked``, up to ~8
        # forward passes at 5-25s each), so offload each batcher to a thread to
        # keep the event loop responsive (/state, /health, submit, archive).
        # Awaited sequentially in dict order so the deterministic fold into
        # combined rewards / window_batches / archives is byte-for-byte unchanged.
        sealed: dict[str, tuple] = {}
        for name, b in self._active_batchers.items():
            sealed[name] = await asyncio.to_thread(b.seal_batch, pool=pool_per_env)
        for name, (batch, rewards) in sealed.items():
            self._active_batchers[name].rewards_by_hotkey = rewards

        # Worker acceptance means "admitted to the auction pool". Publish a
        # second, final /verdicts record after seal so miners can distinguish a
        # selected/rewarded candidate, an honest non-winner, and a deferred-proof
        # failure. This is observability only and cannot change selection.
        for batcher in self._active_batchers.values():
            self._record_auction_final_verdicts(batcher)

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

        window_batches = {
            name: sealed[name][0] for name, _ in self.env_mix if name in sealed
        }

        # Quarantine each window before retaining any of its groups. Rewards
        # and archives remain per-window; this gate only protects model state.
        combined_reject_counts: dict[str, int] = {}
        for _b in self._active_batchers.values():
            for _k, _v in dict(getattr(_b, "reject_counts", {})).items():
                combined_reject_counts[_k] = combined_reject_counts.get(_k, 0) + _v
        flat_window_batch = [
            group for env_batch in window_batches.values() for group in env_batch
        ]
        window_quarantine = assess_training_batch(
            flat_window_batch,
            reject_counts=combined_reject_counts,
        )
        _quarantine_archive = window_quarantine.to_archive()
        for _b in self._active_batchers.values():
            _b.training_quarantine = _quarantine_archive

        checkpoint_revisions = {
            str(getattr(b, "current_checkpoint_hash", ""))
            for b in self._active_batchers.values()
        }
        if len(checkpoint_revisions) != 1:
            logger.error(
                "Window %d has inconsistent checkpoint revisions across envs: %s",
                self._window_n, sorted(checkpoint_revisions),
            )
            discarded = self._training_accumulator.reset()
            accumulator_update = {
                "checkpoint_reset": discarded,
                "counts_before": discarded["counts"],
                "added": {name: 0 for name in per_env_targets},
                "not_accumulated": {
                    name: len(window_batches.get(name, ()))
                    for name in per_env_targets
                },
                "snapshot": self._training_accumulator.snapshot(),
            }
            accumulator_update["blocked_reason"] = "inconsistent_checkpoint"
        else:
            checkpoint_revision = next(iter(checkpoint_revisions))
            accumulator_update = self._training_accumulator.add_window(
                {} if window_quarantine.quarantined else window_batches,
                window_n=self._window_n,
                checkpoint_revision=checkpoint_revision,
            )
            if window_quarantine.quarantined:
                accumulator_update["blocked_reason"] = "window_quarantine"
                accumulator_update["not_accumulated"] = {
                    name: len(window_batches.get(name, ()))
                    for name in per_env_targets
                }

        accumulator_meta: dict[str, Any] = {
            "schema_version": 1,
            "window_groups": {
                name: len(window_batches.get(name, ()))
                for name in per_env_targets
            },
            **accumulator_update,
            "training_attempted": False,
            "trained": False,
            "reset_reason": None,
        }

        env_order = [name for name, _ in self.env_mix]
        accumulator_ready = (
            len(checkpoint_revisions) == 1 and self._training_accumulator.ready
        )
        batches = (
            self._training_accumulator.training_batches(env_order)
            if accumulator_ready else []
        )

        # Assess the balanced retained batch as a second model-health gate.
        # Reject spikes are window-scoped and were checked above, so they are
        # deliberately not summed across source windows here.
        accumulated_quarantine = assess_training_batch(
            [group for env_batch in batches for group in env_batch],
            reject_counts={},
        )
        accumulator_meta["accumulated_quarantine"] = (
            accumulated_quarantine.to_archive()
        )
        if accumulator_ready and accumulated_quarantine.quarantined:
            logger.warning(
                "Window %d accumulated batch quarantined from training: "
                "reasons=%s metrics=%s",
                self._window_n,
                accumulated_quarantine.reasons,
                accumulated_quarantine.metrics,
            )
            accumulator_meta["reset_reason"] = "accumulated_quarantine"
            accumulator_meta["discarded"] = self._training_accumulator.reset()
            accumulator_ready = False
            batches = []

        trained = False
        # Env-controlled skip: ``RELIQUARY_DISABLE_TRAIN=1`` bypasses the
        # train_step call entirely. Useful when the validator is configured
        # in inference-only mode (e.g. a frozen policy phase) or when the
        # train_step has a known OOM/leak pattern that's poisoning the
        # GPU pool across windows. With this flag set the balanced retained
        # batch stays pending while this window is archived normally.
        emergency_freeze = os.environ.get(
            "RELIQUARY_DISABLE_TRAIN", ""
        ).lower() in {"1", "true", "yes", "on"}
        publication_retry_pending = (
            self._trained_windows_since_publish >= self._publish_every
        )
        checkpoint_ceiling_reached = (
            TRAIN_UNTIL_CHECKPOINT_N > 0
            and self._checkpoint_n >= TRAIN_UNTIL_CHECKPOINT_N
        )
        skip_train = (
            emergency_freeze
            or checkpoint_ceiling_reached
            or publication_retry_pending
        )
        if accumulator_ready and skip_train:
            if emergency_freeze:
                blocked_reason = "emergency_training_freeze"
            elif checkpoint_ceiling_reached:
                blocked_reason = "training_checkpoint_ceiling"
            else:
                blocked_reason = "checkpoint_publication_pending"
            accumulator_meta["blocked_reason"] = blocked_reason
            logger.info(
                "Window %d: %s — retaining balanced batch and skipping "
                "train_step + publish (checkpoint=%d ceiling=%d)",
                self._window_n,
                blocked_reason,
                self._checkpoint_n,
                TRAIN_UNTIL_CHECKPOINT_N,
            )
        elif accumulator_ready:
            accumulator_meta["training_attempted"] = True
            try:
                # Forward/backward is the longest blocking step in the loop;
                # run it in a thread so the HTTP server keeps serving /state
                # and /submit while a window trains.
                self.train_model = await asyncio.to_thread(
                    train_step,
                    self.train_model, batches,
                    ref_model=(
                        self.base_ref_model
                        if self.base_ref_model is not None
                        else self.verify_model
                    ),
                    window_index=self._window_n,
                    **(
                        {"behavior_model": self.verify_model}
                        if RECOMPUTE_PI_OLD_FROM_VERIFY
                        else {}
                    ),
                )
                trained = True
            except TrainingStepSkipped as exc:
                logger.warning(
                    "train_step rejected for window %d: reason=%s "
                    "grad_norm=%s; archiving without checkpoint publication",
                    self._window_n,
                    exc.reason,
                    exc.grad_norm,
                )
                accumulator_meta["reset_reason"] = (
                    f"training_health_gate:{exc.reason}"
                )
            except Exception:
                # Don't let a training failure (e.g. CUDA OOM) skip
                # _archive_window — miners still need their R2 contribution
                # recorded so the EMA / on-chain weights reflect this window.
                logger.exception(
                    "train_step failed for window %d; archiving anyway and "
                    "skipping publish", self._window_n,
                )
                accumulator_meta["reset_reason"] = "train_step_failed"
            finally:
                # Reclaim any GPU memory the failed/successful train_step
                # held in its activation cache. This is critical when
                # train_step OOMs intermittently — without explicit cleanup
                # the partial allocations fragment the CUDA pool over
                # successive windows and eventually starve verify_commitment.
                _try_empty_cuda_cache()
                accumulator_meta["discarded"] = self._training_accumulator.reset()
                if accumulator_meta["reset_reason"] is None:
                    accumulator_meta["reset_reason"] = "training_consumed"
        else:
            total_subs = sum(len(b) for b in window_batches.values())
            total_target = sum(per_env_targets.values())
            retained = accumulator_meta["snapshot"]["counts"]
            logger.info(
                "Window %d sealed with %d/%d submissions; retained=%s — "
                "waiting for balanced training batch",
                self._window_n, total_subs, total_target, retained,
            )

        accumulator_meta["trained"] = trained
        accumulator_meta["post_action"] = self._training_accumulator.snapshot()
        self.server.set_training_accumulator_state(accumulator_meta["post_action"])
        log_structured(
            logger,
            logging.INFO,
            "validator_training_accumulator",
            {
                "window_n": self._window_n,
                "window_groups": accumulator_meta["window_groups"],
                "added": accumulator_meta["added"],
                "not_accumulated": accumulator_meta["not_accumulated"],
                "counts_after_add": accumulator_meta["snapshot"]["counts"],
                "ready_after_add": accumulator_meta["snapshot"]["ready"],
                "training_attempted": accumulator_meta["training_attempted"],
                "trained": trained,
                "blocked_reason": accumulator_meta.get("blocked_reason"),
                "reset_reason": accumulator_meta["reset_reason"],
                "post_action_counts": accumulator_meta["post_action"]["counts"],
            },
        )
        for _b in self._active_batchers.values():
            _b.training_accumulator = accumulator_meta

        self._set_state(WindowState.PUBLISHING)
        if trained:
            self._trained_windows_since_publish += 1
        # checkpoint_n only advances on publish. Publish cadence is based on
        # successful trained windows rather than exact window number so a
        # quarantined boundary window cannot freeze the public checkpoint. Once
        # the cadence is reached, retry a failed upload without applying another
        # optimizer step to the pending candidate.
        next_n = self._checkpoint_n + 1
        should_publish = not emergency_freeze and (
            self._trained_windows_since_publish >= self._publish_every
            or (
                trained
                and self._checkpoint_store.current_manifest() is None
            )
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
                if publication_retry_pending:
                    discarded = self._training_accumulator.reset()
                    post_publish_state = self._training_accumulator.snapshot()
                    accumulator_meta["post_publish_discarded"] = discarded
                    accumulator_meta["post_action"] = post_publish_state
                    self.server.set_training_accumulator_state(
                        post_publish_state
                    )
                    for _b in self._active_batchers.values():
                        _b.training_accumulator = accumulator_meta
                    logger.info(
                        "Published pending checkpoint %d; discarded %d "
                        "retained groups generated against its parent",
                        entry.checkpoint_n,
                        sum(discarded["counts"].values()),
                    )
                logger.info(
                    "Published checkpoint %d to %s@%s and refreshed verify_model",
                    entry.checkpoint_n, entry.repo_id, entry.revision[:12],
                )
            except Exception:
                logger.exception("HF publish failed; staying on previous checkpoint")
        elif trained:
            logger.info(
                "Skipping HF publish for window_n=%d "
                "(%d/%d trained windows since last publish)",
                self._window_n,
                self._trained_windows_since_publish,
                self._publish_every,
            )
        self.server.set_training_publish_state({
            "trained_windows_since_publish": (
                self._trained_windows_since_publish
            ),
            "publish_interval": self._publish_every,
            "publication_pending": (
                self._trained_windows_since_publish >= self._publish_every
            ),
        })

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
            difficulty_by_id = getattr(
                batcher, "difficulty_auction_metadata_by_id", {}
            )
            difficulty_meta = (
                difficulty_by_id.get(id(s), {})
                if isinstance(difficulty_by_id, dict)
                else {}
            )
            arrival_ts = getattr(s, "arrival_ts", None)
            window_opened_wall_ts = getattr(
                batcher, "window_opened_wall_ts", None
            )
            arrival_age_seconds = None
            if arrival_ts is not None and window_opened_wall_ts is not None:
                try:
                    candidate_age = float(arrival_ts) - float(
                        window_opened_wall_ts
                    )
                except (TypeError, ValueError):
                    candidate_age = float("nan")
                if math.isfinite(candidate_age) and candidate_age >= 0.0:
                    arrival_age_seconds = candidate_age
            return {
                "arrival_ts": arrival_ts,
                "arrival_age_seconds": arrival_age_seconds,
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
                "difficulty_auction_value": difficulty_meta.get("value"),
                "difficulty_auction_mean_reward": difficulty_meta.get(
                    "mean_reward"
                ),
                "difficulty_auction_reward_std": difficulty_meta.get(
                    "reward_std"
                ),
                "difficulty_auction_reward_count": difficulty_meta.get(
                    "reward_count"
                ),
                "difficulty_auction_mode": (
                    "production"
                    if getattr(batcher, "difficulty_auction_enabled", False)
                    else "observation_only"
                ),
                "difficulty_auction_eligible": difficulty_meta.get(
                    "eligible",
                    True if "status" in difficulty_meta else None,
                ),
                "difficulty_auction_rank": difficulty_meta.get("rank"),
                "difficulty_auction_selected": difficulty_meta.get(
                    "selected", difficulty_meta.get("shadow_selected")
                ),
                "difficulty_auction_status": difficulty_meta.get("status"),
                "difficulty_auction_proof_attempted": difficulty_meta.get(
                    "proof_attempted"
                ),
                "difficulty_auction_proof_passed": difficulty_meta.get(
                    "proof_passed"
                ),
                "difficulty_auction_forensic_sampled": difficulty_meta.get(
                    "forensic_sampled", False
                ),
                "difficulty_auction_forensic_passed": difficulty_meta.get(
                    "forensic_passed"
                ),
                "difficulty_auction_rank_entropy_source": difficulty_meta.get(
                    "rank_entropy_source"
                ),
                "difficulty_auction_operator_id": difficulty_meta.get(
                    "operator_id"
                ),
            }

        def _difficulty_auction_payload(batcher):
            payload = getattr(batcher, "difficulty_auction_shadow", None)
            if isinstance(payload, dict):
                return payload
            return {
                "schema_version": 1,
                "status": "unavailable",
                "mode": "observation_only",
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
        logical_group_dedup: dict[str, dict[str, int]] = {}
        grader_failures: dict[str, int] = {}
        grader_failures_by_environment: dict[str, dict[str, int]] = {}

        for env_name, batcher in batcher_dict.items():
            env_obj = self.envs.get(env_name, self.env)
            env_batch, env_rewards = sealed_dict.get(env_name, ([], {}))

            batched_keys = {(s.hotkey, s.prompt_idx) for s in env_batch}

            for s in env_batch:
                try:
                    problem = env_obj.get_problem(s.prompt_idx)
                except Exception:
                    # A lazy-dataset fetch failure must not abort the whole
                    # window's archive — keep the entry (prompt_idx/rewards are
                    # what the cooldown rebuild needs), just without prompt text.
                    logger.warning(
                        "archive: get_problem(%d) failed; archiving without prompt text",
                        s.prompt_idx,
                    )
                    problem = {}
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
                    "selection_digest": s.selection_digest.hex(),
                    "claimed_checkpoint_hash": s.claimed_checkpoint_hash,
                    "sketch_diff_max": s.sketch_diff_max,
                    "lp_dev_max": s.lp_dev_max,
                    "dist_q10_min": s.dist_q10_min,
                    "all_token_auth_shadow_findings": getattr(
                        s, "all_token_auth_shadow_findings", 0
                    ),
                    "all_token_auth_shadow_min_prob": getattr(
                        s, "all_token_auth_shadow_min_prob", None
                    ),
                    "all_token_auth_shadow_positive_findings": getattr(
                        s, "all_token_auth_shadow_positive_findings", 0
                    ),
                    "all_token_auth_shadow_positive_min_prob": getattr(
                        s, "all_token_auth_shadow_positive_min_prob", None
                    ),
                    "code_semantic_auth_findings": getattr(
                        s, "code_semantic_auth_findings", 0
                    ),
                    "code_semantic_auth_min_prob": getattr(
                        s, "code_semantic_auth_min_prob", None
                    ),
                    "code_semantic_auth_positive_findings": getattr(
                        s, "code_semantic_auth_positive_findings", 0
                    ),
                    "code_semantic_auth_positive_min_prob": getattr(
                        s, "code_semantic_auth_positive_min_prob", None
                    ),
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
                    "selection_digest": s.selection_digest.hex(),
                    "sketch_diff_max": s.sketch_diff_max,
                    "lp_dev_max": s.lp_dev_max,
                    "dist_q10_min": s.dist_q10_min,
                    "all_token_auth_shadow_findings": getattr(
                        s, "all_token_auth_shadow_findings", 0
                    ),
                    "all_token_auth_shadow_min_prob": getattr(
                        s, "all_token_auth_shadow_min_prob", None
                    ),
                    "all_token_auth_shadow_positive_findings": getattr(
                        s, "all_token_auth_shadow_positive_findings", 0
                    ),
                    "all_token_auth_shadow_positive_min_prob": getattr(
                        s, "all_token_auth_shadow_positive_min_prob", None
                    ),
                    "code_semantic_auth_findings": getattr(
                        s, "code_semantic_auth_findings", 0
                    ),
                    "code_semantic_auth_min_prob": getattr(
                        s, "code_semantic_auth_min_prob", None
                    ),
                    "code_semantic_auth_positive_findings": getattr(
                        s, "code_semantic_auth_positive_findings", 0
                    ),
                    "code_semantic_auth_positive_min_prob": getattr(
                        s, "code_semantic_auth_positive_min_prob", None
                    ),
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

            reservations = getattr(
                batcher, "logical_group_reservation_count", 0
            )
            duplicates = getattr(
                batcher, "logical_group_duplicate_rejects", 0
            )
            logical_group_dedup[env_name] = {
                "reservations": (
                    reservations
                    if isinstance(reservations, int)
                    and not isinstance(reservations, bool)
                    else 0
                ),
                "duplicate_rejects": (
                    duplicates
                    if isinstance(duplicates, int)
                    and not isinstance(duplicates, bool)
                    else 0
                ),
            }
            env_grader_failures = {
                str(reason): int(count)
                for reason, count in dict(
                    getattr(batcher, "grader_failures", {})
                ).items()
            }
            grader_failures_by_environment[env_name] = env_grader_failures
            for reason, count in env_grader_failures.items():
                grader_failures[reason] = grader_failures.get(reason, 0) + count

        server_reject_summary = {
            str(reason): int(count)
            for reason, count in dict(
                getattr(self.server, "_recent_reject_counts", {})
            ).items()
            if isinstance(count, int) and not isinstance(count, bool)
        }
        # Worker rejects appear in both counters while HTTP cheap rejects only
        # appear in the server counter. Taking the maximum preserves a total
        # without double-counting and closes the public archive blind spot.
        for reason, count in server_reject_summary.items():
            combined_reject_counts[reason] = max(
                combined_reject_counts.get(reason, 0), count
            )

        env_names_list = list(batcher_dict.keys())
        # Backward-compat: keep "environment" (singular) pointing to the first
        # env so older readers that pre-date multi-env don't silently break.
        difficulty_auction_payload = {
            env_name: _difficulty_auction_payload(env_batcher)
            for env_name, env_batcher in batcher_dict.items()
        }
        archive = {
            "window_start": first_batcher.window_start,
            "validator_hotkey": self.wallet.hotkey.ss58_address,  # provenance
            "randomness": first_batcher.randomness,
            "environment": env_names_list[0],   # legacy singular, kept for compat
            "environments": env_names_list,      # multi-env canonical field
            "force_seal_reason": getattr(first_batcher, "force_seal_reason", None),
            "window_opened_wall_ts_by_environment": {
                env_name: getattr(env_batcher, "window_opened_wall_ts", None)
                for env_name, env_batcher in batcher_dict.items()
            },
            "batch": batch_entries,
            "runners_up": runners_up,
            "reject_summary": combined_reject_counts,
            "server_reject_summary": server_reject_summary,
            "logical_group_dedup": logical_group_dedup,
            # Canonical production name plus the historical alias consumed by
            # existing dashboards and replay scripts.
            "difficulty_auction": difficulty_auction_payload,
            "difficulty_auction_shadow": difficulty_auction_payload,
            "grader_failures": grader_failures,
            "grader_failures_by_environment": (
                grader_failures_by_environment
            ),
            "rejected": rejected_entries,
            "training_quarantine": getattr(
                batcher,
                "training_quarantine",
                {"quarantined": False, "reasons": [], "metrics": {}},
            ),
            "training_accumulator": getattr(
                first_batcher,
                "training_accumulator",
                {"schema_version": 1, "trained": False},
            ),
            "training_kl_reference": dict(self.kl_reference_state),
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
                "training_kl_reference": dict(self.kl_reference_state),
                "batch_size": B_BATCH,
                "m_rollouts_per_prompt": M_ROLLOUTS,
                "environment": self.env.name,
                "netuid": self.netuid,
                "sigma_min": SIGMA_MIN,
                "bootstrap_sigma_min": BOOTSTRAP_SIGMA_MIN,
                "min_eos_probability": MIN_EOS_PROBABILITY,
                "forced_seed_enforce": FORCED_SEED_ENFORCE,
                "forced_seed_protocol_version": FORCED_SEED_PROTOCOL_VERSION,
                "forced_seed_consistency_floor": (
                    FORCED_SEED_CONSISTENCY_FLOOR
                ),
                "forced_seed_rollout_floor": FORCED_SEED_ROLLOUT_FLOOR,
                "forced_seed_cdf_enforce": FORCED_SEED_CDF_ENFORCE,
                "forced_seed_cdf_boundary_epsilon": (
                    FORCED_SEED_CDF_BOUNDARY_EPSILON
                ),
                "legacy_merkle_root_enforce": LEGACY_MERKLE_ROOT_ENFORCE,
                "difficulty_auction_enforce": DIFFICULTY_AUCTION_ENFORCE,
                "difficulty_auction_environments": list(
                    DIFFICULTY_AUCTION_ENVIRONMENTS
                ),
                "difficulty_auction_collection_seconds": (
                    WINDOW_COLLECTION_SECONDS
                ),
                "difficulty_auction_max_slots_per_operator": (
                    MAX_AUCTION_SLOTS_PER_OPERATOR
                ),
                "difficulty_auction_proof_attempt_limit": (
                    MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
                ),
                "difficulty_auction_proof_wall_limit_seconds": (
                    MAX_PROOF_WALL_SECONDS
                ),
                "difficulty_auction_operator_proof_failure_cap": (
                    MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW
                ),
                "difficulty_auction_shadow_enabled": (
                    DIFFICULTY_AUCTION_SHADOW_ENABLED
                ),
                "difficulty_auction_shadow_environments": list(
                    DIFFICULTY_AUCTION_SHADOW_ENVIRONMENTS
                ),
                "difficulty_auction_shadow_delta": DIFFICULTY_AUCTION_DELTA,
                "difficulty_auction_shadow_max_candidates": (
                    DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES
                ),
                "difficulty_auction_shadow_max_slots_per_operator": (
                    DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR
                ),
                "logprob_is_eps": LOGPROB_IS_EPS,
                "r2_bucket": os.getenv("R2_BUCKET_ID", "reliquary"),
                "http_host": self.server.host,
                "http_port": self.server.port,
                "external_ip_configured": bool(self.external_ip),
                "external_port": self.external_port,
            },
        )

    async def run(self, subtensor) -> None:
        from reliquary.infrastructure.archive_queue import get_archive_queue

        archive_queue = get_archive_queue()
        self.server.configure_archive_queue_telemetry(archive_queue.snapshot)
        self.server.configure_registration_gate(
            lambda: self._refresh_registered_hotkeys(
                force=True,
                reason="on_demand",
            ),
        )
        await self._refresh_registered_hotkeys(force=True, reason="startup")
        await self.server.start()
        await self._serve_axon_on_chain(subtensor)
        await self._apply_resume_from()                  # ← resume before bootstrap
        await self._bootstrap_state_from_external()
        self._publish_window_preparation_state()
        await self._rebuild_cooldown_from_history()
        await self._rebuild_hashes_from_history()
        self._log_startup_config_banner()

        # Start the background archive-upload worker. It scans the queue
        # directory for any pending payloads (from before this restart
        # or accumulated during R2 downtime) and uploads them via sync
        # boto3 with exponential backoff. Cancelled cleanly on shutdown.
        self._archive_worker_task = asyncio.create_task(
            archive_queue.run_forever(),
            name="archive_queue_worker",
        )
        self._registration_refresh_task = asyncio.create_task(
            self._maintain_registration_cache(),
            name="registration_cache_refresh",
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
                    if self._candidate_window_n is not None:
                        self._set_window_preparation_stage(
                            "registration_refresh"
                        )
                    await self._refresh_registered_hotkeys(
                        reason="window_boundary"
                    )
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

                    # Persist the cooldown on a fixed window cadence, independent
                    # of the publish cadence (which can stall): keeps the snapshot
                    # within COOLDOWN_SNAPSHOT_INTERVAL_WINDOWS of current_window
                    # so the gap replay always covers it.
                    self._windows_since_cooldown_snapshot += 1
                    if (
                        self._windows_since_cooldown_snapshot
                        >= COOLDOWN_SNAPSHOT_INTERVAL_WINDOWS
                    ):
                        await self._snapshot_cooldown()
                        self._windows_since_cooldown_snapshot = 0

                    # set_weights is owned by a concurrent WeightOnlyValidator
                    # task running off the same R2 archives; no need to do it
                    # here. The trainer is purely about training + uploads.
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("Window iteration failed")
                    self._rollback_preopen_window(exc)
                    # Reset to READY so the next iteration doesn't spin on error state.
                    self.server.set_active_batchers({})
                    self._active_batchers = {}
                    self._set_state(WindowState.READY)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
        finally:
            registration_task = self._registration_refresh_task
            if registration_task is not None and not registration_task.done():
                registration_task.cancel()
                try:
                    await asyncio.wait_for(registration_task, timeout=5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
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
        """At startup, restore per-env cooldown from the run-keyed R2 snapshot,
        then replay only the windows recorded since it was taken — so the FULL
        cooldown survives a restart, not just the last COOLDOWN_REBUILD_LOOKBACK
        windows (the old replay exploit). Falls back to a bounded archive scan
        when no snapshot exists for the DEFAULT run (first start / pre-snapshot
        transition); an explicit fresh RELIQUARY_TRAINING_RUN_ID with no snapshot
        starts empty — a new model must be allowed to re-see every prompt.
        """
        current_window = self._window_n
        snapshot = None
        try:
            snapshot = await storage.download_json(
                _cooldown_snapshot_key(TRAINING_RUN_ID)
            )
        except Exception:
            logger.exception("Failed to read cooldown snapshot")

        if snapshot and snapshot.get("run_id") == TRAINING_RUN_ID:
            try:
                envs = snapshot.get("envs", {}) or {}
                for env_name, cooldown_map in self._cooldown_per_env.items():
                    cooldown_map.import_state(envs.get(env_name, {}))
                snapshot_window = int(snapshot.get("snapshot_window", current_window))
            except Exception:
                # Corrupt / partially-written / tampered snapshot — must not
                # crash startup. Discard any partial restore and fall through.
                logger.exception(
                    "Corrupt cooldown snapshot for run=%s; discarding it", TRAINING_RUN_ID,
                )
                for cooldown_map in self._cooldown_per_env.values():
                    cooldown_map.import_state({})
            else:
                gap = max(0, current_window - snapshot_window)
                if gap > 0:
                    await self._replay_cooldown_gap(current_window, gap)
                logger.info(
                    "Restored cooldown from snapshot run=%s snapshot_window=%d "
                    "gap=%d (current=%d, sizes=%s)",
                    TRAINING_RUN_ID, snapshot_window, gap, current_window,
                    {n: len(m) for n, m in self._cooldown_per_env.items()},
                )
                return

        if TRAINING_RUN_ID != "default":
            logger.info(
                "No cooldown snapshot for fresh run=%s — starting empty (reset).",
                TRAINING_RUN_ID,
            )
            return

        # Default run, no snapshot (first start / pre-snapshot transition):
        # bounded archive rebuild — better than empty, and the first snapshot
        # makes subsequent restarts complete.
        await self._rebuild_cooldown_from_archives(
            current_window, COOLDOWN_REBUILD_LOOKBACK,
        )

    async def _rebuild_cooldown_from_archives(self, current_window: int, n: int) -> None:
        """Rebuild every env's cooldown from scratch from the last ``n`` R2
        archives (used only when no snapshot is available)."""
        try:
            archives = await storage.list_recent_datasets(
                current_window=current_window + 1, n=n,
            )
            for env_name, cooldown_map in self._cooldown_per_env.items():
                cooldown_map.rebuild_from_history(
                    _filter_archives_for_env(archives, env_name),
                    current_window=current_window,
                )
            logger.info(
                "Rebuilt cooldown from %d archive windows (no snapshot; "
                "current=%d, sizes=%s)",
                len(archives), current_window,
                {n2: len(m) for n2, m in self._cooldown_per_env.items()},
            )
        except Exception:
            logger.exception(
                "Failed to rebuild cooldown from history; starting empty"
            )

    async def _replay_cooldown_gap(self, current_window: int, gap: int) -> None:
        """Merge the windows recorded since the snapshot into the restored
        cooldown. Bounded by COOLDOWN_REBUILD_LOOKBACK; in normal operation the
        gap is ~the snapshot (publish) cadence."""
        n = min(gap + 1, COOLDOWN_REBUILD_LOOKBACK)
        try:
            archives = await storage.list_recent_datasets(
                current_window=current_window + 1, n=n,
            )
            for env_name, cooldown_map in self._cooldown_per_env.items():
                cooldown_map.apply_history(
                    _filter_archives_for_env(archives, env_name),
                    current_window=current_window,
                )
            if gap + 1 > COOLDOWN_REBUILD_LOOKBACK:
                logger.warning(
                    "Cooldown gap %d exceeds replay cap %d; prompts in the "
                    "uncovered span may be re-eligible. Widen "
                    "COOLDOWN_REBUILD_LOOKBACK if this recurs.",
                    gap, COOLDOWN_REBUILD_LOOKBACK,
                )
        except Exception:
            logger.exception("Cooldown gap-replay failed; using snapshot only")

    async def _snapshot_cooldown(self) -> None:
        """Persist the per-env cooldown maps to R2, keyed by the training run id,
        so a restart restores the full cooldown without replaying history. Best
        effort — a snapshot failure must never break the window loop."""
        try:
            window = self._window_n

            def _build() -> dict:
                # Copy can be multi-MB (cooldown never expires) — build it off
                # the event loop. Safe: the window loop is sequential here, no
                # concurrent record_batched between seal and the next window.
                return {
                    "run_id": TRAINING_RUN_ID,
                    "snapshot_window": window,
                    "envs": {
                        name: cd.export_state()
                        for name, cd in self._cooldown_per_env.items()
                    },
                }

            snapshot = await asyncio.to_thread(_build)
            if await storage.upload_json(
                _cooldown_snapshot_key(TRAINING_RUN_ID), snapshot
            ):
                logger.info(
                    "Snapshotted cooldown run=%s window=%d (sizes=%s)",
                    TRAINING_RUN_ID, self._window_n,
                    {n: len(m) for n, m in self._cooldown_per_env.items()},
                )
        except Exception:
            logger.exception("Cooldown snapshot failed (non-fatal)")

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
        target_window = (
            self._candidate_window_n
            if self._candidate_window_n is not None
            else self._window_n
        )
        self._set_window_preparation_stage("drand_boundary")
        from reliquary.infrastructure.drand import get_current_chain
        ci = get_current_chain()
        delay = chain.seconds_until_next_drand_boundary(
            time.time(), ci["genesis_time"], ci["period"],
        )
        if delay > 0:
            logger.info(
                "Window %d: waiting %.2fs for next drand boundary before OPEN",
                target_window, delay,
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
            # Both calls do synchronous HTTP to the drand relays; run them off
            # the event loop so a slow relay can't stall the window-open path
            # (and the HTTP server) while the seed is fetched.
            chain_info = await asyncio.to_thread(get_current_chain)
            drand_round = chain.compute_current_drand_round(
                time.time(), chain_info["genesis_time"], chain_info["period"],
            )
            beacon = await asyncio.to_thread(
                get_beacon, round_id=str(drand_round), use_drand=True,
            )
            randomness = chain.compute_window_randomness(
                None, beacon["randomness"], drand_round=beacon["round"],
            )
            return randomness, beacon
        # Legacy mock-only path: still uses block_hash so tests that
        # disable drand keep working without a live drand fetch.
        block_hash = await chain.get_block_hash(subtensor, target_window)
        return chain.compute_window_randomness(block_hash), None

    async def _fetch_seal_randomness(self) -> str:
        """Fetch the drand beacon current at seal (post-deadline) to key the
        forensic sample. Telemetry only, so any failure returns "" (no sample)
        rather than blocking the seal. Off-loop: the HTTP fetch must not stall
        the event loop while the window seals.
        """
        if not self.use_drand:
            return ""
        try:
            import time
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = await asyncio.to_thread(get_current_chain)
            drand_round = chain.compute_current_drand_round(
                time.time(), chain_info["genesis_time"], chain_info["period"],
            )
            beacon = await asyncio.to_thread(
                get_beacon, round_id=str(drand_round), use_drand=True,
            )
            return str(beacon["randomness"])
        except Exception:
            logger.warning("seal-randomness fetch failed; forensic sample skipped")
            return ""
