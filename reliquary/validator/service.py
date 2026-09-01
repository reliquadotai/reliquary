"""Validator main loop — v2.1 batch-driven state machine (OPEN→TRAINING→PUBLISHING→READY)."""

from __future__ import annotations

import asyncio
import functools
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable

from reliquary.constants import (
    AUCTION_ADMISSION_DRAIN_DEADLINE_SECONDS,
    AUCTION_EARLY_CLOSE_MIN_SECONDS,
    AUCTION_EARLY_CLOSE_MODE,
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
    EXPERIMENTAL_CHECKPOINT_EPOCH_CANDIDATES_PER_LANE,
    EXPERIMENTAL_CHECKPOINT_EPOCH_COLLECTION_SECONDS,
    EXPERIMENTAL_CHECKPOINT_EPOCH_COMMITMENTS_PER_OPERATOR_PER_LANE,
    EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED,
    EXPERIMENTAL_CHECKPOINT_EPOCH_REVEAL_SECONDS,
    EXPERIMENTAL_CHECKPOINT_EPOCH_TRAINING_MODE,
    EXPERIMENTAL_CHECKPOINT_EPOCH_WARMUP_ROUNDS,
    FILL_CLOSED_ADMISSION_BUDGET_PER_ENV,
    FILL_CLOSED_ENABLED,
    FILL_CLOSED_FIRST_PICK_SECONDS,
    FILL_CLOSED_MAX_SECONDS,
    FILL_CLOSED_PICK_PIPELINE_DEPTH,
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
    LR_WARMUP_WINDOWS,
    MATH_ADMISSION_WORKERS,
    M_ROLLOUTS,
    MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW,
    MAX_GRADING_STARTS_PER_WINDOW,
    MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW,
    MAX_PROOF_WALL_SECONDS,
    MAX_SEAL_QUEUE_DRAIN_SECONDS,
    MIN_EOS_PROBABILITY,
    POLL_INTERVAL_SECONDS,
    PPO_CLIP_EPSILON_HIGH,
    PPO_CLIP_EPSILON_LOW,
    PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD,
    PROMPT_RANGE_SIZE,
    PROTOCOL_GENERATION_CONTRACT,
    PROTOCOL_PROFILE_ID,
    PROTOCOL_VERSION,
    PROOF_ADMISSION_STALL_POLL_SECONDS,
    PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    REGISTERED_HOTKEY_CACHE_TTL_SECONDS,
    REGISTERED_HOTKEY_REFRESH_TIMEOUT_SECONDS,
    RECOMPUTE_PI_OLD_FROM_VERIFY,
    SHAPE_LEN_FRAC,
    SHAPE_PENALTY,
    SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS,
    SPARSE_VALID_IDLE_SEAL_SECONDS,
    SPARSE_VALID_MAX_WINDOW_SECONDS,
    SIGMA_MIN,
    SUBMISSION_UPLOAD_GRACE_SECONDS,
    SUBNET_START_BLOCK,
    TRAIN_UNTIL_CHECKPOINT_N,
    VALIDATOR_HTTP_PORT,
    WANDB_TRAINING_VERSION,
    WINDOW_LENGTH,
    WINDOW_COLLECTION_SECONDS,
    WINDOW_TIMEOUT_SECONDS,
    CODE_ADMISSION_WORKERS,
)
from reliquary.environment import load_environments
from reliquary.environment.base import Environment
from reliquary.infrastructure import chain, storage
from reliquary.protocol.submission import RejectReason, RolloutSubmission, WindowState
from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CHECKPOINT_EPOCH_REQUIRED_WINDOW_COUNT,
    CHECKPOINT_EPOCH_SCHEDULE_MODE,
    EpochPlan,
    EpochWindow,
    ProtocolBinding,
    SignedEpochCommitmentSet,
    WindowSchedule,
    commitment_set_sha256,
    commitment_set_signing_bytes,
    generation_contract_sha256,
    manifest_sha256,
)
from reliquary.validator import telemetry
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.checkpoint import CheckpointStore
from reliquary.validator.checkpoint_epoch_runtime import (
    EpochCommitIntent,
    EpochStore,
    SignedEpochIntent,
    build_epoch_intent,
    canonical_signed_intent_bytes,
    intent_signing_bytes,
    plan_from_intent,
)
from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap
from reliquary.validator.dedup import RolloutHashSet
from reliquary.validator.errors import FatalProofPlaneError
from reliquary.validator.observability import log_structured, runtime_revision
from reliquary.validator.proof_scheduler import (
    GlobalProofScheduler,
    ProofExecution,
    ProofInvocation,
    SchedulerState,
)
from reliquary.validator.quarantine import assess_training_batch
from reliquary.validator.resume import checkpoint_n_from_commit_title
from reliquary.validator.server import ValidatorServer
from reliquary.validator.training import TrainingStepSkipped, train_step
from reliquary.validator.training_accumulator import BalancedTrainingAccumulator
from reliquary.validator.utility_telemetry import UtilityTelemetryWriter


_WINDOW_ACTIVATION_RANDOMNESS_DOMAIN = b"reliquary/window-activation/v1\x00"


def _bind_window_activation_randomness(
    beacon_randomness: str,
    *,
    target_window: int,
    activation_nonce: bytes,
) -> str:
    """Bind public beacon material to a nonce fixed before publication."""
    digest = hashlib.sha256()
    digest.update(_WINDOW_ACTIVATION_RANDOMNESS_DOMAIN)
    digest.update(int(target_window).to_bytes(8, "big", signed=False))
    encoded_randomness = str(beacon_randomness).encode("utf-8")
    digest.update(len(encoded_randomness).to_bytes(4, "big"))
    digest.update(encoded_randomness)
    digest.update(bytes(activation_nonce))
    return digest.hexdigest()


logger = logging.getLogger(__name__)

_HF_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_STARTUP_HASH_REBUILD_TIMEOUT_SECONDS = 30.0

# v6.1 (R39): how often the between-windows rotation gate re-asks the
# trainer's consumption cursor. Deliberately NOT the seal loop's 0.5 s --
# each ask is a bounded R2 GET, and the gate is a wait of
# seconds-to-a-minute, not a hot loop. Not an operator knob: nothing
# downstream is sized off it.
FILL_CLOSED_ROTATION_POLL_SECONDS = 2.0


class CheckpointEpochExecutionError(RuntimeError):
    """A partially consumed experimental epoch requires a clean restart."""


def _cooldown_snapshot_key(run_id: str) -> str:
    """R2 key for the run-keyed cooldown snapshot."""
    return f"cooldown_snapshots/{run_id}.json"


def _content_cooldown_snapshot_key(run_id: str) -> str:
    return f"content_cooldown_snapshots/{run_id}.json.gz"


def _content_cooldown_local_path(run_id: str) -> Path:
    state_dir = Path(
        os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    )
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    return state_dir / "content_cooldown" / f"{safe_run_id}.json.gz"


def _prompt_mismatch_circuit_local_path(
    run_id: str,
    *,
    netuid: int,
    validator_hotkey: str,
) -> Path:
    state_dir = Path(
        os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    )
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    validator_tag = hashlib.sha256(
        str(validator_hotkey).encode("utf-8")
    ).hexdigest()[:12]
    return (
        state_dir
        / "prompt_mismatch_circuit"
        / f"{safe_run_id}.netuid-{int(netuid)}.{validator_tag}.json"
    )


def _no_reveal_circuit_local_path(
    run_id: str,
    *,
    netuid: int,
    validator_hotkey: str,
) -> Path:
    state_dir = Path(
        os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    )
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    validator_tag = hashlib.sha256(
        str(validator_hotkey).encode("utf-8")
    ).hexdigest()[:12]
    return (
        state_dir
        / "no_reveal_circuit"
        / f"{safe_run_id}.netuid-{int(netuid)}.{validator_tag}.json"
    )


def _prompt_source_identity(environment: Environment) -> dict[str, str]:
    """Return the immutable, secret-free identity of one prompt source."""
    snapshot: dict[str, Any] = {}
    source_health = getattr(environment, "source_health", None)
    if callable(source_health):
        try:
            candidate = source_health()
            if isinstance(candidate, dict):
                snapshot = candidate
        except Exception as exc:  # noqa: BLE001 - optional third-party health hook
            # Namespace construction must not make validator startup depend on
            # an optional health implementation. Concrete environments expose
            # the same values on their lazy dataset below.
            logger.debug(
                "prompt source identity health unavailable environment=%s error=%s",
                getattr(environment, "name", type(environment).__name__),
                type(exc).__name__,
            )
    dataset = getattr(environment, "_dataset", None)
    repo = snapshot.get("repo") or getattr(dataset, "_repo", None)
    revision = snapshot.get("revision") or getattr(dataset, "_revision", None)
    implementation = (
        f"{type(environment).__module__}.{type(environment).__qualname__}"
    )
    return {
        "implementation": implementation,
        "repo": str(repo).strip() if repo is not None else "<unreported>",
        "revision": (
            str(revision).strip() if revision is not None else "<unreported>"
        ),
    }


def _prompt_mismatch_circuit_namespace(
    *,
    run_id: str,
    netuid: int,
    validator_hotkey: str,
    environments: dict[str, Environment],
) -> str:
    """Fingerprint every validator-owned input that defines prompt binding."""
    payload = {
        "schema": 2,
        "run_id": str(run_id),
        "network": os.environ.get("BT_NETWORK", "").strip(),
        "netuid": int(netuid),
        "validator_hotkey": str(validator_hotkey).strip(),
        "generation_contract": dict(PROTOCOL_GENERATION_CONTRACT),
        "prompt_sources": {
            name: _prompt_source_identity(environment)
            for name, environment in sorted(environments.items())
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"prompt-binding-v2:{digest}"


def _read_gzip_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("content cooldown snapshot must be a JSON object")
    return value


def _write_gzip_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".content-cooldown.", suffix=".json.gz", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
                compressed.write(
                    json.dumps(
                        value, separators=(",", ":"), sort_keys=True
                    ).encode("utf-8")
                )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _filter_archives_for_env(archives: list[dict], env_name: str) -> list[dict]:
    """Return a filtered view of archives containing only entries for ``env_name``.

    Handles both old (pre-multi-env) and new archive shapes:
      * Old: top-level ``"environment"`` (singular), no per-entry ``env_name``.
            All batch entries belong to that env.
      * New: per-entry ``"env_name"`` field. Filter to matching entries.
    """
    out = []
    for archive in archives:
        if archive.get("window_status", "completed") == "aborted":
            continue
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
    content_cooldown_map: ContentCooldownMap | None = None,
    hash_set: RolloutHashSet | None,
    tokenizer,
    bootstrap: bool = False,
    queue_drained_predicate=None,
    emit_training_batch_fn=None,
    operator_by_hotkey: dict[str, str] | None = None,
    proof_scheduler=None,
    verify_commitment_proofs_fn=None,
    experimental_epoch_ranking: bool = False,
    experimental_prompt_range: tuple[int, int] | None = None,
    collection_seconds: float | None = None,
    max_productive_candidates: int | None = None,
    max_ranked_proof_attempts: int | None = None,
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

    ``emit_training_batch_fn`` is v6 only: called with one environment's
    picked B_BATCH chunk every time that environment takes a pick, from
    whichever thread called ``pick_training_batch`` (see
    ``GrpoWindowBatcher.pick_training_batch``). ``None`` here -- no
    caller wires a detached-trainer writer into it yet; see
    ``docs/superpowers/plans/2026-08-28-fill-closed-window-v6.md``,
    Component 4.
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
        content_cooldown_map=content_cooldown_map,
        hash_set=hash_set,
        bootstrap=bootstrap,
        completion_text_fn=_completion_text,
        canonical_prompt_tokens_fn=_canonical_prompt_tokens,
        queue_drained_predicate=queue_drained_predicate,
        emit_training_batch_fn=emit_training_batch_fn,
        operator_by_hotkey=operator_by_hotkey,
        proof_scheduler=proof_scheduler,
        verify_commitment_proofs_fn=verify_commitment_proofs_fn,
        experimental_epoch_ranking=experimental_epoch_ranking,
        experimental_prompt_range=experimental_prompt_range,
        collection_seconds=collection_seconds,
        max_productive_candidates=max_productive_candidates,
        max_ranked_proof_attempts=max_ranked_proof_attempts,
    )



def load_validator_replica(
    local_path: str, *, isolated_plane: bool, **load_kwargs,
):
    """Default: load a HF checkpoint in bfloat16 with the configured attention
    implementation, onto whichever device this process's replicas belong to
    (the CPU once an isolated proof plane holds the working replicas).

    ``isolated_plane`` must say whether a plane was BUILT, not whether the flag
    is set — see ``validator_replica_device``. ``load_kwargs`` carries the
    CLI's pinned base-model revision. Both the boot load and the resume load
    come through here so the device decision cannot drift between them.
    """
    import torch
    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared.modeling import load_text_generation_model
    from reliquary.validator.proof_worker import validator_replica_device

    return load_text_generation_model(
        local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        **load_kwargs,
    ).to(validator_replica_device(isolated_plane=isolated_plane)).eval()


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


def _coerce_lr_schedule_step(raw) -> int | None:
    """Profile values are external data: accept only non-negative real ints
    (bool is an int subclass; floats/strings from a hand-edited profile are
    rejected -> full warmup, the fail-closed direction)."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw if raw >= 0 else None


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
        proof_devices: tuple[str, ...] | None = None,
        proof_models: dict[str, Any] | None = None,
        proof_capacity_qualification: dict[str, Any] | None = None,
        proof_worker_pool: Any = None,
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
                "ppo_clip_epsilon_low": PPO_CLIP_EPSILON_LOW,
                "ppo_clip_epsilon_high": PPO_CLIP_EPSILON_HIGH,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "lr_warmup_windows": LR_WARMUP_WINDOWS,
                "lr_schedule": "warmup_then_flat",
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
        # This label is advanced only after a complete train -> verify state
        # copy. A checkpoint manifest alone never certifies in-memory weights.
        self._verify_model_checkpoint_revision: str | None = None

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
        self._proof_models: dict[str, Any] = {}
        self._default_proof_proxy_cursor = 0
        self._proof_worker_pool = proof_worker_pool
        # Directory the weights now in verify_model were read from. The
        # isolated proof workers reload from it, so it must track every
        # install: the resume at boot and each staged-checkpoint swap.
        self._verify_model_snapshot_dir: str | None = None
        self._remote_commitment_verifier = None
        if proof_worker_pool is not None:
            from reliquary.validator.proof_worker import (
                remote_commitment_verifier,
            )

            self._remote_commitment_verifier = remote_commitment_verifier(
                proof_worker_pool
            )
        self.proof_capacity_qualification = dict(
            proof_capacity_qualification or {}
        )
        self.proof_scheduler: GlobalProofScheduler | None = None
        if proof_devices:
            normalized_devices = tuple(
                str(device).strip() for device in proof_devices
            )
            if (
                any(not device for device in normalized_devices)
                or len(set(normalized_devices)) != len(normalized_devices)
            ):
                raise ValueError(
                    "proof devices must be non-empty and unique"
                )
            supplied_models = dict(proof_models or {})
            verify_device = _model_device(self.verify_model)
            for device in normalized_devices:
                candidate = supplied_models.get(device)
                if candidate is None and verify_device == device:
                    candidate = self.verify_model
                if candidate is None:
                    raise RuntimeError(
                        f"proof device {device!r} has no model replica"
                    )
                self._proof_models[device] = candidate
            self.proof_scheduler = GlobalProofScheduler(
                devices=normalized_devices,
                environments=tuple(self.envs),
                proof_callable=self._execute_scheduled_proof,
            )

        self._last_processed_window: int = -1
        self._windows_in_interval: int = 0
        # One CooldownMap per env so prompt-cooldown is independent across
        # environments (a math prompt cooling down doesn't block code prompts).
        self._cooldown_per_env: dict[str, CooldownMap] = {
            name: CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
            for name in self.envs
        }
        self._content_cooldown_per_env: dict[str, ContentCooldownMap] = {
            name: ContentCooldownMap(
                cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS
            )
            for name in self.envs
        }
        self._content_cooldown_health: dict[str, Any] = {
            "complete": False,
            "source": "not_restored",
            "snapshot_window": None,
            "counts_by_environment": {
                name: 0 for name in self.envs
            },
            "last_error_type": None,
            "last_snapshot_success_ts": None,
            "last_snapshot_failure_ts": None,
        }
        # Legacy accessor pointing to the first env's map.  Kept so
        # ``_rebuild_cooldown_from_history`` and tests that read ``_cooldown_map``
        # still work without change.
        self._cooldown_map = self._cooldown_per_env[first_env_name]
        self._hash_set = RolloutHashSet(
            retention_windows=HASH_DEDUP_RETENTION_WINDOWS,
        )
        self._late_drops: dict[str, dict[str, int]] = {}
        # Window-starts whose archive (real or tombstone) is already
        # enqueued. Per-window because pipelining keeps two windows in
        # flight: a shared boolean would let the GPU half's archive of the
        # stashed window suppress the collecting window's tombstone.
        self._archive_enqueued_windows: set[int] = set()
        self._window_iteration_stage = "startup"
        self._utility_telemetry = UtilityTelemetryWriter()
        # Detached-trainer payload worker handle; the queue itself is a
        # lazy process-wide singleton (same pattern as get_archive_queue).
        self._training_payload_worker_task: asyncio.Task | None = None
        # Detached-trainer checkpoint intake (lazy; DETACHED_TRAINER only).
        self._checkpoint_intake = None
        self._intake_stage_task: asyncio.Task | None = None
        self._windows_since_checkpoint_swap = 0

        validator_hotkey = str(wallet.hotkey.ss58_address)
        prompt_mismatch_namespace = _prompt_mismatch_circuit_namespace(
            run_id=TRAINING_RUN_ID,
            netuid=self.netuid,
            validator_hotkey=validator_hotkey,
            environments=self.envs,
        )
        self.server = ValidatorServer(
            host=http_host,
            port=http_port,
            prompt_mismatch_state_path=_prompt_mismatch_circuit_local_path(
                TRAINING_RUN_ID,
                netuid=self.netuid,
                validator_hotkey=validator_hotkey,
            ),
            prompt_mismatch_namespace=prompt_mismatch_namespace,
            no_reveal_state_path=_no_reveal_circuit_local_path(
                TRAINING_RUN_ID,
                netuid=self.netuid,
                validator_hotkey=validator_hotkey,
            ),
            no_reveal_namespace=f"no-reveal-v1:{prompt_mismatch_namespace}",
        )
        self.server.set_late_drop_callback(self.record_late_drop)
        self.server.configure_prompt_source_health(
            self._prompt_source_health_snapshot
        )
        self.server.configure_content_cooldown_health(
            self._content_cooldown_health_snapshot
        )
        self.server.configure_utility_telemetry_health(
            self._utility_telemetry.snapshot
        )
        self.server.configure_proof_scheduler_health(
            self._proof_scheduler_health_snapshot
        )

        # v2.1 state machine infrastructure — in-memory only, bootstrapped at
        # startup from R2 + HF (no local JSON state file).
        self._window_n: int = 0
        self._candidate_window_n: int | None = None
        self._candidate_activation_nonce: bytes | None = None
        self._window_preparation_stage: str | None = None
        self._checkpoint_n: int = 0
        self._publish_every = CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
        self._trained_windows_since_publish = 0
        self._training_tombstoned_windows: set[int] = set()
        self._adaptive_publication_pending = False
        self._adaptive_publication_reason: str | None = None
        self.server.set_training_publish_state(
            {
                "trained_windows_since_publish": 0,
                "publish_interval": self._publish_every,
                "publication_pending": False,
                "adaptive_publication_pending": False,
                "adaptive_publication_reason": None,
            }
        )
        accumulator_targets = dict(self.env_mix)
        if (
            EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED
            and EXPERIMENTAL_CHECKPOINT_EPOCH_TRAINING_MODE == "aggregate_one_step"
        ):
            accumulator_targets = {
                name: target * CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
                for name, target in accumulator_targets.items()
            }
        self._training_accumulator = BalancedTrainingAccumulator(accumulator_targets)
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
        self._checkpoint_epoch_plan: EpochPlan | None = None
        self._checkpoint_epoch_store: EpochStore | None = None
        if EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED:
            state_root = Path(
                os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
            )
            self._checkpoint_epoch_store = EpochStore(
                state_root / "checkpoint_epochs"
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
        # Serializes startup and quiescent-boundary registration refreshes.
        self._registration_refresh_lock = asyncio.Lock()
        self._current_window_state: WindowState = WindowState.READY

        self._resume_from = resume_from
        # The resume load must land where the boot load did. A pool means the
        # working replicas live in the workers, so this process's pair stays
        # on the CPU; no pool means it still proves in-process.
        self._load_model_fn = load_model_fn or functools.partial(
            load_validator_replica,
            isolated_plane=self._proof_worker_pool is not None,
        )

        # Fixed mode is opt-in. An explicit fixed reference is a load-bearing
        # training control, so it is immutable and fail-closed. Empty config keeps
        # the legacy rolling reference (verify_model) exactly as before.
        self.base_ref_model = None
        # Run id + exact LR schedule step read from the resumed checkpoint's
        # profile (None until _apply_resume_from runs, or when resuming
        # pre-field checkpoints — then the LR schedule warms up in full).
        self._resumed_training_run_id: str | None = None
        self._resumed_lr_schedule_step: int | None = None
        # Pipelined window collection: the sealed-window GPU work stashed for
        # the next loop iteration, and the verify-model swap deferred until
        # the last window generated under the previous checkpoint is proven.
        # (batchers, window_n, verify_task, late_drops_snapshot)
        self._gpu_backlog: tuple | None = None
        # v6 (R20). The assembler that computed a window's per-token
        # payment, keyed by window, so ``_archive_window`` can read the
        # map belonging to the window it is ARCHIVING. ``_fill_closed_
        # assembler`` alone is not enough: in pipelined mode the next
        # window's ``_open_window_batchers`` has already replaced it by
        # the time the stashed window is archived.
        self._fill_closed_assemblers: dict[int, Any] = {}
        # v6.1 (R39). The journal key of the LAST batch the last CLOSED
        # v6 window emitted; the next window's open waits for the
        # trainer's cursor to reach it. None means the gate is not armed
        # (no v6 window has closed, or the one that did emitted nothing,
        # in which case there is nothing to consume and nothing to wait
        # for). Consumed by the wait -- one close, one wait.
        self._fill_closed_rotation_key: int | None = None
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

    def _execute_scheduled_proof(
        self,
        invocation: ProofInvocation,
    ) -> ProofExecution:
        model = self._proof_models[invocation.device_id]
        payload = invocation.candidate.payload
        execute = getattr(payload, "execute", None)
        if not callable(execute):
            raise TypeError("scheduled proof payload is not executable")
        submission = execute(model)
        return ProofExecution(
            passed=submission is not None,
            value=submission,
            reason=None if submission is not None else "proof_rejected",
        )

    def _default_proof_proxy(self) -> Any:
        """Proxy used when a proof path does not name a device.

        Any configured slot serves — these paths are single-shot and the
        scheduler is not choosing between replicas for them — but always
        answering with the first would queue them behind slot 0's scheduled
        work on its pipe lock while the other slots idle, inflating slot 0's
        active-job age. Round-robin instead.
        """
        proxies = list(self._proof_models.values())
        if not proxies:
            raise RuntimeError("isolated proof plane has no device proxy")
        proxy = proxies[self._default_proof_proxy_cursor % len(proxies)]
        self._default_proof_proxy_cursor += 1
        return proxy

    def _synchronize_proof_workers(
        self,
        checkpoint_revision: str,
        snapshot_dir: str | None,
    ) -> None:
        pool = self._proof_worker_pool
        assert pool is not None
        scheduler = self.proof_scheduler
        assert scheduler is not None
        # At boot the workers hold the BOOTSTRAP replica and report no
        # revision, so this installs the resumed snapshot before any device is
        # marked ready. Never mark ready on an uninstalled revision.
        snapshot_dir = snapshot_dir or self._verify_model_snapshot_dir
        if snapshot_dir and not Path(snapshot_dir).is_dir():
            # ``CheckpointIntake.mark_installed`` rmtree's the staged dir after
            # each swap, so this path goes stale between publications.
            snapshot_dir = None
        repo_id = getattr(self._checkpoint_store, "repo_id", None)
        for device in self._proof_models:
            if pool.revision(device) != checkpoint_revision:
                if not snapshot_dir and not repo_id:
                    raise RuntimeError(
                        f"isolated proof worker {device!r} holds "
                        f"{pool.revision(device)!r} and no source is "
                        f"available for {checkpoint_revision!r}"
                    )
                logger.info(
                    "Reloading isolated proof worker %s to %s (source=%s)",
                    device, checkpoint_revision[:12],
                    snapshot_dir or f"hub:{repo_id}",
                )
                pool.reload(
                    device, snapshot_dir, checkpoint_revision, repo_id,
                )
            scheduler.mark_device_ready(device, checkpoint_revision)

    def _proof_scheduler_health_snapshot(self) -> dict[str, Any]:
        scheduler = getattr(self, "proof_scheduler", None)
        if scheduler is None:
            return {
                "state": "disabled",
                "required": PROTOCOL_VERSION >= 3,
                "profile_id": PROTOCOL_PROFILE_ID,
                "configured_devices": [],
                "degraded_reasons": (
                    ["required_scheduler_missing"]
                    if PROTOCOL_VERSION >= 3
                    else []
                ),
            }
        snapshot = scheduler.snapshot()
        degraded_reasons: list[str] = []
        state = snapshot.get("state")
        if PROTOCOL_VERSION >= 3 and state != SchedulerState.RUNNING.value:
            degraded_reasons.append("scheduler_not_running")
        active_revision = snapshot.get("active_checkpoint_revision")
        device_revisions = snapshot.get("device_revisions", {})
        if active_revision and any(
            revision != active_revision
            for revision in device_revisions.values()
        ):
            degraded_reasons.append("checkpoint_replica_mismatch")
        active_ages = [
            float(active["age_seconds"])
            for active in snapshot.get("active_by_device", {}).values()
            if active is not None and active.get("age_seconds") is not None
        ]
        if active_ages and max(active_ages) > MAX_PROOF_WALL_SECONDS:
            degraded_reasons.append("active_proof_over_wall")
        abort_age = snapshot.get("last_capacity_abort_age_seconds")
        if (
            abort_age is not None
            and float(abort_age) <= MAX_PROOF_WALL_SECONDS * 2.0
        ):
            degraded_reasons.append("recent_capacity_abort")
        if (
            PROTOCOL_VERSION >= 3
            and getattr(self, "proof_capacity_qualification", {}).get(
                "qualified"
            )
            is not True
        ):
            degraded_reasons.append("capacity_not_qualified")
        published_revision: str | None = None
        checkpoint_store = getattr(self, "_checkpoint_store", None)
        if checkpoint_store is not None:
            try:
                manifest = checkpoint_store.current_manifest()
            except Exception:
                degraded_reasons.append("checkpoint_manifest_unavailable")
            else:
                if manifest is not None:
                    published_revision = manifest.revision
        verify_revision = getattr(
            self,
            "_verify_model_checkpoint_revision",
            None,
        )
        if published_revision and verify_revision != published_revision:
            degraded_reasons.append("verify_checkpoint_mismatch")
        if published_revision and active_revision != published_revision:
            degraded_reasons.append("scheduler_checkpoint_mismatch")
        snapshot.update({
            "required": PROTOCOL_VERSION >= 3,
            "profile_id": PROTOCOL_PROFILE_ID,
            "configured_devices": list(
                getattr(self, "_proof_models", {})
            ),
            "capacity_qualification": dict(
                getattr(self, "proof_capacity_qualification", {})
            ),
            "published_checkpoint_revision": published_revision,
            "verify_checkpoint_revision": verify_revision,
            "degraded_reasons": degraded_reasons,
        })
        return snapshot

    def _refresh_verify_model_from_train(
        self,
        checkpoint_revision: str,
    ) -> None:
        """Install one complete frozen verifier snapshot and then label it."""

        if not checkpoint_revision:
            raise RuntimeError(
                "verify model refresh requires a checkpoint revision"
            )
        self._verify_model_checkpoint_revision = None
        self.verify_model.load_state_dict(self.train_model.state_dict())
        self.verify_model.eval()
        for parameter in self.verify_model.parameters():
            parameter.requires_grad = False
        self._verify_model_checkpoint_revision = checkpoint_revision

    def _detached_intake_ref(self):
        """Lazy CheckpointIntake (DETACHED_TRAINER mode only)."""
        if self._checkpoint_intake is None:
            import os as _os

            from reliquary.validator.checkpoint_intake import (
                CheckpointIntake,
                default_r2_client,
            )
            from reliquary.shared.training_payload import (
                active_training_identity,
            )

            current = self._checkpoint_store.current_manifest()
            self._checkpoint_intake = CheckpointIntake(
                r2_client=default_r2_client(),
                bucket=_os.getenv("R2_BUCKET_ID", "reliquary"),
                staging_dir=_os.path.join(
                    _os.environ.get(
                        "RELIQUARY_STATE_DIR", "/root/reliquary/state",
                    ),
                    "checkpoint_intake",
                ),
                installed_revision=(
                    current.revision if current is not None else None
                ),
                expected_identity=(
                    active_training_identity()
                    if PROTOCOL_VERSION >= 5
                    else None
                ),
            )
        return self._checkpoint_intake

    def _refresh_verify_model_from_dir(
        self, snapshot_dir, checkpoint_revision: str,
    ) -> None:
        """Install a trainer-published snapshot into the verify plane.

        The full state dict is assembled on CPU BEFORE any device copy so
        a read failure cannot leave verify_model half-updated."""
        if not checkpoint_revision:
            raise RuntimeError(
                "verify model refresh requires a checkpoint revision"
            )
        from pathlib import Path as _Path

        from safetensors.torch import load_file

        state: dict = {}
        for path in sorted(_Path(snapshot_dir).glob("*.safetensors")):
            state.update(load_file(str(path), device="cpu"))
        if not state:
            raise RuntimeError(f"no safetensors under {snapshot_dir}")
        self._verify_model_checkpoint_revision = None
        # save_pretrained(safe_serialization=True) omits tied weights
        # (lm_head.weight on every protocol Qwen model), so a strict load
        # raises. Allow EXACTLY the model's declared tied keys, nothing
        # else, then re-tie.
        tied = set(
            getattr(self.verify_model, "_tied_weights_keys", None) or []
        )
        result = self.verify_model.load_state_dict(state, strict=False)
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        missing = [
            k for k in getattr(result, "missing_keys", []) or []
            if k not in tied
        ]
        if unexpected or missing:
            raise RuntimeError(
                "staged checkpoint state mismatch: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        if tied:
            self.verify_model.tie_weights()
        self._verify_model_snapshot_dir = str(snapshot_dir)
        self.verify_model.eval()
        for parameter in self.verify_model.parameters():
            parameter.requires_grad = False
        self._verify_model_checkpoint_revision = checkpoint_revision

    async def _swap_staged_checkpoint(self, window_n: int) -> None:
        """Serial-beat swap: verify plane first, manifest install LAST.

        On any failure the current manifest still names the old revision,
        the staged copy is dropped, and the next poll re-stages the same
        candidate — degradation is staleness, never a wrong-weights proof.
        """
        import shutil as _shutil

        intake = self._checkpoint_intake
        manifest, staged_dir = intake.take_staged()
        revision = str(manifest["revision"])
        try:
            await asyncio.to_thread(
                self._refresh_verify_model_from_dir, staged_dir, revision,
            )
            if self.proof_scheduler is not None:
                await asyncio.to_thread(
                    self._synchronize_proof_models, revision, str(staged_dir),
                )
            entry = self._checkpoint_store.install_external(
                int(manifest["checkpoint_n"]), revision,
            )
            self._checkpoint_n = int(manifest["checkpoint_n"])
            self.server.set_current_checkpoint(entry)
            intake.mark_installed(revision, staged_dir)
            self._windows_since_checkpoint_swap = 0
            # Retained groups were generated against the parent revision;
            # parity with the in-process post-publish discard.
            self._training_accumulator.reset()
            logger.info(
                "Window %d: swapped verify plane to trainer checkpoint "
                "%d (%s)", window_n, entry.checkpoint_n, revision[:12],
            )
        except FatalProofPlaneError:
            raise
        except Exception:
            logger.exception(
                "staged checkpoint swap failed for %s; staying on the "
                "current revision", revision[:12],
            )
            _shutil.rmtree(staged_dir, ignore_errors=True)

    async def _detached_checkpoint_tick(
        self, *, owns_routing: bool, window_n: int,
    ) -> None:
        """Per-window intake driver (DETACHED_TRAINER mode): poll the
        candidate manifest, stage in the background, swap on the serial
        beat only."""
        intake = self._detached_intake_ref()
        self._windows_since_checkpoint_swap += 1
        # The emergency freeze must also stop ADOPTING checkpoints, or a
        # collapsing model keeps rolling out to miners mid-incident.
        if os.environ.get("RELIQUARY_DISABLE_TRAIN", "").lower() in {
            "1", "true", "yes", "on",
        }:
            logger.warning(
                "Window %d: emergency freeze active; not polling or "
                "swapping trainer checkpoints", window_n,
            )
            return
        await self._poll_and_stage_checkpoint_candidate(intake)
        if intake.staged_ready:
            if owns_routing:
                await self._swap_staged_checkpoint(window_n)
            else:
                logger.info(
                    "Window %d: staged checkpoint ready but this half is "
                    "pipelined; deferring swap to the serial beat",
                    window_n,
                )

    async def _poll_and_stage_checkpoint_candidate(self, intake) -> bool:
        """Ask R2 for a new candidate manifest; start staging it if there
        is one. Returns whether a NEW candidate was DETECTED.

        Extracted from ``_detached_checkpoint_tick`` because v6.1's
        between-windows gate (R35) needs the same poll-and-stage without
        the swap: staging is a multi-gigabyte download and must overlap
        the next window's collection, so detection -- not installation --
        is what releases the gate. Never starts a second staging task
        while one is in flight, exactly as the tick did.
        """
        task = self._intake_stage_task
        if task is not None and not task.done():
            return False
        manifest = await asyncio.to_thread(intake.poll)
        if manifest is None:
            return False
        self._intake_stage_task = asyncio.create_task(
            asyncio.to_thread(intake.stage, manifest),
            name="checkpoint_intake_stage",
        )
        return True

    def _arm_fill_closed_rotation_gate(self) -> None:
        """R39: remember the journal key of the LAST batch this window
        emitted, so the next window opens only once the trainer has
        CONSUMED it.

        Supersedes the revision-baseline gate this method used to arm.
        Waiting for a PUBLICATION was structurally wrong: the trainer
        publishes at 16 TRAINED batches CUMULATIVE, not per window, so an
        underfilled window armed a wait for a checkpoint that would never
        come (a full backstop of dead air), and a publication landing
        mid-window over-armed the next one. Consumption has neither
        failure mode -- a window waits for exactly what it put in the
        journal, and nothing else. Checkpoint adoption is untouched and
        stays on ``_detached_checkpoint_tick``'s serial beat; if the
        consumption being waited on happened to cross the publish
        boundary, the staged revision is adopted there as usual.

        ``next_batch_index`` is the assembler's count of real emissions
        and has not yet been padded by ``close()`` at this point in the
        loop, so key ``next_batch_index - 1`` is the last batch this
        window actually emitted. Zero emitted disarms outright: there is
        nothing to consume, so there is nothing to wait for.
        """
        from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW
        from reliquary.infrastructure.training_payload_queue import (
            encoded_window_journal_key,
        )

        assembler = getattr(self, "_fill_closed_assembler", None)
        emitted = int(getattr(assembler, "next_batch_index", 0) or 0)
        window_start = getattr(assembler, "window_start", None)
        if emitted <= 0 or window_start is None:
            # No emissions, or no assembler to say which window they were
            # filed under: disarm rather than guess a key, since a wrong
            # key gates the next window on something nobody will consume.
            self._fill_closed_rotation_key = None
            return
        # Clamped rather than trusted: the encoder raises outside the
        # window's reserved range, and this runs on the window loop where
        # that would cost the whole iteration for a telemetry value.
        index = min(emitted - 1, FILL_CLOSED_EMISSIONS_PER_WINDOW - 1)
        self._fill_closed_rotation_key = encoded_window_journal_key(
            int(window_start), index
        )

    async def _wait_for_fill_closed_rotation(self) -> str:
        """R39: hold the next v6 window's construction until the trainer
        has consumed every batch the closed window emitted.

        The amendment closes a window at its 16th pick and opens the next
        one on a synchronisation point, so that a window is never
        collected against a policy the trainer has already moved past and
        intra-window pacing drift resets between windows. The point is
        the trainer's own cursor reaching this window's last batch --
        measured consumption, never a declared interval and never a
        publication (see ``_arm_fill_closed_rotation_gate`` for why the
        publication reading was wrong).

        The arming is CONSUMED here: one close, one wait. A second call
        finds nothing armed and returns immediately, so a stale key can
        never gate a later window.

        Skipped entirely under ``RELIQUARY_DISABLE_TRAIN``: the freeze
        stops the trainer consuming anything, so the gate would stall the
        validator for a backstop per window during the very incident the
        freeze exists to contain.

        Bounded by ``FILL_CLOSED_MAX_SECONDS``, the same backstop that
        bounds the window itself -- a dead trainer consumes nothing ever,
        and an unbounded wait would take the validator off the air rather
        than merely stop its learning. Same discipline as
        ``_wait_for_window_seal``: a bounded async wait with an explicit
        poll interval, never a busy loop, and a preparation stage on
        ``/state`` throughout so a held rotation is legible from outside.
        """
        if not FILL_CLOSED_ENABLED:
            return "disabled"
        required = getattr(self, "_fill_closed_rotation_key", None)
        if required is None:
            return "not_armed"
        self._fill_closed_rotation_key = None
        if os.environ.get("RELIQUARY_DISABLE_TRAIN", "").lower() in {
            "1", "true", "yes", "on",
        }:
            logger.warning(
                "Window %d: emergency freeze active; not holding the next "
                "window's open on trainer consumption (nothing will ever "
                "consume journal key %s under the freeze)",
                self._window_n, required,
            )
            return "emergency_freeze"

        def consumed() -> bool:
            cursor = self._read_trainer_step_cursor()
            return cursor is not None and int(cursor) >= int(required)

        if consumed():
            return "batches_consumed"
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + FILL_CLOSED_MAX_SECONDS
        self._set_window_preparation_stage("fill_closed_rotation_wait")
        logger.info(
            "Window %d closed; holding the next window's open until the "
            "trainer has consumed its last batch (journal key %s, backstop "
            "%.0f s). Opening now would collect against a policy the "
            "trainer has already moved past.",
            self._window_n, required, FILL_CLOSED_MAX_SECONDS,
        )
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.error(
                    "Trainer has not consumed window %d's last batch "
                    "(journal key %s) %.0f s after it closed; opening the "
                    "next window anyway. The trainer is stalled or dead -- "
                    "its cursor will not move for that window either, so it "
                    "will take its first %d picks on the time floor and then "
                    "end on its own backstop.",
                    self._window_n, required, FILL_CLOSED_MAX_SECONDS,
                    FILL_CLOSED_PICK_PIPELINE_DEPTH,
                )
                return "rotation_wait_timeout"
            await asyncio.sleep(
                min(FILL_CLOSED_ROTATION_POLL_SECONDS, remaining)
            )
            if consumed():
                logger.info(
                    "Window %d's batches consumed in %.1f s; opening the "
                    "next window",
                    self._window_n, loop.time() - started,
                )
                return "batches_consumed"

    async def _seal_wait_and_close(
        self,
        *,
        early_close_ready: Callable[[], bool] | None = None,
    ) -> str:
        """Seal-wait as a concurrent task during a pipelined GPU half.

        poll_deadline() is only driven by _wait_for_window_seal; if the loop
        only reached it AFTER the stashed GPU half (~2 GPU-half durations
        into a 100 s collection), the collecting window would seal late and
        /state would show OPEN for the entire cycle — out-of-phase miners
        would then never see an OPEN -> not-OPEN edge and never re-sync.
        Running it concurrently seals on the ceiling and flips the FSM to READY
        at that moment. The callback also lets adaptive close observe the exact
        instant the previous window's GPU half finishes, without a shared flag
        that could leak across iterations.
        """
        if early_close_ready is None:
            reason = await self._wait_for_window_seal()
        else:
            reason = await self._wait_for_window_seal(
                early_close_ready=early_close_ready
            )
        self._set_state(WindowState.READY)
        return reason

    def _publication_due_next_half(self) -> bool:
        """Forecast, at seal time, whether this window's GPU half may publish.

        SERIAL BEAT: a publishing half must run serially (no window
        collecting), because publication swaps the verify plane and changes
        the pinned checkpoint mid-collection otherwise. The forecast
        over-approximates ``should_publish`` (assumes the train step will
        count); a miss in the other direction (adaptive drift raised DURING
        the half) is caught by the in-half deferral, which pushes the
        publication to the next — serial — window. Cost of a false positive:
        one serial window (~100 s), at most once per publish interval.
        """
        from reliquary.constants import DETACHED_TRAINER as _detached

        if _detached:
            # Detached mode: the serial beat exists for the SWAP, which
            # is due exactly when a staged candidate is ready.
            intake = self._checkpoint_intake
            return bool(intake is not None and intake.staged_ready)
        emergency_freeze = os.environ.get(
            "RELIQUARY_DISABLE_TRAIN", ""
        ).lower() in {"1", "true", "yes", "on"}
        ceiling_reached = (
            TRAIN_UNTIL_CHECKPOINT_N > 0
            and self._checkpoint_n >= TRAIN_UNTIL_CHECKPOINT_N
        )
        if emergency_freeze or ceiling_reached:
            # should_publish is permanently False in these regimes; without
            # this gate a counter frozen at publish_every-1 would pin the
            # loop to the serial path forever.
            return False
        try:
            bootstrap = self._checkpoint_store.current_manifest() is None
        except Exception:
            bootstrap = False
        return (
            self._trained_windows_since_publish + 1 >= self._publish_every
            or self._adaptive_publication_pending
            or bootstrap
        )

    def _synchronize_proof_models(
        self,
        checkpoint_revision: str,
        snapshot_dir: str | None = None,
    ) -> None:
        """Quiesce, refresh every replica, then atomically resume proving.

        With an isolated proof plane the validator no longer holds the
        replicas, so the refresh is a reload inside each worker. It needs the
        staged snapshot directory: a worker already certified for this
        revision is simply marked ready, and one that is not — a replacement
        spawned after a crash, or a publication we have no snapshot for —
        must fail loudly rather than keep proving on unknown weights.
        """

        scheduler = self.proof_scheduler
        if scheduler is None:
            return
        if not checkpoint_revision:
            raise RuntimeError(
                "proof scheduler requires a published checkpoint revision"
            )
        if self._verify_model_checkpoint_revision != checkpoint_revision:
            raise RuntimeError(
                "verify model weights are not certified for the checkpoint"
            )
        if (
            scheduler.state is SchedulerState.RUNNING
            and scheduler.checkpoint_ready(checkpoint_revision)
        ):
            return
        if scheduler.state is SchedulerState.FAULTED:
            raise FatalProofPlaneError(
                "proof scheduler is faulted and requires process restart"
            )
        if scheduler.state is SchedulerState.RUNNING:
            if not scheduler.drain(
                timeout=MAX_PROOF_WALL_SECONDS + 60.0
            ):
                raise RuntimeError(
                    "proof scheduler did not drain before checkpoint refresh"
                )
        elif scheduler.state in {
            SchedulerState.DRAINING,
            SchedulerState.QUIESCING,
        }:
            if not scheduler.quiesce(
                timeout=MAX_PROOF_WALL_SECONDS + 60.0
            ):
                raise RuntimeError(
                    "proof scheduler did not quiesce before checkpoint refresh"
                )
        if scheduler.state is not SchedulerState.QUIESCED:
            raise RuntimeError(
                "proof scheduler is not quiesced for checkpoint refresh"
            )

        for device in self._proof_models:
            scheduler.mark_device_not_ready(device)
        if self._proof_worker_pool is not None:
            self._synchronize_proof_workers(
                checkpoint_revision, snapshot_dir,
            )
            scheduler.resume(checkpoint_revision)
            return
        reference_state = self.verify_model.state_dict()
        for device, model in self._proof_models.items():
            if model is not self.verify_model:
                model.load_state_dict(reference_state)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            actual_device = _model_device(model)
            if actual_device is not None and actual_device != device:
                raise RuntimeError(
                    f"proof replica {device!r} is on {actual_device!r}"
                )
            scheduler.mark_device_ready(device, checkpoint_revision)
        scheduler.resume(checkpoint_revision)

    async def _ensure_proof_scheduler_ready(self) -> None:
        """Recover replicas at a quiescent boundary before opening a window."""

        scheduler = self.proof_scheduler
        if scheduler is None:
            return
        if scheduler.state is SchedulerState.FAULTED:
            raise FatalProofPlaneError(
                "proof scheduler is faulted and requires process restart"
            )
        checkpoint = self._checkpoint_store.current_manifest()
        if checkpoint is None or not checkpoint.revision:
            raise RuntimeError(
                "scheduled proving requires a published checkpoint"
            )
        if self._verify_model_checkpoint_revision != checkpoint.revision:
            from reliquary.constants import DETACHED_TRAINER as _detached

            if _detached:
                # Detached mode: train_model no longer tracks the
                # published checkpoint, so a refresh-from-train would
                # label STALE weights with the new revision and every
                # proof would run against the wrong model. Restart and
                # reload from the published revision instead.
                raise FatalProofPlaneError(
                    "verify model revision diverged under detached "
                    "trainer; restart required to reload the published "
                    "checkpoint"
                )
            if getattr(self, "_gpu_backlog", None) is not None:
                # Should be unreachable: publishes are serial-only, so the
                # revisions can only diverge in an iteration with no backlog.
                # If it fires anyway, the stashed window's proofs would run
                # against the wrong weights — loud is better than silent.
                logger.error(
                    "verify model revision diverged (%s != %s) while a "
                    "pipelined backlog is stashed; refreshing anyway — "
                    "stashed window %s proofs may be mis-verified",
                    self._verify_model_checkpoint_revision,
                    checkpoint.revision,
                    self._gpu_backlog[1],
                )
            await asyncio.to_thread(
                self._refresh_verify_model_from_train,
                checkpoint.revision,
            )
        if (
            scheduler.state is SchedulerState.RUNNING
            and scheduler.checkpoint_ready(checkpoint.revision)
        ):
            return
        await asyncio.to_thread(
            self._synchronize_proof_models,
            checkpoint.revision,
        )
        if not (
            scheduler.state is SchedulerState.RUNNING
            and scheduler.checkpoint_ready(checkpoint.revision)
        ):
            raise RuntimeError(
                "proof scheduler replica recovery did not become ready"
            )

    async def _close_proof_scheduler(self) -> None:
        scheduler = self.proof_scheduler
        if scheduler is None:
            return
        timeout = (
            5.0
            if scheduler.state is SchedulerState.FAULTED
            else MAX_PROOF_WALL_SECONDS + 60.0
        )
        closed = await asyncio.to_thread(scheduler.close, timeout)
        if not closed:
            logger.error(
                "Proof scheduler did not close within %.1fs; process exit "
                "will retire daemon proof workers",
                timeout,
            )
        # Retire the isolated workers. A polite shutdown writes into the very
        # pipe a device thread may still be reading, so when the scheduler
        # did NOT close (a worker thread is still in flight) kill instead.
        pool = self._proof_worker_pool
        if pool is not None:
            try:
                await asyncio.to_thread(pool.close, not closed)
            except Exception:
                logger.exception("isolated proof workers did not close cleanly")

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

    def _content_cooldown_health_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._content_cooldown_health)
        snapshot["counts_by_environment"] = {
            name: len(content_map)
            for name, content_map in self._content_cooldown_per_env.items()
        }
        return snapshot

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
        from reliquary.validator.checkpoint_profile import (
            validate_checkpoint_profile,
        )

        # Historical auction-v2 checkpoints predate lineage metadata and remain
        # loadable. Auction-v3 must never silently resume those 2B weights.
        resumed_profile = validate_checkpoint_profile(
            local_path,
            required=PROTOCOL_VERSION >= 3,
        )
        # Run identity + exact LR schedule position of the resumed
        # checkpoint. Both None on checkpoints published before the fields
        # existed — in that case the LR schedule falls back to the full
        # warmup (fail-closed), and the guard hardens at the next publish.
        self._resumed_training_run_id = (resumed_profile or {}).get(
            "training_run_id"
        )
        self._resumed_lr_schedule_step = _coerce_lr_schedule_step(
            (resumed_profile or {}).get("lr_schedule_step")
        )
        # Load weights — this replaces both models loaded at __init__.
        # verify_model gets the resumed weights too (so the batcher
        # verifies miners against the resumed checkpoint, which is what
        # they have access to via HF).
        self.train_model = self._load_model_fn(local_path)
        self._verify_model_snapshot_dir = str(local_path)
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
        self._verify_model_checkpoint_revision = revision_str
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

    @staticmethod
    def _checkpoint_epoch_protocol_binding() -> ProtocolBinding:
        return ProtocolBinding(
            profile_id=PROTOCOL_PROFILE_ID,
            protocol_version=PROTOCOL_VERSION,
            generation_contract_sha256=generation_contract_sha256(
                PROTOCOL_GENERATION_CONTRACT
            ),
        )

    @staticmethod
    def _checkpoint_epoch_matches(plan: EpochPlan, checkpoint) -> bool:
        return bool(
            checkpoint is not None
            and plan.checkpoint.number == int(checkpoint.checkpoint_n)
            and plan.checkpoint.repo_id == str(checkpoint.repo_id)
            and plan.checkpoint.revision == str(checkpoint.revision)
        )

    def _validate_checkpoint_epoch_runtime_config(self, plan: EpochPlan) -> None:
        expected_schedule = WindowSchedule(
            mode=CHECKPOINT_EPOCH_SCHEDULE_MODE,
            collection_seconds=(
                EXPERIMENTAL_CHECKPOINT_EPOCH_COLLECTION_SECONDS
            ),
            timeout_seconds=WINDOW_TIMEOUT_SECONDS,
        )
        expected_universes = {
            name: len(environment)
            for name, environment in self.envs.items()
        }
        plan_universes = {
            item.environment: item.universe_size
            for item in plan.windows[0].prompt_slices
        }
        if (
            plan.protocol != self._checkpoint_epoch_protocol_binding()
            or plan.window_count != CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
            or plan.window_schedule != expected_schedule
            or plan.training_mode
            != EXPERIMENTAL_CHECKPOINT_EPOCH_TRAINING_MODE
            or plan.warmup_rounds
            != EXPERIMENTAL_CHECKPOINT_EPOCH_WARMUP_ROUNDS
            or plan.prompt_range_size != PROMPT_RANGE_SIZE
            or plan.target_groups_per_environment_lane != B_BATCH
            or plan.candidate_limit_per_environment_lane
            != EXPERIMENTAL_CHECKPOINT_EPOCH_CANDIDATES_PER_LANE
            or plan.commitments_per_operator_per_environment_lane
            != EXPERIMENTAL_CHECKPOINT_EPOCH_COMMITMENTS_PER_OPERATOR_PER_LANE
            or plan.reveal_seconds
            != EXPERIMENTAL_CHECKPOINT_EPOCH_REVEAL_SECONDS
            or plan.candidate_limit_per_environment_lane
            > MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW
            or plan_universes != expected_universes
        ):
            raise RuntimeError("checkpoint epoch runtime configuration changed")

    def _checkpoint_epoch_window(
        self,
        window_number: int,
    ) -> EpochWindow | None:
        plan = getattr(self, "_checkpoint_epoch_plan", None)
        if plan is None:
            return None
        offset = int(window_number) - plan.first_window
        if offset < 0 or offset >= plan.window_count:
            return None
        window = plan.windows[offset]
        if window.offset != offset or window.window_number != window_number:
            raise RuntimeError("checkpoint epoch window mapping is invalid")
        return window

    async def _checkpoint_epoch_drand_snapshot(self) -> tuple[dict, int]:
        from reliquary.infrastructure.drand import get_current_chain

        chain_info = await asyncio.to_thread(get_current_chain)
        current_round = chain.compute_current_drand_round(
            time.time(),
            chain_info["genesis_time"],
            chain_info["period"],
        )
        return chain_info, int(current_round)

    async def _verify_checkpoint_epoch_beacon(
        self,
        beacon: BeaconBinding,
    ) -> None:
        from reliquary.infrastructure.drand import verify_beacon_signature

        from reliquary.infrastructure.drand import get_beacon

        fetched = await asyncio.to_thread(
            get_beacon,
            round_id=str(beacon.round),
            use_drand=True,
            use_fallback=False,
        )
        signature = fetched.get("signature")
        if (
            str(fetched.get("source")) != beacon.source
            or str(fetched.get("chain")) != beacon.chain
            or str(fetched.get("chain_hash")) != beacon.chain_hash
            or int(fetched.get("round", -1)) != beacon.round
            or str(fetched.get("randomness")) != beacon.randomness
            or not signature
        ):
            raise RuntimeError("checkpoint epoch beacon does not match drand")
        verified = await asyncio.to_thread(
            verify_beacon_signature,
            beacon.chain_hash,
            beacon.round,
            beacon.randomness,
            str(signature),
        )
        if verified is not True:
            raise RuntimeError("checkpoint epoch beacon verification failed")

    async def _checkpoint_epoch_intent(
        self,
        checkpoint,
        *,
        next_window: int,
    ) -> EpochCommitIntent:
        store = self._checkpoint_epoch_store
        if store is None:
            raise RuntimeError("checkpoint epoch store is unavailable")
        protocol = self._checkpoint_epoch_protocol_binding()
        existing = store.load_current_intent()
        if (
            existing is not None
            and existing.protocol == protocol
            and existing.checkpoint.number == int(checkpoint.checkpoint_n)
            and existing.checkpoint.repo_id == str(checkpoint.repo_id)
            and existing.checkpoint.revision == str(checkpoint.revision)
            and existing.first_window
            <= next_window
            < existing.first_window + existing.window_count
            and store.is_confirmed(existing)
        ):
            publication = store.load_signed_intent(existing)
            if publication is None:
                raise RuntimeError("confirmed checkpoint epoch intent is unsigned")
            self.server.set_checkpoint_epoch_signed_intent(
                canonical_signed_intent_bytes(publication),
                intent_sha256=publication.intent_sha256,
            )
            return existing

        for _attempt in range(4):
            chain_info, observed_round = await self._checkpoint_epoch_drand_snapshot()
            intent = build_epoch_intent(
                protocol=protocol,
                checkpoint_number=int(checkpoint.checkpoint_n),
                checkpoint_repo_id=str(checkpoint.repo_id),
                checkpoint_revision=str(checkpoint.revision),
                commit_observed_round=observed_round,
                first_window=next_window,
                window_count=CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
                beacon_chain=str(chain_info["name"]),
                beacon_chain_hash=str(chain_info["hash"]),
                warmup_rounds=EXPERIMENTAL_CHECKPOINT_EPOCH_WARMUP_ROUNDS,
                window_schedule=WindowSchedule(
                    mode=CHECKPOINT_EPOCH_SCHEDULE_MODE,
                    collection_seconds=(
                        EXPERIMENTAL_CHECKPOINT_EPOCH_COLLECTION_SECONDS
                    ),
                    timeout_seconds=WINDOW_TIMEOUT_SECONDS,
                ),
                training_mode=EXPERIMENTAL_CHECKPOINT_EPOCH_TRAINING_MODE,
                prompt_range_size=PROMPT_RANGE_SIZE,
                target_groups_per_environment_lane=B_BATCH,
                candidate_limit_per_environment_lane=(
                    EXPERIMENTAL_CHECKPOINT_EPOCH_CANDIDATES_PER_LANE
                ),
                commitments_per_operator_per_environment_lane=(
                    EXPERIMENTAL_CHECKPOINT_EPOCH_COMMITMENTS_PER_OPERATOR_PER_LANE
                ),
                reveal_seconds=EXPERIMENTAL_CHECKPOINT_EPOCH_REVEAL_SECONDS,
                environment_universes={
                    name: len(environment) for name, environment in self.envs.items()
                },
            )
            store.install_intent(intent)
            publication = SignedEpochIntent(
                intent=intent,
                intent_sha256=intent.intent_id,
                validator_hotkey=str(self.wallet.hotkey.ss58_address),
                validator_signature=self.wallet.hotkey.sign(
                    intent_signing_bytes(intent)
                ).hex(),
            )
            signed_raw = store.install_signed_intent(publication)
            _, confirmed_round = await self._checkpoint_epoch_drand_snapshot()
            if confirmed_round < intent.beacon_target_round:
                store.confirm_before_beacon(
                    intent,
                    observed_round=confirmed_round,
                )
                self.server.set_checkpoint_epoch_signed_intent(
                    signed_raw,
                    intent_sha256=publication.intent_sha256,
                )
                return intent
        raise RuntimeError(
            "could not persist checkpoint epoch intent before its beacon"
        )

    async def _plan_from_checkpoint_epoch_intent(
        self,
        intent: EpochCommitIntent,
    ) -> EpochPlan:
        store = self._checkpoint_epoch_store
        if store is None or not store.is_confirmed(intent):
            raise RuntimeError("checkpoint epoch intent is not confirmed")

        while True:
            _, current_round = await self._checkpoint_epoch_drand_snapshot()
            if current_round >= intent.beacon_target_round:
                break
            await asyncio.sleep(0.25)

        from reliquary.infrastructure.drand import get_beacon

        fetched = await asyncio.to_thread(
            get_beacon,
            round_id=str(intent.beacon_target_round),
            use_drand=True,
            use_fallback=False,
        )
        beacon = BeaconBinding(
            source=str(fetched["source"]),
            chain=str(fetched["chain"]),
            chain_hash=str(fetched["chain_hash"]),
            round=int(fetched["round"]),
            randomness=str(fetched["randomness"]),
        )
        await self._verify_checkpoint_epoch_beacon(beacon)
        plan = plan_from_intent(intent, beacon=beacon)
        store.install_plan(intent, plan)
        return plan

    async def _ensure_checkpoint_epoch_plan(self) -> EpochPlan | None:
        if not EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED:
            return None
        if (
            CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
            != CHECKPOINT_EPOCH_REQUIRED_WINDOW_COUNT
        ):
            raise RuntimeError(
                "concurrent checkpoint epochs require a configured horizon "
                f"of {CHECKPOINT_EPOCH_REQUIRED_WINDOW_COUNT} windows"
            )
        if not self.use_drand:
            raise RuntimeError("checkpoint epoch requires verified drand")
        checkpoint = self._checkpoint_store.current_manifest()
        if checkpoint is None:
            self._checkpoint_epoch_plan = None
            self.server.set_checkpoint_epoch_plan(None)
            return None

        next_window = (
            self._candidate_window_n
            if self._candidate_window_n is not None
            else self._window_n + 1
        )
        active = self._checkpoint_epoch_plan
        if active is not None and not self._checkpoint_epoch_matches(
            active, checkpoint
        ):
            self._checkpoint_epoch_plan = None
            self.server.set_checkpoint_epoch_plan(None)
            active = None
        if active is not None:
            self._validate_checkpoint_epoch_runtime_config(active)
            if self._checkpoint_epoch_window(next_window) is not None:
                return active
            raise CheckpointEpochExecutionError(
                "checkpoint epoch ended without a successor checkpoint"
            )

        store = self._checkpoint_epoch_store
        if store is None:
            raise RuntimeError("checkpoint epoch store is unavailable")
        restored = store.load_current_plan()
        if (
            restored is not None
            and self._checkpoint_epoch_matches(restored, checkpoint)
            and store.is_activated(restored)
        ):
            terminal_status = store.terminal_status(restored)
            if terminal_status is None:
                terminal_status = "aborted"
                store.mark_terminal(restored, status=terminal_status)
            if terminal_status == "aborted":
                self._abort_training_epoch_journal(
                    restored,
                    failure_stage="checkpoint_epoch_restart",
                    failure_type="InterruptedCheckpointEpoch",
                )
            else:
                self._write_training_epoch_marker(
                    restored,
                    status="completed",
                )
            self._window_n = max(
                self._window_n,
                restored.first_window + restored.window_count - 1,
            )
            self._candidate_window_n = None
            next_window = self._window_n + 1
            logger.warning(
                "Retired previously activated checkpoint epoch %s status=%s",
                restored.epoch_id[:12],
                terminal_status,
            )
            self._checkpoint_epoch_plan = None
            self.server.set_checkpoint_epoch_plan(None)
            raise CheckpointEpochExecutionError(
                "activated checkpoint epoch requires a successor checkpoint"
            )
        if (
            restored is not None
            and self._checkpoint_epoch_matches(restored, checkpoint)
            and restored.first_window
            <= next_window
            < restored.first_window + restored.window_count
        ):
            self._validate_checkpoint_epoch_runtime_config(restored)
            await self._verify_checkpoint_epoch_beacon(restored.epoch_beacon)
            restored_intent = store.load_current_intent()
            if (
                restored_intent is None
                or not store.is_confirmed(restored_intent)
                or plan_from_intent(
                    restored_intent,
                    beacon=restored.epoch_beacon,
                )
                != restored
            ):
                raise RuntimeError("restored epoch plan lacks its confirmed intent")
            restored_publication = store.load_signed_intent(restored_intent)
            if restored_publication is None:
                raise RuntimeError("restored epoch plan lacks its signed intent")
            self.server.set_checkpoint_epoch_signed_intent(
                canonical_signed_intent_bytes(restored_publication),
                intent_sha256=restored_publication.intent_sha256,
            )
            plan = restored
        else:
            intent = await self._checkpoint_epoch_intent(
                checkpoint,
                next_window=next_window,
            )
            plan = await self._plan_from_checkpoint_epoch_intent(intent)
            latest = self._checkpoint_store.current_manifest()
            if not self._checkpoint_epoch_matches(plan, latest):
                raise RuntimeError(
                    "checkpoint changed while constructing epoch plan"
                )

        self._checkpoint_epoch_plan = plan
        self.server.set_checkpoint_epoch_plan(plan)
        logger.info(
            "Checkpoint epoch active id=%s manifest=%s windows=%d..%d",
            plan.epoch_id[:12],
            manifest_sha256(plan)[:12],
            plan.first_window,
            plan.first_window + plan.window_count - 1,
        )
        return plan

    async def _wait_for_checkpoint_epoch_activation(self) -> None:
        plan = self._checkpoint_epoch_plan
        next_window = (
            self._candidate_window_n
            if self._candidate_window_n is not None
            else self._window_n + 1
        )
        if plan is None or next_window != plan.first_window:
            return
        while True:
            _, current_round = await self._checkpoint_epoch_drand_snapshot()
            if current_round >= plan.activation_not_before_round:
                return
            await asyncio.sleep(0.5)

    def _open_window(self) -> None:
        """Prepare one legacy window without exposing it to HTTP yet."""
        if self._candidate_window_n is None:
            self._candidate_window_n = self._window_n + 1
        self._active_batchers = self._build_window_batchers(
            self._candidate_window_n
        )

    def _build_window_batchers(
        self,
        target_window: int,
    ) -> dict[str, GrpoWindowBatcher]:
        """Create one environment batcher set for an exact logical window.

        Builds all batchers and wires the active checkpoint hash, but does
        NOT expose them to the HTTP server yet — call ``_activate_window``
        after ``_set_window_randomness`` succeeds. This two-phase open
        prevents miner submissions from reaching a batcher whose
        ``randomness`` is still the default ``""``, which crashes commitment
        verification in ``indices_from_root`` if the chain call that fills
        randomness fails (e.g. finney WebSocket returns 503).
        """
        from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW
        from reliquary.validator.fill_closed_batch_assembler import (
            FillClosedBatchAssembler,
        )
        from reliquary.validator.fill_window import FillState

        if self._candidate_activation_nonce is None:
            self._candidate_activation_nonce = os.urandom(32)
        target_window = int(target_window)
        epoch_window = self._checkpoint_epoch_window(target_window)
        self._set_window_preparation_stage("batcher_construction")
        bootstrap = is_bootstrap_window(
            window_start=target_window,
            subnet_start=SUBNET_START_BLOCK,
        )
        cp = self._checkpoint_store.current_manifest()
        cp_hash = cp.revision if cp else ""
        if self.proof_scheduler is not None and not (
            cp_hash
            and self.proof_scheduler.state is SchedulerState.RUNNING
            and self.proof_scheduler.checkpoint_ready(cp_hash)
        ):
            raise RuntimeError(
                "proof scheduler is not ready for the active checkpoint"
            )
        operator_by_hotkey = self.server.operator_by_hotkey_snapshot()
        batchers: dict[str, GrpoWindowBatcher] = {}
        # v6 only (R10). One ``FillState`` is shared across every
        # per-environment batcher for this window: the service builds one
        # ``GrpoWindowBatcher`` per environment, but ``FillState`` is
        # multi-key (budgets) and window-wide (picks), so each batcher
        # can't own its own instance. This is the single place a v6
        # window's ``FillState`` is constructed -- every batcher below
        # just gets the same object assigned to ``.fill_state``; nothing
        # else in the codebase constructs one.
        #
        # Amendment v6.1 (R33, R35): admission is bounded by
        # ``FILL_CLOSED_ADMISSION_BUDGET_PER_ENV``, not the old per-env
        # proven target -- the window no longer closes on a proven count,
        # it closes at the ``FILL_CLOSED_EMISSIONS_PER_WINDOW``-th pick
        # (``picks_target``), which is window-wide, not per-environment.
        shared_fill_state = (
            FillState(
                budgets={
                    env_name: FILL_CLOSED_ADMISSION_BUDGET_PER_ENV
                    for env_name in self.envs
                },
                picks_target=FILL_CLOSED_EMISSIONS_PER_WINDOW,
            )
            if FILL_CLOSED_ENABLED
            else None
        )
        # v6 only (R13). Batch ASSEMBLY moves here too, beside FillState,
        # same gate: a GrpoWindowBatcher only ever holds its own
        # environment's proven groups (see batcher.py:_emit_training_
        # batch), so it cannot join every environment's B_BATCH chunk
        # into one DAPO training batch by itself -- only the service,
        # which sees every environment's batcher for this window, can.
        # One assembler per window, injected as every batcher's
        # ``emit_training_batch_fn`` below. Logged (not raised) before
        # being replaced: a still-open PREVIOUS window's assembler may
        # hold a remainder that never became a full B_BATCH-per-
        # environment payload -- by design (see FillClosedBatchAssembler's
        # module docstring: a partial batch is exactly what its
        # carry-forward exists to avoid), but it should stay observable.
        #
        # R16: ``close()`` is what actually disposes of that remainder
        # now -- emitted as one final partial batch when every
        # environment contributed at least one group, tombstoned
        # otherwise -- instead of the WARNING below being the only
        # signal, one window later, that proven, paid rollouts were
        # dropped with no marker. The snapshot is still read FIRST,
        # before ``close()`` resets it, purely so this log line keeps
        # reporting what was actually outstanding.
        previous_assembler = getattr(self, "_fill_closed_assembler", None)
        if previous_assembler is not None:
            remainder = previous_assembler.remainder_snapshot()
            has_remainder = any(remainder["in_accumulator"].values()) or any(
                remainder["pending"].values()
            )
            previous_assembler.close()
            logger.log(
                logging.WARNING if has_remainder else logging.DEBUG,
                "FillClosedBatchAssembler: window %d closed with remainder "
                "%s (close() emitted it as a final partial batch, or "
                "tombstoned it, so the trainer's cursor still advances -- "
                "see the assembler's own log line for which)",
                previous_assembler.window_start, remainder,
            )
        fill_closed_assembler = (
            FillClosedBatchAssembler(
                window_start=target_window,
                env_order=[name for name, _ in self.env_mix],
                enqueue_fn=self._write_fill_closed_training_payload,
                tombstone_fn=self._write_fill_closed_training_tombstone,
                # R20: one window's whole emission budget. The assembler
                # splits it per environment and per batch itself -- it is
                # the only place a v6 window's assembled batches are
                # known, and under v6 there is no auction to pay at seal.
                window_pool=1.0,
            )
            if FILL_CLOSED_ENABLED
            else None
        )
        self._fill_closed_assembler = fill_closed_assembler
        if fill_closed_assembler is not None:
            self._fill_closed_assemblers[target_window] = (
                fill_closed_assembler
            )
        for env_name, env in self.envs.items():
            open_kwargs = {
                "window_start": target_window,
                "env": env,
                "model": self.verify_model,
                "cooldown_map": self._cooldown_per_env[env_name],
                "content_cooldown_map": (
                    self._content_cooldown_per_env[env_name]
                ),
                "hash_set": self._hash_set,
                "tokenizer": self.tokenizer,
                "bootstrap": bootstrap,
                "queue_drained_predicate": (
                    self._queue_and_proofs_drained
                ),
                "operator_by_hotkey": operator_by_hotkey,
            }
            if fill_closed_assembler is not None:
                open_kwargs["emit_training_batch_fn"] = (
                    fill_closed_assembler.accept
                )
            if epoch_window is not None:
                plan = self._checkpoint_epoch_plan
                if plan is None:
                    raise RuntimeError("checkpoint epoch plan disappeared")
                slices = [
                    item
                    for item in epoch_window.prompt_slices
                    if item.environment == env_name
                ]
                if len(slices) != 1:
                    raise RuntimeError(
                        "checkpoint epoch has no unique environment slice"
                    )
                open_kwargs.update({
                    "experimental_epoch_ranking": True,
                    "experimental_prompt_range": (
                        slices[0].start,
                        slices[0].stop,
                    ),
                    "collection_seconds": plan.window_schedule.collection_seconds,
                    "max_productive_candidates": (
                        plan.candidate_limit_per_environment_lane
                    ),
                    "max_ranked_proof_attempts": (
                        plan.candidate_limit_per_environment_lane
                    ),
                })
            if self.proof_scheduler is not None:
                open_kwargs["proof_scheduler"] = self.proof_scheduler
            if self._proof_worker_pool is not None:
                open_kwargs["verify_commitment_proofs_fn"] = (
                    self._remote_commitment_verifier
                )
                # Paths that prove without naming a device — the forensic
                # sample, and the legacy non-auction admission — fall back to
                # the batcher's own model. In isolated mode that has to be a
                # proxy too, or they would hand the remote verifier a real
                # model and abort the plane.
                open_kwargs["model"] = self._default_proof_proxy()
            batcher = open_grpo_window(
                **open_kwargs,
            )
            batcher.current_checkpoint_hash = cp_hash
            if shared_fill_state is not None:
                batcher.fill_state = shared_fill_state
            if epoch_window is not None:
                plan = self._checkpoint_epoch_plan
                if plan is None:
                    raise RuntimeError("checkpoint epoch plan disappeared")
                batcher.checkpoint_epoch_id = plan.epoch_id
                batcher.checkpoint_epoch_manifest_sha256 = manifest_sha256(plan)
                batcher.checkpoint_epoch_window_offset = epoch_window.offset
                batcher.checkpoint_epoch_generation_randomness = (
                    epoch_window.generation_randomness
                )
            batchers[env_name] = batcher
        return batchers

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
        self._candidate_activation_nonce = None
        self._window_preparation_stage = None
        self.server.clear_window_preparation_failure()
        self._publish_window_preparation_state()
        self._set_state(WindowState.OPEN)

    async def _wait_for_checkpoint_epoch_phase_deadline(
        self,
        *,
        seconds_from_open: float | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Wait for an epoch phase without sealing any logical lane."""
        loop = asyncio.get_running_loop()
        if seconds_from_open is not None:
            batchers = list(self._active_batchers.values())
            if not batchers:
                raise RuntimeError("checkpoint epoch lanes are unavailable")
            opened = {
                float(
                    getattr(batcher, "window_opened_at")
                    if hasattr(batcher, "window_opened_at")
                    else getattr(batcher, "opened_at")
                )
                for batcher in batchers
            }
            if len(opened) != 1:
                raise RuntimeError("checkpoint epoch lanes did not open together")
            deadline = next(iter(opened)) + float(seconds_from_open)
        elif duration_seconds is not None:
            deadline = loop.time() + float(duration_seconds)
        else:
            raise ValueError("checkpoint epoch phase deadline is missing")
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.25, remaining))

    def _open_checkpoint_epoch(self) -> None:
        """Prepare every logical lane in one checkpoint-wide collection."""
        plan = self._checkpoint_epoch_plan
        if plan is None:
            raise RuntimeError("checkpoint epoch plan is unavailable")
        if plan.window_count != CHECKPOINT_EPOCH_REQUIRED_WINDOW_COUNT:
            raise RuntimeError("checkpoint epoch does not contain sixteen lanes")
        if self._gpu_backlog is not None:
            raise RuntimeError("checkpoint epoch cannot overlap pipelined backlog")
        if self._candidate_window_n is None:
            self._candidate_window_n = self._window_n + 1
        if self._candidate_window_n != plan.first_window:
            raise RuntimeError("checkpoint epoch does not start at candidate window")

        batchers: dict[str, GrpoWindowBatcher] = {}
        for epoch_window in plan.windows:
            lane = self._build_window_batchers(epoch_window.window_number)
            for environment, batcher in lane.items():
                if (
                    getattr(
                        batcher,
                        "checkpoint_epoch_generation_randomness",
                        None,
                    )
                    != epoch_window.generation_randomness
                ):
                    raise RuntimeError("checkpoint epoch batcher binding changed")
                batcher.randomness = epoch_window.generation_randomness
                batcher.set_prompt_range()
                batchers[f"{epoch_window.window_number}:{environment}"] = batcher
        self._active_batchers = batchers
        self._last_beacon = None
        self._verify_task = None

    def _activate_checkpoint_epoch(self, chain_info: dict) -> None:
        """Atomically expose all lanes with one exact OPEN timestamp."""
        plan = self._checkpoint_epoch_plan
        if plan is None or not self._active_batchers:
            raise RuntimeError("checkpoint epoch batchers are unavailable")
        expected_windows = {window.window_number for window in plan.windows}
        actual_windows = {
            int(batcher.window_start)
            for batcher in self._active_batchers.values()
        }
        if actual_windows != expected_windows:
            raise RuntimeError("checkpoint epoch lane set is incomplete")

        if (
            str(chain_info["name"]) != plan.epoch_beacon.chain
            or str(chain_info["hash"]) != plan.epoch_beacon.chain_hash
        ):
            raise RuntimeError("drand chain changed before epoch activation")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        monotonic_open = loop.time() if loop is not None else time.monotonic()
        wall_open = time.time()
        routes: dict[tuple[str, int], GrpoWindowBatcher] = {}
        for batcher in self._active_batchers.values():
            batcher._drand_chain_info = chain_info
            batcher.mark_window_opened(
                monotonic_time=monotonic_open,
                wall_time=wall_open,
            )
            if loop is not None:
                batcher.bind_event_loop(loop)
            environment = str(getattr(batcher.env, "name", ""))
            route = (environment, int(batcher.window_start))
            if route in routes:
                raise RuntimeError("duplicate checkpoint epoch lane")
            routes[route] = batcher

        store = self._checkpoint_epoch_store
        if store is None:
            raise RuntimeError("checkpoint epoch store is unavailable")
        store.mark_activated(plan)
        self.server.set_active_epoch_batchers(routes)
        self._window_n = plan.first_window
        self._candidate_window_n = None
        self._window_preparation_stage = None
        self.server.clear_window_preparation_failure()
        self._publish_window_preparation_state()
        self.server.set_checkpoint_epoch_phase("commitment")
        self._set_state(WindowState.OPEN)

    async def _run_checkpoint_epoch(self) -> None:
        """Collect all lanes together, then consume the frozen reservoir."""
        plan = self._checkpoint_epoch_plan
        if plan is None:
            raise RuntimeError("checkpoint epoch plan is unavailable")
        activated = False
        late_drops: dict | None = None
        completed_windows: set[int] = set()
        lane_batchers: dict[int, dict[str, GrpoWindowBatcher]] = {
            window.window_number: {} for window in plan.windows
        }

        def close_failed_epoch(failure_type: str) -> None:
            self.server.set_active_epoch_batchers({})
            if activated:
                failure_stage = self._window_iteration_stage
                for window in plan.windows:
                    if window.window_number in completed_windows:
                        continue
                    lane = lane_batchers[window.window_number]
                    if not lane:
                        continue
                    try:
                        self._enqueue_aborted_window(
                            failure_stage=failure_stage,
                            failure_type=failure_type,
                            batchers=lane,
                            late_drops=late_drops,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to tombstone checkpoint epoch lane %d",
                            window.window_number,
                        )
            self._active_batchers = {}
            if activated:
                self._window_n = plan.first_window + plan.window_count - 1
                store = self._checkpoint_epoch_store
                if store is not None:
                    store.mark_terminal(plan, status="aborted")
                self._abort_training_epoch_journal(
                    plan,
                    failure_stage=failure_stage,
                    failure_type=failure_type,
                )
            self._set_state(WindowState.READY)

        try:
            self._window_iteration_stage = "checkpoint_epoch_open"
            self._open_checkpoint_epoch()
            for batcher in self._active_batchers.values():
                environment = str(getattr(batcher.env, "name", ""))
                lane_batchers[int(batcher.window_start)][environment] = batcher
            if any(
                set(lane_batchers[window.window_number]) != set(self.envs)
                for window in plan.windows
            ):
                raise RuntimeError(
                    "checkpoint epoch environment reservoir is incomplete"
                )

            first_lane = lane_batchers[plan.first_window]
            self._window_iteration_stage = "checkpoint_epoch_admission_pools"
            await self.server.prepare_admission_pools(first_lane)
            activation_chain_info, _ = (
                await self._checkpoint_epoch_drand_snapshot()
            )
            self._activate_checkpoint_epoch(activation_chain_info)
            activated = True

            self._window_iteration_stage = "checkpoint_epoch_commitment"
            await self._wait_for_checkpoint_epoch_phase_deadline(
                seconds_from_open=plan.window_schedule.collection_seconds
            )
            # Close ingress synchronously before any await. Requests already
            # inside the HTTP handler are drained; later ones see selection and
            # cannot enter the frozen set.
            self.server.set_checkpoint_epoch_phase("selection")
            await self.server.drain_checkpoint_epoch_commitments()
            _, commitment_close_round = await self._checkpoint_epoch_drand_snapshot()
            frozen_set = self.server.freeze_checkpoint_epoch_commitment_set(
                commitment_close_round=commitment_close_round,
                validator_hotkey=str(self.wallet.hotkey.ss58_address),
            )
            publication = SignedEpochCommitmentSet(
                commitment_set=frozen_set,
                commitment_set_sha256=commitment_set_sha256(frozen_set),
                validator_signature=self.wallet.hotkey.sign(
                    commitment_set_signing_bytes(frozen_set)
                ).hex(),
            )
            store = self._checkpoint_epoch_store
            if store is None:
                raise RuntimeError("checkpoint epoch store is unavailable")
            store.install_commitment_set(plan, publication)
            self.server.install_checkpoint_epoch_commitment_set(publication)
            _, admission_beacon = await self._fetch_checkpoint_epoch_admission_beacon(
                commitment_close_round=commitment_close_round,
            )
            reveal_deadline_ts = time.time() + plan.reveal_seconds
            selected_counts = self.server.select_checkpoint_epoch_reveals(
                commitment_close_round=commitment_close_round,
                admission_beacon=admission_beacon,
                reveal_deadline_ts=reveal_deadline_ts,
            )
            logger.info(
                "Checkpoint epoch %s commitments closed lanes=%d "
                "commit_round=%d admission_round=%d selected=%d",
                plan.epoch_id[:12],
                plan.window_count,
                commitment_close_round,
                admission_beacon.round,
                sum(selected_counts.values()),
            )

            self._window_iteration_stage = "checkpoint_epoch_reveal"
            await self._wait_for_checkpoint_epoch_phase_deadline(
                duration_seconds=plan.reveal_seconds
            )
            self._set_state(WindowState.TRAINING)
            for batcher in self._active_batchers.values():
                batcher.force_seal("checkpoint_epoch_reveal_closed")
            drain_timeouts = await self._freeze_auction_populations(
                list(self._active_batchers.values())
            )
            if any(drain_timeouts.values()):
                raise RuntimeError("checkpoint epoch reveal drain timed out")

            late_drops = dict(self._late_drops)
            self._late_drops.clear()
            reject_counts = dict(
                getattr(self.server, "_recent_reject_counts", {})
            )
            epoch_seal = await self._fetch_checkpoint_epoch_seal_beacon(
                after_round=admission_beacon.round
            )
            # Close every route after selected reveals are frozen.
            self.server.set_active_epoch_batchers({})

            for index, window in enumerate(plan.windows):
                final_lane = index == plan.window_count - 1
                lane = lane_batchers[window.window_number]
                self._active_batchers = lane
                self._window_n = window.window_number
                self._window_iteration_stage = (
                    "checkpoint_epoch_finalize"
                    if final_lane
                    else "checkpoint_epoch_reservoir"
                )
                await self._train_and_publish(
                    batchers=lane,
                    window_n=window.window_number,
                    verify_task=None,
                    late_drops=late_drops if final_lane else {},
                    server_reject_counts=(
                        reject_counts if final_lane else {}
                    ),
                    epoch_seal=epoch_seal,
                    epoch_finalize=final_lane,
                )
                completed_windows.add(window.window_number)
        except asyncio.CancelledError:
            close_failed_epoch("CancelledError")
            raise
        except FatalProofPlaneError:
            close_failed_epoch("FatalProofPlaneError")
            raise
        except Exception as exc:
            close_failed_epoch(type(exc).__name__)
            if activated:
                raise CheckpointEpochExecutionError(
                    "checkpoint epoch failed after its common OPEN"
                ) from exc
            raise

        self._active_batchers = {}
        self._window_n = plan.first_window + plan.window_count - 1
        store = self._checkpoint_epoch_store
        if store is None:
            raise RuntimeError("checkpoint epoch store is unavailable")
        epoch_windows = {window.window_number for window in plan.windows}
        training_status = (
            "aborted"
            if epoch_windows.intersection(self._training_tombstoned_windows)
            else "completed"
        )
        store.mark_terminal(plan, status=training_status)
        if training_status == "aborted":
            self._abort_training_epoch_journal(
                plan,
                failure_stage="checkpoint_epoch_training_journal",
                failure_type="IncompleteCheckpointEpoch",
            )
        else:
            self._write_training_epoch_marker(plan, status="completed")
        self._set_state(WindowState.READY)

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

        Gated on the never-refunded grading-starts backstop, not on the
        productive admission budget: non-productive rejects (protocol
        conformance and out-of-zone) refund the latter, so for a
        degenerate-reward env it never reaches its cap. Only the ceiling that
        nothing gives back is a real "can't fill anymore" signal.
        """
        if batcher is None or batcher.is_sealed():
            return False
        distinct_valid = self._distinct_valid_prompt_count(batcher)
        if distinct_valid >= B_BATCH:
            return False
        if (
            getattr(batcher, "proof_grading_attempts", 0)
            < getattr(
                batcher,
                "max_grading_starts",
                MAX_GRADING_STARTS_PER_WINDOW,
            )
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
            if int(getattr(batcher, "pending_upload_precommits", 0) or 0):
                return False
        return True

    async def _freeze_auction_populations(
        self, batchers: list[Any]
    ) -> dict[str, bool]:
        """Freeze only a completely drained auction population or abort it."""
        auction_batchers = [
            batcher for batcher in batchers
            if getattr(batcher, "difficulty_auction_enabled", False)
        ]
        if not auction_batchers:
            return {}

        loop = asyncio.get_running_loop()
        drain_started = loop.time()
        drain_deadline = (
            loop.time() + AUCTION_ADMISSION_DRAIN_DEADLINE_SECONDS
        )
        while not self._queue_and_proofs_drained():
            if loop.time() >= drain_deadline:
                break
            await asyncio.sleep(PROOF_ADMISSION_STALL_POLL_SECONDS)

        timed_out = not self._queue_and_proofs_drained()
        if timed_out:
            for batcher in auction_batchers:
                begin_snapshot = getattr(
                    batcher, "begin_seal_snapshot", None
                )
                if callable(begin_snapshot):
                    begin_snapshot()
            abort_stats = await self.server.abort_auction_admission(
                auction_batchers
            )
        else:
            abort_stats = {}

        queue_by_env = dict(
            getattr(self.server, "submit_queue_depth_by_environment", {}) or {}
        )
        inflight_by_env = dict(
            getattr(
                self.server,
                "proof_verification_inflight_by_environment",
                {},
            )
            or {}
        )
        timed_out_by_env: dict[str, bool] = {}

        def _env_name(active_batcher: Any) -> str:
            candidate = getattr(
                getattr(active_batcher, "env", None), "name", None
            )
            if isinstance(candidate, str):
                return candidate
            for configured_name, configured_batcher in self._active_batchers.items():
                if configured_batcher is active_batcher:
                    return str(configured_name)
            return "unknown"

        for batcher in auction_batchers:
            env_name = _env_name(batcher)
            timed_out_by_env[env_name] = timed_out
            conservation_fn = getattr(
                type(batcher), "upload_precommit_conservation", None
            )
            conservation = (
                conservation_fn(batcher)
                if callable(conservation_fn)
                else {}
            )
            batcher.auction_seal_drain = {
                "elapsed_seconds": max(0.0, loop.time() - drain_started),
                "timed_out": timed_out,
                "outcome": "aborted" if timed_out else "complete",
                "queue_depth_at_snapshot": int(
                    queue_by_env.get(env_name, 0) or 0
                ),
                "inflight_workers_at_snapshot": int(
                    inflight_by_env.get(env_name, 0) or 0
                ),
                "pending_reservations_at_snapshot": int(
                    getattr(batcher, "pending_proof_reservations", 0) or 0
                ),
                "inflight_reservations_at_snapshot": int(
                    getattr(batcher, "inflight_proof_reservations", 0) or 0
                ),
                "receipt_conservation": conservation,
                "abort_terminalization": dict(abort_stats),
            }
            if timed_out:
                batcher.auction_admission_aborted = True
                existing_reason = getattr(batcher, "force_seal_reason", None)
                if not isinstance(existing_reason, str) or not existing_reason:
                    batcher.force_seal_reason = "auction_admission_drain_abort"
            else:
                if not conservation.get("conserved", True) or conservation.get(
                    "pending", 0
                ):
                    raise RuntimeError(
                        f"receipt conservation failed for {env_name}: "
                        f"{conservation}"
                    )
                begin_snapshot = getattr(
                    batcher, "begin_seal_snapshot", None
                )
                if callable(begin_snapshot):
                    begin_snapshot()

        if timed_out:
            logger.warning(
                "Window %d auction admission drain reached %.1fs; aborted "
                "the complete window without ranking or training",
                self._window_n,
                AUCTION_ADMISSION_DRAIN_DEADLINE_SECONDS,
            )
        return timed_out_by_env

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
            # Auction timing belongs exclusively to poll_deadline. An exhausted
            # grading-start backstop can occur before the 60 s fairness floor
            # and must not bypass the GPU/quiet/population gates. The 100 s
            # ceiling remains unconditional, so no auction can hang here.
            return None
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

    def _read_trainer_step_cursor(self) -> int | None:
        """The last journal key the trainer reported CONSUMING, or None.

        Amendment v6.1 point 2, transport in R38/R40:
        ``fetch_step_cursor`` performs the remote R2 GET that reads back
        what the detached trainer's own drain uploaded (the validator and
        the trainer run on different hosts and never share a local
        queue_dir, so the LOCAL-file ``read_step_cursor`` cannot see it).
        It is fire-and-collect and internally cached (R40 #1): this call
        never blocks on network I/O, returning the last value a
        background fetch completed with and kicking a new one only when
        that value is stale and none is already in flight -- safe to call
        on every poll tick.

        ``fetch_step_cursor`` already swallows a missing/torn/unparseable
        object, a network error, or a timeout into None -- pacing
        telemetry must never take the poll loop down -- and this adds the
        same treatment for a queue reference that refuses outright. None
        means "unknown", which the gate reads as "not yet", so every
        failure here degrades to picks stopping and the backstop sealing
        a partial window, never to a pick firing on a cadence nobody
        measured.
        """
        try:
            return self._training_payload_queue_ref().fetch_step_cursor()
        except Exception:
            logger.warning(
                "trainer step cursor unreadable; picks stay gated",
                exc_info=True,
            )
            return None

    def _fill_closed_pick_gate_open(self, batcher, *, next_pick: int) -> bool:
        """Is the pacing gate for pick ``next_pick`` (1-indexed) open?

        R34 as amended by R41, and the one place the off-by-ones live:

        * ``next_pick == 1``: nothing has been emitted into this window's
          key range yet, so no cursor value could ever release it -- the
          ONLY pick on the ``FILL_CLOSED_FIRST_PICK_SECONDS`` floor,
          measured on the BATCHER's clock (the same one
          ``FILL_CLOSED_MAX_SECONDS`` uses, so the floor and the backstop
          can never disagree about how old a window is).
        * ``next_pick >= 2``: pick k emits batch index ``k - 1``, so the
          batch ``depth`` picks back is index ``k - depth - 1``, CLAMPED
          at 0 (R41): pick 2 waits on batch 0 -- the end of the first
          training step -- rather than sharing the floor. The trade,
          measured when R41 was ruled: one emit-to-fetch bubble (~8-10 s)
          at the end of step 1, once per window, and in exchange the
          floor seats fall from 2/16 to 1/16 of the window and pick 2's
          candidates get a whole step of extra generation time. Picks 2
          and 3 share the batch-0 gate (max(0, k-depth-1) collides
          there), which is what refills the depth-2 buffer immediately
          after the bubble: from step 2 on the trainer always holds one
          batch in hand, exactly the buffer R34 asks for.

        ``>=`` rather than ``==`` because the trainer skips keys it never
        trains (tombstones, quarantined batches, health skips all advance
        the cursor) -- and because the encoding is monotone across
        windows, so a cursor left in a PREVIOUS window is simply too
        small and needs no separate staleness rule.
        """
        if next_pick <= 1:
            age = self._window_open_age_seconds(batcher)
            return age is not None and age >= FILL_CLOSED_FIRST_PICK_SECONDS
        from reliquary.infrastructure.training_payload_queue import (
            encoded_window_journal_key,
        )

        required = encoded_window_journal_key(
            int(batcher.window_start),
            max(0, next_pick - FILL_CLOSED_PICK_PIPELINE_DEPTH - 1),
        )
        cursor = self._read_trainer_step_cursor()
        return cursor is not None and int(cursor) >= required

    def _drive_fill_closed_picks(self, batchers) -> bool:
        """Fire at most ONE window-wide pick event; return whether it did.

        R36: a pick is a WINDOW event, not a per-environment one. One pick
        k is one DAPO batch built from every environment's own k-th chunk,
        so this fires only when EVERY environment can seat ``B_BATCH`` and
        then drives all of them in a single pass. Driving a ready
        environment alone would emit half a batch and, worse, advance the
        window-wide count -- closing a two-environment window at half the
        batches R35 asks for.

        Readiness is checked before the pacing gate on purpose: a fleet
        that is not producing then costs no cursor read at all, and the
        cursor is read at most once per poll tick (once per call, inside
        ``_fill_closed_pick_gate_open``, and only on the branch that
        actually needs it).

        Check-then-pick without holding the lock across the readiness
        scan and the picks is safe because THIS LOOP OWNS BOTH. Picking,
        the readiness check and ``poll_deadline``'s seal all run on the
        one ``_wait_for_window_seal`` task, so no seal and no rival pick
        can interleave between them; the proof threads that run
        concurrently only ever GROW the pool, which can invalidate a
        False (harmless -- the next 0.5 s tick sees it) but never a True.
        That ownership, not the pool's growth, is what excludes the
        seal-vs-pick interleaving. If an environment still refuses, the
        invariant is broken somewhere: log at ERROR rather than emit a
        half batch silently -- an incomplete event costs nothing wrong
        under R37, since ``picks_emitted`` is the MIN over environments'
        ordinals and a half-taken event simply does not count.
        """
        if not FILL_CLOSED_ENABLED:
            return False
        picking = [
            batcher for batcher in batchers
            if getattr(batcher, "fill_state", None) is not None
        ]
        if not picking:
            return False
        fill_state = picking[0].fill_state
        with fill_state.lock:
            if fill_state.is_closed():
                return False
            next_pick = int(fill_state.snapshot()["picks_emitted"]) + 1
        if not all(batcher.can_pick() for batcher in picking):
            return False
        if not self._fill_closed_pick_gate_open(
            picking[0], next_pick=next_pick
        ):
            return False
        refused = [
            str(getattr(getattr(batcher, "env", None), "name", "?"))
            for batcher in picking
            if not batcher.pick_training_batch()
        ]
        if refused:
            # Both shapes are the same broken invariant and both are
            # loud. A PARTIAL refusal leaves the event half-taken, which
            # R37's per-environment ordinals make survivable (the
            # window-wide count is the min, so it simply does not move)
            # but never correct. A TOTAL refusal means readiness said
            # every environment could seat a batch and then none did.
            logger.error(
                "Window %d: pick event %d was driven on %d environment(s) "
                "and %s refused after passing the readiness check (%s); "
                "the window-wide count does not advance for an incomplete "
                "event",
                int(getattr(picking[0], "window_start", 0)),
                next_pick,
                len(picking),
                ",".join(refused),
                "no environment took it"
                if len(refused) == len(picking)
                else "the event is half-taken",
            )
        return len(refused) < len(picking)

    async def _wait_for_window_seal(
        self,
        *,
        fixed_deadline_only: bool = False,
        early_close_ready: Callable[[], bool] | None = None,
    ) -> str:
        """Wait until every active env's batcher seals.

        Auction batchers have a 100 s ceiling. In enforce mode they may close
        after 60 s once the primary population exists, the candidate stream is
        quiet, uploads/admission are drained, and ``early_close_ready`` says the
        previous pipelined GPU half has finished. Legacy batchers retain their
        B-distinct/drand-boundary seal. Per-environment liveness guards cannot
        let a fast environment cut a slower one short. The window advances only
        once all are sealed (or the global timeout).
        """
        batchers = list(self._active_batchers.values())
        if not batchers:
            return "no_active_batcher"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + WINDOW_TIMEOUT_SECONDS
        dup_since: dict[str, float] = {}
        reasons: dict[str, str] = {}
        while True:
            # v6.1 (R34/R36): the pick event runs on this same 0.5 s
            # cadence, BEFORE the seal poll -- so the 16th pick closes the
            # window and ``poll_deadline`` seals it on the same tick
            # rather than a tick later. Inert with the gate off.
            self._drive_fill_closed_picks(batchers)
            for b in batchers:
                # The hard ceiling ignores GPU readiness; only adaptive close
                # is gated by it.
                poll = getattr(b, "poll_deadline", None)
                if callable(poll):
                    pipeline_ready = (
                        True
                        if early_close_ready is None
                        else bool(early_close_ready())
                    )
                    poll(pipeline_ready=pipeline_ready)
                if b.is_sealed():
                    continue
                r = (
                    None
                    if fixed_deadline_only
                    else self._force_seal_dead_batcher(b, dup_since)
                )
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

        drain_timeouts = await self._freeze_auction_populations(batchers)
        for env_name, timed_out in drain_timeouts.items():
            if timed_out:
                reasons[env_name] = "auction_admission_drain_abort"

        if not reasons:
            return "sealed"
        if len(reasons) == 1:
            return next(iter(reasons.values()))
        return ",".join(f"{e}={r}" for e, r in reasons.items())

    async def _set_window_randomness(self, subtensor) -> None:
        """Populate all active batchers' per-window randomness seed.

        GRAIL sketch verification re-derives challenge indices from this
        seed; miner and validator must agree on the value published by
        ``/state``. All batchers share the same randomness for a given window.

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
        epoch_window = self._checkpoint_epoch_window(target_window)
        if epoch_window is not None:
            for batcher in self._active_batchers.values():
                if (
                    getattr(batcher, "checkpoint_epoch_generation_randomness", None)
                    != epoch_window.generation_randomness
                ):
                    raise RuntimeError(
                        "prepared batcher does not match checkpoint epoch"
                    )
                batcher.randomness = epoch_window.generation_randomness
                batcher.set_prompt_range()
            self._last_beacon = None
            self._verify_task = None
            return
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

        if self.use_drand:
            activation_nonce = getattr(self, "_candidate_activation_nonce", None)
            if activation_nonce is None:
                raise RuntimeError("window activation nonce is unavailable")
            randomness = _bind_window_activation_randomness(
                randomness,
                target_window=int(target_window),
                activation_nonce=activation_nonce,
            )

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

            from reliquary.validator.verifier import rewards_std

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
                    sigma=rewards_std(list(pending.rewards or ())),
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

    def _lr_global_step_hint(self) -> int:
        """Restored LR-schedule position for a same-run restart.

        The EXACT scheduler step count the resumed checkpoint recorded in
        its profile at publish — never derived from checkpoint numbers
        (this run inherited its counter across a weight reset, and the
        publish cadence has changed across protocol versions, so
        checkpoint_n x interval is not a step count). Returns 0 — full
        warmup — fail-closed: missing field (all pre-field checkpoints),
        or a checkpoint published under a DIFFERENT training run id
        (new run id on old weights => new run => warmup).
        """
        resumed = getattr(self, "_resumed_training_run_id", None)
        if resumed is not None and resumed != TRAINING_RUN_ID:
            return 0
        step = getattr(self, "_resumed_lr_schedule_step", None)
        if step is None:
            return 0
        return max(0, int(step))

    async def _train_and_publish(
        self,
        batchers: dict | None = None,
        window_n: int | None = None,
        verify_task=None,
        late_drops: dict | None = None,
        server_reject_counts: dict | None = None,
        epoch_seal: tuple[int, BeaconBinding] | None = None,
        epoch_finalize: bool = False,
    ) -> None:
        """TRAINING + PUBLISHING + READY phases for one sealed window."""
        # Pipelined mode passes the SEALED window's batchers explicitly while
        # self._active_batchers already points at the next, collecting window.
        # owns_routing gates every mutation of the live routing state: the GPU
        # half of a stashed window must never clear the collecting window's
        # batchers out from under the HTTP server.
        owns_routing = batchers is None
        if batchers is None:
            batchers = self._active_batchers
        if window_n is None:
            window_n = self._window_n
        if verify_task is None and owns_routing:
            # Serial mode: the live task belongs to this (the only) window.
            # Pipelined mode passes the STASHED window's task explicitly —
            # self._verify_task by now belongs to the collecting window and
            # must be neither awaited nor cancelled here.
            verify_task = self._verify_task
        if not batchers:
            logger.warning("_train_and_publish called with no active batchers")
            return

        # Background drand cross-check flips beacon_invalid if the beacon
        # was forged or the verify crashed. Await up to 2s for its verdict
        # before checking — by seal-time (~3s after OPEN) it's almost always
        # done. Plain wait_for (no shield): if it times out, cancel the task
        # and check the flag below with whatever state it reached.
        if verify_task is not None and not verify_task.done():
            try:
                await asyncio.wait_for(verify_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Window %d: drand verify still running at train-time; "
                    "proceeding without final verdict (will check flag below)",
                    window_n,
                )
        # Check if any batcher has been invalidated (beacon_invalid propagated
        # to all batchers by _verify_beacon_async, so checking any one suffices).
        if any(b.beacon_invalid for b in batchers.values()):
            logger.error(
                "Window %d: dropping seal+train+archive — beacon invalid",
                window_n,
            )
            self._enqueue_aborted_window(
                failure_stage="beacon_verification",
                failure_type="InvalidBeacon",
                batchers=batchers,
                late_drops=late_drops,
            )
            if owns_routing:
                self.server.set_active_batchers({})
                self._active_batchers = {}
            if owns_routing:
                self._set_state(WindowState.READY)
            return

        if any(
            bool(getattr(b, "auction_admission_aborted", False))
            for b in batchers.values()
        ):
            logger.error(
                "Window %d: admission drain aborted; skipping ranking, "
                "rewards, training and checkpoint publication",
                window_n,
            )
            self._enqueue_aborted_window(
                failure_stage="admission_drain",
                failure_type="AdmissionDrainTimeout",
                batchers=batchers,
                late_drops=late_drops,
            )
            if owns_routing:
                self.server.set_active_batchers({})
                self._active_batchers = {}
            if owns_routing:
                self._set_state(WindowState.READY)
            return

        if owns_routing:
            self._set_state(WindowState.TRAINING)
        # Seal every environment after its adaptive close or hard ceiling.
        # Auction mode
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
        # Fetch a fresh drand beacon AFTER the populations freeze. It strictly
        # orders v5 candidates equal on score and throughput, and keys the
        # forensic sample. If the bounded fetch fails, v5 uses its deterministic
        # operator/prompt ticket (legacy non-throughput profiles retain exact
        # validator-arrival fallback) and forensics are disabled.
        epoch_batchers = [
            batcher
            for batcher in batchers.values()
            if getattr(batcher, "experimental_epoch_ranking", False) is True
        ]
        if epoch_batchers:
            if len(epoch_batchers) != len(batchers):
                raise RuntimeError("mixed epoch and production ranking window")
            close_round, seal_beacon = (
                epoch_seal
                if epoch_seal is not None
                else await self._fetch_checkpoint_epoch_seal_beacon()
            )
            for batcher in epoch_batchers:
                batcher.collection_close_drand_round = close_round
                batcher.seal_randomness = seal_beacon.randomness
                batcher.seal_beacon_round = seal_beacon.round
        else:
            seal_randomness = await self._fetch_seal_randomness()
            for b in batchers.values():
                b.seal_randomness = seal_randomness
        # Both environments submit their strict rank order to one global,
        # device-owning scheduler. The scheduler applies decisions in rank
        # order even when distinct replicas finish out of order.
        seal_items = tuple(batchers.items())
        if self.proof_scheduler is None:
            # Exact v2 compatibility: both batchers share one verify model.
            seal_results = []
            for _name, batcher in seal_items:
                seal_results.append(
                    await asyncio.to_thread(
                        batcher.seal_batch,
                        pool=pool_per_env,
                    )
                )
        else:
            # Pipelined mode joins BOTH env threads before any propagation:
            # without it, one env's raise leaves the other's seal_batch
            # proving on the GPU while the loop continues into the next
            # window's half on the same device. Serial mode keeps main's
            # immediate propagation (flag-off parity).
            from reliquary.constants import PIPELINED_WINDOWS as _pipelined

            seal_results = await asyncio.gather(*(
                asyncio.to_thread(
                    batcher.seal_batch,
                    pool=pool_per_env,
                    commit_side_effects=False,
                )
                for _name, batcher in seal_items
            ), return_exceptions=bool(_pipelined))
            for _res in seal_results:
                if isinstance(_res, BaseException):
                    raise _res
        sealed: dict[str, tuple] = {
            name: result
            for (name, _batcher), result in zip(
                seal_items, seal_results
            )
        }
        proof_capacity_aborts = {
            name: getattr(
                batcher,
                "proof_capacity_abort_reason",
                None,
            )
            for name, batcher in batchers.items()
            if bool(
                getattr(batcher, "proof_capacity_aborted", False)
            )
        }
        if (
            self.proof_scheduler is not None
            and self.proof_scheduler.state is SchedulerState.FAULTED
        ):
            snapshot_fn = getattr(self.proof_scheduler, "snapshot", None)
            scheduler_snapshot = (
                snapshot_fn() if callable(snapshot_fn) else {}
            )
            proof_capacity_aborts["proof_scheduler"] = (
                scheduler_snapshot.get("fault_reason")
                or "faulted"
            )
        if proof_capacity_aborts:
            for batcher in batchers.values():
                batcher.discard_seal_side_effects()
            logger.error(
                "Window %d: proof capacity aborted %s; skipping rewards, "
                "training and checkpoint publication",
                window_n,
                proof_capacity_aborts,
            )
            self._enqueue_aborted_window(
                failure_stage="proof_capacity",
                failure_type="ProofCapacityAbort",
                batchers=batchers,
                late_drops=late_drops,
            )
            if owns_routing:
                self.server.set_active_batchers({})
                self._active_batchers = {}
            if owns_routing:
                self._set_state(WindowState.READY)
            if (
                self.proof_scheduler is not None
                and self.proof_scheduler.state is SchedulerState.FAULTED
            ):
                raise FatalProofPlaneError(
                    "proof scheduler faulted; restart required before "
                    "another window"
                )
            return
        if self.proof_scheduler is not None:
            # Cooldowns and rollout-hash reservations are committed only after
            # every environment has sealed successfully.
            for batcher in batchers.values():
                batcher.commit_seal_side_effects()
        for name, (batch, rewards) in sealed.items():
            batchers[name].rewards_by_hotkey = rewards

        # Worker acceptance means "admitted to the auction pool". Publish a
        # second, final /verdicts record after seal so miners can distinguish a
        # selected/rewarded candidate, an honest non-winner, and a deferred-proof
        # failure. This is observability only and cannot change selection.
        for batcher in batchers.values():
            self._record_auction_final_verdicts(batcher)

        # Emit per-submission lifecycle telemetry for every env's accepted
        # pool. Carried over from PR #40 (validator observability) and
        # extended with env_name so downstream consumers can split by env.
        for env_name, batcher in batchers.items():
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
                    # Which mechanism actually credited it (R20): under v6
                    # the seal path's slot share is reported but not paid.
                    "payment_source": meta.get("payment_source"),
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
        for _b in batchers.values():
            _snapshot_fn = getattr(
                type(_b), "rejection_telemetry_snapshot", None
            )
            _reject_counts = (
                _snapshot_fn(_b).get("reject_counts", {})
                if callable(_snapshot_fn)
                else getattr(_b, "reject_counts", {})
            )
            for _k, _v in dict(_reject_counts).items():
                combined_reject_counts[_k] = combined_reject_counts.get(_k, 0) + _v
        flat_window_batch = [
            group for env_batch in window_batches.values() for group in env_batch
        ]
        window_quarantine = assess_training_batch(
            flat_window_batch,
            reject_counts=combined_reject_counts,
        )
        _quarantine_archive = window_quarantine.to_archive()
        for _b in batchers.values():
            _b.training_quarantine = _quarantine_archive

        checkpoint_revisions = {
            str(getattr(b, "current_checkpoint_hash", ""))
            for b in batchers.values()
        }
        if len(checkpoint_revisions) != 1:
            logger.error(
                "Window %d has inconsistent checkpoint revisions across envs: %s",
                window_n, sorted(checkpoint_revisions),
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
            # No single behavior revision exists — the trainer cannot use
            # this window. Tombstone instead so its journal stays gapless.
            self._write_training_tombstone(
                window_n, "inconsistent_checkpoint", "InconsistentCheckpoint",
            )
        else:
            checkpoint_revision = next(iter(checkpoint_revisions))
            # Off the event loop: the encode walks every token of the
            # window (np.savez_compressed included) — seconds of CPU that
            # would otherwise freeze /state and the drain workers.
            await asyncio.to_thread(
                self._write_training_payload,
                window_batches, window_n, checkpoint_revision,
                _quarantine_archive,
            )
            accumulator_update = self._training_accumulator.add_window(
                {} if window_quarantine.quarantined else window_batches,
                window_n=window_n,
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
        allow_partial_epoch_batch = bool(
            epoch_finalize
            and EXPERIMENTAL_CHECKPOINT_EPOCH_TRAINING_MODE
            == "aggregate_one_step"
        )
        accumulator_ready = (
            len(checkpoint_revisions) == 1
            and (
                self._training_accumulator.ready
                or (
                    allow_partial_epoch_batch
                    and self._training_accumulator.has_groups_for_all_targets
                )
            )
        )
        batches = (
            self._training_accumulator.training_batches(
                env_order,
                allow_partial=allow_partial_epoch_batch,
            )
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
                window_n,
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
        from reliquary.constants import DETACHED_TRAINER as _detached

        detached_trainer = bool(_detached)
        cadence_publication_pending = (
            self._trained_windows_since_publish >= self._publish_every
        )
        publication_retry_pending = (
            cadence_publication_pending
            or self._adaptive_publication_pending
        )
        checkpoint_ceiling_reached = (
            TRAIN_UNTIL_CHECKPOINT_N > 0
            and self._checkpoint_n >= TRAIN_UNTIL_CHECKPOINT_N
        )
        skip_train = (
            emergency_freeze
            or detached_trainer
            or checkpoint_ceiling_reached
            or publication_retry_pending
        )
        if accumulator_ready and skip_train:
            if emergency_freeze:
                blocked_reason = "emergency_training_freeze"
            elif detached_trainer:
                blocked_reason = "detached_trainer"
            elif checkpoint_ceiling_reached:
                blocked_reason = "training_checkpoint_ceiling"
            else:
                blocked_reason = "checkpoint_publication_pending"
            accumulator_meta["blocked_reason"] = blocked_reason
            logger.info(
                "Window %d: %s — retaining balanced batch and skipping "
                "train_step + publish (checkpoint=%d ceiling=%d)",
                window_n,
                blocked_reason,
                self._checkpoint_n,
                TRAIN_UNTIL_CHECKPOINT_N,
            )
            if detached_trainer and epoch_finalize:
                accumulator_meta["discarded"] = self._training_accumulator.reset()
                accumulator_meta["reset_reason"] = "detached_epoch_journal_owns_batch"
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
                    window_index=window_n,
                    global_step_hint=self._lr_global_step_hint(),
                    **(
                        {"behavior_model": self.verify_model}
                        if RECOMPUTE_PI_OLD_FROM_VERIFY
                        else {}
                    ),
                )
                trained = True
            except TrainingStepSkipped as exc:
                outside_clip_ratio = exc.metrics.get(
                    "train/ppo_ratio_outside_clip_ratio"
                )
                logger.warning(
                    "train_step rejected for window %d: reason=%s "
                    "grad_norm=%s ppo_outside_clip=%s",
                    window_n,
                    exc.reason,
                    exc.grad_norm,
                    outside_clip_ratio,
                )
                accumulator_meta["reset_reason"] = (
                    f"training_health_gate:{exc.reason}"
                )
                accumulator_meta["training_skip"] = {
                    "reason": exc.reason,
                    "grad_norm": exc.grad_norm,
                    "metrics": exc.metrics,
                }
                if (
                    exc.reason == "policy_ratio_drift"
                    and self._trained_windows_since_publish > 0
                ):
                    self._adaptive_publication_pending = True
                    self._adaptive_publication_reason = exc.reason
                    accumulator_meta["adaptive_publication_triggered"] = True
                    logger.warning(
                        "Window %d detected stale behavior-policy drift after "
                        "%d safe updates; publishing the accumulated model "
                        "without the rejected step",
                        window_n,
                        self._trained_windows_since_publish,
                    )
            except Exception:
                # Don't let a training failure (e.g. CUDA OOM) skip
                # _archive_window — miners still need their R2 contribution
                # recorded so the EMA / on-chain weights reflect this window.
                logger.exception(
                    "train_step failed for window %d; archiving anyway and "
                    "skipping publish", window_n,
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
                window_n, total_subs, total_target, retained,
            )

        accumulator_meta["trained"] = trained
        accumulator_meta["post_action"] = self._training_accumulator.snapshot()
        self.server.set_training_accumulator_state(accumulator_meta["post_action"])
        log_structured(
            logger,
            logging.INFO,
            "validator_training_accumulator",
            {
                "window_n": window_n,
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
        for _b in batchers.values():
            _b.training_accumulator = accumulator_meta

        if owns_routing:
            self._set_state(WindowState.PUBLISHING)
        if trained:
            self._trained_windows_since_publish += 1
        # checkpoint_n only advances on publish. Publish cadence is based on
        # successful trained windows rather than exact window number so a
        # quarantined boundary window cannot freeze the public checkpoint. Once
        # the cadence is reached, retry a failed upload without applying another
        # optimizer step to the pending candidate.
        next_n = self._checkpoint_n + 1
        should_publish = (
            not emergency_freeze
            and not detached_trainer
            and not checkpoint_ceiling_reached
            and (
                self._trained_windows_since_publish >= self._publish_every
                or self._adaptive_publication_pending
                or (
                    epoch_finalize
                    and self._trained_windows_since_publish > 0
                )
                or (
                    trained
                    and self._checkpoint_store.current_manifest() is None
                )
            )
        )
        if should_publish and not owns_routing and not epoch_finalize:
            # SERIAL BEAT: publication only ever runs in a serial iteration,
            # where no window is collecting against the old checkpoint. This
            # half is pipelined (adaptive drift or a quarantined train step
            # shifted the cadence after the seal-time forecast), so defer:
            # the pending counters/flags survive untouched, the next seal's
            # forecast sees them and runs that window serially, and the
            # publish happens there.
            logger.warning(
                "Window %d: publication due but this GPU half is pipelined; "
                "deferring publication to the next serial-beat window",
                window_n,
            )
            should_publish = False
        if should_publish:
            if self._adaptive_publication_pending:
                publication_reason = "adaptive_policy_ratio_drift"
            elif self._trained_windows_since_publish >= self._publish_every:
                publication_reason = "cadence"
            elif epoch_finalize:
                publication_reason = "checkpoint_epoch"
            else:
                publication_reason = "bootstrap"
            try:
                from reliquary.validator.training import (
                    current_lr_schedule_step,
                )

                lr_step = current_lr_schedule_step()
                entry = await self._checkpoint_store.publish(
                    checkpoint_n=next_n,
                    model=self.train_model,
                    profile_extra=(
                        {"lr_schedule_step": int(lr_step)}
                        if lr_step is not None
                        else None
                    ),
                )
                self._checkpoint_n = next_n
                self._trained_windows_since_publish = 0
                self._adaptive_publication_pending = False
                self._adaptive_publication_reason = None
                self.server.set_current_checkpoint(entry)
                # Refresh verify_model in-place so the next window's
                # batcher verifies miners against the just-published
                # checkpoint. In-place copy: no new allocation. The serial
                # beat guarantees no window is collecting right now, so the
                # in-place copy from train_model is exact (train == published)
                # and no miner sees a mid-collection checkpoint change.
                try:
                    self._refresh_verify_model_from_train(entry.revision)
                except (AttributeError, RuntimeError):
                    logger.exception(
                        "verify_model refresh failed; verify_model now "
                        "stale wrt checkpoint %d", entry.checkpoint_n,
                    )
                    raise
                if self.proof_scheduler is not None:
                    await asyncio.to_thread(
                        self._synchronize_proof_models,
                        entry.revision,
                    )
                if publication_retry_pending or epoch_finalize:
                    discarded = self._training_accumulator.reset()
                    post_publish_state = self._training_accumulator.snapshot()
                    accumulator_meta["post_publish_discarded"] = discarded
                    accumulator_meta["post_action"] = post_publish_state
                    self.server.set_training_accumulator_state(
                        post_publish_state
                    )
                    for _b in batchers.values():
                        _b.training_accumulator = accumulator_meta
                    logger.info(
                        "Published pending checkpoint %d; discarded %d "
                        "retained groups generated against its parent",
                        entry.checkpoint_n,
                        sum(discarded["counts"].values()),
                    )
                logger.info(
                    "Published checkpoint %d to %s@%s and refreshed "
                    "verify_model (reason=%s)",
                    entry.checkpoint_n,
                    entry.repo_id,
                    entry.revision[:12],
                    publication_reason,
                )
            except FatalProofPlaneError:
                raise
            except Exception:
                logger.exception(
                    "checkpoint publish or proof-replica refresh failed; "
                    "validator will remain closed until replicas are coherent"
                )
        elif trained:
            logger.info(
                "Skipping HF publish for window_n=%d "
                "(%d/%d trained windows since last publish)",
                window_n,
                self._trained_windows_since_publish,
                self._publish_every,
            )
        if detached_trainer:
            try:
                await self._detached_checkpoint_tick(
                    owns_routing=owns_routing, window_n=window_n,
                )
            except FatalProofPlaneError:
                raise
            except Exception:
                logger.exception(
                    "detached checkpoint tick failed for window %d",
                    window_n,
                )
        self.server.set_training_publish_state({
            "trained_windows_since_publish": (
                self._trained_windows_since_publish
            ),
            "detached_trainer": detached_trainer,
            "windows_since_checkpoint_swap": (
                self._windows_since_checkpoint_swap
                if detached_trainer else None
            ),
            "checkpoint_intake": (
                self._checkpoint_intake.snapshot()
                if detached_trainer and self._checkpoint_intake is not None
                else None
            ),
            "publish_interval": self._publish_every,
            "publication_pending": (
                self._trained_windows_since_publish >= self._publish_every
                or self._adaptive_publication_pending
            ),
            "adaptive_publication_pending": (
                self._adaptive_publication_pending
            ),
            "adaptive_publication_reason": (
                self._adaptive_publication_reason
            ),
        })

        try:
            await self._archive_window(
                batchers,
                sealed,
                late_drops=late_drops,
                server_reject_counts=server_reject_counts,
            )
        except Exception as exc:
            logger.exception("window archive failed")
            self._enqueue_aborted_window(
                failure_stage="archive_enqueue",
                failure_type=type(exc).__name__,
                batchers=batchers,
                late_drops=late_drops,
            )

        if owns_routing:
            self.server.set_active_batchers({})
            self._active_batchers = {}
        if owns_routing:
            self._set_state(WindowState.READY)

    async def _archive_window(
        self, batchers, sealed, late_drops=None, server_reject_counts=None,
    ) -> None:
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
                **dict(getattr(s, "ingress_observability", {}) or {}),
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
                "prompt_content_sha256": getattr(
                    s, "prompt_content_sha256", None
                ) or difficulty_meta.get("prompt_content_sha256"),
                "canonical_rank": meta.get("canonical_rank"),
                "accepted_into_pool": not rejected,
                "selected_for_batch": bool(meta.get("selected_for_batch", False)),
                "rewarded": bool(meta.get("rewarded", False)),
                "payment_source": meta.get("payment_source"),
                "batch_filled_reason": (
                    meta.get("selection_reason")
                    if not meta.get("selected_for_batch", False)
                    else None
                ),
                "reject_stage": getattr(s, "reject_stage", None),
                "reject_reason": getattr(s, "reason", None) if rejected else None,
                "reward_vector": getattr(s, "reward_vector", None),
                "truncated_count": getattr(s, "truncated_count", None),
                "unboxed_count": getattr(s, "unboxed_count", None),
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
                "difficulty_auction_proof_device": difficulty_meta.get(
                    "proof_device"
                ),
                "difficulty_auction_proof_duration_seconds": (
                    difficulty_meta.get("proof_duration_seconds")
                ),
                "difficulty_auction_forensic_sampled": difficulty_meta.get(
                    "forensic_sampled", False
                ),
                "difficulty_auction_forensic_passed": difficulty_meta.get(
                    "forensic_passed"
                ),
                "difficulty_auction_arrival_drand_round": difficulty_meta.get(
                    "arrival_drand_round"
                ),
                "difficulty_auction_arrival_round_source": difficulty_meta.get(
                    "arrival_round_source"
                ),
                "difficulty_auction_throughput_rank": difficulty_meta.get(
                    "throughput_rank"
                ),
                "difficulty_auction_tier": difficulty_meta.get("tier"),
                "difficulty_auction_tier_size": difficulty_meta.get(
                    "tier_size"
                ),
                "difficulty_auction_operator_id": difficulty_meta.get(
                    "operator_id"
                ),
                "difficulty_auction_operator_tiebreak": difficulty_meta.get(
                    "operator_tiebreak"
                ),
                "difficulty_auction_rank_entropy_source": difficulty_meta.get(
                    "rank_entropy_source"
                ),
                "difficulty_auction_precommit_arrival_ts": difficulty_meta.get(
                    "precommit_arrival_ts"
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

        # R20/R24: under v6 the seal path pays nothing and selects nothing
        # -- the spec removes the auction and the seal path IS the auction --
        # so BOTH the archive's per-hotkey emission and its ``batch`` entries
        # come from the assembler: the token split it computed, over exactly
        # the groups it paid. ``sealed_dict`` is empty under v6 (see
        # ``_seal_fill_closed_window``), and a weight-only validator replaying
        # the map from ``eos_tokens`` must divide over the same set.
        # Resolved BEFORE the per-env loop below, which reads its batches.
        fill_closed_assembler = None
        fill_closed_batches: dict[str, list] = {}
        if FILL_CLOSED_ENABLED:
            archived_window = int(getattr(first_batcher, "window_start", 0))
            fill_closed_assembler = self._fill_closed_assemblers.pop(
                archived_window, None
            )
            # Any assembler older than the window being archived belongs to
            # a window that was dropped before it ever reached here; nothing
            # will read it again.
            for stale in [
                key for key in self._fill_closed_assemblers
                if key < archived_window
            ]:
                self._fill_closed_assemblers.pop(stale, None)
            if fill_closed_assembler is None:
                # Not a silent zero: with no assembler this window pays
                # nobody and archives an empty batch, which is a wiring
                # failure, not an outcome.
                logger.error(
                    "Window %d: v6 archive found no batch assembler; the "
                    "window pays nothing and archives no batch",
                    archived_window,
                )
            else:
                # Idempotent (R16). The window's LAST batch is the partial
                # remainder ``close()`` forces out, and in serial mode
                # ``close()`` otherwise runs at the next window's open --
                # after this archive is written -- so that batch's pay and
                # its groups would never reach the archive a weight-only
                # validator replays. In pipelined mode this is a no-op.
                fill_closed_assembler.close()
                fill_closed_batches = fill_closed_assembler.paid_groups()

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
            rejection_snapshot_fn = getattr(
                type(batcher), "rejection_telemetry_snapshot", None
            )
            rejection_snapshot = (
                rejection_snapshot_fn(batcher)
                if callable(rejection_snapshot_fn)
                else {
                    "reject_counts": dict(
                        getattr(batcher, "reject_counts", {})
                    ),
                    "rejected_submissions": list(
                        getattr(batcher, "rejected_submissions", [])
                    ),
                    "grader_failures": dict(
                        getattr(batcher, "grader_failures", {})
                    ),
                }
            )
            env_obj = self.envs.get(env_name, self.env)
            env_batch, env_rewards = sealed_dict.get(env_name, ([], {}))
            # Parallel to ``env_batch``: which assembled batch paid each
            # group, or None off the v6 path (v4/v5 pay once per window,
            # so there is no batch to name and the field is not written).
            paid_batch_indices: list[int | None] = [None] * len(env_batch)
            if FILL_CLOSED_ENABLED:
                # R24: the paid set, not the auction's winners. R28: with
                # the batch index each group was paid in.
                paid = list(fill_closed_batches.get(env_name, ()))
                env_batch = [group for _index, group in paid]
                paid_batch_indices = [index for index, _group in paid]

            batched_keys = {(s.hotkey, s.prompt_idx) for s in env_batch}

            for paid_batch_index, s in zip(paid_batch_indices, env_batch):
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
                entry = {
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
                    # v6: completion tokens over genuinely EOS-terminated
                    # rollouts, produced once at admission. Additive field --
                    # a weight-only validator replaying per-token payment
                    # from the archive needs this to divide the same way the
                    # live validator did (see token_rewards.py).
                    "eos_tokens": int(getattr(s, "eos_tokens", 0) or 0),
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
                }
                if paid_batch_index is not None:
                    # R28: v6 pays per assembled batch, so a weight-only
                    # validator replaying the map from ``eos_tokens`` has to
                    # divide within each batch. Additive: v4/v5 entries and
                    # older readers are untouched.
                    entry["batch_index"] = int(paid_batch_index)
                batch_entries.append(entry)

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
                # Legacy archives may contain paid runners. Auction mode keeps
                # this path for schema compatibility but its alignment invariant
                # requires every paid group to be selected for training.
                if obs.get("rewarded") and s.rollout_hashes:
                    runner_entry["rollout_hashes"] = [h.hex() for h in s.rollout_hashes]
                runners_up.append(runner_entry)

            for r in rejection_snapshot["rejected_submissions"]:
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

            for k, v in rejection_snapshot["reject_counts"].items():
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
                    rejection_snapshot["grader_failures"]
                ).items()
            }
            grader_failures_by_environment[env_name] = env_grader_failures
            for reason, count in env_grader_failures.items():
                grader_failures[reason] = grader_failures.get(reason, 0) + count

        if FILL_CLOSED_ENABLED:
            combined_rewards = (
                dict(fill_closed_assembler.reward_map())
                if fill_closed_assembler is not None
                else {}
            )

        # Pipelined mode passes a stash-time snapshot: the live counter was
        # reset at the NEXT window's activation and holds its rejects, not
        # this window's.
        server_reject_summary = {
            str(reason): int(count)
            for reason, count in dict(
                server_reject_counts
                if server_reject_counts is not None
                else getattr(self.server, "_recent_reject_counts", {})
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
            "archive_schema_version": 2,
            "window_status": "completed",
            "window_start": first_batcher.window_start,
            "validator_hotkey": self.wallet.hotkey.ss58_address,  # provenance
            "randomness": first_batcher.randomness,
            "environment": env_names_list[0],   # legacy singular, kept for compat
            "environments": env_names_list,      # multi-env canonical field
            "force_seal_reason": getattr(first_batcher, "force_seal_reason", None),
            "force_seal_reason_by_environment": {
                env_name: getattr(env_batcher, "force_seal_reason", None)
                for env_name, env_batcher in batcher_dict.items()
            },
            "auction_seal_drain_by_environment": {
                env_name: dict(
                    getattr(env_batcher, "auction_seal_drain", {}) or {}
                )
                for env_name, env_batcher in batcher_dict.items()
            },
            "upload_precommit_conservation_by_environment": {
                env_name: (
                    env_batcher.upload_precommit_conservation()
                    if callable(
                        getattr(
                            type(env_batcher),
                            "upload_precommit_conservation",
                            None,
                        )
                    )
                    else {}
                )
                for env_name, env_batcher in batcher_dict.items()
            },
            "reward_alignment_by_environment": {
                env_name: dict(
                    getattr(env_batcher, "reward_alignment", {}) or {}
                )
                for env_name, env_batcher in batcher_dict.items()
            },
            "content_selection_by_environment": {
                env_name: dict(
                    getattr(env_batcher, "content_selection", {}) or {}
                )
                for env_name, env_batcher in batcher_dict.items()
            },
            "proof_scheduler": self._proof_scheduler_health_snapshot(),
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
            # Authoritative per-hotkey emission. In auction mode this is derived
            # only from the proven groups selected for training.
            "rewards_by_hotkey": combined_rewards,
            "rewarded_but_not_selected_by_hotkey": combined_rewarded_not_selected,
            "late_drops": {
                hk: dict(counts) for hk, counts in (
                    late_drops if late_drops is not None else self._late_drops
                ).items()
            },
        }
        await asyncio.to_thread(
            self._utility_telemetry.write_window,
            window=int(first_batcher.window_start),
            checkpoint_revision=str(
                getattr(first_batcher, "current_checkpoint_hash", "") or ""
            ),
            batchers=batcher_dict,
            selected_by_environment={
                env_name: list(sealed_dict.get(env_name, ([], {}))[0])
                for env_name in batcher_dict
            },
        )
        # Reset the in-memory counter for the next window. New events
        # arriving while this window's payload is uploading land in the
        # fresh dict and will appear in the next archive.
        if late_drops is None:
            # Serial mode: the ledger belonged to this window. Pipelined mode
            # received a seal-time snapshot; the live ledger now accumulates
            # the COLLECTING window's drops and must not be wiped.
            self._late_drops.clear()
        # Non-blocking archive: enqueue to disk and return immediately.
        # The background ``ArchiveQueue`` worker (started in run()) picks
        # this up and uploads via the same sync-boto3 path used in
        # storage.upload_window_dataset, with persistent retry-on-failure.
        # Main-loop window iteration is unblocked even if R2 is down for
        # hours, and queued payloads survive process restarts.
        from reliquary.infrastructure.archive_queue import get_archive_queue
        get_archive_queue().enqueue(first_batcher.window_start, archive)
        self._archive_enqueued_windows.add(int(first_batcher.window_start))

    def _write_training_payload(
        self,
        window_batches,
        window_n,
        checkpoint_revision,
        window_quarantine,
    ) -> None:
        """Enqueue this sealed window's detached-trainer payload (spec
        2026-08-21-detached-trainer-r2). Independent of in-process
        training so a shadow trainer can consume live data. Best-effort:
        a write failure degrades the trainer, never the window."""
        from reliquary.constants import WRITE_TRAINING_PAYLOADS

        if not WRITE_TRAINING_PAYLOADS:
            return
        if FILL_CLOSED_ENABLED:
            # C2. This enqueues under the BARE window number, but under v6
            # the journal key space is window * FILL_CLOSED_EMISSIONS_PER_
            # WINDOW + batch_index -- so a raw-key write for window w lands
            # in window w // EMISSIONS's first slot, on top of whatever the
            # assembler put there. The window's payloads were already
            # written, batch by batch, by FillClosedBatchAssembler under
            # their own encoded keys (R13); the seal path has nothing to
            # add and nowhere valid to put it.
            return
        from reliquary.shared.training_payload import encode_training_payload

        try:
            checkpoint_epoch = ValidationService._training_payload_epoch_binding(
                self,
                window_n,
            )
            data = encode_training_payload(
                window_batches,
                window_start=int(window_n),
                checkpoint_revision=str(checkpoint_revision),
                env_order=[name for name, _ in self.env_mix],
                window_quarantine=dict(window_quarantine or {}),
                checkpoint_epoch=checkpoint_epoch,
            )
            self._training_payload_queue_ref().enqueue_payload(
                int(window_n),
                data,
            )
            tombstoned_windows = getattr(
                self,
                "_training_tombstoned_windows",
                None,
            )
            if tombstoned_windows is not None:
                tombstoned_windows.discard(int(window_n))
        except Exception:
            logger.exception(
                "training payload write failed for window %s",
                window_n,
            )
            # The journal contract is "never advance on absence": a hole
            # with neither payload nor tombstone stalls the trainer
            # forever. A failed encode must still produce a marker.
            self._write_training_tombstone(
                window_n,
                "payload_encode",
                "PayloadEncodeError",
            )

    def _training_payload_epoch_binding(
        self,
        window_start: int,
        *,
        plan: EpochPlan | None = None,
    ):
        epoch_plan = plan or getattr(self, "_checkpoint_epoch_plan", None)
        if epoch_plan is None:
            return None
        offset = int(window_start) - epoch_plan.first_window
        if offset < 0 or offset >= epoch_plan.window_count:
            return None
        from reliquary.shared.training_payload import (
            CheckpointEpochTrainingBinding,
        )

        return CheckpointEpochTrainingBinding(
            epoch_id=epoch_plan.epoch_id,
            manifest_sha256=manifest_sha256(epoch_plan),
            training_run_id=TRAINING_RUN_ID,
            training_mode=epoch_plan.training_mode,
            first_window=epoch_plan.first_window,
            lane_offset=offset,
            window_count=epoch_plan.window_count,
            target_groups_per_environment_lane=(
                epoch_plan.target_groups_per_environment_lane
            ),
        )

    def _write_training_tombstone(
        self,
        window_start: int,
        failure_stage: str,
        failure_type: str,
        *,
        checkpoint_epoch_plan: EpochPlan | None = None,
    ) -> None:
        """Tombstone keeps the trainer's journal gapless: the trainer
        never advances on absence, only on an explicit marker."""
        from reliquary.constants import WRITE_TRAINING_PAYLOADS

        if not WRITE_TRAINING_PAYLOADS:
            return
        from reliquary.shared.training_payload import encode_tombstone

        checkpoint_epoch = ValidationService._training_payload_epoch_binding(
            self,
            window_start,
            plan=checkpoint_epoch_plan,
        )
        if FILL_CLOSED_ENABLED and checkpoint_epoch is None:
            self._write_fill_closed_window_tombstones(
                int(window_start), str(failure_stage), str(failure_type),
            )
            return
        try:
            if checkpoint_epoch is not None:
                self._training_tombstoned_windows.add(int(window_start))
            self._training_payload_queue_ref().enqueue_tombstone(
                int(window_start),
                encode_tombstone(
                    window_start=int(window_start),
                    failure_stage=str(failure_stage),
                    failure_type=str(failure_type),
                    checkpoint_epoch=checkpoint_epoch,
                ),
            )
        except Exception:
            logger.exception(
                "training tombstone write failed for window %s",
                window_start,
            )
            if checkpoint_epoch is not None:
                raise

    def _write_training_epoch_marker(
        self,
        plan: EpochPlan,
        *,
        status: str,
    ) -> None:
        from reliquary.constants import WRITE_TRAINING_PAYLOADS

        if not WRITE_TRAINING_PAYLOADS:
            return
        from reliquary.shared.training_payload import (
            encode_checkpoint_epoch_marker,
        )

        binding = self._training_payload_epoch_binding(
            plan.first_window,
            plan=plan,
        )
        if binding is None:
            raise RuntimeError("checkpoint epoch marker has no plan binding")
        try:
            data = encode_checkpoint_epoch_marker(binding, status=status)
            self._training_payload_queue_ref().enqueue_epoch_marker(
                plan.epoch_id,
                data,
            )
            self._training_tombstoned_windows.difference_update(
                range(
                    plan.first_window,
                    plan.first_window + plan.window_count,
                )
            )
        except Exception:
            logger.exception(
                "training epoch marker write failed for epoch %s",
                plan.epoch_id[:12],
            )
            raise

    def _abort_training_epoch_journal(
        self,
        plan: EpochPlan,
        *,
        failure_stage: str,
        failure_type: str,
    ) -> None:
        """Make an aborted epoch gapless before publishing its commit marker."""
        failures: list[int] = []
        for window in plan.windows:
            try:
                self._write_training_tombstone(
                    window.window_number,
                    failure_stage,
                    failure_type,
                    checkpoint_epoch_plan=plan,
                )
            except Exception:
                failures.append(window.window_number)
        if failures:
            raise RuntimeError(
                "checkpoint epoch abort journal is incomplete for windows "
                + ",".join(str(window) for window in failures)
            )
        self._write_training_epoch_marker(plan, status="aborted")

    def _training_payload_queue_ref(self):
        """Injected queue for tests; process singleton in production."""
        queue = getattr(self, "_training_payload_queue", None)
        if queue is not None:
            return queue
        from reliquary.infrastructure.training_payload_queue import (
            get_training_payload_queue,
        )
        return get_training_payload_queue()

    def _write_fill_closed_window_tombstones(
        self,
        window_start: int,
        failure_stage: str,
        failure_type: str,
    ) -> None:
        """C2: mark a failed v6 window across its WHOLE encoded key range.

        ``WindowJournal.next_entry`` walks the encoded key space one integer
        at a time and never advances on absence, only on an explicit marker
        (journal.py). A v6 window owns
        ``FILL_CLOSED_EMISSIONS_PER_WINDOW`` consecutive keys, so the seal
        path's single raw-key tombstone leaves that whole range unwritten
        and parks the trainer's cursor on the first hole forever -- it would
        never reach any later window either.

        Padding starts at the assembler's ``next_batch_index``: a window can
        abort AFTER emitting real batches, and those slots already hold
        payloads. This is the same rule ``FillClosedBatchAssembler.close()``
        applies at a normal close (R18), reached here for the windows that
        never get to close.
        """
        from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW
        from reliquary.infrastructure.training_payload_queue import (
            encoded_window_journal_key,
        )
        from reliquary.shared.training_payload import encode_tombstone

        assembler = getattr(self, "_fill_closed_assemblers", {}).get(
            int(window_start)
        )
        first_index = max(
            0, int(getattr(assembler, "next_batch_index", 0) or 0)
        )
        data = encode_tombstone(
            window_start=int(window_start),
            failure_stage=str(failure_stage),
            failure_type=str(failure_type),
        )
        for index in range(first_index, FILL_CLOSED_EMISSIONS_PER_WINDOW):
            key = encoded_window_journal_key(int(window_start), index)
            try:
                self._training_payload_queue_ref().enqueue_tombstone(key, data)
            except Exception:
                logger.exception(
                    "fill-closed window tombstone write failed for journal "
                    "key %s", key,
                )

    def _write_fill_closed_training_payload(
        self, key: int, data: bytes,
    ) -> None:
        """R13: enqueue_fn injected into this window's
        ``FillClosedBatchAssembler``. Independent of the seal-time
        ``_write_training_payload`` -- no checkpoint-epoch binding here
        (v6 windows are not epoch-final windows). The assembler itself
        now runs ``assess_training_batch`` per batch (R14) and only
        calls this for batches it did NOT quarantine; a quarantined
        batch goes to ``_write_fill_closed_training_tombstone`` instead.
        Best-effort and silent on ``WRITE_TRAINING_PAYLOADS`` off,
        matching the seal path's own kill-switch; a write failure here
        is logged and dropped rather than tombstoned, since -- unlike a
        sealed window -- there is no single terminal journal slot for a
        mid-window emission to mark failed against.
        """
        from reliquary.constants import WRITE_TRAINING_PAYLOADS

        if not WRITE_TRAINING_PAYLOADS:
            return
        try:
            self._training_payload_queue_ref().enqueue_payload(key, data)
        except Exception:
            logger.exception(
                "fill-closed training payload write failed for journal "
                "key %s", key,
            )

    def _write_fill_closed_training_tombstone(
        self, key: int, data: bytes,
    ) -> None:
        """R14: tombstone_fn injected into this window's
        ``FillClosedBatchAssembler``, called instead of ``_write_fill_
        closed_training_payload`` for a batch ``assess_training_batch``
        quarantines. Written under the batch's OWN encoded journal key
        (not the bare window) so ``WindowJournal.next_entry`` finds it at
        the same cursor position a payload would have occupied -- the
        trainer's cursor advances on this marker instead of stalling.
        Mirrors the payload sibling: best-effort, silent on
        ``WRITE_TRAINING_PAYLOADS`` off, and a write failure here is
        logged and dropped for the same reason (no single terminal slot
        to mark failed against mid-window).
        """
        from reliquary.constants import WRITE_TRAINING_PAYLOADS

        if not WRITE_TRAINING_PAYLOADS:
            return
        try:
            self._training_payload_queue_ref().enqueue_tombstone(key, data)
        except Exception:
            logger.exception(
                "fill-closed training tombstone write failed for journal "
                "key %s", key,
            )

    def _enqueue_aborted_window(
        self,
        *,
        failure_stage: str,
        failure_type: str,
        batchers: dict | None = None,
        late_drops: dict | None = None,
    ) -> None:
        """Durably record an opened window that could not complete.

        Tombstones carry no rewards or training data. They preserve archive
        continuity and make the failure auditable without exposing exception
        messages or partially validated miner payloads.

        ``batchers`` defaults to the collecting window's routing; pipelined
        callers pass the stashed window's batchers explicitly so the
        tombstone carries that window's metadata.
        """
        if batchers is None:
            batchers = self._active_batchers
        if not batchers:
            return
        first_batcher = next(iter(batchers.values()))
        window_start = int(first_batcher.window_start)
        if window_start in getattr(self, "_archive_enqueued_windows", set()):
            return
        iteration_stage = getattr(
            self, "_window_iteration_stage", "seal_train_archive"
        )
        if iteration_stage not in {
            "open",
            "drand_boundary",
            "randomness",
            "active",
            "seal_wait",
            "seal_train_archive",
            "archive_enqueue",
            "pipelined_stash",
            "pipelined_train_archive",
        } and not str(iteration_stage).startswith("checkpoint_epoch_"):
            return
        # The archive-dedup guard above also dedups this tombstone: one
        # marker per aborted window, matching the trainer's journal.
        self._write_training_tombstone(
            window_start, failure_stage, failure_type,
        )
        # An aborted window never reaches ``_archive_window``, the only other
        # place an assembler is popped; without this the dict keeps one dead
        # assembler per consecutive abort. Dropped AFTER the tombstones above,
        # which read its ``next_batch_index`` to know where padding starts.
        # ``getattr`` for the same reason the dedup set above uses it: test
        # stubs and partially-built services call this method too.
        getattr(self, "_fill_closed_assemblers", {}).pop(window_start, None)
        env_names = list(batchers)
        validator_hotkey = str(
            getattr(getattr(getattr(self, "wallet", None), "hotkey", None),
                    "ss58_address", "")
        )
        try:
            kl_reference = dict(self.kl_reference_state)
        except (AttributeError, TypeError, ValueError):
            kl_reference = {}
        archive = {
            "archive_schema_version": 2,
            "window_status": "aborted",
            "window_start": int(first_batcher.window_start),
            "validator_hotkey": validator_hotkey,
            "randomness": str(getattr(first_batcher, "randomness", "")),
            "environment": env_names[0] if env_names else "",
            "environments": env_names,
            "failure_stage": str(failure_stage),
            "failure_type": str(failure_type),
            "force_seal_reason": getattr(
                first_batcher, "force_seal_reason", None
            ),
            "force_seal_reason_by_environment": {
                name: getattr(batcher, "force_seal_reason", None)
                for name, batcher in batchers.items()
            },
            "proof_capacity_abort_by_environment": {
                name: {
                    "aborted": bool(
                        getattr(
                            batcher,
                            "proof_capacity_aborted",
                            False,
                        )
                    ),
                    "reason": getattr(
                        batcher,
                        "proof_capacity_abort_reason",
                        None,
                    ),
                }
                for name, batcher in batchers.items()
            },
            "proof_scheduler": self._proof_scheduler_health_snapshot(),
            "auction_seal_drain_by_environment": {
                name: dict(
                    getattr(batcher, "auction_seal_drain", {}) or {}
                )
                for name, batcher in batchers.items()
            },
            "upload_precommit_conservation_by_environment": {
                name: (
                    batcher.upload_precommit_conservation()
                    if callable(
                        getattr(
                            type(batcher),
                            "upload_precommit_conservation",
                            None,
                        )
                    )
                    else {}
                )
                for name, batcher in batchers.items()
            },
            "reward_alignment_by_environment": {
                name: dict(getattr(batcher, "reward_alignment", {}) or {})
                for name, batcher in batchers.items()
            },
            "batch": [],
            "runners_up": [],
            "rejected": [],
            "reject_summary": {},
            "server_reject_summary": {},
            "rewards_by_hotkey": {},
            "rewarded_but_not_selected_by_hotkey": {},
            "training_quarantine": {
                "quarantined": True,
                "reasons": ["aborted_window"],
                "metrics": {},
            },
            "training_accumulator": {
                "schema_version": 1,
                "trained": False,
                "blocked_reason": "aborted_window",
            },
            "training_kl_reference": kl_reference,
            "late_drops": {
                hotkey: dict(counts)
                for hotkey, counts in (
                    late_drops
                    if late_drops is not None
                    else getattr(self, "_late_drops", {})
                ).items()
            },
        }
        from reliquary.infrastructure.archive_queue import get_archive_queue

        get_archive_queue().enqueue(first_batcher.window_start, archive)
        self._archive_enqueued_windows.add(window_start)
        logger.error(
            "Window %d archived as aborted stage=%s error_type=%s",
            first_batcher.window_start,
            failure_stage,
            failure_type,
        )

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
                "upload_precommit_enabled": True,
                "submission_upload_grace_seconds": (
                    SUBMISSION_UPLOAD_GRACE_SECONDS
                ),
                "math_admission_workers": MATH_ADMISSION_WORKERS,
                "code_admission_workers": CODE_ADMISSION_WORKERS,
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
                "difficulty_auction_early_close_mode": (
                    AUCTION_EARLY_CLOSE_MODE
                ),
                "difficulty_auction_early_close_min_seconds": (
                    AUCTION_EARLY_CLOSE_MIN_SECONDS
                ),
                "difficulty_auction_primary_candidate_target": (
                    PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW
                ),
                "difficulty_auction_productive_candidate_limit": (
                    MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
                ),
                "difficulty_auction_challenger_capacity": (
                    MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
                    - PRIMARY_PROOF_GRADING_ATTEMPTS_PER_WINDOW
                ),
                "difficulty_auction_proof_attempt_limit": (
                    MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW
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
        self.server.configure_registration_gate()
        await self._refresh_registered_hotkeys(force=True, reason="startup")
        await self.server.start()
        await self._serve_axon_on_chain(subtensor)
        await self._apply_resume_from()                  # ← resume before bootstrap
        await self._bootstrap_state_from_external()
        if PROTOCOL_VERSION >= 3 and self.proof_scheduler is None:
            raise RuntimeError(
                "auction-v3 requires configured proof replicas; set "
                "RELIQUARY_PROOF_DEVICES and qualify capacity before launch"
            )
        if (
            PROTOCOL_VERSION >= 3
            and self.proof_capacity_qualification.get("qualified") is not True
        ):
            raise RuntimeError(
                "auction-v3 proof capacity is not qualified"
            )
        await self._ensure_proof_scheduler_ready()
        self._publish_window_preparation_state()
        await self._rebuild_cooldown_from_history()
        await self._restore_content_cooldown()
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
        from reliquary.constants import WRITE_TRAINING_PAYLOADS

        if WRITE_TRAINING_PAYLOADS:
            self._training_payload_worker_task = asyncio.create_task(
                self._training_payload_queue_ref().run_forever(),
                name="training_payload_queue_worker",
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
                    # Safe to clear even in pipelined mode: at loop top the
                    # in-flight windows (stashed + about-to-open) have not
                    # archived yet, so no live entry is lost.
                    self._archive_enqueued_windows.clear()
                    self._window_iteration_stage = "registration_refresh"
                    if self._candidate_window_n is not None:
                        self._set_window_preparation_stage(
                            "registration_refresh"
                        )
                    await self._refresh_registered_hotkeys(
                        reason="epoch_boundary"
                    )
                    self._window_iteration_stage = "proof_replica_refresh"
                    if self._candidate_window_n is not None:
                        self._set_window_preparation_stage(
                            "proof_replica_refresh"
                        )
                    await self._ensure_proof_scheduler_ready()
                    self._window_iteration_stage = "checkpoint_epoch"
                    checkpoint_epoch = await self._ensure_checkpoint_epoch_plan()
                    await self._wait_for_checkpoint_epoch_activation()
                    if checkpoint_epoch is not None:
                        await self._run_checkpoint_epoch()
                        self._windows_since_cooldown_snapshot += (
                            checkpoint_epoch.window_count
                        )
                        if (
                            self._windows_since_cooldown_snapshot
                            >= COOLDOWN_SNAPSHOT_INTERVAL_WINDOWS
                        ):
                            await self._snapshot_cooldown()
                            await self._snapshot_content_cooldown()
                            self._windows_since_cooldown_snapshot = 0
                        continue
                    self._window_iteration_stage = "open"
                    # v6.1 (R39): the next window opens once the trainer
                    # has CONSUMED the last one's batches. A no-op with
                    # the gate off, under an emergency freeze, and before
                    # the first v6 close (nothing armed) -- rotation stays
                    # byte-identical for v4/v5.
                    await self._wait_for_fill_closed_rotation()
                    self._open_window()
                    self._window_iteration_stage = "admission_pools"
                    self._set_window_preparation_stage("admission_pools")
                    await self.server.prepare_admission_pools(
                        self._active_batchers
                    )
                    self._window_iteration_stage = "drand_boundary"
                    await self._wait_for_next_drand_boundary()
                    self._window_iteration_stage = "randomness"
                    await self._set_window_randomness(subtensor)
                    self._activate_window()
                    self._window_iteration_stage = "active"
                    seal_wait_task = None
                    if self._gpu_backlog is not None:
                        # Pipelined mode: the previous window sealed last
                        # iteration; its GPU half (proofs + train + archive)
                        # runs NOW, hidden under the collection of the window
                        # we just activated. Nothing in it is speculative —
                        # the stashed window's ranking froze at its seal.
                        # The collecting window's seal-wait runs CONCURRENTLY
                        # so poll_deadline fires on the deadline and miners
                        # get their OPEN edge on time (see
                        # _seal_wait_and_close).
                        (
                            stashed_batchers,
                            stashed_n,
                            stashed_verify_task,
                            stashed_drops,
                            stashed_reject_counts,
                        ) = self._gpu_backlog
                        self._gpu_backlog = None
                        gpu_half_task = asyncio.create_task(
                            self._train_and_publish(
                                batchers=stashed_batchers,
                                window_n=stashed_n,
                                verify_task=stashed_verify_task,
                                late_drops=stashed_drops,
                                server_reject_counts=stashed_reject_counts,
                            ),
                            name=f"pipelined_gpu_half_{stashed_n}",
                        )
                        seal_wait_task = asyncio.create_task(
                            self._seal_wait_and_close(
                                early_close_ready=gpu_half_task.done
                            )
                        )
                        self._window_iteration_stage = (
                            "pipelined_train_archive"
                        )
                        try:
                            await gpu_half_task
                        except asyncio.CancelledError:
                            seal_wait_task.cancel()
                            try:
                                # Retrieve the outcome so a real seal-path
                                # error stored in the task is not silently
                                # dropped at GC during shutdown.
                                await seal_wait_task
                            except asyncio.CancelledError:
                                pass
                            except Exception:
                                logger.exception(
                                    "seal-wait task failed during "
                                    "cancellation"
                                )
                            raise
                        except FatalProofPlaneError:
                            # Proof plane is unrecoverable in-process; the
                            # outer handler tombstones and terminates for a
                            # supervisor restart. Give the stashed window its
                            # own tombstone first (the handler only sees the
                            # collecting window's routing).
                            try:
                                self._enqueue_aborted_window(
                                    failure_stage="pipelined_train_archive",
                                    failure_type="FatalProofPlaneError",
                                    batchers=stashed_batchers,
                                    late_drops=stashed_drops,
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to tombstone stashed window %d",
                                    stashed_n,
                                )
                            seal_wait_task.cancel()
                            try:
                                await seal_wait_task
                            except asyncio.CancelledError:
                                if not seal_wait_task.cancelled():
                                    # The CancelledError hit OUR await (an
                                    # external shutdown racing the fatal),
                                    # not the task we cancelled — propagate
                                    # the cancellation, not the fatal.
                                    raise
                            except Exception:
                                logger.exception(
                                    "seal-wait task failed during fatal "
                                    "teardown"
                                )
                            raise
                        except Exception:
                            # One incident, one window: the stashed window is
                            # tombstoned and forfeited, but the COLLECTING
                            # window is untouched (its routing and server
                            # state live outside this half), so the iteration
                            # continues to its seal. If the failure was a
                            # persistent GPU fault, that window's own half
                            # will surface it as a separate incident.
                            logger.exception(
                                "Stashed window %d GPU half failed; "
                                "tombstoning it and continuing with the "
                                "collecting window",
                                stashed_n,
                            )
                            try:
                                self._enqueue_aborted_window(
                                    failure_stage="pipelined_train_archive",
                                    failure_type="PipelinedTrainFailure",
                                    batchers=stashed_batchers,
                                    late_drops=stashed_drops,
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to tombstone stashed window %d",
                                    stashed_n,
                                )
                        except BaseException:
                            # SystemExit/KeyboardInterrupt: don't orphan the
                            # seal-wait task on the way out.
                            seal_wait_task.cancel()
                            raise
                    self._window_iteration_stage = "seal_wait"
                    if seal_wait_task is not None:
                        seal_reason = await seal_wait_task
                    else:
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
                    if FILL_CLOSED_ENABLED:
                        # R39: arm the next open's consumption gate while
                        # this window's assembler is still the current one
                        # (``_train_and_publish`` pops it below) and its
                        # ``next_batch_index`` still counts only the real
                        # mid-window emissions, before ``close()`` pads.
                        self._arm_fill_closed_rotation_gate()

                    from reliquary.constants import PIPELINED_WINDOWS

                    if PIPELINED_WINDOWS and self._publication_due_next_half():
                        logger.info(
                            "Window %d: publication due next half — running "
                            "this window on the SERIAL path",
                            self._window_n,
                        )
                    if PIPELINED_WINDOWS and not self._publication_due_next_half():
                        # Stash the sealed window; its GPU half runs at the
                        # top of the next iteration, after the next window's
                        # collection has opened. State is frozen at seal, so
                        # waiting costs nothing but latency.
                        self._window_iteration_stage = "pipelined_stash"
                        # Capture everything the GPU half will need so the
                        # next window's open cannot alias it: the beacon
                        # verify task belongs to THIS window (the next open
                        # will overwrite self._verify_task), and the
                        # late-drop ledger up to this seal belongs to this
                        # window's archive.
                        stashed_drops = dict(self._late_drops)
                        self._late_drops.clear()
                        self._gpu_backlog = (
                            dict(self._active_batchers),
                            self._window_n,
                            self._verify_task,
                            stashed_drops,
                            dict(getattr(
                                self.server, "_recent_reject_counts", {},
                            )),
                        )
                        self._verify_task = None
                        # Release routing and FSM exactly as the serial path
                        # does at end-of-window: the window is sealed (server
                        # hard-rejects arrivals), miners polling /state see
                        # READY instead of a stale OPEN, and the loop
                        # handlers' default-batchers tombstone can no longer
                        # alias the stashed window (which the salvage path
                        # owns from here).
                        self.server.set_active_batchers({})
                        self._active_batchers = {}
                        self._set_state(WindowState.READY)
                    else:
                        self._window_iteration_stage = "seal_train_archive"
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
                        await self._snapshot_content_cooldown()
                        self._windows_since_cooldown_snapshot = 0

                    # set_weights is owned by a concurrent WeightOnlyValidator
                    # task running off the same R2 archives; no need to do it
                    # here. The trainer is purely about training + uploads.
                except asyncio.CancelledError:
                    raise
                except CheckpointEpochExecutionError:
                    logger.exception(
                        "Checkpoint epoch stopped after partial execution; "
                        "terminating for a clean checkpoint reload"
                    )
                    self.server.set_active_batchers({})
                    self._active_batchers = {}
                    self._set_state(WindowState.READY)
                    raise
                except FatalProofPlaneError:
                    logger.exception(
                        "Fatal proof-plane failure; terminating for "
                        "supervisor restart"
                    )
                    try:
                        # Per-window dedup happens inside the helper.
                        self._enqueue_aborted_window(
                            failure_stage=self._window_iteration_stage,
                            failure_type="FatalProofPlaneError",
                        )
                    except Exception:
                        logger.exception(
                            "Failed to enqueue fatal proof tombstone"
                        )
                    if self._gpu_backlog is not None:
                        # The proof plane is unrecoverable, so the sealed
                        # backlog cannot be salvaged — but it MUST leave a
                        # tombstone: without one, restart's max(R2 windows)
                        # can advance past it (silent permanent archive gap)
                        # or re-collect under the same number.
                        (
                            _fb, _fn, _fvt, _fd, _frc,
                        ) = self._gpu_backlog
                        self._gpu_backlog = None
                        logger.error(
                            "Tombstoning pipelined backlog window %d on "
                            "fatal proof-plane failure (unpaid)",
                            _fn,
                        )
                        # The fatal may fire in a non-allowlisted stage
                        # (e.g. proof_replica_refresh); set the truthful
                        # pipelined stage so the tombstone is not skipped.
                        self._window_iteration_stage = (
                            "pipelined_train_archive"
                        )
                        try:
                            self._enqueue_aborted_window(
                                failure_stage="pipelined_train_archive",
                                failure_type="FatalProofPlaneError",
                                batchers=_fb,
                                late_drops=_fd,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to tombstone backlog window %d", _fn,
                            )
                    self.server.set_active_batchers({})
                    self._active_batchers = {}
                    self._set_state(WindowState.READY)
                    raise
                except Exception as exc:
                    logger.exception("Window iteration failed")
                    try:
                        self._enqueue_aborted_window(
                            failure_stage=self._window_iteration_stage,
                            failure_type=type(exc).__name__,
                        )
                    except Exception:
                        logger.exception("Failed to enqueue aborted-window tombstone")
                    self._rollback_preopen_window(exc)
                    # Reset to READY so the next iteration doesn't spin on error state.
                    if self._gpu_backlog is not None:
                        # The backlog is non-None only BEFORE the stashed GPU
                        # half runs (open/randomness/activate stages), so the
                        # failure was in the collecting window's open — the
                        # sealed backlog is intact and its miners are owed
                        # payment. Salvage it serially before resetting.
                        (
                            _sb, _sn, _svt, _sd, _src,
                        ) = self._gpu_backlog
                        self._gpu_backlog = None
                        logger.error(
                            "Iteration failed before the stashed GPU half "
                            "ran; salvaging sealed window %d serially",
                            _sn,
                        )
                        # Truthful stage for the salvage run — also keeps a
                        # salvage-failure tombstone inside the helper's stage
                        # allowlist (the failed stage may be outside it, e.g.
                        # registration_refresh).
                        self._window_iteration_stage = (
                            "pipelined_train_archive"
                        )
                        try:
                            await self._train_and_publish(
                                batchers=_sb,
                                window_n=_sn,
                                verify_task=_svt,
                                late_drops=_sd,
                                server_reject_counts=_src,
                            )
                        except asyncio.CancelledError:
                            raise
                        except FatalProofPlaneError:
                            # Do NOT downgrade a fatal to a salvage failure:
                            # the proof plane needs a supervisor restart, and
                            # opening another window against it would burn it
                            # too. Tombstone, then let the fatal escape.
                            logger.exception(
                                "Fatal proof-plane failure during salvage "
                                "of window %d; terminating for restart", _sn,
                            )
                            try:
                                self._enqueue_aborted_window(
                                    failure_stage="pipelined_train_archive",
                                    failure_type="FatalProofPlaneError",
                                    batchers=_sb,
                                    late_drops=_sd,
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to tombstone stashed window %d",
                                    _sn,
                                )
                            raise
                        except Exception:
                            logger.exception(
                                "Salvage of stashed window %d failed", _sn,
                            )
                            try:
                                self._enqueue_aborted_window(
                                    failure_stage="pipelined_train_archive",
                                    failure_type="PipelinedSalvageFailure",
                                    batchers=_sb,
                                    late_drops=_sd,
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to tombstone stashed window %d",
                                    _sn,
                                )
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
            await self._close_proof_scheduler()
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
            from huggingface_hub import HfApi
            repo_id = self._checkpoint_store.repo_id
            api = HfApi()
            commits = api.list_repo_commits(repo_id=repo_id)
            latest_n = -1
            latest_sha: str | None = None
            count = 0
            for c in commits:
                n = checkpoint_n_from_commit_title(getattr(c, "title", None))
                if n is None:
                    continue
                count += 1
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
        except Exception as exc:
            from reliquary.validator.checkpoint_profile import (
                CheckpointProfileMismatch,
            )

            if isinstance(exc, CheckpointProfileMismatch):
                raise
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

    @staticmethod
    def _validate_content_snapshot(
        snapshot: dict[str, Any],
        env_names: set[str],
    ) -> int:
        if snapshot.get("run_id") != TRAINING_RUN_ID:
            raise ValueError("content cooldown run id mismatch")
        if snapshot.get("complete") is not True:
            raise ValueError("content cooldown snapshot is incomplete")
        envs = snapshot.get("envs")
        if not isinstance(envs, dict) or set(envs) != env_names:
            raise ValueError("content cooldown environments are incomplete")
        return int(snapshot.get("snapshot_window", -1))

    def _top_up_content_cooldown_from_prompt_state(
        self,
        snapshot_window: int,
    ) -> int:
        """Resolve prompt-index cooldown entries newer than a content snapshot."""
        from reliquary.validator.prompt_content import (
            prompt_content_sha256,
            render_canonical_prompt,
        )

        resolved = 0
        for env_name, prompt_map in self._cooldown_per_env.items():
            env = self.envs[env_name]
            content_map = self._content_cooldown_per_env[env_name]
            content_state = content_map.export_state()
            for prompt_idx, selected_window in prompt_map.export_state().items():
                if int(selected_window) <= snapshot_window:
                    continue
                problem = env.get_problem(int(prompt_idx))
                rendered = render_canonical_prompt(
                    self.tokenizer, str(problem["prompt"])
                )
                digest = prompt_content_sha256(env_name, rendered)
                prior = content_state.get(digest, -1)
                if int(selected_window) > prior:
                    content_map.record_selected(digest, int(selected_window))
                    content_state[digest] = int(selected_window)
                resolved += 1
        return resolved

    async def _restore_content_cooldown(self) -> None:
        """Restore exact-content cooldown, deriving the first snapshot safely.

        The existing prompt-index snapshot is the complete source of selected
        prompts for a training run. On the first deployment, resolve every one
        through the pinned environment and tokenizer. Later restarts load the
        content snapshot and resolve only prompt entries newer than it.
        """
        env_names = set(self.envs)
        snapshot: dict[str, Any] | None = None
        source = "none"
        try:
            candidate = await storage.download_json(
                _content_cooldown_snapshot_key(TRAINING_RUN_ID)
            )
            if candidate is not None:
                self._validate_content_snapshot(candidate, env_names)
                snapshot = candidate
                source = "r2"
        except Exception:
            logger.exception("Failed to restore R2 content cooldown snapshot")

        if snapshot is None:
            try:
                candidate = await asyncio.to_thread(
                    _read_gzip_json,
                    _content_cooldown_local_path(TRAINING_RUN_ID),
                )
                if candidate is not None:
                    self._validate_content_snapshot(candidate, env_names)
                    snapshot = candidate
                    source = "local"
            except Exception:
                logger.exception("Failed to restore local content cooldown snapshot")

        snapshot_window = -1
        try:
            if snapshot is not None:
                snapshot_window = self._validate_content_snapshot(
                    snapshot, env_names
                )
                envs = snapshot["envs"]
                for env_name, content_map in self._content_cooldown_per_env.items():
                    content_map.import_state(envs[env_name])
            else:
                for content_map in self._content_cooldown_per_env.values():
                    content_map.import_state({})

            resolved = await asyncio.to_thread(
                self._top_up_content_cooldown_from_prompt_state,
                snapshot_window,
            )
            self._content_cooldown_health.update({
                "complete": True,
                "source": source if snapshot is not None else "prompt_backfill",
                "snapshot_window": max(snapshot_window, self._window_n),
                "counts_by_environment": {
                    name: len(content_map)
                    for name, content_map in self._content_cooldown_per_env.items()
                },
                "last_error_type": None,
            })
            logger.info(
                "Restored content cooldown source=%s snapshot_window=%d "
                "resolved=%d counts=%s",
                self._content_cooldown_health["source"],
                snapshot_window,
                resolved,
                self._content_cooldown_health["counts_by_environment"],
            )
            if not await self._snapshot_content_cooldown():
                raise RuntimeError(
                    "content cooldown bootstrap was not persisted locally"
                )
        except Exception as exc:
            self._content_cooldown_health.update({
                "complete": False,
                "source": source,
                "last_error_type": type(exc).__name__,
            })
            logger.exception(
                "Content cooldown restore incomplete; refusing to open windows"
            )
            raise RuntimeError("content cooldown restore incomplete") from exc

    async def _snapshot_content_cooldown(self) -> bool:
        """Persist locally before the best-effort R2 mirror.

        Returns whether a restart-safe local snapshot exists. The caller uses
        this as a startup gate; periodic callers may continue from the complete
        in-memory map while health reports a later disk failure.
        """
        window = self._window_n

        def _build() -> dict[str, Any]:
            return {
                "schema_version": 1,
                "run_id": TRAINING_RUN_ID,
                "snapshot_window": window,
                "complete": True,
                "envs": {
                    name: content_map.export_state()
                    for name, content_map in self._content_cooldown_per_env.items()
                },
            }

        snapshot = await asyncio.to_thread(_build)
        try:
            await asyncio.to_thread(
                _write_gzip_json_atomic,
                _content_cooldown_local_path(TRAINING_RUN_ID),
                snapshot,
            )
        except Exception as exc:
            self._content_cooldown_health.update({
                "last_snapshot_failure_ts": time.time(),
                "last_error_type": type(exc).__name__,
            })
            logger.exception(
                "Content cooldown local snapshot failed; durability unavailable"
            )
            return False

        now = time.time()
        try:
            uploaded = await storage.upload_json(
                _content_cooldown_snapshot_key(TRAINING_RUN_ID), snapshot
            )
            if not uploaded:
                raise RuntimeError("content cooldown upload returned false")
            self._content_cooldown_health.update({
                "complete": True,
                "source": "r2_and_local",
                "snapshot_window": window,
                "counts_by_environment": {
                    name: len(content_map)
                    for name, content_map in self._content_cooldown_per_env.items()
                },
                "last_snapshot_success_ts": now,
                "last_snapshot_failure_ts": None,
                "last_error_type": None,
            })
        except Exception as exc:
            self._content_cooldown_health.update({
                "complete": True,
                "source": "local",
                "snapshot_window": window,
                "counts_by_environment": {
                    name: len(content_map)
                    for name, content_map in self._content_cooldown_per_env.items()
                },
                "last_snapshot_success_ts": now,
                "last_snapshot_failure_ts": time.time(),
                "last_error_type": type(exc).__name__,
            })
            logger.exception(
                "Content cooldown R2 mirror failed; local snapshot is durable"
            )
        return True

    async def _rebuild_hashes_from_history(self) -> None:
        """Rebuild ``self._hash_set`` from the last HASH_DEDUP_RETENTION_WINDOWS
        archives. Horizon is independent of cooldown — see constants docstring.
        Compat path covers pre-feature archives (no ``hash`` field) by
        recomputing from ``tokens``.
        """
        try:
            current_window = self._window_n
            archives = await asyncio.wait_for(
                storage.list_recent_datasets(
                    current_window=current_window + 1,
                    n=HASH_DEDUP_RETENTION_WINDOWS,
                ),
                timeout=_STARTUP_HASH_REBUILD_TIMEOUT_SECONDS,
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
        except asyncio.TimeoutError:
            logger.warning(
                "Hash-set rebuild from history timed out after %.1fs; "
                "starting empty",
                _STARTUP_HASH_REBUILD_TIMEOUT_SECONDS,
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
        """Fetch public drand material for the next window seed.

        Returns ``(window_randomness, beacon_or_None)``. ``beacon`` is the
        raw drand beacon dict (``{round, randomness, signature, ...}``)
        when the drand path is active, so the caller can schedule a
        background bittensor_drand cross-check. ``None`` on the legacy
        mock path (no cross-check possible).

        The caller binds this public value to the candidate's preselected
        activation nonce before exposing the final seed at OPEN.
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

    async def _fetch_checkpoint_epoch_post_phase_beacon(
        self,
        *,
        after_round: int = 0,
        phase_close_round: int | None = None,
    ) -> tuple[int, BeaconBinding]:
        """Return the first verified beacon after a locally observed phase end."""
        plan = self._checkpoint_epoch_plan
        if plan is None or not self.use_drand:
            raise RuntimeError("checkpoint epoch seal requires drand")
        chain_info, observed_round = await self._checkpoint_epoch_drand_snapshot()
        close_round = (
            observed_round if phase_close_round is None else int(phase_close_round)
        )
        if (
            str(chain_info["name"]) != plan.epoch_beacon.chain
            or str(chain_info["hash"]) != plan.epoch_beacon.chain_hash
        ):
            raise RuntimeError("drand chain changed during checkpoint epoch")
        # Choose the target only after all phase data was frozen and persisted.
        # If persistence crossed a drand boundary, skip that now-known output.
        target_round = max(observed_round, close_round, int(after_round)) + 1
        while True:
            _, current_round = await self._checkpoint_epoch_drand_snapshot()
            if current_round >= target_round:
                break
            await asyncio.sleep(0.25)

        from reliquary.infrastructure.drand import get_beacon

        fetched = await asyncio.to_thread(
            get_beacon,
            round_id=str(target_round),
            use_drand=True,
            use_fallback=False,
        )
        beacon = BeaconBinding(
            source=str(fetched["source"]),
            chain=str(fetched["chain"]),
            chain_hash=str(fetched["chain_hash"]),
            round=int(fetched["round"]),
            randomness=str(fetched["randomness"]),
        )
        await self._verify_checkpoint_epoch_beacon(beacon)
        if beacon.round <= close_round:
            raise RuntimeError("checkpoint epoch phase beacon is not post-close")
        return close_round, beacon

    async def _fetch_checkpoint_epoch_admission_beacon(
        self,
        *,
        commitment_close_round: int,
    ) -> tuple[int, BeaconBinding]:
        plan = self._checkpoint_epoch_plan
        if plan is None:
            raise RuntimeError("checkpoint epoch plan is unavailable")
        return await self._fetch_checkpoint_epoch_post_phase_beacon(
            after_round=plan.epoch_beacon.round,
            phase_close_round=commitment_close_round,
        )

    async def _fetch_checkpoint_epoch_seal_beacon(
        self,
        *,
        after_round: int = 0,
    ) -> tuple[int, BeaconBinding]:
        return await self._fetch_checkpoint_epoch_post_phase_beacon(
            after_round=after_round
        )

    async def _fetch_seal_randomness(self) -> str:
        """Fetch post-seal drand with a bounded retry budget.

        The beacon strictly orders exact auction ties and keys the forensic
        sample. A total outage returns ``""`` after six seconds; ranking then
        uses the active profile's deterministic fallback rather than known
        window randomness.
        """
        if not self.use_drand:
            return ""
        from reliquary.infrastructure.drand import get_beacon, get_current_chain

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 6.0
        attempts = 0
        last_error: Exception | None = None
        while loop.time() < deadline:
            attempts += 1
            remaining = deadline - loop.time()

            def _fetch_once() -> str:
                chain_info = get_current_chain()
                drand_round = chain.compute_current_drand_round(
                    time.time(),
                    chain_info["genesis_time"],
                    chain_info["period"],
                )
                beacon = get_beacon(
                    round_id=str(drand_round), use_drand=True
                )
                return str(beacon["randomness"])

            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_fetch_once),
                    timeout=max(0.01, remaining),
                )
            except Exception as exc:
                last_error = exc
                remaining = deadline - loop.time()
                if remaining <= 0.0:
                    break
                await asyncio.sleep(min(0.5, remaining))

        logger.warning(
            "seal-randomness unavailable after %.1fs attempts=%d; "
            "auction ties use validator-arrival fallback and forensic sample "
            "is skipped error=%s",
            6.0,
            attempts,
            type(last_error).__name__ if last_error is not None else "unknown",
        )
        return ""
