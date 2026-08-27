"""Process-safe auction admission preparation.

This module deliberately has no model or CUDA dependency. Production passes
raw signed reveals through these functions in spawned processes, then commits
only the compact validated result in the trainer process.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from pydantic import ValidationError

from reliquary.constants import (
    BOOTSTRAP_SIGMA_MIN,
    BFT_ANSWER_BUDGET,
    BFT_THINKING_BUDGET,
    GRADER_EVAL_TIMEOUT_SECONDS,
    MATH_ANSWER_FORMAT,
    ROBUST_TRUNCATION_UTILITY_ENABLED,
    SIGMA_MIN,
    episode_limits_for_environment,
    max_new_tokens_for_environment,
    max_truncated_for_environment,
)
from reliquary.environment.grader_client import (
    GraderClient,
    GraderInfrastructureError,
)
from reliquary.environment.opencodeinstruct import _entry_function_name, _extract_python
from reliquary.environment.openmathinstruct import _compute_omi_reward
from reliquary.environment.registry import get_environment_spec
from reliquary.protocol.legacy_merkle import legacy_submission_merkle_matches
from reliquary.protocol.signatures import (
    verify_commit_signature,
    verify_envelope_signature,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    CommitModel,
    RejectReason,
)
from reliquary.validator.boxed_integrity import (
    has_malformed_final_answer,
    is_missing_final_answer_box,
)
from reliquary.validator.dedup import compute_rollout_hash
from reliquary.validator.rollout_patterns import detect_opposite_reward_clones
from reliquary.validator.selection_digest import (
    compute_rollouts_selection_digest,
)


@dataclass(frozen=True)
class AdmissionReceiptBinding:
    miner_hotkey: str
    prompt_idx: int
    window_start: int
    merkle_root: str
    checkpoint_hash: str
    environment: str
    payload_bytes: int
    drand_round: int
    protocol_version: int
    nonce: str
    generation_profile_id: str = ""


@dataclass(frozen=True)
class AdmissionContext:
    randomness: str
    environment: str
    vocab_size: int | None
    max_sequence_length: int
    eos_token_ids: tuple[int, ...]
    canonical_force_ids: tuple[int, ...]
    think_close_ids: tuple[int, ...]
    bootstrap: bool
    enforce_envelope_signature: bool
    enforce_legacy_merkle: bool


@dataclass
class ParsedSubmission:
    request: BatchSubmissionRequest | None
    rollout_hashes: list[bytes]
    selection_digest: bytes | None
    reject_reason: RejectReason | None = None
    reject_stage: str | None = None
    legacy_merkle_status: str | None = None
    body_parse_ms: float = 0.0
    preparation_ms: float = 0.0
    timed_out: bool = False


@dataclass(frozen=True)
class AdmissionRuntimeMaterials:
    canonical_prompt_tokens: list[int] | None
    problem: dict[str, Any]
    completion_texts: list[str]
    code_cases: list[dict[str, Any]] | None = None
    reward_materials: Any = None

    @property
    def effective_reward_materials(self) -> Any:
        return (
            self.reward_materials
            if self.reward_materials is not None
            else self.code_cases
        )


@dataclass(frozen=True)
class AdmissionProblemMaterials:
    problem: dict[str, Any]
    rendered_prompt: str
    code_cases: list[dict[str, Any]] | None = None
    reward_materials: Any = None

    @property
    def effective_reward_materials(self) -> Any:
        return (
            self.reward_materials
            if self.reward_materials is not None
            else self.code_cases
        )


@dataclass
class PreparedSubmission:
    request: BatchSubmissionRequest | None
    completion_texts: list[str]
    rewards: list[float]
    rollout_hashes: list[bytes]
    selection_digest: bytes | None
    prompt_content_sha256: str = ""
    target_content_sha256: str = ""
    reject_reason: RejectReason | None = None
    reject_stage: str | None = None
    grader_failure_reason: str | None = None
    legacy_merkle_status: str | None = None
    body_parse_ms: float = 0.0
    preparation_ms: float = 0.0
    reward_grading_ms: float = 0.0
    timed_out: bool = False
    # True only after every rollout prefix matched the validator-rendered
    # canonical prompt.  The parent process uses this authenticated signal to
    # clear prompt-mismatch compatibility strikes even when a later, unrelated
    # admission check rejects the group.
    prompt_binding_verified: bool = False
    # Rollouts that ran to the protocol cap without terminating. Carried to the
    # auction so the group can be valued conservatively (a truncated rollout has
    # no gradeable answer, and a miner can create one on purpose).
    truncated_count: int = 0
    truncated_index: int | None = None
    # Math completions with no box score zero, but are uncertain for auction
    # eligibility so deleting a box cannot manufacture useful variance.
    unboxed_count: int = 0
    attainable_rewards: tuple[float, ...] = ()
    robust_utility: float | None = None
    task_family: str | None = None
    generator_version: str | None = None
    operation_id: str | None = None
    difficulty: int | None = None


class _AdmissionTimeout(TimeoutError):
    pass


_WORKER_TOKENIZER: Any | None = None


def _guard_bittensor_queue_listener_eof_in_child() -> bool:
    """Make bittensor's admission-child QueueListener exit cleanly on EOF.

    Importing the signature verifier imports bittensor, whose global logger
    starts a multiprocessing queue listener.  ``ProcessPoolExecutor`` retires
    admission children after a bounded number of tasks; Python's process
    teardown can close the queue pipe before that listener stops, producing a
    noisy ``QueueListener ... EOFError`` thread traceback.

    Do not close or join bittensor's queue here: doing so in a spawn initializer
    can stall process-pool startup.  QueueListener treats ``None`` as its stop
    sentinel, so translating teardown-only EOF/OSError to ``None`` preserves
    normal logging while making retirement graceful.  The parent validator
    logger is untouched.
    """

    if multiprocessing.parent_process() is None:
        return False
    try:
        import bittensor as bt

        logging_machine = getattr(bt, "logging", None)
        listener = getattr(logging_machine, "_listener", None)
        dequeue = getattr(listener, "dequeue", None)
        if listener is None or not callable(dequeue):
            return False
        if getattr(listener, "_reliquary_eof_guarded", False):
            return True

        def _safe_dequeue(block: bool) -> Any:
            try:
                return dequeue(block)
            except (EOFError, OSError):
                return None

        listener.dequeue = _safe_dequeue
        listener._reliquary_eof_guarded = True
        return True
    except Exception:
        # Logging cleanup must never prevent an admission worker from starting.
        return False


def initialize_admission_worker(tokenizer_json: str | None = None) -> None:
    """Keep spawned admission children CPU-only and below control-plane priority."""
    global _WORKER_TOKENIZER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    _guard_bittensor_queue_listener_eof_in_child()
    try:
        os.nice(5)
    except OSError:
        pass
    if tokenizer_json:
        from tokenizers import Tokenizer

        _WORKER_TOKENIZER = Tokenizer.from_str(tokenizer_json)


def admission_worker_ready(hold_seconds: float = 0.0) -> int:
    """Small warm-up task that proves a spawned child ran its initializer."""
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("admission worker tokenizer unavailable")
    if hold_seconds > 0.0:
        time.sleep(float(hold_seconds))
    return os.getpid()


@contextmanager
def _deadline(deadline_monotonic: float) -> Iterator[None]:
    remaining = float(deadline_monotonic) - time.monotonic()
    if remaining <= 0.0:
        raise _AdmissionTimeout("admission worker deadline exceeded")

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise _AdmissionTimeout("admission worker deadline exceeded")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _reject_parsed(
    reason: RejectReason,
    stage: str,
    *,
    request: BatchSubmissionRequest | None = None,
    body_parse_ms: float = 0.0,
    preparation_started: float,
    legacy_merkle_status: str | None = None,
    rollout_hashes: list[bytes] | None = None,
    selection_digest: bytes | None = None,
    timed_out: bool = False,
) -> ParsedSubmission:
    return ParsedSubmission(
        request=request,
        rollout_hashes=list(rollout_hashes or ()),
        selection_digest=selection_digest,
        reject_reason=reason,
        reject_stage=stage,
        legacy_merkle_status=legacy_merkle_status,
        body_parse_ms=body_parse_ms,
        preparation_ms=max(
            0.0, (time.perf_counter() - preparation_started) * 1000.0
        ),
        timed_out=timed_out,
    )


def _binding_matches(
    request: BatchSubmissionRequest,
    binding: AdmissionReceiptBinding,
) -> bool:
    environments = {rollout.env_name for rollout in request.rollouts}
    return (
        request.miner_hotkey == binding.miner_hotkey
        and request.prompt_idx == binding.prompt_idx
        and request.window_start == binding.window_start
        and request.merkle_root.lower() == binding.merkle_root.lower()
        and request.checkpoint_hash == binding.checkpoint_hash
        and environments == {binding.environment}
        and request.drand_round == binding.drand_round
        and request.protocol_version == binding.protocol_version
        and request.generation_profile_id == binding.generation_profile_id
        and request.nonce == binding.nonce
    )


def _tokens_valid(
    tokens: list[int],
    *,
    vocab_size: int | None,
    max_sequence_length: int,
) -> bool:
    if not tokens or len(tokens) > max_sequence_length:
        return False
    if vocab_size is None:
        return all(isinstance(token, int) and token >= 0 for token in tokens)
    return all(
        isinstance(token, int) and 0 <= token < vocab_size for token in tokens
    )


def _force_span_valid(
    tokens: list[int],
    meta: dict[str, Any],
    context: AdmissionContext,
) -> bool:
    if not meta.get("forced"):
        return True
    # Mirror of verifier.validate_force_span: without BFT there is no
    # profile-sanctioned force span, so a forced claim is tampering.
    from reliquary.constants import BFT_ENABLED

    if not BFT_ENABLED:
        return False
    span = meta.get("force_span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    try:
        start, end = int(span[0]), int(span[1])
        prompt_length = int(meta.get("prompt_length", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    if not (prompt_length <= start < end <= len(tokens)):
        return False
    if start - prompt_length != BFT_THINKING_BUDGET:
        return False
    think_close = set(context.think_close_ids)
    if any(int(token) in think_close for token in tokens[prompt_length:start]):
        return False
    return list(tokens[start:end]) == list(context.canonical_force_ids)


def _forced_cap_termination(meta: dict[str, Any]) -> bool:
    if not meta.get("forced"):
        return False
    span = meta.get("force_span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    try:
        start, end = int(span[0]), int(span[1])
        completion_length = int(meta.get("completion_length", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    if end <= start:
        return False
    return completion_length == (
        BFT_THINKING_BUDGET + (end - start) + BFT_ANSWER_BUDGET
    )


def _natural_cap_termination(
    tokens: list[int],
    meta: dict[str, Any],
    context: AdmissionContext,
) -> bool:
    try:
        uses_math_bft = (
            get_environment_spec(context.environment).termination_policy
            == "math_bft"
        )
    except ValueError:
        uses_math_bft = False
    if not uses_math_bft:
        return False
    if meta.get("forced") or meta.get("force_span") not in (None, []):
        return False
    try:
        prompt_length = int(meta.get("prompt_length", 0))
        completion_length = int(meta.get("completion_length", 0))
    except (TypeError, ValueError):
        return False
    if completion_length != BFT_THINKING_BUDGET + BFT_ANSWER_BUDGET:
        return False
    if prompt_length + completion_length != len(tokens):
        return False
    phase_one = tokens[
        prompt_length: prompt_length + BFT_THINKING_BUDGET
    ]
    think_close = set(context.think_close_ids)
    return any(int(token) in think_close for token in phase_one)


def _classify_termination(rollout, context: AdmissionContext) -> str:
    """Classify one rollout's termination.

    Returns ``"ok"`` (terminated, or an accepted BFT/natural cap shape),
    ``"truncated"`` (no EOS, ran to the protocol cap), or a reject reason name
    (``"bad_schema"`` / ``"tampered"`` / ``"bad_termination"``). Shared by the
    admission gate and the truncated-rollout count so both read the same
    predicate.
    """
    eos_ids = set(context.eos_token_ids)
    commit = rollout.commit
    tokens = list(commit.get("tokens") or [])
    meta = commit.get("rollout", {}) or {}
    # Episode termination is an environment state transition, not a trailing
    # tokenizer EOS. The authoritative replay gate below validates it.
    if isinstance(meta.get("episode"), dict):
        return "ok"
    try:
        prompt_length = int(meta.get("prompt_length", 0))
        completion_length = int(meta.get("completion_length", 0))
    except (TypeError, ValueError, OverflowError):
        return "bad_schema"
    completion = tokens[prompt_length: prompt_length + completion_length]
    if not completion:
        return "bad_schema"
    eos_positions = [
        index
        for index, token in enumerate(completion)
        if int(token) in eos_ids
    ]
    if eos_positions:
        if len(eos_positions) > 1 or eos_positions[0] != len(completion) - 1:
            return "bad_termination"
        return "ok"
    try:
        uses_math_bft = (
            get_environment_spec(context.environment).termination_policy
            == "math_bft"
        )
    except ValueError:
        uses_math_bft = False
    if uses_math_bft and _forced_cap_termination(meta):
        if not _force_span_valid(tokens, meta, context):
            return "tampered"
        return "ok"
    if _natural_cap_termination(tokens, meta, context):
        return "ok"
    if (
        prompt_length + completion_length
        < max_new_tokens_for_environment(context.environment)
    ):
        return "bad_termination"
    return "truncated"


def count_truncated_rollouts(
    request: BatchSubmissionRequest,
    context: AdmissionContext,
) -> int:
    """How many rollouts ran to the protocol cap without terminating.

    Feeds the auction's conservative valuation: a truncated rollout has no
    gradeable answer, so the group is scored under the interpretation least
    favourable to the miner (see ``conservative_difficulty_score``). Counting
    only — admission acceptance is decided by ``_termination_reject``.
    """
    if not context.eos_token_ids:
        return 0
    return sum(
        1
        for rollout in request.rollouts
        if _classify_termination(rollout, context) == "truncated"
    )


def truncated_rollout_indices(
    request: BatchSubmissionRequest,
    context: AdmissionContext,
) -> tuple[int, ...]:
    if not context.eos_token_ids:
        return ()
    return tuple(
        index
        for index, rollout in enumerate(request.rollouts)
        if _classify_termination(rollout, context) == "truncated"
    )


def _termination_reject(
    request: BatchSubmissionRequest,
    context: AdmissionContext,
) -> RejectReason | None:
    eos_ids = set(context.eos_token_ids)
    if not eos_ids:
        return None
    max_truncated = max_truncated_for_environment(
        context.environment,
        bootstrap=context.bootstrap,
    )
    truncated = 0
    for rollout in request.rollouts:
        kind = _classify_termination(rollout, context)
        if kind == "bad_schema":
            return RejectReason.BAD_SCHEMA
        if kind == "tampered":
            return RejectReason.TOKEN_TAMPERED
        if kind == "bad_termination":
            return RejectReason.BAD_TERMINATION
        if kind == "truncated":
            truncated += 1
            if truncated > max_truncated:
                return RejectReason.BAD_TERMINATION
    return None


def parse_and_validate_submission(
    raw_body: bytes,
    binding: AdmissionReceiptBinding,
    context: AdmissionContext,
    deadline_monotonic: float,
) -> ParsedSubmission:
    """Parse and run every immutable structural/authenticity gate."""
    started = time.perf_counter()
    parse_ms = 0.0
    request: BatchSubmissionRequest | None = None
    try:
        with _deadline(deadline_monotonic):
            parse_started = time.perf_counter()
            try:
                request = BatchSubmissionRequest.model_validate_json(raw_body)
            except ValidationError:
                return _reject_parsed(
                    RejectReason.BAD_SCHEMA,
                    "schema",
                    preparation_started=started,
                )
            parse_ms = (time.perf_counter() - parse_started) * 1000.0
            request._payload_bytes = binding.payload_bytes

            if not _binding_matches(request, binding):
                return _reject_parsed(
                    RejectReason.PRECOMMIT_INVALID,
                    "upload_precommit",
                    request=request,
                    body_parse_ms=parse_ms,
                    preparation_started=started,
                )
            if context.enforce_envelope_signature and not verify_envelope_signature(
                miner_hotkey=request.miner_hotkey,
                window_start=request.window_start,
                prompt_idx=request.prompt_idx,
                merkle_root=request.merkle_root,
                checkpoint_hash=request.checkpoint_hash,
                drand_round=request.drand_round,
                randomness=context.randomness,
                nonce=request.nonce,
                protocol_version=request.protocol_version,
                generation_profile_id=request.generation_profile_id,
                envelope_signature=request.envelope_signature,
            ):
                return _reject_parsed(
                    RejectReason.BAD_ENVELOPE_SIGNATURE,
                    "envelope",
                    request=request,
                    body_parse_ms=parse_ms,
                    preparation_started=started,
                )

            legacy_status = "disabled"
            try:
                legacy_matches, _ = legacy_submission_merkle_matches(request)
                legacy_status = "match" if legacy_matches else "mismatch"
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                legacy_matches = False
                legacy_status = "error"
            if context.enforce_legacy_merkle and not legacy_matches:
                return _reject_parsed(
                    RejectReason.MERKLE_ROOT_MISMATCH,
                    "legacy_merkle",
                    request=request,
                    body_parse_ms=parse_ms,
                    preparation_started=started,
                    legacy_merkle_status=legacy_status,
                )
            request._legacy_merkle_verified = legacy_matches

            rollout_hashes: list[bytes] = []
            local_hashes: set[bytes] = set()
            for rollout in request.rollouts:
                try:
                    commit = CommitModel.model_validate(rollout.commit)
                except ValidationError:
                    return _reject_parsed(
                        RejectReason.BAD_SCHEMA,
                        "schema",
                        request=request,
                        body_parse_ms=parse_ms,
                        preparation_started=started,
                        legacy_merkle_status=legacy_status,
                    )
                tokens = list(commit.tokens)
                if tokens != list(rollout.tokens):
                    return _reject_parsed(
                        RejectReason.TOKENS_MISMATCH,
                        "token_invariant",
                        request=request,
                        body_parse_ms=parse_ms,
                        preparation_started=started,
                        legacy_merkle_status=legacy_status,
                    )
                if not _tokens_valid(
                    tokens,
                    vocab_size=context.vocab_size,
                    max_sequence_length=context.max_sequence_length,
                ):
                    return _reject_parsed(
                        RejectReason.BAD_TOKENS,
                        "tokens",
                        request=request,
                        body_parse_ms=parse_ms,
                        preparation_started=started,
                        legacy_merkle_status=legacy_status,
                    )
                if not verify_commit_signature(
                    rollout.commit, request.miner_hotkey
                ):
                    return _reject_parsed(
                        RejectReason.BAD_SIGNATURE,
                        "rollout_signature",
                        request=request,
                        body_parse_ms=parse_ms,
                        preparation_started=started,
                        legacy_merkle_status=legacy_status,
                    )
                claimed_randomness = (
                    (rollout.commit.get("beacon") or {}).get("randomness", "")
                )
                if claimed_randomness != context.randomness:
                    return _reject_parsed(
                        RejectReason.WRONG_RANDOMNESS,
                        "randomness",
                        request=request,
                        body_parse_ms=parse_ms,
                        preparation_started=started,
                        legacy_merkle_status=legacy_status,
                    )
                rollout_hash = compute_rollout_hash(tokens)
                if rollout_hash in local_hashes:
                    return _reject_parsed(
                        RejectReason.HASH_DUPLICATE,
                        "dedup",
                        request=request,
                        body_parse_ms=parse_ms,
                        preparation_started=started,
                        legacy_merkle_status=legacy_status,
                    )
                local_hashes.add(rollout_hash)
                rollout_hashes.append(rollout_hash)

            termination_reason = _termination_reject(request, context)
            if termination_reason is not None:
                stage = (
                    "force_span_preflight"
                    if termination_reason is RejectReason.TOKEN_TAMPERED
                    else "termination_preflight"
                )
                return _reject_parsed(
                    termination_reason,
                    stage,
                    request=request,
                    body_parse_ms=parse_ms,
                    preparation_started=started,
                    legacy_merkle_status=legacy_status,
                    rollout_hashes=rollout_hashes,
                    selection_digest=compute_rollouts_selection_digest(
                        request.rollouts
                    ),
                )

            return ParsedSubmission(
                request=request,
                rollout_hashes=rollout_hashes,
                selection_digest=compute_rollouts_selection_digest(
                    request.rollouts
                ),
                legacy_merkle_status=legacy_status,
                body_parse_ms=parse_ms,
                preparation_ms=(time.perf_counter() - started) * 1000.0,
            )
    except _AdmissionTimeout:
        return _reject_parsed(
            RejectReason.WORKER_DROPPED,
            "admission_timeout",
            request=request,
            body_parse_ms=parse_ms,
            preparation_started=started,
            timed_out=True,
        )
    except Exception:
        return _reject_parsed(
            RejectReason.WORKER_DROPPED,
            "admission_worker",
            request=request,
            body_parse_ms=parse_ms,
            preparation_started=started,
        )


def _reward_matches(actual: float, claimed: float) -> bool:
    return (
        math.isfinite(float(actual))
        and math.isfinite(float(claimed))
        and abs(float(actual) - float(claimed)) <= 1e-6
    )


def _in_zone(rewards: list[float], *, bootstrap: bool) -> bool:
    if len(rewards) < 2:
        return False
    mean = sum(rewards) / len(rewards)
    sigma = (
        sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    ) ** 0.5
    if sigma < 1e-8:
        return False
    return sigma >= (BOOTSTRAP_SIGMA_MIN if bootstrap else SIGMA_MIN)


def _compute_code_rewards(
    completion_texts: list[str],
    cases: list[dict[str, Any]],
) -> list[float]:
    client = GraderClient()
    # Must grade the same span as OpenCodeInstructEnvironment.compute_reward:
    # a divergence here rejects honest miners on reward_mismatch.
    entry_name = _entry_function_name(cases)

    def _grade(text: str) -> float:
        return float(
            client.evaluate_cases(
                _extract_python(text, entry_name=entry_name),
                cases,
                timeout_s=GRADER_EVAL_TIMEOUT_SECONDS,
            )
        )

    executor = ThreadPoolExecutor(max_workers=len(completion_texts))
    try:
        return list(executor.map(_grade, completion_texts))
    finally:
        # A process deadline must not block in context-manager shutdown while
        # a bounded grader call is still unwinding.
        executor.shutdown(wait=False, cancel_futures=True)


def _score_openmath_adapter(
    problem: dict[str, Any],
    completion_texts: list[str],
    reward_materials: Any = None,
) -> list[float]:
    del reward_materials
    return [
        float(_compute_omi_reward(problem, text))
        for text in completion_texts
    ]


def _score_opencode_adapter(
    problem: dict[str, Any],
    completion_texts: list[str],
    reward_materials: Any = None,
) -> list[float]:
    del problem
    return _compute_code_rewards(
        completion_texts,
        list(reward_materials or []),
    )


def score_and_finalize_submission(
    parsed: ParsedSubmission,
    materials: AdmissionRuntimeMaterials,
    context: AdmissionContext,
    deadline_monotonic: float,
) -> PreparedSubmission:
    """Bind the canonical prompt, grade rewards and validate group structure."""
    started = time.perf_counter()
    request = parsed.request
    prompt_binding_verified = False

    def result(**kwargs: Any) -> PreparedSubmission:
        return PreparedSubmission(
            prompt_binding_verified=prompt_binding_verified,
            **kwargs,
        )

    if parsed.reject_reason is not None or request is None:
        return result(
            request=request,
            completion_texts=[],
            rewards=[],
            rollout_hashes=parsed.rollout_hashes,
            selection_digest=parsed.selection_digest,
            reject_reason=parsed.reject_reason or RejectReason.BAD_SCHEMA,
            reject_stage=parsed.reject_stage or "schema",
            body_parse_ms=parsed.body_parse_ms,
            preparation_ms=parsed.preparation_ms,
            timed_out=parsed.timed_out,
        )

    reward_ms = 0.0
    try:
        with _deadline(deadline_monotonic):
            canonical = materials.canonical_prompt_tokens
            if canonical is not None:
                for rollout in request.rollouts:
                    meta = rollout.commit.get("rollout", {}) or {}
                    prompt_length = int(meta.get("prompt_length", 0))
                    if list(rollout.commit.get("tokens", []))[
                        :prompt_length
                    ] != list(canonical):
                        return result(
                            request=request,
                            completion_texts=materials.completion_texts,
                            rewards=[],
                            rollout_hashes=parsed.rollout_hashes,
                            selection_digest=parsed.selection_digest,
                            reject_reason=RejectReason.PROMPT_MISMATCH,
                            reject_stage="prompt_binding",
                            body_parse_ms=parsed.body_parse_ms,
                            preparation_ms=(
                                parsed.preparation_ms
                                + (time.perf_counter() - started) * 1000.0
                            ),
                        )

                prompt_binding_verified = True

            reward_started = time.perf_counter()
            try:
                try:
                    environment_spec = get_environment_spec(
                        context.environment
                    )
                except ValueError:
                    return result(
                        request=request,
                        completion_texts=materials.completion_texts,
                        rewards=[],
                        rollout_hashes=parsed.rollout_hashes,
                        selection_digest=parsed.selection_digest,
                        reject_reason=RejectReason.WORKER_DROPPED,
                        reject_stage="unsupported_environment",
                        body_parse_ms=parsed.body_parse_ms,
                        preparation_ms=parsed.preparation_ms,
                    )
                if environment_spec.interaction_mode == "episode":
                    if _WORKER_TOKENIZER is None:
                        raise RuntimeError("episode replay tokenizer unavailable")
                    from reliquary.environment.agentic.replay import (
                        replay_tokenized_episode,
                    )

                    def encode_piece(text: str) -> list[int]:
                        encoded = _WORKER_TOKENIZER.encode(
                            text, add_special_tokens=False
                        )
                        return list(getattr(encoded, "ids", encoded))

                    def decode_piece(ids: list[int]) -> str:
                        return str(_WORKER_TOKENIZER.decode(
                            list(ids), skip_special_tokens=False
                        ))

                    computed = []
                    for rollout in request.rollouts:
                        raw_meta = rollout.commit.get("rollout", {}) or {}
                        episode = raw_meta.get("episode")
                        if not isinstance(episode, dict):
                            raise ValueError("episode metadata missing")
                        spans = tuple(
                            (int(span[0]), int(span[1]))
                            for span in episode.get("assistant_spans", [])
                        )
                        limits = episode_limits_for_environment(
                            context.environment
                        )
                        if limits is None:
                            raise ValueError("episode limits missing from profile")
                        (
                            max_action_tokens,
                            max_episode_tokens,
                            max_observation_bytes,
                        ) = limits
                        if len(rollout.commit["tokens"]) > max_episode_tokens:
                            raise ValueError("episode token budget exceeded")
                        if any(
                            end - start > max_action_tokens
                            for start, end in spans
                        ):
                            raise ValueError("episode action token budget exceeded")
                        trace = replay_tokenized_episode(
                            environment_spec.create(),
                            task_index=request.prompt_idx,
                            seed=int(episode["seed"]),
                            tokens=list(rollout.commit["tokens"]),
                            assistant_spans=spans,
                            decode=decode_piece,
                            encode=encode_piece,
                            max_episode_tokens=max_episode_tokens,
                            max_observation_bytes=max_observation_bytes,
                        )
                        reward = trace.reward
                        if reward is None:
                            raise ValueError("episode replay did not produce reward")
                        if trace.task_id != str(episode.get("task_id")):
                            raise ValueError("episode task binding mismatch")
                        if [action.to_wire() for action in trace.actions] != list(
                            episode.get("actions") or []
                        ):
                            raise ValueError("episode action binding mismatch")
                        if tuple(trace.tokens) != tuple(rollout.commit["tokens"]):
                            raise ValueError("episode canonical transcript mismatch")
                        if trace.assistant_spans != spans:
                            raise ValueError("episode assistant span mismatch")
                        if list(trace.observation_digests) != list(
                            episode.get("observation_digests") or []
                        ):
                            raise ValueError("episode observation mismatch")
                        if trace.termination_reason != str(
                            episode.get("termination_reason")
                        ):
                            raise ValueError("episode termination mismatch")
                        if reward.state_digest != str(episode.get("state_digest")):
                            raise ValueError("episode state digest mismatch")
                        if trace.trace_digest != str(episode.get("trace_digest")):
                            raise ValueError("episode trace digest mismatch")
                        rollout._validated_assistant_spans = spans
                        rollout._validated_episode_trace_digest = trace.trace_digest
                        computed.append(float(reward.reward))
                else:
                    computed = environment_spec.score_many(
                        materials.problem,
                        materials.completion_texts,
                        materials.effective_reward_materials,
                    )
                authoritative = (
                    environment_spec.validator_authoritative_reward
                )
            except GraderInfrastructureError:
                raise
            except Exception:
                return result(
                    request=request,
                    completion_texts=materials.completion_texts,
                    rewards=[],
                    rollout_hashes=parsed.rollout_hashes,
                    selection_digest=parsed.selection_digest,
                    reject_reason=RejectReason.REWARD_MISMATCH,
                    reject_stage="reward",
                    body_parse_ms=parsed.body_parse_ms,
                    preparation_ms=parsed.preparation_ms,
                )
            reward_ms = (time.perf_counter() - reward_started) * 1000.0

            for rollout, reward in zip(request.rollouts, computed, strict=True):
                if not math.isfinite(reward):
                    return result(
                        request=request,
                        completion_texts=materials.completion_texts,
                        rewards=[],
                        rollout_hashes=parsed.rollout_hashes,
                        selection_digest=parsed.selection_digest,
                        reject_reason=RejectReason.REWARD_MISMATCH,
                        reject_stage="reward",
                        body_parse_ms=parsed.body_parse_ms,
                        preparation_ms=parsed.preparation_ms,
                        reward_grading_ms=reward_ms,
                    )
                if authoritative:
                    rollout.reward = reward
                    meta = rollout.commit.get("rollout")
                    if isinstance(meta, dict):
                        meta["success"] = reward > 0.5
                        meta["total_reward"] = reward
                elif not _reward_matches(reward, rollout.reward):
                    return result(
                        request=request,
                        completion_texts=materials.completion_texts,
                        rewards=[],
                        rollout_hashes=parsed.rollout_hashes,
                        selection_digest=parsed.selection_digest,
                        reject_reason=RejectReason.REWARD_MISMATCH,
                        reject_stage="reward",
                        body_parse_ms=parsed.body_parse_ms,
                        preparation_ms=parsed.preparation_ms,
                        reward_grading_ms=reward_ms,
                    )

            rewards = [float(rollout.reward) for rollout in request.rollouts]
            truncated_indices = truncated_rollout_indices(request, context)
            unboxed_indices = (
                tuple(
                    index
                    for index, text in enumerate(materials.completion_texts)
                    if is_missing_final_answer_box(text)
                )
                if (
                    environment_spec.final_answer_policy == "boxed"
                    and MATH_ANSWER_FORMAT == "boxed"
                )
                else ()
            )
            uncertain_indices = tuple(
                dict.fromkeys((*truncated_indices, *unboxed_indices))
            )
            robust_utility: float | None = None
            attainable_rewards: tuple[float, ...] = ()
            if (
                ROBUST_TRUNCATION_UTILITY_ENABLED
                and uncertain_indices
            ):
                from reliquary.validator.difficulty_auction import (
                    fractional_reward_lattice,
                    robust_uncertain_reward_utility,
                )

                if environment_spec.attainable_rewards:
                    attainable_rewards = (
                        environment_spec.attainable_rewards
                    )
                else:
                    total_tests = max(
                        1,
                        len(materials.effective_reward_materials or ()),
                    )
                    attainable_rewards = fractional_reward_lattice(
                        total_tests
                    )
                robust_utility = robust_uncertain_reward_utility(
                    rewards,
                    sigma_min=(
                        BOOTSTRAP_SIGMA_MIN
                        if context.bootstrap
                        else SIGMA_MIN
                    ),
                    uncertain_indices=uncertain_indices,
                    attainable_rewards=attainable_rewards,
                )
            in_zone = (
                robust_utility > 0.0
                if robust_utility is not None
                else _in_zone(rewards, bootstrap=context.bootstrap)
            )
            if not in_zone:
                return result(
                    request=request,
                    completion_texts=materials.completion_texts,
                    rewards=rewards,
                    rollout_hashes=parsed.rollout_hashes,
                    selection_digest=parsed.selection_digest,
                    reject_reason=RejectReason.OUT_OF_ZONE,
                    reject_stage="zone",
                    body_parse_ms=parsed.body_parse_ms,
                    preparation_ms=parsed.preparation_ms,
                    reward_grading_ms=reward_ms,
                )

            if environment_spec.final_answer_policy == "boxed":
                for index, text in enumerate(materials.completion_texts):
                    meta = request.rollouts[index].commit.get("rollout", {}) or {}
                    malformed, _ = has_malformed_final_answer(
                        rewards[index],
                        text,
                        completion_length=int(meta.get("completion_length", 0)),
                        cap=max_new_tokens_for_environment(
                            context.environment
                        ),
                    )
                    if malformed:
                        return result(
                            request=request,
                            completion_texts=materials.completion_texts,
                            rewards=rewards,
                            rollout_hashes=parsed.rollout_hashes,
                            selection_digest=parsed.selection_digest,
                            reject_reason=RejectReason.MALFORMED_FINAL_ANSWER,
                            reject_stage="malformed_final_answer",
                            body_parse_ms=parsed.body_parse_ms,
                            preparation_ms=parsed.preparation_ms,
                            reward_grading_ms=reward_ms,
                        )

            clone_metrics = detect_opposite_reward_clones(
                materials.completion_texts, rewards
            )
            if clone_metrics.suspicious:
                return result(
                    request=request,
                    completion_texts=materials.completion_texts,
                    rewards=rewards,
                    rollout_hashes=parsed.rollout_hashes,
                    selection_digest=parsed.selection_digest,
                    reject_reason=RejectReason.DISTRIBUTION_SUSPICIOUS,
                    reject_stage="distribution",
                    body_parse_ms=parsed.body_parse_ms,
                    preparation_ms=parsed.preparation_ms,
                    reward_grading_ms=reward_ms,
                )

            for rollout in request.rollouts:
                meta = rollout.commit.get("rollout")
                if isinstance(meta, dict):
                    meta["truncated"] = False
                    if environment_spec.termination_policy != "math_bft":
                        meta["forced"] = False

            return result(
                request=request,
                completion_texts=materials.completion_texts,
                rewards=rewards,
                rollout_hashes=parsed.rollout_hashes,
                selection_digest=parsed.selection_digest,
                # Derived from the token stream, so it is unaffected by the
                # meta["truncated"] reset above.
                truncated_count=len(truncated_indices),
                truncated_index=(
                    truncated_indices[0] if truncated_indices else None
                ),
                unboxed_count=len(unboxed_indices),
                attainable_rewards=attainable_rewards,
                robust_utility=robust_utility,
                body_parse_ms=parsed.body_parse_ms,
                preparation_ms=(
                    parsed.preparation_ms
                    + (time.perf_counter() - started) * 1000.0
                ),
                reward_grading_ms=reward_ms,
            )
    except _AdmissionTimeout:
        return result(
            request=request,
            completion_texts=materials.completion_texts,
            rewards=[],
            rollout_hashes=parsed.rollout_hashes,
            selection_digest=parsed.selection_digest,
            reject_reason=RejectReason.WORKER_DROPPED,
            reject_stage="admission_timeout",
            body_parse_ms=parsed.body_parse_ms,
            preparation_ms=parsed.preparation_ms,
            reward_grading_ms=reward_ms,
            timed_out=True,
        )
    except GraderInfrastructureError as exc:
        return result(
            request=request,
            completion_texts=materials.completion_texts,
            rewards=[],
            rollout_hashes=parsed.rollout_hashes,
            selection_digest=parsed.selection_digest,
            reject_reason=RejectReason.WORKER_DROPPED,
            reject_stage="code_grader",
            grader_failure_reason=exc.reason,
            body_parse_ms=parsed.body_parse_ms,
            preparation_ms=parsed.preparation_ms,
            reward_grading_ms=reward_ms,
        )
    except Exception:
        return result(
            request=request,
            completion_texts=materials.completion_texts,
            rewards=[],
            rollout_hashes=parsed.rollout_hashes,
            selection_digest=parsed.selection_digest,
            reject_reason=RejectReason.WORKER_DROPPED,
            reject_stage="admission_worker",
            body_parse_ms=parsed.body_parse_ms,
            preparation_ms=parsed.preparation_ms,
            reward_grading_ms=reward_ms,
        )


def materialize_and_score_submission(
    parsed: ParsedSubmission,
    materials: AdmissionProblemMaterials,
    context: AdmissionContext,
    deadline_monotonic: float,
) -> PreparedSubmission:
    """Decode and grade entirely inside the isolated admission process."""
    request = parsed.request
    if parsed.reject_reason is not None or request is None:
        return score_and_finalize_submission(
            parsed,
            AdmissionRuntimeMaterials(
                canonical_prompt_tokens=None,
                problem=materials.problem,
                completion_texts=[],
                code_cases=materials.code_cases,
                reward_materials=materials.reward_materials,
            ),
            context,
            deadline_monotonic,
        )
    try:
        with _deadline(deadline_monotonic):
            if _WORKER_TOKENIZER is None:
                raise RuntimeError("admission worker tokenizer unavailable")
            encoded_prompt = _WORKER_TOKENIZER.encode(
                materials.rendered_prompt,
                add_special_tokens=False,
            )
            canonical_prompt_tokens = list(
                getattr(encoded_prompt, "ids", encoded_prompt)
            )
            completion_texts = []
            for rollout in request.rollouts:
                meta = rollout.commit.get("rollout", {}) or {}
                prompt_length = int(meta.get("prompt_length", 0))
                completion_texts.append(
                    _WORKER_TOKENIZER.decode(
                        list(rollout.commit["tokens"])[prompt_length:],
                        skip_special_tokens=False,
                    )
                )
        return score_and_finalize_submission(
            parsed,
            AdmissionRuntimeMaterials(
                canonical_prompt_tokens=canonical_prompt_tokens,
                problem=materials.problem,
                completion_texts=completion_texts,
                code_cases=materials.code_cases,
                reward_materials=materials.reward_materials,
            ),
            context,
            deadline_monotonic,
        )
    except _AdmissionTimeout:
        return PreparedSubmission(
            request=request,
            completion_texts=[],
            rewards=[],
            rollout_hashes=parsed.rollout_hashes,
            selection_digest=parsed.selection_digest,
            reject_reason=RejectReason.WORKER_DROPPED,
            reject_stage="admission_timeout",
            body_parse_ms=parsed.body_parse_ms,
            preparation_ms=parsed.preparation_ms,
            timed_out=True,
        )
    except Exception:
        return PreparedSubmission(
            request=request,
            completion_texts=[],
            rewards=[],
            rollout_hashes=parsed.rollout_hashes,
            selection_digest=parsed.selection_digest,
            reject_reason=RejectReason.WORKER_DROPPED,
            reject_stage="admission_worker",
            body_parse_ms=parsed.body_parse_ms,
            preparation_ms=parsed.preparation_ms,
        )


def prepare_submission(
    raw_body: bytes,
    binding: AdmissionReceiptBinding,
    materials: AdmissionProblemMaterials,
    context: AdmissionContext,
    deadline_monotonic: float,
) -> PreparedSubmission:
    """Authoritative one-pass auction preparation entry point."""
    parsed = parse_and_validate_submission(
        raw_body,
        binding,
        context,
        deadline_monotonic,
    )
    prepared = materialize_and_score_submission(
        parsed,
        materials,
        context,
        deadline_monotonic,
    )
    from reliquary.validator.prompt_content import (
        prompt_content_sha256,
        target_content_sha256,
    )

    prepared.prompt_content_sha256 = prompt_content_sha256(
        context.environment,
        materials.rendered_prompt,
    )
    prepared.target_content_sha256 = target_content_sha256(
        context.environment,
        materials.problem,
        code_cases=materials.code_cases,
    )
    prepared.task_family = (
        str(materials.problem.get("task_family"))
        if materials.problem.get("task_family") is not None
        else None
    )
    prepared.generator_version = (
        str(materials.problem.get("generator_version"))
        if materials.problem.get("generator_version") is not None
        else None
    )
    prepared.operation_id = (
        str(materials.problem.get("operation_id"))
        if materials.problem.get("operation_id") is not None
        else None
    )
    try:
        prepared.difficulty = (
            int(materials.problem["difficulty"])
            if "difficulty" in materials.problem
            else None
        )
    except (TypeError, ValueError, OverflowError):
        prepared.difficulty = None
    prepared.legacy_merkle_status = parsed.legacy_merkle_status
    return prepared
