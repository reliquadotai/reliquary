"""Process-safe auction admission preparation.

This module deliberately has no model or CUDA dependency. Production passes
raw signed reveals through these functions in spawned processes, then commits
only the compact validated result in the trainer process.
"""

from __future__ import annotations

import math
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from pydantic import ValidationError

from reliquary.constants import (
    BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION,
    BOOTSTRAP_SIGMA_MIN,
    BFT_ANSWER_BUDGET,
    BFT_THINKING_BUDGET,
    GRADER_EVAL_TIMEOUT_SECONDS,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MAX_TRUNCATED_PER_SUBMISSION,
    SIGMA_MIN,
)
from reliquary.environment.grader_client import (
    GraderClient,
    GraderInfrastructureError,
)
from reliquary.environment.opencodeinstruct import _extract_python
from reliquary.environment.openmathinstruct import _compute_omi_reward
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
from reliquary.validator.boxed_integrity import has_malformed_final_answer
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


@dataclass(frozen=True)
class AdmissionProblemMaterials:
    problem: dict[str, Any]
    rendered_prompt: str
    code_cases: list[dict[str, Any]] | None = None


@dataclass
class PreparedSubmission:
    request: BatchSubmissionRequest | None
    completion_texts: list[str]
    rewards: list[float]
    rollout_hashes: list[bytes]
    selection_digest: bytes | None
    reject_reason: RejectReason | None = None
    reject_stage: str | None = None
    grader_failure_reason: str | None = None
    legacy_merkle_status: str | None = None
    body_parse_ms: float = 0.0
    preparation_ms: float = 0.0
    reward_grading_ms: float = 0.0
    timed_out: bool = False


class _AdmissionTimeout(TimeoutError):
    pass


_WORKER_TOKENIZER: Any | None = None


def initialize_admission_worker(tokenizer_json: str | None = None) -> None:
    """Keep spawned admission children CPU-only and below control-plane priority."""
    global _WORKER_TOKENIZER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    try:
        os.nice(5)
    except OSError:
        pass
    if tokenizer_json:
        from tokenizers import Tokenizer

        _WORKER_TOKENIZER = Tokenizer.from_str(tokenizer_json)


def admission_worker_ready() -> int:
    """Small warm-up task that proves a spawned child ran its initializer."""
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("admission worker tokenizer unavailable")
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
    if context.environment != "openmathinstruct":
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


def _termination_reject(
    request: BatchSubmissionRequest,
    context: AdmissionContext,
) -> RejectReason | None:
    eos_ids = set(context.eos_token_ids)
    if not eos_ids:
        return None
    max_truncated = (
        BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION
        if context.bootstrap
        else MAX_TRUNCATED_PER_SUBMISSION
    )
    truncated = 0
    for rollout in request.rollouts:
        commit = rollout.commit
        tokens = list(commit.get("tokens") or [])
        meta = commit.get("rollout", {}) or {}
        try:
            prompt_length = int(meta.get("prompt_length", 0))
            completion_length = int(meta.get("completion_length", 0))
        except (TypeError, ValueError, OverflowError):
            return RejectReason.BAD_SCHEMA
        completion = tokens[prompt_length: prompt_length + completion_length]
        if not completion:
            return RejectReason.BAD_SCHEMA
        eos_positions = [
            index
            for index, token in enumerate(completion)
            if int(token) in eos_ids
        ]
        if eos_positions:
            if len(eos_positions) > 1 or eos_positions[0] != len(completion) - 1:
                return RejectReason.BAD_TERMINATION
            continue
        if (
            context.environment == "openmathinstruct"
            and _forced_cap_termination(meta)
        ):
            if not _force_span_valid(tokens, meta, context):
                return RejectReason.TOKEN_TAMPERED
            continue
        if _natural_cap_termination(tokens, meta, context):
            continue
        if prompt_length + completion_length < MAX_NEW_TOKENS_PROTOCOL_CAP:
            return RejectReason.BAD_TERMINATION
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

    def _grade(text: str) -> float:
        return float(
            client.evaluate_cases(
                _extract_python(text),
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


def score_and_finalize_submission(
    parsed: ParsedSubmission,
    materials: AdmissionRuntimeMaterials,
    context: AdmissionContext,
    deadline_monotonic: float,
) -> PreparedSubmission:
    """Bind the canonical prompt, grade rewards and validate group structure."""
    started = time.perf_counter()
    request = parsed.request
    if parsed.reject_reason is not None or request is None:
        return PreparedSubmission(
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
                        return PreparedSubmission(
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

            reward_started = time.perf_counter()
            try:
                if context.environment == "openmathinstruct":
                    computed = [
                        float(_compute_omi_reward(materials.problem, text))
                        for text in materials.completion_texts
                    ]
                    authoritative = False
                elif context.environment == "opencodeinstruct":
                    computed = _compute_code_rewards(
                        materials.completion_texts,
                        materials.code_cases or [],
                    )
                    authoritative = True
                else:
                    return PreparedSubmission(
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
            except GraderInfrastructureError:
                raise
            except Exception:
                return PreparedSubmission(
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
                    return PreparedSubmission(
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
                    return PreparedSubmission(
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
            if not _in_zone(rewards, bootstrap=context.bootstrap):
                return PreparedSubmission(
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

            for index, text in enumerate(materials.completion_texts):
                meta = request.rollouts[index].commit.get("rollout", {}) or {}
                malformed, _ = has_malformed_final_answer(
                    rewards[index],
                    text,
                    completion_length=int(meta.get("completion_length", 0)),
                    cap=MAX_NEW_TOKENS_PROTOCOL_CAP,
                )
                if malformed:
                    return PreparedSubmission(
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
                return PreparedSubmission(
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
                    if context.environment != "openmathinstruct":
                        meta["forced"] = False

            return PreparedSubmission(
                request=request,
                completion_texts=materials.completion_texts,
                rewards=rewards,
                rollout_hashes=parsed.rollout_hashes,
                selection_digest=parsed.selection_digest,
                body_parse_ms=parsed.body_parse_ms,
                preparation_ms=(
                    parsed.preparation_ms
                    + (time.perf_counter() - started) * 1000.0
                ),
                reward_grading_ms=reward_ms,
            )
    except _AdmissionTimeout:
        return PreparedSubmission(
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
        return PreparedSubmission(
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
        return PreparedSubmission(
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
            ),
            context,
            deadline_monotonic,
        )
    try:
        with _deadline(deadline_monotonic):
            if _WORKER_TOKENIZER is None:
                raise RuntimeError("admission worker tokenizer unavailable")
            canonical_prompt_tokens = list(
                _WORKER_TOKENIZER.encode(
                    materials.rendered_prompt,
                    add_special_tokens=False,
                ).ids
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
    prepared.legacy_merkle_status = parsed.legacy_merkle_status
    return prepared
