"""Arrival-neutral market primitives for experimental checkpoint epochs.

The module is deliberately dependency-light.  It separates two decisions that
the production window currently conflates:

* which miner-chosen prompt intentions may spend generation compute; and
* which proven groups form the checkpoint-bound training portfolio.

Neither decision accepts arrival timestamps, payload sizes, token counts, or a
throughput measurement.  Public post-close randomness is used only after the
relevant population has been frozen.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from reliquary.shared.checkpoint_epoch import (
    CHECKPOINT_EPOCH_ADMISSION_POLICY as CHECKPOINT_EPOCH_INTENT_POLICY,
    CHECKPOINT_EPOCH_RANKING_POLICY as CHECKPOINT_EPOCH_MARKET_RANKING_POLICY,
    CHECKPOINT_EPOCH_VALUATION_POLICY as CHECKPOINT_EPOCH_PORTFOLIO_POLICY,
    canonical_json_bytes,
)

_INTENT_OPERATOR_DOMAIN = b"reliquary/checkpoint-epoch/intent-operator/v1"
_INTENT_CANDIDATE_DOMAIN = b"reliquary/checkpoint-epoch/intent-candidate/v1"
_PORTFOLIO_OPERATOR_DOMAIN = b"reliquary/checkpoint-epoch/portfolio-operator/v1"
_PORTFOLIO_CANDIDATE_DOMAIN = b"reliquary/checkpoint-epoch/portfolio-candidate/v1"
_INTENT_SET_ROOT_DOMAIN = b"reliquary/checkpoint-epoch/generation-intent-set/v1"
_INTENT_SET_SIGNING_DOMAIN = b"reliquary/checkpoint-epoch/generation-intent-set-signing/v1"


@dataclass(frozen=True, slots=True)
class GenerationIntent:
    """A miner-selected prompt claim made before expensive generation."""

    intent_id: str
    operator_id: str
    miner_hotkey: str
    window_number: int
    environment: str
    prompt_idx: int
    prompt_content_sha256: str
    generation_nonce: str


@dataclass(frozen=True, slots=True)
class GenerationTicket:
    """One deterministic generation right or inactive backup right."""

    intent_id: str
    role: str
    activation_wave: int
    operator_round: int
    selection_rank: int


@dataclass(frozen=True, slots=True)
class GenerationIntentSet:
    schema_version: int
    epoch_id: str
    manifest_sha256: str
    intent_close_round: int
    validator_hotkey: str
    intent_root: str
    intents: tuple[GenerationIntent, ...]


@dataclass(frozen=True, slots=True)
class SignedGenerationIntentSet:
    intent_set: GenerationIntentSet
    intent_set_sha256: str
    validator_signature: str


@dataclass(frozen=True, slots=True)
class DifficultyStratum:
    """One manifest-bound target share of a varied training portfolio."""

    stratum_id: str
    minimum_mean: float
    maximum_mean: float
    weight: int


@dataclass(frozen=True, slots=True)
class PortfolioCandidate:
    """Proof-eligible group metadata used by the final market selection."""

    candidate_id: str
    operator_id: str
    prompt_idx: int
    prompt_content_sha256: str
    mean_reward: float
    reward_std: float
    robust_utility: float


@dataclass(frozen=True, slots=True)
class PortfolioSelection:
    candidate_id: str
    stratum_id: str
    operator_round: int
    selection_rank: int
    quota_fill: bool


BALANCED_ADVANTAGE_STRATA = (
    DifficultyStratum("frontier", 0.0, 0.25, 1),
    DifficultyStratum("learning", 0.25, 0.75, 2),
    DifficultyStratum("consolidation", 0.75, 1.0, 1),
)


def _hex64(name: str, value: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return bytes.fromhex(value)


def _text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _hash(domain: bytes, randomness: bytes, binding: dict) -> bytes:
    encoded = canonical_json_bytes(binding)
    digest = hashlib.sha256()
    for part in (domain, randomness, encoded):
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return digest.digest()


def _validate_intent(intent: GenerationIntent) -> None:
    _text("intent_id", intent.intent_id)
    _text("operator_id", intent.operator_id)
    _text("miner_hotkey", intent.miner_hotkey)
    _non_negative_int("window_number", intent.window_number)
    _text("environment", intent.environment)
    _non_negative_int("prompt_idx", intent.prompt_idx)
    _hex64("prompt_content_sha256", intent.prompt_content_sha256)
    _text("generation_nonce", intent.generation_nonce)


def generation_intent_to_dict(intent: GenerationIntent) -> dict[str, Any]:
    _validate_intent(intent)
    return {
        "environment": intent.environment,
        "generation_nonce": intent.generation_nonce,
        "intent_id": intent.intent_id,
        "miner_hotkey": intent.miner_hotkey,
        "operator_id": intent.operator_id,
        "prompt_content_sha256": intent.prompt_content_sha256,
        "prompt_idx": intent.prompt_idx,
        "window_number": intent.window_number,
    }


def build_generation_intent_set(
    intents: Sequence[GenerationIntent],
    *,
    epoch_id: str,
    manifest_sha256_hex: str,
    intent_close_round: int,
    validator_hotkey: str,
) -> GenerationIntentSet:
    _hex64("epoch_id", epoch_id)
    _hex64("manifest_sha256_hex", manifest_sha256_hex)
    _non_negative_int("intent_close_round", intent_close_round)
    _text("validator_hotkey", validator_hotkey)
    ordered = tuple(
        sorted(
            intents,
            key=lambda item: (
                item.window_number,
                item.environment,
                item.operator_id,
                item.prompt_idx,
                item.intent_id,
            ),
        )
    )
    if len({item.intent_id for item in ordered}) != len(ordered):
        raise ValueError("duplicate generation intent")
    for intent in ordered:
        _validate_intent(intent)
    root = _hash(
        _INTENT_SET_ROOT_DOMAIN,
        bytes.fromhex(epoch_id),
        {
            "intent_close_round": intent_close_round,
            "intents": [generation_intent_to_dict(item) for item in ordered],
            "manifest_sha256": manifest_sha256_hex,
            "validator_hotkey": validator_hotkey,
        },
    ).hex()
    return GenerationIntentSet(
        schema_version=1,
        epoch_id=epoch_id,
        manifest_sha256=manifest_sha256_hex,
        intent_close_round=intent_close_round,
        validator_hotkey=validator_hotkey,
        intent_root=root,
        intents=ordered,
    )


def generation_intent_set_to_dict(value: GenerationIntentSet) -> dict[str, Any]:
    if not isinstance(value, GenerationIntentSet) or value.schema_version != 1:
        raise ValueError("unsupported generation intent set")
    expected = build_generation_intent_set(
        value.intents,
        epoch_id=value.epoch_id,
        manifest_sha256_hex=value.manifest_sha256,
        intent_close_round=value.intent_close_round,
        validator_hotkey=value.validator_hotkey,
    )
    if value != expected:
        raise ValueError("generation intent set does not match derivation")
    return {
        "epoch_id": value.epoch_id,
        "intent_close_round": value.intent_close_round,
        "intent_root": value.intent_root,
        "intents": [generation_intent_to_dict(item) for item in value.intents],
        "manifest_sha256": value.manifest_sha256,
        "schema_version": value.schema_version,
        "validator_hotkey": value.validator_hotkey,
    }


def canonical_generation_intent_set_bytes(value: GenerationIntentSet) -> bytes:
    return canonical_json_bytes(generation_intent_set_to_dict(value))


def generation_intent_set_sha256(value: GenerationIntentSet) -> str:
    return hashlib.sha256(canonical_generation_intent_set_bytes(value)).hexdigest()


def generation_intent_set_signing_bytes(value: GenerationIntentSet) -> bytes:
    raw = canonical_generation_intent_set_bytes(value)
    digest = hashlib.sha256()
    for part in (_INTENT_SET_SIGNING_DOMAIN, raw):
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return digest.digest()


def canonical_signed_generation_intent_set_bytes(
    value: SignedGenerationIntentSet,
) -> bytes:
    if not isinstance(value, SignedGenerationIntentSet):
        raise TypeError("value must be SignedGenerationIntentSet")
    expected = generation_intent_set_sha256(value.intent_set)
    if value.intent_set_sha256 != expected:
        raise ValueError("signed generation intent-set hash differs")
    if (
        not value.validator_signature
        or len(value.validator_signature) > 256
        or len(value.validator_signature) % 2
        or any(
            character not in "0123456789abcdef"
            for character in value.validator_signature
        )
    ):
        raise ValueError("signed generation intent-set signature is invalid")
    return canonical_json_bytes(
        {
            "intent_set": generation_intent_set_to_dict(value.intent_set),
            "intent_set_sha256": value.intent_set_sha256,
            "validator_signature": value.validator_signature,
        }
    )


def parse_signed_generation_intent_set(
    raw: bytes | str,
) -> SignedGenerationIntentSet:
    raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    try:
        value = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid signed generation intent set") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "intent_set",
        "intent_set_sha256",
        "validator_signature",
    }:
        raise ValueError("signed generation intent-set keys differ")
    body = value["intent_set"]
    if not isinstance(body, Mapping) or set(body) != {
        "epoch_id",
        "intent_close_round",
        "intent_root",
        "intents",
        "manifest_sha256",
        "schema_version",
        "validator_hotkey",
    }:
        raise ValueError("generation intent-set keys differ")
    raw_intents = body["intents"]
    if not isinstance(raw_intents, list):
        raise ValueError("generation intent-set intents must be a list")
    intents = []
    expected_keys = {
        "environment",
        "generation_nonce",
        "intent_id",
        "miner_hotkey",
        "operator_id",
        "prompt_content_sha256",
        "prompt_idx",
        "window_number",
    }
    for item in raw_intents:
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            raise ValueError("generation intent keys differ")
        intents.append(GenerationIntent(**item))
    intent_set = GenerationIntentSet(
        schema_version=body["schema_version"],
        epoch_id=body["epoch_id"],
        manifest_sha256=body["manifest_sha256"],
        intent_close_round=body["intent_close_round"],
        validator_hotkey=body["validator_hotkey"],
        intent_root=body["intent_root"],
        intents=tuple(intents),
    )
    publication = SignedGenerationIntentSet(
        intent_set=intent_set,
        intent_set_sha256=value["intent_set_sha256"],
        validator_signature=value["validator_signature"],
    )
    if raw_bytes != canonical_signed_generation_intent_set_bytes(publication):
        raise ValueError("signed generation intent set is not canonical")
    return publication


def select_generation_tickets(
    intents: Sequence[GenerationIntent],
    *,
    admission_randomness: str,
    epoch_id: str,
    manifest_sha256_hex: str,
    intent_set_sha256_hex: str,
    primary_limit: int,
    backup_limit: int,
    backup_waves: int,
    per_prompt_limit: int,
) -> tuple[GenerationTicket, ...]:
    """Select primary and standby generation rights without an arrival key.

    The miner chooses the lane and prompt before this function runs.  Operators
    are then visited in rounds so additional intents have diminishing returns.
    Backups are ordered now but remain inactive until an advertised wave.
    """

    randomness = _hex64("admission_randomness", admission_randomness)
    _hex64("epoch_id", epoch_id)
    _hex64("manifest_sha256_hex", manifest_sha256_hex)
    _hex64("intent_set_sha256_hex", intent_set_sha256_hex)
    primary_limit = _non_negative_int("primary_limit", primary_limit)
    backup_limit = _non_negative_int("backup_limit", backup_limit)
    backup_waves = _non_negative_int("backup_waves", backup_waves)
    per_prompt_limit = _non_negative_int("per_prompt_limit", per_prompt_limit)
    if primary_limit < 1 or per_prompt_limit < 1:
        raise ValueError("primary_limit and per_prompt_limit must be positive")
    if backup_limit and backup_waves < 1:
        raise ValueError("backup_waves must be positive when backups exist")

    by_operator: dict[str, list[GenerationIntent]] = defaultdict(list)
    seen_ids: set[str] = set()
    lane: tuple[int, str] | None = None
    for intent in intents:
        if not isinstance(intent, GenerationIntent):
            raise TypeError("intents must contain GenerationIntent")
        _validate_intent(intent)
        if intent.intent_id in seen_ids:
            raise ValueError("duplicate generation intent")
        candidate_lane = (intent.window_number, intent.environment)
        if lane is None:
            lane = candidate_lane
        elif candidate_lane != lane:
            raise ValueError("generation intents must share one lane")
        seen_ids.add(intent.intent_id)
        by_operator[intent.operator_id].append(intent)

    common = {
        "epoch_id": epoch_id,
        "manifest_sha256": manifest_sha256_hex,
        "intent_set_sha256": intent_set_sha256_hex,
        "lane": None if lane is None else [lane[0], lane[1]],
    }
    operators = sorted(
        by_operator,
        key=lambda operator: (
            _hash(
                _INTENT_OPERATOR_DOMAIN,
                randomness,
                {**common, "operator_id": operator},
            ),
            operator,
        ),
    )
    queues: dict[str, list[GenerationIntent]] = {}
    for operator, values in by_operator.items():
        queues[operator] = sorted(
            values,
            key=lambda intent: (
                _hash(
                    _INTENT_CANDIDATE_DOMAIN,
                    randomness,
                    {
                        **common,
                        "generation_nonce": intent.generation_nonce,
                        "intent_id": intent.intent_id,
                        "operator_id": operator,
                        "prompt_content_sha256": (
                            intent.prompt_content_sha256
                        ),
                        "prompt_idx": intent.prompt_idx,
                    },
                ),
                intent.intent_id,
            ),
        )

    ordered: list[tuple[GenerationIntent, int]] = []
    prompt_counts: dict[int, int] = defaultdict(int)
    operator_prompts: set[tuple[str, int]] = set()
    round_index = 0
    total_limit = primary_limit + backup_limit
    while len(ordered) < total_limit:
        progressed = False
        for operator in operators:
            queue = queues[operator]
            while queue:
                intent = queue.pop(0)
                operator_prompt = (operator, intent.prompt_idx)
                if (
                    prompt_counts[intent.prompt_idx] >= per_prompt_limit
                    or operator_prompt in operator_prompts
                ):
                    continue
                ordered.append((intent, round_index))
                prompt_counts[intent.prompt_idx] += 1
                operator_prompts.add(operator_prompt)
                progressed = True
                break
            if len(ordered) >= total_limit:
                break
        if not progressed:
            break
        round_index += 1

    tickets: list[GenerationTicket] = []
    for rank, (intent, operator_round) in enumerate(ordered):
        if rank < primary_limit:
            role = "primary"
            wave = 0
        else:
            role = "backup"
            backup_index = rank - primary_limit
            wave = min(
                backup_waves,
                1 + (backup_index * backup_waves // max(1, backup_limit)),
            )
        tickets.append(
            GenerationTicket(
                intent_id=intent.intent_id,
                role=role,
                activation_wave=wave,
                operator_round=operator_round,
                selection_rank=rank,
            )
        )
    return tuple(tickets)


def portfolio_quotas(
    target: int,
    strata: Sequence[DifficultyStratum] = BALANCED_ADVANTAGE_STRATA,
) -> dict[str, int]:
    """Allocate an exact target by deterministic largest remainders."""

    target = _non_negative_int("target", target)
    if target < 1:
        raise ValueError("target must be positive")
    if not strata:
        raise ValueError("strata must be non-empty")
    identifiers: set[str] = set()
    total_weight = 0
    for stratum in strata:
        _text("stratum_id", stratum.stratum_id)
        if stratum.stratum_id in identifiers:
            raise ValueError("duplicate difficulty stratum")
        identifiers.add(stratum.stratum_id)
        if (
            not math.isfinite(stratum.minimum_mean)
            or not math.isfinite(stratum.maximum_mean)
            or not 0.0 <= stratum.minimum_mean < stratum.maximum_mean <= 1.0
            or isinstance(stratum.weight, bool)
            or not isinstance(stratum.weight, int)
            or stratum.weight < 1
        ):
            raise ValueError("invalid difficulty stratum")
        total_weight += stratum.weight

    exact = {
        item.stratum_id: target * item.weight / total_weight for item in strata
    }
    result = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(result.values())
    order = sorted(
        strata,
        key=lambda item: (
            -(exact[item.stratum_id] - result[item.stratum_id]),
            item.stratum_id,
        ),
    )
    for item in order[:remaining]:
        result[item.stratum_id] += 1
    return result


def _candidate_stratum(
    candidate: PortfolioCandidate,
    strata: Sequence[DifficultyStratum],
) -> str | None:
    if (
        not math.isfinite(candidate.mean_reward)
        or not math.isfinite(candidate.reward_std)
        or not math.isfinite(candidate.robust_utility)
        or candidate.reward_std <= 0.0
        or candidate.robust_utility <= 0.0
        or not 0.0 <= candidate.mean_reward <= 1.0
    ):
        return None
    for stratum in strata:
        if (
            candidate.mean_reward > stratum.minimum_mean
            and candidate.mean_reward <= stratum.maximum_mean
        ):
            return stratum.stratum_id
    return None


def _round_robin_candidates(
    candidates: Iterable[PortfolioCandidate],
    *,
    seal_randomness: bytes,
    epoch_id: str,
    manifest_sha256_hex: str,
    stratum_id: str,
) -> list[tuple[PortfolioCandidate, int]]:
    by_operator: dict[str, list[PortfolioCandidate]] = defaultdict(list)
    for candidate in candidates:
        _text("candidate_id", candidate.candidate_id)
        _text("operator_id", candidate.operator_id)
        _non_negative_int("prompt_idx", candidate.prompt_idx)
        _hex64("prompt_content_sha256", candidate.prompt_content_sha256)
        by_operator[candidate.operator_id].append(candidate)

    common = {
        "epoch_id": epoch_id,
        "manifest_sha256": manifest_sha256_hex,
        "stratum_id": stratum_id,
    }
    operators = sorted(
        by_operator,
        key=lambda operator: (
            _hash(
                _PORTFOLIO_OPERATOR_DOMAIN,
                seal_randomness,
                {**common, "operator_id": operator},
            ),
            operator,
        ),
    )
    for operator, values in by_operator.items():
        values.sort(
            key=lambda candidate: (
                -candidate.robust_utility,
                _hash(
                    _PORTFOLIO_CANDIDATE_DOMAIN,
                    seal_randomness,
                    {
                        **common,
                        "candidate_id": candidate.candidate_id,
                        "operator_id": operator,
                        "prompt_content_sha256": (
                            candidate.prompt_content_sha256
                        ),
                        "prompt_idx": candidate.prompt_idx,
                    },
                ),
                candidate.candidate_id,
            )
        )

    ordered: list[tuple[PortfolioCandidate, int]] = []
    round_index = 0
    while True:
        progressed = False
        for operator in operators:
            queue = by_operator[operator]
            if round_index >= len(queue):
                continue
            ordered.append((queue[round_index], round_index))
            progressed = True
        if not progressed:
            return ordered
        round_index += 1


def select_training_portfolio(
    candidates: Sequence[PortfolioCandidate],
    *,
    seal_randomness: str,
    epoch_id: str,
    manifest_sha256_hex: str,
    target: int,
    strata: Sequence[DifficultyStratum] = BALANCED_ADVANTAGE_STRATA,
) -> tuple[PortfolioSelection, ...]:
    """Build a diverse, operator-rounded portfolio after proof eligibility.

    Quotas prevent one difficulty optimum from absorbing the batch.  Operator
    rounds are primary inside each stratum; utility chooses an operator's best
    candidate, not which operator receives the next opportunity.  Unfilled
    quotas spill into one deterministic overflow pass.
    """

    randomness = _hex64("seal_randomness", seal_randomness)
    _hex64("epoch_id", epoch_id)
    _hex64("manifest_sha256_hex", manifest_sha256_hex)
    quotas = portfolio_quotas(target, strata)
    by_stratum: dict[str, list[PortfolioCandidate]] = defaultdict(list)
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, PortfolioCandidate):
            raise TypeError("candidates must contain PortfolioCandidate")
        if candidate.candidate_id in seen_ids:
            raise ValueError("duplicate portfolio candidate")
        seen_ids.add(candidate.candidate_id)
        stratum_id = _candidate_stratum(candidate, strata)
        if stratum_id is not None:
            by_stratum[stratum_id].append(candidate)

    queues = {
        stratum.stratum_id: _round_robin_candidates(
            by_stratum[stratum.stratum_id],
            seal_randomness=randomness,
            epoch_id=epoch_id,
            manifest_sha256_hex=manifest_sha256_hex,
            stratum_id=stratum.stratum_id,
        )
        for stratum in strata
    }
    selected: list[PortfolioSelection] = []
    selected_ids: set[str] = set()
    prompt_ids: set[int] = set()
    content_ids: set[str] = set()

    def consume(stratum_id: str, limit: int, *, quota_fill: bool) -> None:
        for candidate, operator_round in queues[stratum_id]:
            if sum(item.stratum_id == stratum_id for item in selected) >= limit:
                return
            if (
                candidate.candidate_id in selected_ids
                or candidate.prompt_idx in prompt_ids
                or candidate.prompt_content_sha256 in content_ids
            ):
                continue
            selected.append(
                PortfolioSelection(
                    candidate_id=candidate.candidate_id,
                    stratum_id=stratum_id,
                    operator_round=operator_round,
                    selection_rank=len(selected),
                    quota_fill=quota_fill,
                )
            )
            selected_ids.add(candidate.candidate_id)
            prompt_ids.add(candidate.prompt_idx)
            content_ids.add(candidate.prompt_content_sha256)

    for stratum in strata:
        consume(stratum.stratum_id, quotas[stratum.stratum_id], quota_fill=True)

    if len(selected) < target:
        overflow: list[tuple[PortfolioCandidate, int, str]] = []
        for stratum in strata:
            for candidate, operator_round in queues[stratum.stratum_id]:
                if candidate.candidate_id not in selected_ids:
                    overflow.append((candidate, operator_round, stratum.stratum_id))
        overflow.sort(
            key=lambda item: (
                item[1],
                _hash(
                    _PORTFOLIO_CANDIDATE_DOMAIN,
                    randomness,
                    {
                        "candidate_id": item[0].candidate_id,
                        "epoch_id": epoch_id,
                        "manifest_sha256": manifest_sha256_hex,
                        "overflow": True,
                        "stratum_id": item[2],
                    },
                ),
            )
        )
        for candidate, operator_round, stratum_id in overflow:
            if len(selected) >= target:
                break
            if (
                candidate.prompt_idx in prompt_ids
                or candidate.prompt_content_sha256 in content_ids
            ):
                continue
            selected.append(
                PortfolioSelection(
                    candidate_id=candidate.candidate_id,
                    stratum_id=stratum_id,
                    operator_round=operator_round,
                    selection_rank=len(selected),
                    quota_fill=False,
                )
            )
            selected_ids.add(candidate.candidate_id)
            prompt_ids.add(candidate.prompt_idx)
            content_ids.add(candidate.prompt_content_sha256)

    return tuple(selected)


__all__ = [
    "BALANCED_ADVANTAGE_STRATA",
    "CHECKPOINT_EPOCH_INTENT_POLICY",
    "CHECKPOINT_EPOCH_MARKET_RANKING_POLICY",
    "CHECKPOINT_EPOCH_PORTFOLIO_POLICY",
    "DifficultyStratum",
    "GenerationIntent",
    "GenerationIntentSet",
    "GenerationTicket",
    "PortfolioCandidate",
    "PortfolioSelection",
    "SignedGenerationIntentSet",
    "build_generation_intent_set",
    "canonical_generation_intent_set_bytes",
    "canonical_signed_generation_intent_set_bytes",
    "generation_intent_set_sha256",
    "generation_intent_set_signing_bytes",
    "generation_intent_set_to_dict",
    "generation_intent_to_dict",
    "parse_signed_generation_intent_set",
    "portfolio_quotas",
    "select_generation_tickets",
    "select_training_portfolio",
]
