from __future__ import annotations

from dataclasses import replace

import pytest

from reliquary.shared.checkpoint_epoch_market import (
    GenerationIntent,
    PortfolioCandidate,
    SignedGenerationIntentSet,
    build_generation_intent_set,
    canonical_signed_generation_intent_set_bytes,
    generation_intent_set_sha256,
    parse_signed_generation_intent_set,
    portfolio_quotas,
    select_generation_tickets,
    select_training_portfolio,
)


EPOCH = "1" * 64
MANIFEST = "2" * 64
SET_HASH = "3" * 64


def _intent(operator: str, index: int, *, prompt: int | None = None):
    return GenerationIntent(
        intent_id=f"intent-{operator}-{index}",
        operator_id=operator,
        miner_hotkey=f"miner-{operator}",
        window_number=500,
        environment="math",
        prompt_idx=index if prompt is None else prompt,
        prompt_content_sha256=f"{index + 1:064x}",
        generation_nonce=f"nonce-{operator}-{index}",
    )


def _tickets(intents):
    return select_generation_tickets(
        intents,
        admission_randomness="4" * 64,
        epoch_id=EPOCH,
        manifest_sha256_hex=MANIFEST,
        intent_set_sha256_hex=SET_HASH,
        primary_limit=4,
        backup_limit=4,
        backup_waves=2,
        per_prompt_limit=2,
    )


def test_generation_tickets_are_arrival_free_deterministic_and_rounded():
    intents = [
        _intent(operator, index + operator_index * 10)
        for operator_index, operator in enumerate(("alice", "bob", "carol"))
        for index in range(4)
    ]

    first = _tickets(intents)
    second = _tickets(list(reversed(intents)))

    assert first == second
    assert [ticket.selection_rank for ticket in first] == list(range(8))
    assert [ticket.role for ticket in first] == ["primary"] * 4 + ["backup"] * 4
    assert [ticket.activation_wave for ticket in first] == [0] * 4 + [1, 1, 2, 2]
    assert sorted(ticket.operator_round for ticket in first[:3]) == [0, 0, 0]


def test_generation_intent_set_is_canonical_and_mutation_bound():
    intents = [_intent("alice", index) for index in range(4)]
    first = build_generation_intent_set(
        intents,
        epoch_id=EPOCH,
        manifest_sha256_hex=MANIFEST,
        intent_close_round=120,
        validator_hotkey="validator",
    )
    second = build_generation_intent_set(
        list(reversed(intents)),
        epoch_id=EPOCH,
        manifest_sha256_hex=MANIFEST,
        intent_close_round=120,
        validator_hotkey="validator",
    )
    assert first == second
    publication = SignedGenerationIntentSet(
        intent_set=first,
        intent_set_sha256=generation_intent_set_sha256(first),
        validator_signature="aa",
    )
    assert parse_signed_generation_intent_set(
        canonical_signed_generation_intent_set_bytes(publication)
    ) == publication
    changed = build_generation_intent_set(
        [replace(intents[0], prompt_idx=99), *intents[1:]],
        epoch_id=EPOCH,
        manifest_sha256_hex=MANIFEST,
        intent_close_round=120,
        validator_hotkey="validator",
    )
    assert generation_intent_set_sha256(changed) != (
        generation_intent_set_sha256(first)
    )


def test_generation_intent_set_rejects_duplicate_publication_key():
    intents = [_intent("alice", index) for index in range(2)]
    intent_set = build_generation_intent_set(
        intents,
        epoch_id=EPOCH,
        manifest_sha256_hex=MANIFEST,
        intent_close_round=120,
        validator_hotkey="validator",
    )
    publication = SignedGenerationIntentSet(
        intent_set=intent_set,
        intent_set_sha256=generation_intent_set_sha256(intent_set),
        validator_signature="aa",
    )
    encoded = canonical_signed_generation_intent_set_bytes(publication)
    ambiguous = encoded.replace(
        b'"intent_set_sha256":',
        b'"intent_set_sha256":"' + b"0" * 64 + b'","intent_set_sha256":',
        1,
    )

    with pytest.raises(ValueError, match="invalid signed generation intent set"):
        parse_signed_generation_intent_set(ambiguous)


def test_generation_ticket_binding_changes_with_intent_mutation():
    intents = [_intent("alice", index) for index in range(8)]
    original = _tickets(intents)
    changed = _tickets(
        [replace(intents[0], prompt_content_sha256="f" * 64), *intents[1:]]
    )

    assert original != changed


def test_generation_ticket_prompt_and_operator_prompt_caps_apply():
    same_prompt = [
        _intent(f"operator-{index}", index, prompt=7)
        for index in range(8)
    ]
    one_operator = [_intent("operator", index, prompt=7) for index in range(8)]

    assert len(_tickets(same_prompt)) == 2
    assert len(_tickets(one_operator)) == 1


def _candidate(
    operator: str,
    index: int,
    mean: float,
    *,
    utility: float = 0.2,
) -> PortfolioCandidate:
    return PortfolioCandidate(
        candidate_id=f"candidate-{operator}-{index}",
        operator_id=operator,
        prompt_idx=index,
        prompt_content_sha256=f"{index + 1:064x}",
        mean_reward=mean,
        reward_std=0.3,
        robust_utility=utility,
    )


def test_balanced_portfolio_has_exact_four_eight_four_targets():
    assert portfolio_quotas(16) == {
        "frontier": 4,
        "learning": 8,
        "consolidation": 4,
    }


def test_portfolio_is_stratified_and_operator_rounded():
    candidates = []
    for stratum_offset, mean in enumerate((0.125, 0.5, 0.875)):
        for operator_offset, operator in enumerate(("alice", "bob", "carol")):
            for local_index in range(6):
                index = stratum_offset * 100 + operator_offset * 10 + local_index
                candidates.append(
                    _candidate(
                        operator,
                        index,
                        mean,
                        utility=0.3 - local_index * 0.01,
                    )
                )

    selected = select_training_portfolio(
        candidates,
        seal_randomness="5" * 64,
        epoch_id=EPOCH,
        manifest_sha256_hex=MANIFEST,
        target=16,
    )

    assert len(selected) == 16
    assert {
        stratum: sum(item.stratum_id == stratum for item in selected)
        for stratum in ("frontier", "learning", "consolidation")
    } == {"frontier": 4, "learning": 8, "consolidation": 4}
    assert all(item.quota_fill for item in selected)
    assert len({item.candidate_id for item in selected}) == 16


def test_portfolio_spills_unused_quota_without_underfilling():
    candidates = [
        _candidate(f"operator-{index % 4}", index, 0.5)
        for index in range(24)
    ]

    selected = select_training_portfolio(
        candidates,
        seal_randomness="6" * 64,
        epoch_id=EPOCH,
        manifest_sha256_hex=MANIFEST,
        target=16,
    )

    assert len(selected) == 16
    assert all(item.stratum_id == "learning" for item in selected)
    assert sum(item.quota_fill for item in selected) == 8


def test_portfolio_does_not_accept_arrival_size_or_token_inputs():
    fields = set(PortfolioCandidate.__dataclass_fields__)

    assert fields.isdisjoint(
        {
            "arrival_round",
            "arrival_ts",
            "completion_tokens",
            "payload_bytes",
            "throughput",
        }
    )
