"""Throughput ordering among EQUAL-difficulty candidates (v3 profile only).

A tie-break never trades against training utility — difficulty ranks first, so
this only orders submissions already judged equally useful. Ordering those by
arrival round penalises long generation, which is precisely what the 4B does.
"""

import pytest

from reliquary.protocol.profiles import PROFILES
from reliquary.validator.batch_selection import throughput_rank

CAP = 15616
BUCKET = 50


def _rank(tokens, arrival, open_round=0):
    return throughput_rank(
        tokens, arrival_round=arrival, window_open_round=open_round,
        token_cap=CAP, bucket_tokens_per_round=BUCKET,
    )


def test_length_neutral_at_equal_hardware():
    """15616 tokens in 32 rounds and 488 in 1 round are both ~488 tok/round —
    the long generation lands in the SAME bucket, not a worse one."""
    assert _rank(15616, 32) == _rank(488, 1)


def test_higher_throughput_ranks_first():
    assert _rank(15616, 16) < _rank(15616, 32)      # more negative sorts first


def test_padding_past_the_cap_earns_no_rank():
    assert _rank(CAP, 32) == _rank(CAP * 4, 32)


def test_integer_arithmetic_for_a_consensus_key():
    rank = _rank(15616, 33)
    assert rank == -(15616 // (33 * BUCKET))
    assert isinstance(rank, int)


def test_missing_tokens_degrade_to_last_bucket():
    assert _rank(0, 10) == 0


def test_only_v3_carries_the_tiebreak():
    """v2 is the deployed 2B ordering and must stay untouched; the rule is part
    of the versioned profile so a miner reads which contract applies."""
    assert PROFILES["qwen35-2b-auction-v2"].throughput_tiebreak is None
    v3 = PROFILES["qwen35-4b-auction-v3"].throughput_tiebreak
    assert v3 is not None
    assert v3.token_cap == 15616


def test_contract_publishes_the_tiebreak_to_miners():
    contract = PROFILES["qwen35-4b-auction-v3"].to_generation_contract()
    assert contract["throughput_tiebreak"]["token_cap"] == 15616
    assert PROFILES["qwen35-2b-auction-v2"].to_generation_contract()[
        "throughput_tiebreak"
    ] is None


def test_generated_tokens_exclude_the_prompt():
    """The rank numerator counts generated tokens: len(tokens) carries the
    prompt once per rollout, and the miner picks the prompt."""
    from types import SimpleNamespace
    from reliquary.validator.batcher import _generated_tokens_of

    roll = SimpleNamespace(commit={"rollout": {
        "prompt_length": 500, "completion_length": 1000,
    }})
    pending = SimpleNamespace(request=SimpleNamespace(rollouts=[roll, roll]))
    assert _generated_tokens_of(pending) == 2000        # not 3000


def test_generated_tokens_degrade_instead_of_raising():
    from types import SimpleNamespace
    from reliquary.validator.batcher import _generated_tokens_of

    assert _generated_tokens_of(SimpleNamespace(request=None)) == 0
    bad = SimpleNamespace(commit={"rollout": {"completion_length": "nope"}})
    assert _generated_tokens_of(
        SimpleNamespace(request=SimpleNamespace(rollouts=[bad]))
    ) == 0
