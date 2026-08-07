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


def test_generated_tokens_cap_each_rollout():
    from types import SimpleNamespace
    from reliquary.validator.batcher import _generated_tokens_of

    long_roll = SimpleNamespace(
        commit={"rollout": {"completion_length": 16384}}
    )
    pending = SimpleNamespace(
        request=SimpleNamespace(rollouts=[long_roll] * 8)
    )
    assert _generated_tokens_of(pending, per_rollout_cap=15616) == 15616 * 8
    assert _generated_tokens_of(pending) == 16384 * 8  # uncapped by default


def test_group_totals_stay_discriminant_under_group_scale_cap():
    """Regression: token_cap=15616 (a single-rollout scale) applied to the
    8-rollout SUM clamped 53% of production groups onto one constant, so the
    throughput tie-break degenerated into raw arrival speed — the exact
    behavior it was designed to replace. At group scale (M_ROLLOUTS x cap),
    real measured totals (3k-129k) must map to distinct ranks again."""
    same_arrival = dict(
        arrival_round=1060,
        window_open_round=1000,
        bucket_tokens_per_round=50,
    )
    # old group cap: both real-world totals collapse onto the same bucket
    assert throughput_rank(
        47_572, token_cap=15616, **same_arrival
    ) == throughput_rank(20_000, token_cap=15616, **same_arrival)
    # group-scale cap: they are discriminant again
    assert throughput_rank(
        47_572, token_cap=15616 * 8, **same_arrival
    ) != throughput_rank(20_000, token_cap=15616 * 8, **same_arrival)


def test_batcher_wires_the_group_scale_cap():
    import inspect

    import reliquary.validator.batcher as B

    src = inspect.getsource(B)
    assert "per_rollout_cap=throughput_profile.token_cap" in src
    assert "token_cap=throughput_profile.token_cap * M_ROLLOUTS" in src
