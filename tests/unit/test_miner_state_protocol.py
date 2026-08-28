from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from reliquary.protocol.submission import (
    MinerEnvironmentState,
    decode_cooldown_bitmap,
    encode_cooldown_bitmap,
)


def test_cooldown_bitmap_round_trip_is_bounded_to_prompt_slice():
    encoded, count = encode_cooldown_bitmap(
        {1, 100, 103, 107, 108, 999_999},
        (100, 109),
    )

    assert count == 4
    assert len(base64.b64decode(encoded)) == 2
    assert decode_cooldown_bitmap(encoded, (100, 109)) == {
        100, 103, 107, 108,
    }


def test_environment_state_rejects_count_or_trailing_bit_corruption():
    encoded, _ = encode_cooldown_bitmap({10}, (10, 11))
    with pytest.raises(ValidationError, match="cooldown_count"):
        MinerEnvironmentState(
            prompt_range=(10, 11),
            cooldown_bitmap=encoded,
            cooldown_count=0,
        )

    corrupt = base64.b64encode(bytes([0b1000_0000])).decode("ascii")
    with pytest.raises(ValidationError, match="out-of-range bits"):
        MinerEnvironmentState(
            prompt_range=(10, 11),
            cooldown_bitmap=corrupt,
            cooldown_count=1,
        )


def test_large_membership_set_is_not_iterated_outside_bounded_slice():
    class _MembershipOnlySet(set):
        def __iter__(self):
            raise AssertionError("full cooldown history must not be scanned")

    cooldown = _MembershipOnlySet(range(100_000))
    encoded, count = encode_cooldown_bitmap(cooldown, (20_000, 25_000))

    assert len(base64.b64decode(encoded)) == 625
    assert count == 5_000
