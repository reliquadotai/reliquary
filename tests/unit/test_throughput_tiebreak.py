"""Throughput (tokens/round) draw tie-break — replaces raw arrival speed so a
model that must reason long (the 4B) is not penalized for arriving later."""

from dataclasses import dataclass

from reliquary.validator.batch_selection import (
    make_throughput_slot_key,
    select_batch_and_distribute,
)
from reliquary.validator.cooldown import CooldownMap

CAP = 16000
BUCKET = 50


@dataclass
class FakeSubmission:
    hotkey: str
    prompt_idx: int
    drand_round: int
    completion_length: int
    merkle_root: bytes = b"\x00" * 32
    selection_digest: bytes = b"\x00" * 32


def _sub(hotkey, prompt_idx, drand_round, completion_length):
    digest = hotkey.encode().ljust(32, b"\x00")
    return FakeSubmission(
        hotkey=hotkey, prompt_idx=prompt_idx, drand_round=drand_round,
        completion_length=completion_length, merkle_root=digest,
        selection_digest=digest,
    )


def _key(window_open=0):
    return make_throughput_slot_key(
        window_open, token_cap=CAP, bucket_tokens_per_round=BUCKET,
    )


def test_length_neutral_same_throughput_same_tier():
    """16k tokens in 32 rounds and 500 in 1 round are both 500 tok/round —
    the long generation lands in the SAME tier, not a worse one."""
    key = _key()
    long_gen = _sub("long", 1, drand_round=32, completion_length=16000)   # 500/r
    short_gen = _sub("short", 2, drand_round=1, completion_length=500)    # 500/r
    assert key(long_gen)[0] == key(short_gen)[0]      # same -bucket


def test_higher_throughput_sorts_first():
    """Higher tok/round → earlier (more negative) tier → fills the batch first."""
    key = _key()
    fast = _sub("fast", 1, drand_round=16, completion_length=16000)  # 1000/r
    slow = _sub("slow", 2, drand_round=32, completion_length=16000)  # 500/r
    assert key(fast) < key(slow)


def test_padding_past_cap_earns_no_rank():
    """min(tokens, cap) — generating past the cap does not raise throughput."""
    key = _key()
    capped = _sub("a", 1, drand_round=32, completion_length=16000)
    padded = _sub("b", 2, drand_round=32, completion_length=48000)
    assert key(capped)[0] == key(padded)[0]


def test_arrival_breaks_within_bucket_ties():
    """Same throughput bucket → earlier arrival wins (deterministic fallback)."""
    key = _key()
    early = _sub("early", 1, drand_round=10, completion_length=5000)
    late = _sub("late", 2, drand_round=11, completion_length=5500)
    # both ~500/r → same bucket; the (-bucket, arrival) tuple orders by arrival
    assert key(early)[0] == key(late)[0]
    assert key(early) < key(late)


def test_missing_completion_length_degrades_to_last_tier():
    """A submission without completion_length ranks throughput 0, never raises."""
    key = _key()

    class Bare:
        drand_round = 5
    k = key(Bare())
    assert k[0] == 0          # bucket 0 (highest tier value → sorts last)


def test_long_efficient_beats_short_inefficient_for_scarce_slot():
    """End-to-end: with one slot, the long-but-efficient miner wins the draw the
    old arrival ordering would have handed to the early, low-throughput miner."""
    cd = CooldownMap(cooldown_windows=50)
    long_efficient = _sub("long", 1, drand_round=32, completion_length=16000)  # 500/r
    short_inefficient = _sub("short", 2, drand_round=5, completion_length=100)  # 20/r
    subs = [short_inefficient, long_efficient]

    # Arrival ordering (default) would pick the early short one.
    arrival_batch, _ = select_batch_and_distribute(
        subs, b=1, cooldown_map=cd, current_window=100,
    )
    assert arrival_batch[0].hotkey == "short"

    # Throughput ordering picks the efficient long one instead.
    thr_batch, _ = select_batch_and_distribute(
        subs, b=1, cooldown_map=cd, current_window=100, slot_round_of=_key(),
    )
    assert thr_batch[0].hotkey == "long"
