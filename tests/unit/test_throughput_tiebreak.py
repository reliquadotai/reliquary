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


def test_key_reads_the_group_level_completion_length():
    """The key reads whatever ``completion_length`` the submission exposes — a
    group-level aggregate of GENERATED tokens (see the ValidSubmission test
    below for how that aggregate excludes the prompt)."""
    key = _key()

    class Roll:
        def __init__(self, n):
            self.tokens = [0] * n

    class GroupSub:
        drand_round = 16
        rollouts = [Roll(2000), Roll(2000), Roll(2000), Roll(2000)]  # sum 8000

        @property
        def completion_length(self):
            return sum(len(r.tokens) for r in self.rollouts)

    # 8000 tokens / 16 rounds = 500/round -> bucket 10
    assert key(GroupSub())[0] == -int((8000 / 16) / BUCKET)


def test_composes_with_difficulty_tiers_value_dominates():
    """Composite key (value tier, -throughput bucket, arrival): across value
    tiers the better value wins even with lower throughput."""
    cd = CooldownMap(cooldown_windows=50)
    thr = _key()
    high_val = _sub("hv", 1, drand_round=40, completion_length=400)     # tier 0, ~10/r
    low_val = _sub("lv", 2, drand_round=16, completion_length=16000)    # tier 1, 1000/r
    tier = {id(high_val): 0, id(low_val): 1}
    compose = lambda s: (tier.get(id(s), 0), *thr(s))
    batch, _ = select_batch_and_distribute(
        [low_val, high_val], b=1, cooldown_map=cd, current_window=100,
        slot_round_of=compose,
    )
    assert batch[0].hotkey == "hv"       # value dominates throughput


def test_throughput_breaks_draw_within_same_tier():
    """Same value tier (a draw): the higher-throughput submission wins the slot."""
    cd = CooldownMap(cooldown_windows=50)
    thr = _key()
    slow = _sub("slow", 1, drand_round=32, completion_length=16000)   # 500/r
    fast = _sub("fast", 2, drand_round=16, completion_length=16000)   # 1000/r
    tier = {id(slow): 0, id(fast): 0}                                  # same value tier
    compose = lambda s: (tier.get(id(s), 0), *thr(s))
    batch, _ = select_batch_and_distribute(
        [slow, fast], b=1, cooldown_map=cd, current_window=100,
        slot_round_of=compose,
    )
    assert batch[0].hotkey == "fast"


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


def test_bucket_uses_integer_arithmetic():
    """The slot key orders emission, so it must be bit-identical across
    validators: floor division, never a float divide."""
    key = _key()
    sub = _sub("m", 1, drand_round=33, completion_length=16000)
    bucket = -key(sub)[0]
    assert bucket == 16000 // (33 * BUCKET)      # exact integer form
    assert isinstance(bucket, int)


def test_arrival_prefers_validator_observed_round():
    """Arrival is the throughput DENOMINATOR, so the miner-submitted round must
    not win over the validator-observed one."""
    key = _key()
    sub = _sub("m", 1, drand_round=2, completion_length=16000)   # miner claims early
    sub.arrival_drand_round = 32                                 # validator saw late
    assert key(sub)[1] == 32
    assert -key(sub)[0] == 16000 // (32 * BUCKET)


def test_pending_and_ranking_share_one_scorer(monkeypatch):
    """PendingSubmission.value and the _prove_ranked ranking must come from the
    same scorer, or they can drift apart."""
    monkeypatch.setattr("reliquary.constants.CONSERVATIVE_TRUNCATION_VALUE", True)
    from reliquary.validator.batcher import PendingSubmission, _pending_difficulty_score

    p = PendingSubmission(
        hotkey="hk", prompt_idx=1, request=None,
        rewards=[1.0] * 6 + [0.0] * 2, drand_round=1,
        merkle_root=b"\x00" * 32, selection_digest=b"\x00" * 32,
        truncated_count=1,
    )
    assert _pending_difficulty_score(p).value == p.value


def test_valid_submission_completion_length_excludes_the_prompt():
    """The throughput numerator must count GENERATED tokens only: len(tokens)
    carries the prompt once per rollout, and the miner picks the prompt, so a
    long prompt would inflate throughput for free."""
    from reliquary.validator.batcher import ValidSubmission

    class Roll:
        def __init__(self, prompt_len, gen):
            self.tokens = [0] * (prompt_len + gen)
            self.commit = {"rollout": {
                "prompt_length": prompt_len, "completion_length": gen,
            }}

    sub = ValidSubmission(
        hotkey="hk", prompt_idx=1, merkle_root_bytes=b"\x00" * 32,
        rollouts=[Roll(500, 1000), Roll(500, 1000)],
    )
    assert sub.completion_length == 2000          # not 3000 (prompt excluded)


def test_valid_submission_completion_length_falls_back_without_meta():
    """Unexpected shape degrades to tokens-minus-prompt, never raises."""
    from reliquary.validator.batcher import ValidSubmission

    class Roll:
        tokens = [0] * 1500
        commit = {"rollout": {"prompt_length": 500}}       # no completion_length

    sub = ValidSubmission(
        hotkey="hk", prompt_idx=1, merkle_root_bytes=b"\x00" * 32,
        rollouts=[Roll()],
    )
    assert sub.completion_length == 1000
