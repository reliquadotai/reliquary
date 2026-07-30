"""Throughput (tokens/round) draw tie-break — replaces raw arrival speed so a
model that must reason long (the 4B) is not penalized for arriving later.

Since auction v3 the tie-break is a component of the STRICT auction rank
(``_prove_ranked``), not a slot key handed to ``select_batch_and_distribute``:
value first, then throughput, then arrival.
"""

from reliquary.constants import (
    CHALLENGE_K,
    THROUGHPUT_BUCKET_TOKENS_PER_ROUND,
    THROUGHPUT_TOKEN_CAP,
)
from reliquary.protocol.submission import BatchSubmissionRequest, RolloutSubmission
from reliquary.validator.batch_selection import throughput_rank
from tests.unit.test_deferred_proof import _accept
from tests.unit.test_grpo_window_batcher import _make_batcher, _make_commit

CAP = THROUGHPUT_TOKEN_CAP
BUCKET = THROUGHPUT_BUCKET_TOKENS_PER_ROUND
PROMPT_LEN = 4


def _rank(tokens, arrival, window_open=0, cap=CAP, width=BUCKET):
    return throughput_rank(
        tokens,
        arrival_round=arrival,
        window_open_round=window_open,
        token_cap=cap,
        bucket_tokens_per_round=width,
    )


# ────────────────  the ratio itself  ────────────────


def test_length_neutral_same_throughput_same_rank():
    """16k tokens in 32 rounds and 500 in 1 round are both 500 tok/round —
    the long generation gets the SAME rank, not a worse one."""
    assert _rank(16000, arrival=32) == _rank(500, arrival=1)


def test_higher_throughput_sorts_first():
    """Higher tok/round → more negative rank → proven and paid first."""
    fast = _rank(16000, arrival=16)      # 1000/r
    slow = _rank(16000, arrival=32)      # 500/r
    assert fast < slow


def test_padding_past_cap_earns_no_rank():
    """min(tokens, cap) — generating past the cap does not raise throughput."""
    assert _rank(CAP, arrival=32) == _rank(CAP * 3, arrival=32)


def test_instant_arrival_floors_the_denominator():
    """A submission stamped at the window-open round must not divide by zero."""
    assert _rank(16000, arrival=7, window_open=7) == -(16000 // BUCKET)


def test_uses_integer_arithmetic():
    """The rank orders emission, so it must be bit-identical across
    validators: floor division, never a float divide."""
    rank = _rank(16000, arrival=33)
    assert rank == -(16000 // (33 * BUCKET))
    assert isinstance(rank, int)


def test_unusable_token_count_degrades_to_zero():
    """An unexpected shape ranks throughput 0 (last), it never raises."""
    assert _rank(None, arrival=5) == 0
    assert _rank("many", arrival=5) == 0
    assert _rank(-10, arrival=5) == 0


# ────────────────  wiring into the auction rank  ────────────────


def _request(prompt_idx, hotkey, *, completion_tokens, rewards=None, salt=1):
    """A submission whose rollouts carry ``completion_tokens`` GENERATED tokens
    each. Token ids are salted per candidate so two candidates never collide on
    the logical-group / rollout dedup."""
    if rewards is None:
        rewards = [1.0] * 2 + [0.0] * 6          # k=2, the difficulty peak
    rollouts = []
    for idx, reward in enumerate(rewards):
        tokens = [
            1 + (salt * 131 + idx * 17 + t) % 8000
            for t in range(PROMPT_LEN + completion_tokens)
        ]
        commit = _make_commit(
            tokens=tokens,
            prompt_length=PROMPT_LEN,
            success=reward > 0.5,
            total_reward=reward,
        )
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"],
                reward=reward,
                commit=commit,
                env_name="openmathinstruct",
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey,
        prompt_idx=prompt_idx,
        window_start=500,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
        protocol_version=2,
    )


def _thr_batcher(monkeypatch, *, enabled=True, width=1, window_open=100):
    """Auction batcher with the tie-break armed. The bucket width is narrowed so
    the test's small token counts still land in distinct buckets."""
    monkeypatch.setattr(
        "reliquary.validator.batcher.THROUGHPUT_TIEBREAK_ENABLED", enabled
    )
    monkeypatch.setattr(
        "reliquary.validator.batcher.THROUGHPUT_BUCKET_TOKENS_PER_ROUND", width
    )
    b = _make_batcher()
    assert b.difficulty_auction_enabled
    b.window_open_drand_round = window_open
    return b


def test_throughput_wins_the_draw_over_an_earlier_arrival(monkeypatch):
    """The point of the feature: a long-but-efficient submission arriving later
    outranks an early, low-throughput one at EQUAL value — the ordering the raw
    arrival tie-break got backwards."""
    b = _thr_batcher(monkeypatch)
    _accept(b, _request(1, "long", completion_tokens=8 * CHALLENGE_K, salt=1),
            arrival_round=110)                    # 2048 tok / 10 rounds = 204/r
    _accept(b, _request(2, "short", completion_tokens=CHALLENGE_K, salt=2),
            arrival_round=102)                    # 256 tok / 2 rounds = 128/r
    b.seal_batch()

    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["long"]["rank"] < rows["short"]["rank"]
    assert rows["long"]["arrival_drand_round"] == 110      # it really was later


def test_value_still_dominates_throughput(monkeypatch):
    """Throughput only orders EQUAL value. A harder prompt outranks a
    high-throughput easy one."""
    b = _thr_batcher(monkeypatch)
    _accept(
        b,
        _request(1, "easy-fast", completion_tokens=8 * CHALLENGE_K, salt=1,
                 rewards=[1.0] * 6 + [0.0] * 2),
        arrival_round=101,
    )
    _accept(
        b,
        _request(2, "hard-slow", completion_tokens=CHALLENGE_K, salt=2,
                 rewards=[1.0] * 2 + [0.0] * 6),
        arrival_round=160,
    )
    b.seal_batch()

    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["hard-slow"]["rank"] == 1


def test_disabled_keeps_the_arrival_ordering_bit_for_bit(monkeypatch):
    """Default off: the rank contributes 0 for everyone, so the earlier arrival
    wins the draw exactly as before the feature."""
    b = _thr_batcher(monkeypatch, enabled=False)
    _accept(b, _request(1, "long", completion_tokens=8 * CHALLENGE_K, salt=1),
            arrival_round=110)
    _accept(b, _request(2, "short", completion_tokens=CHALLENGE_K, salt=2),
            arrival_round=102)
    b.seal_batch()

    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["short"]["rank"] < rows["long"]["rank"]


def test_unanchored_window_disables_the_tiebreak(monkeypatch):
    """No window-open round (drand hiccup) means elapsed is unmeasurable, so the
    rank degrades to 0 rather than misreading it."""
    b = _thr_batcher(monkeypatch)
    b.window_open_drand_round = None
    _accept(b, _request(1, "long", completion_tokens=4 * CHALLENGE_K, salt=1),
            arrival_round=110)
    pending = b.pending_submissions()[-1]
    assert b._throughput_rank_of(pending, 110) == 0


def test_rank_reads_the_candidates_generated_tokens(monkeypatch):
    """The numerator is the group's generated tokens, taken from the payload the
    proof will re-derive — not the prompt, and not a separate declaration."""
    b = _thr_batcher(monkeypatch, width=BUCKET)
    _accept(b, _request(1, "m", completion_tokens=CHALLENGE_K, salt=1),
            arrival_round=110)
    pending = b.pending_submissions()[-1]

    assert pending.completion_length == 8 * CHALLENGE_K       # prompt excluded
    assert b._throughput_rank_of(pending, 110) == _rank(
        8 * CHALLENGE_K, arrival=110, window_open=100
    )


def test_early_close_and_throughput_are_mutually_exclusive(monkeypatch):
    """Early close proves dominance from arrival monotonicity, which the
    throughput rank breaks (a later arrival CAN outrank). With the tie-break
    armed the prover must not start; the collection deadline seals."""
    b = _thr_batcher(monkeypatch)
    b.mark_window_opened()
    assert b._early_close_thread is None

    b2 = _thr_batcher(monkeypatch, enabled=False)
    b2.mark_window_opened()
    assert b2._early_close_thread is not None
    b2.force_seal("test")


# ────────────────  the numerator on both sides of the proof  ────────────────


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
