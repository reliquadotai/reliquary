"""Conservative valuation of truncated rollouts — closes the manufactured-zero
hole in the difficulty auction.

The auction pays more for hard prompts, so breaking one of your own correct
rollouts (suppress EOS -> runs to the cap -> grades 0) makes a prompt look
harder and pays more. Scoring the minimum over every interpretation of the
truncated rollouts removes that gain entirely.
"""

import pytest

from reliquary.validator.difficulty_auction import (
    conservative_difficulty_score,
    difficulty_score,
)

DELTA = 1.0


def _vec(k, n=8):
    """Binary reward vector with k correct out of n."""
    return [1.0] * k + [0.0] * (n - k)


def _honest(k, n=8):
    return difficulty_score(_vec(k, n), delta=DELTA).value


def _manipulated(k, truncated_correct, truncated_wrong, n=8):
    """Miner truncates `truncated_correct` of its correct rollouts and
    `truncated_wrong` of its wrong ones; truncated rollouts grade 0."""
    c = k - truncated_correct
    t = truncated_correct + truncated_wrong
    rewards = _vec(c, n)          # the t truncated ones are already zeros here
    return conservative_difficulty_score(
        rewards, truncated_count=t, delta=DELTA
    ).value


def test_no_truncation_matches_plain_score():
    for k in range(9):
        plain = difficulty_score(_vec(k), delta=DELTA)
        conservative = conservative_difficulty_score(
            _vec(k), truncated_count=0, delta=DELTA
        )
        assert conservative.value == pytest.approx(plain.value)


def test_manufacturing_a_zero_gains_nothing_at_k6():
    """The headline hole: at k=6 one manufactured zero is worth +68% today."""
    honest = _honest(6)
    naive_gain = difficulty_score(_vec(5), delta=DELTA).value
    assert naive_gain / honest > 1.6          # the hole exists without the fix
    assert _manipulated(6, truncated_correct=1, truncated_wrong=0) == pytest.approx(honest)


def test_truncating_a_wrong_rollout_gains_nothing_at_low_k():
    """Low-k side: truncating a WRONG rollout must not walk the ratio toward
    the value peak (this is what breaks the 'exclude truncated' approach)."""
    honest = _honest(1)
    assert _manipulated(1, truncated_correct=0, truncated_wrong=1) == pytest.approx(honest)


def test_truncating_a_correct_rollout_is_a_loss():
    assert _manipulated(1, truncated_correct=1, truncated_wrong=0) < _honest(1)


def test_manufacturing_never_profitable_exhaustive():
    """Every way to manufacture truncations, at every k: never above honest."""
    checked = 0
    for k in range(9):
        honest = _honest(k)
        for a in range(k + 1):                      # truncate `a` correct
            for b in range(8 - k + 1):              # truncate `b` wrong
                if a + b == 0 or a + b > 8:
                    continue
                checked += 1
                assert _manipulated(k, a, b) <= honest + 1e-12, (
                    f"manufacturing profitable at k={k} a={a} b={b}"
                )
    assert checked > 100                            # the grid really was walked


def test_honest_truncation_is_penalised_but_not_zeroed_in_auction_zone():
    """The cost of the rule lands mostly outside the competitive zone: at the
    low-k prompts the auction actually pays for, an honest truncation loses
    little; the harsh case is high-k (easy) prompts that are near-worthless."""
    # k=2 is the value peak; one honest truncation costs <10%
    mild = conservative_difficulty_score(
        _vec(2), truncated_count=1, delta=DELTA
    ).value
    assert mild > 0.9 * _honest(2)
    # k=6 (easy prompt) is hit harder, but was already worth ~3x less than peak
    harsh = conservative_difficulty_score(
        _vec(6), truncated_count=1, delta=DELTA
    ).value
    assert harsh < _honest(6)
    assert _honest(6) < _honest(2)


def test_fractional_rewards_supported():
    """Code rewards are fractional (passed/total); the rule still applies and
    never scores above the plain value."""
    rewards = [1.0, 0.75, 0.5, 0.0, 0.0, 0.25, 1.0, 0.0]
    plain = difficulty_score(rewards, delta=DELTA).value
    conservative = conservative_difficulty_score(
        rewards, truncated_count=2, delta=DELTA
    ).value
    assert conservative <= plain


def test_auction_value_respects_kill_switch(monkeypatch):
    """Default off: value is the plain score. Flag on: conservative."""
    from reliquary.validator.difficulty_auction import auction_value

    monkeypatch.setattr("reliquary.constants.CONSERVATIVE_TRUNCATION_VALUE", False)
    assert auction_value(_vec(6), truncated_count=1) == pytest.approx(_honest(6))

    monkeypatch.setattr("reliquary.constants.CONSERVATIVE_TRUNCATION_VALUE", True)
    assert auction_value(_vec(6), truncated_count=1) < _honest(6)


def test_auction_value_no_truncation_is_unchanged_either_way(monkeypatch):
    from reliquary.validator.difficulty_auction import auction_value

    for flag in (False, True):
        monkeypatch.setattr(
            "reliquary.constants.CONSERVATIVE_TRUNCATION_VALUE", flag
        )
        assert auction_value(_vec(4), truncated_count=0) == pytest.approx(_honest(4))


def test_pending_submission_uses_conservative_value(monkeypatch):
    """End-to-end through the batcher's pending candidate."""
    monkeypatch.setattr("reliquary.constants.CONSERVATIVE_TRUNCATION_VALUE", True)
    from reliquary.validator.batcher import PendingSubmission

    def _pending(truncated):
        return PendingSubmission(
            hotkey="hk", prompt_idx=1, request=None, rewards=_vec(6),
            drand_round=1, merkle_root=b"\x00" * 32,
            selection_digest=b"\x00" * 32, truncated_count=truncated,
        )

    assert _pending(0).value == pytest.approx(_honest(6))
    assert _pending(1).value < _honest(6)          # manufactured zero gains nothing


def test_more_truncations_never_score_higher():
    """Monotone: adding truncations can only widen the interpretation set, so
    the minimum can only fall."""
    prev = None
    for t in range(0, 5):
        value = conservative_difficulty_score(
            _vec(4), truncated_count=t, delta=DELTA
        ).value
        if prev is not None:
            assert value <= prev + 1e-12
        prev = value


def test_token_budget_is_internally_consistent():
    """One uniform ~16k ceiling: a forced completion (thinking + force span +
    answer) must fit under the global cap, with margin for the template."""
    from reliquary import constants as C

    assert C.MAX_NEW_TOKENS_PROTOCOL_CAP == 16384
    forced_total = C.BFT_THINKING_BUDGET + C.BFT_ANSWER_BUDGET
    assert forced_total < C.MAX_NEW_TOKENS_PROTOCOL_CAP
    assert C.MAX_NEW_TOKENS_PROTOCOL_CAP - forced_total >= 256   # force-span room


def test_truncation_allowance_is_per_env():
    """Math keeps the strict allowance (BFT terminates every rollout); code gets
    room for the 4B's honest looping, which conservative valuation prices."""
    from reliquary import constants as C

    by_env = C.MAX_TRUNCATED_PER_SUBMISSION_BY_ENV
    assert by_env.get("openmathinstruct", C.MAX_TRUNCATED_PER_SUBMISSION) == 1
    assert by_env["opencodeinstruct"] >= 2


def test_collection_window_can_absorb_a_full_budget_generation():
    """The collection deadline must track BFT_THINKING_BUDGET.

    A miner generates its M_ROLLOUTS in one batch, so a group's generation time
    is set by its LONGEST rollout — and with the 4B a long rollout lands in ~96%
    of groups, so essentially every submission takes budget / per-seq-rate. If
    the window is too short for that, honest miners are cut off precisely on the
    hard prompts that reason longest. The live 2B fleet ran ~130 tok/s/seq and a
    4B decodes roughly half as fast, so the window must clear ~65 tok/s/seq.
    """
    from reliquary import constants as C

    required_rate = C.BFT_THINKING_BUDGET / C.WINDOW_COLLECTION_SECONDS
    assert required_rate <= 65.0, (
        f"window {C.WINDOW_COLLECTION_SECONDS}s demands "
        f"{required_rate:.0f} tok/s/seq to generate {C.BFT_THINKING_BUDGET} "
        "tokens; the expected 4B fleet rate is ~65 tok/s/seq"
    )
