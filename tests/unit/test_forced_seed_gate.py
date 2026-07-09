from reliquary.validator.batcher import _forced_seed_verdict
from reliquary.protocol.submission import RejectReason


def test_gate_rejects_below_floor_when_enforcing():
    # 100 stochastic positions, 60 matches (0.60) -> below 0.80 floor
    reject = _forced_seed_verdict(n_stoch=100, n_match=60, window=200,
                                  enforce_from=100)
    assert reject is True


def test_gate_accepts_above_floor():
    reject = _forced_seed_verdict(n_stoch=100, n_match=95, window=200, enforce_from=100)
    assert reject is False


def test_gate_abstains_below_min_positions():
    reject = _forced_seed_verdict(n_stoch=10, n_match=0, window=200, enforce_from=100)
    assert reject is False       # too few positions -> abstain, never false-reject


def test_gate_shadow_before_cutover():
    reject = _forced_seed_verdict(n_stoch=100, n_match=0, window=50, enforce_from=100)
    assert reject is False       # before window -> shadow only
