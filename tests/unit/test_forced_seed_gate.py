from reliquary.validator.batcher import _forced_seed_verdict
from reliquary.validator import batcher as _batcher
from reliquary.protocol.submission import RejectReason


def test_gate_rejects_below_floor_when_enforcing():
    # 100 stochastic positions, 60 matches (0.60) -> below 0.80 floor
    reject = _forced_seed_verdict(n_stoch=100, n_match=60, enforce=True)
    assert reject is True


def test_gate_accepts_above_floor():
    reject = _forced_seed_verdict(n_stoch=100, n_match=95, enforce=True)
    assert reject is False


def test_gate_abstains_below_min_positions():
    reject = _forced_seed_verdict(n_stoch=10, n_match=0, enforce=True)
    assert reject is False       # too few positions -> abstain, never false-reject


def test_gate_shadow_when_not_enforcing():
    reject = _forced_seed_verdict(n_stoch=100, n_match=0, enforce=False)
    assert reject is False       # enforcement off -> shadow only


# ── Per-rollout hardening (H3): the group average dilutes a partial swap. ──

def test_rollout_gate_rejects_single_swap_the_group_average_hides():
    # 7 honest rollouts (~0.96) + 1 fully-swapped (0.60). The GROUP average
    # (366/400 = 0.915) sails past the 0.80 floor, but the per-rollout check
    # catches the one off-stream rollout.
    per_rollout = [(50, 48)] * 7 + [(50, 30)]
    g_stoch = sum(s for s, _ in per_rollout)
    g_match = sum(m for _, m in per_rollout)
    assert g_match / g_stoch >= 0.80          # group verdict would accept
    assert _forced_seed_rollout_reject(per_rollout, enforce=True) is True


def test_rollout_gate_accepts_all_honest():
    per_rollout = [(50, 48)] * 8              # every rollout ~0.96
    assert _forced_seed_rollout_reject(per_rollout, enforce=True) is False


def test_rollout_gate_abstains_on_thin_rollout():
    # A rollout with too few stochastic positions is never judged -> a short or
    # peaked honest rollout can't be false-rejected on thin signal.
    per_rollout = [(50, 48)] * 7 + [(5, 0)]   # 0.0 but only 5 positions
    assert _forced_seed_rollout_reject(per_rollout, enforce=True) is False


def test_rollout_gate_shadow_when_not_enforcing():
    per_rollout = [(50, 48)] * 7 + [(50, 10)]  # one clearly off-stream
    assert _forced_seed_rollout_reject(per_rollout, enforce=False) is False


def _forced_seed_rollout_reject(per_rollout, enforce):
    return _batcher._forced_seed_rollout_reject(per_rollout, enforce)
