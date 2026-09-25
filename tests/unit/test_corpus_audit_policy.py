"""The per-hotkey audit state machine, the draw, and its parameters — no I/O."""

import pytest

from reliquary.corpus.audit_policy import (
    AuditParams, MinerState, after_confirmed_failure, after_pass, decision,
    drawn, effective_state, validate_audit_params,
)

P = AuditParams(q=0.1, probation_submissions=3, hold_seconds=100, suspect_seconds=50,
                ban_after_failures=2, ban_window_seconds=1000, ban_seconds=500)
R = "ab" * 32


def test_defaults_are_v0():
    assert AuditParams.from_params({}).q == 1.0


@pytest.mark.parametrize("bad", [
    {"audit_q": 0.0}, {"audit_q": 1.5}, {"audit_probation_submissions": 0},
    {"audit_hold_seconds": -1}, {"audit_ban_after_failures": 0}, {"audit_q": "0.1"},
])
def test_nonsense_parameters_are_refused(bad):
    with pytest.raises(ValueError):
        validate_audit_params(bad)


def test_a_new_hotkey_is_audited_until_its_probation_passes():
    m = MinerState()
    for _ in range(3):
        assert decision(m, params=P, now=0, received_at=0, recent_submissions=50, randomness_hex=R) == "audit"
        m = after_pass(m, P, 1.0)
    assert effective_state(m, 0, P) == "sampled"


def test_the_draw_is_deterministic_and_close_to_q():
    ids = [f"{i:064x}" for i in range(4000)]
    hits = sum(drawn(R, i, 0.1) for i in ids)
    assert 300 < hits < 500
    assert [drawn(R, i, 0.1) for i in ids[:50]] == [drawn(R, i, 0.1) for i in ids[:50]]


def _sampled():
    return MinerState(state="sampled", audited_passed=3)


def test_a_sampled_miner_waits_then_passes_unaudited_unless_drawn():
    m = _sampled()
    sid_not = next(f"{i:064x}" for i in range(1000) if not drawn(R, f"{i:064x}", 0.1))
    sid_yes = next(f"{i:064x}" for i in range(1000) if drawn(R, f"{i:064x}", 0.1))
    kw = dict(params=P, received_at=0, recent_submissions=50, randomness_hex=R)
    assert decision(m, now=10, submission_id=sid_yes, **kw) == "audit"
    assert decision(m, now=10, submission_id=sid_not, **kw) == "wait"
    assert decision(m, now=100, submission_id=sid_not, **kw) == "pass_unaudited"


def test_no_beacon_means_audit():
    assert decision(_sampled(), params=P, now=10, received_at=0, recent_submissions=50,
                    randomness_hex=None, submission_id="c" * 64) == "audit"


def test_a_slow_hotkey_is_audited_in_full():
    assert decision(_sampled(), params=P, now=10, received_at=0, recent_submissions=5,
                    randomness_hex=R, submission_id="c" * 64) == "audit"


def test_a_confirmed_failure_makes_it_suspect_then_a_second_bans():
    m = after_confirmed_failure(_sampled(), P, now=0)
    assert effective_state(m, 10, P) == "suspect"
    assert decision(m, params=P, now=10, received_at=0, recent_submissions=50,
                    randomness_hex=R, submission_id="c" * 64) == "audit"
    m = after_confirmed_failure(m, P, now=20)
    assert effective_state(m, 30, P) == "banned"
    assert decision(m, params=P, now=30, received_at=25, recent_submissions=50,
                    randomness_hex=R, submission_id="c" * 64) == "void_banned"


def test_a_ban_ends_into_a_fresh_probation():
    m = after_confirmed_failure(after_confirmed_failure(_sampled(), P, now=0), P, now=1)
    assert effective_state(m, 1 + 500 + 1, P) == "probation"


def test_state_round_trips():
    m = after_pass(_sampled(), P, 2.5)
    assert MinerState.from_dict(m.to_dict()) == m
