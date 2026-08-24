"""The 64-receipt admission budget must only be spent on productive work.

``MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW`` exists to bound grading CPU and
seal-time GPU proof. A submission that never reaches the grader (protocol
conformance failures) and a submission whose reward simply landed outside the
difficulty band both leave that budget unspent, so neither may hold a receipt
against the miners still trying to fill the window.
"""

import pytest

from reliquary.constants import MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW


def _batcher(**kwargs):
    from tests.unit.test_grpo_window_batcher import _make_batcher

    return _make_batcher(**kwargs)


def _request(**kwargs):
    from tests.unit.test_grpo_window_batcher import _request as make

    return make(**kwargs)


def _admit(batcher, request):
    """Drive one submission through the production admission sequence."""
    reserved, reason = batcher.try_reserve_proof_admission(request)
    if not reserved:
        return reserved, reason
    started, reason = batcher.start_proof_admission(request)
    if not started:
        return started, reason
    try:
        return True, batcher.accept_submission(request)
    finally:
        batcher.finish_proof_admission(request)


IN_ZONE = [1.0] * 2 + [0.0] * 6
OUT_OF_ZONE = [1.0] + [0.0] * 7


def test_prompt_binding_reject_refunds_its_admission_charge():
    batcher = _batcher(canonical_prompt_tokens_fn=lambda _idx: [999999])

    ok, response = _admit(batcher, _request(prompt_idx=1, hotkey="flood"))

    assert ok is True
    assert response.accepted is False
    assert batcher.proof_grading_charged == 0


def test_out_of_zone_reject_refunds_its_admission_charge():
    batcher = _batcher()

    ok, response = _admit(
        batcher, _request(prompt_idx=1, hotkey="honest", rewards=OUT_OF_ZONE)
    )

    assert ok is True
    assert response.accepted is False
    assert batcher.proof_grading_charged == 0


def test_accepted_submission_keeps_its_admission_charge():
    batcher = _batcher()

    ok, response = _admit(
        batcher, _request(prompt_idx=1, hotkey="honest", rewards=IN_ZONE)
    )

    assert ok is True
    assert response.accepted is True
    assert batcher.proof_grading_charged == 1


def _prompt_bound_request(prompt_idx, hotkey, canonical, rewards):
    """A request whose rollouts all carry ``canonical`` as their prompt prefix.

    ``_request`` shifts every rollout's tokens so no two are identical, which
    also shifts the prompt prefix and makes canonical binding unpassable. Here
    the prefix is pinned and only the completion tail varies.
    """
    request = _request(prompt_idx=prompt_idx, hotkey=hotkey, rewards=rewards)
    for rollout_idx, rollout in enumerate(request.rollouts):
        tail_len = len(rollout.commit["tokens"]) - len(canonical)
        tokens = list(canonical) + [
            500 + rollout_idx * 100 + t for t in range(tail_len)
        ]
        rollout.tokens = tokens
        rollout.commit["tokens"] = tokens
    return request


CANONICAL = [7, 8, 9, 10]


def test_a_flood_of_prompt_mismatches_never_closes_admission():
    """Today's live symptom: two hotkeys whose every submission dies at
    prompt binding must not consume the budget honest miners need."""
    batcher = _batcher(canonical_prompt_tokens_fn=lambda _idx: CANONICAL)

    for idx in range(MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW * 2):
        ok, _ = _admit(batcher, _request(prompt_idx=idx, hotkey="flood"))
        assert ok is True

    ok, response = _admit(
        batcher,
        _prompt_bound_request(999, "honest", CANONICAL, IN_ZONE),
    )
    assert ok is True
    assert response.accepted is True


def test_a_flood_of_out_of_zone_never_closes_admission():
    """The same two hotkeys flooded with out-of-zone submissions a day
    earlier. Honest prospecting must not be able to starve the fleet."""
    batcher = _batcher()

    for idx in range(MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW * 2):
        ok, _ = _admit(
            batcher,
            _request(prompt_idx=idx, hotkey="flood", rewards=OUT_OF_ZONE),
        )
        assert ok is True

    ok, response = _admit(
        batcher, _request(prompt_idx=999, hotkey="honest", rewards=IN_ZONE)
    )
    assert ok is True
    assert response.accepted is True


def test_total_grading_starts_stay_bounded_when_rejects_are_refunded():
    """Refunding must not remove the anti-DoS ceiling: the never-refunded
    start counter still bounds how much grading work a window can be made to
    run, at a deliberately higher backstop."""
    from reliquary.constants import MAX_GRADING_STARTS_PER_WINDOW

    assert MAX_GRADING_STARTS_PER_WINDOW > MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    batcher = _batcher(canonical_prompt_tokens_fn=lambda _idx: [999999])

    admitted = 0
    for idx in range(MAX_GRADING_STARTS_PER_WINDOW + 5):
        ok, _ = _admit(batcher, _request(prompt_idx=idx, hotkey="flood"))
        if not ok:
            break
        admitted += 1

    assert admitted == MAX_GRADING_STARTS_PER_WINDOW
    assert batcher.proof_grading_attempts == MAX_GRADING_STARTS_PER_WINDOW


def test_dishonest_post_grading_reject_keeps_its_charge():
    """A reward mismatch consumed real grading work and is miner-attributable,
    so it stays charged even though it produced no candidate."""
    from reliquary.validator.batcher import RejectReason

    batcher = _batcher(completion_text_fn=lambda rollout: "wrong")

    ok, response = _admit(
        batcher, _request(prompt_idx=1, hotkey="liar", rewards=IN_ZONE)
    )

    assert ok is True
    assert response.accepted is False
    assert response.reason == RejectReason.REWARD_MISMATCH
    assert batcher.proof_grading_charged == 1


def test_isolated_admission_worker_reject_also_refunds():
    """Production grades in a separate process, so the prompt-binding reject
    that starves the live math windows arrives as a PreparedSubmission and
    never passes through the in-process accept path. It must refund too."""
    from reliquary.protocol.submission import RejectReason

    batcher = _batcher()
    request = _request(prompt_idx=1, hotkey="flood")

    assert batcher.try_reserve_proof_admission(request) == (True, None)
    assert batcher.start_proof_admission(request) == (True, None)
    response = batcher.reject_prepared_submission(
        request, RejectReason.PROMPT_MISMATCH, "prompt_binding"
    )
    batcher.finish_proof_admission(request)

    assert response.accepted is False
    assert batcher.proof_grading_charged == 0
