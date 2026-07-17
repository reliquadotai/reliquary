"""GrpoWindowBatcher: accepts submissions, enforces verification pipeline,
exposes select_batch at window close."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pytest

from reliquary.constants import (
    B_BATCH,
    CHALLENGE_K,
    FORCED_SEED_PROTOCOL_VERSION,
    M_ROLLOUTS,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RejectReason,
    RolloutSubmission,
)
from reliquary.validator.batcher import GrpoWindowBatcher


class FakeEnv:
    # Generic batcher tests exercise the production Math lane. Tests for the
    # legacy Code lane inject ``PrivateRewardFakeEnv`` explicitly.
    name = "openmathinstruct"
    def __len__(self):
        return 1000
    def get_problem(self, idx):
        return {"prompt": f"p{idx}", "ground_truth": "a", "id": f"pid-{idx}"}
    def compute_reward(self, problem, completion):
        return 1.0 if "CORRECT" in completion else 0.0


class PrivateRewardFakeEnv(FakeEnv):
    name = "opencodeinstruct"
    validator_authoritative_reward = True


def _always_true_grail(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))


def _always_false_grail(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=False, passed=0, checked=1, logits=torch.empty(0))


def _always_true_sig(commit, hotkey):
    return True


def _make_commit(
    *,
    tokens: list[int] | None = None,
    prompt_length: int = 4,
    success: bool = False,
    total_reward: float = 0.0,
) -> dict:
    """Build a minimal commit that passes CommitModel.model_validate.

    Default produces a ``CHALLENGE_K + 4`` token sequence: 4 prompt tokens,
    ``CHALLENGE_K`` completion tokens (the minimum the proof needs).
    """
    if tokens is None:
        tokens = list(range(CHALLENGE_K + prompt_length))
    seq_len = len(tokens)
    completion_length = seq_len - prompt_length
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v7",
        "model": {"name": "test-model", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
            "success": success,
            "total_reward": total_reward,
            "advantage": 0.0,
            "token_logprobs": [0.0] * seq_len,
        },
    }


def _request(
    prompt_idx=42, window_start=500,
    rewards=None, hotkey="hk",
) -> BatchSubmissionRequest:
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for idx, r in enumerate(rewards):
        # Shift token ids by idx so each rollout has a unique sequence;
        # offsets stay well within any test model's vocab_size.
        tokens = [t + idx for t in range(CHALLENGE_K + 4)]
        commit = _make_commit(tokens=tokens, success=r > 0.5, total_reward=r)
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"],
                reward=r,
                commit=commit,
                env_name="openmathinstruct",
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey,
        prompt_idx=prompt_idx,
        window_start=window_start,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
        protocol_version=2,
    )


def _request_with_prompt_unique_tokens(
    prompt_idx=42, window_start=500, rewards=None, hotkey="hk",
) -> BatchSubmissionRequest:
    req = _request(
        prompt_idx=prompt_idx,
        window_start=window_start,
        rewards=rewards,
        hotkey=hotkey,
    )
    for rollout_idx, rollout in enumerate(req.rollouts):
        tokens = [
            prompt_idx * 100 + rollout_idx * 10 + t
            for t in range(CHALLENGE_K + 4)
        ]
        rollout.tokens = tokens
        rollout.commit["tokens"] = tokens
    return req


def _make_batcher(**overrides) -> GrpoWindowBatcher:
    class _DefaultFakeTokenizer:
        eos_token_id = 99

    class _DefaultModelStub:
        """Minimal stub satisfying resolve_vocab_size + resolve_max_context_length.

        vocab_size=10000 is comfortably above any test token id (existing tests
        use ids in [0, CHALLENGE_K + 4) ~ 36).
        """
        class config:
            vocab_size = 10000
            max_position_embeddings = 4096

    kwargs = dict(
        window_start=500,
        env=FakeEnv(),
        model=_DefaultModelStub(),
        tokenizer=_DefaultFakeTokenizer(),
        verify_commitment_proofs_fn=_always_true_grail,
        verify_signature_fn=_always_true_sig,
        completion_text_fn=lambda rollout: (
            "CORRECT" if rollout.reward > 0.5 else "wrong"
        ),
        hash_set=None,
        # The vast majority of legacy tests construct requests without an
        # attached drand_round (default 0). Disable the check by default
        # in the test helper; tests that exercise the drand timing gate
        # explicitly override `drand_round_check_enabled=True`.
        drand_round_check_enabled=False,
    )
    kwargs.update(overrides)
    b = GrpoWindowBatcher(**kwargs)
    # Match the per-window randomness used by ``_make_commit`` so the new
    # randomness-binding check (BAD_SIGNATURE → WRONG_RANDOMNESS → GRAIL)
    # doesn't reject every test request. Production sets this via
    # ``service._set_window_randomness`` before the window opens for
    # submissions; tests skip that hop.
    b.randomness = "cd" * 16
    return b


def _prove_one(b: GrpoWindowBatcher, req) -> "ValidSubmission | None":
    """Run one request through its environment's configured proof path.

    Auction-enabled Math and Code both defer proof to seal. Returns None when
    the configured path rejects the submission.
    """
    response = b.accept_submission(req)
    if not response.accepted:
        return None
    if b.difficulty_auction_enabled:
        return b._verify_expensive(b.pending_submissions()[-1])
    return b.valid_submissions()[-1]


def test_constructor_sets_window():
    b = _make_batcher()
    assert b.window_start == 500


def test_reject_window_mismatch():
    b = _make_batcher()
    req = _request(window_start=999)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WINDOW_MISMATCH


def test_accept_in_zone_submission():
    b = _make_batcher()
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED
    assert len(b.pending_submissions()) == 1   # proven at seal, not at admission
    b.seal_batch()
    assert len(b.valid_submissions()) == 1


def test_accepted_submission_uses_validator_computed_selection_digest():
    from reliquary.validator.selection_digest import (
        compute_rollouts_selection_digest,
    )

    b = _make_batcher()
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)

    assert b.accept_submission(req).accepted is True
    b.seal_batch()  # proof runs at seal; selection digest carries to _valid
    accepted = b.valid_submissions()[0]
    assert accepted.selection_digest == compute_rollouts_selection_digest(
        req.rollouts
    )
    assert accepted.selection_digest_bytes == accepted.selection_digest


def test_ingestion_resets_miner_supplied_truncated_flag():
    """`truncated` is a validator-set reward-shaping flag. A miner-supplied
    value must be wiped at ingestion so it can't clamp a losing rollout's
    advantage to -SHAPE_PENALTY via _shape_advantages. These rollouts are not
    cap-truncated, so after the reset the validator leaves the flag False."""
    b = _make_batcher()
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    for rollout in req.rollouts:
        rollout.commit["rollout"]["truncated"] = True  # miner-forged

    resp = b.accept_submission(req)

    assert resp.accepted is True, resp
    assert all(
        rollout.commit["rollout"]["truncated"] is False
        for rollout in req.rollouts
    )


def test_ingestion_resets_forced_flag_in_non_math_env():
    """BFT is math-only (mirror the miner's env gate): a `forced` flag on a
    non-math submission is wiped at ingestion so the carve-out stays scoped to
    openmathinstruct. PrivateRewardFakeEnv.name == "opencodeinstruct"."""
    b = _make_batcher(env=PrivateRewardFakeEnv())
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    for rollout in req.rollouts:
        rollout.commit["rollout"]["forced"] = True  # miner-supplied

    resp = b.accept_submission(req)

    assert resp.accepted is True, resp
    assert all(
        rollout.commit["rollout"]["forced"] is False for rollout in req.rollouts
    )


def test_grail_verifier_receives_tokenizer_for_sparse_pstop():
    import torch
    from reliquary.validator.verifier import ProofResult

    seen_tokenizers = []

    def tokenizer_aware_grail(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        seen_tokenizers.append(tokenizer)
        return ProofResult(
            all_passed=True,
            passed=1,
            checked=1,
            logits=torch.empty(0),
        )

    b = _make_batcher(verify_commitment_proofs_fn=tokenizer_aware_grail)
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    # The GRAIL proof (and thus the tokenizer hand-off) runs at seal now.
    assert _prove_one(b, req) is not None
    assert seen_tokenizers
    assert all(tok is b.tokenizer for tok in seen_tokenizers)


def test_reject_out_of_zone_all_fail():
    b = _make_batcher()
    req = _request(rewards=[0.0] * 8)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.OUT_OF_ZONE
    assert len(b.valid_submissions()) == 0


def test_reject_out_of_zone_all_pass():
    b = _make_batcher()
    req = _request(rewards=[1.0] * 8)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.OUT_OF_ZONE


def test_out_of_zone_does_not_charge_grail_candidate():
    """An out_of_zone reject is a reward error (degenerate rollout rewards),
    not a proof failure or cheating. The scarce GRAIL candidate budget is
    charged only AFTER the reward-zone gate passes, so a degenerate submission
    never consumes it — no refund needed, and a high out_of_zone rate (e.g.
    opencode's binary rewards) cannot starve the env below B distinct. The
    grading-attempts capacity (anti-DoS / queue bound) IS reserved at admission,
    then charged irreversibly only when grading starts.
    """
    b = _make_batcher()
    req = _request(rewards=[1.0] * 8, hotkey="hkz")  # degenerate -> out_of_zone

    admitted, _ = b.try_reserve_proof_admission(req)
    assert admitted
    assert b.proof_admission_count == 0      # GRAIL budget NOT charged at admission
    assert b.proof_grading_attempts == 0
    assert b.pending_proof_reservations == 1
    assert b.start_proof_admission(req) == (True, None)
    assert b.proof_grading_attempts == 1

    try:
        resp = b.accept_submission(req)
    finally:
        b.finish_proof_admission(req)
    assert resp.reason == RejectReason.OUT_OF_ZONE

    assert b.proof_admission_count == 0      # never reached GRAIL -> never charged
    assert b.proof_grading_attempts == 1     # grading ceiling unaffected


def test_grading_attempts_ceiling_blocks_admission(monkeypatch):
    """The grading-attempts ceiling bounds total grading work and gates
    admission — even when no submission ever charges the GRAIL budget (here all
    are out_of_zone). Otherwise a degenerate-reward flood would grow the
    unbounded submit queue without limit."""
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW", 3)
    b = _make_batcher()
    for i in range(3):
        req = _request(rewards=[1.0] * 8, hotkey=f"hk{i}")
        ok, _ = b.try_reserve_proof_admission(req)
        assert ok
        assert b.start_proof_admission(req) == (True, None)
        try:
            b.accept_submission(req)  # out_of_zone -> never charges GRAIL budget
        finally:
            b.finish_proof_admission(req)

    assert b.proof_admission_count == 0          # GRAIL budget never charged
    assert b.proof_grading_attempts == 3         # ceiling reached, not refunded

    ok, reason = b.try_reserve_proof_admission(
        _request(rewards=[1.0] * 8, hotkey="hkX")
    )
    assert ok is False
    assert reason == "proof_grading_attempts_full"


def test_cancelled_pending_reservation_returns_unused_capacity(monkeypatch):
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW", 1)
    b = _make_batcher()
    dropped = _request(hotkey="hk-drop")
    replacement = _request(prompt_idx=43, hotkey="hk-replacement")

    assert b.try_reserve_proof_admission(dropped) == (True, None)
    assert b.try_reserve_proof_admission(replacement) == (
        False,
        "proof_grading_attempts_full",
    )
    assert b.cancel_proof_admission(dropped) is True
    assert b.proof_grading_attempts == 0
    assert b.pending_proof_reservations == 0
    assert b.try_reserve_proof_admission(replacement) == (True, None)


def test_started_attempt_is_never_refunded(monkeypatch):
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW", 1)
    b = _make_batcher()
    started = _request(hotkey="hk-started")
    later = _request(prompt_idx=43, hotkey="hk-later")

    assert b.try_reserve_proof_admission(started) == (True, None)
    assert b.start_proof_admission(started) == (True, None)
    b.finish_proof_admission(started)

    assert b.proof_grading_attempts == 1
    assert b.inflight_proof_reservations == 0
    assert b.try_reserve_proof_admission(later) == (
        False,
        "proof_grading_attempts_full",
    )


def test_pending_burst_rechecks_hotkey_debt_at_proof_start():
    from reliquary.constants import (
        MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW,
    )

    b = _make_batcher()
    pending = _request(hotkey="hk-burst")
    assert b.try_reserve_proof_admission(pending) == (True, None)
    for _ in range(MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW):
        b._reject(
            RejectReason.TOKEN_TAMPERED,
            hotkey="hk-burst",
            prompt_idx=42,
            reject_stage="forced_seed",
        )

    assert b.start_proof_admission(pending) == (
        False,
        "proof_failure_debt_hotkey",
    )
    assert b.pending_proof_reservations == 0
    assert b.proof_grading_attempts == 0


def test_reject_manufactured_opposite_reward_clones_before_grail_compute():
    def fail_if_called(commit, model, randomness):  # pragma: no cover
        raise AssertionError("GRAIL proof should not run for clone-pattern rejects")

    def paired_text(slot: int, answer: int, *, correct: bool) -> str:
        verdict = "CORRECT" if correct else "near miss"
        return (
            f"To solve template {slot}, compute the same intermediate values. "
            "First expand the expression, then simplify every term carefully. "
            "The derivation is intentionally long enough to look like a real "
            "math completion and all steps are repeated across the paired "
            "rollouts. After substitution, the final numeric value is "
            f"{answer}. Therefore the final answer is \\boxed{{{answer}}}. "
            f"{verdict}"
        )

    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    texts: dict[int, str] = {}
    for slot in range(4):
        texts[slot] = paired_text(slot, 40, correct=True)
        texts[slot + 4] = paired_text(slot, 41 + slot, correct=False)

    b = _make_batcher(
        completion_text_fn=lambda rollout: texts[rollout.tokens[0]],
        verify_commitment_proofs_fn=fail_if_called,
    )

    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.DISTRIBUTION_SUSPICIOUS


def test_allow_less_than_three_opposite_reward_clone_pairs():
    def paired_text(slot: int, answer: int, *, correct: bool) -> str:
        verdict = "CORRECT" if correct else "near miss"
        return (
            f"To solve template {slot}, compute the same intermediate values. "
            "First expand the expression, then simplify every term carefully. "
            "Only two opposite-reward pairs are deliberately similar here. "
            f"The final answer is \\boxed{{{answer}}}. {verdict}"
        )

    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    texts = {
        0: paired_text(0, 40, correct=True),
        1: paired_text(1, 40, correct=True),
        2: "A separate solution path reaches the answer. CORRECT",
        3: "Another unrelated derivation also reaches the answer. CORRECT",
        4: paired_text(0, 41, correct=False),
        5: paired_text(1, 42, correct=False),
        6: "This wrong path makes an arithmetic slip and ends elsewhere.",
        7: "This different wrong path estimates instead of calculating.",
    }
    b = _make_batcher(completion_text_fn=lambda rollout: texts[rollout.tokens[0]])

    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_accept_all_sigma_zone_binary_configs(k):
    b = _make_batcher()
    req = _request(rewards=[1.0] * k + [0.0] * (M_ROLLOUTS - k))
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED


def test_reject_grail_fail():
    """The GRAIL proof runs at seal now, so admission accepts and the reject
    lands in _verify_expensive."""
    b = _make_batcher(verify_commitment_proofs_fn=_always_false_grail)
    req = _request()
    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.GRAIL_FAIL.value] == 1


def test_reject_reward_mismatch():
    # Override completion_text_fn to always return "wrong", creating a reward mismatch
    # when claim is 1.0
    b = _make_batcher(completion_text_fn=lambda rollout: "wrong")
    rollouts = []
    for i in range(M_ROLLOUTS):
        commit = _make_commit(success=False, total_reward=0.0)
        # Claim high reward for first 4, but completion_text_fn will return "wrong"
        # which computes to 0.0 reward, triggering REWARD_MISMATCH
        claimed_reward = 1.0 if i < 4 else 0.0
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"],
                reward=claimed_reward,
                commit=commit,
                env_name="openmathinstruct",
            )
        )
    req = BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=42,
        window_start=500,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
        protocol_version=2,
    )
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.REWARD_MISMATCH


def test_validator_authoritative_reward_overwrites_placeholder_claims():
    """Private-reward envs cannot require miners to know hidden cases."""
    req = _request(rewards=[0.0] * M_ROLLOUTS)
    for rollout in req.rollouts:
        rollout.env_name = "opencodeinstruct"

    b = _make_batcher(
        env=PrivateRewardFakeEnv(),
        completion_text_fn=lambda rollout: (
            "CORRECT" if int(rollout.tokens[0]) < 4 else "wrong"
        ),
    )

    resp = b.accept_submission(req)
    assert resp.accepted is True, resp.reason
    b.seal_batch()  # validator-authoritative reward is applied at admission;
    # the proof at seal promotes it into _valid unchanged.

    expected = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert [r.reward for r in b._valid[0].rollouts] == expected
    assert [
        r.commit["rollout"]["total_reward"] for r in b._valid[0].rollouts
    ] == expected
    assert [
        r.commit["rollout"]["success"] for r in b._valid[0].rollouts
    ] == [True, True, True, True, False, False, False, False]


def test_validator_authoritative_non_finite_reward_is_rejected():
    class NonFinitePrivateRewardEnv(PrivateRewardFakeEnv):
        def compute_reward(self, problem, completion):
            return float("nan")

    req = _request(rewards=[0.0] * M_ROLLOUTS)
    for rollout in req.rollouts:
        rollout.env_name = "opencodeinstruct"

    b = _make_batcher(env=NonFinitePrivateRewardEnv())
    response = b.accept_submission(req)

    assert response.accepted is False
    assert response.reason == RejectReason.REWARD_MISMATCH


def test_reject_outer_inner_token_split_even_if_constructed():
    b = _make_batcher()
    req = _request()
    honest_tokens = list(req.rollouts[0].commit["tokens"])
    fake_outer_tokens = honest_tokens[:-1] + [999]
    req.rollouts[0] = RolloutSubmission.model_construct(
        tokens=fake_outer_tokens,
        reward=req.rollouts[0].reward,
        commit=req.rollouts[0].commit,
    )

    resp = b.accept_submission(req)

    assert resp.accepted is False
    assert resp.reason == RejectReason.TOKENS_MISMATCH


# --- seal_batch + cooldown lifecycle ---

def test_seal_batch_empty_pool_returns_empty():
    b = _make_batcher()
    batch, rewards = b.seal_batch()
    assert batch == [] and rewards == {}


def test_seal_batch_chronological_by_drand_round():
    """v2.3 (design A'): submissions in earlier drand rounds fill the batch
    first, regardless of TCP arrival order."""
    b = _make_batcher()
    # Both submissions accepted; round 1 must come out first at seal time
    # even though round 2 arrived first (insertion order below).
    req_late = _request(prompt_idx=42, hotkey="late")
    req_early = _request(prompt_idx=7, hotkey="early")
    # Stamp the rounds on the requests (the check is disabled in the helper, so
    # accept_submission won't re-validate them) — the proof carries drand_round
    # onto the ValidSubmission, which drives seal-time ordering.
    req_late.drand_round = 2  # late: round 2
    req_early.drand_round = 1  # early: round 1
    assert b.accept_submission(req_late).accepted
    assert b.accept_submission(req_early).accepted
    batch, _ = b.seal_batch()
    assert len(batch) == 2
    assert batch[0].hotkey == "early"
    assert batch[1].hotkey == "late"


def test_seal_batch_cooldown_recorded():
    b = _make_batcher()
    req = _request(prompt_idx=42)
    b.accept_submission(req)
    batch, rewards = b.seal_batch()
    assert len(batch) == 1
    assert b._cooldown.is_in_cooldown(42, b.window_start + 1) is True
    # Each slot pays pool / B_BATCH. One slot filled, K_p=1 → 1/8.
    assert abs(rewards["hk"] - 1 / B_BATCH) < 1e-9


def test_sealed_batch_respects_cooldown_from_previous_window():
    from reliquary.validator.cooldown import CooldownMap
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS
    cd = CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
    cd.record_batched(prompt_idx=42, window=100)
    b = _make_batcher(window_start=120, cooldown_map=cd)
    req = _request(prompt_idx=42, window_start=120)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_IN_COOLDOWN


def test_state_endpoint_exposes_cooldown():
    from reliquary.validator.cooldown import CooldownMap
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS
    cd = CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
    cd.record_batched(prompt_idx=42, window=100)
    cd.record_batched(prompt_idx=7, window=105)
    b = _make_batcher(window_start=110, cooldown_map=cd)
    state = b.get_state()
    assert set(state.cooldown_prompts) == {42, 7}
    assert state.valid_submissions == 0


def test_distinct_prompts_in_batch_only():
    """One submission per winning prompt enters the training batch, even when
    multiple miners successfully submitted on the same prompt.

    BEHAVIOUR CHANGE (ranked proving): only the top-ranked candidate per prompt
    is proven, so a prompt's slot now pays ONE miner in full instead of being
    split K ways. Alice and Bob tie on value and drand round, so the canonical
    tie-break picks one — expected-value identical to the old 1/16 split, still
    sybil-neutral (N hotkeys on one prompt win at most that prompt's one slot).
    """
    b = _make_batcher()
    b.accept_submission(_request(prompt_idx=42, hotkey="alice"))
    b.accept_submission(_request(prompt_idx=42, hotkey="bob"))
    b.accept_submission(_request(prompt_idx=7, hotkey="carol"))
    batch, rewards = b.seal_batch()
    assert len(batch) == 2
    assert {s.prompt_idx for s in batch} == {42, 7}
    # Each filled slot pays pool / B_BATCH = 1/8, to the prompt's single winner.
    prompt_42_winner = {"alice", "bob"} & set(rewards)
    assert len(prompt_42_winner) == 1
    assert abs(rewards[prompt_42_winner.pop()] - 1 / 8) < 1e-9
    assert abs(rewards["carol"] - 1 / 8) < 1e-9


# --- v2.1 seal_event + checkpoint_hash gating ---

import asyncio

import pytest


def _request_v21(prompt_idx=42, window_start=500,
                 rewards=None, hotkey="hk", checkpoint_hash="sha256:abc"):
    """v2.1 request: includes the required checkpoint_hash field."""
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for r in rewards:
        commit = _make_commit(success=r > 0.5, total_reward=r)
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"], reward=r,
                commit=commit,
                env_name="openmathinstruct",
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey, prompt_idx=prompt_idx,
        window_start=window_start,
        merkle_root="00" * 32, rollouts=rollouts,
        checkpoint_hash=checkpoint_hash,
        protocol_version=FORCED_SEED_PROTOCOL_VERSION,
    )


def test_reject_wrong_checkpoint():
    """Submission with checkpoint_hash != batcher's current is rejected."""
    b = _make_batcher()
    b.current_checkpoint_hash = "sha256:current"
    req = _request_v21(checkpoint_hash="sha256:stale")
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WRONG_CHECKPOINT


def test_accept_matching_checkpoint():
    b = _make_batcher()
    b.current_checkpoint_hash = "sha256:current"
    req = _request_v21(checkpoint_hash="sha256:current")
    resp = b.accept_submission(req)
    assert resp.accepted is True


def test_reject_old_forced_seed_protocol_before_auction_admission():
    b = _make_batcher()
    b.current_checkpoint_hash = "sha256:current"
    req = _request_v21(checkpoint_hash="sha256:current")
    req.protocol_version = FORCED_SEED_PROTOCOL_VERSION - 1

    resp = b.accept_submission(req)

    assert resp.accepted is False
    assert resp.reason == RejectReason.SEED_MISMATCH
    assert b.pending_count == 0
    assert b.proof_grading_attempts == 0


def test_empty_checkpoint_hash_disables_gate():
    """When batcher.current_checkpoint_hash is "", any hash is accepted
    (test convenience — simulates pre-first-publish)."""
    b = _make_batcher()
    b.current_checkpoint_hash = ""
    req = _request_v21(checkpoint_hash="anything")
    resp = b.accept_submission(req)
    assert resp.accepted is True


def test_seal_event_not_set_with_only_duplicate_prompts():
    """Two submissions on same prompt → only first counts → seal_event not set."""
    b = _make_batcher()
    b.current_checkpoint_hash = "sha256:hash"
    for i in range(2):
        req = _request_v21(
            prompt_idx=42, hotkey=f"hk{i}",
            checkpoint_hash="sha256:hash",
        )
        b.accept_submission(req)
    # Only 1 distinct prompt → not enough for seal
    assert not b.seal_event.is_set()


def test_seal_event_not_set_with_fewer_than_b():
    """Fewer than B valid submissions → no seal."""
    b = _make_batcher()
    b.current_checkpoint_hash = "sha256:hash"
    for i in range(B_BATCH - 1):
        req = _request_v21(
            prompt_idx=i, hotkey=f"hk{i}",
            checkpoint_hash="sha256:hash",
        )
        b.accept_submission(req)
    assert not b.seal_event.is_set()


# ---------------------------------------------------------------------------
# Prompt-binding (canonical_prompt_tokens_fn)
# ---------------------------------------------------------------------------
#
# A miner can pass every other check while having generated under a modified
# prompt (CoT prefix, alternate chat template, few-shot examples) by:
#   1. Running their forward pass on prompt_modified
#   2. Sending the resulting completions + GRAIL sketch to the validator
#   3. Claiming the canonical prompt_idx
# GRAIL alone won't catch this because the validator re-runs forward on the
# *miner-supplied tokens* — both produce the same sketch.
#
# canonical_prompt_tokens_fn closes the gap: the validator computes the
# canonical prompt tokens for the claimed prompt_idx from its own env +
# tokenizer, and rejects any submission whose tokens[:prompt_length] diverges.


def _request_with_prompt_tokens(
    *, prompt_idx: int, prompt_tokens: list[int],
    completion_tokens: list[int] | None = None,
    rewards: list[float] | None = None, hotkey: str = "hk",
):
    """Like ``_request`` but sets ``commit['rollout']['prompt_length']`` and
    builds ``commit['tokens']`` = prompt + completion explicitly so the
    validator's prompt-binding check has something to inspect.

    Pads completion_tokens to ensure total sequence length >= CHALLENGE_K so
    CommitModel schema validation passes.
    """
    prompt_list = list(prompt_tokens)
    if completion_tokens is None:
        completion_tokens = [99]
    comp_list = list(completion_tokens)
    # Ensure total >= CHALLENGE_K (CommitModel min_length requirement)
    min_comp_len = max(len(comp_list), CHALLENGE_K - len(prompt_list))
    if len(comp_list) < min_comp_len:
        comp_list = comp_list + [0] * (min_comp_len - len(comp_list))
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for r in rewards:
        full_tokens = prompt_list + comp_list
        commit = _make_commit(
            tokens=full_tokens,
            prompt_length=len(prompt_list),
            success=r > 0.5,
            total_reward=r,
        )
        rollouts.append(
            RolloutSubmission(
                tokens=full_tokens, reward=r,
                commit=commit,
                env_name="openmathinstruct",
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey, prompt_idx=prompt_idx,
        window_start=500,
        merkle_root="00" * 32, rollouts=rollouts,
        checkpoint_hash="",  # gate disabled for these tests
        protocol_version=2,
    )


def test_prompt_mismatch_rejected_when_canonical_differs():
    """Miner runs forward pass on a CoT-prefixed prompt but claims the
    canonical prompt_idx → validator detects the prompt_tokens don't match
    its env's canonical version → PROMPT_MISMATCH before any GRAIL compute."""
    canonical = [10, 11, 12]            # what the env says prompt 42 is
    miner_used = [99, 10, 11, 12]       # CoT prefix + canonical question

    b = _make_batcher(
        canonical_prompt_tokens_fn=lambda idx: canonical if idx == 42 else [],
    )
    req = _request_with_prompt_tokens(
        prompt_idx=42, prompt_tokens=miner_used, completion_tokens=[200, 201],
    )
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_MISMATCH


def test_prompt_match_accepted_when_canonical_equals():
    """Honest miner: prompt_tokens match the env's canonical version → check
    is a no-op, submission proceeds through the rest of the pipeline."""
    canonical = [10, 11, 12]
    b = _make_batcher(
        canonical_prompt_tokens_fn=lambda idx: canonical if idx == 42 else [],
    )
    req = _request_with_prompt_tokens(
        prompt_idx=42, prompt_tokens=canonical, completion_tokens=[200, 201],
    )
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED


def test_no_canonical_fn_disables_check():
    """When ``canonical_prompt_tokens_fn`` is None (test stubs), the binding
    check is skipped — preserves backward compatibility for existing tests
    that don't carry a real tokenizer."""
    b = _make_batcher()  # no canonical_prompt_tokens_fn passed
    # Use an arbitrary prompt_tokens; without a canonical, nothing to compare.
    req = _request_with_prompt_tokens(
        prompt_idx=42, prompt_tokens=[7, 8, 9],
    )
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED


# ---------------------------------------------------------------------------
# Per-prompt multi-miner acceptance (v2.3+)
# ---------------------------------------------------------------------------
#
# Drand-anchored ordering at seal time replaced the FIFO SUPERSEDED short-
# circuit. Multiple miners may submit on the same prompt within a window,
# capped at MAX_SUBMISSIONS_PER_PROMPT. Each pays its own GRAIL verify; the
# cap is the only thing bounding worst-case validator GPU load.


def test_same_prompt_multi_miner_accepted():
    """v2.3: two different miners on the same prompt both pass verification.
    The first does not 'claim' the prompt — emission is split at seal time."""
    b = _make_batcher()
    first = _request_v21(prompt_idx=42, hotkey="A", checkpoint_hash="")
    second = _request_v21(prompt_idx=42, hotkey="B", checkpoint_hash="")
    assert b.accept_submission(first).accepted is True
    r2 = b.accept_submission(second)
    assert r2.accepted is True
    # Admitted to the pending pool; the same-prompt winner is resolved at seal.
    assert len(b.pending_submissions()) == 2
    assert b.prompt_submission_count(42) == 2


def test_prompt_full_rejected_at_cap():
    """Beyond MAX_SUBMISSIONS_PER_PROMPT submissions on a prompt, further
    arrivals are rejected PROMPT_FULL before any heavy verify."""
    from reliquary.constants import MAX_SUBMISSIONS_PER_PROMPT
    b = _make_batcher()
    for i in range(MAX_SUBMISSIONS_PER_PROMPT):
        req = _request_v21(prompt_idx=42, hotkey=f"hk{i}", checkpoint_hash="")
        assert b.accept_submission(req).accepted is True, f"miner {i} should fit"
    overflow = _request_v21(prompt_idx=42, hotkey="overflow", checkpoint_hash="")
    resp = b.accept_submission(overflow)
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_FULL
    # The PROMPT_FULL reject did not enter the pending bucket.
    assert b.prompt_submission_count(42) == MAX_SUBMISSIONS_PER_PROMPT
    assert len(b.pending_submissions()) == MAX_SUBMISSIONS_PER_PROMPT


def test_different_prompts_tracked_independently():
    """Each prompt has its own bucket — filling prompt 42 must not affect
    prompt 43's acceptance budget."""
    b = _make_batcher()
    r_a = b.accept_submission(_request_v21(
        prompt_idx=42, hotkey="A", checkpoint_hash="",
    ))
    r_b = b.accept_submission(_request_v21(
        prompt_idx=43, hotkey="A", checkpoint_hash="",
    ))
    assert r_a.accepted is True
    assert r_b.accepted is True
    assert len(b.pending_submissions()) == 2
    assert set(b._submissions_per_prompt) == {42, 43}


def test_failed_submission_does_not_consume_bucket_slot():
    """v2 (deferred proof): admission is cheap and always buckets, so a
    submission that later FAILS its seal-time proof still occupied a pending
    slot (the cap bounds pending, bounding worst-case GPU load). The
    anti-starvation invariant moved to seal: a squatter that fails the proof
    does NOT lock the prompt — the honest same-prompt submission behind it is
    promoted and wins the slot."""
    import torch
    from reliquary.validator.verifier import ProofResult

    calls = {"n": 0}

    def grail_fail_first(commit, model, randomness):
        # The top-ranked candidate (the squatter, admitted first) fails on its
        # first proved rollout; every later proof call passes.
        calls["n"] += 1
        passed = calls["n"] > 1
        return ProofResult(
            all_passed=passed, passed=int(passed), checked=1,
            logits=torch.empty(0),
        )

    b = _make_batcher(verify_commitment_proofs_fn=grail_fail_first)
    squatter = _request_v21(prompt_idx=42, hotkey="A", checkpoint_hash="")
    honest = _request_v21(prompt_idx=42, hotkey="B", checkpoint_hash="")
    assert b.accept_submission(squatter).accepted is True
    assert b.accept_submission(honest).accepted is True
    # Both occupy the pending bucket (cheap admission).
    assert b.prompt_submission_count(42) == 2

    batch, rewards = b.seal_batch()
    # The squatter failed its proof; the honest miner behind it wins the slot.
    assert b.reject_counts.get(RejectReason.GRAIL_FAIL.value) == 1
    failed_hotkey = next(
        row["hotkey"]
        for row in b.auction_candidates
        if row["status"] == "proof_failed"
    )
    assert len(batch) == 1
    assert batch[0].hotkey in {"A", "B"} - {failed_hotkey}
    assert set(rewards) == {batch[0].hotkey}


# ---------------------------------------------------------------------------
# Drand-round timing gate (v2.3 design A')
# ---------------------------------------------------------------------------

def _make_batcher_with_drand_check(*, fixed_round: int = 100, **overrides):
    """Helper: batcher with the drand timing gate ENABLED and a stable
    chain info so we can reason about exact round numbers in tests."""
    # Pin wall clock so current_round = fixed_round.
    # genesis_time=1000, period=3 → current_round at wall_clock = 1000 +
    # (fixed_round - 1) * 3 = 1000 + (fixed_round - 1) * 3.
    wall = 1000 + (fixed_round - 1) * 3 + 1.0  # +1s into the round
    overrides.setdefault("drand_round_check_enabled", True)
    overrides.setdefault("wall_clock_fn", lambda: wall)
    overrides.setdefault(
        "drand_chain_info", {"genesis_time": 1000, "period": 3},
    )
    return _make_batcher(**overrides)


# The drand-round timing gate is now an HTTP-arrival-only check (see
# ``server.py``'s ``stamp_arrival`` middleware + cheap-reject path). The
# batcher's ``_accept_locked`` no longer re-validates drand_round at worker
# dequeue — that re-check was using ``time.time()`` minutes after arrival
# under GRAIL queue backpressure and turning on-time submissions into
# spurious STALE_ROUND rejections (the bug this refactor removes). These
# tests pin the gate's behaviour via the public ``validate_drand_round``
# method that the HTTP cheap-reject calls directly.

def test_drand_round_current_accepted():
    b = _make_batcher_with_drand_check(fixed_round=100)
    assert b.validate_drand_round(100) is None


def test_drand_round_one_behind_stale_under_default_tolerance():
    """Default ``DRAND_ROUND_BACKWARD_TOLERANCE = 0`` is strict equality:
    a miner whose attached round is even one behind the validator's
    current is STALE_ROUND. Honest miners use the miner-side boundary-
    safety check (``RELIQUARY_DRAND_BOUNDARY_SAFETY_S``) to ensure they
    don't fire right on a drand-round boundary, so RTT can't push them
    across. Anything one behind at the validator is either a clock-lag
    miner or an antedating attempt.
    """
    b = _make_batcher_with_drand_check(fixed_round=100)
    assert b.validate_drand_round(99) == RejectReason.STALE_ROUND


def test_drand_round_future_rejected():
    """Forward direction is always zero-tolerance: a future round means
    the miner claims to have seen a beacon that hasn't been signed yet."""
    b = _make_batcher_with_drand_check(fixed_round=100)
    assert b.validate_drand_round(101) == RejectReason.FUTURE_ROUND


def test_drand_round_zero_tolerance_one_behind_stale():
    """Explicit ``drand_round_backward_tolerance = 0`` matches the
    default — one round behind is STALE. Kept as an explicit pin for
    readability (so a future widening of the default doesn't silently
    break the zero-tolerance contract callers rely on)."""
    b = _make_batcher_with_drand_check(
        fixed_round=100, drand_round_backward_tolerance=0,
    )
    assert b.validate_drand_round(99) == RejectReason.STALE_ROUND


def test_drand_round_explicit_tolerance_one_allows_one_behind():
    """Operators with cross-continent RTT profiles can override the
    tolerance via the env var or per-batcher kwarg. Pin that an
    explicit ``tolerance = 1`` accepts the previous-round case so
    operators have a documented escape hatch."""
    b = _make_batcher_with_drand_check(
        fixed_round=100, drand_round_backward_tolerance=1,
    )
    assert b.validate_drand_round(99) is None  # within explicit tol=1


def test_drand_round_default_backward_tolerance_is_zero():
    """Pin the default. Changing this in constants is a deliberate
    protocol-tuning decision, not an incidental refactor — make any
    drift loud.

    History:
      * v2.3 original spec: 0 (strict equality).
      * commit 8b7f483: 1 (absorb RTT boundary crossing).
      * PR #31 (1f5d4e7): 10 (absorb worker-side dequeue lag during
        GRAIL queue backpressure).
      * commit 62cbf3f: 1 (worker re-check removed, only RTT remains).
      * This commit: 0 (arrival-time stamping + seal extension + drain
        wait + miner-side boundary safety eliminate the remaining
        legitimate sources of STALE for honest miners).

    Operators can re-widen via the ``DRAND_ROUND_BACKWARD_TOLERANCE``
    env var if their cross-continent RTT profile demands it.
    """
    import os
    # Pin the *unset* default — env-var override would skew the test.
    prior = os.environ.pop("DRAND_ROUND_BACKWARD_TOLERANCE", None)
    try:
        import importlib
        import reliquary.constants
        importlib.reload(reliquary.constants)
        assert reliquary.constants.DRAND_ROUND_BACKWARD_TOLERANCE == 0
    finally:
        if prior is not None:
            os.environ["DRAND_ROUND_BACKWARD_TOLERANCE"] = prior
        import importlib
        import reliquary.constants
        importlib.reload(reliquary.constants)


def test_drand_round_backward_tolerance_env_var_override():
    """``DRAND_ROUND_BACKWARD_TOLERANCE`` env var overrides the constant.
    Lets operators tune for their validator's typical stall profile
    without a code push — same ``RELIQUARY_*``-style ergonomic the
    miner / validator CLI already exposes for env name and resume."""
    import os
    import importlib
    import reliquary.constants
    prior = os.environ.get("DRAND_ROUND_BACKWARD_TOLERANCE")
    os.environ["DRAND_ROUND_BACKWARD_TOLERANCE"] = "25"
    try:
        importlib.reload(reliquary.constants)
        assert reliquary.constants.DRAND_ROUND_BACKWARD_TOLERANCE == 25
    finally:
        if prior is None:
            os.environ.pop("DRAND_ROUND_BACKWARD_TOLERANCE", None)
        else:
            os.environ["DRAND_ROUND_BACKWARD_TOLERANCE"] = prior
        importlib.reload(reliquary.constants)


def test_drand_round_explicit_tolerance_three_allows_three_behind():
    """Tolerance is a per-batcher knob — operators can dial it down when
    arrival-time stamping makes the wide default less necessary."""
    b = _make_batcher_with_drand_check(
        fixed_round=100, drand_round_backward_tolerance=3,
    )
    assert b.validate_drand_round(97) is None  # current - 3


def test_drand_round_explicit_tolerance_three_rejects_four_behind():
    """Tolerance = 3 still rejects four rounds behind — the gate is a
    hard cliff at ``current - tolerance``, not a soft penalty."""
    b = _make_batcher_with_drand_check(
        fixed_round=100, drand_round_backward_tolerance=3,
    )
    assert b.validate_drand_round(96) == RejectReason.STALE_ROUND  # current-4


# ---------------------------------------------------------------------------
# Arrival-time stamping (decouples drand check from handler-execution latency)
# ---------------------------------------------------------------------------

def test_drand_round_arrival_stamp_overrides_wall_clock():
    """When ``t_arrival`` is provided and lies inside an earlier drand round
    than the batcher's current wall clock, the check uses ``t_arrival``.
    This decouples the gate from validator-side processing latency — a
    submission that landed inside its round is still accepted even if the
    handler runs many rounds later (event-loop stall behind trainer GIL).

    Pinned with ``drand_round_backward_tolerance = 0`` so the test is sharp
    against the wall-clock vs arrival-time split without the production
    10-round tolerance papering over it.
    """
    # batcher's wall clock returns a time in round 110; t_arrival is in
    # round 100 (~30 s earlier).
    b = _make_batcher_with_drand_check(
        fixed_round=110, drand_round_backward_tolerance=0,
    )
    t_arrival = 1000 + (100 - 1) * 3 + 1.0  # +1 s into round 100

    # With t_arrival, round 100 IS the current round → accepted.
    assert b.validate_drand_round(100, t_arrival=t_arrival) is None

    # Without t_arrival, the batcher's wall clock dominates → current=110,
    # tolerance=0 → round 100 is STALE.
    assert (
        b.validate_drand_round(100) == RejectReason.STALE_ROUND
    )


def test_drand_round_arrival_stamp_still_rejects_antedating():
    """Arrival-time stamping doesn't weaken the antedating defense: a
    miner claiming a round many drand periods earlier than ``t_arrival``
    is still STALE. The bound is still ``tolerance × period`` of
    chronological antedate — the cap moves with the timestamp, but the
    cap itself doesn't go away.
    """
    b = _make_batcher_with_drand_check(
        fixed_round=110, drand_round_backward_tolerance=2,
    )
    t_arrival = 1000 + (100 - 1) * 3 + 1.0  # in round 100
    # Antedating attempt: claim 50 rounds earlier than arrival.
    assert (
        b.validate_drand_round(50, t_arrival=t_arrival)
        == RejectReason.STALE_ROUND
    )
    # At the cap (current=100 by t_arrival, tolerance=2) round 98 is OK.
    assert b.validate_drand_round(98, t_arrival=t_arrival) is None
    # One past the cap is STALE.
    assert (
        b.validate_drand_round(97, t_arrival=t_arrival)
        == RejectReason.STALE_ROUND
    )


def test_drand_round_arrival_stamp_future_still_rejected():
    """A miner attaching a round AHEAD of ``t_arrival`` still gets
    FUTURE_ROUND. Forward direction stays zero-tolerance whether the
    check timestamp comes from arrival stamp or wall clock — claiming a
    beacon you haven't seen is unrecoverable cheating in both cases.
    """
    b = _make_batcher_with_drand_check(fixed_round=110)
    t_arrival = 1000 + (100 - 1) * 3 + 1.0  # in round 100
    assert (
        b.validate_drand_round(101, t_arrival=t_arrival)
        == RejectReason.FUTURE_ROUND
    )


def test_drand_round_falls_back_to_wall_clock_when_no_arrival():
    """``t_arrival=None`` means use ``_wall_clock()`` — backward-compatible
    for callers that don't go through the HTTP middleware (direct unit
    tests, the worker re-check, integration fixtures from before this
    feature)."""
    b = _make_batcher_with_drand_check(fixed_round=100)
    assert b.validate_drand_round(100) is None
    assert b.validate_drand_round(101) == RejectReason.FUTURE_ROUND


def test_accept_submission_no_longer_rechecks_drand_round():
    """Regression: ``accept_submission`` (the worker path) must NOT
    re-validate drand_round. Drand is an arrival-time gate decided once
    by the HTTP cheap-reject path; re-checking here would use
    ``time.time()`` at worker dequeue, which under GRAIL queue
    backpressure can be minutes after arrival — turning on-time
    submissions into STALE_ROUND rejections (the bug this design fixes).

    A drand_round that the cheap-reject would have rejected as STALE
    must STILL be accepted by ``accept_submission`` if it passes the
    other gates. The arrival path is the single source of truth for
    drand timing.
    """
    b = _make_batcher_with_drand_check(fixed_round=100)
    req = _request_v21(prompt_idx=42, hotkey="A", checkpoint_hash="")
    req.drand_round = 50  # far stale — would be rejected at HTTP arrival

    # Direct ``validate_drand_round`` would reject (cheap-reject path).
    assert b.validate_drand_round(50) == RejectReason.STALE_ROUND

    # But ``accept_submission`` does NOT re-check, so it proceeds past
    # the drand gate. (It may still reject for other reasons like GRAIL,
    # but never STALE_ROUND from the worker path.)
    resp = b.accept_submission(req)
    assert resp.reason != RejectReason.STALE_ROUND, (
        "worker path must not re-validate drand_round — that re-check "
        "is the staleness-on-queue-wait bug we're fixing"
    )


def test_constructor_accepts_tokenizer():
    """Tokenizer must be passable to the batcher (used by TerminationValidator)."""
    class FakeTokenizer:
        eos_token_id = 99

    fake_tok = FakeTokenizer()
    b = _make_batcher(tokenizer=fake_tok)
    assert b.tokenizer is fake_tok


import torch
from reliquary.validator.verifier import ProofResult


def _grail_with_logits(seq_len: int, eos_id: int = 99):
    """Stub that opts into behavioural checks with high EOS probability
    at the termination position. The actual EOS pass/fail depends on
    whether the surrounding model/tokenizer stub declares ``eos_id`` —
    this fixture leaves that to the test setup so we can exercise both
    the EOS-present and EOS-missing termination paths.
    """
    def _fn(commit, model, randomness):
        prompt_length = int(
            (commit.get("rollout") or {}).get("prompt_length", 0)
        )
        challenge_idxs = list(range(prompt_length, prompt_length + CHALLENGE_K))
        return ProofResult(
            all_passed=True, passed=1, checked=1, sketch_diff_max=0,
            has_sparse_outputs=True,
            p_stop=0.99,
            challenge_lp_indices=challenge_idxs,
            challenge_lp_values=[0.0] * CHALLENGE_K,
        )
    return _fn


# ----- SchemaValidator wiring -----

def test_reject_bad_schema_missing_proof_version():
    b = _make_batcher()
    req = _request()
    # Mutate one rollout's commit to break schema
    req.rollouts[0].commit.pop("proof_version")
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_SCHEMA


def test_reject_bad_schema_extra_field():
    b = _make_batcher()
    req = _request()
    req.rollouts[0].commit["unauthorized_field"] = "x"
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_SCHEMA


def test_reject_bad_schema_inconsistent_lengths():
    b = _make_batcher()
    req = _request()
    req.rollouts[0].commit["rollout"]["prompt_length"] = 999
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_SCHEMA


# ----- TokenValidator wiring -----
# The verify_tokens function in protocol/tokens.py is now wired into the
# batcher AFTER schema validation. We stub model.config so verify_tokens
# can resolve vocab_size.

class _ModelStubWithVocab:
    """Minimal stub satisfying resolve_vocab_size(model.config)."""
    class config:
        vocab_size = 1000
        max_position_embeddings = 4096


def test_reject_bad_tokens_above_vocab():
    b = _make_batcher(model=_ModelStubWithVocab())
    req = _request()
    # vocab_size=1000, inject a token == vocab_size (out of bounds)
    req.rollouts[0].commit["tokens"] = [1000] * (CHALLENGE_K + 4)
    # Re-sync the outer field so RolloutSubmission stays consistent
    req.rollouts[0].tokens = req.rollouts[0].commit["tokens"]
    # Re-sync commitments + token_logprobs lengths for schema
    req.rollouts[0].commit["commitments"] = [
        {"sketch": 0} for _ in range(CHALLENGE_K + 4)
    ]
    req.rollouts[0].commit["rollout"]["token_logprobs"] = [0.0] * (CHALLENGE_K + 4)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_TOKENS


def test_reject_bad_tokens_negative_id():
    b = _make_batcher(model=_ModelStubWithVocab())
    req = _request()
    req.rollouts[0].commit["tokens"] = [-1] + list(range(CHALLENGE_K + 3))
    req.rollouts[0].tokens = req.rollouts[0].commit["tokens"]
    req.rollouts[0].commit["commitments"] = [
        {"sketch": 0} for _ in range(CHALLENGE_K + 4)
    ]
    req.rollouts[0].commit["rollout"]["token_logprobs"] = [0.0] * (CHALLENGE_K + 4)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_TOKENS


# ----- TerminationValidator wiring -----

def test_reject_bad_termination_when_last_token_not_eos():
    # A non-EOS last token that did NOT hit the cap is a mid-generation stop →
    # BAD_TERMINATION. (Cap-hit truncations go through the cap path and are
    # bounded by MAX_TRUNCATED_PER_SUBMISSION instead.)
    seq_len = CHALLENGE_K + 4
    b = _make_batcher(
        model=_ModelStubWithVocab(),
        verify_commitment_proofs_fn=_grail_with_logits(seq_len),
    )
    req = _request()
    # Last token != 99 (EOS) — sequence ends in seq_len-1
    req.rollouts[0].commit["tokens"] = list(range(seq_len))
    req.rollouts[0].tokens = req.rollouts[0].commit["tokens"]
    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.BAD_TERMINATION.value] == 1


class _LongContextModelStub:
    class config:
        vocab_size = 20000
        # ≥ prompt + MAX_NEW_TOKENS_PROTOCOL_CAP (32768) so a full-cap rollout
        # clears the sequence-length check and reaches the truncation logic.
        max_position_embeddings = 40000


def _request_with_cap_truncations(n_truncated: int, eos_id: int = 99):
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt_length = 4
    cap_seq_len = prompt_length + MAX_NEW_TOKENS_PROTOCOL_CAP
    req = _request()
    # Cap-truncated rollouts: completion filled with a constant non-EOS token
    # so has_eos_padding does not fire on a stray EOS inside the body.
    for idx in range(n_truncated):
        tokens = [10 + idx] * prompt_length + [5] * MAX_NEW_TOKENS_PROTOCOL_CAP
        commit = _make_commit(
            tokens=tokens,
            prompt_length=prompt_length,
            success=req.rollouts[idx].reward > 0.5,
            total_reward=req.rollouts[idx].reward,
        )
        req.rollouts[idx].commit = commit
        req.rollouts[idx].tokens = commit["tokens"]
    # Remaining rollouts: short, naturally EOS-terminated.
    for idx in range(n_truncated, len(req.rollouts)):
        tokens = [20 + idx] * prompt_length + [5] * (CHALLENGE_K - 1) + [eos_id]
        commit = _make_commit(
            tokens=tokens,
            prompt_length=prompt_length,
            success=req.rollouts[idx].reward > 0.5,
            total_reward=req.rollouts[idx].reward,
        )
        req.rollouts[idx].commit = commit
        req.rollouts[idx].tokens = commit["tokens"]
    return req, cap_seq_len


def _set_eos_completion_lengths(req: BatchSubmissionRequest, lengths: list[int]) -> None:
    prompt_length = 4
    for idx, completion_length in enumerate(lengths):
        body_token = 200 + idx
        tokens = (
            [10 + idx] * prompt_length
            + [body_token] * (completion_length - 1)
            + [99]
        )
        commit = _make_commit(
            tokens=tokens,
            prompt_length=prompt_length,
            success=req.rollouts[idx].reward > 0.5,
            total_reward=req.rollouts[idx].reward,
        )
        req.rollouts[idx].commit = commit
        req.rollouts[idx].tokens = commit["tokens"]


def test_reject_cap_path_truncations_over_budget():
    from reliquary.constants import MAX_TRUNCATED_PER_SUBMISSION

    req, seq_len = _request_with_cap_truncations(MAX_TRUNCATED_PER_SUBMISSION + 1)
    b = _make_batcher(
        model=_LongContextModelStub(),
        verify_commitment_proofs_fn=_grail_with_logits(seq_len),
    )
    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.BAD_TERMINATION.value] == 1


def test_accept_cap_path_truncations_at_budget():
    from reliquary.constants import MAX_TRUNCATED_PER_SUBMISSION

    req, seq_len = _request_with_cap_truncations(MAX_TRUNCATED_PER_SUBMISSION)
    b = _make_batcher(
        model=_LongContextModelStub(),
        verify_commitment_proofs_fn=_grail_with_logits(seq_len),
    )
    # At-budget truncation is tolerated: the proof passes at seal.
    assert _prove_one(b, req) is not None


def test_cap_path_records_private_termination_shadow(monkeypatch):
    import reliquary.validator.batcher as batcher_mod

    recorded = []
    monkeypatch.setattr(
        batcher_mod,
        "record_termination_shadow",
        lambda **kwargs: recorded.append(kwargs),
    )
    req, seq_len = _request_with_cap_truncations(1)
    b = _make_batcher(
        model=_LongContextModelStub(),
        verify_commitment_proofs_fn=_grail_with_logits(seq_len),
    )

    # The private termination shadow is recorded during the seal-time proof.
    assert _prove_one(b, req) is not None
    assert len(recorded) == 1
    assert recorded[0]["cap_truncated"] is True
    assert recorded[0]["would_exceed_truncation_budget"] is False
    assert recorded[0]["window_start"] == b.window_start


def test_reward_shape_no_longer_rejects_repeated_zero_tail():
    """The reward-shape filter was removed: it was trivially bypassed
    (reorder rollouts / vary loser lengths) yet false-rejected honest
    miners. A same-length zero tail is no longer grounds for rejection."""
    b = _make_batcher(
        model=_ModelStubWithVocab(),
        verify_commitment_proofs_fn=_grail_with_logits(220),
    )
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    _set_eos_completion_lengths(req, [239, 212, 213, 232, 120, 120, 120, 120])

    resp = b.accept_submission(req)

    assert resp.accepted is True


def test_accept_ordered_rewards_with_varied_zero_lengths():
    b = _make_batcher(
        model=_ModelStubWithVocab(),
        verify_commitment_proofs_fn=_grail_with_logits(220),
    )
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    _set_eos_completion_lengths(req, [239, 212, 213, 232, 120, 133, 147, 161])

    resp = b.accept_submission(req)

    assert resp.accepted is True


def test_reject_eos_padding_after_natural_stop():
    seq_len = CHALLENGE_K + 4
    b = _make_batcher(
        model=_ModelStubWithVocab(),
        verify_commitment_proofs_fn=_grail_with_logits(seq_len),
    )
    req = _request()

    tokens = list(range(seq_len - 2)) + [99, 99]
    commit = _make_commit(
        tokens=tokens,
        success=req.rollouts[0].reward > 0.5,
        total_reward=req.rollouts[0].reward,
    )
    req.rollouts[0].commit = commit
    req.rollouts[0].tokens = commit["tokens"]

    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.BAD_TERMINATION.value] == 1


def test_termination_skipped_when_grail_returns_empty_logits():
    """Backward-compat: when the GRAIL stub returns empty logits, the
    termination check is skipped. The default ``_always_true_grail`` does
    this (it predates the cached-logits path), so the existing
    full-pipeline tests stay green without becoming termination-aware.
    """
    b = _make_batcher(model=_ModelStubWithVocab())
    req = _request()  # default rewards [1,1,1,1,0,0,0,0] → sigma above SIGMA_MIN
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED


# Note on "happy path with logits" test: a positive case where the rollout
# DOES end with EOS *and* survives the full pipeline (logprob + distribution)
# requires synthetic logits whose log_softmax matches the miner-claimed
# token_logprobs (which the test fixture sets to all-zero). Building such a
# fixture pulls the test toward an end-to-end integration test. We cover the
# wiring with the reject case above and the empty-logits skip case; the
# happy path is exercised by the existing pipeline tests through the
# empty-logits branch. A real end-to-end happy path lives in tests/integration.


def test_rejected_submissions_list_initialised_empty():
    from reliquary.validator.batcher import GrpoWindowBatcher, RejectedSubmission
    b = _make_batcher()  # existing helper in this file
    assert hasattr(b, "rejected_submissions")
    assert b.rejected_submissions == []
    # Confirm the dataclass exists and has the documented fields.
    fields = {f.name for f in RejectedSubmission.__dataclass_fields__.values()}
    assert {
        "hotkey", "prompt_idx", "reason",
        "sketch_diff_max", "lp_dev_max", "dist_q10_min",
    }.issubset(fields)


def _empty_logits():
    import torch
    return torch.empty(0)


def _build_request(*, hotkey: str = "hk", prompt_idx: int = 42, window_start: int = 500):
    """Thin wrapper around ``_request`` for the rejection-archive tests."""
    return _request(prompt_idx=prompt_idx, window_start=window_start, hotkey=hotkey)


def test_rejected_grail_fail_omits_sketch_diff_max(monkeypatch):
    """GRAIL_FAIL must NOT expose sketch_diff_max — anti-tuning."""
    from reliquary.validator.verifier import ProofResult
    from reliquary.protocol.submission import RejectReason

    b = _make_batcher()  # existing helper

    # Stub verify_commitment to return a failing proof with a known diff.
    def fake_verify(commit, model, randomness):
        return ProofResult(
            all_passed=False,
            passed=2,
            checked=4,
            sketch_diff_max=4242,  # MUST NOT leak into archive
            logits=_empty_logits(),
        )
    b._verify_commitment = fake_verify
    b._verify_signature = lambda commit, hk: True

    req = _build_request(hotkey="hk_grail", prompt_idx=3)  # existing helper
    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.GRAIL_FAIL.value] == 1

    assert len(b.rejected_submissions) == 1
    rec = b.rejected_submissions[0]
    assert rec.hotkey == "hk_grail"
    assert rec.prompt_idx == 3
    assert rec.reason == "grail_fail"
    # Anti-tuning invariant: NO diagnostic field may surface on GRAIL_FAIL.
    # Identity fields (hotkey, prompt_idx, reason) are explicitly excluded;
    # everything else must be scrubbed to None.
    identity_fields = {"hotkey", "prompt_idx", "reason"}
    for field_name in rec.__dataclass_fields__:
        if field_name in identity_fields:
            continue
        assert getattr(rec, field_name) is None, (
            f"GRAIL_FAIL leaked tuning signal via field {field_name!r} "
            f"(value={getattr(rec, field_name)!r}); add scrubbing in _reject."
        )


def test_rejected_submissions_capped_per_hotkey(monkeypatch):
    """6th rejection from same hotkey must NOT grow the list (cap = 5)."""
    from reliquary.protocol.submission import RejectReason
    from reliquary.constants import REJECTED_LIST_CAP_PER_HOTKEY

    assert REJECTED_LIST_CAP_PER_HOTKEY == 5  # plan invariant

    b = _make_batcher()
    # Trigger BAD_PROMPT_IDX repeatedly — cheapest reject path that needs no
    # heavy stubbing (just send prompt_idx >= len(env)).
    spam_hotkey = "hk_spam"
    for i in range(REJECTED_LIST_CAP_PER_HOTKEY + 3):
        req = _build_request(
            hotkey=spam_hotkey,
            prompt_idx=10_000 + i,  # past env size to force BAD_PROMPT_IDX
        )
        resp = b.accept_submission(req)
        assert resp.reason == RejectReason.BAD_PROMPT_IDX

    # List capped, but counter keeps climbing.
    assert len(b.rejected_submissions) == REJECTED_LIST_CAP_PER_HOTKEY
    assert b.reject_counts["bad_prompt_idx"] == REJECTED_LIST_CAP_PER_HOTKEY + 3

    # Different hotkey gets its own quota.
    other_req = _build_request(hotkey="hk_other", prompt_idx=99_999)
    b.accept_submission(other_req)
    assert len(b.rejected_submissions) == REJECTED_LIST_CAP_PER_HOTKEY + 1
    assert b.rejected_submissions[-1].hotkey == "hk_other"


def test_valid_submission_has_rollout_hashes_field():
    """ValidSubmission exposes a per-rollout hash list (default empty)."""
    from reliquary.validator.batcher import ValidSubmission
    s = ValidSubmission(
        hotkey="hk", prompt_idx=42,
        merkle_root_bytes=b"\x00" * 32,
    )
    assert s.rollout_hashes == []


def test_hash_dup_rejects_replay_from_persistent_set():
    """A rollout whose tokens are already in the shared hash_set is rejected."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash

    hs = RolloutHashSet(retention_windows=50)
    # Seed the set with the hash of the rollout the test will resubmit.
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    h = compute_rollout_hash(req.rollouts[0].commit["tokens"])
    hs.add(h, window=499)

    b = _make_batcher(hash_set=hs)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.HASH_DUPLICATE


def test_hash_dup_intra_submission_collision_rejects():
    """Two rollouts in the same submission with identical tokens → reject."""
    from reliquary.validator.dedup import RolloutHashSet

    hs = RolloutHashSet(retention_windows=50)
    # Build a request whose 8 rollouts all share identical commit["tokens"].
    rollouts = []
    for i in range(M_ROLLOUTS):
        commit = _make_commit(success=(i < 4), total_reward=(1.0 if i < 4 else 0.0))
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"], reward=(1.0 if i < 4 else 0.0),
                commit=commit,
                env_name="openmathinstruct",
            )
        )
    req = BatchSubmissionRequest(
        miner_hotkey="hk", prompt_idx=42, window_start=500,
        merkle_root="00" * 32, rollouts=rollouts, checkpoint_hash="sha256:test",
        protocol_version=2,
    )

    b = _make_batcher(hash_set=hs)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.HASH_DUPLICATE


def test_hash_dup_none_set_disables_check():
    """Passing hash_set=None disables the check (back-compat for tests)."""
    b = _make_batcher(hash_set=None)
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True


def test_hash_dup_accept_when_not_in_set():
    """Fresh content with no prior hash entry passes."""
    from reliquary.validator.dedup import RolloutHashSet

    hs = RolloutHashSet(retention_windows=50)
    b = _make_batcher(hash_set=hs)
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True
    b.seal_batch()  # rollout_hashes are populated when the proof promotes to _valid
    stored = b.valid_submissions()[0]
    assert len(stored.rollout_hashes) == M_ROLLOUTS
    assert all(isinstance(h, bytes) and len(h) == 32 for h in stored.rollout_hashes)


def test_logical_group_duplicate_rejects_same_hotkey_wrapper_grind():
    """Changing wrapper metadata cannot mint a second in-window claim."""
    b = _make_batcher()
    first = _request(hotkey="hk-copy")
    second = _request(hotkey="hk-copy")
    second.merkle_root = "ff" * 32
    second.nonce = "new-wrapper"
    for rollout in second.rollouts:
        rollout.reward += 0.01

    assert b.accept_submission(first).accepted is True
    duplicate = b.accept_submission(second)

    assert duplicate.accepted is False
    assert duplicate.reason == RejectReason.HASH_DUPLICATE
    assert b.logical_group_reservation_count == 1
    assert b.logical_group_duplicate_rejects == 1
    assert len(b.pending_submissions()) == 1


def test_logical_group_identity_is_scoped_per_hotkey():
    b = _make_batcher()

    assert b.accept_submission(_request(hotkey="hk-a")).accepted is True
    assert b.accept_submission(_request(hotkey="hk-b")).accepted is True
    assert b.logical_group_reservation_count == 2


def test_auction_logical_claim_is_scoped_per_operator_and_prompt():
    b = _make_batcher(
        operator_by_hotkey={
            "hk-a": "operator-a",
            "hk-a-sybil": "operator-a",
        }
    )
    first = _request(prompt_idx=7, hotkey="hk-a")
    second = _request(prompt_idx=7, hotkey="hk-a-sybil")
    # A different token group must not mint a second ticket for the same
    # economic identity and prompt.
    for rollout in second.rollouts:
        changed = list(rollout.tokens)
        changed[-1] += 100
        rollout.tokens = changed
        rollout.commit["tokens"] = changed

    assert b.accept_submission(first).accepted is True
    duplicate = b.accept_submission(second)

    assert duplicate.accepted is False
    assert duplicate.reason == RejectReason.HASH_DUPLICATE
    assert b.logical_group_reservation_count == 1
    assert b.logical_group_duplicate_rejects == 1
    assert b.accept_submission(
        _request(prompt_idx=8, hotkey="hk-a-sybil")
    ).accepted is True


def test_auction_same_prompt_remains_open_to_distinct_operators():
    b = _make_batcher(
        operator_by_hotkey={
            "hk-a": "operator-a",
            "hk-b": "operator-b",
        }
    )

    assert b.accept_submission(
        _request(prompt_idx=7, hotkey="hk-a")
    ).accepted is True
    assert b.accept_submission(
        _request(prompt_idx=7, hotkey="hk-b")
    ).accepted is True
    assert b.logical_group_reservation_count == 2


def test_logical_group_reservation_can_cancel_before_proof_start():
    b = _make_batcher()
    request = _request(hotkey="hk-retry")

    assert b.try_reserve_logical_group(request) == (True, None)
    assert b.cancel_logical_group_reservation(request) is True
    assert b.logical_group_reservation_count == 0
    assert b.try_reserve_logical_group(request) == (True, None)


def test_logical_group_reservation_is_atomic_under_race():
    b = _make_batcher()
    requests = [
        _request(hotkey="hk-race").model_copy(deep=True) for _ in range(16)
    ]

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(b.try_reserve_logical_group, requests))

    assert sum(admitted for admitted, _ in results) == 1
    assert b.logical_group_reservation_count == 1
    assert b.logical_group_duplicate_rejects == 15


def test_seal_batch_populates_hash_set():
    """After seal_batch, every batched rollout's hash is in the shared set."""
    from reliquary.validator.dedup import RolloutHashSet

    hs = RolloutHashSet(retention_windows=50)
    b = _make_batcher(hash_set=hs)
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True

    batch, _ = b.seal_batch()
    assert len(batch) == 1
    for sub in batch:
        assert len(sub.rollout_hashes) == M_ROLLOUTS
        for h in sub.rollout_hashes:
            assert h in hs


def test_seal_batch_prunes_expired_hashes():
    """seal_batch calls prune so the set stays bounded across windows."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash

    hs = RolloutHashSet(retention_windows=50)
    # Seed a stale hash from a window way past retention.
    stale = compute_rollout_hash([1234, 5678])
    hs.add(stale, window=100)

    b = _make_batcher(hash_set=hs)
    # window_start defaults to 500 — stale (w=100) is 400 windows old, well
    # past retention=50.
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    b.accept_submission(req)
    b.seal_batch()
    assert stale not in hs


def test_seal_batch_with_none_hash_set_is_noop():
    """seal_batch must not crash when hash_set=None (test fixture path)."""
    b = _make_batcher(hash_set=None)
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    b.accept_submission(req)
    batch, _ = b.seal_batch()
    assert len(batch) == 1  # behaviour unchanged


def test_distinct_pending_prompt_count_ignores_duplicate_submissions():
    """Duplicate-prompt submissions inflate the raw pending count but not the
    distinct-prompt count that bounds trainable slots. (v2: submissions live in
    the pending pool until the seal-time proof; the distinct count is taken over
    pending, not valid.)"""
    b = _make_batcher()
    b.current_checkpoint_hash = ""

    for prompt_idx in range(B_BATCH - 1):
        req = _request_v21(
            prompt_idx=prompt_idx,
            hotkey=f"hk{prompt_idx}",
            checkpoint_hash="",
        )
        req.drand_round = 100
        b.accept_submission(req)
    duplicate = _request_v21(prompt_idx=0, hotkey="hk_dup", checkpoint_hash="")
    duplicate.drand_round = 100
    b.accept_submission(duplicate)

    assert len(b.pending_submissions()) == B_BATCH
    assert b.distinct_pending_prompt_count() == B_BATCH - 1


def test_admission_gated_by_grading_ceiling_not_grail_budget():
    """Admission counts grading attempts; there is no GRAIL candidate budget.

    The window-open burst queues for grading up to the grading ceiling instead
    of hard-bouncing on a candidate budget. (Entering the proof after the
    reward-zone gate only bumps a telemetry counter now; see
    ``test_grail_candidate_reserved_after_zone_gate`` and
    ``test_grail_candidate_budget_removed_no_reject_on_burst``.)
    """
    from reliquary.constants import (
        MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    )

    b = _make_batcher()

    # A burst larger than the GRAIL budget is still fully admitted for grading,
    # bounded only by the grading-attempts ceiling.
    for i in range(MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW):
        admitted, reason = b.try_reserve_proof_admission(
            _request(prompt_idx=i, hotkey=f"hk{i}")
        )
        assert admitted is True, (i, reason)
        assert reason is None
    assert b.proof_admission_count == 0          # admission never charges GRAIL
    assert b.proof_grading_attempts == 0
    assert b.pending_proof_reservations == MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    assert (
        b.proof_grading_capacity_used
        == MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    )

    # One past the grading ceiling is the anti-DoS bound that does reject.
    admitted, reason = b.try_reserve_proof_admission(
        _request(prompt_idx=999, hotkey="hkX")
    )
    assert admitted is False
    assert reason == "proof_grading_attempts_full"


def test_grail_candidate_reserved_after_zone_gate():
    """The GRAIL/GPU candidate budget is charged when a submission passes the
    reward-zone gate (heading to the proof), not at HTTP admission. A reserved-
    but-not-yet-graded submission holds a grading slot but no GRAIL slot."""
    b = _make_batcher()
    req = _request(hotkey="hkok")  # default rewards are in-zone (k=4, sigma=0.5)

    admitted, _ = b.try_reserve_proof_admission(req)
    assert admitted
    assert b.proof_admission_count == 0      # GRAIL not charged at admission
    assert b.proof_grading_attempts == 0
    assert b.pending_proof_reservations == 1
    assert b.start_proof_admission(req) == (True, None)

    try:
        resp = b.accept_submission(req)
    finally:
        b.finish_proof_admission(req)
    assert resp.accepted is True, resp.reason
    assert b.proof_admission_count == 1      # charged once the zone gate passed


def test_grail_candidate_budget_removed_no_reject_on_burst():
    """There is no per-window GRAIL candidate budget: a burst of zone-valid
    submissions well past the old cap (32) is accepted, bounded only by the
    grading ceiling. Entering the proof only bumps a telemetry counter, and no
    submission is rejected BATCH_FILLED on a candidate budget."""
    b = _make_batcher()

    n = 40  # > the old MAX_PROOF_CANDIDATES_PER_WINDOW (32)
    for i in range(n):
        resp = b.accept_submission(_request(prompt_idx=i, hotkey=f"hk{i}"))
        assert resp.accepted is True, (i, resp.reason)

    assert b.proof_admission_count == n
    assert b.reject_counts.get(RejectReason.BATCH_FILLED.value, 0) == 0


def test_proof_admission_rejects_hotkey_after_expensive_failure_debt():
    """Repeated post-proof failures from one hotkey should not consume all
    proof slots in the window.
    """
    from reliquary.constants import (
        MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW,
    )

    b = _make_batcher()
    req = _request(hotkey="hk-debt")
    for _ in range(MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW):
        b._reject(
            RejectReason.BAD_TERMINATION,
            hotkey="hk-debt",
            prompt_idx=42,
            reject_stage="termination",
        )

    admitted, reason = b.try_reserve_proof_admission(req)
    assert admitted is False
    assert reason == "proof_failure_debt_hotkey"
    assert b.proof_admission_count == 0
    assert (
        b.expensive_proof_failures_by_hotkey["hk-debt"]
        == MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW
    )


@pytest.mark.parametrize(
    "stage",
    [
        "force_span",
        "token_authenticity",
        "all_token_authenticity",
        "code_semantic_auth",
        "forced_seed",
    ],
)
def test_integrity_failures_all_accrue_proof_debt(stage):
    b = _make_batcher()

    b._reject(
        RejectReason.TOKEN_TAMPERED,
        hotkey="hk-integrity",
        prompt_idx=42,
        reject_stage=stage,
    )

    assert b.proof_failure_debt("hk-integrity") == 1


def test_proof_failure_debt_ignores_pre_proof_reject_stages():
    """Reward/zone/schema rejects can happen after HTTP reservation in some
    test paths, but they are not the post-proof failure pattern being
    throttled here.
    """
    from reliquary.constants import (
        MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW,
    )

    b = _make_batcher()
    req = _request(hotkey="hk-zone")
    for _ in range(MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW + 1):
        b._reject(
            RejectReason.OUT_OF_ZONE,
            hotkey="hk-zone",
            prompt_idx=42,
            reject_stage="zone",
        )

    admitted, reason = b.try_reserve_proof_admission(req)
    assert admitted is True
    assert reason is None
    assert b.proof_failure_debt("hk-zone") == 0


def test_proof_admission_post_trigger_cap_rejects_same_round_tail():
    """After the seal trigger round is known, only a bounded same-round
    tail can enter the expensive proof queue.
    """
    from reliquary.constants import MAX_POST_TRIGGER_PROOF_CANDIDATES

    b = _make_batcher()
    b._seal_trigger_round = 100
    requests = []
    for i in range(MAX_POST_TRIGGER_PROOF_CANDIDATES):
        req = _request(prompt_idx=i, hotkey=f"hk{i}")
        req.drand_round = 100
        requests.append(req)
        admitted, reason = b.try_reserve_proof_admission(req)
        assert admitted is True
        assert reason is None

    req = _request(prompt_idx=999, hotkey="hk-tail")
    req.drand_round = 100
    admitted, reason = b.try_reserve_proof_admission(req)
    assert admitted is False
    assert reason == "proof_admission_post_trigger_full"
    assert b.post_trigger_proof_admission_count == (
        MAX_POST_TRIGGER_PROOF_CANDIDATES
    )


def _prove_all_pending(b):
    """Prove every pending candidate into ``_valid``, bypassing the B_BATCH cap
    that ``_prove_ranked`` normally enforces, so the seal's boundary fair-split
    (which needs > B distinct candidate prompts) is reachable in the test."""
    def _prove(pool=1.0):
        proven = [
            s for s in (
                b._verify_expensive(p) for p in b.pending_submissions()
            ) if s is not None
        ]
        b._valid = proven
        b.valid_count = len(proven)
        return proven
    b._prove_ranked = _prove
    b._prove_forensic_sample = lambda: []


def test_seal_batch_cooldowns_every_rewarded_boundary_prompt():
    """Boundary runners earn emission, so their prompts must enter cooldown."""
    b = _make_batcher()
    for i in range(B_BATCH + 4):
        req = _request(prompt_idx=i, hotkey=f"hk{i}")
        req.drand_round = 100
        assert b.accept_submission(req).accepted is True

    _prove_all_pending(b)
    batch, rewards = b.seal_batch()

    selected_prompts = {s.prompt_idx for s in batch}
    runner_prompts = set(range(B_BATCH + 4)) - selected_prompts
    assert len(batch) == B_BATCH
    assert set(rewards) == {f"hk{i}" for i in range(B_BATCH + 4)}
    assert runner_prompts
    for prompt_idx in range(B_BATCH + 4):
        assert b._cooldown.is_in_cooldown(prompt_idx, b.window_start + 1)
    assert b.rewarded_but_not_selected_by_hotkey == {
        f"hk{prompt_idx}": 1 for prompt_idx in runner_prompts
    }


def test_seal_batch_hash_dedup_records_rewarded_boundary_runners():
    """Rollout hashes for paid runners must be blocked in later windows."""
    from reliquary.validator.dedup import RolloutHashSet

    hs = RolloutHashSet(retention_windows=50)
    b = _make_batcher(hash_set=hs)
    for i in range(B_BATCH + 4):
        req = _request_with_prompt_unique_tokens(prompt_idx=i, hotkey=f"hk{i}")
        req.drand_round = 100
        assert b.accept_submission(req).accepted is True

    _prove_all_pending(b)
    batch, _ = b.seal_batch()

    selected_prompts = {s.prompt_idx for s in batch}
    runner_submissions = [
        s for s in b.valid_submissions() if s.prompt_idx not in selected_prompts
    ]
    assert runner_submissions
    for sub in runner_submissions:
        assert len(sub.rollout_hashes) == M_ROLLOUTS
        for h in sub.rollout_hashes:
            assert h in hs


def test_batcher_beacon_invalid_defaults_false():
    """The background drand-verify task sets ``beacon_invalid`` to True
    if the cross-check against bittensor_drand fails post-OPEN.
    Default at construction must be False so an un-flagged batcher
    proceeds to train+publish normally.
    """
    b = _make_batcher()
    assert b.beacon_invalid is False


# ----- BoxedAnswerValidator wiring -----


class _CharTokenizerWithEos:
    """One-char-per-token tokenizer with eos_token_id=99 for batcher wiring."""

    eos_token_id = 99

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids if int(i) != 99)


def _grail_with_chosen_probs(
    seq_len: int,
    completion_probs: list[float],
    prompt_length: int = 4,
    argmax_ids: list[int] | None = None,
):
    """Stub returning sparse outputs incl. completion_chosen_probs.

    Also pre-populates challenge_lp_indices/values with zeros at the first K
    completion positions, matching the miner-claimed all-zero logprobs from
    ``_make_commit`` so verify_logprobs_claim does not fire before us.
    """
    challenge_idxs = list(range(prompt_length, prompt_length + CHALLENGE_K))
    challenge_vals = [0.0] * CHALLENGE_K

    def _fn(commit, model, randomness):
        return ProofResult(
            all_passed=True, passed=1, checked=1, sketch_diff_max=0,
            has_sparse_outputs=True,
            p_stop=0.99,
            challenge_lp_indices=challenge_idxs,
            challenge_lp_values=challenge_vals,
            completion_chosen_probs=list(completion_probs),
            completion_argmax_probs=[1.0] * len(completion_probs),
            completion_argmax_ids=argmax_ids or [0] * len(completion_probs),
        )
    return _fn


def _boxed_completion_padded(answer_text: str = "the final answer is \\boxed{42}") -> tuple[str, list[int]]:
    # completion_length must be >= CHALLENGE_K to clear verify_logprobs_claim.
    pad_chars = max(0, (CHALLENGE_K + 1) - len(answer_text))
    text = "x" * pad_chars + answer_text
    completion = [ord(c) for c in text] + [99]
    return text, completion


def test_reject_boxed_answer_tampered():
    """One token inside the last \\boxed{...} at tampered prob → reject."""
    prompt = [10, 11, 12, 13]
    text, completion = _boxed_completion_padded()
    tokens = prompt + completion
    seq_len = len(tokens)

    probs = [0.99] * len(completion)
    probs[text.index("{") + 1] = 1e-6

    class _LongCtxModel:
        class config:
            vocab_size = 1000
            max_position_embeddings = 4096

    b = _make_batcher(
        model=_LongCtxModel(),
        tokenizer=_CharTokenizerWithEos(),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(seq_len, probs),
    )
    req = _request()
    commit = _make_commit(
        tokens=tokens,
        prompt_length=len(prompt),
        success=True,
        total_reward=1.0,
    )
    req.rollouts[0].commit = commit
    req.rollouts[0].tokens = commit["tokens"]

    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.BOXED_ANSWER_TAMPERED.value] == 1


def test_reject_forced_rollout_with_noncanonical_force_span(monkeypatch):
    # A forced rollout whose declared force_span tokens differ from the canonical
    # FORCE ids (atomic </think>=200 + tail [201,202]) → reject. The thinking
    # budget is monkeypatched to the test's thinking length so the force-position
    # check passes and the byte-exact content check is what fires.
    monkeypatch.setattr("reliquary.constants.BFT_THINKING_BUDGET", CHALLENGE_K)

    class _ForceTokenizer:
        eos_token_id = 99

        def decode(self, ids, *, skip_special_tokens=False):
            return "".join(chr(int(i)) for i in ids if int(i) != 99)

        def convert_tokens_to_ids(self, t):
            return 200 if t == "</think>" else None

        def encode(self, text, add_special_tokens=False):
            return [201, 202]

    class _LongCtxModel:
        class config:
            vocab_size = 1000
            max_position_embeddings = 4096

    prompt = [10, 11, 12, 13]
    thinking = [5] * CHALLENGE_K
    force_noncanonical = [200, 999, 202]  # 999 differs from the canonical 201
    answer = [55, 99]
    tokens = prompt + thinking + force_noncanonical + answer
    seq_len = len(tokens)
    fstart = len(prompt) + len(thinking)

    b = _make_batcher(
        model=_LongCtxModel(),
        tokenizer=_ForceTokenizer(),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(
            seq_len, [0.99] * (seq_len - len(prompt)), prompt_length=len(prompt),
        ),
    )
    b.env.name = "openmathinstruct"  # BFT forcing is honoured only in math
    req = _request()
    commit = _make_commit(
        tokens=tokens, prompt_length=len(prompt), success=True, total_reward=1.0,
    )
    commit["rollout"]["forced"] = True
    commit["rollout"]["force_span"] = [fstart, fstart + 3]
    req.rollouts[0].commit = commit
    req.rollouts[0].tokens = commit["tokens"]

    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.TOKEN_TAMPERED.value] == 1


def test_accept_boxed_answer_high_prob():
    """All tokens inside \\boxed{...} above threshold → no boxed reject."""
    prompt = [10, 11, 12, 13]
    _, completion = _boxed_completion_padded()
    tokens = prompt + completion
    seq_len = len(tokens)
    probs = [0.95] * len(completion)

    class _LongCtxModel:
        class config:
            vocab_size = 1000
            max_position_embeddings = 4096

    b = _make_batcher(
        model=_LongCtxModel(),
        tokenizer=_CharTokenizerWithEos(),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(seq_len, probs),
    )
    req = _request()
    for idx in range(M_ROLLOUTS):
        commit = _make_commit(
            tokens=tokens,
            prompt_length=len(prompt),
            success=req.rollouts[idx].reward > 0.5,
            total_reward=req.rollouts[idx].reward,
        )
        req.rollouts[idx].commit = commit
        req.rollouts[idx].tokens = commit["tokens"]

    # High in-box probs → no boxed tamper; the proof passes at seal.
    assert _prove_one(b, req) is not None


def test_all_token_auth_shadow_records_without_rejecting(monkeypatch, tmp_path):
    import reliquary.validator.batcher as batcher_mod

    # Telemetry-recording path: with enforcement off, findings are counted but
    # the submission is still accepted.
    monkeypatch.setattr(batcher_mod, "ALL_TOKEN_AUTH_ENFORCE", False)
    forensics_path = tmp_path / "auth-forensics.jsonl"
    monkeypatch.setenv("RELIQUARY_AUTH_FORENSICS_ENABLED", "1")
    monkeypatch.setenv("RELIQUARY_AUTH_FORENSICS_PATH", str(forensics_path))

    monkeypatch.setattr(
        batcher_mod,
        "evaluate_all_token_auth_shadow",
        lambda _proof, **_kwargs: (
            False,
            {
                "n_tokens": CHALLENGE_K,
                "findings": 2,
                "min_prob": 4.0e-7,
                "threshold": 1.0e-5,
                "argmax_conf": 0.99,
                "finding_min_prob": 7.0e-6,
                "finding_details": [
                    {
                        "completion_pos": 1,
                        "absolute_token_pos": 5,
                        "p_chosen": 4.0e-7,
                        "p_argmax": 0.995,
                        "token_id": 65,
                        "token_text": "A",
                        "argmax_id": 66,
                        "argmax_text": "B",
                        "completion_context": "xAy",
                    },
                    {
                        "completion_pos": 2,
                        "absolute_token_pos": 6,
                        "p_chosen": 7.0e-6,
                        "p_argmax": 0.996,
                        "token_id": 67,
                        "token_text": "C",
                        "argmax_id": 68,
                        "argmax_text": "D",
                        "completion_context": "xCy",
                    },
                ],
            },
        ),
    )
    monkeypatch.setattr(batcher_mod, "has_eos_padding", lambda *_args, **_kw: False)
    monkeypatch.setattr(batcher_mod, "verify_termination", lambda *_args, **_kw: True)
    monkeypatch.setattr(batcher_mod, "is_cap_truncation", lambda *_args, **_kw: False)

    b = _make_batcher(
        env=FakeEnv(),
        tokenizer=_CharTokenizerWithEos(),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(
            CHALLENGE_K + 4,
            [0.99] * CHALLENGE_K,
        ),
    )
    req = _request()

    # The auth shadow is recorded during the seal-time proof; recording does
    # not reject when enforcement is off.
    sub = _prove_one(b, req)
    assert sub is not None
    assert sub.all_token_auth_shadow_findings == M_ROLLOUTS * 2
    assert sub.all_token_auth_shadow_min_prob == pytest.approx(4.0e-7)
    assert sub.all_token_auth_shadow_positive_findings == 4 * 2
    assert sub.all_token_auth_shadow_positive_min_prob == pytest.approx(7.0e-6)

    records = [json.loads(line) for line in forensics_path.read_text().splitlines()]
    assert len(records) == M_ROLLOUTS * 2
    first = records[0]
    assert first["event"] == "all_token_auth_shadow_finding"
    assert first["window_start"] == 500
    assert first["miner_hotkey"] == "hk"
    assert first["prompt_idx"] == 42
    assert first["rollout_idx"] == 0
    assert first["reward_positive"] is True
    assert first["completion_pos"] == 1
    assert first["token_text"] == "A"


def test_all_token_auth_enforce_rejects(monkeypatch):
    """With ALL_TOKEN_AUTH_ENFORCE on, an all-token argmax-gated finding rejects
    the submission (unconditional on reward), like the primary token-auth gate."""
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "ALL_TOKEN_AUTH_ENFORCE", True)
    monkeypatch.setattr(
        batcher_mod,
        "evaluate_all_token_auth_shadow",
        lambda _proof, **_kwargs: (False, {"findings": 1, "min_prob": 4.0e-7}),
    )
    monkeypatch.setattr(batcher_mod, "has_eos_padding", lambda *_args, **_kw: False)
    monkeypatch.setattr(batcher_mod, "verify_termination", lambda *_args, **_kw: True)
    monkeypatch.setattr(batcher_mod, "is_cap_truncation", lambda *_args, **_kw: False)

    b = _make_batcher(
        env=FakeEnv(),
        tokenizer=_CharTokenizerWithEos(),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(
            CHALLENGE_K + 4,
            [0.99] * CHALLENGE_K,
        ),
    )
    req = _request()

    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.TOKEN_TAMPERED.value] == 1


def test_opencode_semantic_auth_shadow_records_without_rejecting(
    monkeypatch,
    tmp_path,
):
    class _CodeTokenizer:
        eos_token_id = 9999

        def decode(self, ids, *, skip_special_tokens=False):
            return "".join(chr(int(i)) for i in ids if int(i) != self.eos_token_id)

    forensics_path = tmp_path / "code-semantic-auth.jsonl"
    monkeypatch.setenv("RELIQUARY_AUTH_FORENSICS_ENABLED", "1")
    monkeypatch.setenv(
        "RELIQUARY_CODE_SEMANTIC_AUTH_FORENSICS_PATH",
        str(forensics_path),
    )

    prompt = [10, 11, 12, 13]
    body = (
        "```python\n"
        "def second_largest(nums):\n"
        "    return sorted(set(nums), reverse=False)[-2]\n"
        "```"
    )
    completion = [ord(c) for c in body] + [_CodeTokenizer.eos_token_id]
    tokens = prompt + completion
    probs = [0.99] * len(completion)
    probs[body.index("False")] = 2.0e-4

    class _LongCtxModel:
        class config:
            vocab_size = 20000
            max_position_embeddings = 4096

    b = _make_batcher(
        env=PrivateRewardFakeEnv(),
        model=_LongCtxModel(),
        tokenizer=_CodeTokenizer(),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(
            len(tokens), probs, prompt_length=len(prompt),
        ),
    )
    req = _request()
    for idx, rollout in enumerate(req.rollouts):
        commit = _make_commit(
            tokens=tokens,
            prompt_length=len(prompt),
            success=idx < 4,
            total_reward=1.0 if idx < 4 else 0.0,
        )
        rollout.commit = commit
        rollout.tokens = commit["tokens"]
        rollout.env_name = "opencodeinstruct"

    # The code-semantic auth shadow is recorded during the seal-time proof.
    sub = _prove_one(b, req)
    assert sub is not None
    assert sub.code_semantic_auth_findings == M_ROLLOUTS
    assert sub.code_semantic_auth_min_prob == pytest.approx(2.0e-4)
    assert sub.code_semantic_auth_positive_findings == 4
    assert sub.code_semantic_auth_positive_min_prob == pytest.approx(2.0e-4)

    records = [json.loads(line) for line in forensics_path.read_text().splitlines()]
    assert len(records) == M_ROLLOUTS
    first = records[0]
    assert first["event"] == "code_semantic_auth_finding"
    assert first["surface"] == "code-semantic"
    assert first["window_start"] == 500
    assert first["miner_hotkey"] == "hk"
    assert first["prompt_idx"] == 42
    assert first["rollout_idx"] == 0
    assert first["reward_positive"] is True
    assert first["label"] == "keyword:reverse"
    assert first["token_text"] == "F"
    assert "reverse=False" in first["code_context"]


def test_opencode_semantic_auth_records_counterfactual_reward_flip(
    monkeypatch,
    tmp_path,
):
    class _CodeTokenizer:
        eos_token_id = 9999

        def decode(self, ids, *, skip_special_tokens=False):
            return "".join(chr(int(i)) for i in ids if int(i) != self.eos_token_id)

    class _CounterfactualEnv(PrivateRewardFakeEnv):
        def compute_reward(self, problem, completion):
            return 1.0 if "reverse=False" in completion else 0.0

    forensics_path = tmp_path / "code-semantic-auth.jsonl"
    monkeypatch.setenv("RELIQUARY_AUTH_FORENSICS_ENABLED", "1")
    monkeypatch.setenv("RELIQUARY_CODE_SEMANTIC_COUNTERFACTUAL_ENABLED", "1")
    monkeypatch.setenv(
        "RELIQUARY_CODE_SEMANTIC_AUTH_FORENSICS_PATH",
        str(forensics_path),
    )

    prompt = [10, 11, 12, 13]
    body = (
        "```python\n"
        "def second_largest(nums):\n"
        "    return sorted(set(nums), reverse=False)[-2]\n"
        "```"
    )
    completion = [ord(c) for c in body] + [_CodeTokenizer.eos_token_id]
    tokens = prompt + completion
    false_pos = body.index("False")
    probs = [0.99] * len(completion)
    probs[false_pos] = 2.0e-4
    argmax_ids = [0] * len(completion)
    argmax_ids[false_pos] = ord("T")

    class _LongCtxModel:
        class config:
            vocab_size = 20000
            max_position_embeddings = 4096

    b = _make_batcher(
        env=_CounterfactualEnv(),
        model=_LongCtxModel(),
        tokenizer=_CodeTokenizer(),
        completion_text_fn=lambda rollout: (
            body if rollout.reward > 0.5 else "wrong"
        ),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(
            len(tokens),
            probs,
            prompt_length=len(prompt),
            argmax_ids=argmax_ids,
        ),
    )
    req = _request()
    for rollout in req.rollouts:
        commit = _make_commit(
            tokens=tokens,
            prompt_length=len(prompt),
            success=False,
            total_reward=0.0,
        )
        rollout.commit = commit
        rollout.tokens = commit["tokens"]
        rollout.env_name = "opencodeinstruct"

    # The counterfactual forensics are written during the seal-time proof.
    assert _prove_one(b, req) is not None
    records = [json.loads(line) for line in forensics_path.read_text().splitlines()]
    assert len(records) == M_ROLLOUTS
    first = records[0]
    assert first["reward_positive"] is True
    assert first["signal_bucket"] == "review"
    assert first["counterfactual_checked"] is True
    assert first["counterfactual_reward"] == 0.0
    assert first["counterfactual_reward_delta"] == -1.0
    assert first["counterfactual_reward_flipped"] is True


def test_opencode_semantic_auth_enforce_ignores_zero_reward_rollout(monkeypatch):
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "CODE_SEMANTIC_AUTH_ENFORCE", True)

    def fake_code_auth(*, tokens, **_kwargs):
        # Only the zero-reward rollout carries this sentinel token.
        if 4242 in tokens:
            return False, {"findings": 1, "min_prob": 2.0e-4}
        return True, {"findings": 0, "min_prob": 0.5}

    monkeypatch.setattr(
        batcher_mod,
        "evaluate_code_semantic_token_authenticity",
        fake_code_auth,
    )
    monkeypatch.setattr(batcher_mod, "has_eos_padding", lambda *_args, **_kw: False)
    monkeypatch.setattr(batcher_mod, "verify_termination", lambda *_args, **_kw: True)
    monkeypatch.setattr(batcher_mod, "is_cap_truncation", lambda *_args, **_kw: False)

    b = _make_batcher(
        env=PrivateRewardFakeEnv(),
        tokenizer=_CharTokenizerWithEos(),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(
            CHALLENGE_K + 4,
            [0.99] * CHALLENGE_K,
        ),
    )
    req = _request()
    for idx, rollout in enumerate(req.rollouts):
        rollout.env_name = "opencodeinstruct"
        if idx == len(req.rollouts) - 1:
            rollout.tokens[0] = 4242
            rollout.commit["tokens"][0] = 4242

    # Enforcement only rejects on a POSITIVE-reward rollout; the finding here is
    # on the zero-reward rollout, so the submission survives the seal-time proof.
    sub = _prove_one(b, req)
    assert sub is not None
    assert sub.code_semantic_auth_findings == 1
    assert sub.code_semantic_auth_min_prob == pytest.approx(2.0e-4)
    assert sub.code_semantic_auth_positive_findings == 0
    assert sub.code_semantic_auth_positive_min_prob is None


def test_opencode_semantic_auth_enforce_rejects_positive_reward_rollout(
    monkeypatch,
):
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "CODE_SEMANTIC_AUTH_ENFORCE", True)

    def fake_code_auth(*, tokens, **_kwargs):
        # First rollout is reward-positive in _request().
        if 4242 in tokens:
            return False, {"findings": 1, "min_prob": 2.0e-4}
        return True, {"findings": 0, "min_prob": 0.5}

    monkeypatch.setattr(
        batcher_mod,
        "evaluate_code_semantic_token_authenticity",
        fake_code_auth,
    )
    monkeypatch.setattr(batcher_mod, "has_eos_padding", lambda *_args, **_kw: False)
    monkeypatch.setattr(batcher_mod, "verify_termination", lambda *_args, **_kw: True)
    monkeypatch.setattr(batcher_mod, "is_cap_truncation", lambda *_args, **_kw: False)

    b = _make_batcher(
        env=PrivateRewardFakeEnv(),
        tokenizer=_CharTokenizerWithEos(),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(
            CHALLENGE_K + 4,
            [0.99] * CHALLENGE_K,
        ),
    )
    req = _request()
    for rollout in req.rollouts:
        rollout.env_name = "opencodeinstruct"
    req.rollouts[0].tokens[0] = 4242
    req.rollouts[0].commit["tokens"][0] = 4242

    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.TOKEN_TAMPERED.value] == 1


def test_cap_truncated_rollout_still_runs_behavioural_checks():
    """Regression: cap-truncated rollouts are budget-tolerated for termination
    but MUST still pass logprob/distribution/boxed integrity checks.

    Without this, a miner force-caps a rollout to bypass behavioural checks
    and tampers the \\boxed{...} content; the truncated-budget path used to
    ``continue`` past every per-rollout check.
    """
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt = [10, 11, 12, 13]
    answer = "the final answer is \\boxed{42}"
    # Body filled with non-EOS, non-`{` token + appended boxed answer.
    # Total completion = MAX_NEW_TOKENS_PROTOCOL_CAP, last token != EOS →
    # cap-truncated path.
    body = [5] * (MAX_NEW_TOKENS_PROTOCOL_CAP - len(answer))
    answer_tokens = [ord(c) for c in answer]
    completion = body + answer_tokens
    tokens = prompt + completion

    probs = [0.99] * len(completion)
    # Tamper: drop chosen prob at the digit inside \boxed{}
    probs[len(body) + answer.index("{") + 1] = 1e-6

    class _LongCtxModel:
        class config:
            vocab_size = 1000
            max_position_embeddings = MAX_NEW_TOKENS_PROTOCOL_CAP + 100

    b = _make_batcher(
        model=_LongCtxModel(),
        tokenizer=_CharTokenizerWithEos(),
        verify_commitment_proofs_fn=_grail_with_chosen_probs(
            len(tokens), probs, prompt_length=len(prompt),
        ),
    )
    req = _request()
    commit = _make_commit(
        tokens=tokens,
        prompt_length=len(prompt),
        success=True,
        total_reward=1.0,
    )
    req.rollouts[0].commit = commit
    req.rollouts[0].tokens = commit["tokens"]

    # Cap-truncation is termination-tolerated but the boxed-content check still
    # runs at seal — the tampered digit must reject.
    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.BOXED_ANSWER_TAMPERED.value] == 1


def test_reject_malformed_final_answer_before_grail():
    # reward=0 rollouts box "a" then dangle a special-token box -> malformed final.
    # FakeEnv.compute_reward returns 0.0 (no "CORRECT"), so the reward claim matches.
    def text_fn(rollout):
        if rollout.reward > 0.5:
            return "CORRECT \\boxed{a}"
        return "work \\boxed{a} then $$\\boxed{<|im_end|>"
    b = _make_batcher(completion_text_fn=text_fn)
    req = _request_with_prompt_unique_tokens(rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.MALFORMED_FINAL_ANSWER


def test_accept_honest_failures_not_flagged_as_malformed():
    # reward=0 rollouts are genuine give-ups (no boxed) -> not flagged.
    b = _make_batcher()  # default text_fn: "CORRECT" / "wrong"
    req = _request_with_prompt_unique_tokens(rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.reason != RejectReason.MALFORMED_FINAL_ANSWER


def test_accept_genuine_wrong_wellformed_answer():
    # reward=0 rollouts cleanly box a wrong value -> legitimate negative, accepted.
    def text_fn(rollout):
        return "CORRECT \\boxed{a}" if rollout.reward > 0.5 else "nope \\boxed{9}"
    b = _make_batcher(completion_text_fn=text_fn)
    req = _request_with_prompt_unique_tokens(rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True


from reliquary.validator import batcher as batcher_mod


def test_set_prompt_range_none_before_cutover(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 10_000)
    b = _make_batcher(window_start=500)
    b.randomness = "deadbeef"
    b.set_prompt_range()
    assert b.prompt_range is None  # window 500 < cutover 10000 -> not armed


def test_set_prompt_range_none_without_randomness(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 0)
    b = _make_batcher(window_start=500)
    b.randomness = ""
    b.set_prompt_range()
    assert b.prompt_range is None  # no randomness yet -> no restriction


def test_set_prompt_range_armed(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 0)
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_SIZE", 100)
    b = _make_batcher(window_start=500)
    b.randomness = "deadbeef"
    b.set_prompt_range()
    lo, hi = b.prompt_range
    assert hi - lo == 100
    assert 0 <= lo and hi <= 1000  # FakeEnv len is 1000


def test_set_prompt_range_matches_shared_function(monkeypatch):
    # Cross-side agreement: the validator must seed the slice with the SAME
    # inputs (env.name, len(env), size) the miner feeds window_prompt_range,
    # or an honest miner gets wrongly rejected. Pin it against future drift.
    from reliquary.shared.prompt_range import window_prompt_range
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 0)
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_SIZE", 100)
    b = _make_batcher(window_start=500)
    b.randomness = "deadbeef"
    b.set_prompt_range()
    assert b.prompt_range == window_prompt_range(
        b.randomness, b.env.name, len(b.env), 100,
    )


def test_accept_rejects_out_of_range(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 0)
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_SIZE", 100)
    b = _make_batcher(window_start=500)
    b.randomness = "deadbeef"
    b.set_prompt_range()
    lo, hi = b.prompt_range
    out = (hi + 1) % 1000
    if lo <= out < hi:
        out = (lo - 1) % 1000
    resp = b.accept_submission(_request(prompt_idx=out, window_start=500))
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_OUT_OF_RANGE


def test_accept_in_range_passes_range_gate(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 0)
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_SIZE", 100)
    b = _make_batcher(window_start=500)
    b.randomness = "deadbeef"
    b.set_prompt_range()
    lo, hi = b.prompt_range
    resp = b.accept_submission(_request(prompt_idx=lo, window_start=500))
    # Passes the range gate; may still hit a later gate, but never this one.
    assert resp.reason != RejectReason.PROMPT_OUT_OF_RANGE


# ---- forced-seed group gate (Task 5) -------------------------------------


def _grail_with_seed_counts(n_stoch: int, n_match: int):
    """Stub verifier that opts into the (tokenizer, seed_u_values) signature
    and reports a fixed per-rollout seed-consistency tally. The batcher sums
    these across all 8 rollouts before deciding the group verdict."""
    def _fn(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        from reliquary.validator.verifier import ProofResult
        return ProofResult(
            all_passed=True, passed=1, checked=1,
            seed_n_stochastic=n_stoch, seed_n_match=n_match,
        )
    return _fn


def test_forced_seed_group_gate_rejects_below_floor_when_enforcing(monkeypatch):
    """Aggregate over 8 rollouts: 80 stochastic positions, 8 matches (0.10)
    is well below FORCED_SEED_CONSISTENCY_FLOOR (0.80). With FORCED_SEED_ENFORCE
    on, the group is rejected SEED_MISMATCH after the per-rollout loop."""
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "FORCED_SEED_ENFORCE", True)
    b = _make_batcher(
        window_start=500,
        verify_commitment_proofs_fn=_grail_with_seed_counts(n_stoch=10, n_match=1),
    )
    b.current_checkpoint_hash = "sha256:test"   # pinned -> seed enforcement active
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.SEED_MISMATCH.value] == 1
    assert len(b.valid_submissions()) == 0


def test_forced_seed_gate_abstains_when_checkpoint_hash_unpinned(monkeypatch):
    """When current_checkpoint_hash is empty the WRONG_CHECKPOINT gate is off,
    so the miner controls checkpoint_hash -- a forced-seed derivation input he
    could grind. Enforcement is coupled to a pinned hash: even with
    FORCED_SEED_ENFORCE on and a failing match-rate, an unpinned-hash window
    abstains instead of rejecting."""
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "FORCED_SEED_ENFORCE", True)
    b = _make_batcher(
        window_start=500,
        verify_commitment_proofs_fn=_grail_with_seed_counts(n_stoch=10, n_match=1),
    )
    b.current_checkpoint_hash = ""              # not yet published -> not pinned
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    # Unpinned hash -> the gate abstains: the proof passes at seal, not rejected.
    assert _prove_one(b, req) is not None


def test_forced_seed_group_gate_shadow_when_not_enforcing(monkeypatch):
    """Same low match-rate as above, but FORCED_SEED_ENFORCE is off -> shadow
    only, submission is NOT rejected for SEED_MISMATCH."""
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "FORCED_SEED_ENFORCE", False)
    b = _make_batcher(
        window_start=500,
        verify_commitment_proofs_fn=_grail_with_seed_counts(n_stoch=10, n_match=1),
    )
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    # Not enforcing -> shadow only: the proof passes at seal, not rejected.
    assert _prove_one(b, req) is not None


def _grail_with_cdf_hard_mismatch():
    def _fn(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        from reliquary.validator.verifier import ProofResult

        return ProofResult(
            all_passed=True,
            passed=1,
            checked=1,
            seed_n_stochastic=10,
            seed_n_match=10,
            seed_n_positions=10,
            seed_n_boundary_match=9,
            seed_n_hard_mismatch=1,
            seed_max_cdf_miss=0.2,
        )

    return _fn


def test_forced_seed_cdf_gate_rejects_sparse_branch_mismatch(monkeypatch):
    import reliquary.validator.batcher as batcher_mod

    recorded = []
    monkeypatch.setattr(batcher_mod, "FORCED_SEED_ENFORCE", False)
    monkeypatch.setattr(batcher_mod, "FORCED_SEED_CDF_ENFORCE", True)
    monkeypatch.setattr(
        batcher_mod,
        "record_forced_seed_shadow",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )
    b = _make_batcher(
        window_start=500,
        verify_commitment_proofs_fn=_grail_with_cdf_hard_mismatch(),
    )
    b.current_checkpoint_hash = "sha256:test"

    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    assert _prove_one(b, req) is None
    assert b.reject_counts[RejectReason.SEED_MISMATCH.value] == 1
    assert len(recorded) == 1
    assert recorded[0][1]["cdf_would_reject"] is True
    assert recorded[0][1]["cdf_enforced"] is True
    assert recorded[0][1]["window_start"] == 500
    assert recorded[0][1]["checkpoint_hash"] == "sha256:test"
    assert recorded[0][1]["env_name"] == "openmathinstruct"


def test_forced_seed_cdf_gate_is_shadow_until_calibrated(monkeypatch):
    import reliquary.validator.batcher as batcher_mod

    monkeypatch.setattr(batcher_mod, "FORCED_SEED_ENFORCE", False)
    monkeypatch.setattr(batcher_mod, "FORCED_SEED_CDF_ENFORCE", False)
    b = _make_batcher(
        window_start=500,
        verify_commitment_proofs_fn=_grail_with_cdf_hard_mismatch(),
    )
    b.current_checkpoint_hash = "sha256:test"

    # CDF enforcement off -> shadow only: the proof passes at seal.
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    assert _prove_one(b, req) is not None
