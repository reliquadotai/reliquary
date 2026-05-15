"""GrpoWindowBatcher: accepts submissions, enforces verification pipeline,
exposes select_batch at window close."""

from dataclasses import dataclass
from typing import Any

import pytest

from reliquary.constants import B_BATCH, CHALLENGE_K, M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RejectReason,
    RolloutSubmission,
)
from reliquary.validator.batcher import GrpoWindowBatcher


class FakeEnv:
    name = "fake"
    def __len__(self):
        return 1000
    def get_problem(self, idx):
        return {"prompt": f"p{idx}", "ground_truth": "a", "id": f"pid-{idx}"}
    def compute_reward(self, problem, completion):
        return 1.0 if "CORRECT" in completion else 0.0


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
        "proof_version": "v5",
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
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey,
        prompt_idx=prompt_idx,
        window_start=window_start,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
    )


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
    assert len(b.valid_submissions()) == 1


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


def test_reject_grail_fail():
    b = _make_batcher(verify_commitment_proofs_fn=_always_false_grail)
    req = _request()
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.GRAIL_FAIL


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
            )
        )
    req = BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=42,
        window_start=500,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
    )
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.REWARD_MISMATCH


# --- seal_batch + cooldown lifecycle ---

def test_seal_batch_empty_pool_returns_empty():
    b = _make_batcher()
    assert b.seal_batch() == []


def test_seal_batch_fifo_across_many_submissions():
    """v2.2: ordering is by ``arrived_at`` (validator-side TCP arrival)."""
    b = _make_batcher()
    for i in range(B_BATCH):
        req = _request(
            prompt_idx=i, hotkey=f"hk{i}",
        )
        resp = b.accept_submission(req)
        assert resp.accepted, f"unexpected reject for {i}: {resp.reason}"
    batch = b.seal_batch()
    assert len(batch) == B_BATCH
    arrivals = [s.arrived_at for s in batch]
    assert arrivals == sorted(arrivals), "batch must be ordered by arrived_at"


def test_seal_batch_cooldown_recorded():
    b = _make_batcher()
    req = _request(prompt_idx=42)
    b.accept_submission(req)
    batch = b.seal_batch()
    assert len(batch) == 1
    assert b._cooldown.is_in_cooldown(42, b.window_start + 1) is True


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
    b = _make_batcher()
    b.accept_submission(_request(prompt_idx=42, hotkey="alice"))
    b.accept_submission(_request(prompt_idx=42, hotkey="bob"))
    b.accept_submission(_request(prompt_idx=7, hotkey="carol"))
    batch = b.seal_batch()
    assert len(batch) == 2
    hotkeys = {s.hotkey for s in batch}
    assert hotkeys == {"alice", "carol"}


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
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey, prompt_idx=prompt_idx,
        window_start=window_start,
        merkle_root="00" * 32, rollouts=rollouts,
        checkpoint_hash=checkpoint_hash,
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


def test_empty_checkpoint_hash_disables_gate():
    """When batcher.current_checkpoint_hash is "", any hash is accepted
    (test convenience — simulates pre-first-publish)."""
    b = _make_batcher()
    b.current_checkpoint_hash = ""
    req = _request_v21(checkpoint_hash="anything")
    resp = b.accept_submission(req)
    assert resp.accepted is True


@pytest.mark.asyncio
async def test_seal_event_set_when_b_valid_distinct_landed():
    """seal_event fires when the B-th valid distinct-prompt non-cooldown
    submission is accepted."""
    b = _make_batcher()
    b.current_checkpoint_hash = "sha256:hash"
    assert not b.seal_event.is_set()
    for i in range(B_BATCH):
        req = _request_v21(
            prompt_idx=i, hotkey=f"hk{i}",
            checkpoint_hash="sha256:hash",
        )
        b.accept_submission(req)
    # Give the event loop a tick to ensure the event is visible.
    await asyncio.wait_for(b.seal_event.wait(), timeout=0.1)
    assert b.seal_event.is_set()


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
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey, prompt_idx=prompt_idx,
        window_start=500,
        merkle_root="00" * 32, rollouts=rollouts,
        checkpoint_hash="",  # gate disabled for these tests
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
# FIFO short-circuit (SUPERSEDED)
# ---------------------------------------------------------------------------
#
# v2.2: ordering is by validator-side TCP arrival (``arrived_at``). The first
# submission to clear the full pipeline for a given ``prompt_idx`` claims
# the slot; every subsequent submission for the same prompt is rejected
# SUPERSEDED before the heavy reward + GRAIL forward pass (~3s GPU).


def test_supersede_blocks_same_miner_duplicate_spam():
    """Same-miner spamming the same prompt is short-circuited: the second
    submission must be rejected SUPERSEDED before any heavy validation."""
    b = _make_batcher()
    first = _request_v21(
        prompt_idx=42, hotkey="A", checkpoint_hash="",
    )
    second = _request_v21(
        prompt_idx=42, hotkey="A", checkpoint_hash="",
    )
    assert b.accept_submission(first).accepted is True
    r2 = b.accept_submission(second)
    assert r2.accepted is False
    assert r2.reason == RejectReason.SUPERSEDED


def test_second_arrival_for_same_prompt_is_short_circuited():
    """v2.2: any subsequent submission for an already-claimed prompt is
    rejected SUPERSEDED before any heavy validation — no fall-through,
    no second entry in _valid."""
    b = _make_batcher()
    incumbent = _request_v21(
        prompt_idx=42, hotkey="B", checkpoint_hash="",
    )
    challenger = _request_v21(
        prompt_idx=42, hotkey="A", checkpoint_hash="",
    )
    assert b.accept_submission(incumbent).accepted is True
    r2 = b.accept_submission(challenger)
    assert r2.accepted is False
    assert r2.reason == RejectReason.SUPERSEDED
    # Only the first arrival is in _valid.
    assert len(b._valid) == 1
    assert b._valid[0].hotkey == "B"


def test_different_prompts_tracked_independently():
    """SUPERSEDED is per-prompt — accepting prompt 42 must not block prompt 43."""
    b = _make_batcher()
    r_a = b.accept_submission(_request_v21(
        prompt_idx=42, hotkey="A", checkpoint_hash="",
    ))
    r_b = b.accept_submission(_request_v21(
        prompt_idx=43, hotkey="A", checkpoint_hash="",
    ))
    assert r_a.accepted is True
    assert r_b.accepted is True
    assert len(b._valid) == 2
    assert b._claimed_prompts == {42, 43}


def test_supersede_only_records_after_full_success():
    """If a submission gets rejected mid-pipeline (e.g. GRAIL fail), it must
    NOT mark the prompt as claimed — otherwise an honest later submission
    for the same prompt would be wrongly rejected as SUPERSEDED."""
    # Use an always-failing GRAIL so the submission is rejected post-cheap-checks.
    b = _make_batcher(verify_commitment_proofs_fn=_always_false_grail)
    first_fails = _request_v21(
        prompt_idx=42, hotkey="A", checkpoint_hash="",
    )
    r1 = b.accept_submission(first_fails)
    assert r1.accepted is False
    assert r1.reason == RejectReason.GRAIL_FAIL
    # No claim recorded for prompt 42.
    assert 42 not in b._claimed_prompts
    # A subsequent honest submission for prompt 42 must NOT be wrongly
    # rejected as SUPERSEDED.
    b._verify_commitment = _always_true_grail
    r2 = b.accept_submission(_request_v21(
        prompt_idx=42, hotkey="B", checkpoint_hash="",
    ))
    assert r2.accepted is True


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
    """Stub that returns logits where EOS is highly probable everywhere."""
    def _fn(commit, model, randomness):
        logits = torch.zeros(seq_len, 100)
        logits[:, eos_id] = 5.0
        return ProofResult(
            all_passed=True, passed=1, checked=1, logits=logits,
            sketch_diff_max=0,
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
    seq_len = CHALLENGE_K + 4
    b = _make_batcher(
        model=_ModelStubWithVocab(),
        verify_commitment_proofs_fn=_grail_with_logits(seq_len),
    )
    req = _request()
    # Last token != 99 (EOS) — sequence ends in seq_len-1
    req.rollouts[0].commit["tokens"] = list(range(seq_len))
    req.rollouts[0].tokens = req.rollouts[0].commit["tokens"]
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_TERMINATION


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
    resp = b.accept_submission(req)
    assert resp.reason == RejectReason.GRAIL_FAIL

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
            )
        )
    req = BatchSubmissionRequest(
        miner_hotkey="hk", prompt_idx=42, window_start=500,
        merkle_root="00" * 32, rollouts=rollouts, checkpoint_hash="sha256:test",
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
    # rollout_hashes populated on the stored ValidSubmission
    stored = b.valid_submissions()[0]
    assert len(stored.rollout_hashes) == M_ROLLOUTS
    assert all(isinstance(h, bytes) and len(h) == 32 for h in stored.rollout_hashes)


def test_seal_batch_populates_hash_set():
    """After seal_batch, every batched rollout's hash is in the shared set."""
    from reliquary.validator.dedup import RolloutHashSet

    hs = RolloutHashSet(retention_windows=50)
    b = _make_batcher(hash_set=hs)
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True

    batch = b.seal_batch()
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
    batch = b.seal_batch()
    assert len(batch) == 1  # behaviour unchanged
