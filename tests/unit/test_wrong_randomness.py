"""Validator must reject submissions whose claimed ``beacon.randomness``
doesn't match the per-window randomness it derived from
``block_hash + drand_round``.

This is the security control that ties the GRAIL r_vec to a per-window
seed. Without it, the sketch-tolerance window (PROOF_SKETCH_TOLERANCE_BASE
= 5000 mod q ≈ 2.15e9) is wide enough that miners can ship commits built
with a pre-computed constant r_vec and still slip under the per-position
sketch_diff threshold — defeating the whole point of per-window randomness.

Empirical motivation: real-world miners using ``block_hash(0) + drand round 1``
(i.e. genesis-derived constants) were observed with sketch_diff_max in the
3000–5000 band on >95 % of their accepted submissions, vs the ~0 we'd see
on randomness-matched commits where only bf16 rounding noise remains.
"""

from __future__ import annotations

import pytest

from reliquary.constants import CHALLENGE_K
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RejectReason,
    RolloutSubmission,
)

from tests.unit.test_grpo_window_batcher import (  # type: ignore[import-not-found]
    _make_batcher,
    _make_commit,
)


def _request_with_randomness(
    rollout_randomness: str,
    *,
    prompt_idx: int = 42,
    window_start: int = 500,
    rewards: list[float] | None = None,
    hotkey: str = "hk",
) -> BatchSubmissionRequest:
    """Build an in-zone request where every rollout's ``commit.beacon.randomness``
    is forced to ``rollout_randomness``."""
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for idx, r in enumerate(rewards):
        tokens = [t + idx for t in range(CHALLENGE_K + 4)]
        commit = _make_commit(tokens=tokens, success=r > 0.5, total_reward=r)
        commit["beacon"]["randomness"] = rollout_randomness
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


def test_match_accepts() -> None:
    """Baseline: when miner-claimed randomness equals the validator's window
    randomness, the binding check is a no-op and the submission flows
    through to GRAIL (which the test stub passes)."""
    b = _make_batcher()
    b.randomness = "ab" * 16  # validator-derived
    req = _request_with_randomness("ab" * 16)
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED


def test_constant_genesis_randomness_rejected() -> None:
    """The exploit we're closing: miner pre-computes randomness from
    ``block_hash(0) + drand_round_1`` (a publicly-known constant) and
    submits it on every window. After this PR, the binding check must
    reject before GRAIL ever runs."""
    b = _make_batcher()
    b.randomness = "aa" * 32  # validator's per-window seed (any non-genesis)
    # 64-hex constant produced from the literal genesis hash + drand round 1,
    # the cheapest "I don't actually fetch chain state" replay payload.
    genesis_constant = "f174fec" + "0" * (64 - len("f174fec"))
    req = _request_with_randomness(genesis_constant)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WRONG_RANDOMNESS


def test_neighbour_window_randomness_rejected() -> None:
    """A miner could try replaying randomness from window N-1 to skip the
    block-hash fetch latency. The check is byte-for-byte equality, so even
    a one-window-old seed is rejected."""
    b = _make_batcher()
    b.randomness = "11" * 32  # window N
    req = _request_with_randomness("12" * 32)  # window N-1, say
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WRONG_RANDOMNESS


def test_empty_or_missing_beacon_is_schema_rejected() -> None:
    """An empty / missing beacon is caught upstream by the CommitModel
    schema (``randomness_hex: str = Field(..., pattern=r"^[0-9a-fA-F]+$")``
    requires ≥1 hex char). We pin this layering so future refactors don't
    silently relax the schema and rely on the binding check alone — those
    are two independent defences and both must hold."""
    b = _make_batcher()
    b.randomness = "ab" * 32
    req = _request_with_randomness("")
    resp = b.accept_submission(req)
    assert resp.accepted is False
    # The schema catches it first — BAD_SCHEMA, not WRONG_RANDOMNESS.
    assert resp.reason == RejectReason.BAD_SCHEMA


def test_rejection_happens_before_grail() -> None:
    """If we reach the GRAIL forward pass with mismatched randomness, every
    OPEN window pays one full ~5-25 s verify per bad submission, defeating
    the purpose of the early-reject. Confirm the binding check shortcuts
    GRAIL by counting how often the stub function is invoked."""
    grail_calls: list[int] = []

    def _counting_grail(commit, model, randomness):
        grail_calls.append(1)
        import torch
        from reliquary.validator.verifier import ProofResult
        return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))

    b = _make_batcher(verify_commitment_proofs_fn=_counting_grail)
    b.randomness = "11" * 32
    req = _request_with_randomness("22" * 32)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WRONG_RANDOMNESS
    assert grail_calls == []   # GRAIL must not have run


def test_rejection_after_signature_check() -> None:
    """A bad-signature commit should still report BAD_SIGNATURE, not
    WRONG_RANDOMNESS — the signature check is cheaper and more specific.
    This pins the relative ordering of the two cheap pre-GRAIL checks."""
    def _always_false_sig(commit, hotkey):
        return False

    b = _make_batcher(verify_signature_fn=_always_false_sig)
    b.randomness = "11" * 32
    req = _request_with_randomness("22" * 32)  # also wrong randomness
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_SIGNATURE


def test_per_rollout_randomness_is_checked() -> None:
    """All M rollouts in a group share one window — any single rollout with
    the wrong randomness must reject the whole submission. Otherwise a
    miner could include 7 correct + 1 constant and slip the bad one through
    the GRAIL tolerance."""
    b = _make_batcher()
    b.randomness = "ab" * 32

    rewards = [1.0] * 4 + [0.0] * 4
    rollouts = []
    for idx, r in enumerate(rewards):
        tokens = [t + idx for t in range(CHALLENGE_K + 4)]
        commit = _make_commit(tokens=tokens, success=r > 0.5, total_reward=r)
        # Force the 5th rollout (idx=4) to carry stale randomness while the
        # rest correctly cite the validator's window seed.
        commit["beacon"]["randomness"] = "ab" * 32 if idx != 4 else "00" * 32
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"],
                reward=r,
                commit=commit,
                env_name="openmathinstruct",
            )
        )
    req = BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=7,
        window_start=500,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
        protocol_version=2,
    )
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WRONG_RANDOMNESS


def test_telemetry_records_wrong_randomness() -> None:
    """Operators need to see WRONG_RANDOMNESS show up in the per-window
    ``reject_summary`` so we can detect a flood of stale-randomness
    submissions hitting the validator (e.g. after the PR lands and a
    miner hasn't updated yet). Touching the existing reject-counts
    accounting is part of the contract."""
    b = _make_batcher()
    b.randomness = "11" * 32
    req = _request_with_randomness("22" * 32)
    b.accept_submission(req)
    assert b.reject_counts.get(RejectReason.WRONG_RANDOMNESS.value, 0) == 1
