"""The proof loop pays one pass for a slice, and is otherwise exactly the loop it was.

Proving a group one rollout at a time is deliberate: each carries its own receipt and deadline,
and a GRAIL failure stops the ones behind it. Warming must not touch any of that — it only means
the model was already driven when a rollout's turn comes.
"""

import torch

from reliquary.constants import M_ROLLOUTS
from reliquary.validator.verifier import ProofResult
from tests.unit.test_grpo_window_batcher import _make_batcher, _request


def _verifier(warm=None):
    """A verifier shaped like the one the isolated plane injects."""
    proved = []

    def verify(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        proved.append(tuple(commit["tokens"]))
        return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))

    if warm is not None:
        verify.warm = warm
    return verify, proved


def _prove(verify):
    batcher = _make_batcher(verify_commitment_proofs_fn=verify)
    batcher.accept_submission(_request(prompt_idx=7, hotkey="miner"))
    return batcher._verify_expensive(batcher.pending_submissions()[0])


def test_one_warming_pass_covers_the_group_the_loop_then_proves():
    warmed = []
    verify, proved = _verifier(warm=lambda inputs, model, randomness: warmed.append(
        [tuple(commit["tokens"]) for commit, _ in inputs]
    ))

    proven = _prove(verify)

    assert proven is not None
    assert len(warmed) == 1, "a group is warmed once, not once per rollout"
    assert warmed[0] == proved, "and it covers exactly the rollouts the loop then proves"
    assert len(proved) == M_ROLLOUTS, "every rollout is still proved on its own"


def test_a_warming_pass_that_fails_costs_the_group_nothing():
    """Warming is an optimisation. Failing a group over it would cost a miner a window."""
    def _broken(inputs, model, randomness):
        raise RuntimeError("the pass did not run")

    verify, proved = _verifier(warm=_broken)

    proven = _prove(verify)

    assert proven is not None, "the group is proved anyway"
    assert len(proved) == M_ROLLOUTS, "each rollout falls back to driving the model itself"


def test_a_verifier_without_a_warm_pass_is_driven_exactly_as_before():
    verify, proved = _verifier()

    proven = _prove(verify)

    assert proven is not None
    assert len(proved) == M_ROLLOUTS


def test_the_seeds_a_warming_pass_is_given_are_the_ones_the_proofs_use():
    """A row computed under different seeds than the proof reads would be a wrong verdict."""
    warmed, used = [], []

    def _warm(inputs, model, randomness):
        warmed.extend(seed for _, seed in inputs)

    verify, _ = _verifier(warm=_warm)
    inner = verify

    def _record(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        used.append(seed_u_values)
        return inner(commit, model, randomness, tokenizer=tokenizer, seed_u_values=seed_u_values)

    _record.warm = _warm
    assert _prove(_record) is not None
    assert warmed == used


def test_a_group_larger_than_a_slice_is_warmed_slice_by_slice(monkeypatch):
    """Rows are held on the card until they are spent, so a slice bounds what is in flight."""
    from reliquary.validator import batcher as batcher_module

    monkeypatch.setattr(batcher_module, "PROOF_WARM_ROLLOUTS", M_ROLLOUTS // 2)
    warmed = []
    verify, proved = _verifier(warm=lambda inputs, model, randomness: warmed.append(
        [tuple(commit["tokens"]) for commit, _ in inputs]
    ))

    assert _prove(verify) is not None

    assert [len(slice_) for slice_ in warmed] == [M_ROLLOUTS // 2] * 2
    assert warmed[0] + warmed[1] == proved, "the slices cover the group, in order, once each"
