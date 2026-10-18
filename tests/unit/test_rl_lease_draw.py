"""One lease, one draw: the miner, the validator (audits) and the trainer key the forced draw identically from the
lease (``protocol/rl_draw.py``). The miner's side is its real path: the lease answer, ``draw_context``, the vLLM
request's extra args and the runner-keyed processor's per-position uniform."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX, pick_rows, u_at
from reliquary.miner.rl_async import HeldLease, draw_context
from reliquary.miner.vllm_generation import FORCED_SEED_KEY, _forced_state, _uniform, forced_seed_extra_args
from reliquary.protocol.rl_draw import lease_draw_context, lease_uniforms, pooled_rate, rollout_rate
from reliquary.validator.rl_buffer import SubmittedGroup
from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseRequest
from reliquary.validator.training import _forced_draw_uniforms

ENV = "math"
LENGTHS = [5, 1, 9] + [3] * 13


@pytest.fixture
def lease(tmp_path):
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}),
                     quota=QuotaController(QuotaConfig(initial=4)), clock=lambda: 1_800_000_000.0,
                     rng=random.Random(0))
    book.adopt_checkpoint(7)
    yield book.grant([LeaseRequest("A", ENV, (42,), 7)])[0].leases[0]
    book.close()


def miner_uniforms(held: HeldLease, rollout_index: int, positions: int) -> list[float]:
    extra = forced_seed_extra_args(**draw_context(held), rollout_index=rollout_index)
    state = _forced_state(extra[FORCED_SEED_KEY])
    return [_uniform(state, t) for t in range(positions)]


def test_miner_validator_and_trainer_draw_the_same_uniforms_for_one_lease(lease):
    held = HeldLease.from_wire(lease.to_public())                     # what the miner is told
    completions = tuple(tuple(range(n)) for n in LENGTHS)
    group = SubmittedGroup(lease.lease_id, "A", ENV, lease.prompt_idx, lease.checkpoint, lease.nonce, 0.0,
                           (0.0,) * 16, (False,) * 16, (1, 2), completions, (None,) * 16)
    group = SubmittedGroup.unpack(group.pack())                        # as the buffer hands it to the trainer
    for index, n in enumerate(LENGTHS):
        miner = miner_uniforms(held, index, n)
        validator = group.draw_uniforms(index)
        rollout = SimpleNamespace(_forced_draw_uniforms=validator)     # what the trainer's loss reads
        trainer = _forced_draw_uniforms(rollout, n)
        assert miner == validator == trainer == lease_uniforms(lease.nonce, 42, 7, index, n)
    # keyed by the nonce, the prompt, k and the rollout index; never by the holder
    assert draw_context(held) == lease_draw_context(lease.nonce, 42, 7) == {
        "randomness": lease.nonce, "prompt_idx": 42, "checkpoint_hash": "k7"}
    assert group.draw_uniforms(0) != group.draw_uniforms(3)[:5]
    assert lease_uniforms(lease.nonce, 42, 8, 0, 4) != lease_uniforms(lease.nonce, 42, 7, 0, 4)
    assert group.draw_uniforms(0)[0] == u_at(lease.nonce, 42, "k7", 0, 0)


def test_the_gumbel_picks_agree_given_the_same_logits(lease):
    held = HeldLease.from_wire(lease.to_public())
    logits = torch.randn(6, 97, generator=torch.Generator().manual_seed(0))
    miner = pick_rows(logits, miner_uniforms(held, 2, 6), draw=DRAW_GUMBEL_MAX)
    validator = pick_rows(logits, lease_uniforms(lease.nonce, lease.prompt_idx, lease.checkpoint, 2, 6),
                          draw=DRAW_GUMBEL_MAX)
    assert torch.equal(miner, validator)


def test_a_malformed_nonce_or_checkpoint_is_refused():
    with pytest.raises(ValueError):
        lease_draw_context("ABCD", 1, 0)
    with pytest.raises(ValueError):
        lease_draw_context("zz", 1, 0)
    with pytest.raises(ValueError):
        lease_draw_context("ab", 1, -1)


def test_a_rollout_too_short_for_a_rate_is_left_to_the_gap_tests():
    """(stochastic, stochastic matches, drawn, drawn matches): one honest near-tie in 3 tokens is no evidence."""
    assert rollout_rate((3, 2, 3, 2)) == 1.0 and rollout_rate((0, 0, 0, 0)) == 0.0
    assert rollout_rate((5, 4, 40, 30)) == 0.75                     # long enough: the rate on every drawn position
    assert rollout_rate((25, 20, 40, 35)) == 0.8                    # enough stochastic positions: their rate
    assert pooled_rate([(3, 2, 3, 2)] * 4) == 8 / 12                # a group pools every position, short or not
