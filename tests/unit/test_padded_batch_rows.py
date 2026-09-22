"""A rollout batched with rollouts of other lengths must read what it reads alone.

This is the assertion the whole batching rests on once lengths are allowed to differ, and it has
two halves. Mathematically it holds by construction: attention here is causal, so a position never
reads a later one, and everything padded onto a rollout comes after it. Numerically it is a
property of the kernel — a wider batch tiles and reduces differently — so it is asserted here, and
must be asserted again on the card and the model that will run it.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary.shared.forward import forward_single_layer
from reliquary.validator.verifier import forward_rows_for_batch, plan_verification_passes

MODEL_TYPES = ("qwen3", "qwen3_moe")
# What production runs. The same comparison in float32 differs by ~2e-6, which bfloat16 cannot
# represent: the padding does not move these numbers, it moves ones below them.
DTYPE = torch.bfloat16


def _model(model_type: str):
    common = dict(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        eos_token_id=2, tie_word_embeddings=True,
    )
    if model_type == "qwen3_moe":
        common |= dict(
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            decoder_sparse_step=1, norm_topk_prob=True, mlp_only_layers=[],
        )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(
        AutoConfig.for_model(model_type, **common), attn_implementation="eager", dtype=DTYPE,
    ).eval()


def _commits(lengths):
    generator = torch.Generator().manual_seed(7)
    return [
        {"tokens": torch.randint(0, 256, (length,), generator=generator).tolist()}
        for length in lengths
    ]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_a_padded_row_is_what_the_rollout_reads_alone(model_type):
    model = _model(model_type)
    # Lengths chosen to share nothing: this is what a real group looks like.
    commits = _commits([17, 24, 31, 40, 41, 63])

    rows = forward_rows_for_batch(commits, model, token_budget=10_000, pad=True)

    assert len(rows) == len(commits)
    for index, commit in enumerate(commits):
        alone, _ = forward_single_layer(
            model, torch.tensor([commit["tokens"]]), None, -1,
        )
        hidden, _ = rows[index]
        assert hidden.shape[0] == len(commit["tokens"]), "a row is cut back to its own tokens"
        assert torch.equal(hidden, alone[0]), f"rollout {index} of {model_type}"


def test_padding_is_what_turns_a_real_group_into_one_pass():
    """Unpadded, a group of rollouts that terminate on their own is not a batch at all."""
    commits = _commits([201, 202, 203, 204, 205, 206, 207, 208])

    unpadded = plan_verification_passes(commits, token_budget=100_000, pad=False)
    padded = plan_verification_passes(commits, token_budget=100_000, pad=True)

    assert [len(p) for p in unpadded] == [1] * 8
    assert [len(p) for p in padded] == [8]


def test_a_pass_is_budgeted_on_the_width_that_lands_on_the_card():
    """Padding makes every row as wide as the widest, and that is what the budget must count."""
    commits = _commits([10, 10, 100])

    passes = plan_verification_passes(commits, token_budget=200, pad=True)

    assert passes == [[0, 1], [2]], "a third row at width 100 would be 300 tokens, over budget"


def test_rollouts_are_grouped_with_their_nearest_lengths():
    """Sorted, so a pass pads as little as it can."""
    commits = _commits([100, 10, 99, 11])

    passes = plan_verification_passes(commits, token_budget=220, pad=True)

    assert passes == [[1, 3], [2, 0]]


def test_a_fused_kernel_moves_the_arithmetic_and_the_gate_absorbs_it():
    """Padding is exact by construction and approximate in practice, and the margin is measured.

    An unfused attention returns the padded row bit for bit. A fused one tiles over keys, and the
    padded width changes how the tiles fall, so the hidden states move — not because a rollout
    read its padding, but because the same sum was added up in another order. What matters is
    whether the check downstream can tell, and the sketch tolerance is what answers that: it
    exists to absorb the gap between the miner's runtime and the validator's, which is far wider
    than this. Asserted here as a fraction of that budget, on the metric the gate reads.
    """
    from reliquary.constants import PROOF_SKETCH_TOLERANCE_BASE
    from reliquary.protocol.grail_verifier import GRAILVerifier
    from reliquary.validator.verifier import (
        verify_commitment_proofs, verify_commitment_proofs_batch,
    )

    randomness = "ab" * 32
    config = AutoConfig.for_model(
        "qwen3", vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=4, max_position_embeddings=1024,
        eos_token_id=2, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="sdpa", dtype=DTYPE,
    ).eval()

    commits = []
    for seed, length in enumerate((137, 250, 251, 399)):
        tokens = torch.randint(
            0, 512, (length,), generator=torch.Generator().manual_seed(seed),
        ).tolist()
        hidden, _ = forward_single_layer(model, torch.tensor([tokens]), None, -1)
        verifier = GRAILVerifier(hidden_dim=config.hidden_size)
        commits.append({
            "tokens": tokens,
            "commitments": verifier.create_commitments_batch(
                hidden[0], verifier.generate_r_vec(randomness),
            ),
            "rollout": {"prompt_length": length // 4, "completion_length": length - length // 4},
        })

    alone = [verify_commitment_proofs(c, model, randomness) for c in commits]
    padded = verify_commitment_proofs_batch(
        commits, model, randomness, token_budget=100_000, pad=True,
    )

    assert all(result.all_passed for result in alone), "the unpadded proof is the reference"
    for one, other in zip(alone, padded):
        assert other.all_passed == one.all_passed, "the verdict is the verdict"
        assert other.sketch_diff_max < PROOF_SKETCH_TOLERANCE_BASE // 4, (
            "padding must stay a small share of a tolerance that absorbs whole runtimes"
        )
