"""One forward for a batch of rollouts must verify them exactly as one forward each did.

The streamed replica pays a full traversal of the model per forward, so verifying rollouts one at
a time would pay it over and over. Batching is what makes that cost divisible — but only if the
verdicts do not move.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary.validator.verifier import verify_commitment_proofs

RANDOMNESS = "ab" * 32


def _model(tmp_path):
    config = AutoConfig.for_model(
        "qwen3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        eos_token_id=2, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config).to(torch.float32).eval()


def _commit(model, length, seed):
    """A commitment shaped the way the proof path receives one."""
    from reliquary.protocol.grail_verifier import GRAILVerifier
    from reliquary.shared.forward import forward_single_layer
    from reliquary.constants import LAYER_INDEX

    tokens = torch.randint(0, 256, (length,), generator=torch.Generator().manual_seed(seed)).tolist()
    hidden, _ = forward_single_layer(model, torch.tensor([tokens]), None, LAYER_INDEX)
    verifier = GRAILVerifier(hidden_dim=model.config.hidden_size)
    r_vec = verifier.generate_r_vec(RANDOMNESS)
    commitments = verifier.create_commitments_batch(hidden[0], r_vec)
    prompt = length // 4
    return {
        "tokens": tokens,
        "commitments": commitments,
        "rollout": {"prompt_length": prompt, "completion_length": length - prompt},
    }


def test_a_batched_forward_gives_the_same_verdicts(tmp_path):
    from reliquary.validator.verifier import verify_commitment_proofs_batch

    model = _model(tmp_path)
    commits = [_commit(model, length, seed) for seed, length in enumerate((24, 24, 40))]

    one_at_a_time = [verify_commitment_proofs(c, model, RANDOMNESS) for c in commits]
    batched = verify_commitment_proofs_batch(commits, model, RANDOMNESS)

    assert len(batched) == len(one_at_a_time)
    for batched_result, single in zip(batched, one_at_a_time):
        assert batched_result.all_passed == single.all_passed
        assert (batched_result.passed, batched_result.checked) == (single.passed, single.checked)
        assert batched_result.sketch_diff_max == single.sketch_diff_max


def test_a_batch_is_split_under_the_token_budget(tmp_path, monkeypatch):
    """A budget too small to hold the batch splits it rather than running out of memory."""
    from reliquary.validator import verifier as verifier_module

    model = _model(tmp_path)
    commits = [_commit(model, 24, seed) for seed in range(4)]
    seen = []
    original = verifier_module.forward_single_layer_for_batch

    def spy(model_, tokens, mask, layer_index, **kwargs):
        seen.append(tokens.shape)
        return original(model_, tokens, mask, layer_index, **kwargs)

    monkeypatch.setattr(verifier_module, "forward_single_layer_for_batch", spy)
    verifier_module.verify_commitment_proofs_batch(commits, model, RANDOMNESS, token_budget=48)

    assert len(seen) == 2, "four rollouts of 24 tokens under a 48-token budget make two passes"
    assert all(shape[0] == 2 for shape in seen)


def test_rollouts_of_different_lengths_stay_in_their_own_pass(tmp_path):
    """Padding a short rollout up to a long one would change the numbers its verdict reads."""
    from reliquary.validator import verifier as verifier_module

    model = _model(tmp_path)
    commits = [_commit(model, length, seed) for seed, length in enumerate((16, 32, 16))]
    passes = verifier_module.plan_verification_passes(commits, token_budget=10_000)
    assert [sorted(group) for group in passes] == [[0, 2], [1]]
