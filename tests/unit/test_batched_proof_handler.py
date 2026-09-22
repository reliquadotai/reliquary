"""The worker proves a list of rollouts in one pass, and says which replica it is."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

RANDOMNESS = "cd" * 32


def _context(tmp_path, monkeypatch, streamed: bool):
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.ATTN_IMPLEMENTATION", "eager")
    if streamed:
        monkeypatch.setenv("RELIQUARY_PROOF_REPLICA", "streamed")
    config = AutoConfig.for_model(
        "qwen3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        eos_token_id=2, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).to(torch.float32).eval().save_pretrained(str(tmp_path))
    config.save_pretrained(str(tmp_path))
    return proof_worker.build_proof_context(checkpoint=str(tmp_path), device="cpu")


def _commit(model, length, seed):
    from reliquary.constants import LAYER_INDEX
    from reliquary.protocol.grail_verifier import GRAILVerifier
    from reliquary.shared.forward import forward_single_layer

    tokens = torch.randint(0, 256, (length,), generator=torch.Generator().manual_seed(seed)).tolist()
    hidden, _ = forward_single_layer(model, torch.tensor([tokens]), None, LAYER_INDEX)
    verifier = GRAILVerifier(hidden_dim=model.config.hidden_size)
    commitments = verifier.create_commitments_batch(hidden[0], verifier.generate_r_vec(RANDOMNESS))
    return {
        "tokens": tokens,
        "commitments": commitments,
        "rollout": {"prompt_length": length // 4, "completion_length": length - length // 4},
    }


def test_the_handler_proves_a_list_in_one_pass(tmp_path, monkeypatch):
    from reliquary.validator import proof_worker, verifier as verifier_module

    context = _context(tmp_path, monkeypatch, streamed=False)
    model = context["model"]
    commits = [_commit(model, 24, seed) for seed in range(3)]

    passes = []
    original = verifier_module.forward_single_layer_for_batch
    monkeypatch.setattr(
        verifier_module, "forward_single_layer_for_batch",
        lambda *a, **k: (passes.append(a[1].shape), original(*a, **k))[1],
    )
    results = proof_worker.run_commitment_proof(context, commits, RANDOMNESS, [None] * 3)

    assert len(results) == 3 and all(r.all_passed for r in results)
    assert passes == [torch.Size([3, 24])], "one pass carried all three"


def test_a_single_commit_still_returns_a_single_result(tmp_path, monkeypatch):
    """The one-at-a-time path is what every resident slot still uses."""
    from reliquary.validator import proof_worker

    context = _context(tmp_path, monkeypatch, streamed=False)
    commit = _commit(context["model"], 24, 7)
    result = proof_worker.run_commitment_proof(context, commit, RANDOMNESS, None)
    assert not isinstance(result, list) and result.all_passed


def test_warming_a_batch_spares_every_later_proof_its_own_pass(tmp_path, monkeypatch):
    """The server proves item by item; warming is how those items share one traversal."""
    import reliquary.shared.forward as forward_module
    from reliquary.validator import proof_worker

    context = _context(tmp_path, monkeypatch, streamed=False)
    model = context["model"]
    commits = [_commit(model, 24, seed) for seed in range(3)]

    shapes = []
    original = forward_module.forward_single_layer
    monkeypatch.setattr(
        forward_module, "forward_single_layer",
        lambda *a, **k: (shapes.append(tuple(a[1].shape)), original(*a, **k))[1],
    )
    proof_worker.warm_commitment_batch(context, commits, RANDOMNESS, [None] * 3)
    warmed = [proof_worker.run_commitment_proof(context, commit, RANDOMNESS, None) for commit in commits]

    assert all(r.all_passed for r in warmed)
    assert shapes == [(3, 24)], "one pass carried the three, and no proof ran one of its own"


def test_a_warmed_batch_is_spent_once(tmp_path, monkeypatch):
    """A rollout proved twice must not silently reuse a forward meant for the first attempt."""
    from reliquary.validator import proof_worker

    context = _context(tmp_path, monkeypatch, streamed=False)
    commit = _commit(context["model"], 24, 11)
    proof_worker.warm_commitment_batch(context, [commit], RANDOMNESS, [None])
    first = proof_worker.run_commitment_proof(context, commit, RANDOMNESS, None)
    second = proof_worker.run_commitment_proof(context, commit, RANDOMNESS, None)
    assert first.all_passed and second.all_passed
    assert not context.get("_warm"), "the warmed rows are released as they are consumed"
