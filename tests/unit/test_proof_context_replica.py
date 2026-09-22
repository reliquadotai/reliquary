"""A proof slot builds the replica its card can carry, and says which one it built."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary.shared.replica_strategy import RESIDENT, STREAMED


def _tiny_checkpoint(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.for_model(
        "qwen3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        eos_token_id=2, tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    model.save_pretrained(str(tmp_path))
    config.save_pretrained(str(tmp_path))
    return tmp_path


def test_a_model_that_fits_is_loaded_resident_as_it_always_was(tmp_path, monkeypatch):
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.ATTN_IMPLEMENTATION", "eager")
    context = proof_worker.build_proof_context(checkpoint=str(_tiny_checkpoint(tmp_path)), device="cpu")
    assert context["replica"] == RESIDENT
    assert hasattr(context["model"], "generate"), "the resident path still returns the model itself"


def test_the_override_builds_a_streamed_replica(tmp_path, monkeypatch):
    from reliquary.shared.streaming_forward import StreamedReplica
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.ATTN_IMPLEMENTATION", "eager")
    monkeypatch.setenv("RELIQUARY_PROOF_REPLICA", "streamed")
    context = proof_worker.build_proof_context(checkpoint=str(_tiny_checkpoint(tmp_path)), device="cpu")
    assert context["replica"] == STREAMED
    assert isinstance(context["model"], StreamedReplica)


def test_both_replicas_verify_a_rollout_the_same_way(tmp_path, monkeypatch):
    """The point of the whole change: downstream cannot tell which replica it was given."""
    from reliquary.shared.forward import forward_single_layer
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.ATTN_IMPLEMENTATION", "eager")
    checkpoint = str(_tiny_checkpoint(tmp_path))
    resident = proof_worker.build_proof_context(checkpoint=checkpoint, device="cpu")["model"]
    monkeypatch.setenv("RELIQUARY_PROOF_REPLICA", "streamed")
    streamed = proof_worker.build_proof_context(checkpoint=checkpoint, device="cpu")["model"]

    tokens = torch.randint(0, 256, (1, 20), generator=torch.Generator().manual_seed(3))
    hidden, logits = forward_single_layer(resident, tokens, None, -1)
    streamed_hidden, streamed_logits = forward_single_layer(streamed, tokens, None, -1)
    assert torch.equal(streamed_hidden, hidden)
    assert torch.equal(streamed_logits, logits)


def test_a_streamed_slot_survives_a_checkpoint_rotation(tmp_path, monkeypatch):
    """A rotation must rebuild the replica, not load a state dict into something that has no layers."""
    import shutil

    import torch

    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.streaming_forward import StreamedReplica
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.ATTN_IMPLEMENTATION", "eager")
    monkeypatch.setenv("RELIQUARY_PROOF_REPLICA", "streamed")
    first = _tiny_checkpoint(tmp_path / "first")
    context = proof_worker.build_proof_context(checkpoint=str(first), device="cpu")

    # a second checkpoint, staged as the intake stages one
    second = _tiny_checkpoint(tmp_path / "second")
    proof_worker.reload_proof_context(context, str(second), "rev-2")

    assert context["revision"] == "rev-2"
    assert isinstance(context["model"], StreamedReplica)
    tokens = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(4))
    resident = AutoModelForCausalLM.from_pretrained(str(second), dtype=torch.bfloat16, attn_implementation="eager").eval()
    hidden, _ = forward_single_layer(resident, tokens, None, -1)
    # Intake drops the staged copy as soon as the swap completes, and this replica reads its
    # layers from disk on every pass: it has to be reading the store, not the download.
    shutil.rmtree(second)
    streamed, _ = forward_single_layer(context["model"], tokens, None, -1)
    assert torch.equal(streamed, hidden), "the rotated replica serves the new checkpoint"
