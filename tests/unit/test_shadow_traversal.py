"""A resident slot can run the streamed traversal beside its own forward and compare.

This is the only place the two can be compared on the real checkpoint with real rollouts, and it
stops being possible the day a model arrives that no card can hold. What it must never do is
change a verdict: the proof is decided by the same forward as before, whatever the shadow says.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary.shared.replica_strategy import RESIDENT, STREAMED


def _context(tmp_path):
    config = AutoConfig.for_model(
        "qwen3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        eos_token_id=2, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    return {"model": model, "replica": RESIDENT, "device": "cpu"}


_COMMIT = {"tokens": list(range(24))}


def test_a_sampled_proof_is_checked_against_the_streamed_traversal(tmp_path, monkeypatch):
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.PROOF_SHADOW_FRACTION", 1.0)
    monkeypatch.setattr("reliquary.constants.ATTN_IMPLEMENTATION", "eager")
    context = _context(tmp_path)

    assert proof_worker.shadow_traversal(context, _COMMIT) is True
    assert context["shadow"] == {"checked": 1, "mismatched": 0, "failed": 0}


def test_nothing_runs_when_the_fraction_is_zero(tmp_path, monkeypatch):
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.PROOF_SHADOW_FRACTION", 0.0)
    context = _context(tmp_path)

    assert proof_worker.shadow_traversal(context, _COMMIT) is False
    assert "shadow" not in context, "an idle shadow leaves no trace to read"


def test_a_streamed_slot_shadows_nothing(tmp_path, monkeypatch):
    """There is no second opinion to be had: the streamed traversal is what it already runs."""
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.PROOF_SHADOW_FRACTION", 1.0)
    context = _context(tmp_path) | {"replica": STREAMED}

    assert proof_worker.shadow_traversal(context, _COMMIT) is False


def test_a_disagreement_is_counted_and_never_raised(tmp_path, monkeypatch, caplog):
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.PROOF_SHADOW_FRACTION", 1.0)
    monkeypatch.setattr("reliquary.constants.ATTN_IMPLEMENTATION", "eager")
    import copy

    from reliquary.shared.streaming_forward import StreamedReplica

    context = _context(tmp_path)
    # A traversal that computes something else is what a silent divergence looks like from here:
    # same rollout, same layer index, a hidden state that is not the resident one.
    drifted = copy.deepcopy(context["model"])
    with torch.no_grad():
        drifted.model.layers[0].mlp.down_proj.weight.mul_(1.5)
    context["_shadow_replica"] = StreamedReplica.from_model(drifted, device="cpu")

    with caplog.at_level("ERROR", logger="reliquary.validator.proof_worker"):
        assert proof_worker.shadow_traversal(context, _COMMIT) is False

    assert context["shadow"]["mismatched"] == 1
    assert context["shadow"]["max_difference"] > 0
    assert "no verdict was changed" in caplog.text


def test_a_shadow_that_breaks_costs_the_proof_nothing(tmp_path, monkeypatch):
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.PROOF_SHADOW_FRACTION", 1.0)
    context = _context(tmp_path)
    monkeypatch.setattr(
        "reliquary.shared.streaming_forward.StreamedReplica.from_model",
        staticmethod(lambda *a, **k: 1 / 0),
    )

    assert proof_worker.shadow_traversal(context, _COMMIT) is False
    assert context["shadow"]["failed"] == 1


def test_the_sample_is_the_fraction_asked_for(tmp_path, monkeypatch):
    from reliquary.validator import proof_worker

    monkeypatch.setattr("reliquary.constants.PROOF_SHADOW_FRACTION", 0.25)
    monkeypatch.setattr("reliquary.constants.ATTN_IMPLEMENTATION", "eager")
    draws = iter([0.1, 0.9, 0.2, 0.3, 0.99])
    monkeypatch.setattr("random.random", lambda: next(draws))
    context = _context(tmp_path)

    for _ in range(5):
        proof_worker.shadow_traversal(context, _COMMIT)

    assert context["shadow"]["checked"] == 2, "only the draws under the fraction"
