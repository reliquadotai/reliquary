"""Which replica a proof slot builds is derived from the model and the card, not configured.

An operator who ticks the wrong box on a task would either waste a card or fail to verify at all,
so the worker decides for itself and the flag exists only to force a path for a test.
"""

import pytest

from reliquary.shared.replica_strategy import STREAMED, RESIDENT, choose_replica


GIGA = 1024 ** 3


def test_a_model_that_fits_stays_resident():
    assert choose_replica(weights_bytes=8 * GIGA, free_bytes=80 * GIGA) == RESIDENT


def test_a_model_that_does_not_fit_is_streamed():
    assert choose_replica(weights_bytes=2000 * GIGA, free_bytes=80 * GIGA) == STREAMED


def test_room_is_kept_for_the_batch_the_pass_needs():
    """A model that fits to the byte leaves nothing to run on; that is not 'fits'."""
    assert choose_replica(weights_bytes=79 * GIGA, free_bytes=80 * GIGA) == STREAMED


def test_the_override_forces_either_way():
    assert choose_replica(weights_bytes=2000 * GIGA, free_bytes=80 * GIGA, override="resident") == RESIDENT
    assert choose_replica(weights_bytes=1 * GIGA, free_bytes=80 * GIGA, override="streamed") == STREAMED


def test_an_unknown_override_is_refused_loudly():
    with pytest.raises(ValueError, match="nonsense"):
        choose_replica(weights_bytes=GIGA, free_bytes=80 * GIGA, override="nonsense")


def test_weights_are_measured_from_the_checkpoint(tmp_path):
    """The decision is taken before anything is loaded, so it reads the files on disk."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from reliquary.shared.replica_strategy import checkpoint_weight_bytes

    config = AutoConfig.for_model(
        "qwen3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    model.save_pretrained(str(tmp_path))
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    measured = checkpoint_weight_bytes(tmp_path)
    assert 0.9 * expected <= measured <= 1.2 * expected
