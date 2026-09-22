"""Which replica a proof slot builds: derived by default, pinned by a task that wants it uniform.

Left alone, the worker decides from the model and its card, so nobody has to know what a model is
made of. A task can pin one instead, which is how a fleet of unequal cards is made to run a single
path — but not a path a card cannot build, which is refused at startup rather than mid-window. A
single validator can still force one for an incident or a shadow run.
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


def test_a_task_that_pins_a_replica_gets_it_on_a_card_that_could_have_held_the_model():
    """A fleet of unequal cards is made to run one path by the task, not by each operator."""
    from reliquary.shared.replica_strategy import STREAMED, choose_replica

    assert choose_replica(weights_bytes=1, free_bytes=1_000, declared=STREAMED) == STREAMED


def test_a_task_cannot_pin_a_replica_a_card_cannot_build():
    """Refused at startup, not by running out of memory on some miner's rollout."""
    from reliquary.shared.replica_strategy import RESIDENT, ReplicaUnavailable, choose_replica

    with pytest.raises(ReplicaUnavailable, match="do not fit"):
        choose_replica(weights_bytes=900, free_bytes=1_000, declared=RESIDENT)


def test_forcing_a_path_on_one_validator_beats_what_the_task_pinned():
    """The override is for an incident or a shadow run, and forcing means forcing."""
    from reliquary.shared.replica_strategy import RESIDENT, STREAMED, choose_replica

    assert choose_replica(
        weights_bytes=900, free_bytes=1_000, override=RESIDENT, declared=STREAMED,
    ) == RESIDENT


def test_a_replica_nobody_implements_is_refused_wherever_it_is_named():
    from reliquary.shared.replica_strategy import choose_replica

    for field in ("override", "declared"):
        with pytest.raises(ValueError, match=field):
            choose_replica(weights_bytes=1, free_bytes=2, **{field: "quantised"})
