from types import SimpleNamespace

from reliquary.eval_dashboard.runner import _generation_kwargs, derive_sample_seed


def test_sample_seed_is_stable_and_partitioned():
    args = ("reliquary-eval-v2-seed", "a" * 64, "task-1", 0)
    first = derive_sample_seed(*args)
    assert first == derive_sample_seed(*args)
    assert first != derive_sample_seed(args[0], args[1], args[2], 1)
    assert first != derive_sample_seed(args[0], "b" * 64, args[2], 0)
    assert 0 <= first < 2**63


def test_generation_kwargs_neutralize_checkpoint_local_processors():
    config = SimpleNamespace(
        generation=SimpleNamespace(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
        )
    )
    kwargs = _generation_kwargs(
        config,
        max_new_tokens=512,
        eos_ids={1, 2},
        pad_token_id=1,
    )
    assert kwargs["do_sample"] is True
    assert kwargs["eos_token_id"] == [1, 2]
    assert kwargs["no_repeat_ngram_size"] == 0
    assert kwargs["min_new_tokens"] == 0
    assert kwargs["forced_eos_token_id"] is None
    assert kwargs["sequence_bias"] is None
