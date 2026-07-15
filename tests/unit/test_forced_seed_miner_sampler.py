import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment import forced_sampling as fs
from reliquary.miner.forced_seed_sampler import (
    ForcedSeedLogitsProcessor,
    forced_seed_generate_kwargs,
    phase2_base_offsets,
)


def _forced_token(logits_row, randomness, hotkey, prompt_idx, ckpt, roll, t):
    probs = fs.warp(logits_row, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    return fs.pick(probs, fs.u_at(randomness, prompt_idx, ckpt, roll, t))


def test_processor_forces_pick_and_advances_position():
    r, hk, pidx, ck = "rr", "hk", 7, "ck"
    proc = ForcedSeedLogitsProcessor(
        randomness=r, hotkey=hk, prompt_idx=pidx, checkpoint_hash=ck,
        rollout_indices=[0], base_offsets=[0], start_len=2,
    )
    # step s=0: input_ids length == start_len -> completion offset t=0
    logits0 = torch.tensor([[0.2, 0.1, 0.0, 0.15]])
    out0 = proc(torch.zeros(1, 2, dtype=torch.long), logits0.clone())
    assert int(out0.argmax(-1)) == _forced_token(logits0[0], r, hk, pidx, ck, 0, 0)
    # step s=1: input_ids length == start_len + 1 -> completion offset t=1
    logits1 = torch.tensor([[0.0, 0.3, 0.1, 0.2]])
    out1 = proc(torch.zeros(1, 3, dtype=torch.long), logits1.clone())
    assert int(out1.argmax(-1)) == _forced_token(logits1[0], r, hk, pidx, ck, 0, 1)


def test_processor_uses_per_row_rollout_index_and_base_offset():
    r, hk, pidx, ck = "rr", "hk", 7, "ck"
    proc = ForcedSeedLogitsProcessor(
        randomness=r, hotkey=hk, prompt_idx=pidx, checkpoint_hash=ck,
        rollout_indices=[3, 5], base_offsets=[0, 10], start_len=4,
    )
    scores = torch.tensor([[0.2, 0.1, 0.0, 0.15], [0.05, 0.2, 0.1, 0.0]])
    out = proc(torch.zeros(2, 4, dtype=torch.long), scores.clone())  # s=0
    # row 0 -> rollout 3 at offset 0; row 1 -> rollout 5 at offset 10
    assert int(out[0].argmax(-1)) == _forced_token(scores[0], r, hk, pidx, ck, 3, 0)
    assert int(out[1].argmax(-1)) == _forced_token(scores[1], r, hk, pidx, ck, 5, 10)


def test_forced_seed_generate_kwargs_strips_warpers_and_forces_greedy():
    # HF must not apply its own temperature/top_k/top_p on top of the
    # processor's protocol warp, or the miner would double-warp and drift
    # off the validator's forced pick. do_sample=False makes greedy select
    # the processor's one-hot token.
    proc = ForcedSeedLogitsProcessor(
        randomness="r", hotkey="h", prompt_idx=0, checkpoint_hash="c",
        rollout_indices=[0], base_offsets=[0], start_len=1,
    )
    base = {"max_new_tokens": 16, "do_sample": True, "temperature": T_PROTO,
            "top_p": TOP_P_PROTO, "top_k": TOP_K_PROTO, "pad_token_id": 0}
    kw = forced_seed_generate_kwargs(base, proc)
    assert "temperature" not in kw and "top_p" not in kw and "top_k" not in kw
    assert kw["do_sample"] is False
    assert kw["max_new_tokens"] == 16 and kw["pad_token_id"] == 0  # untouched
    assert proc in list(kw["logits_processor"])
    assert base["temperature"] == T_PROTO  # input dict not mutated


def test_forced_seed_generate_kwargs_neutralizes_generation_config_processors():
    # HF builds repetition_penalty / no_repeat_ngram / min_length /
    # suppress_tokens (etc.) processors from the model's generation_config and
    # runs them BEFORE the forced processor -- they are NOT do_sample-gated. The
    # miner would then warp penalized logits while the validator warps raw ones,
    # a systematic honest false-mismatch. The forced kwargs must pin every such
    # processor to its inert value (kwargs override generation_config in HF).
    proc = ForcedSeedLogitsProcessor(
        randomness="r", hotkey="h", prompt_idx=0, checkpoint_hash="c",
        rollout_indices=[0], base_offsets=[0], start_len=1,
    )
    base = {
        "max_new_tokens": 8, "pad_token_id": 0,
        "repetition_penalty": 1.3, "encoder_repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3, "encoder_no_repeat_ngram_size": 3,
        "min_length": 5, "min_new_tokens": 5,
        "suppress_tokens": [1, 2], "begin_suppress_tokens": [3],
        "bad_words_ids": [[4]], "forced_bos_token_id": 1,
        "forced_eos_token_id": 2, "exponential_decay_length_penalty": (8, 1.1),
        "sequence_bias": {(1,): 1.0},
    }
    kw = forced_seed_generate_kwargs(base, proc)

    assert kw["repetition_penalty"] == 1.0
    assert kw["encoder_repetition_penalty"] == 1.0
    assert kw["no_repeat_ngram_size"] == 0
    assert kw["encoder_no_repeat_ngram_size"] == 0
    assert kw["min_length"] == 0
    assert kw["min_new_tokens"] == 0
    assert kw["suppress_tokens"] is None
    assert kw["begin_suppress_tokens"] is None
    assert kw["bad_words_ids"] is None
    assert kw["forced_bos_token_id"] is None
    assert kw["forced_eos_token_id"] is None
    assert kw["exponential_decay_length_penalty"] is None
    assert kw["sequence_bias"] is None
    # Non-processor kwargs preserved; input dict not mutated.
    assert kw["max_new_tokens"] == 8 and kw["pad_token_id"] == 0
    assert base["repetition_penalty"] == 1.3


def test_phase2_base_offsets_are_per_row_completion_offsets():
    # Each BFT phase-2 row resumes from its own primed length; the first
    # sampled answer token sits at completion offset (primed_len - prompt_len).
    assert phase2_base_offsets([10, 25, 12], prompt_length=4) == [6, 21, 8]
    # A primed sequence shorter than the prompt clamps to 0, never negative.
    assert phase2_base_offsets([3], prompt_length=4) == [0]
