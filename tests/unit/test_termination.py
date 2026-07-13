"""verify_termination — EOS and protocol-cap termination checks.

The miner must end every rollout with the tokenizer's EOS token, AND the
model must have assigned probability >= MIN_EOS_PROBABILITY to EOS at the
position that produced it. A protocol-cap fallback exists for max-length
runaways; the batcher counts cap hits without natural EOS as truncations.

After the keep-logits-on-GPU refactor, ``verify_termination`` reads a
precomputed ``p_stop`` carried on ``ProofResult`` rather than slicing a
CPU logits tensor itself. The fake-logits helper below computes the
same value test-side so each test pins the contract without having to
recompute the softmax inside the verifier.
"""

import pytest
import torch
from types import SimpleNamespace

from reliquary.constants import MIN_EOS_PROBABILITY
from reliquary.validator.verifier import (
    ProofResult,
    _gpu_terminal_forced_pick,
    _gpu_terminal_forced_pick_diagnostics,
    has_eos_padding,
    is_cap_truncation,
    is_natural_bft_cap_candidate,
    verify_termination,
)


class _FakeTokenizer:
    eos_token_id = 99


class _FakeBFTTokenizer(_FakeTokenizer):
    think_close_id = 77

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.think_close_id if token == "</think>" else -1


def _commit(tokens: list[int]) -> dict:
    """Minimal commit dict — verify_termination only reads ``tokens``."""
    return {"tokens": tokens}


def _make_logits(seq_len: int, vocab_size: int = 100, eos_logit: float = 5.0):
    """Logits where EOS token (id 99) has high probability at every position."""
    logits = torch.zeros(seq_len, vocab_size)
    logits[:, 99] = eos_logit
    return logits


def _proof_from_logits(logits: torch.Tensor, eos_token_id: int) -> ProofResult:
    """Build a ProofResult whose ``p_stop`` mirrors what
    ``verify_commitment_proofs`` would have precomputed on GPU from the
    given fake logits — softmax of the second-to-last row, mass at eos."""
    if logits.size(0) < 2:
        p_stop = None
    else:
        probs = torch.softmax(logits[-2].float(), dim=-1)
        p_stop = float(probs[eos_token_id].item())
    return ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=p_stop,
    )


def test_accepts_when_ends_with_eos_at_high_prob():
    tokens = [10, 20, 30, 99]  # last token = EOS
    logits = _make_logits(seq_len=4, eos_logit=5.0)  # p(EOS) ~ 0.97
    proof = _proof_from_logits(logits, eos_token_id=99)
    assert verify_termination(_commit(tokens), _FakeTokenizer(), proof) is True


def test_rejects_when_does_not_end_with_eos():
    tokens = [10, 20, 30, 40]  # last token != EOS
    logits = _make_logits(seq_len=4)
    proof = _proof_from_logits(logits, eos_token_id=99)
    assert verify_termination(_commit(tokens), _FakeTokenizer(), proof) is False


def test_rejects_when_eos_prob_below_threshold():
    tokens = [10, 20, 30, 99]
    logits = torch.zeros(4, 100)
    logits[:, 99] = -10.0  # p(EOS) ~ 4.5e-5, well below MIN_EOS_PROBABILITY
    proof = _proof_from_logits(logits, eos_token_id=99)
    assert proof.p_stop < MIN_EOS_PROBABILITY
    assert verify_termination(_commit(tokens), _FakeTokenizer(), proof) is False


def test_rejects_when_tokenizer_has_no_eos():
    tokens = [10, 20, 30, 99]
    logits = _make_logits(seq_len=4)
    proof = _proof_from_logits(logits, eos_token_id=99)

    class NoEosTokenizer:
        eos_token_id = None

    assert verify_termination(_commit(tokens), NoEosTokenizer(), proof) is False


def test_uses_p_stop_at_second_to_last_position():
    """p_stop is the EOS probability at logits[seq_len - 2] (the position
    that PRODUCED tokens[-1]). The helper above mirrors that contract;
    here we just confirm a strong probability there passes."""
    tokens = [10, 20, 30, 99]
    logits = torch.zeros(4, 100)
    logits[:, 99] = -10.0
    logits[-2, 99] = 5.0  # p(EOS|context-at-pos-2) ~ 0.97
    proof = _proof_from_logits(logits, eos_token_id=99)
    assert verify_termination(_commit(tokens), _FakeTokenizer(), proof) is True


def test_forced_seed_pick_rescues_low_p_stop_termination():
    """A rollout whose final EOS IS the protocol's forced inverse-CDF pick is a
    legal draw, however improbable: the miner cannot choose where the public u
    selects EOS. Honest mid-reasoning truncations land here (the model drew EOS
    from the nucleus by chance) and must not be rejected as forgeries."""
    tokens = [10, 20, 30, 99]
    logits = torch.zeros(4, 100)
    logits[:, 99] = -10.0  # p(EOS) ~ 4.5e-5 — far below the floor
    proof = _proof_from_logits(logits, eos_token_id=99)
    assert proof.p_stop < MIN_EOS_PROBABILITY
    proof.terminal_pick_ok = True

    assert verify_termination(_commit(tokens), _FakeTokenizer(), proof) is True


def test_forced_seed_pick_mismatch_still_rejects_low_p_stop():
    """The escape is an OR, not a bypass: when the final EOS is NOT the forced
    pick, a sub-floor p_stop stays a rejection (this is the forged-truncation
    case — the attacker injected EOS where u did not select it)."""
    tokens = [10, 20, 30, 99]
    logits = torch.zeros(4, 100)
    logits[:, 99] = -10.0
    proof = _proof_from_logits(logits, eos_token_id=99)
    proof.terminal_pick_ok = False

    assert verify_termination(_commit(tokens), _FakeTokenizer(), proof) is False


def test_forced_seed_pick_cannot_rescue_a_non_eos_final_token():
    """The escape never waives the structural requirement that the rollout
    actually ends on a stop token."""
    tokens = [10, 20, 30, 40]  # final token is not EOS
    proof = ProofResult(all_passed=True, passed=1, checked=1, p_stop=0.99)
    proof.terminal_pick_ok = True

    assert verify_termination(_commit(tokens), _FakeTokenizer(), proof) is False


def test_terminal_forced_pick_must_equal_the_submitted_eos_token():
    """With multiple valid EOS ids, drawing EOS-A cannot authenticate EOS-B."""
    logits = torch.full((3, 8), -20.0)
    logits[1, 6] = 20.0
    tokens = [1, 2, 7]

    assert _gpu_terminal_forced_pick(
        logits,
        tokens,
        prompt_length=1,
        seq_len=len(tokens),
        eos_set={6, 7},
        seed_u_values=[0.5, 0.5],
        force_span=(0, 0),
    ) is False


def test_terminal_forced_pick_reports_near_boundary_miss_without_relaxing():
    logits = torch.zeros(2, 2)
    exact, miss = _gpu_terminal_forced_pick_diagnostics(
        logits,
        tokens=[1, 0],
        prompt_length=1,
        seq_len=2,
        eos_set={0},
        seed_u_values=[0.5005],
        force_span=(0, 0),
    )

    assert exact is False
    assert miss == pytest.approx(0.0005, abs=1e-6)


def test_absent_forced_seed_pick_preserves_current_behaviour():
    """Pre-forced-seed rollouts (no u-stream) carry terminal_pick_ok=None and
    must keep falling back to the p_stop floor alone."""
    tokens = [10, 20, 30, 99]
    logits = torch.zeros(4, 100)
    logits[:, 99] = -10.0
    proof = _proof_from_logits(logits, eos_token_id=99)
    assert proof.terminal_pick_ok is None

    assert verify_termination(_commit(tokens), _FakeTokenizer(), proof) is False


def test_forced_seed_pick_rescues_cap_hit_from_truncation_count():
    """A cap-length rollout that ended on a legally-drawn EOS is a natural stop,
    not a truncation — otherwise it burns the per-submission truncation budget."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    tokens = [1] * (MAX_NEW_TOKENS_PROTOCOL_CAP - 1) + [99]
    commit = {
        "tokens": tokens,
        "rollout": {"prompt_length": 0, "completion_length": len(tokens)},
    }
    proof = ProofResult(all_passed=True, passed=1, checked=1, p_stop=1e-9)
    proof.terminal_pick_ok = True

    assert is_cap_truncation(commit, _FakeTokenizer(), proof) is False


def test_accepts_tokenizer_eos_even_when_model_generation_eos_differs():
    """Qwen3.5 advertises <|endoftext|> on generation config while the chat
    tokenizer ends turns with <|im_end|>; the validator must accept both."""
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=98),
        config=SimpleNamespace(text_config=SimpleNamespace(eos_token_id=98)),
    )
    tokenizer = SimpleNamespace(eos_token_id=99)
    tokens = [1, 2, 99]
    proof = ProofResult(all_passed=True, passed=1, checked=1, p_stop=0.99)

    assert verify_termination(_commit(tokens), tokenizer, proof, model=model) is True


# ---------------------------------------------------------------------
# Path 1 — max-length termination based on total context length.
# Honest miners running under a `max_model_len` ceiling (e.g. vLLM) cap
# at prompt_length + completion_length = max_model_len, so completion_length
# alone never reaches MAX_NEW_TOKENS_PROTOCOL_CAP. Path 1 must check the
# total to accept these.
# ---------------------------------------------------------------------


def _commit_with_lengths(tokens: list[int], prompt_length: int, completion_length: int) -> dict:
    return {
        "tokens": tokens,
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
        },
    }


def test_path1_accepts_max_model_len_bound_termination():
    """Miner with max_model_len=cap and prompt_length>0 hits prompt+compl=cap.
    Last token is not EOS, p_stop is ~0 — Path 2 fails — but Path 1 must
    accept on total-length grounds, regardless of what's in the proof."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt_length = 33
    completion_length = MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length
    seq_len = prompt_length + completion_length
    tokens = [42] * seq_len
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=1e-10,  # Path 2 would fail; Path 1 must short-circuit before
    )
    assert verify_termination(
        _commit_with_lengths(tokens, prompt_length, completion_length),
        _FakeTokenizer(), proof,
    ) is True


def test_cap_hit_without_natural_eos_counts_as_truncation():
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt_length = 33
    completion_length = MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length
    seq_len = prompt_length + completion_length
    tokens = [42] * seq_len
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=1e-10,
    )
    commit = _commit_with_lengths(tokens, prompt_length, completion_length)

    assert verify_termination(commit, _FakeTokenizer(), proof) is True
    assert is_cap_truncation(commit, _FakeTokenizer(), proof) is True


def test_forced_bft_cap_passes_below_global_cap_without_truncation():
    from reliquary.constants import (
        BFT_ANSWER_BUDGET,
        BFT_THINKING_BUDGET,
        MAX_NEW_TOKENS_PROTOCOL_CAP,
    )

    prompt_length = 33
    force_len = 8
    completion_length = BFT_THINKING_BUDGET + force_len + BFT_ANSWER_BUDGET
    assert prompt_length + completion_length < MAX_NEW_TOKENS_PROTOCOL_CAP
    tokens = [42] * (prompt_length + completion_length)
    commit = _commit_with_lengths(tokens, prompt_length, completion_length)
    commit["rollout"]["forced"] = True
    commit["rollout"]["force_span"] = [
        prompt_length + BFT_THINKING_BUDGET,
        prompt_length + BFT_THINKING_BUDGET + force_len,
    ]
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=1e-10,
    )

    assert verify_termination(
        commit,
        _FakeTokenizer(),
        proof,
        env_name="openmathinstruct",
    ) is True
    assert is_cap_truncation(
        commit,
        _FakeTokenizer(),
        proof,
        env_name="openmathinstruct",
    ) is False

    assert verify_termination(
        commit,
        _FakeTokenizer(),
        proof,
        env_name="opencodeinstruct",
    ) is False


def test_forced_bft_cap_requires_exact_completion_length():
    from reliquary.constants import BFT_ANSWER_BUDGET, BFT_THINKING_BUDGET

    prompt_length = 4
    force_len = 3
    completion_length = BFT_THINKING_BUDGET + force_len + BFT_ANSWER_BUDGET + 1
    tokens = [42] * (prompt_length + completion_length)
    commit = _commit_with_lengths(tokens, prompt_length, completion_length)
    commit["rollout"]["forced"] = True
    commit["rollout"]["force_span"] = [
        prompt_length + BFT_THINKING_BUDGET,
        prompt_length + BFT_THINKING_BUDGET + force_len,
    ]
    proof = ProofResult(all_passed=True, passed=1, checked=1, p_stop=1e-10)

    assert verify_termination(commit, _FakeTokenizer(), proof) is False


def test_natural_bft_cap_requires_math_shape_and_validator_pick_proof():
    from reliquary.constants import BFT_ANSWER_BUDGET, BFT_THINKING_BUDGET

    tokenizer = _FakeBFTTokenizer()
    prompt_length = 4
    completion_length = BFT_THINKING_BUDGET + BFT_ANSWER_BUDGET
    completion = [42] * completion_length
    completion[100] = tokenizer.think_close_id
    commit = _commit_with_lengths(
        [11] * prompt_length + completion,
        prompt_length,
        completion_length,
    )
    proof = ProofResult(
        all_passed=True,
        passed=1,
        checked=1,
        p_stop=1e-10,
        natural_close_pick_ok=True,
    )

    assert is_natural_bft_cap_candidate(
        commit, tokenizer, env_name="openmathinstruct"
    ) is True
    assert verify_termination(
        commit,
        tokenizer,
        proof,
        env_name="openmathinstruct",
    ) is True
    assert is_cap_truncation(
        commit,
        tokenizer,
        proof,
        env_name="openmathinstruct",
    ) is False

    proof.natural_close_pick_ok = False
    assert verify_termination(
        commit,
        tokenizer,
        proof,
        env_name="openmathinstruct",
    ) is False
    proof.natural_close_pick_ok = True
    assert verify_termination(
        commit,
        tokenizer,
        proof,
        env_name="opencodeinstruct",
    ) is False


def test_cap_hit_with_natural_eos_is_not_truncation():
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt_length = 33
    completion_length = MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length
    seq_len = prompt_length + completion_length
    tokens = [42] * (seq_len - 1) + [_FakeTokenizer.eos_token_id]
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=0.99,
    )
    commit = _commit_with_lengths(tokens, prompt_length, completion_length)

    assert verify_termination(commit, _FakeTokenizer(), proof) is True
    assert is_cap_truncation(commit, _FakeTokenizer(), proof) is False


def test_single_terminal_eos_is_not_padding():
    tokens = [10, 11, 12, 99]
    commit = _commit_with_lengths(tokens, prompt_length=1, completion_length=3)

    assert has_eos_padding(commit, _FakeTokenizer()) is False


def test_repeated_eos_tokens_are_padding():
    tokens = [10, 11, 99, 99]
    commit = _commit_with_lengths(tokens, prompt_length=1, completion_length=3)

    assert has_eos_padding(commit, _FakeTokenizer()) is True


def test_tokens_after_eos_are_padding():
    tokens = [10, 11, 99, 12]
    commit = _commit_with_lengths(tokens, prompt_length=1, completion_length=3)

    assert has_eos_padding(commit, _FakeTokenizer()) is True


def test_path1_accepts_when_completion_alone_meets_cap():
    """Backwards-compat: a miner running pure max_new_tokens=cap (no
    max_model_len constraint) still passes Path 1."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt_length = 0
    completion_length = MAX_NEW_TOKENS_PROTOCOL_CAP
    seq_len = completion_length
    tokens = [42] * seq_len
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=1e-10,
    )
    assert verify_termination(
        _commit_with_lengths(tokens, prompt_length, completion_length),
        _FakeTokenizer(), proof,
    ) is True


def test_path1_rejects_short_truncation_below_cap():
    """A miner who truncates well below the cap and forges a non-EOS last
    token must be rejected — this is the gaming-safe property of Path 1."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt_length = 33
    completion_length = 100  # total = 133, way below the cap
    seq_len = prompt_length + completion_length
    tokens = [42] * seq_len  # last token NOT EOS
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=1e-10,
    )
    assert prompt_length + completion_length < MAX_NEW_TOKENS_PROTOCOL_CAP
    assert verify_termination(
        _commit_with_lengths(tokens, prompt_length, completion_length),
        _FakeTokenizer(), proof,
    ) is False
