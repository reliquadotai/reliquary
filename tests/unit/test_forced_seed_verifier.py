from unittest.mock import MagicMock, patch

import torch
from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.validator import verifier
from reliquary.environment import forced_sampling as fs


def test_gpu_completion_seed_counts_perfect_for_forced_tokens():
    # 3 completion positions, flat distributions -> all stochastic
    logits = torch.tensor([[0.2, 0.1, 0.0], [0.1, 0.2, 0.15], [0.0, 0.1, 0.2]])
    u = [fs.u_at("r", "h", 0, "c", 0, t) for t in range(3)]
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i]) for i in range(3)]
    n_stoch, n_match = verifier._gpu_seed_consistency(logits, tokens, u)
    assert n_stoch == 3 and n_match == 3


def test_proof_result_has_seed_fields():
    p = verifier.ProofResult(all_passed=True, passed=0, checked=0, has_sparse_outputs=False)
    assert p.seed_n_stochastic == 0 and p.seed_n_match == 0


# ══════════════════════════════════════════════════════════════════════
# Integration coverage for the seed_u_values WIRING inside
# verify_commitment_proofs itself: the offset-skip filter
# (`t - prompt_length < len(seed_u_values)`), the `logits_gpu[pos_tensor]`
# gather, and the `seed_tokens`/`seed_u` construction. The two unit tests
# above only exercise the extracted `_gpu_seed_consistency` helper and the
# ProofResult defaults -- neither drives seed_u_values through the real
# function, so an off-by-one in that wiring would go undetected.
# ══════════════════════════════════════════════════════════════════════

_HIDDEN_DIM = 128
_VOCAB_SIZE = 4
_RANDOMNESS = "aa" * 32
_PROMPT_LENGTH = 2
_COMPLETION_LENGTH = 4
_SEQ_LEN = _PROMPT_LENGTH + _COMPLETION_LENGTH


def _make_mock_model():
    param = torch.zeros(1)
    model = MagicMock()
    model.parameters.return_value = iter([param])
    return model


def _seed_wiring_logits_rows():
    """logits_gpu rows, indexed so row[t - 1] predicts tokens[t].

    Completion positions are t = 2, 3, 4, 5 (prompt_length=2), i.e. rows
    1, 2, 3, 4. Rows 1, 2, 4 are FLAT (uniform -> stochastic under the
    FORCED_SEED_STOCHASTIC_MAXPROB=0.99 threshold); row 3 is PEAKED
    (argmax prob >= 0.99 -> deterministic, excluded from the stochastic
    count regardless of the forced pick). Rows 0 and 5 are never read.
    """
    flat = torch.tensor([1.0, 1.0, 1.0, 1.0])
    peaked = torch.tensor([10.0, -10.0, -10.0, -10.0])
    return [flat, flat, flat, peaked, flat, flat]


def _seed_wiring_u_values():
    return [
        fs.u_at("seedwire", "hk", 7, "ckpt", 0, offset)
        for offset in range(_COMPLETION_LENGTH)
    ]


def _build_seed_wiring_commit(logits_gpu, u_values, corrupt_offset=None):
    """Build a commit whose completion tokens are the forced-u inverse-CDF
    picks against `logits_gpu`, using the same warp/pick primitives
    `_gpu_seed_consistency` reuses -- so an honest rollout scores
    seed_n_match == seed_n_stochastic. `corrupt_offset`, if given,
    overwrites that one completion token with a token guaranteed to
    differ from the forced pick, to exercise the mismatch path.
    """
    tokens = [100, 101]
    for t in range(_PROMPT_LENGTH, _SEQ_LEN):
        offset = t - _PROMPT_LENGTH
        row = logits_gpu[t - 1]
        probs = fs.warp(row, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
        picked = fs.pick(probs, u_values[offset])
        if corrupt_offset is not None and offset == corrupt_offset:
            picked = (picked + 1) % _VOCAB_SIZE
        tokens.append(picked)
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0}] * _SEQ_LEN,
        "rollout": {
            "prompt_length": _PROMPT_LENGTH,
            "completion_length": _COMPLETION_LENGTH,
        },
    }


@patch("reliquary.shared.forward.forward_single_layer")
@patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=_HIDDEN_DIM)
def test_verify_commitment_proofs_honest_seed_matches_all_stochastic(_rhs, mock_fwd):
    """Full-length seed_u_values, honest tokens -> seed_n_match ==
    seed_n_stochastic, and the peaked (deterministic) position is excluded."""
    logits_gpu = torch.stack(_seed_wiring_logits_rows())
    u_values = _seed_wiring_u_values()
    commit = _build_seed_wiring_commit(logits_gpu, u_values)
    mock_fwd.return_value = (
        torch.randn(1, _SEQ_LEN, _HIDDEN_DIM),
        logits_gpu.unsqueeze(0),
    )

    result = verifier.verify_commitment_proofs(
        commit, _make_mock_model(), _RANDOMNESS, seed_u_values=u_values,
    )

    # t=2,3,5 flat (stochastic + matched); t=4 peaked (excluded).
    assert result.seed_n_stochastic == 3
    assert result.seed_n_match == 3


@patch("reliquary.shared.forward.forward_single_layer")
@patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=_HIDDEN_DIM)
def test_verify_commitment_proofs_mismatch_lowers_match_not_stochastic(_rhs, mock_fwd):
    """Corrupting one stochastic completion token drops seed_n_match by
    one but leaves seed_n_stochastic unchanged -- proves the gather picks
    the right logits row for the right token, not just a plausible count."""
    logits_gpu = torch.stack(_seed_wiring_logits_rows())
    u_values = _seed_wiring_u_values()
    commit = _build_seed_wiring_commit(logits_gpu, u_values, corrupt_offset=1)  # t=3
    mock_fwd.return_value = (
        torch.randn(1, _SEQ_LEN, _HIDDEN_DIM),
        logits_gpu.unsqueeze(0),
    )

    result = verifier.verify_commitment_proofs(
        commit, _make_mock_model(), _RANDOMNESS, seed_u_values=u_values,
    )

    assert result.seed_n_stochastic == 3
    assert result.seed_n_match == 2


@patch("reliquary.shared.forward.forward_single_layer")
@patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=_HIDDEN_DIM)
def test_verify_commitment_proofs_excludes_bft_force_span_positions(_rhs, mock_fwd):
    """A BFT-injected force_span token is validator-accepted but not
    policy-sampled, so it must be excluded from the seed-consistency check
    even though its distribution is stochastic. force_span=(3,4) removes
    completion offset 1 (t=3). We corrupt that same token so that WITHOUT
    exclusion it would both count as stochastic AND mismatch (n_stoch=3,
    n_match=2); with exclusion it neither counts nor mismatches
    (n_stoch=2, n_match=2)."""
    logits_gpu = torch.stack(_seed_wiring_logits_rows())
    u_values = _seed_wiring_u_values()
    commit = _build_seed_wiring_commit(logits_gpu, u_values, corrupt_offset=1)  # t=3
    commit["rollout"]["forced"] = True
    commit["rollout"]["force_span"] = [3, 4]  # absolute token positions
    mock_fwd.return_value = (
        torch.randn(1, _SEQ_LEN, _HIDDEN_DIM),
        logits_gpu.unsqueeze(0),
    )

    result = verifier.verify_commitment_proofs(
        commit, _make_mock_model(), _RANDOMNESS, seed_u_values=u_values,
    )

    assert result.seed_n_stochastic == 2   # t=2, t=5 (t=3 excluded as force-span)
    assert result.seed_n_match == 2        # corrupted t=3 does not drag match down


@patch("reliquary.shared.forward.forward_single_layer")
@patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=_HIDDEN_DIM)
def test_verify_commitment_proofs_short_seed_u_values_skips_tail_positions(_rhs, mock_fwd):
    """seed_u_values shorter than completion_length must SKIP the
    uncovered tail positions, not score them. With only 2 values supplied,
    t=4 (peaked) and t=5 (flat/stochastic) must both be excluded -- if the
    offset-skip filter had an off-by-one and let t=5 slip through,
    seed_n_stochastic would read 3 instead of 2."""
    logits_gpu = torch.stack(_seed_wiring_logits_rows())
    u_values = _seed_wiring_u_values()
    commit = _build_seed_wiring_commit(logits_gpu, u_values)
    mock_fwd.return_value = (
        torch.randn(1, _SEQ_LEN, _HIDDEN_DIM),
        logits_gpu.unsqueeze(0),
    )

    result = verifier.verify_commitment_proofs(
        commit, _make_mock_model(), _RANDOMNESS, seed_u_values=u_values[:2],
    )

    assert result.seed_n_stochastic == 2
    assert result.seed_n_match == 2
