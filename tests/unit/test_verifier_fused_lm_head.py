"""The verifier must never materialise the whole [seq_len, vocab] logits block.

At production scale that block is 8.2 GB (16.5k tokens x 248k vocab). It is
allocated once per rollout and 8 times per proof on a GPU that already sits
above the ~88% allocator cliff, which is what stalls the proof plane. Every
consumer only ever reads rows out of it, so the rows can be computed on demand
from the hidden states (84 MB at the same scale) instead.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from reliquary.constants import T_PROTO
from reliquary.validator import verifier

_PROMPT_LENGTH = 8
# Longer than CHALLENGE_K (32) so the logprob-challenge rows are really read.
_COMPLETION_LENGTH = 40
_SEQ_LEN = _PROMPT_LENGTH + _COMPLETION_LENGTH
_VOCAB_SIZE = 16
_EOS_ID = 3
_RANDOMNESS = "aa" * 32


class _RecordingLmHead:
    """An lm_head over one-hot hidden rows: output row i is ``table[pos_i]``.

    Hidden states are the identity matrix, so the argmax of each input row is
    the sequence position it stands for. Recording the rows of every call is
    what lets a test tell per-chunk computation from one full-sequence
    materialisation.
    """

    def __init__(self, table: torch.Tensor) -> None:
        self._table = table
        # A production lm_head is an nn.Linear; the vocab width has to be
        # readable without projecting anything through it.
        self.out_features = int(table.shape[1])
        self.rows_per_call: list[int] = []
        self.positions: set[int] = set()

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        flat = hidden.reshape(-1, hidden.shape[-1])
        self.rows_per_call.append(int(flat.shape[0]))
        self.positions.update(int(p) for p in flat.argmax(dim=-1).tolist())
        return hidden @ self._table


def _build_model(table: torch.Tensor) -> tuple[_RecordingLmHead, SimpleNamespace]:
    """A model whose hidden states are one-hot position indicators."""
    hidden = torch.eye(_SEQ_LEN).unsqueeze(0)  # [1, seq_len, seq_len]
    lm_head = _RecordingLmHead(table)

    def base(input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(last_hidden_state=hidden)

    model = SimpleNamespace(
        base_model_prefix="model",
        model=base,
        lm_head=lm_head,
        config=SimpleNamespace(eos_token_id=_EOS_ID),
        parameters=lambda: iter([torch.zeros(1)]),
    )
    return lm_head, model


def _build_commit() -> dict:
    tokens = [(i % (_VOCAB_SIZE - 1)) + 1 for i in range(_SEQ_LEN)]
    tokens[-1] = _EOS_ID
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0}] * _SEQ_LEN,
        "rollout": {
            "prompt_length": _PROMPT_LENGTH,
            "completion_length": _COMPLETION_LENGTH,
        },
    }


def _run_verification() -> _RecordingLmHead:
    torch.manual_seed(0)
    table = torch.randn(_SEQ_LEN, _VOCAB_SIZE)
    lm_head, model = _build_model(table)
    with patch(
        "reliquary.shared.hf_compat.resolve_hidden_size", return_value=_SEQ_LEN,
    ):
        verifier.verify_commitment_proofs(_build_commit(), model, _RANDOMNESS)
    return lm_head


def test_lm_head_is_never_applied_to_the_whole_sequence_at_once():
    """No single lm_head call may cover every position.

    One call spanning all of them is exactly the [seq_len, vocab] allocation
    this change exists to remove.
    """
    lm_head = _run_verification()

    assert lm_head.rows_per_call, "lm_head was never called"
    assert max(lm_head.rows_per_call) < _SEQ_LEN


_HIDDEN_DIM = 32  # > PROOF_TOPK (16), which the sketch check top-k's over


def _linear_head_model() -> tuple[torch.nn.Linear, torch.Tensor, SimpleNamespace]:
    """A model with a real nn.Linear head and non-degenerate hidden states."""
    torch.manual_seed(7)
    hidden_dim = _HIDDEN_DIM
    hidden = torch.randn(1, _SEQ_LEN, hidden_dim)
    lm_head = torch.nn.Linear(hidden_dim, _VOCAB_SIZE, bias=False)

    def base(input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(last_hidden_state=hidden)

    model = SimpleNamespace(
        base_model_prefix="model",
        model=base,
        lm_head=lm_head,
        config=SimpleNamespace(eos_token_id=_EOS_ID),
        parameters=lambda: iter([torch.zeros(1)]),
    )
    return lm_head, hidden[0], model


# The forced-seed gate turns logits into an inverse-CDF interval and asks
# whether the miner's public uniform falls inside it, so what matters is how
# far the CDF moves, not how far the logits move. Measured at ~6e-8; the
# configured FORCED_SEED_CDF_BOUNDARY_EPSILON is 2e-3, ~33000x larger. Bound
# it well under that but well over the measurement, so this catches a real
# regression without tripping on hardware noise.
_MAX_CDF_SHIFT = 1e-5


def test_lazy_rows_shift_the_cdf_far_below_the_forced_seed_boundary():
    """Rows are NOT bit-identical to the full-sequence projection.

    A GEMM over [seq_len, hidden] and one over [n, hidden] select different
    kernels and accumulation orders, so results differ in the last place. That
    is a property of the hardware, not something this class can remove. What
    it must guarantee is that the drift stays far below the granularity of a
    forced-seed decision — the enforced gate counts *exact* picks
    (FORCED_SEED_CDF_ENFORCE is off, so the boundary tolerance does not shield
    it), and drift is what flips one.
    """
    lm_head, hidden, _ = _linear_head_model()
    with torch.no_grad():
        materialised = lm_head(hidden)
    lazy = verifier._LazyLogitRows(hidden, lm_head, _VOCAB_SIZE)
    positions = torch.tensor([0, 5, _SEQ_LEN - 1], dtype=torch.long)

    for got, want in (
        (lazy[3], materialised[3]),
        (lazy[positions], materialised[positions]),
        (lazy.index_select(0, positions), materialised.index_select(0, positions)),
    ):
        got_cdf = torch.softmax(got.float() / T_PROTO, dim=-1).cumsum(dim=-1)
        want_cdf = torch.softmax(want.float() / T_PROTO, dim=-1).cumsum(dim=-1)
        assert float((got_cdf - want_cdf).abs().max()) < _MAX_CDF_SHIFT


def test_lazy_and_materialised_paths_agree_on_every_gating_value():
    """The values the accept/reject path reads must not move.

    Rows drift by an ulp, so this asserts the decisions built on them —
    the forced-seed counters the 0.75/0.80 floors consume, and p_stop —
    rather than the raw floats.
    """
    _, _, model = _linear_head_model()
    commit = _build_commit()
    u_values = [0.1 + 0.02 * i for i in range(_COMPLETION_LENGTH)]

    def run() -> object:
        return verifier.verify_commitment_proofs(
            commit, model, _RANDOMNESS, seed_u_values=u_values,
        )

    with patch(
        "reliquary.shared.hf_compat.resolve_hidden_size", return_value=_HIDDEN_DIM,
    ):
        lazy_result = run()
        # vocab_size unresolvable => verify_commitment_proofs keeps the old
        # materialised block, which is the reference this must reproduce.
        with patch.object(verifier, "_lm_head_vocab_size", return_value=None):
            materialised_result = run()

    for field in (
        "seed_n_stochastic",
        "seed_n_match",
        "seed_n_positions",
        "seed_n_boundary_match",
        "seed_n_hard_mismatch",
        "seed_n_deterministic_hard_mismatch",
        "terminal_pick_ok",
        "all_passed",
        "passed",
        "checked",
        "challenge_lp_indices",
        "completion_argmax_ids",
    ):
        assert getattr(lazy_result, field) == getattr(materialised_result, field), (
            f"{field} moved between the lazy and materialised paths"
        )


def test_logits_are_not_computed_for_prompt_only_positions():
    """Rows before ``prompt_length - 1`` predict prompt tokens.

    No consumer reads them, so computing them is pure waste; the first row any
    consumer needs is the one predicting the first completion token.
    """
    lm_head = _run_verification()

    assert lm_head.positions, "lm_head was never called"
    assert min(lm_head.positions) >= _PROMPT_LENGTH - 1
