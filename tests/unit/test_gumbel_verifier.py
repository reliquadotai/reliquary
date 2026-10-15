"""The validator checks the profile's forced draw: Gumbel-max picks when ``sampling.draw`` says so,
and an fp32 LM head in the proof forward when ``sampling.lm_head_fp32`` does."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from reliquary.environment import forced_sampling as fs
from reliquary.validator import verifier

_HIDDEN_DIM = 64
_VOCAB = 32
_EOS = 3
_PROMPT = 2
_RANDOMNESS = "aa" * 32
_GUMBEL = verifier.DrawSettings(draw=fs.DRAW_GUMBEL_MAX, t=1.0, top_k=0, top_p=1.0,
                                stochastic_threshold=0.99, boundary_epsilon=0.0)
_CDF = verifier.DrawSettings(draw=fs.DRAW_INVERSE_CDF, t=1.0, top_k=0, top_p=1.0,
                             stochastic_threshold=0.99, boundary_epsilon=0.002)


@pytest.fixture
def gumbel(monkeypatch):
    monkeypatch.setattr(verifier, "draw_settings", lambda: _GUMBEL)


def _logits(n: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, _VOCAB, generator=gen)


def _u(n: int, stream: str = "s") -> list[float]:
    return [fs.u_at(stream, 7, "ckpt", 0, j) for j in range(n)]


def _commit(logits: torch.Tensor, us: list[float], *, draw: str, end_on_eos: bool = False):
    """tokens[t] is the forced pick of logits[t - 1] for u[t - prompt]."""
    n = logits.shape[0] - _PROMPT + 1
    tokens = [100, 101]
    rows = logits[_PROMPT - 1:_PROMPT - 1 + n]
    tokens += fs.pick_rows(rows, us[:n], draw=draw).tolist()
    if end_on_eos:
        tokens[-1] = _EOS
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0}] * len(tokens),
        "rollout": {"prompt_length": _PROMPT, "completion_length": n},
    }


def _model():
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    return model


def _verify(commit, logits, us):
    seq = len(commit["tokens"])
    with patch("reliquary.shared.forward.forward_single_layer",
               return_value=(torch.randn(1, seq, _HIDDEN_DIM), logits[:seq].unsqueeze(0))), \
         patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=_HIDDEN_DIM), \
         patch.object(verifier, "_eos_set_from_model", return_value={_EOS}):
        return verifier.verify_commitment_proofs(commit, _model(), _RANDOMNESS, seed_u_values=us)


def test_draw_settings_follow_the_active_profile():
    from reliquary import constants

    settings = verifier.draw_settings()
    assert settings.draw == constants.FORCED_DRAW
    assert (settings.t, settings.top_k, settings.top_p) == (
        constants.T_PROTO, constants.TOP_K_PROTO, constants.TOP_P_PROTO)
    assert settings.stochastic_threshold == constants.FORCED_SEED_STOCHASTIC_MAXPROB
    expected = (constants.FORCED_SEED_GUMBEL_BOUNDARY_EPSILON
                if constants.FORCED_DRAW == fs.DRAW_GUMBEL_MAX
                else constants.FORCED_SEED_CDF_BOUNDARY_EPSILON)
    assert settings.boundary_epsilon == expected


def test_honest_gumbel_rollout_matches_every_stochastic_position(gumbel):
    logits = _logits(40)
    us = _u(40)
    result = _verify(_commit(logits, us, draw=fs.DRAW_GUMBEL_MAX), logits, us)
    assert result.seed_n_stochastic > 30
    assert result.seed_n_match == result.seed_n_stochastic
    assert result.seed_n_hard_mismatch == 0 and result.seed_max_cdf_miss == 0.0


def test_an_inverse_cdf_rollout_fails_a_gumbel_validator(gumbel):
    logits = _logits(40)
    us = _u(40)
    result = _verify(_commit(logits, us, draw=fs.DRAW_INVERSE_CDF), logits, us)
    assert result.seed_n_match < 0.6 * result.seed_n_stochastic
    assert 0.0 < result.seed_max_cdf_miss < 1.0


def test_gumbel_terminal_pick_is_checked_with_the_gumbel_draw(gumbel):
    logits = _logits(40, seed=1)
    us = _u(40)
    # make EOS the Gumbel pick at the last position
    row = logits.shape[0] - 1  # predicts the final token, completion offset row - (prompt - 1)
    scores = fs._gumbel_scores(logits[row:row + 1], [us[row - (_PROMPT - 1)]])[0]
    logits[row, _EOS] += float(scores.max() - scores[_EOS]) + 1.0
    commit = _commit(logits, us, draw=fs.DRAW_GUMBEL_MAX)
    assert commit["tokens"][-1] == _EOS
    assert _verify(commit, logits, us).terminal_pick_ok is True

    other = _logits(40, seed=2)
    other[:, _EOS] -= 5.0
    forged = _commit(other, us, draw=fs.DRAW_GUMBEL_MAX, end_on_eos=True)
    result = _verify(forged, other, us)
    assert result.terminal_pick_ok is False
    assert 0.0 < result.terminal_pick_cdf_miss < 1.0


def test_episode_stop_picks_use_the_gumbel_draw(gumbel):
    logits = _logits(12, seed=3)
    us = _u(12)
    positions = list(range(2, 12))
    tokens = [100, 101] + fs.pick_rows(logits[1:11], us[:10], draw=fs.DRAW_GUMBEL_MAX).tolist()
    stop = tokens[6]
    ok, bad_turn, miss = verifier._episode_stop_picks(logits, tokens, [(2, 7), (8, 12)], positions,
                                                      {stop}, us)
    assert ok is True and bad_turn is None
    tokens[6] = (stop + 1) % _VOCAB  # not the pick: the pick there is `stop`
    ok, bad_turn, miss = verifier._episode_stop_picks(logits, tokens, [(2, 7), (8, 12)], positions,
                                                      {tokens[6]}, us)
    assert ok is False and bad_turn == 0 and 0.0 < miss < 1.0


def test_gumbel_boundary_epsilon_comes_from_the_settings(monkeypatch):
    logits = _logits(1, seed=4)
    us = _u(1)
    scores = fs._gumbel_scores(logits, us)[0]
    runner_up = int(scores.argsort(descending=True)[1])
    gap_miss = float(-torch.expm1(-(scores.max() - scores[runner_up])))
    for eps, hard in ((gap_miss * 1.01, 0), (gap_miss * 0.99, 1)):
        monkeypatch.setattr(verifier, "draw_settings",
                            lambda eps=eps: verifier.DrawSettings(**{**_GUMBEL._asdict(),
                                                                     "boundary_epsilon": eps}))
        d = verifier._gpu_seed_consistency_diagnostics(logits, [runner_up], us)
        assert d.n_hard_mismatch == hard


def test_inverse_cdf_settings_keep_the_old_verdicts(monkeypatch):
    monkeypatch.setattr(verifier, "draw_settings", lambda: _CDF)
    logits = _logits(40)
    us = _u(40)
    result = _verify(_commit(logits, us, draw=fs.DRAW_INVERSE_CDF), logits, us)
    assert result.seed_n_match == result.seed_n_stochastic


# ---------------------------------------------------------------- fp32 LM head

class _Head(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.out_features = int(weight.shape[0])

    def forward(self, rows):
        return rows @ self.weight.T


def test_fp32_head_projects_bf16_rows_in_fp32():
    gen = torch.Generator().manual_seed(5)
    weight = torch.randn(_VOCAB, _HIDDEN_DIM, generator=gen).to(torch.bfloat16)
    hidden = torch.randn(6, _HIDDEN_DIM, generator=gen).to(torch.bfloat16)
    head = _Head(weight)
    rows = verifier._LazyLogitRows(hidden, head, _VOCAB, fp32_head=True)
    got = rows[1:4]
    assert got.dtype == torch.float32
    assert torch.equal(got, hidden[1:4].float() @ weight.float().T)
    assert torch.equal(rows.index_select(0, torch.tensor([5, 0])),
                       hidden[[5, 0]].float() @ weight.float().T)
    default = verifier._LazyLogitRows(hidden, head, _VOCAB)
    assert default[1:4].dtype == torch.bfloat16


def test_proof_forward_uses_the_fp32_head_when_the_profile_does(monkeypatch):
    seen = {}
    real = verifier._LazyLogitRows

    def spy(hidden, lm_head, vocab, *, fp32_head=False):
        seen["fp32_head"] = fp32_head
        return real(hidden, lm_head, vocab, fp32_head=fp32_head)

    monkeypatch.setattr(verifier, "_LazyLogitRows", spy)
    monkeypatch.setattr(verifier, "lm_head_fp32", lambda: True)
    gen = torch.Generator().manual_seed(6)
    head = _Head(torch.randn(_VOCAB, _HIDDEN_DIM, generator=gen))
    model = SimpleNamespace(lm_head=head, parameters=lambda: iter([torch.zeros(1)]))
    commit = {"tokens": [1, 2, 3, 4], "commitments": [{"sketch": 0}] * 4,
              "rollout": {"prompt_length": 2, "completion_length": 2}}
    with patch("reliquary.shared.forward.forward_single_layer",
               return_value=(torch.randn(1, 4, _HIDDEN_DIM), None)), \
         patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=_HIDDEN_DIM):
        verifier.verify_commitment_proofs(commit, model, _RANDOMNESS)
    assert seen["fp32_head"] is True


def test_fp32_head_refuses_a_materialised_logits_block(monkeypatch):
    monkeypatch.setattr(verifier, "lm_head_fp32", lambda: True)
    model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
    commit = {"tokens": [1, 2, 3, 4], "commitments": [{"sketch": 0}] * 4,
              "rollout": {"prompt_length": 2, "completion_length": 2}}
    with patch("reliquary.shared.forward.forward_single_layer",
               return_value=(torch.randn(1, 4, _HIDDEN_DIM), torch.randn(1, 4, _VOCAB))), \
         patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=_HIDDEN_DIM), \
         pytest.raises(RuntimeError, match="fp32"):
        verifier.verify_commitment_proofs(commit, model, _RANDOMNESS)
