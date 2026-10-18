"""The out-of-zone audit's proof forward (``rl_audit``): the verifier's forward and Gumbel check over a submitted
group, the per-rollout floor and pooled group rule, and the failure classes."""
from __future__ import annotations

import pytest
import torch

from reliquary.environment import forced_sampling as fs
from reliquary.protocol.rl_draw import lease_uniforms
from reliquary.services.exploration import AUDIT_FAILURE_DETERMINISTIC, AUDIT_FAILURE_STATISTICAL
from reliquary.validator import rl_audit, verifier
from reliquary.validator.rl_buffer import SubmittedGroup
from reliquary.validator.rl_submit import AuditResult

V = 32
NONCE, OTHER_NONCE = "ab" * 32, "cd" * 32
PROMPT = (5, 20)
LENGTH = 48


def _transitions() -> torch.Tensor:
    """Next-token logits per current token: tokens < 16 are flat (stochastic positions), tokens >= 16 are peaked
    on one successor below 16 (max p > 0.99: near-deterministic positions)."""
    gen = torch.Generator().manual_seed(0)
    table = torch.randn(V, V, generator=gen)
    for a in range(16, V):
        table[a] = 0.0
        table[a, (a * 7) % 16] = 10.0
    return table


class _Base(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(V, V)
        self.embed.weight.data = torch.eye(V)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return (self.embed(input_ids),)


class TinyLM(torch.nn.Module):
    """A one-token-context LM: hidden = one-hot(token), logits = the transition table's row."""
    base_model_prefix = "model"

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.model = _Base()
        self.lm_head = torch.nn.Linear(V, V, bias=False)
        self.lm_head.weight.data = table.T.contiguous()


@pytest.fixture(autouse=True)
def gumbel(monkeypatch):
    monkeypatch.setattr(verifier, "draw_settings", lambda: verifier.DrawSettings(
        draw=fs.DRAW_GUMBEL_MAX, t=1.0, top_k=0, top_p=1.0, stochastic_threshold=0.99, boundary_epsilon=0.05))
    monkeypatch.setattr(verifier, "lm_head_fp32", lambda: True)


def _generate(table, uniforms, override=None):
    """Autoregressive Gumbel picks; ``override(j)`` replaces the pick at completion offset j (then generation goes
    on from the replaced token, as a forger editing a rollout and regenerating after would)."""
    tokens, last = [], PROMPT[-1]
    for j, u in enumerate(uniforms):
        pick = int(fs.pick_rows(table[last:last + 1], [u], draw=fs.DRAW_GUMBEL_MAX)[0])
        edited = None if override is None else override(j, last)
        if edited is not None:
            pick = edited
        tokens.append(pick)
        last = pick
    return tuple(tokens)


def _group(table, *, nonce_for_draw=NONCE, override_rollout=None, override=None) -> SubmittedGroup:
    completions = tuple(
        _generate(table, lease_uniforms(nonce_for_draw, 9, 3, i, LENGTH),
                  override if i == override_rollout else None)
        for i in range(16))
    return SubmittedGroup("L1", "A", "env", 9, 3, NONCE, 0.0, (0.0,) * 16, (False,) * 16, PROMPT, completions,
                          (None,) * 16)


def test_an_honest_group_passes_with_full_match():
    table = _transitions()
    result = rl_audit.audit_forward(TinyLM(table))(_group(table))
    assert result == AuditResult(True, None, (1.0,) * 16)


def test_a_group_drawn_with_other_noise_fails_statistically():
    table = _transitions()
    result = rl_audit.audit_forward(TinyLM(table))(_group(table, nonce_for_draw=OTHER_NONCE))
    assert result.passed is False and result.failure_class == AUDIT_FAILURE_STATISTICAL
    assert max(result.rollout_match) < 0.9


def test_tampered_tokens_at_near_deterministic_positions_are_forgery():
    table = _transitions()
    edits = []

    def tamper(j, last):                      # 3 edits at peaked positions: a token the draw never yields there
        if last >= 16 and len(edits) < 3 and j > 4:
            edits.append(j)
            return 31 if (last * 7) % 16 != 31 else 30
        return None

    group = _group(table, override_rollout=5, override=tamper)
    assert len(edits) == 3
    result = rl_audit.audit_forward(TinyLM(table))(group)
    assert result.passed is False and result.failure_class == AUDIT_FAILURE_DETERMINISTIC
    assert result.rollout_match[5] >= rl_audit.RL_ROLLOUT_MATCH_FLOOR


def test_a_rollout_under_the_floor_is_excluded_the_group_is_judged_pooled():
    def d(stochastic, match):
        return fs.SeedConsistencyDiagnostics(n_positions=stochastic, n_stochastic=stochastic, n_exact_match=match)

    result = rl_audit.judge([d(100, 85)] + [d(100, 99)] * 15)           # pooled 0.981
    assert result.passed is True and result.rollout_match[0] == 0.85 < rl_audit.RL_ROLLOUT_MATCH_FLOOR
    result = rl_audit.judge([d(100, 80)] * 8 + [d(100, 99)] * 8)        # pooled 0.895 < the group floor
    assert result.passed is False and result.failure_class == AUDIT_FAILURE_STATISTICAL
    result = rl_audit.judge([d(1, 0)] * 16)                              # 16 drawn positions: abstain, suspect
    assert result.passed is None and result.suspect


def _drawn(drawn, match, stochastic=0, stochastic_match=0):
    return fs.SeedConsistencyDiagnostics(n_positions=drawn, n_stochastic=stochastic, n_exact_match=stochastic_match,
                                         n_drawn_exact_match=match)


def test_one_token_rollouts_never_pass_an_audit():
    """16 one-token rollouts (e.g. just EOS): too few drawn positions to judge, the audit abstains as a suspect
    inconclusive (no pass, no pay); a few more tokens off the draw fail on the every-position match."""
    result = rl_audit.judge([_drawn(1, 0)] * 16)
    assert result == AuditResult(None, None, (0.0,) * 16, suspect=True)
    result = rl_audit.judge([_drawn(1, 1)] * 16)                         # even when the token is the pick
    assert result.passed is None and result.suspect
    result = rl_audit.judge([_drawn(4, 0)] * 16)
    assert result.passed is False and result.failure_class == AUDIT_FAILURE_STATISTICAL
    assert rl_audit.judge([_drawn(4, 4)] * 16).passed is True             # a peaked honest short group


def test_one_rollout_under_the_forgery_floor_fails_the_group():
    honest = [_drawn(100, 100, 100, 99)] * 15
    result = rl_audit.judge(honest + [_drawn(1, 0)])                     # pooled 0.99, one [EOS] off the draw
    assert result.passed is False and result.failure_class == AUDIT_FAILURE_STATISTICAL
    assert rl_audit.judge(honest + [_drawn(100, 85, 100, 80)]).passed is True   # honest noise: excluded from pay


def test_force_span_positions_are_not_checked():
    table = _transitions()
    honest = _group(table)
    forced = list(honest.completions[2])
    forced[10:14] = [31, 31, 31, 31]        # injected text the draw did not produce...
    spans = list(honest.force_spans)
    spans[2] = (10, 14)                      # ...declared as the BFT span
    group = SubmittedGroup(**{**honest.__dict__, "completions": honest.completions[:2] + (tuple(forced),)
                              + honest.completions[3:], "force_spans": tuple(spans)})
    diagnostics = rl_audit.rollout_draw_diagnostics(TinyLM(table), group, 2)
    assert diagnostics.n_positions == LENGTH - 4


def test_an_unavailable_model_is_inconclusive():
    table = _transitions()
    assert rl_audit.audit_forward(None)(_group(table)) == AuditResult(None)


def test_the_runner_drains_run_audits_with_the_forward_and_limit():
    table = _transitions()
    calls = []

    class Submissions:
        def run_audits(self, forward, limit):
            calls.append(limit)
            return int(forward(_group(table)).passed is not None)

    assert rl_audit.run_rl_audits(TinyLM(table), Submissions(), limit=3) == 1 and calls == [3]
    assert rl_audit.run_rl_audits(None, Submissions()) == 0 and calls[-1] == rl_audit.DEFAULT_AUDIT_LIMIT
