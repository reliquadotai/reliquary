"""Out-of-zone audits on the Miles gateway (reliquary/trainer/audits.gateway_audit, diagnostics_from_rows)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX, gumbel_gaps, seed_consistency_diagnostics
from reliquary.trainer.audits import diagnostics_from_rows
from reliquary.validator.training import _draw_stats


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_diagnostics_from_rows_match_cpu_reference(seed):
    g = torch.Generator().manual_seed(seed)
    rows, vocab = 64, 50
    logits = torch.randn(rows, vocab, generator=g) * 2
    logits[: rows // 3, 5] += 30                        # peaked: deterministic positions
    u = torch.rand(rows, generator=g, dtype=torch.float64).tolist()
    picks, _ = gumbel_gaps(logits, u, torch.zeros(rows, dtype=torch.long))
    tokens = picks.clone()
    flip = torch.rand(rows, generator=g) < 0.3
    tokens[flip] = torch.randint(0, vocab, (int(flip.sum()),), generator=g)
    tokens[0] = (picks[0] + 1) % vocab                  # a hard miss on a deterministic row
    offsets = list(range(3, 3 + rows))
    ref = seed_consistency_diagnostics(logits, tokens.tolist(), u, t=1.0, top_k=0, top_p=1.0,
                                       stochastic_threshold=0.99, boundary_epsilon=0.02,
                                       position_offsets=offsets, draw=DRAW_GUMBEL_MAX)
    # the gateway's rows (reliquary_gate: v*, Gumbel gap, match, stochastic over drawn positions)
    vstar, gap = gumbel_gaps(logits, u, tokens)
    stochastic, _ = _draw_stats(logits, 0.99, scratch=False)
    got = diagnostics_from_rows(match=(vstar == tokens).tolist(), gap=gap.tolist(), stochastic=stochastic.tolist(),
                                offsets=offsets, boundary_epsilon=0.02)
    assert got.n_deterministic_hard_mismatch >= 1
    for field in ("n_positions", "n_stochastic", "n_exact_match", "n_boundary_match", "n_hard_mismatch",
                  "n_deterministic_hard_mismatch", "n_miss_gt_0_01", "n_miss_gt_0_05", "n_miss_gt_0_10",
                  "first_hard_mismatch_offset", "n_drawn_exact_match", "n_far_miss"):
        assert getattr(got, field) == getattr(ref, field), field
    assert got.max_cdf_miss == pytest.approx(ref.max_cdf_miss, abs=1e-5)
    assert got.last_miss == pytest.approx(ref.last_miss, abs=1e-5)


class Client:
    def __init__(self, match_rate):
        self.match_rate, self.calls = match_rate, []

    def forward(self, datums, loss_fn, config):
        self.calls.append((len(datums), loss_fn, config))
        outs = []
        for d in datums:
            mask = np.array(d.loss_fn_inputs["draw_mask"].data)
            n = len(mask)
            match = np.zeros(n)
            hits = np.flatnonzero(mask)[: int(round(self.match_rate * mask.sum()))]
            match[hits] = 1.0
            gap = np.where((mask > 0) & (match == 0), 5.0, 0.0)
            outs.append({"match": match, "gap": gap, "stochastic": np.ones(n), "vstar": np.zeros(n),
                         "gate": np.zeros(8)})
        return SimpleNamespace(result=lambda: SimpleNamespace(loss_fn_outputs=outs))


def _submitted(n=4, length=200):
    return SimpleNamespace(prompt_tokens=(1, 2, 3), completions=((7,) * length,) * n, force_spans=(None,) * n,
                           spans=None, truncated=(False,) * n, rewards=(0.0,) * n, environment="math",
                           prompt_idx=1, lease_id="lease", model_tokens=lambda i: length,
                           draw_uniforms=lambda i: [0.5] * length)


def test_audit_verdict_passes_and_fails(monkeypatch):
    pytest.importorskip("tinker")
    from reliquary.trainer.audits import gateway_audit
    from reliquary.validator import verifier

    monkeypatch.setattr(verifier, "draw_settings", lambda: SimpleNamespace(
        draw=DRAW_GUMBEL_MAX, t=1.0, top_k=0, top_p=1.0, boundary_epsilon=0.02))

    honest = Client(1.0)
    result = gateway_audit(honest, eos_ids=(9,))(None, _submitted())
    assert result.passed is True and honest.calls == [(4, "reliquary_gate", {"eos_ids": "9"})]
    assert gateway_audit(Client(0.3), eos_ids=())(None, _submitted()).passed is False
    episodes = Client(1.0)
    group = _submitted(length=200)
    group.spans = (((0, 100), (150, 200)),) * 4
    group.model_tokens = lambda i: 150
    group.draw_uniforms = lambda i: [0.5] * 150
    gateway_audit(episodes, eos_ids=(9,))(None, group)
    assert episodes.calls[0][2] == {"eos_ids": "9", "terminal_check": 0}


def test_gateway_audit_refuses_a_warped_draw(monkeypatch):
    from reliquary.trainer.audits import gateway_audit
    from reliquary.validator import verifier

    monkeypatch.setattr(verifier, "draw_settings", lambda: SimpleNamespace(
        draw=DRAW_GUMBEL_MAX, t=1.0, top_k=20, top_p=1.0, boundary_epsilon=0.02))
    with pytest.raises(RuntimeError, match="unwarped"):
        gateway_audit(Client(1.0), eos_ids=())


class VaryingClient(Client):
    """Each datum's rows depend on the datum alone (its tokens), never on what else shares the call."""

    def forward(self, datums, loss_fn, config):
        self.calls.append((len(datums), loss_fn, config))
        outs = []
        for d in datums:
            mask = np.array(d.loss_fn_inputs["draw_mask"].data)
            tokens = np.array(d.loss_fn_inputs["target_tokens"].data)
            n = len(mask)
            match = ((tokens % 3) != 0).astype(float) * (mask > 0)
            gap = np.where((mask > 0) & (match == 0), 0.001 * (1 + tokens % 7), 0.0)
            outs.append({"match": match, "gap": gap, "stochastic": (tokens % 2).astype(float),
                         "vstar": np.zeros(n), "gate": np.zeros(8)})
        return SimpleNamespace(result=lambda: SimpleNamespace(loss_fn_outputs=outs))


def _varied(seed, n=16, length=120, episode=False):
    rng = np.random.default_rng(seed)
    completions = tuple(tuple(int(t) for t in rng.integers(0, 50, length)) for _ in range(n))
    group = SimpleNamespace(prompt_tokens=(1, 2, 3), completions=completions, force_spans=(None,) * n,
                            spans=None, truncated=(False,) * n, rewards=(0.0,) * n, environment="math",
                            prompt_idx=seed, lease_id=f"lease-{seed}", model_tokens=lambda i: length,
                            draw_uniforms=lambda i: [0.5] * length)
    if episode:
        group.spans = (((0, 60), (90, length)),) * n
        group.model_tokens = lambda i: 90
        group.draw_uniforms = lambda i: [0.5] * 90
    return group


def test_judging_groups_together_gives_each_group_its_own_verdict(monkeypatch):
    """``audit.many`` sends several groups in one forward (one per draw config: episodes apart) and judges each
    group exactly as a forward of that group alone would."""
    pytest.importorskip("tinker")
    from reliquary.trainer.audits import gateway_audit
    from reliquary.validator import verifier

    monkeypatch.setattr(verifier, "draw_settings", lambda: SimpleNamespace(
        draw=DRAW_GUMBEL_MAX, t=1.0, top_k=0, top_p=1.0, boundary_epsilon=0.02))
    groups = [_varied(0), _varied(1, episode=True), _varied(2, n=4), _varied(3)]
    alone = VaryingClient(1.0)
    one = gateway_audit(alone, eos_ids=(9,))
    expected = [one(None, g) for g in groups]
    assert len(alone.calls) == 4
    together = VaryingClient(1.0)
    audit = gateway_audit(together, eos_ids=(9,))
    assert audit.many(None, groups) == expected
    assert sorted(together.calls, key=lambda c: c[0]) == [
        (16, "reliquary_gate", {"eos_ids": "9", "terminal_check": 0}), (36, "reliquary_gate", {"eos_ids": "9"})]
    assert audit.many(None, []) == []


def test_a_short_gateway_reply_is_a_clear_error(monkeypatch):
    pytest.importorskip("tinker")
    from reliquary.trainer.audits import gateway_audit
    from reliquary.validator import verifier

    monkeypatch.setattr(verifier, "draw_settings", lambda: SimpleNamespace(
        draw=DRAW_GUMBEL_MAX, t=1.0, top_k=0, top_p=1.0, boundary_epsilon=0.02))

    class Short(VaryingClient):
        def forward(self, datums, loss_fn, config):
            full = super().forward(datums, loss_fn, config).result().loss_fn_outputs
            return SimpleNamespace(result=lambda: SimpleNamespace(loss_fn_outputs=full[:-1]))

    with pytest.raises(RuntimeError, match="31 rows for 32 rollouts"):
        gateway_audit(Short(1.0), eos_ids=(9,)).many(None, [_varied(0), _varied(1)])
