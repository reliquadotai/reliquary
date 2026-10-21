"""Out-of-zone audits judged by the async trainer (``trainer.audits``): the same diagnostics and verdict as the
validator's proof forward, through the trainer's own model (plain head or the FSDP2 fused head), and the loop that
takes jobs from the validator and reports them."""
from __future__ import annotations

import pytest

import dataclasses

from reliquary.trainer.audits import TrainerAudits, audit_group, group_diagnostics, rollout_diagnostics
from reliquary.validator import rl_audit
from reliquary.validator.rl_submit import AuditResult
from tests.unit.test_rl_audit import TinyLM, _group, _transitions, gumbel  # noqa: F401  (gumbel: autouse fixture)


def _forged(table):
    return _group(table, override_rollout=2, override=lambda j, last: (last * 7 + 3) % 16 if j == 30 else None)


@pytest.mark.parametrize("make", [_group, _forged, lambda t: _group(t, nonce_for_draw="cd" * 32)])
def test_the_trainer_forward_judges_as_the_proof_forward(make):
    table = _transitions()
    group = make(table)
    model = TinyLM(table)
    assert audit_group(model, group) == rl_audit.audit_forward(model)(group)
    for i in range(len(group.completions)):
        assert rollout_diagnostics(model, group, i) == rl_audit.rollout_draw_diagnostics(model, group, i)


def test_the_fused_head_projects_the_audit_rows_inside_its_own_call():
    from reliquary.trainer.distributed import FusedLogprobHead

    table = _transitions()
    group = _forged(table)
    plain = rl_audit.audit_forward(TinyLM(table))(group)
    model = TinyLM(table)
    calls = []
    model.lm_head.__class__ = FusedLogprobHead           # what fsdp2's shard_model installs
    original = FusedLogprobHead.forward

    def counted(self, *args, **kw):
        calls.append(kw.get("audit") is not None)
        return original(self, *args, **kw)

    FusedLogprobHead.forward = counted
    try:
        assert audit_group(model, group) == plain
    finally:
        FusedLogprobHead.forward = original
    assert calls == [True]                               # one head call for the packed group: one gather under FSDP2


class Source:
    def __init__(self, jobs, gone=()):
        self.jobs, self.gone, self.reported, self.asked = list(jobs), set(gone), {}, []

    def take_audits(self, limit):
        self.asked.append(limit)
        return self.jobs[:limit]

    def report_audit(self, lease_id, result):
        if lease_id in self.gone:
            raise LookupError(lease_id)
        self.reported[lease_id] = result
        self.jobs = [j for j in self.jobs if j.lease_id != lease_id]


class Job:
    def __init__(self, lease_id):
        self.lease_id = lease_id


def test_the_trainer_loop_takes_judges_and_reports_within_its_budget():
    t = [0.0]
    judged = []

    def audit_fn(model, group):
        judged.append(group.lease_id)
        t[0] += 10.0
        return AuditResult(True, None, (1.0,) * 16)

    source = Source([Job(f"L{i}") for i in range(5)], gone={"L1"})
    audits = TrainerAudits(source, audit_fn=audit_fn, clock=lambda: t[0])
    assert audits.run(object(), limit=4, budget_s=25.0) == 3              # the 4th waits: 30 s spent > 25 s
    assert judged == ["L0", "L1", "L2"] and set(source.reported) == {"L0", "L2"}   # L1 concluded meanwhile
    assert audits.metrics() == {"async/audits": 3.0, "async/audit_s": 30.0}
    assert audits.metrics() == {"async/audits": 0.0, "async/audit_s": 0.0}
    assert audits.run(object(), limit=0, budget_s=25.0) == 0 and source.asked == [4]


def _ragged(group, lengths):
    """The group with rollout i cut to ``lengths[i]`` tokens (mixed lengths: padded micro-batches)."""
    completions = tuple(c[:n] for c, n in zip(group.completions, lengths))
    return dataclasses.replace(group, completions=completions)


@pytest.mark.parametrize("make", [_group, _forged, lambda t: _group(t, nonce_for_draw="cd" * 32)])
@pytest.mark.parametrize("budget", [10_000, 120, 40])
def test_the_packed_group_forward_gives_each_rollouts_own_diagnostics(make, budget):
    """Packed into right-padded micro-batches under any token budget (one rollout over it bins alone), every rollout
    keeps exactly the diagnostics of its own forward, and the verdict is the proof forward's."""
    table = _transitions()
    full = make(table)
    group = _ragged(full, [len(full.completions[0]) - (i * 5) % 23 for i in range(16)])
    model = TinyLM(table)
    assert group_diagnostics(model, group, token_budget=budget) == [
        rollout_diagnostics(model, group, i) for i in range(16)]
    assert audit_group(model, group) == rl_audit.audit_forward(model)(group)
