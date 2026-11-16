"""DrawGate state from the Miles gateway's reliquary_gate outputs (reliquary/trainer/miles_gate.py)."""
from types import SimpleNamespace

import numpy as np
import pytest

from reliquary.trainer.async_trainer import AsyncTrainer, DrawGate
from reliquary.trainer.miles_gate import apply_gateway_verdicts, fill_gate


def _rollout(tokens, prompt_length, *, episode=False):
    meta = {"prompt_length": prompt_length, "completion_length": len(tokens) - prompt_length}
    if episode:
        meta["episode"] = {"assistant_spans": []}
    return SimpleNamespace(commit={"tokens": list(tokens), "rollout": meta})


def _out(gate, *, vstar=None, gap=None, match=None, n=5):
    return {"gate": np.array(gate, dtype=np.float32),
            "vstar": np.array(vstar if vstar is not None else [0] * n),
            "gap": np.array(gap if gap is not None else [0.0] * n, dtype=np.float32),
            "match": np.array(match if match is not None else [1.0] * n, dtype=np.float32),
            "stochastic": np.ones(n, dtype=np.float32)}


EOS = 99


def test_fill_gate_tallies_and_substitutions():
    # tokens: prompt (1, 2), completion (10, 11, 12, 13); targets j = 0..4 predict tokens[1..5]
    rollouts = [_rollout([1, 2, 10, 11, 12, 13], 2), _rollout([1, 2, 20, 21, 22, EOS], 2)]
    gate = DrawGate([[0, 1]], eos_ids=(EOS,))
    outputs = [
        _out([1, 40, 39, 40, 39, 3.5, 0.0, 1], vstar=[2, 10, 50, 12, 13], gap=[0, 0, 3.5, 0, 0],
             match=[0, 1, 0, 1, 1]),
        _out([0, 40, 30, 40, 30, 2.0, 2.0, 0], vstar=[2, 20, 21, 22, EOS], gap=[0, 0, 0, 0, 2.0],
             match=[0, 1, 1, 1, 0]),
    ]
    fill_gate(gate, [(0, 0), (0, 1)], outputs, rollouts)
    assert gate.tallies == [[40, 39, 40, 39], [40, 30, 40, 30]]
    assert gate.forwarded == [1, 1] and gate.admitted == [1, 0]
    assert gate.max_gap == [3.5, 2.0]
    # target j=2 is completion position 1; the EOS pick at the last position is re-graded whatever its gap
    assert gate.substitutions == [[(1, 50)], [(3, EOS)]]
    assert gate.bad_terminal == [0, 1]                       # ends on EOS with last gap 2.0 > RL_TERMINAL_GAP


def test_episode_never_has_a_bad_terminal():
    gate = DrawGate([[0]], eos_ids=(EOS,))
    fill_gate(gate, [(0, 0)], [_out([1, 40, 40, 40, 40, 2.0, 2.0, 0])],
              [_rollout([1, 2, 10, 11, 12, EOS], 2, episode=True)])
    assert gate.bad_terminal == [0]


def test_report_follows_gateway_verdict():
    rollouts = [_rollout([1, 2, 10, 11, 12, 13], 2) for _ in range(4)]
    gate = DrawGate([[0, 1], [2, 3]])
    good = _out([1, 40, 40, 40, 40, 0.0, 0.0, 0])
    fill_gate(gate, [(0, 0), (0, 1), (1, 2), (1, 3)], [good] * 4, rollouts)
    # the local tallies would accept both groups; the gateway rejected group 1
    apply_gateway_verdicts(gate, {"gate_groups": 2, "gate_groups_rejected": 1, "gate_rollouts_backed_out": 2,
                                  "gate_group_rejected/1": 1.0})
    assert gate.accepted == [True, False] and gate.undone == 1
    reports = []
    trainer = SimpleNamespace(_reporter=None, submissions=SimpleNamespace(
        report_training=lambda lease_id, **kw: reports.append((lease_id, kw["accepted"]))))
    groups = [SimpleNamespace(lease_id=f"l{i}", completions=[[0]] * 2, environment="math") for i in range(2)]
    AsyncTrainer._report(trainer, groups, gate)
    assert reports == [("l0", True), ("l1", False)]


def test_unforwarded_group_is_never_accepted():
    gate = DrawGate([[0, 1], [2]])
    fill_gate(gate, [(0, 0), (0, 1)], [_out([1, 40, 40, 40, 40, 0.0, 0.0, 0])] * 2,
              [_rollout([1, 2, 10, 11, 12, 13], 2)] * 2)
    apply_gateway_verdicts(gate, {"gate_groups": 1, "gate_groups_rejected": 0, "gate_rollouts_backed_out": 0})
    assert gate.accepted == [True, False]


def test_verdict_count_mismatch_raises():
    gate = DrawGate([[0]])
    fill_gate(gate, [(0, 0)], [_out([1, 40, 40, 40, 40, 0.0, 0.0, 0])], [_rollout([1, 2, 10, 11, 12, 13], 2)])
    with pytest.raises(RuntimeError, match="groups"):
        apply_gateway_verdicts(gate, {"gate_groups": 2, "gate_groups_rejected": 0})
    with pytest.raises(RuntimeError, match="verdicts"):
        apply_gateway_verdicts(gate, {})
