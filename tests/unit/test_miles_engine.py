"""The async trainer's step through a Miles Tinker gateway (reliquary/trainer/miles_engine.py)."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tinker")

from reliquary import constants as C  # noqa: E402
from reliquary.trainer import miles_engine  # noqa: E402
from reliquary.trainer.async_trainer import DrawGate, training_group  # noqa: E402
from reliquary.trainer.miles_engine import MilesTrainStep  # noqa: E402
from reliquary.validator.training import TrainingStepSkipped  # noqa: E402


class Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class FakeClient:
    def __init__(self, metrics=None, rejected=(), fail=None):
        self.calls, self.metrics, self.rejected, self.fail = [], metrics, set(rejected), fail
        self.groups = set()

    def forward_backward(self, datums, loss_fn, config):
        self.calls.append(("forward_backward", len(datums), loss_fn, dict(config)))
        if self.fail is not None:
            return Future(self.fail)
        outs = []
        for d in datums:
            n = len(d.loss_fn_inputs["target_tokens"].data)
            self.groups.add(d.loss_fn_inputs["group_ids"].data[0])
            outs.append({"gate": np.array([1, 40, 40, 40, 40, 0.0, 0.0, 0], dtype=np.float32),
                         "vstar": np.zeros(n), "gap": np.zeros(n), "match": np.ones(n), "stochastic": np.ones(n)})
        return Future(SimpleNamespace(loss_fn_outputs=outs, metrics={}))

    def optim_step(self, adam):
        self.calls.append(("optim_step", adam))
        metrics = dict(self.metrics) if self.metrics is not None else {"grad_norm": 0.5}
        metrics.update({"gate_groups": len(self.groups), "gate_groups_rejected": len(self.rejected),
                        **{f"gate_group_rejected/{g}": 1.0 for g in self.rejected}})
        return Future(SimpleNamespace(metrics=metrics))


def _batch(n_groups, size=2):
    groups, slot = [], 0
    for g in range(n_groups):
        raw = SimpleNamespace(prompt_tokens=(1, 2), completions=((10, 11, 12),) * size, force_spans=(None,) * size,
                              spans=None, truncated=(False,) * size, rewards=tuple(float(i % 2) for i in range(size)),
                              environment="math", prompt_idx=g, lease_id=f"l{g}", model_tokens=lambda i: 3)
        groups.append(training_group(raw, slot0=slot, index=g, uniforms=[[0.5] * 3] * size))
        slot += size
    gate = DrawGate([[r._async_slot for r in g.rollouts] for g in groups])
    return groups, gate


@pytest.fixture
def logged(monkeypatch):
    rows = []
    monkeypatch.setattr(miles_engine.telemetry, "log_training_step", lambda m, step: rows.append((step, m)))
    return rows


def test_one_step_sends_chunks_then_one_optim_step(logged):
    groups, gate = _batch(65)                         # 130 rollouts
    client = FakeClient()
    model = object()
    out = MilesTrainStep(client, chunk=64, eos_ids=(7, 9))(model, [groups], gate=gate, window_index=1)
    assert out is model
    kinds = [c[0] for c in client.calls]
    assert kinds == ["forward_backward"] * 3 + ["optim_step"]
    assert [c[1] for c in client.calls[:3]] == [64, 64, 2]
    assert all(c[2] == "reliquary_gate" and c[3] == {"eos_ids": "7,9"} for c in client.calls[:3])
    adam = client.calls[-1][1]
    assert (adam.beta1, adam.beta2, adam.eps, adam.weight_decay, adam.grad_clip_norm) == (
        0.9, C.ADAM_BETA2, 1e-8, 0.01, C.GRAD_CLIP_NORM)
    assert gate.accepted == [True] * 65 and sum(gate.forwarded) == 130


def test_lr_warmup_and_resume():
    fresh = MilesTrainStep(FakeClient(), eos_ids=())
    assert fresh.lr_for(0) == pytest.approx(C.LEARNING_RATE / C.LR_WARMUP_WINDOWS)
    assert fresh.lr_for(C.LR_WARMUP_WINDOWS + 3) == pytest.approx(C.LEARNING_RATE)
    resumed = MilesTrainStep(FakeClient(), eos_ids=(), global_step_hint=20)
    # Adam restarts fresh at a resume: the trainer's short re-warmup
    rewarm = C.LR_RESTART_REWARMUP_WINDOWS
    assert resumed.lr_for(20) == pytest.approx(C.LEARNING_RATE / (rewarm + 1))
    assert resumed.lr_for(20 + rewarm) == pytest.approx(C.LEARNING_RATE)


def test_step_counter_advances_only_on_a_real_step(logged):
    groups, gate = _batch(2)
    step = MilesTrainStep(FakeClient(), eos_ids=())
    step(None, [groups], gate=gate)
    assert step.step == 1
    groups, gate = _batch(2)
    with pytest.raises(TrainingStepSkipped):
        MilesTrainStep(FakeClient({"skipped_grad_spike": 1, "grad_norm": 500.0}), eos_ids=())(None, [groups],
                                                                                             gate=gate)


@pytest.mark.parametrize("metrics,reason", [({"skipped_grad_spike": 1, "grad_norm": 500.0}, "gradient_spike"),
                                            ({"skipped_nonfinite": 1}, "nonfinite_gradient")])
def test_skips_raise_training_step_skipped(logged, metrics, reason):
    groups, gate = _batch(2)
    step = MilesTrainStep(FakeClient(metrics), eos_ids=())
    with pytest.raises(TrainingStepSkipped) as info:
        step(None, [groups], gate=gate)
    assert info.value.reason == reason and step.step == 0
    assert logged and logged[-1][1][f"train/step_skipped_{'grad_spike' if 'spike' in reason else 'nonfinite'}"] == 1.0


def test_report_follows_rejected_groups(logged):
    groups, gate = _batch(3)
    MilesTrainStep(FakeClient(rejected={1}), eos_ids=())(None, [groups], gate=gate)
    assert gate.accepted == [True, False, True]


def test_gateway_error_propagates(logged):
    groups, gate = _batch(2)
    client = FakeClient(fail=ConnectionError("gateway down"))
    with pytest.raises(ConnectionError):
        MilesTrainStep(client, eos_ids=())(None, [groups], gate=gate)
    assert "optim_step" not in [c[0] for c in client.calls]


def test_no_plan_no_call(logged):
    raw = SimpleNamespace(prompt_tokens=(1,), completions=((5, 6),) * 2, force_spans=(None,) * 2, spans=None,
                          truncated=(False,) * 2, rewards=(1.0, 1.0), environment="math", prompt_idx=0,
                          lease_id="l", model_tokens=lambda i: 2)
    group = training_group(raw, slot0=0, index=0, uniforms=[[0.5] * 2] * 2)
    client = FakeClient()
    assert MilesTrainStep(client, eos_ids=())("m", [[group]], gate=DrawGate([[0, 1]])) == "m"
    assert client.calls == []


def test_metrics_logged(logged):
    groups, gate = _batch(2)
    MilesTrainStep(FakeClient(), eos_ids=())(None, [groups], gate=gate, window_index=4)
    step, metrics = logged[-1]
    assert step == 4 and metrics["train/grad_norm"] == 0.5
    assert metrics["train/lr_applied"] == pytest.approx(C.LEARNING_RATE / C.LR_WARMUP_WINDOWS)
    assert metrics["async/groups_accepted"] == 2 and "rewards/mean" in metrics


def test_episodes_go_in_their_own_chunks_without_the_final_eos_test(logged):
    groups, gate = _batch(2)
    raw = SimpleNamespace(prompt_tokens=(1, 2), completions=((10, 11, 12),) * 2, force_spans=(None,) * 2,
                          spans=(((0, 1), (2, 3)),) * 2, truncated=(False,) * 2, rewards=(0.0, 1.0),
                          environment="swe", prompt_idx=9, lease_id="ep", model_tokens=lambda i: 2)
    groups.append(training_group(raw, slot0=4, index=2, uniforms=[[0.5] * 2] * 2))
    gate = DrawGate([[r._async_slot for r in g.rollouts] for g in groups])
    client = FakeClient()
    MilesTrainStep(client, eos_ids=(7,))(None, [groups], gate=gate)
    sent = [(c[1], c[3]) for c in client.calls if c[0] == "forward_backward"]
    assert sent == [(4, {"eos_ids": "7"}), (2, {"eos_ids": "7", "terminal_check": 0})]
    assert gate.accepted == [True, True, True]
