"""fp32 master weights: a 1e-6 Adam step must survive bf16 storage."""

import pytest
import torch

import reliquary.constants as C
import reliquary.validator.training as training
from reliquary.validator.training import (
    _MasterWeightOptimizer,
    _build_optimizer,
    _lazy_init,
    _use_master_weights,
    reset_training_state,
)

# ~0.0075 is the median |w| of Teutonic-I's MLP weights: its bf16 ulp is
# 3.05e-5, so a single 1e-6 step is a thirtieth of the rounding step.
W0 = torch.tensor(0.0075, dtype=torch.bfloat16).item()
STEPS = 40


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(training, "LEARNING_RATE", 1e-6)
    reset_training_state()
    yield
    reset_training_state()


def _bf16_weight(n: int = 64) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.full((n,), W0, dtype=torch.bfloat16))


def _run(optimizer, parameter, steps: int = STEPS) -> None:
    for _ in range(steps):
        optimizer.zero_grad()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()


def test_in_place_bf16_steps_round_every_update_away():
    parameter = _bf16_weight()
    _run(_build_optimizer([parameter]), parameter)
    assert torch.all(parameter == W0)


def test_masters_accumulate_what_bf16_cannot_hold():
    parameter = _bf16_weight()
    optimizer = _MasterWeightOptimizer([parameter], _build_optimizer)
    _run(optimizer, parameter)

    expected = W0 - STEPS * 1e-6
    master = optimizer.masters[0]
    assert master.dtype == torch.float32
    assert torch.allclose(master, torch.full_like(master, expected), atol=5e-8)
    # The model holds the rounded master, i.e. exactly what gets published.
    assert parameter.dtype == torch.bfloat16
    assert torch.equal(parameter, master.to(torch.bfloat16))
    assert torch.all(parameter < W0)


def test_a_chunked_step_updates_every_parameter_exactly_once(monkeypatch):
    def run(chunk_numel):
        monkeypatch.setattr(training, "_MASTER_STEP_CHUNK_NUMEL", chunk_numel)
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(n, dtype=torch.bfloat16)) for n in (5, 7, 3)]
        optimizer = _MasterWeightOptimizer(params, _build_optimizer)
        for step in range(3):
            optimizer.zero_grad()
            for index, parameter in enumerate(params):
                parameter.grad = torch.full_like(parameter, float(index + step + 1))
            optimizer.step()
        steps = [float(optimizer.inner.state[m]["step"]) for m in optimizer.masters]
        return [m.detach().clone() for m in optimizer.masters], steps

    whole, whole_steps = run(1 << 28)
    chunked, chunked_steps = run(1)
    assert whole_steps == chunked_steps == [3.0, 3.0, 3.0]
    for a, b in zip(whole, chunked):
        assert torch.equal(a, b)


def test_a_parameter_without_gradient_is_left_alone():
    frozen, moving = _bf16_weight(4), _bf16_weight(4)
    optimizer = _MasterWeightOptimizer([frozen, moving], _build_optimizer)
    optimizer.zero_grad()
    moving.grad = torch.ones_like(moving)
    optimizer.step()
    assert optimizer.masters[0] not in optimizer.inner.state
    assert torch.all(optimizer.masters[0] == W0)


def test_zero_grad_clears_the_model_gradients():
    parameter = _bf16_weight(4)
    optimizer = _MasterWeightOptimizer([parameter], _build_optimizer)

    parameter.grad = torch.ones_like(parameter)
    optimizer.zero_grad(set_to_none=False)
    assert torch.equal(parameter.grad, torch.zeros_like(parameter))

    optimizer.zero_grad()
    assert parameter.grad is None


def test_the_flag_and_a_low_precision_parameter_are_both_required(monkeypatch):
    bf16 = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    fp32 = torch.nn.Parameter(torch.ones(2))

    monkeypatch.setattr(C, "OPTIMIZER_MASTER_WEIGHTS", True)
    assert _use_master_weights([fp32, bf16]) is True
    assert _use_master_weights([fp32]) is False

    monkeypatch.setattr(C, "OPTIMIZER_MASTER_WEIGHTS", False)
    assert _use_master_weights([bf16]) is False


def test_lazy_init_schedules_the_masters(monkeypatch):
    monkeypatch.setattr(C, "OPTIMIZER_MASTER_WEIGHTS", True)
    model = torch.nn.Linear(4, 4).to(torch.bfloat16)

    assert _lazy_init(model)
    assert isinstance(training._optimizer, _MasterWeightOptimizer)
    assert training._scheduler.optimizer is training._optimizer.inner
    assert training._optimizer.param_groups[0]["lr"] == pytest.approx(
        training._scheduler.get_last_lr()[0]
    )


def test_lazy_init_keeps_the_in_place_optimizer_when_off(monkeypatch):
    monkeypatch.setattr(C, "OPTIMIZER_MASTER_WEIGHTS", False)
    model = torch.nn.Linear(4, 4).to(torch.bfloat16)

    assert _lazy_init(model)
    assert isinstance(training._optimizer, torch.optim.AdamW)
    assert training._scheduler.optimizer is training._optimizer
