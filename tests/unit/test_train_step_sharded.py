import copy
from dataclasses import dataclass, field

import pytest
import torch

from tests.unit.trainer_dist_helpers import run_ranks, tiny_qwen35


@dataclass
class _Rollout:
    tokens: list
    reward: float
    commit: dict = field(default_factory=dict)


@dataclass
class _Group:
    rollouts: list
    prompt_idx: int = 0


def _batches(model, *, n_groups=3, outlier_group=None, seed=7):
    """Groups whose pi_old is the model's own logprobs (ratio 1), so no gate
    fires unless ``outlier_group`` shifts one group's pi_old far away."""
    from reliquary.validator.training import _batched_completion_logprobs

    g = torch.Generator().manual_seed(seed)
    groups = []
    for gi in range(n_groups):
        rollouts = []
        for ri in range(4):
            length = int(torch.randint(24, 64, (1,), generator=g))
            tokens = torch.randint(0, 300, (length,), generator=g).tolist()
            with torch.no_grad():
                x = torch.tensor([tokens])
                lp, _ = _batched_completion_logprobs(model, x, torch.ones_like(x), [8], [length])
            old = lp.tolist()
            if gi == outlier_group:
                old = [v - 3.0 for v in old]
            rollouts.append(_Rollout(tokens=tokens, reward=float(ri % 2), commit={
                "tokens": tokens,
                "rollout": {"prompt_length": 8, "token_logprobs": old},
            }))
        groups.append(_Group(rollouts=rollouts, prompt_idx=gi))
    return [groups]


def _use_substitution(batches):
    """The substitution loss under a Gumbel draw, with public uniforms on every rollout."""
    from reliquary import constants
    from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX, u_at
    from reliquary.validator import verifier

    constants.TRAINING_LOSS = "substitution"
    verifier.draw_settings = lambda: verifier.DrawSettings(
        draw=DRAW_GUMBEL_MAX, t=1.0, top_k=0, top_p=1.0,
        stochastic_threshold=0.99, boundary_epsilon=0.05,
    )
    for group in batches[0]:
        for index, r in enumerate(group.rollouts):
            n = len(r.commit["rollout"]["token_logprobs"])
            r._forced_draw_uniforms = [u_at("cd" * 32, group.prompt_idx, "c", index, j) for j in range(n)]


def _single_step(substitution=False):
    from reliquary.validator import training

    training.reset_training_state()
    training.KL_BETA = 0.0  # the run trains with KL 0 (decision N)
    model = tiny_qwen35()
    batches = _batches(model)
    if substitution:
        _use_substitution(batches)
    training.train_step(model, batches, ref_model=None)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _sharded_step(budget, substitution=False):
    from reliquary.trainer.distributed import init_distributed, shard_model
    from reliquary.validator import training

    ctx = init_distributed(device_type="cpu")
    training.reset_training_state()
    training.KL_BETA = 0.0  # the run trains with KL 0 (decision N)
    training.configure_distributed(ctx)
    training.MICROBATCH_MAX_PADDED_TOKENS = budget
    model = tiny_qwen35()
    batches = _batches(model)
    if substitution:
        _use_substitution(batches)
    model = shard_model(model, ctx, param_dtype=torch.float32)
    training.train_step(model, batches, ref_model=None)
    from reliquary.trainer.distributed import gather_full_state_dict
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return state if ctx.is_main else None


@pytest.mark.parametrize("budget", [32768, 100])
def test_sharded_step_matches_single_process(budget):
    want = _single_step()
    got = run_ranks(_sharded_step, 2, budget)[0]
    worst = max(float((got[k].float() - want[k].float()).abs().max()) for k in want)
    assert worst < 1e-5


def _substituted_single():
    return _single_step(substitution=True)


def test_sharded_substitution_step_matches_single_process():
    """The fused FSDP2 head picks v* on the gathered weight like the one-process chunk loop."""
    want = run_ranks(_substituted_single, 1)[0]
    plain = run_ranks(_single_step, 1)[0]
    got = run_ranks(_sharded_step, 2, 100, True)[0]
    worst = max(float((got[k].float() - want[k].float()).abs().max()) for k in want)
    assert worst < 1e-5
    assert any(not torch.equal(want[k], plain[k]) for k in want)  # v* moved the step


def _unequal(budget):
    # budget 80 packs one rollout per bin: 12 bins of uneven cost over 3
    # ranks, so ranks run different numbers of passes before padding.
    _sharded_step(budget)
    return True


def test_unequal_bins_do_not_deadlock():
    out = run_ranks(_unequal, 3, 80, timeout=300)
    assert all(out.values())


def _gate_global():
    from reliquary.trainer.distributed import init_distributed, shard_model
    from reliquary.validator import training

    ctx = init_distributed(device_type="cpu")
    training.reset_training_state()
    training.KL_BETA = 0.0  # the run trains with KL 0 (decision N)
    training.configure_distributed(ctx)
    training.MICROBATCH_MAX_PADDED_TOKENS = 100
    training.PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD = 0.5  # protocol >= 4 value
    model = tiny_qwen35()
    # Two of three groups shifted: more than half of the tokens are outside
    # the clip globally, but the bins carrying them land on some ranks only.
    batches = _batches(model, n_groups=3, outlier_group=0)
    for group in batches[0][1:2]:
        for r in group.rollouts:
            r.commit["rollout"]["token_logprobs"] = [v - 3.0 for v in r.commit["rollout"]["token_logprobs"]]
    model = shard_model(model, ctx, param_dtype=torch.float32)
    try:
        training.train_step(model, batches, ref_model=None)
    except training.TrainingStepSkipped as exc:
        return exc.reason
    return "stepped"


def test_gate_decision_is_global():
    out = run_ranks(_gate_global, 2)
    assert out[0] == out[1] == "policy_ratio_drift"
