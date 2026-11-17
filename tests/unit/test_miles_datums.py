"""Tinker datums of the Miles trainer backend (reliquary/trainer/miles_datums.py)."""
from types import SimpleNamespace

import pytest

pytest.importorskip("tinker")

from reliquary.environment.forced_sampling import gumbel_key  # noqa: E402
from reliquary.trainer.async_trainer import training_group  # noqa: E402
from reliquary.trainer.miles_datums import step_datums  # noqa: E402
from reliquary.validator.training import _plan_from_batches  # noqa: E402


def _group(completions, rewards, *, prompt=(1, 2, 3), force_spans=None, spans=None, index=0, slot0=0):
    n = len(completions)
    spans = None if spans is None else tuple(tuple(tuple(s) for s in r) for r in spans)

    def model_tokens(i):
        return len(completions[i]) if spans is None else sum(e - s for s, e in spans[i])

    raw = SimpleNamespace(
        prompt_tokens=tuple(prompt), completions=tuple(tuple(c) for c in completions),
        force_spans=tuple(force_spans or [None] * n), spans=spans, truncated=(False,) * n, rewards=tuple(rewards),
        environment="math", prompt_idx=7, lease_id=f"lease-{index}", model_tokens=model_tokens)
    uniforms = [[(i + 1) / 10 + t / 1000 for t in range(model_tokens(i))] for i in range(n)]
    return training_group(raw, slot0=slot0, index=index, uniforms=uniforms), uniforms


def _inputs(datum):
    return {k: list(v.data) for k, v in datum.loss_fn_inputs.items()}


def test_single_turn_advantage_and_keys():
    group, uniforms = _group([[10, 11, 12, 13], [20, 21, 22, 23]], [1.0, 0.0])
    plan, _ = _plan_from_batches([[group]])
    datums, slots = step_datums(plan)
    assert slots == [(0, 0), (0, 1)]
    (_, advantages, scale), = plan
    d = _inputs(datums[0])
    assert list(datums[0].model_input.to_ints()) == [1, 2, 3, 10, 11, 12]
    assert d["target_tokens"] == [2, 3, 10, 11, 12, 13]
    assert d["advantages"] == pytest.approx([0, 0] + [advantages[0] * scale] * 4)
    assert scale == pytest.approx(1 / 8)
    assert d["draw_mask"] == [0, 0, 1, 1, 1, 1]
    assert d["draw_keys"] == [0, 0] + [gumbel_key(u) for u in uniforms[0]]
    assert d["logprobs"] == [0.0] * 6
    assert d["group_ids"] == [0] * 6


def test_force_span_gets_no_loss_no_draw():
    group, uniforms = _group([[10, 11, 12, 13], [20, 21, 22, 23]], [1.0, 0.0], force_spans=[(1, 3), None])
    plan, _ = _plan_from_batches([[group]])
    (_, advantages, scale), = plan
    assert scale == pytest.approx(1 / 6)                    # the 2 forced tokens leave the denominator
    d = _inputs(step_datums(plan)[0][0])
    a = advantages[0] * scale
    assert d["advantages"] == pytest.approx([0, 0, a, 0, 0, a])
    assert d["draw_mask"] == [0, 0, 1, 0, 0, 1]
    # draw t indexes model positions: completion positions 1-2 are forced, their uniforms are not read
    assert d["draw_keys"] == [0, 0, gumbel_key(uniforms[0][0]), 0, 0, gumbel_key(uniforms[0][3])]


def test_episode_masks_observations():
    completion = [10, 11, 90, 91, 12, 13]                    # 90, 91: a tool observation
    group, uniforms = _group([completion, completion], [1.0, 0.0], spans=[[(0, 2), (4, 6)]] * 2)
    plan, _ = _plan_from_batches([[group]])
    (_, advantages, scale), = plan
    d = _inputs(step_datums(plan)[0][0])
    a = advantages[0] * scale
    # targets 2..7 are completion positions 0..5
    assert d["advantages"] == pytest.approx([0, 0, a, a, 0, 0, a, a])
    assert d["draw_mask"] == [0, 0, 1, 1, 0, 0, 1, 1]
    keys = [gumbel_key(u) for u in uniforms[0]]
    assert d["draw_keys"] == [0, 0, keys[0], keys[1], 0, 0, keys[2], keys[3]]


def test_degenerate_group_not_in_datums():
    live, _ = _group([[10, 11], [20, 21]], [1.0, 0.0], index=0)
    flat, _ = _group([[30, 31], [40, 41]], [1.0, 1.0], index=1, slot0=2)
    plan, skipped = _plan_from_batches([[live, flat]])
    datums, slots = step_datums(plan)
    assert skipped == 1 and len(datums) == 2 and {g for g, _ in slots} == {0}


def test_group_ids_follow_the_batch_index():
    a, _ = _group([[10, 11], [20, 21]], [1.0, 0.0], index=0)
    b, _ = _group([[30, 31], [40, 41]], [0.0, 1.0], index=1, slot0=2)
    datums, slots = step_datums(_plan_from_batches([[a, b]])[0])
    assert slots == [(0, 0), (0, 1), (1, 2), (1, 3)]
    for datum, (g, _) in zip(datums, slots):
        assert set(_inputs(datum)["group_ids"]) == {g}


def test_rollout_forced_over_every_token_is_not_forwarded():
    group, _ = _group([[10, 11], [20, 21]], [1.0, 0.0], force_spans=[(0, 2), None])
    datums, slots = step_datums(_plan_from_batches([[group]])[0])
    assert slots == [(0, 1)]
