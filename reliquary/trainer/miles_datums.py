"""The async trainer's step as Tinker datums for the Miles gateway's ``reliquary_gate`` loss.

One datum per forwarded rollout (``training._plan_from_batches``: degenerate groups and masked rollouts are not
forwarded, as in ``train_step``). Tinker convention: ``model_input = tokens[:-1]``, target j predicts ``tokens[j+1]``.

* ``advantages``: the rollout's advantage times its plan scale (the token-level global mean) on trainable policy
  positions, 0 elsewhere (prompt, BFT force span, episode observations);
* ``draw_mask`` / ``draw_keys``: the positions the draw reads (``SubmittedGroup.model_positions`` minus the force
  span), keyed by the lease uniform of that model token (draw order);
* ``group_ids``: the group's index in the batch (``_async_group``): the gateway judges and backs out per group.
"""
from __future__ import annotations

from reliquary.environment.forced_sampling import gumbel_key
from reliquary.validator.training import _completion_keep_list, _is_masked_from_loss, _policy_token_positions


def _draw_positions(rollout) -> list[tuple[int, int]]:
    """(draw index t, completion-relative position) of every position the draw reads (``audits._drawn``)."""
    meta = rollout.commit["rollout"]
    p, n = int(meta["prompt_length"]), int(meta["completion_length"])
    spans = getattr(rollout, "_validated_assistant_spans", None)
    if spans is not None:
        positions = [q - p for s, e in spans for q in range(int(s), int(e))]
    else:
        positions = list(range(n))
    force = getattr(rollout, "_validated_force_span", None)
    start, end = (int(force[0]) - p, int(force[1]) - p) if force else (0, 0)
    return [(t, q) for t, q in enumerate(positions) if not start <= q < end]


def rollout_datum(rollout, advantage: float, group_index: int):
    import tinker.types as T

    tokens = [int(t) for t in rollout.commit["tokens"]]
    p = int(rollout.commit["rollout"]["prompt_length"])
    width = len(tokens) - 1
    keep = _completion_keep_list(rollout, p, len(tokens) - p)
    advantages = [0.0] * width
    for position in _policy_token_positions(rollout):
        if keep is None or keep[position - p]:
            advantages[position - 1] = float(advantage)
    mask, keys = [0] * width, [0] * width
    uniforms = rollout._forced_draw_uniforms
    for t, q in _draw_positions(rollout):
        mask[p + q - 1] = 1
        keys[p + q - 1] = gumbel_key(uniforms[t])

    def tensor(values, dtype):
        return T.TensorData(data=values, dtype=dtype, shape=[width])

    return T.Datum(model_input=T.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs={
        "target_tokens": tensor(tokens[1:], "int64"),
        "logprobs": tensor([0.0] * width, "float32"),
        "advantages": tensor(advantages, "float32"),
        "draw_keys": tensor(keys, "int64"),
        "draw_mask": tensor(mask, "float32"),
        "group_ids": tensor([int(group_index)] * width, "int64"),
    })


def step_datums(plan) -> tuple[list, list[tuple[int, int]]]:
    """Datums of every forwarded rollout of ``plan`` (``(group, advantages, scale)`` entries), and each datum's
    ``(group_index, slot)``."""
    datums, slots = [], []
    for group, advantages, scale in plan:
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            if _is_masked_from_loss(rollout) or not _draw_positions(rollout):
                continue                    # as train_step: never forwarded (a force span over every token)
            g = int(rollout._async_group)
            datums.append(rollout_datum(rollout, advantage * scale, g))
            slots.append((g, int(rollout._async_slot)))
    return datums, slots


__all__ = ["rollout_datum", "step_datums"]
