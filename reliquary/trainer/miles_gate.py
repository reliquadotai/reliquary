"""``DrawGate`` state from the Miles gateway (``reliquary_gate`` loss outputs and ``optim_step`` metrics).

The gateway verifies every rollout in its forward and judges the groups at ``optim_step`` (backing the rejected ones
out before Adam). The trainer only reads the outcome back into its ``DrawGate`` so that ``AsyncTrainer._report`` and
the telemetry stay as they are. The group verdicts are the gateway's (``gate_group_rejected/<g>``): pay follows what
was actually trained, never a re-judgement here.
"""
from __future__ import annotations

from reliquary.trainer.miles_datums import _draw_positions

GATE_FIELDS = ("admitted", "n_stoch", "m_stoch", "n_drawn", "m_drawn", "max_gap", "last_gap", "n_substitute")


def _values(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def fill_gate(gate, slots: list[tuple[int, int]], outputs: list, rollouts: list) -> None:
    """Per forwarded rollout (``slots[i] = (group_index, slot)``, ``outputs[i]`` its loss outputs, ``rollouts[i]``
    the trainer's rollout object): tallies, admission, largest gap, final-EOS test and the re-graded substitutions
    (completion positions, as ``training._draw_suspects``)."""
    for (_, slot), output, rollout in zip(slots, outputs, rollouts, strict=True):
        verdict = dict(zip(GATE_FIELDS, _values(output["gate"]), strict=True))
        gate.tallies[slot] = [int(verdict[k]) for k in ("n_stoch", "m_stoch", "n_drawn", "m_drawn")]
        gate.forwarded[slot] = 1
        gate.admitted[slot] = int(bool(verdict["admitted"]))
        gate.max_gap[slot] = float(verdict["max_gap"])
        tokens, meta = rollout.commit["tokens"], rollout.commit["rollout"]
        last = float(verdict["last_gap"])
        ends_on_eos = bool(tokens) and int(tokens[-1]) in gate.eos_ids and "episode" not in meta
        gate.bad_terminal[slot] = int(ends_on_eos and last >= 0 and last > gate.terminal_gap)
        p = int(meta["prompt_length"])
        drawn = {p + q - 1 for _, q in _draw_positions(rollout)}
        gaps, picks, match = _values(output["gap"]), _values(output["vstar"]), _values(output["match"])
        gate.substitutions[slot] = [
            (j + 1 - p, int(picks[j])) for j in range(len(gaps)) if j + 1 >= p and (
                gaps[j] > gate.substitute_gap
                or (j in drawn and not match[j] and int(picks[j]) in gate.eos_ids))]


def apply_gateway_verdicts(gate, metrics: dict) -> None:
    """``gate.accepted`` from the gateway's ``optim_step`` metrics. A group with no forwarded rollout is not
    accepted (nothing verified); the gateway must have judged exactly the forwarded groups."""
    forwarded = [g for g, slots in enumerate(gate.groups) if any(gate.forwarded[s] for s in slots)]
    if forwarded and "gate_groups" not in metrics:
        raise RuntimeError("optim_step returned no gate verdicts")
    if int(metrics.get("gate_groups", 0)) != len(forwarded):
        raise RuntimeError(f"the gateway judged {metrics.get('gate_groups')} groups, {len(forwarded)} were forwarded")
    rejected = {int(k.rsplit("/", 1)[1]) for k, v in metrics.items() if k.startswith("gate_group_rejected/") and v}
    if len(rejected) != int(metrics.get("gate_groups_rejected", 0)) or not rejected <= set(forwarded):
        raise RuntimeError(f"inconsistent gateway verdicts: {sorted(rejected)}")
    alive = set(forwarded)
    gate.accepted = [g in alive and g not in rejected for g in range(len(gate.groups))]
    gate.undone = sum(1 for g in rejected if any(gate.admitted[s] for s in gate.groups[g]))


__all__ = ["GATE_FIELDS", "apply_gateway_verdicts", "fill_gate"]
