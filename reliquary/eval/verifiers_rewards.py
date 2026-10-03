"""Deterministic rewards that replace a Verifiers task's judged one.

Plugged through the taskset args of a set, e.g. GPQA's
``{"task": {"rewards": {"correct": {"fn": "reliquary.eval.verifiers_rewards:gpqa_letter"}}}}``:
the plugged function replaces the task's reward of the same name, so the
upstream judge fallback is never reached.
"""

from __future__ import annotations


async def gpqa_letter(task, trace) -> float:
    """GPQA on the upstream letter extractor alone: an answer it cannot read is 0."""
    from gpqa.mcq import extract_mcq_answer

    return 1.0 if extract_mcq_answer(trace.last_reply) == task.answer else 0.0


__all__ = ["gpqa_letter"]
