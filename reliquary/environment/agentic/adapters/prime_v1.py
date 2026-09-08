"""Export helpers for Prime Verifiers v1's task/trace-oriented data model.

The adapter intentionally returns JSON-native dictionaries and does not import
``verifiers``. A pinned deployment can construct its exact Task/Trace classes
from these values without making that external package part of consensus.
"""

from __future__ import annotations

from reliquary.environment.agentic.types import EpisodeTask, EpisodeTrace


def export_prime_v1_task(task: EpisodeTask) -> dict:
    return {
        "id": task.id,
        "prompt": task.prompt,
        "tools": [tool.to_wire() for tool in task.tools],
        "info": dict(task.metadata),
    }


def export_prime_v1_trace(trace: EpisodeTrace) -> dict:
    assistant_messages = []
    tool_messages = []
    for event in trace.events:
        value = event.to_wire()
        if event.role == "assistant":
            assistant_messages.append(value)
        elif event.role == "tool":
            tool_messages.append(value)
    return {
        "task": {"id": trace.task_id},
        "assistant_messages": assistant_messages,
        "tool_messages": tool_messages,
        "state": {
            "environment": trace.environment,
            "seed": trace.seed,
            "state_digest": (
                trace.reward.state_digest if trace.reward is not None else None
            ),
        },
        "info": {
            "schema": trace.schema,
            "trace_digest": trace.trace_digest,
            "assistant_spans": [list(span) for span in trace.assistant_spans],
        },
        "rewards": (
            {} if trace.reward is None else {"reliquary": trace.reward.reward}
        ),
        "metrics": {
            "turns": len(trace.actions),
            "success": bool(trace.reward and trace.reward.success),
        },
        "stop_condition": trace.termination_reason,
    }
