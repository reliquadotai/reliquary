"""Legacy JSON exports and an optional, pinned native Verifiers v1 bridge.

Only explicitly registered compatible Tasksets are qualified. This module does
not make arbitrary Prime environments deterministic or consensus-compatible.
Verifiers is imported lazily; normal core operation needs no Prime dependency.
"""

from __future__ import annotations

import importlib.metadata
import json

from reliquary.environment.agentic.types import (
    MAX_ACTION_BYTES,
    AssistantAction,
    EpisodeTask,
    EpisodeTrace,
    _load_json_object,
)

VERIFIERS_COMMIT = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"


def pinned_verifiers_v1():
    distribution = importlib.metadata.distribution("verifiers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != VERIFIERS_COMMIT:
        raise ValueError("native interop requires the pinned Verifiers source commit")
    import verifiers.v1 as vf

    return vf


def native_prime_v1_trace(task, *, completion: str | None = None,
                          episode: EpisodeTrace | None = None):
    """Construct a real Trace, including typed tool calls and tool observations.

    ``task`` is an actual Task loaded by its pinned wheel's native Taskset.
    The caller controls neither reward nor task data through the transcript.
    Reward is recomputed by that Task after a WireTrace serialization roundtrip.
    """
    vf = pinned_verifiers_v1()
    if not isinstance(task, vf.Task):
        raise TypeError("native interop requires a Verifiers Task")
    if (completion is None) == (episode is None):
        raise ValueError("provide exactly one completion or episode trace")
    nodes = []

    def append(message, *, sampled=False):
        nodes.append(vf.MessageNode(parent=len(nodes) - 1 if nodes else None,
                                    message=message, sampled=sampled))

    if completion is not None:
        if not isinstance(completion, str):
            raise TypeError("completion must be text")
        append(vf.AssistantMessage(content=completion), sampled=True)
    else:
        if episode.task_id != task.key:
            raise ValueError("episode and Verifiers task identity mismatch")
        observations = {}
        action_index = -1
        for event in episode.events:
            if event.role == "assistant":
                action_index += 1
            elif event.role == "tool":
                index = event.action_index if event.action_index is not None else action_index
                observations.setdefault(index, []).append(event)
        for index, action in enumerate(episode.actions):
            wire = action.to_wire()
            if "tool" in wire:
                call_id = f"reliquary-{index}"
                append(vf.AssistantMessage(tool_calls=[vf.ToolCall(
                    id=call_id, name=wire["tool"],
                    arguments=json.dumps(wire["arguments"], sort_keys=True),
                )]), sampled=True)
                for event in observations.get(index, ()):
                    append(vf.ToolMessage(tool_call_id=call_id,
                                          name=event.name or wire["tool"],
                                          content=event.content))
            else:
                append(vf.AssistantMessage(content=wire["final"]), sampled=True)
    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data,
                          key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()), nodes=nodes, state=vf.State(),
    )


def actions_from_prime_v1_trace(trace) -> tuple[AssistantAction, ...]:
    """Recover bounded Episode actions from a native Trace/WireTrace.

    Use sampled assistant nodes on the final branch, as the pinned native
    Taskset does. Tool observations and trace rewards are ignored: the Reliquary
    environment must replay actions to establish an authoritative outcome.
    """
    vf = pinned_verifiers_v1()
    if not isinstance(trace, (vf.Trace, vf.WireTrace)):
        raise TypeError("native interop requires a Verifiers Trace or WireTrace")
    if trace.version != 1:
        raise ValueError("unsupported Verifiers trace version")
    # Validate before asking Verifiers to walk parents: malformed wire graphs
    # can otherwise loop forever or resolve negative Python list indices.
    for index, node in enumerate(trace.nodes):
        if node.parent is not None and not 0 <= node.parent < index:
            raise ValueError("Verifiers trace parents must precede their children")
    branches = trace.branches
    messages = [node.message for node in branches[-1].nodes
                if node.sampled and isinstance(node.message, vf.AssistantMessage)] if branches else []
    actions = []
    call_ids = set()
    for index, message in enumerate(messages):
        if message.tool_calls:
            if message.content or len(message.tool_calls) != 1:
                raise ValueError("Episode v1 requires one tool call per assistant turn")
            call = message.tool_calls[0]
            if call.type != "function" or not call.id or call.id in call_ids:
                raise ValueError("Episode tool calls require unique nonempty function IDs")
            call_ids.add(call.id)
            # This is structured API JSON, not free-form model reasoning.
            # Never extract a later action-shaped object from malformed input.
            arguments = _load_json_object(call.arguments, max_bytes=MAX_ACTION_BYTES)
            actions.append(AssistantAction.from_wire(
                {"tool": call.name, "arguments": arguments}
            ))
        else:
            if index != len(messages) - 1 or not isinstance(message.content, str):
                raise ValueError("Episode final text must be the last sampled turn")
            actions.append(AssistantAction.final(message.content))
    return tuple(actions)


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
