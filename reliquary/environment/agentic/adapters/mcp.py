"""Dependency-free MCP tool-shape bridge.

This converts public MCP ``tools/list`` and ``tools/call`` JSON objects only.
MCP transport/session state never becomes Reliquary consensus state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from reliquary.environment.agentic.types import AssistantAction, ToolSpec


def task_tools_from_mcp(tools: Iterable[Mapping[str, Any]]) -> tuple[ToolSpec, ...]:
    result = []
    for tool in tools:
        schema = tool.get("inputSchema", {"type": "object", "properties": {}})
        if not isinstance(schema, Mapping):
            raise ValueError("MCP inputSchema must be an object")
        result.append(ToolSpec(
            name=str(tool["name"]),
            description=str(tool.get("description", "")),
            parameters=dict(schema),
        ))
    return tuple(result)


def action_from_mcp_call(call: Mapping[str, Any]) -> AssistantAction:
    name = call.get("name")
    arguments = call.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("invalid MCP tool call")
    return AssistantAction(kind="tool", tool=name, arguments=dict(arguments))
