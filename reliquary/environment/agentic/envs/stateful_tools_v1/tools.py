"""Public tool schemas for the stateful SQLite reference environment."""

from __future__ import annotations

from reliquary.environment.agentic.types import ToolSpec


def _object(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS = (
    ToolSpec(
        "search_customers",
        "Find customers whose email exactly matches the supplied email.",
        _object({"email": {"type": "string"}}, ["email"]),
    ),
    ToolSpec(
        "get_customer",
        "Get one customer by customer_id.",
        _object({"customer_id": {"type": "string"}}, ["customer_id"]),
    ),
    ToolSpec(
        "list_orders",
        "List a customer's orders, newest first.",
        _object({"customer_id": {"type": "string"}}, ["customer_id"]),
    ),
    ToolSpec(
        "get_order",
        "Get one order and its refundable amount.",
        _object({"order_id": {"type": "string"}}, ["order_id"]),
    ),
    ToolSpec(
        "update_shipping_address",
        "Update the shipping address for one pending order.",
        _object(
            {
                "order_id": {"type": "string"},
                "address": {"type": "string"},
            },
            ["order_id", "address"],
        ),
    ),
    ToolSpec(
        "create_refund",
        "Create a refund for a delivered order. Duplicate refunds are rejected.",
        _object(
            {
                "order_id": {"type": "string"},
                "amount_cents": {"type": "integer", "minimum": 1},
            },
            ["order_id", "amount_cents"],
        ),
    ),
    ToolSpec(
        "add_support_note",
        "Add an audit-visible note to a customer account.",
        _object(
            {
                "customer_id": {"type": "string"},
                "note": {"type": "string"},
            },
            ["customer_id", "note"],
        ),
    ),
    ToolSpec(
        "finish",
        "Finish the task and provide the final response to the user.",
        _object({"response": {"type": "string"}}, ["response"]),
    ),
)
