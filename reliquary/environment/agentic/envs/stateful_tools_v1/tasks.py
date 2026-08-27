"""Procedural, Reliquary-authored tasks for stateful tool training."""

from __future__ import annotations

import hashlib
import random

from reliquary.environment.agentic.types import AssistantAction, EpisodeTask
from reliquary.environment.agentic.envs.stateful_tools_v1.tools import TOOLS


GENERATOR_VERSION = "reliquary-stateful-tools-generator-v1"
TASK_COUNT = 1 << 31
FAMILIES = ("address_update", "refund", "support_note")


def _rng(index: int) -> random.Random:
    digest = hashlib.sha256(
        f"{GENERATOR_VERSION}:{int(index) % TASK_COUNT}".encode("ascii")
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _task_id(index: int) -> str:
    return hashlib.sha256(
        f"reliquary_stateful_tools_v1:{GENERATOR_VERSION}:{index}".encode("ascii")
    ).hexdigest()[:16]


def build_task(index: int) -> EpisodeTask:
    normalized = int(index) % TASK_COUNT
    rng = _rng(normalized)
    family = FAMILIES[normalized % len(FAMILIES)]
    customer_number = rng.randrange(100_000, 999_999)
    other_number = rng.randrange(100_000, 999_999)
    customer_id = f"cus_{customer_number}"
    other_customer_id = f"cus_{other_number}"
    order_id = f"ord_{rng.randrange(100_000, 999_999)}"
    other_order_id = f"ord_{rng.randrange(100_000, 999_999)}"
    email = f"customer{customer_number}@example.test"
    refund_cents = rng.randrange(12, 90) * 100
    note = f"verified-request-{rng.randrange(1000, 9999)}"
    new_address = (
        f"{rng.randrange(10, 999)} Cedar Street, "
        f"Unit {rng.randrange(1, 40)}, Testville"
    )
    refund_id = f"ref_{order_id[4:]}"

    rows = {
        "customers": [
            (customer_id, email, "Primary Customer"),
            (other_customer_id, f"other{other_number}@example.test", "Distractor"),
        ],
        "orders": [
            (
                order_id,
                customer_id,
                "delivered" if family == "refund" else "pending",
                "12 Old Road, Testville",
                refund_cents,
                2,
            ),
            (
                other_order_id,
                other_customer_id,
                "pending",
                "99 Unrelated Avenue, Testville",
                2500,
                1,
            ),
        ],
    }

    if family == "address_update":
        prompt = (
            f"Find the customer with email {email}. Update the shipping address "
            f"of their pending order to '{new_address}'. Add the support note "
            f"'{note}', then finish with the customer ID and order ID."
        )
        expected = {
            "address": new_address,
            "note": note,
            "final_terms": [customer_id, order_id],
            "allowed_mutations": ["update_address", "add_note"],
        }
        reference = [
            AssistantAction.tool_call("search_customers", email=email),
            AssistantAction.tool_call("list_orders", customer_id=customer_id),
            AssistantAction.tool_call(
                "update_shipping_address", order_id=order_id, address=new_address
            ),
            AssistantAction.tool_call(
                "add_support_note", customer_id=customer_id, note=note
            ),
            AssistantAction.tool_call(
                "finish", response=f"Updated {order_id} for {customer_id}."
            ),
        ]
    elif family == "refund":
        prompt = (
            f"Find the customer with email {email}. Refund exactly "
            f"{refund_cents} cents on their latest delivered order, add the "
            f"support note '{note}', and finish with the refund ID and order ID."
        )
        expected = {
            "refund_id": refund_id,
            "refund_cents": refund_cents,
            "note": note,
            "final_terms": [refund_id, order_id],
            "allowed_mutations": ["create_refund", "add_note"],
        }
        reference = [
            AssistantAction.tool_call("search_customers", email=email),
            AssistantAction.tool_call("list_orders", customer_id=customer_id),
            AssistantAction.tool_call("get_order", order_id=order_id),
            AssistantAction.tool_call(
                "create_refund", order_id=order_id, amount_cents=refund_cents
            ),
            AssistantAction.tool_call(
                "add_support_note", customer_id=customer_id, note=note
            ),
            AssistantAction.tool_call(
                "finish", response=f"Created {refund_id} for {order_id}."
            ),
        ]
    else:
        prompt = (
            f"Find the customer with email {email}, inspect their latest order, "
            f"add the exact support note '{note}', and finish with the customer "
            "ID, order ID, and current order status. Do not change the order."
        )
        expected = {
            "note": note,
            "status": "pending",
            "final_terms": [customer_id, order_id, "pending"],
            "allowed_mutations": ["add_note"],
        }
        reference = [
            AssistantAction.tool_call("search_customers", email=email),
            AssistantAction.tool_call("list_orders", customer_id=customer_id),
            AssistantAction.tool_call("get_order", order_id=order_id),
            AssistantAction.tool_call(
                "add_support_note", customer_id=customer_id, note=note
            ),
            AssistantAction.tool_call(
                "finish",
                response=f"{customer_id} {order_id} is pending.",
            ),
        ]

    return EpisodeTask(
        id=_task_id(normalized),
        prompt=prompt,
        tools=TOOLS,
        metadata={
            "family": family,
            "generator_version": GENERATOR_VERSION,
            "generator_index": normalized,
            "difficulty": 1 + normalized % 3,
        },
        private={
            "rows": rows,
            "target_customer_id": customer_id,
            "target_order_id": order_id,
            "other_customer_id": other_customer_id,
            "other_order_id": other_order_id,
            "expected": expected,
            "reference_actions": tuple(reference),
        },
    )
