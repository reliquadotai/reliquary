"""Exact state-invariant verifier for stateful tool episodes."""

from __future__ import annotations

from typing import Any

from reliquary.environment.agentic.types import (
    EpisodeTask,
    EpisodeTrace,
    RewardCheck,
    RewardReport,
    sha256_json,
)


def database_snapshot(connection: Any) -> dict[str, list[list[Any]]]:
    snapshot: dict[str, list[list[Any]]] = {}
    for table, order in (
        ("customers", "customer_id"),
        ("orders", "order_id"),
        ("refunds", "refund_id"),
        ("notes", "note_id"),
    ):
        rows = connection.execute(
            f"SELECT * FROM {table} ORDER BY {order}"  # noqa: S608 - static identifiers
        ).fetchall()
        snapshot[table] = [list(row) for row in rows]
    return snapshot


def grade_state(task: EpisodeTask, state: Any, trace: EpisodeTrace) -> RewardReport:
    expected = dict(task.private["expected"])
    customer_id = str(task.private["target_customer_id"])
    order_id = str(task.private["target_order_id"])
    family = str(task.metadata["family"])
    connection = state.connection
    checks: list[RewardCheck] = []

    note_rows = connection.execute(
        "SELECT note FROM notes WHERE customer_id = ? ORDER BY note_id",
        (customer_id,),
    ).fetchall()
    checks.append(
        RewardCheck(
            "required_support_note",
            (expected.get("note"),) in note_rows,
            weight=1.0,
        )
    )

    if family == "address_update":
        address = connection.execute(
            "SELECT shipping_address FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        checks.append(
            RewardCheck(
                "shipping_address_updated",
                bool(address and address[0] == expected["address"]),
                weight=2.0,
            )
        )
    elif family == "refund":
        refund = connection.execute(
            "SELECT refund_id, amount_cents FROM refunds WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        checks.append(
            RewardCheck(
                "refund_created",
                bool(
                    refund
                    and refund[0] == expected["refund_id"]
                    and refund[1] == expected["refund_cents"]
                ),
                weight=2.0,
            )
        )
    else:
        status = connection.execute(
            "SELECT status FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        checks.append(
            RewardCheck(
                "order_status_preserved",
                bool(status and status[0] == expected["status"]),
                weight=2.0,
            )
        )

    final_text = state.final_response or ""
    checks.append(
        RewardCheck(
            "final_response_identifiers",
            all(term.lower() in final_text.lower() for term in expected["final_terms"]),
            weight=1.0,
        )
    )
    checks.append(
        RewardCheck(
            "finished_explicitly",
            trace.termination_reason == "finished",
            weight=0.5,
        )
    )
    allowed = set(expected["allowed_mutations"])
    forbidden = [mutation for mutation in state.mutations if mutation[0] not in allowed]
    checks.append(
        RewardCheck(
            "no_forbidden_mutations",
            not forbidden,
            weight=2.0,
            detail="" if not forbidden else repr(forbidden[:3]),
        )
    )
    checks.append(
        RewardCheck(
            "no_invalid_tool_calls",
            state.invalid_actions == 0,
            weight=0.5,
        )
    )

    snapshot = database_snapshot(connection)
    fatal = bool(forbidden)
    return RewardReport.from_checks(
        checks,
        state_digest=sha256_json(snapshot),
        fatal=fatal,
    )
