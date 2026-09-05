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


def expected_database_snapshot(task: EpisodeTask) -> dict[str, list[list[Any]]]:
    """Build the only final database state accepted for this task.

    Checking an exact snapshot prevents a model from satisfying the target
    while also applying an otherwise-allowed mutation to a distractor row.
    """

    rows = task.private["rows"]
    expected = dict(task.private["expected"])
    family = str(task.metadata["family"])
    customer_id = str(task.private["target_customer_id"])
    order_id = str(task.private["target_order_id"])
    orders = [list(row) for row in rows["orders"]]
    if family == "address_update":
        for row in orders:
            if row[0] == order_id:
                row[3] = expected["address"]
                break
    refunds: list[list[Any]] = []
    if family == "refund":
        refunds.append([
            expected["refund_id"],
            order_id,
            expected["refund_cents"],
        ])
    return {
        "customers": sorted(
            [list(row) for row in rows["customers"]], key=lambda row: row[0]
        ),
        "orders": sorted(orders, key=lambda row: row[0]),
        "refunds": sorted(refunds, key=lambda row: row[0]),
        "notes": [[1, customer_id, expected["note"]]],
    }


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
    expected_mutations = {
        "address_update": [
            ("update_address", order_id),
            ("add_note", customer_id),
        ],
        "refund": [
            ("create_refund", order_id),
            ("add_note", customer_id),
        ],
        "support_note": [("add_note", customer_id)],
    }[family]
    mutations_exact = sorted(state.mutations) == sorted(expected_mutations)
    checks.append(
        RewardCheck(
            "mutations_exact",
            mutations_exact,
            weight=2.0,
            detail="" if mutations_exact else repr(state.mutations[:6]),
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
    expected_snapshot = expected_database_snapshot(task)
    state_exact = snapshot == expected_snapshot
    checks.append(
        RewardCheck(
            "database_state_exact",
            state_exact,
            weight=3.0,
        )
    )
    return RewardReport.from_checks(
        checks,
        state_digest=sha256_json(snapshot),
        fatal=not state_exact,
        binary=True,
    )
