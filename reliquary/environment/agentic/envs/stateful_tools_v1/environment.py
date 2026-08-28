"""Reference stateful tool environment backed by isolated in-memory SQLite."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sqlite3
from typing import Any, Mapping

from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeEvent,
    EpisodeTask,
    EpisodeTrace,
    ResetResult,
    RewardReport,
    StepResult,
    canonical_json,
)
from reliquary.environment.agentic.envs.stateful_tools_v1.tasks import (
    TASK_COUNT,
    build_task,
)
from reliquary.environment.agentic.envs.stateful_tools_v1.verifier import grade_state


@dataclass(slots=True)
class StatefulToolsState:
    connection: sqlite3.Connection
    seed: int
    final_response: str | None = None
    mutations: list[tuple[str, str]] = field(default_factory=list)
    invalid_actions: int = 0
    closed: bool = False


class StatefulToolsEnvironment:
    name = "reliquary_stateful_tools_v1"
    validator_authoritative_reward = True
    max_turns = 8

    def __len__(self) -> int:
        return TASK_COUNT

    def get_task(self, index: int) -> EpisodeTask:
        return build_task(index)

    def get_problem(self, index: int) -> dict[str, Any]:
        from reliquary.environment.agentic.compat import episode_problem

        return episode_problem(self, index)

    def reset(self, task: EpisodeTask, seed: int) -> ResetResult:
        connection = sqlite3.connect(":memory:")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL
            );
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                status TEXT NOT NULL,
                shipping_address TEXT NOT NULL,
                refundable_cents INTEGER NOT NULL,
                created_seq INTEGER NOT NULL
            );
            CREATE TABLE refunds (
                refund_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
                amount_cents INTEGER NOT NULL
            );
            CREATE TABLE notes (
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                note TEXT NOT NULL
            );
            """
        )
        rows = task.private["rows"]
        connection.executemany(
            "INSERT INTO customers(customer_id, email, name) VALUES (?, ?, ?)",
            rows["customers"],
        )
        connection.executemany(
            """INSERT INTO orders(
                order_id, customer_id, status, shipping_address,
                refundable_cents, created_seq
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            rows["orders"],
        )
        connection.commit()
        return ResetResult(state=StatefulToolsState(connection=connection, seed=int(seed)))

    @staticmethod
    def _event(tool: str, value: Mapping[str, Any]) -> tuple[EpisodeEvent, ...]:
        return (
            EpisodeEvent(role="tool", name=tool, content=canonical_json(dict(value))),
        )

    @staticmethod
    def _arguments(
        action: AssistantAction,
        *,
        required: tuple[str, ...],
    ) -> dict[str, Any]:
        arguments = dict(action.arguments)
        if set(arguments) != set(required):
            raise ValueError(f"expected arguments {required}")
        return arguments

    def step(
        self,
        task: EpisodeTask,
        state: StatefulToolsState,
        action: AssistantAction,
    ) -> StepResult:
        del task
        if state.closed:
            raise RuntimeError("episode state is closed")
        if action.kind == "final":
            state.final_response = action.content or ""
            return StepResult(
                state,
                self._event("final", {"accepted": True}),
                done=True,
                termination_reason="finished",
            )
        tool = str(action.tool)
        try:
            if tool == "search_customers":
                args = self._arguments(action, required=("email",))
                rows = state.connection.execute(
                    "SELECT customer_id, email, name FROM customers WHERE email = ?",
                    (str(args["email"]),),
                ).fetchall()
                value = {"customers": [dict(zip(("customer_id", "email", "name"), row)) for row in rows]}
            elif tool == "get_customer":
                args = self._arguments(action, required=("customer_id",))
                row = state.connection.execute(
                    "SELECT customer_id, email, name FROM customers WHERE customer_id = ?",
                    (str(args["customer_id"]),),
                ).fetchone()
                value = {"customer": None if row is None else dict(zip(("customer_id", "email", "name"), row))}
            elif tool == "list_orders":
                args = self._arguments(action, required=("customer_id",))
                rows = state.connection.execute(
                    """SELECT order_id, status, shipping_address, refundable_cents
                    FROM orders WHERE customer_id = ? ORDER BY created_seq DESC""",
                    (str(args["customer_id"]),),
                ).fetchall()
                keys = ("order_id", "status", "shipping_address", "refundable_cents")
                value = {"orders": [dict(zip(keys, row)) for row in rows]}
            elif tool == "get_order":
                args = self._arguments(action, required=("order_id",))
                row = state.connection.execute(
                    """SELECT order_id, customer_id, status, shipping_address,
                    refundable_cents FROM orders WHERE order_id = ?""",
                    (str(args["order_id"]),),
                ).fetchone()
                keys = ("order_id", "customer_id", "status", "shipping_address", "refundable_cents")
                value = {"order": None if row is None else dict(zip(keys, row))}
            elif tool == "update_shipping_address":
                args = self._arguments(action, required=("order_id", "address"))
                order_id = str(args["order_id"])
                address = str(args["address"])
                if not address or len(address) > 512:
                    raise ValueError("address must contain 1..512 characters")
                row = state.connection.execute(
                    "SELECT status FROM orders WHERE order_id = ?", (order_id,)
                ).fetchone()
                if row is None or row[0] != "pending":
                    raise ValueError("only pending orders may be updated")
                state.connection.execute(
                    "UPDATE orders SET shipping_address = ? WHERE order_id = ?",
                    (address, order_id),
                )
                state.connection.commit()
                state.mutations.append(("update_address", order_id))
                value = {"updated": True, "order_id": order_id}
            elif tool == "create_refund":
                args = self._arguments(action, required=("order_id", "amount_cents"))
                order_id = str(args["order_id"])
                amount = int(args["amount_cents"])
                row = state.connection.execute(
                    "SELECT status, refundable_cents FROM orders WHERE order_id = ?",
                    (order_id,),
                ).fetchone()
                if row is None or row[0] != "delivered":
                    raise ValueError("only delivered orders may be refunded")
                if amount != int(row[1]):
                    raise ValueError("refund must equal the refundable amount")
                refund_id = f"ref_{order_id[4:]}"
                state.connection.execute(
                    "INSERT INTO refunds(refund_id, order_id, amount_cents) VALUES (?, ?, ?)",
                    (refund_id, order_id, amount),
                )
                state.connection.commit()
                state.mutations.append(("create_refund", order_id))
                value = {"created": True, "refund_id": refund_id}
            elif tool == "add_support_note":
                args = self._arguments(action, required=("customer_id", "note"))
                customer_id = str(args["customer_id"])
                note = str(args["note"])
                if not note or len(note) > 1024:
                    raise ValueError("note must contain 1..1024 characters")
                state.connection.execute(
                    "INSERT INTO notes(customer_id, note) VALUES (?, ?)",
                    (customer_id, note),
                )
                state.connection.commit()
                state.mutations.append(("add_note", customer_id))
                value = {"created": True, "customer_id": customer_id}
            elif tool == "finish":
                args = self._arguments(action, required=("response",))
                state.final_response = str(args["response"])
                return StepResult(
                    state,
                    self._event("finish", {"accepted": True}),
                    done=True,
                    termination_reason="finished",
                )
            else:
                raise ValueError(f"unknown tool: {tool}")
            return StepResult(state, self._event(tool, {"ok": True, **value}))
        except (KeyError, TypeError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
            state.invalid_actions += 1
            return StepResult(
                state,
                self._event(tool, {"ok": False, "error": str(exc)[:512]}),
                done=True,
                termination_reason="invalid_action",
            )

    def grade(
        self,
        task: EpisodeTask,
        state: StatefulToolsState,
        trace: EpisodeTrace,
    ) -> RewardReport:
        return grade_state(task, state, trace)

    def close(self, state: StatefulToolsState) -> None:
        if not state.closed:
            state.connection.close()
            state.closed = True
