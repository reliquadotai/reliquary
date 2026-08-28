"""Small durable idempotency and monotonic-replay journal for signer operations."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


class JournalConflict(RuntimeError):
    pass


class JournalReplay(RuntimeError):
    pass


class JournalUncertain(RuntimeError):
    pass


@dataclass(frozen=True)
class Reservation:
    cached_response: dict | None = None


class SignerJournal:
    def __init__(self, path: str) -> None:
        journal_path = Path(path)
        journal_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(journal_path.parent, 0o700)
        self._connection = sqlite3.connect(
            journal_path,
            timeout=10.0,
            check_same_thread=False,
            isolation_level=None,
        )
        os.chmod(journal_path, 0o600)
        self._lock = threading.Lock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA busy_timeout=10000")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    response_json TEXT,
                    error_type TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cursors (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def reserve(
        self,
        *,
        operation_id: str,
        kind: str,
        payload_digest: str,
        cursor_name: str | None = None,
        cursor_value: int | None = None,
    ) -> Reservation:
        now = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT kind, payload_digest, state, response_json "
                    "FROM operations WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if row is not None:
                    existing_kind, existing_digest, state, response_json = row
                    if existing_kind != kind or existing_digest != payload_digest:
                        raise JournalConflict(
                            "operation ID was reused with different content"
                        )
                    if state == "completed":
                        self._connection.execute("COMMIT")
                        return Reservation(cached_response=json.loads(response_json))
                    if state == "uncertain":
                        raise JournalUncertain(
                            "operation outcome is uncertain and cannot be replayed"
                        )
                    if state == "pending":
                        raise JournalConflict("operation is already in progress")
                    # A definite pre-side-effect failure may retry the exact request.
                    self._connection.execute(
                        "UPDATE operations SET state='pending', error_type=NULL, updated_at=? "
                        "WHERE operation_id=?",
                        (now, operation_id),
                    )
                    self._connection.execute("COMMIT")
                    return Reservation()

                if cursor_name is not None:
                    if cursor_value is None:
                        raise ValueError("cursor_value is required with cursor_name")
                    cursor = self._connection.execute(
                        "SELECT value FROM cursors WHERE name=?", (cursor_name,)
                    ).fetchone()
                    if cursor is not None and int(cursor_value) <= int(cursor[0]):
                        raise JournalReplay(
                            f"{cursor_name} must increase beyond {int(cursor[0])}"
                        )
                    self._connection.execute(
                        "INSERT INTO cursors(name, value) VALUES(?, ?) "
                        "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                        (cursor_name, int(cursor_value)),
                    )

                self._connection.execute(
                    "INSERT INTO operations(operation_id, kind, payload_digest, state, "
                    "created_at, updated_at) VALUES(?, ?, ?, 'pending', ?, ?)",
                    (operation_id, kind, payload_digest, now, now),
                )
                self._connection.execute("COMMIT")
                return Reservation()
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def complete(self, operation_id: str, response: dict) -> None:
        encoded = json.dumps(
            response, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self._lock:
            updated = self._connection.execute(
                "UPDATE operations SET state='completed', response_json=?, "
                "error_type=NULL, updated_at=? WHERE operation_id=? AND state='pending'",
                (encoded, time.time(), operation_id),
            ).rowcount
            if updated != 1:
                raise JournalConflict("operation was not pending at completion")

    def fail_definite(self, operation_id: str, error_type: str) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE operations SET state='failed', error_type=?, updated_at=? "
                "WHERE operation_id=? AND state='pending'",
                (error_type[:100], time.time(), operation_id),
            )

    def mark_uncertain(self, operation_id: str, error_type: str) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE operations SET state='uncertain', error_type=?, updated_at=? "
                "WHERE operation_id=? AND state='pending'",
                (error_type[:100], time.time(), operation_id),
            )
