"""Unix-socket IPC client for the grader server.

Used by OpenCodeInstructEnvironment.compute_reward to dispatch
structured case evaluation requests. Frames JSON-lines over SOCK_STREAM.
Retries once on transient connection failures, then returns 0.0 — the
Environment Protocol forbids raising from compute_reward.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import uuid
from typing import Any

from reliquary.constants import GRADER_SOCKET_PATH

logger = logging.getLogger(__name__)

# Extra wall-clock budget on top of the eval timeout for socket setup
# + round-trip + the server's own dispatch overhead. The grader server
# enforces the inner per-eval timeout (GRADER_EVAL_TIMEOUT_SECONDS);
# this just keeps the outer socket from hanging forever if the server
# dies mid-response.
_SOCKET_TIMEOUT_HEADROOM_S = 5.0


class GraderUnavailableError(RuntimeError):
    """The trusted grader could not provide an authoritative result."""


class GraderClient:
    """Thin JSON-over-Unix-socket client.

    Stateless per-call (opens a new socket per evaluate). The grader
    server handles concurrent connections in its accept loop, so we
    don't need connection pooling on the client side.
    """

    def __init__(self, socket_path: str = GRADER_SOCKET_PATH) -> None:
        self.socket_path = socket_path

    def evaluate_cases(
        self, code: str, cases: list[dict[str, Any]], timeout_s: float
    ) -> float:
        """Send (code, structured cases) and return passed/total in [0, 1].

        Returns 0.0 if the grader is unreachable, the response is
        malformed, the worker timed out, the worker crashed, or
        total is zero. Never raises.
        """
        if not isinstance(cases, list) or not cases:
            return 0.0
        response: dict = {}
        req = {
            "req_id": uuid.uuid4().hex,
            "code": code,
            "cases": cases,
            "timeout_s": timeout_s,
        }
        # One retry with short backoff for transient failures (grader
        # restarting, accept queue full).
        for attempt in (1, 2):
            try:
                response = self._round_trip(req)
                break
            except (OSError, ConnectionError) as e:
                if attempt == 1:
                    logger.debug("grader_client: connect failed (%s), retrying", e)
                    time.sleep(0.1)
                    continue
                logger.warning("grader_client: unreachable after retry: %s", e)
                return 0.0

        if response.get("status") != "ok":
            return 0.0
        try:
            passed = int(response["passed"])
            total = int(response["total"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        if total <= 0:
            return 0.0
        return passed / total

    def evaluate_cases_strict(
        self,
        code: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
    ) -> float:
        """Evaluate cases, raising when grader authority is unavailable.

        Validator reward computation intentionally fails soft through
        :meth:`evaluate_cases`. Offline model evaluation must fail closed: a
        dead grader must never be published as a real all-zero checkpoint.
        """
        if not isinstance(cases, list) or not cases:
            raise ValueError("strict grading requires at least one case")
        response: dict = {}
        last_error: Exception | None = None
        request = {
            "req_id": uuid.uuid4().hex,
            "code": code,
            "cases": cases,
            "timeout_s": timeout_s,
        }
        for attempt in (1, 2):
            try:
                response = self._round_trip(request)
                break
            except (OSError, ConnectionError) as exc:
                last_error = exc
                if attempt == 1:
                    time.sleep(0.1)
        else:
            raise GraderUnavailableError("grader is unreachable") from last_error

        if response.get("status") != "ok":
            raise GraderUnavailableError(
                f"grader returned non-authoritative status {response.get('status')!r}"
            )
        try:
            passed = int(response["passed"])
            total = int(response["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GraderUnavailableError("grader response is malformed") from exc
        if total != len(cases) or not 0 <= passed <= total:
            raise GraderUnavailableError("grader response counts are inconsistent")
        return passed / total

    def _round_trip(self, req: dict) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(req["timeout_s"] + _SOCKET_TIMEOUT_HEADROOM_S)
            s.connect(self.socket_path)
            s.sendall(json.dumps(req).encode() + b"\n")
            buf = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            if not buf:
                return {}
            try:
                return json.loads(buf.split(b"\n", 1)[0])
            except json.JSONDecodeError:
                return {}
