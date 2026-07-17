"""Unix-socket IPC client for the grader server.

Used by OpenCodeInstructEnvironment.compute_reward to dispatch structured case
evaluation requests. Frames JSON-lines over SOCK_STREAM. Candidate-caused
failures score zero; infrastructure failures raise ``GraderInfrastructureError``
so the auction cannot mistake a flaky grader for a hard negative.
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


class GraderInfrastructureError(RuntimeError):
    """The trusted grading service failed before producing a valid score."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(f"code grader infrastructure failure: {self.reason}")


_CANDIDATE_FAILURE_STATUSES = frozenset({
    "bad_output",
    "forbidden_import",
    "runtime_error",
    "tampered",
    "timeout",
})


class GraderClient:
    """Thin JSON-over-Unix-socket client.

    Stateless per-call (opens a new socket per evaluate). The grader
    server handles concurrent connections in its accept loop, so we
    don't need connection pooling on the client side.
    """

    def __init__(self, socket_path: str = GRADER_SOCKET_PATH) -> None:
        self.socket_path = socket_path

    def evaluate_cases(self, code: str, cases: list[dict[str, Any]], timeout_s: float) -> float:
        """Send (code, structured cases) and return passed/total in [0, 1].

        Candidate-caused failures return ``0.0``. Trusted-service failures
        raise :class:`GraderInfrastructureError`; callers must not turn those
        into negative training labels.
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
                raise GraderInfrastructureError("unreachable") from e

        status = response.get("status")
        if status in _CANDIDATE_FAILURE_STATUSES:
            return 0.0
        if status != "ok":
            raise GraderInfrastructureError(
                str(status) if status else "malformed_response"
            )
        try:
            passed = int(response["passed"])
            total = int(response["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GraderInfrastructureError("malformed_response") from exc
        if total <= 0 or passed < 0 or passed > total:
            raise GraderInfrastructureError("invalid_score")
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
