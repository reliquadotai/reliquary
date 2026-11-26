"""Host side of a reward worker: the sealed container, and a JSON-lines client over its stdio.

Infrastructure failures (the container would not start, died, hung, or answered garbage) raise
``RewardInfrastructureError``: a caller never reads them as a zero score, and never faults a miner for them.
"""
from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass

_DIGEST_RE = re.compile(r"\A[a-z0-9][a-z0-9./:_-]{0,254}@sha256:[0-9a-f]{64}\Z")


class RewardInfrastructureError(RuntimeError):
    """The reward worker failed before answering; the request may be retried, never scored."""


@dataclass(frozen=True)
class Limits:
    cpus: float = 1.0
    memory_mb: int = 2048
    pids: int = 256
    tmp_mb: int = 256


def sealed_argv(image: str, *, limits: Limits = Limits(), runtime: str = "runsc",
                docker: str = "docker") -> list[str]:
    """``docker run`` for a reward worker: gVisor, no network, read-only root, no capability, no
    privilege escalation, nobody user, bounded CPU/memory/pids/tmp, no mount and no environment of ours.
    The image is named by digest only."""
    if not _DIGEST_RE.match(image):
        raise ValueError("a customer env image is named by repository@sha256:<digest>")
    return [docker, "run", "--rm", "-i", "--runtime", runtime, "--network", "none", "--read-only",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,size={limits.tmp_mb}m", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--user", "65534:65534",
            "--cpus", str(limits.cpus), "--memory", f"{limits.memory_mb}m",
            "--memory-swap", f"{limits.memory_mb}m", "--pids-limit", str(limits.pids),
            "--pull", "never", "--log-driver", "none", image]


class WorkerProcess:
    """One reward worker. Requests are serialised; a hung or dead worker is killed and restarted on the
    next request."""

    def __init__(self, argv: Sequence[str], *, start_timeout_s: float = 300.0) -> None:
        self._argv = list(argv)
        self._start_timeout = start_timeout_s
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue | None = None
        self._lock = threading.Lock()
        self.provenance: dict | None = None

    def _spawn(self) -> None:
        try:
            proc = subprocess.Popen(self._argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, bufsize=0)
        except OSError as exc:
            raise RewardInfrastructureError(f"start_failed: {exc}") from exc
        lines: queue.Queue = queue.Queue()

        def pump() -> None:
            for line in iter(proc.stdout.readline, b""):
                lines.put(line)
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        self._proc, self._lines = proc, lines
        hello = self._exchange({"op": "hello"}, self._start_timeout)
        if not hello.get("ok"):
            self.kill()
            raise RewardInfrastructureError(f"start_failed: {hello.get('error')}")
        self.provenance = hello.get("provenance")

    def _exchange(self, request: dict, timeout_s: float) -> dict:
        try:
            self._proc.stdin.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.kill()
            raise RewardInfrastructureError(f"worker_died: {exc}") from exc
        try:
            line = self._lines.get(timeout=timeout_s)
        except queue.Empty:
            self.kill()
            raise RewardInfrastructureError(f"worker_hung: no answer in {timeout_s:.0f} s") from None
        if line is None:
            self.kill()
            raise RewardInfrastructureError("worker_died: its output closed")
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            self.kill()
            raise RewardInfrastructureError("worker_garbled: an answer that is not JSON") from None
        if not isinstance(response, dict):
            self.kill()
            raise RewardInfrastructureError("worker_garbled: an answer that is not an object")
        return response

    def request(self, request: dict, timeout_s: float) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()
            return self._exchange(request, timeout_s)

    def kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    close = kill


@dataclass(frozen=True)
class ScoreResult:
    score: float | None
    error: str | None = None


class RewardClient:
    """Scores and freezes through one worker. ``score`` answers one result per item: a score, or the item's
    error (``source_drift``, ``timeout``, ``grader_error: ...``) — the customer's code failing on that item."""

    def __init__(self, worker: WorkerProcess, *, per_item_timeout_s: float = 60.0) -> None:
        self._worker = worker
        self._per_item = per_item_timeout_s

    def _ok(self, response: dict) -> dict:
        if not response.get("ok"):
            raise RewardInfrastructureError(f"worker_refused: {response.get('error')}"[:500])
        return response

    def score(self, items: Sequence[dict]) -> list[ScoreResult]:
        from reliquary.customer_env.worker import MAX_SCORE_ITEMS

        out: list[ScoreResult] = []
        for start in range(0, len(items), MAX_SCORE_ITEMS):
            chunk = list(items[start:start + MAX_SCORE_ITEMS])
            response = self._ok(self._worker.request({"op": "score", "items": chunk},
                                                     self._per_item * len(chunk) + 30.0))
            results = response.get("results")
            if not isinstance(results, list) or len(results) != len(chunk):
                raise RewardInfrastructureError("worker_garbled: one result per item expected")
            for result in results:
                score = result.get("score") if isinstance(result, dict) else None
                if isinstance(score, (int, float)) and not isinstance(score, bool) and 0.0 <= score <= 1.0:
                    out.append(ScoreResult(float(score)))
                elif isinstance(result, dict) and isinstance(result.get("error"), str):
                    out.append(ScoreResult(None, result["error"][:300]))
                else:
                    raise RewardInfrastructureError("worker_garbled: a result is a score or an error")
        return out

    def freeze_all(self, *, page: int = 1000, max_rows: int = 1_000_000) -> tuple[list[dict], list[dict]]:
        rows, refused, offset = [], [], 0
        while True:
            response = self._ok(self._worker.request({"op": "freeze", "offset": offset, "limit": page}, 600.0))
            rows += response["rows"]
            refused += response["refused"]
            offset += page
            if offset >= int(response["total"]) or offset >= max_rows:
                return rows, refused

    @property
    def provenance(self) -> dict | None:
        return self._worker.provenance

    def close(self) -> None:
        self._worker.close()


__all__ = ["Limits", "RewardClient", "RewardInfrastructureError", "ScoreResult", "WorkerProcess", "sealed_argv"]
