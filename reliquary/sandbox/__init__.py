"""Catalyst's side of signed-episode sandboxes (reliquary-sandbox).

Everything under `reliquary.sandbox` imports `reliquary_sandbox` (attest, observation,
episode client), which comes from the optional extras `reliquary[sandbox]` (validator)
and `reliquary[sandbox-miner]` (miner, with the verifiers bridge). Replay jobs never
import it. reliquary-sandbox is a PRIVATE repository: a third party cannot install the
extras until it is published, which is a user decision.
"""

from __future__ import annotations

from collections.abc import Mapping

SANDBOX_DISTRIBUTION = "reliquary-sandbox"
# The reliquary-sandbox commit this build verifies and drives. A job's contract pins
# its own (`episode.sandbox.sandbox_commit`); both must agree.
SANDBOX_COMMIT = "bfce1e8dec8fa365cbc3098e3bad4c51ff321d84"


class SandboxUnavailable(RuntimeError):
    """reliquary-sandbox is not installed in this environment."""


def _import_sandbox() -> None:
    import reliquary_sandbox.attest
    import reliquary_sandbox.observation  # noqa: F401


def require_sandbox() -> None:
    try:
        _import_sandbox()
    except ImportError as exc:
        raise SandboxUnavailable(
            "signed-sandbox jobs need reliquary-sandbox: install reliquary[sandbox] "
            "(validator) or reliquary[sandbox-miner] (miner); the repository is private"
        ) from exc


def _installed_commit() -> str | None:
    from reliquary.environment.agentic_swe import _dist_commit

    return _dist_commit(SANDBOX_DISTRIBUTION)


def sandbox_commit_refusal(pinned: str) -> str | None:
    """Why this process cannot serve a job pinning reliquary-sandbox at `pinned`, or None."""
    if pinned != SANDBOX_COMMIT:
        return (f"the job pins reliquary-sandbox {pinned}, this build supports "
                f"{SANDBOX_COMMIT}")
    installed = _installed_commit()
    if installed != SANDBOX_COMMIT:
        return f"reliquary-sandbox is installed at {installed}, this build needs {SANDBOX_COMMIT}"
    return None


def transcript_cap_refusal(caps) -> str | None:
    """Why a machine advertising these capacity-report `caps` cannot be placed, or None.

    The machine stops a transcript at its `max_transcript_bytes` (compact UTF-8 JSON,
    final record included); the submission body carries it under Catalyst's own
    `MAX_TRANSCRIPT_BYTES`, measured the same way. A machine whose cap is larger
    would sign honest episodes no miner can submit, so the fleet refuses it."""
    from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES

    value = caps.get("max_transcript_bytes") if isinstance(caps, Mapping) else None
    if type(value) is not int or value <= 0:
        return "the capacity report carries no max_transcript_bytes"
    if value > MAX_TRANSCRIPT_BYTES:
        return (f"the machine's max_transcript_bytes {value} is over the submission "
                f"cap {MAX_TRANSCRIPT_BYTES}")
    return None


__all__ = ["SANDBOX_COMMIT", "SANDBOX_DISTRIBUTION", "SandboxUnavailable",
           "require_sandbox", "sandbox_commit_refusal", "transcript_cap_refusal"]
