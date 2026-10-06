"""Catalyst's side of signed-episode sandboxes (reliquary-sandbox).

Everything under `reliquary.sandbox` imports `reliquary_sandbox` (attest, observation,
episode client), which comes from the optional extras `reliquary[sandbox]` (validator)
and `reliquary[sandbox-miner]` (miner, with the verifiers bridge). Replay jobs never
import it. reliquary-sandbox is a PRIVATE repository: a third party cannot install the
extras until it is published, which is a user decision.
"""

from __future__ import annotations

SANDBOX_DISTRIBUTION = "reliquary-sandbox"
# The reliquary-sandbox commit this build verifies and drives. A job's contract pins
# its own (`episode.sandbox.sandbox_commit`); both must agree.
SANDBOX_COMMIT = "e7c765c596d0017fe9698916efd73eadd7f1b010"


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


__all__ = ["SANDBOX_COMMIT", "SANDBOX_DISTRIBUTION", "SandboxUnavailable",
           "require_sandbox", "sandbox_commit_refusal"]
