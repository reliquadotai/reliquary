"""Validator failures that must cross the CLI process boundary."""


class FatalProofPlaneError(RuntimeError):
    """Proof infrastructure is unsafe to reuse without a process restart."""
