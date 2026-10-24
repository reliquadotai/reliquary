"""Resume-point resolution for the detached trainer.

The candidate manifest is a hint for WHICH checkpoint to load; the
checkpoint PROFILE inside the snapshot is authoritative for the cursor
and LR position once downloaded. First-run bootstrap requires an explicit
cursor — the trainer refuses to guess where the journal starts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Mapping

from reliquary.shared.checkpoint_identity import (
    canonical_checkpoint_identity,
    require_checkpoint_number,
    require_immutable_checkpoint_revision,
)
from reliquary.shared.strict_json import strict_json_loads
from reliquary.shared.checkpoint_namespace import (
    CheckpointNamespace, active_checkpoint_namespace,
)

logger = logging.getLogger(__name__)


def validate_scoped_resume_snapshot(
    snapshot_dir: str | Path,
    *,
    namespace: CheckpointNamespace,
    repo_id: str,
    revision: str,
    checkpoint_n: int,
    candidate: Mapping[str, object] | None,
) -> dict:
    """Bind exact-run weights and state before restoring a cursor or LR step."""
    from reliquary.trainer.publisher import PUBLICATION_RECEIPT
    from reliquary.validator.checkpoint_profile import (
        active_checkpoint_profile, validate_checkpoint_profile,
    )

    if not namespace.scoped:
        raise ValueError("scoped resume validation requires a task/run namespace")
    profile = validate_checkpoint_profile(
        snapshot_dir, required=True,
        expected=active_checkpoint_profile(namespace=namespace),
    )
    receipt = strict_json_loads((Path(snapshot_dir) / PUBLICATION_RECEIPT).read_bytes())
    if not isinstance(receipt, dict) or not isinstance(receipt.get("manifest"), dict):
        raise ValueError("scoped resume snapshot publication receipt is invalid")
    manifest = receipt["manifest"]
    namespace.require_identity(manifest)
    receipt_identity = canonical_checkpoint_identity(
        manifest.get("checkpoint_n"), manifest.get("repo_id"), revision,
        field="scoped resume snapshot",
    )
    if receipt_identity != (checkpoint_n, repo_id, revision):
        raise ValueError("scoped resume snapshot publication identity mismatch")
    if candidate is not None:
        namespace.require_identity(candidate)
        if candidate.get("revision") != revision:
            raise ValueError("scoped resume candidate changed during download")
        if manifest != {k: v for k, v in candidate.items() if k != "revision"}:
            raise ValueError("scoped resume snapshot receipt mismatch")
    bindings = {"trained_window_cursor": "trained_window_cursor", "journal_key_space": "journal_key_space",
                "generation_contract_sha256": "generation_contract_sha256", "profile_id": "protocol_profile_id",
                "protocol_version": "protocol_version"}
    for profile_key, manifest_key in bindings.items():
        if profile_key not in profile or profile[profile_key] != manifest.get(manifest_key):
            raise ValueError(f"scoped resume snapshot mismatch for {profile_key}")
    return profile


def _environment_string(
    env: Mapping[str, str],
    key: str,
) -> str:
    value = env.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be configured as a string")
    return value.strip()


def _environment_nonnegative_int(
    env: Mapping[str, str],
    key: str,
) -> int:
    raw = _environment_string(env, key)
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise ValueError(f"{key} must contain a base-10 integer") from exc
    return require_checkpoint_number(value, field=key)


def resolve_resume_point(
    fetch_fn: Callable[[str], bytes | None],
    *,
    env: Mapping[str, str],
    expected_identity: Mapping[str, object] | None = None,
    namespace: CheckpointNamespace | None = None,
) -> tuple[str | None, int, int]:
    """Return ``(revision, cursor, checkpoint_n)``: the checkpoint
    revision to load (None = bootstrap), the journal cursor to resume
    after, and the last published checkpoint number (0 = none yet —
    checkpoint numbering must never regress across restarts)."""
    namespace = namespace or active_checkpoint_namespace(env)
    raw = fetch_fn(namespace.candidate_manifest_key)
    if raw is not None:
        manifest = strict_json_loads(raw)
        if not isinstance(manifest, dict):
            raise ValueError("trainer resume manifest must be a JSON object")
        namespace.require_identity(manifest)
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in (expected_identity or {}).items()
            if manifest.get(key) != expected
        }
        if not mismatches:
            checkpoint_n, _, revision = canonical_checkpoint_identity(
                manifest.get("checkpoint_n"),
                manifest.get("repo_id"),
                manifest.get("revision"),
                field="trainer resume manifest checkpoint",
            )
            return (
                revision,
                require_checkpoint_number(
                    manifest.get("trained_window_cursor"),
                    field="trainer resume manifest cursor",
                ),
                checkpoint_n,
            )
        if namespace.scoped:
            raise ValueError("scoped trainer resume manifest identity mismatch")
        logger.warning(
            "candidate manifest belongs to another protocol/run (%s); "
            "using the explicit bootstrap configuration",
            ", ".join(sorted(mismatches)),
        )
    bootstrap = _environment_string(
        env,
        "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR",
    )
    if not bootstrap:
        logger.critical(
            "no candidate manifest in R2 and no "
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR set — refusing to guess "
            "the journal start"
        )
        raise SystemExit(2)
    # Mid-run bootstrap (shadow start, cutover from in-process training):
    # begin from the validator's last PUBLISHED checkpoint, not the base
    # model, so the shadow comparison and the cutover are seamless.
    raw_revision = _environment_string(
        env,
        "RELIQUARY_TRAINER_BOOTSTRAP_REVISION",
    )
    revision = (
        require_immutable_checkpoint_revision(
            raw_revision,
            field="trainer bootstrap revision",
        )
        if raw_revision
        else None
    )
    raw_n = _environment_string(env, "RELIQUARY_TRAINER_CHECKPOINT_N")
    checkpoint_n = (
        _environment_nonnegative_int(env, "RELIQUARY_TRAINER_CHECKPOINT_N")
        if raw_n
        else 0
    )
    return (
        revision,
        _environment_nonnegative_int(
            env,
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR",
        ),
        checkpoint_n,
    )
