"""The git patch inside a signed SWE episode's graded state.

A signed-sandbox SWE job's `final_diff` is the state the gateway graded (§5.D: its
sha256 is the final record's `state_sha256`). Through reliquary-sandbox's verifiers
bridge that state is not a diff: it is `reliquary-vf-bridge-state/1\\n`, a JSON header
line (`{"missing_required": [...], "present": [...]}`) and the state archive of the
task's artifact roots (`state_archive`: a tar whose members are `<i>/<path>` under
root i). reliquary-swe's finalize writes the agent's `git diff` to
`/logs/artifacts/patch.diff`, the file its grading applies.

`swe_patch` returns that file's text, so an export row's `final_diff` has the shape of
a replay SWE job's (a git patch). The archive is read with the sandbox's own vetting
(`rebuild_for_restore`, links contained as in the bridge's grading restore), never
extracted to disk.
"""

from __future__ import annotations

import io
import json
import tarfile

BRIDGE_STATE_MAGIC = b"reliquary-vf-bridge-state/1\n"
SWE_ARTIFACT_ROOT = "/logs/artifacts"
SWE_PATCH_FILE = "patch.diff"
MAX_HEADER_BYTES = 64 * 1024


class SignedStateUnreadable(ValueError):
    """The stored state is not a bridge state this export can read."""


def swe_patch(final_diff: str) -> str:
    """The git patch a signed SWE episode was graded on: `""` for an empty state (graded
    0 before extract), a state without the artifact root, or an absent or empty
    `patch.diff`; a state that is not a bridge state (an env whose state is its diff)
    unchanged. Raises SignedStateUnreadable (a ValueError) for anything else that
    is not a bridge state with a readable archive."""
    if final_diff == "":
        return ""
    state = final_diff.encode("utf-8")
    if not state.startswith(BRIDGE_STATE_MAGIC):
        return final_diff        # an env that publishes its diff as its state (not bridged)
    rest = state[len(BRIDGE_STATE_MAGIC):]
    end = rest.find(b"\n", 0, MAX_HEADER_BYTES)
    if end < 0:
        raise SignedStateUnreadable("the bridge state header is unterminated")
    try:
        header = json.loads(rest[:end])
    except ValueError as exc:
        raise SignedStateUnreadable("the bridge state header is not JSON") from exc
    present = header.get("present") if isinstance(header, dict) else None
    if not isinstance(present, list) or not all(isinstance(p, str) for p in present):
        raise SignedStateUnreadable("the bridge state header names no present roots")
    archive = rest[end + 1:]
    if SWE_ARTIFACT_ROOT not in present or not archive:
        return ""
    from reliquary_sandbox_service.episodes.state_archive import (
        StateUnreadable, rebuild_for_restore,
    )

    try:
        rebuilt = rebuild_for_restore(archive, list(present), contained_links=True)
    except StateUnreadable as exc:
        raise SignedStateUnreadable(f"the state archive is refused: {exc}") from exc
    clean = next(tar for root, _, tar in rebuilt if root == SWE_ARTIFACT_ROOT)
    try:
        with tarfile.open(fileobj=io.BytesIO(clean), mode="r:") as tar:
            try:
                member = tar.getmember(SWE_PATCH_FILE)
            except KeyError:
                return ""
            handle = tar.extractfile(member)        # follows an in-archive link
            data = b"" if handle is None else handle.read()
    except (tarfile.TarError, OSError) as exc:
        raise SignedStateUnreadable(f"the artifact archive is unreadable: {exc}") from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SignedStateUnreadable("patch.diff is not UTF-8") from exc


__all__ = ["BRIDGE_STATE_MAGIC", "SWE_ARTIFACT_ROOT", "SWE_PATCH_FILE", "SignedStateUnreadable",
           "swe_patch"]
