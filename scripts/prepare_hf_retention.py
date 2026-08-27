#!/usr/bin/env python3
"""Audit an existing HF checkpoint run and protect its first publications.

The default is read-only.  ``--apply`` creates one branch pointing at the
50th (configurable) run-local checkpoint; it never squashes, deletes, or moves
``main``.  The reported publication sequence must be supplied to the detached
trainer when bounded retention is first enabled for a legacy manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.errors import EntryNotFoundError  # noqa: E402

from reliquary.trainer.retention import CheckpointRetentionPolicy  # noqa: E402


CHECKPOINT_TITLE = re.compile(r"^checkpoint\s+(\d+)(?:\s|$)", re.IGNORECASE)
CHECKPOINT_PROFILE_NAME = "reliquary_protocol_profile.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--keep-initial", type=int, default=50)
    parser.add_argument(
        "--start-revision",
        help=(
            "Optional first trainer-publication SHA of this run. By default "
            "the boundary is inferred from checkpoint profiles."
        ),
    )
    parser.add_argument(
        "--expected-head",
        help="Required with --apply; prevents creating a branch during a race.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create the protected first-history branch (non-destructive).",
    )
    return parser


def _checkpoint_profile(api: HfApi, repo_id: str, revision: str) -> dict | None:
    try:
        path = api.hf_hub_download(
            repo_id=repo_id,
            filename=CHECKPOINT_PROFILE_NAME,
            revision=revision,
        )
    except EntryNotFoundError:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _checkpoint_rows(commits: list) -> list[dict]:
    checkpoints = []
    for commit in commits:
        match = CHECKPOINT_TITLE.match(str(commit.title or ""))
        if match:
            checkpoints.append({
                "checkpoint_n": int(match.group(1)),
                "revision": str(commit.commit_id),
                "title": str(commit.title),
                "created_at": commit.created_at.isoformat(),
            })
    return checkpoints


def _infer_run_checkpoints(
    api: HfApi, repo_id: str, run_id: str, checkpoints: list[dict],
) -> list[dict]:
    """Walk backward over profile-stamped trainer publications for one run."""

    selected_newest = []
    for checkpoint in reversed(checkpoints):
        profile = _checkpoint_profile(
            api, repo_id, str(checkpoint["revision"]),
        )
        matches = (
            profile is not None
            and profile.get("training_run_id") == run_id
            and profile.get("trained_window_cursor") is not None
        )
        if matches:
            selected_newest.append(checkpoint)
            continue
        if selected_newest:
            break
    return list(reversed(selected_newest))


def main() -> int:
    args = _parser().parse_args()
    if args.keep_initial <= 0:
        raise SystemExit("--keep-initial must be positive")
    if args.apply and not args.expected_head:
        raise SystemExit("--apply requires --expected-head")

    api = HfApi()
    head = str(api.model_info(args.repo_id).sha)
    commits = list(reversed(list(api.list_repo_commits(repo_id=args.repo_id))))
    all_checkpoints = _checkpoint_rows(commits)
    if args.start_revision:
        indexes = [
            index for index, commit in enumerate(commits)
            if commit.commit_id == args.start_revision
        ]
        if not indexes:
            raise SystemExit("--start-revision is not in repository history")
        checkpoints = _checkpoint_rows(commits[indexes[0]:])
        if checkpoints:
            first_profile = _checkpoint_profile(
                api, args.repo_id, checkpoints[0]["revision"],
            )
            if (
                first_profile is None
                or first_profile.get("training_run_id") != args.run_id
                or first_profile.get("trained_window_cursor") is None
            ):
                raise SystemExit(
                    "--start-revision does not select a trainer publication "
                    "with the requested run id and trained-window cursor"
                )
        boundary_source = "explicit_start_revision"
    else:
        checkpoints = _infer_run_checkpoints(
            api, args.repo_id, args.run_id, all_checkpoints,
        )
        boundary_source = "checkpoint_profiles"

    if not checkpoints:
        raise SystemExit(
            "no profile-stamped trainer publications found for --run-id"
        )
    latest_profile = _checkpoint_profile(
        api, args.repo_id, checkpoints[-1]["revision"],
    )
    if (
        latest_profile is None
        or latest_profile.get("training_run_id") != args.run_id
    ):
        raise SystemExit("repository HEAD does not belong to --run-id")

    policy = CheckpointRetentionPolicy(
        enabled=True,
        keep_initial=args.keep_initial,
    )
    branch = policy.first_history_branch(args.run_id)
    retained = (
        checkpoints[args.keep_initial - 1]
        if len(checkpoints) >= args.keep_initial else None
    )
    refs = api.list_repo_refs(repo_id=args.repo_id)
    branches = {
        str(item.name): str(item.target_commit)
        for item in getattr(refs, "branches", [])
    }
    target = str(retained["revision"]) if retained is not None else None
    branch_target = branches.get(branch)
    if branch_target is None:
        branch_state = "missing"
    elif branch_target == target:
        branch_state = "protected_history"
    else:
        branch_commits = api.list_repo_commits(
            repo_id=args.repo_id,
            revision=branch,
        )
        if (
            len(branch_commits) == 1
            and str(branch_commits[0].title or "")
            == policy.run_start_root_message()
        ):
            branch_state = "r2_archived_root"
        else:
            branch_state = "conflict"

    first_selected_revision = str(checkpoints[0]["revision"])
    first_selected_index = next(
        index for index, row in enumerate(all_checkpoints)
        if str(row["revision"]) == first_selected_revision
    )

    result = {
        "repo_id": args.repo_id,
        "run_id": args.run_id,
        "head": head,
        "boundary_source": boundary_source,
        "publication_seq": len(checkpoints),
        "first_checkpoint": checkpoints[0] if checkpoints else None,
        "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        "keep_initial": args.keep_initial,
        "preceding_checkpoint_count": first_selected_index,
        "legacy_r2_backfill_required": (
            branch_state != "r2_archived_root"
        ),
        "first_history_branch": branch,
        "first_history_revision": (
            retained["revision"] if retained is not None else None
        ),
        "branch_already_exists": branch in branches,
        "branch_state": branch_state,
        "branch_target": branch_target,
        "applied": False,
    }

    if args.apply:
        if head != args.expected_head:
            raise SystemExit(
                f"HF HEAD moved: expected {args.expected_head}, observed {head}"
            )
        if retained is None:
            raise SystemExit(
                f"run has only {len(checkpoints)} checkpoints; nothing to "
                f"protect until publication {args.keep_initial}"
            )
        target = str(retained["revision"])
        if branch_state == "conflict":
            raise SystemExit(
                f"branch {branch!r} has an unexpected target {branch_target}"
            )
        if branch_state == "missing":
            api.create_branch(
                repo_id=args.repo_id,
                branch=branch,
                revision=target,
                exist_ok=False,
            )
            result["branch_already_exists"] = True
            result["branch_state"] = "protected_history"
            result["branch_target"] = target
        result["applied"] = True

    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "\nRELIQUARY_TRAINER_PUBLICATION_SEQ="
        f"{result['publication_seq']}"
    )
    if not args.apply and retained is not None and branch_state == "missing":
        print(
            "# Re-run with --apply --expected-head " + head
            + " to create the protected first-history branch."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
