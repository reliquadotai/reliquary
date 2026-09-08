#!/usr/bin/env python3
"""Prepare a new three-environment run from a drained V5 checkpoint.

No remote writes. The new profile explicitly changes curriculum and resets
optimizer/LR warmup. This is preparation evidence, not GPU or model-quality
qualification. The existing Math+Code continuation has a separate command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliquary.constants import FILL_CLOSED_ENABLED, PROTOCOL_PROFILE_ID  # noqa: E402
from reliquary.protocol.profiles import resolve_protocol_profile  # noqa: E402
from reliquary.shared.checkpoint_identity import canonical_checkpoint_identity, require_checkpoint_number  # noqa: E402
from reliquary.shared.strict_json import strict_json_loads  # noqa: E402
from reliquary.validator.checkpoint_profile import CHECKPOINT_PROFILE_NAME, active_checkpoint_profile  # noqa: E402
from scripts.prepare_v5_fill_checkpoint import _json_bytes, source_checkpoint_number, validate_v5_source  # noqa: E402

PROFILE = "qwen3-4b-base-dapo-reliquary-v1"
TRANSITION = "reliquary_v1_bootstrap.json"


def prepare_bootstrap(source: dict, *, repo_id: str, revision: str, checkpoint_n: int,
                      last_archived_window: int, source_bucket: str, target_bucket: str,
                      lr_start_step: int = 0) -> dict:
    checkpoint_n, repo_id, revision = canonical_checkpoint_identity(checkpoint_n, repo_id, revision)
    validate_v5_source(source)
    require_checkpoint_number(last_archived_window, field="last archived window")
    require_checkpoint_number(lr_start_step, field="new run LR step")
    if not FILL_CLOSED_ENABLED or PROTOCOL_PROFILE_ID != PROFILE:
        raise ValueError("bootstrap requires the explicit three-environment V1 profile and fill capability")
    target = active_checkpoint_profile()
    old_run, new_run = source.get("training_run_id"), target["training_run_id"]
    if (not isinstance(old_run, str) or not old_run or old_run.strip() != old_run
            or not isinstance(new_run, str) or not new_run or new_run == "default" or new_run == old_run):
        raise ValueError("V1 requires a new explicit run identity and a canonical source run")
    for bucket in (source_bucket, target_bucket):
        if not isinstance(bucket, str) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
            raise ValueError("invalid storage bucket identity")
    if source_bucket == target_bucket:
        raise ValueError("RUN_ID is not storage isolation; use a distinct V1 bucket")
    cursor = require_checkpoint_number(source.get("trained_window_cursor"), field="source cursor")
    require_checkpoint_number(source.get("lr_schedule_step"), field="source LR step")
    if cursor != last_archived_window or source.get("journal_key_space", "raw") != "raw":
        raise ValueError("final V5 checkpoint must cover the exact drained raw-window boundary")
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW

    first_window = last_archived_window + 1
    target.update(trained_window_cursor=first_window * FILL_CLOSED_EMISSIONS_PER_WINDOW - 1,
                  journal_key_space="fill_closed", lr_schedule_step=lr_start_step)
    provenance = {
        "schema_version": 1, "kind": "new_v1_three_environment_run",
        "source_repo": repo_id, "source_revision": revision,
        "source_checkpoint_n": checkpoint_n, "source_profile": source,
        "target_checkpoint_n": checkpoint_n + 1, "target_profile": target,
        "first_v1_window": first_window,
        "weights": "inherited_unchanged_from_parent_commit",
        "optimizer_state": "recreated", "lr_schedule": "explicit_new_run_warmup",
        "environment_targets": {name: 16 for name in resolve_protocol_profile(PROFILE).environments},
    }
    return {
        "repo_id": repo_id, "parent_commit": revision,
        "commit_message": f"checkpoint {checkpoint_n + 1} (reliquary-v1-bootstrap)",
        "files": {CHECKPOINT_PROFILE_NAME: target, TRANSITION: provenance},
        "private_storage_migration": {
            "source_bucket": source_bucket, "target_bucket": target_bucket,
            "source_run": old_run, "target_run": new_run,
            "archive_boundary": last_archived_window, "scoring_history_windows": 216,
            "archive_history_min_windows": 300,
            "carry_prompt_and_content_cooldowns": ["openmathinstruct", "opencodeinstruct"],
            "initialize_empty_cooldowns": ["reliquary_logic_v2"],
            "exclude": ["reliquary/training/", "pending_training_payloads/", "control.json"],
        },
        "status": "prepared_only_requires_evaluation_target_oid_qualification_and_cutover",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--last-archived-window", type=int, required=True)
    parser.add_argument("--source-bucket", required=True)
    parser.add_argument("--target-bucket", required=True)
    parser.add_argument("--lr-start-step", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("output directory already exists")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    number = source_checkpoint_number(list(api.list_repo_commits(args.repo_id)), args.source_revision)
    source = strict_json_loads(Path(hf_hub_download(
        args.repo_id, CHECKPOINT_PROFILE_NAME, revision=args.source_revision,
    )).read_bytes())
    plan = prepare_bootstrap(source, repo_id=args.repo_id, revision=args.source_revision,
                             checkpoint_n=number, last_archived_window=args.last_archived_window,
                             source_bucket=args.source_bucket, target_bucket=args.target_bucket,
                             lr_start_step=args.lr_start_step)
    source_checkpoint_number(list(api.list_repo_commits(args.repo_id)), args.source_revision)
    args.output_dir.mkdir(mode=0o700, parents=True)
    for name, value in plan["files"].items():
        (args.output_dir / name).write_bytes(_json_bytes(value))
    (args.output_dir / "commit-plan.json").write_bytes(_json_bytes({
        **{key: value for key, value in plan.items() if key != "files"},
        "add_files": {name: hashlib.sha256(_json_bytes(value)).hexdigest() for name, value in plan["files"].items()},
    }))
    print(json.dumps({"prepared": str(args.output_dir), "published": False}))


if __name__ == "__main__":
    main()
