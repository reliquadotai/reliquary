#!/usr/bin/env python3
"""Prepare a V5 -> fill-closed checkpoint commit; never publish or load weights.

Run after the final V5 checkpoint covers every archived window. Publication
must add only the two prepared JSON files at the exact recorded parent commit.
The validator, trainer and miners still require a coordinated V6 activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliquary.constants import FILL_CLOSED_ENABLED  # noqa: E402
from reliquary.infrastructure.training_payload_queue import encoded_window_journal_key  # noqa: E402
from reliquary.protocol.profiles import resolve_protocol_profile  # noqa: E402
from reliquary.shared.checkpoint_identity import (  # noqa: E402
    canonical_checkpoint_identity,
    require_checkpoint_number,
)
from reliquary.shared.strict_json import strict_json_loads  # noqa: E402
from reliquary.validator.checkpoint_profile import (  # noqa: E402
    CHECKPOINT_PROFILE_NAME,
    active_checkpoint_profile,
)
from reliquary.validator.resume import checkpoint_n_from_commit_title  # noqa: E402

TRANSITION_NAME = "reliquary_protocol_transition.json"


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def prepare_transition(source: dict, *, repo_id: str, revision: str,
                       checkpoint_n: int, last_archived_window: int) -> dict:
    """Keep weights/run/LR, advance identity, and start at the first V6 batch."""
    checkpoint_n, repo_id, revision = canonical_checkpoint_identity(
        checkpoint_n, repo_id, revision,
    )
    require_checkpoint_number(last_archived_window, field="last archived window")
    if not FILL_CLOSED_ENABLED:
        raise ValueError("select the exact fill-closed V6 profile and capability")
    if not isinstance(source, dict):
        raise ValueError("source checkpoint profile must be an object")
    v5 = resolve_protocol_profile("qwen3-4b-base-dapo-reasoning-v5")
    v6 = resolve_protocol_profile("qwen3-4b-base-dapo-fill-closed-v6")
    old_contract = v5.to_generation_contract()
    new_contract = v6.to_generation_contract()
    window_fields = {"profile_id", "protocol_version", "throughput_tiebreak"}
    if ({k: v for k, v in old_contract.items() if k not in window_fields} !=
            {k: v for k, v in new_contract.items() if k not in window_fields}):
        raise ValueError("generation semantics differ beyond the window transition")
    expected = {
        "schema_version": 2, "profile_id": v5.profile_id,
        "protocol_version": 5, "base_model_id": v5.model_id,
        "base_model_revision": v5.model_revision,
        "generation_contract_sha256": hashlib.sha256(json.dumps(
            old_contract, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
    }
    for key, value in expected.items():
        if type(source.get(key)) is not type(value) or source.get(key) != value:
            raise ValueError(f"source V5 lineage mismatch for {key}")
    target = active_checkpoint_profile()
    run_id = source.get("training_run_id")
    if (not isinstance(run_id, str) or not run_id or run_id.strip() != run_id
            or run_id != target["training_run_id"]):
        raise ValueError("set RELIQUARY_TRAINING_RUN_ID to the exact source run")
    cursor = require_checkpoint_number(source.get("trained_window_cursor"),
                                       field="source trained window cursor")
    lr_step = require_checkpoint_number(source.get("lr_schedule_step"),
                                        field="source LR schedule step")
    if source.get("journal_key_space", "raw") != "raw":
        raise ValueError("source journal must use raw V5 window keys")
    if cursor != last_archived_window:
        raise ValueError("final V5 checkpoint must cover exactly the archive boundary")
    first_window = cursor + 1
    target.update({
        "trained_window_cursor": encoded_window_journal_key(first_window, 0) - 1,
        "journal_key_space": "fill_closed", "lr_schedule_step": lr_step,
    })
    provenance = {
        "schema_version": 1, "kind": "v5_to_fill_closed_v6",
        "source_repo": repo_id, "source_revision": revision,
        "source_checkpoint_n": checkpoint_n, "source_profile": source,
        "target_checkpoint_n": checkpoint_n + 1, "target_profile": target,
        "first_v6_window": first_window,
        "optimizer_state": "not_persisted_restart_rewarmup_required",
        "weights": "inherited_unchanged_from_parent_commit",
    }
    return {
        "repo_id": repo_id, "parent_commit": revision,
        "commit_message": f"checkpoint {checkpoint_n + 1} (v5-to-fill-v6)",
        "files": {CHECKPOINT_PROFILE_NAME: target, TRANSITION_NAME: provenance},
    }


def source_checkpoint_number(commits: list, revision: str) -> int:
    if not commits or commits[0].commit_id != revision:
        raise ValueError("source revision is no longer repository HEAD; prepare again")
    number = checkpoint_n_from_commit_title(commits[0].title)
    if number is None:
        raise ValueError("source HEAD must be an explicitly numbered checkpoint")
    numbers = [checkpoint_n_from_commit_title(c.title) for c in commits]
    if max(n for n in numbers if n is not None) != number:
        raise ValueError("source HEAD is behind an existing checkpoint number")
    if numbers.count(number) != 1:
        raise ValueError("source checkpoint number was rebound in repository history")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-revision", required=True, help="Exact current HF HEAD SHA")
    parser.add_argument("--last-archived-window", type=int, required=True,
                        help="Verified archive boundary after draining and freezing V5")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    canonical_checkpoint_identity(0, args.repo_id, args.source_revision)
    if args.output_dir.exists():
        raise ValueError("output directory already exists")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    commits = list(api.list_repo_commits(args.repo_id))
    number = source_checkpoint_number(commits, args.source_revision)
    profile = strict_json_loads(Path(hf_hub_download(
        args.repo_id, CHECKPOINT_PROFILE_NAME, revision=args.source_revision,
    )).read_bytes())
    info = api.model_info(args.repo_id, revision=args.source_revision)
    names = {f.rfilename for f in info.siblings or []}
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if info.sha != args.source_revision or not required <= names:
        raise ValueError("immutable source checkpoint lacks model/tokenizer metadata")
    if "model.safetensors" not in names:
        index_name = "model.safetensors.index.json"
        if index_name not in names:
            raise ValueError("source checkpoint has no model safetensors weights")
        index = strict_json_loads(Path(hf_hub_download(
            args.repo_id, index_name, revision=args.source_revision,
        )).read_bytes())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map or not all(
            isinstance(name, str) and name.endswith(".safetensors") and name in names
            for name in weight_map.values()
        ):
            raise ValueError("source checkpoint has an incomplete model weight index")
    plan = prepare_transition(profile, repo_id=args.repo_id,
                              revision=args.source_revision, checkpoint_n=number,
                              last_archived_window=args.last_archived_window)
    # Verify again after reads: this never promises a live run is already frozen.
    source_checkpoint_number(list(api.list_repo_commits(args.repo_id)), args.source_revision)
    args.output_dir.mkdir(mode=0o700, parents=True)
    for name, value in plan["files"].items():
        (args.output_dir / name).write_bytes(_json_bytes(value))
    (args.output_dir / "commit-plan.json").write_bytes(_json_bytes({
        **{k: v for k, v in plan.items() if k != "files"},
        "add_files": {name: hashlib.sha256(_json_bytes(value)).hexdigest()
                      for name, value in plan["files"].items()},
        "status": "prepared_only_no_remote_write",
    }))
    print(f"Prepared {plan['commit_message']} in {args.output_dir}; nothing published.")


if __name__ == "__main__":
    main()
