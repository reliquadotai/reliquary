#!/usr/bin/env python3
"""Verify a prepared V1 metadata commit; publish only with an explicit fenced parent.

By default weights and other files are inherited unchanged. An explicit prepared
base reset copies the pinned base repository tree, preserving the V5 parent in
history. This command never opens admission, writes R2, signs, or changes the old
writer. Freeze that writer and complete the private storage migration before --apply.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliquary.shared.strict_json import strict_json_loads  # noqa: E402
from reliquary.trainer.publisher import _file_identity  # noqa: E402
from reliquary.validator.control import write_json  # noqa: E402
from scripts.prepare_v1_bootstrap import (  # noqa: E402
    CHECKPOINT_PROFILE_NAME, TRANSITION, prepare_bootstrap, source_checkpoint_number,
)


def _tree(api, repo: str, revision: str) -> dict:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("checkpoint metadata did not resolve the exact revision")
    result = {}
    for entry in info.siblings or []:
        lfs = getattr(entry, "lfs", None)
        digest = lfs.sha256 if lfs is not None else entry.blob_id
        if not digest or type(entry.size) is not int or entry.size < 0:
            raise ValueError("checkpoint file metadata is incomplete")
        result[entry.rfilename] = (entry.size, digest, lfs is not None)
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if not required <= result.keys() or not any(name.endswith(".safetensors") for name in result):
        raise ValueError("source lacks model/tokenizer files")
    return result


def _validate_weight_index(tree, *, download, repo, revision):
    if "model.safetensors" not in tree:
        index = strict_json_loads(Path(download(repo, "model.safetensors.index.json", revision=revision)).read_bytes())
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping or not all(
            isinstance(name, str) and name.endswith(".safetensors") and name in tree
            for name in mapping.values()
        ):
            raise ValueError("source checkpoint has an incomplete weight index")


def publish_prepared(directory: Path, *, api, download, apply=False,
                     fenced_parent: str | None = None) -> dict:
    plan = strict_json_loads((directory / "commit-plan.json").read_bytes())
    names = {CHECKPOINT_PROFILE_NAME, TRANSITION}
    if set(plan["add_files"]) != names:
        raise ValueError("bootstrap must add exactly its two metadata files")
    files = {}
    identities = {}
    for name in names:
        identities[name] = _file_identity(directory / name)
        if identities[name]["sha256"] != plan["add_files"][name]:
            raise ValueError("prepared bootstrap file changed")
        files[name] = strict_json_loads((directory / name).read_bytes())
    transition = files[TRANSITION]
    migration = plan["private_storage_migration"]
    expected = prepare_bootstrap(
        transition["source_profile"], repo_id=plan["repo_id"],
        revision=plan["parent_commit"], checkpoint_n=transition["source_checkpoint_n"],
        last_archived_window=migration["archive_boundary"],
        source_bucket=migration["source_bucket"], target_bucket=migration["target_bucket"],
        storage_mode=migration.get("storage_mode", "distinct-bucket"),
        lr_start_step=files[CHECKPOINT_PROFILE_NAME]["lr_schedule_step"],
        reset_to_base="base_weights" in transition,
    )
    if files != expected["files"] or any(
        plan.get(key) != value for key, value in expected.items() if key != "files"
    ):
        raise ValueError("prepared plan does not match the active V1 contract")
    repo, parent = plan["repo_id"], plan["parent_commit"]
    source = strict_json_loads(Path(download(repo, CHECKPOINT_PROFILE_NAME, revision=parent)).read_bytes())
    if source != transition["source_profile"]:
        raise ValueError("source profile changed from prepared provenance")
    if source_checkpoint_number(list(api.list_repo_commits(repo, revision=parent)), parent) != transition["source_checkpoint_n"]:
        raise ValueError("source checkpoint number differs from prepared provenance")
    source_tree = _tree(api, repo, parent)
    _validate_weight_index(source_tree, download=download, repo=repo, revision=parent)
    base_weights = transition.get("base_weights")
    expected_tree = source_tree
    if base_weights is not None:
        expected_tree = _tree(api, base_weights["repo_id"], base_weights["revision"])
        _validate_weight_index(expected_tree, download=download,
                               repo=base_weights["repo_id"], revision=base_weights["revision"])
        if names.intersection(expected_tree):
            raise ValueError("base repository already contains Reliquary bootstrap metadata")
    commits = list(api.list_repo_commits(repo))
    if not commits:
        raise ValueError("checkpoint repository is empty")
    head = commits[0].commit_id
    if head == parent:
        source_checkpoint_number(commits, parent)
        if not apply:
            return {"status": "verified_not_published", "parent_commit": parent,
                    "checkpoint_n": transition["target_checkpoint_n"]}
        if fenced_parent != parent:
            raise ValueError("--apply requires the exact confirmed fenced parent")
        from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete

        operations = [CommitOperationAdd(path_in_repo=name,
                      path_or_fileobj=str(directory / name)) for name in sorted(names)]
        if base_weights is not None:
            # Hub >=1.30 copies LFS server-side and regular files byte-for-byte.
            # No model load/re-serialization and no learned V5 files survive in HEAD.
            operations += [CommitOperationCopy(src_path_in_repo=name, path_in_repo=name,
                           src_repo_id=base_weights["repo_id"], src_repo_type="model",
                           src_revision=base_weights["revision"]) for name in sorted(expected_tree)]
            operations += [CommitOperationDelete(path_in_repo=name)
                           for name in sorted(source_tree.keys() - expected_tree.keys() - names)]

        # HF compare-and-swap owns the race with any writer advancing HEAD.
        # A lost acknowledgement leaves the unchanged local plan for exact retry.
        head = api.create_commit(
            repo_id=repo, parent_commit=parent, commit_message=plan["commit_message"],
            operations=operations,
        ).oid
    target_commits = list(api.list_repo_commits(repo, revision=head))
    if (len(target_commits) < 2 or target_commits[0].commit_id != head
            or target_commits[0].title != plan["commit_message"]
            or target_commits[1].commit_id != parent):
        raise ValueError("HEAD is not the exact bootstrap child of the fenced parent")
    if source_checkpoint_number(target_commits, head) != transition["target_checkpoint_n"]:
        raise ValueError("bootstrap checkpoint number differs")
    target_tree = _tree(api, repo, head)
    if {k: v for k, v in expected_tree.items() if k not in names} != {
        k: v for k, v in target_tree.items() if k not in names
    }:
        raise ValueError("bootstrap changed weights or unrelated repository files from the expected source")
    for name, identity in identities.items():
        found = target_tree.get(name)
        if found is None or found != (
            identity["size"], identity["sha256"] if found[2] else identity["blob_id"], found[2],
        ):
            raise ValueError("bootstrap remote metadata differs from the prepared files")
    if api.model_info(repo).sha != head:
        raise ValueError("checkpoint HEAD advanced; keep admission closed and reconcile writer")
    profile = files[CHECKPOINT_PROFILE_NAME]
    receipt = {"status": "published_verified_requires_gpu_qualification_and_activation",
               "repo_id": repo, "revision": head,
               "checkpoint_n": transition["target_checkpoint_n"],
               "trained_window_cursor": profile["trained_window_cursor"],
               "training_run_id": profile["training_run_id"], "parent_commit": parent,
               "metadata_sha256": {name: value["sha256"] for name, value in identities.items()}}
    if base_weights is not None:
        receipt["base_weights"] = base_weights
    if apply:
        write_json(directory / "published-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-fenced-parent")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    print(json.dumps(publish_prepared(args.directory, api=HfApi(), download=hf_hub_download,
                                    apply=args.apply, fenced_parent=args.confirm_fenced_parent), indent=2))
