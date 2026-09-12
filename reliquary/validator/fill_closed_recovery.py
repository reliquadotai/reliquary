"""Abort an interrupted fill window without losing committed miner payments.

Pending proof/accumulator memory is deliberately not replayed. Queue receipts
bind paid-group evidence to the exact durable training body; unwritten slots
become tombstones. A complete archive is retained until its queue commit.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from reliquary.constants import B_BATCH, FILL_CLOSED_EMISSIONS_PER_WINDOW, FILL_CLOSED_PICKS_PER_WINDOW
from reliquary.shared.checkpoint_identity import require_immutable_checkpoint_revision
from reliquary.shared.strict_json import strict_json_loads
from reliquary.shared.training_payload import (
    active_training_identity,
    encode_tombstone,
    validate_training_identity,
)
from reliquary.validator.control import write_json
from reliquary.validator.fill_closed_rotation import FillClosedRotationGate
from reliquary.validator.token_rewards import AcceptedGroup, split_environment_pool


_FIXED_GROUP_PAYMENT_POLICY = "fixed-selected-group/v1"
_FIFO_SELECTION_POLICY = "fifo-ingress/v1"


def accounting_rows(batches: dict | None, *, batch_index: int) -> list[dict]:
    rows = []
    for environment, groups in (batches or {}).items():
        for group in groups:
            rows.append({
                "env_name": environment, "batch_index": batch_index,
                "hotkey": str(group.hotkey), "prompt_idx": int(group.prompt_idx),
                "sigma": float(group.sigma),
                "eos_tokens": int(getattr(group, "eos_tokens", 0) or 0),
                "claimed_checkpoint_hash": str(group.claimed_checkpoint_hash),
                "merkle_root": group.merkle_root_bytes.hex(),
                "selection_digest": group.selection_digest.hex(),
                "rollouts": [
                    {"tokens": list(rollout.commit["tokens"]),
                     "reward": float(rollout.reward),
                     **({"hash": group.rollout_hashes[index].hex()}
                        if index < len(group.rollout_hashes) else {})}
                    for index, rollout in enumerate(group.rollouts)
                ],
            })
    return rows


class FillClosedRecoveryStore:
    def __init__(self, state_dir: str | Path):
        self.directory = Path(state_dir) / "fill_active"
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, window: int) -> Path:
        if type(window) is not int or window < 0:
            raise ValueError("invalid active window")
        return self.directory / f"window-{window}.json"

    def load(self, window: int) -> dict:
        value = strict_json_loads(self._path(window).read_bytes())
        base_keys = {
            "schema_version", "window_start", "identity", "parent_checkpoint_n",
            "parent_revision", "environments", "batch_targets", "archive",
        }
        if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
            raise ValueError("invalid active window record")
        schema_version = value["schema_version"]
        if schema_version == 1:
            valid_fields = set(value) - {"picks_target"} == base_keys
        elif schema_version == 2:
            valid_fields = set(value) == base_keys | {
                "picks_target", "journal_slots", "payment_policy",
                "selection_policy", "window_pool",
            }
        else:
            valid_fields = False
        if not valid_fields:
            raise ValueError("invalid active window record")
        if type(value["window_start"]) is not int or value["window_start"] != window:
            raise ValueError("active window identity mismatch")
        validate_training_identity(value["identity"], active_training_identity(), artifact="active window")
        require_immutable_checkpoint_revision(value["parent_revision"], field="active window parent")
        environments = value["environments"]
        targets = value["batch_targets"]
        if (not isinstance(environments, list) or not environments
                or len(set(environments)) != len(environments)
                or not isinstance(targets, dict) or set(targets) != set(environments)
                or any(type(n) is not int or n <= 0 for n in targets.values())):
            raise ValueError("invalid active window environments")
        picks = value.get("picks_target", FILL_CLOSED_EMISSIONS_PER_WINDOW)
        if type(picks) is not int or not 1 <= picks <= FILL_CLOSED_EMISSIONS_PER_WINDOW:
            raise ValueError("invalid active window pick target")
        if schema_version == 2:
            if any(target != B_BATCH for target in targets.values()):
                raise ValueError("active window batch target mismatch")
            if (
                type(value["journal_slots"]) is not int
                or value["journal_slots"] != FILL_CLOSED_EMISSIONS_PER_WINDOW
                or picks > value["journal_slots"]
            ):
                raise ValueError("active window journal stride mismatch")
            if value["payment_policy"] != _FIXED_GROUP_PAYMENT_POLICY:
                raise ValueError("invalid active window payment policy")
            if value["selection_policy"] != _FIFO_SELECTION_POLICY:
                raise ValueError("invalid active window selection policy")
            window_pool = value["window_pool"]
            if (
                isinstance(window_pool, bool)
                or not isinstance(window_pool, (int, float))
                or not math.isfinite(float(window_pool))
                or float(window_pool) < 0.0
            ):
                raise ValueError("invalid active window pool")
        return value

    def windows(self) -> list[int]:
        windows = []
        for path in sorted(self.directory.glob("window-*.json")):
            window = int(path.name.removeprefix("window-").removesuffix(".json"))
            self.load(window)
            windows.append(window)
        return sorted(windows)

    def begin(self, window: int, *, checkpoint_n: int, revision: str, targets: dict) -> None:
        if self._path(window).exists():
            raise RuntimeError("active window requires recovery before reuse")
        write_json(self._path(window), {
            "schema_version": 1, "window_start": window,
            "identity": active_training_identity(), "parent_checkpoint_n": checkpoint_n,
            "parent_revision": revision, "environments": list(targets),
            "batch_targets": targets, "archive": None,
            "picks_target": FILL_CLOSED_PICKS_PER_WINDOW,
        })
        self.load(window)

    def quarantine_uncommitted(self, queue_dir: Path) -> None:
        """Only a known active window permits discarding an unpaid staging body."""
        commit_dir = queue_dir / "journal_commits"
        for body in sorted(commit_dir.glob("window-*.body")):
            key = int(body.name.split(".")[0].removeprefix("window-"))
            receipt = commit_dir / f"window-{key}.json"
            if receipt.exists():
                continue
            self.load(key // FILL_CLOSED_EMISSIONS_PER_WINDOW)
            quarantine = self.directory / "uncommitted"
            quarantine.mkdir(exist_ok=True)
            destination = quarantine / body.name
            if destination.exists():
                raise RuntimeError("uncommitted body already quarantined; refusing overwrite")
            os.replace(body, destination)
            self._sync(commit_dir)
            self._sync(quarantine)

    @staticmethod
    def _sync(directory: Path) -> None:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def finish(self, window: int, archive: dict, archive_queue: Any) -> None:
        record = self.load(window)
        if record["schema_version"] == 2 and (
            archive.get("payment_policy") != record["payment_policy"]
            or archive.get("selection_policy") != record["selection_policy"]
            or archive.get("picks_target") != record["picks_target"]
            or archive.get("journal_slots") != record["journal_slots"]
            or archive.get("window_pool") != record["window_pool"]
        ):
            raise RuntimeError("archive policy differs from active window")
        if record["archive"] is not None and record["archive"] != archive:
            raise RuntimeError("recovered archive differs from committed archive")
        record["archive"] = archive
        write_json(self._path(window), record)
        archive_queue.enqueue(window, archive)
        self._path(window).unlink()
        self._sync(self.directory)

    def recover(self, window: int, *, queue: Any, archives: Any, rotation: Any) -> None:
        record = self.load(window)
        if record["archive"] is not None:
            self.finish(window, record["archive"], archives)
            return
        rows, rewards, payload_count = [], {}, 0
        environments = record["environments"]
        schema_version = record["schema_version"]
        picks_target = record.get("picks_target", FILL_CLOSED_EMISSIONS_PER_WINDOW)
        journal_slots = record.get("journal_slots", FILL_CLOSED_EMISSIONS_PER_WINDOW)
        window_pool = float(record.get("window_pool", 1.0))
        for index in range(journal_slots):
            key = window * journal_slots + index
            path = queue._journal_commit_dir / f"window-{key}.json"
            if not path.exists():
                queue.enqueue_committed_tombstone(key, encode_tombstone(
                    window_start=window, failure_stage="active_window_recovery",
                    failure_type="interrupted_before_commit",
                ))
            receipt = queue._validate_journal_receipt(strict_json_loads(path.read_bytes()), source=path.name)
            if receipt["journal_key"] != key:
                raise RuntimeError("recovery receipt belongs to another key")
            if receipt["kind"] == "tombstone" and receipt["schema_version"] == 1:
                continue
            if receipt["schema_version"] != 2:
                raise RuntimeError("paid window recovery requires accounting receipts")
            payload_count += int(receipt["kind"] == "payload")
            paid = receipt["accounting"]
            if (paid or receipt["kind"] == "payload") and index >= picks_target:
                raise RuntimeError("paid batch exceeds the window target")
            for row in paid:
                if (row["env_name"] not in environments or row["batch_index"] != index
                        or type(row["eos_tokens"]) is not int or row["eos_tokens"] < 0
                        or row["claimed_checkpoint_hash"] != record["parent_revision"]):
                    raise RuntimeError("paid group does not match active window")
            if schema_version == 2:
                paid = [
                    {
                        **row,
                        "selected_for_batch": True,
                        "rewarded": True,
                        "payment_source": "fill_closed_fixed_group",
                    }
                    for row in paid
                ]
            rows.extend(paid)
            for environment in environments:
                groups = [
                    AcceptedGroup(row["hotkey"], row["hotkey"], row["eos_tokens"])
                    for row in paid if row["env_name"] == environment
                ]
                pool = window_pool / len(environments) / picks_target
                if schema_version == 2:
                    slots = record["batch_targets"][environment]
                    if len(groups) > slots:
                        raise ValueError("groups exceed fixed payment slots")
                    share = pool / slots
                    shares = {}
                    for group in groups:
                        shares[group.hotkey] = shares.get(group.hotkey, 0.0) + share
                else:
                    shares = split_environment_pool(groups, pool=pool)
                for hotkey, reward in shares.items():
                    rewards[hotkey] = rewards.get(hotkey, 0.0) + reward
        gate = FillClosedRotationGate(
            source_window=window,
            required_journal_key=(window + 1) * journal_slots - 1,
            parent_checkpoint_n=record["parent_checkpoint_n"], parent_revision=record["parent_revision"],
            durable_payload_count=payload_count,
            requires_successor=payload_count >= journal_slots,
        )
        existing = rotation.load()
        if existing is None or existing.source_window <= window:
            rotation.save(gate)
        archive = {
            "archive_schema_version": 2, "window_start": window,
            "window_status": "recovered_partial" if rows else "aborted",
            "failure_stage": "active_window_recovery", "failure_type": "interrupted_window",
            "environments": environments, "environment": environments[0],
            "batch_targets": record["batch_targets"], "batch": rows,
            "rewards_by_hotkey": rewards, "training_identity": record["identity"],
            "checkpoint_revision": record["parent_revision"],
            "durable_payload_count": payload_count,
            "picks_target": picks_target,
            **({
                "journal_slots": journal_slots,
                "payment_policy": record["payment_policy"],
                "selection_policy": record["selection_policy"],
                "window_pool": window_pool,
            } if schema_version == 2 else {}),
            "runners_up": [], "rejected": [], "reject_summary": {},
            "training_quarantine": {"quarantined": False, "reasons": [], "metrics": {}},
        }
        self.finish(window, archive, archives)
