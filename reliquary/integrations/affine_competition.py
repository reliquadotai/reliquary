"""A bounded native epoch, enrolled identities, and one immutable reward archive.

Generation remains the pinned native runtime. The task trusts the Affine
authority's authenticated audit/vector; it does not claim an independent replay
or complete visibility of unchecked duplicate claims.
"""
from __future__ import annotations

import base64
import hashlib
import math
import os
from pathlib import Path
import re
import time

from reliquary.integrations.affine import _private_file, _private_open, _private_path
from reliquary.integrations.affine_evidence import canonical, manifest_bindings, require, verify_bundle
from reliquary.shared.strict_json import strict_json_loads
from reliquary.shared.task_id import normalise_task_id
from reliquary.shared.task_registry import (
    MECHANISM_NATIVE_AFFINE_POINTS, TaskEntry, validate_entry,
    require_fleet_knows_native_affine_points,
)
from reliquary.validator import corpus_periods as periods

SCHEMA = "affine-native-competition/v1"
REWARD_BASIS = "native-finalized-observed-subset-weight"
PREFIX = "reliquary/affine-native/epochs/"


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(raw, authority):
    """Native Ed25519 encoding, with no upstream/model import."""
    from nacl.signing import VerifyKey

    value = strict_json_loads(raw) if isinstance(raw, bytes) else raw
    require(isinstance(value, dict) and set(value) == {"payload", "signer", "signature"}
            and value["signer"] == authority and isinstance(value["payload"], dict), "signed_envelope")
    VerifyKey(bytes.fromhex(authority)).verify(
        canonical(value["payload"]), base64.b64decode(value["signature"], validate=True))
    return value["payload"]


def read_private(path, limit=8 * 1024 * 1024):
    path = Path(path)
    _private_file(path)
    require(path.stat().st_size <= limit, "private_document_budget")
    return strict_json_loads(path.read_bytes())


def write_private(path, document):
    path = Path(path)
    require(path.is_absolute(), "private_absolute_path")
    _private_path(path)
    parent = path.parent.stat()
    require(parent.st_uid == os.getuid() and not parent.st_mode & 0o077, "private_parent_directory")
    fd = _private_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical(document))


def prepare(snapshot, *, authority, task_id, cap, env_id=None):
    manifest = signed(snapshot["manifest_envelope"], authority)
    require(snapshot["manifest"] == manifest, "snapshot_manifest")
    require(normalise_task_id(task_id) == task_id and task_id != "default", "competition_task_id")
    require(type(cap) in (int, float) and math.isfinite(cap) and 0 <= cap <= 1, "competition_cap")
    selected = env_id or manifest["environments"][0]["env_id"]
    bindings = manifest_bindings(manifest, selected)
    require(manifest["transport_policy"] == "direct-r2-v1"
            and type(manifest["start"]) in (int, float) and type(manifest["deadline"]) in (int, float)
            and math.isfinite(manifest["start"]) and math.isfinite(manifest["deadline"])
            and manifest["start"] < manifest["deadline"]
            and type(manifest["K"]) is int and type(manifest["L"]) is int
            and manifest["K"] == manifest["L"] == 1
            and type(manifest.get("max_batches")) is int and 0 < manifest["max_batches"] <= 10000,
            "native_epoch_contract")
    contract = dict(schema=SCHEMA, authority=authority,
                    epoch_digest=digest(dict(authority=authority, epoch=manifest["epoch"])),
                    manifest_digest=digest(manifest), bindings=bindings,
                    start=manifest["start"], deadline=manifest["deadline"], reward_basis=REWARD_BASIS)
    return dict(schema="affine-competition-draft/v1", task_id=task_id, cap=float(cap),
                env_id=selected, manifest_envelope=snapshot["manifest_envelope"], contract=contract,
                base_digest=digest(dict(task_id=task_id, contract=contract)))


def validate_draft(draft):
    require(isinstance(draft, dict) and set(draft) == {
        "schema", "task_id", "cap", "env_id", "manifest_envelope", "contract", "base_digest"
    } and draft["schema"] == "affine-competition-draft/v1", "competition_draft")
    authority = draft["contract"]["authority"]
    manifest = signed(draft["manifest_envelope"], authority)
    rebuilt = prepare(dict(manifest=manifest, manifest_envelope=draft["manifest_envelope"]),
                      authority=authority, task_id=draft["task_id"], cap=draft["cap"], env_id=draft["env_id"])
    require(rebuilt == draft, "competition_draft_binding")
    return manifest


def enrollment_payload(draft, native_id, hotkey):
    manifest = validate_draft(draft)
    require(isinstance(native_id, str) and re.fullmatch("[0-9a-f]{64}", native_id)
            and native_id in manifest["capabilities"], "native_registered_identity")
    require(isinstance(hotkey, str) and 1 <= len(hotkey) <= 128, "miner_hotkey")
    return dict(schema="affine-native-enrollment/v1", task_id=draft["task_id"],
                base_digest=draft["base_digest"], native_id=native_id, hotkey=hotkey)


def sign_enrollment(draft, native_seed, wallet):
    from nacl.signing import SigningKey

    key = SigningKey(native_seed)
    payload = enrollment_payload(draft, key.verify_key.encode().hex(), wallet.hotkey.ss58_address)
    data = canonical(payload)
    return dict(payload=payload,
                native_signature=base64.b64encode(key.sign(data).signature).decode(),
                hotkey_signature=wallet.hotkey.sign(data).hex())


def verify_hotkey(hotkey, data, signature):
    from bittensor_wallet import Keypair

    return Keypair(ss58_address=hotkey).verify(data, signature)


def roster(draft, enrollments, *, hotkey_verifier=verify_hotkey):
    from nacl.signing import VerifyKey

    require(isinstance(enrollments, list) and 0 < len(enrollments) <= 4096, "enrollment_count")
    identities, hotkeys = set(), set()
    rows = []
    for envelope in enrollments:
        require(isinstance(envelope, dict) and set(envelope) == {
            "payload", "native_signature", "hotkey_signature"}, "enrollment_envelope")
        payload = envelope["payload"]
        require(payload == enrollment_payload(draft, payload["native_id"], payload["hotkey"]), "enrollment_binding")
        data = canonical(payload)
        VerifyKey(bytes.fromhex(payload["native_id"])).verify(
            data, base64.b64decode(envelope["native_signature"], validate=True))
        require(hotkey_verifier(payload["hotkey"], data, bytes.fromhex(envelope["hotkey_signature"])),
                "enrollment_hotkey_signature")
        require(payload["native_id"] not in identities and payload["hotkey"] not in hotkeys,
                "one_identity_per_hotkey")
        identities.add(payload["native_id"])
        hotkeys.add(payload["hotkey"])
        rows.append(envelope)
    return sorted(rows, key=lambda row: row["payload"]["native_id"])


def task_entry(draft, enrollments, *, hotkey_verifier=verify_hotkey):
    rows = roster(draft, enrollments, hotkey_verifier=hotkey_verifier)
    contract = {**draft["contract"], "enrollment_digest": digest(rows)}
    cap = draft["cap"]
    params = dict(start=cap, decay=1.0, rounds_per_step=1, deadband=0.0, snap=1.0,
                  floor=cap, cap=cap, median_rounds=1, last_good_fills=1,
                  min_incentive_share=0.0, min_incentive_ramp_start=0.0,
                  settlement=periods.SETTLEMENT_PERIOD_EMA)
    entry = TaskEntry(task_id=draft["task_id"], profile_id="affine-native-v1",
                      profile_sha256=digest(contract), mechanism=MECHANISM_NATIVE_AFFINE_POINTS,
                      params=params, status="active", retired_at=None, contract=contract)
    validate_entry(entry)
    return entry


def epoch_key(entry, suffix):
    validate_entry(entry)
    require(entry.mechanism == MECHANISM_NATIVE_AFFINE_POINTS, "native_competition_task")
    return f"{PREFIX}{entry.contract['epoch_digest']}/{suffix}.json"


def commitment(entry):
    return dict(schema="affine-native-commitment/v1", task_id=entry.task_id,
                contract_digest=entry.profile_sha256, enrollment_digest=entry.contract["enrollment_digest"])


class NativeStore:
    """Cold-path R2 objects, create-only and bounded. No allocation credentials."""

    def __init__(self, **client_kwargs):
        self.bucket = client_kwargs.pop("bucket_name", None) or os.getenv("R2_BUCKET_ID", "reliquary")
        self.client_kwargs = client_kwargs

    async def read(self, key):
        from botocore.exceptions import ClientError
        from reliquary.infrastructure.storage import get_s3_client

        async with get_s3_client(**self.client_kwargs) as client:
            try:
                response = await client.get_object(Bucket=self.bucket, Key=key)
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                    return None, None
                raise
            require(int(response.get("ContentLength", 0)) <= 1024 * 1024, "native_store_budget")
            raw = await response["Body"].read(1024 * 1024 + 1)
            require(len(raw) <= 1024 * 1024, "native_store_budget")
            return strict_json_loads(raw), response["LastModified"].timestamp()

    async def create(self, key, document):
        from botocore.exceptions import ClientError
        from reliquary.infrastructure.storage import get_s3_client

        async with get_s3_client(**self.client_kwargs) as client:
            try:
                await client.put_object(Bucket=self.bucket, Key=key, Body=canonical(document),
                                        ContentType="application/json", IfNoneMatch="*")
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in {
                    "PreconditionFailed", "412", "ConditionalRequestConflict"}:
                    raise
        saved, received = await self.read(key)
        require(saved == document, "immutable_native_object_collision")
        return received


async def declare(draft, enrollments, *, acknowledged, store=None, create_task=None,
                  read_registry=None, now=None):
    entry = task_entry(draft, enrollments)
    require_fleet_knows_native_affine_points(entry, acknowledged=acknowledged is True)
    require((time.time() if now is None else now) < entry.contract["deadline"], "enrollment_deadline")
    if read_registry is None:
        from reliquary.infrastructure.task_registry_store import read_registry
    entries, _ = await read_registry()
    existing = entries.get(entry.task_id)
    if existing is not None:
        require(existing == entry, "existing_competition_changed")
    else:
        from reliquary.shared.task_registry import add_task, require_default_declared_first

        require_default_declared_first(entries, entry)
        add_task(entries, entry)
    store = store or NativeStore()
    received = await store.create(epoch_key(entry, "commitment"), commitment(entry))
    require(entry.contract["start"] <= received < entry.contract["deadline"], "precommit_before_deadline")
    if create_task is None:
        from reliquary.infrastructure.task_registry_store import create_task
    # Registry CAS retains default-first and pool-cap safeguards. Identical
    # retries succeed; a changed contract is never accepted as a retry.
    if existing is None:
        await create_task(entry)
    return entry


def _score_scope(manifest, scores):
    require(type(scores.get("provisional")) is bool
            and type(scores.get("unchecked_duplicate_claims_unresolved")) is bool
            and scores.get("duplicate_coverage") in {"complete", "incomplete"}
            and scores.get("score_basis") in {"full-audit", "fully-audited-subset"}, "native_score_scope")
    incomplete = (scores.get("provisional") is not False or scores.get("duplicate_coverage") != "complete"
                  or scores.get("unchecked_duplicate_claims_unresolved") is not False)
    if incomplete:
        policy = manifest.get("live_reward_contract")
        require(isinstance(policy, dict)
                and policy.get("version") == "live-verified-subset-reward-v1"
                and policy.get("payable") is True and policy.get("netuid") == 120
                and policy.get("epoch") == manifest["epoch"]
                and policy.get("starts_at") == manifest["start"]
                and policy.get("checkpoint") == manifest["checkpoint"]["id"]
                and policy.get("source_sha256") == manifest["source_bundle"]["sha256"]
                and policy.get("basis") == "fully-audited-unique-observed-subset"
                and policy.get("unchecked_duplicate_claims") == "unresolved-no-global-uniqueness-claim"
                and policy.get("penalties") == manifest.get("audit_policy", {}).get("penalties")
                and scores.get("penalty_policy") == policy.get("penalties")
                and policy.get("units_per_point") == 1_000_000
                and policy.get("rounding") == "floor-after-hour-aggregation"
                and policy.get("compute_chain_transactions") is False
                and manifest.get("payable") is False
                and manifest.get("chain_execution_scope") == "operator-live-reward-bridge-only-v1"
                and scores.get("score_basis") == "fully-audited-subset", "unsupported_provisional_scope")
    return dict(duplicate_coverage=scores["duplicate_coverage"],
                unchecked_duplicate_claims_unresolved=scores["unchecked_duplicate_claims_unresolved"],
                provisional=scores["provisional"], score_basis=scores["score_basis"],
                independent_gpu_replay=False, native_payment_verified=False)


def reward_archive(entry, draft, enrollments, bundles, *, now, hotkey_verifier=verify_hotkey):
    rows = roster(draft, enrollments, hotkey_verifier=hotkey_verifier)
    expected = task_entry(draft, enrollments, hotkey_verifier=hotkey_verifier)
    require(entry.task_id == expected.task_id and entry.profile_sha256 == expected.profile_sha256,
            "declared_competition_binding")
    require(type(entry.params["cap"]) in (int, float) and math.isfinite(entry.params["cap"])
            and 0 <= entry.params["cap"] <= 1, "settlement_cap")
    require(now >= entry.contract["deadline"] and isinstance(bundles, list) and bool(bundles),
            "finalized_evidence_required")
    authority = entry.contract["authority"]
    manifest = validate_draft(draft)
    documents = bundles[0]["envelopes"]
    scores = signed(documents["scores"], authority)
    challenge = signed(documents["audit_challenge"], authority)
    history = signed(documents["history"], authority)
    require(history.get("authority") == authority and history.get("version") == 1
            and sum(row.get("epoch_id") == manifest["epoch"] for row in history["epochs"]) == 1,
            "signed_finalized_history")
    history_row = next(row for row in history["epochs"] if row["epoch_id"] == manifest["epoch"])
    checkpoint, source = history_row["checkpoint"], history_row["source_bundle"]
    require(checkpoint["id"] == manifest["checkpoint"]["id"]
            and checkpoint["files"] == manifest["checkpoint"]["files"]
            and history_row["deadline"] == manifest["deadline"]
            and history_row["payable"] is manifest["payable"], "history_checkpoint_binding")
    require(source.get("binding") == "epoch-signed"
            and source.get("sha256") == manifest["source_bundle"]["sha256"]
            and source.get("size") == manifest["source_bundle"]["size"], "history_source_binding")
    for name, envelope in (("manifest", draft["manifest_envelope"]), ("scores", documents["scores"]),
                           ("audit-challenge", documents["audit_challenge"])):
        route = history_row.get("objects", {}).get(name)
        if isinstance(route, dict) and "sha256" in route:
            require(route["sha256"] == digest(envelope), "history_document_digest")
    require(scores["epoch_id"] == manifest["epoch"] and scores["checkpoint"] == manifest["checkpoint"]["id"]
            and type(scores["finalized_at"]) in (int, float) and math.isfinite(scores["finalized_at"])
            and entry.contract["deadline"] <= scores["finalized_at"] <= now
            and scores["receipts"] == challenge["receipts"]
            and challenge["generated_after_freeze_at"] >= entry.contract["deadline"], "finalized_score_binding")
    points, weights = scores["points"], scores["weights"]
    require(all(type(v) is int and 0 <= v <= manifest["max_batches"] for v in points.values())
            and type(scores["total"]) is int and scores["total"] == sum(points.values()), "native_point_total")
    mass = scores.get("adjusted_points", points)
    require(set(mass) == set(points) and set(weights) <= set(points)
            and set(points) <= set(scores["receipts"])
            and set(scores["receipts"]) <= set(manifest["capabilities"])
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= points[k]
                    for k, v in mass.items()), "native_adjusted_points")
    total = sum(mass.values())
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                and math.isclose(v, mass[k] / total if total else 0, rel_tol=1e-12, abs_tol=1e-12)
                for k, v in weights.items())
            and (math.isclose(sum(weights.values()), 1, abs_tol=1e-12) if total else not weights),
            "native_adjusted_weights")
    scope = _score_scope(manifest, scores)
    by_miner = {}
    for bundle in bundles:
        miner = bundle["miner_id"]
        require(miner not in by_miner and bundle["authority"] == authority
                and bundle["envelopes"]["manifest"] == draft["manifest_envelope"]
                and bundle["envelopes"]["scores"] == documents["scores"]
                and bundle["envelopes"]["audit_challenge"] == documents["audit_challenge"], "shared_frozen_evidence")
        result = verify_bundle(bundle, signed, entry.contract["bindings"])
        require(result.get("points", 0) == points.get(miner, 0), "own_native_points")
        require(result.get("accepted_batches", 0) <= manifest["max_batches"], "own_native_quota")
        if miner in scores["receipts"]:
            frozen, receipt = history_row["frozen"][miner], scores["receipts"][miner]
            require(frozen["sha256"] == receipt["sha256"] == bundle["submission_sha256"]
                    and type(receipt["size"]) is int and receipt["size"] >= 0
                    and frozen["size"] == receipt["size"], "history_frozen_receipt")
        by_miner[miner] = result
    rewards, raw_points = {}, {}
    for row in rows:
        native, hotkey = row["payload"]["native_id"], row["payload"]["hotkey"]
        if native in scores["receipts"]:
            require(native in by_miner, "own_audit_required")
        reward = float(entry.params["cap"]) * weights.get(native, 0)
        if reward:
            rewards[hotkey] = reward
        raw_points[hotkey] = points.get(native, 0)
    return dict(schema="affine-native-period/v1", task_id=entry.task_id,
                contract_digest=entry.profile_sha256, epoch_digest=entry.contract["epoch_digest"],
                work_period=periods.period_of(scores["finalized_at"]), entry_period=periods.period_of(now) + 1,
                settlement_cap=float(entry.params["cap"]), rewards_by_hotkey=rewards,
                raw_unique_observed_points=raw_points,
                reward_basis=REWARD_BASIS, scope=scope, score_envelope_digest=digest(documents["scores"]),
                assurance="authority-authenticated-native-audits-and-finalized-observed-vector")


async def settle(entry, draft, enrollments, bundles, *, store=None, now=None):
    store = store or NativeStore()
    locked, received = await store.read(epoch_key(entry, "commitment"))
    require(locked == commitment(entry) and entry.contract["start"] <= received < entry.contract["deadline"],
            "immutable_precommitted_enrollment")
    archive = reward_archive(entry, draft, enrollments, bundles, now=time.time() if now is None else now)
    key = epoch_key(entry, "settlement")
    existing, _ = await store.read(key)
    if existing is not None:
        # A retry has a later entry_period; every substantive binding must
        # still agree, and the original archive remains the only payment.
        require({k: v for k, v in archive.items() if k != "entry_period"}
                == {k: v for k, v in existing.items() if k != "entry_period"}, "immutable_epoch_settlement")
        return existing
    require(entry.status == "active", "retired_competition_cannot_create_new_settlement")
    await store.create(key, archive)
    return archive


async def read_archive(entry, *, store=None):
    store = store or NativeStore()
    document, _ = await store.read(epoch_key(entry, "settlement"))
    if document is None:
        return None
    locked, received = await store.read(epoch_key(entry, "commitment"))
    require(locked == commitment(entry) and type(received) in (int, float)
            and entry.contract["start"] <= received < entry.contract["deadline"], "native_archive_precommit")
    require(document.get("schema") == "affine-native-period/v1"
            and document.get("task_id") == entry.task_id
            and document.get("contract_digest") == entry.profile_sha256
            and document.get("epoch_digest") == entry.contract["epoch_digest"]
            and type(document.get("entry_period")) is int
            and type(document.get("work_period")) is int
            and 0 <= document["work_period"] <= document["entry_period"], "native_archive_binding")
    rewards = document.get("rewards_by_hotkey")
    require(isinstance(rewards, dict) and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                            for v in rewards.values()), "native_archive_rewards")
    cap = document.get("settlement_cap")
    require(type(cap) in (int, float) and math.isfinite(cap) and 0 <= cap <= 1
            and sum(rewards.values()) <= cap + 1e-12, "native_archive_budget")
    return document
