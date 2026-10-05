"""Real native/hotkey signatures, observed native penalties, immutable pay and replay."""
import asyncio
import base64
import copy
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from bittensor_wallet import Keypair
from nacl.signing import SigningKey

from reliquary.integrations import affine_competition as ac
from reliquary.integrations.affine_evidence import canonical
from reliquary.shared.task_registry import RegistryError, validate_registry, set_cap
from reliquary.validator import corpus_periods as periods


class Store:
    def __init__(self, clock):
        self.clock = clock
        self.objects = {}

    async def read(self, key):
        return self.objects.get(key, (None, None))

    async def create(self, key, value):
        if key not in self.objects:
            self.objects[key] = (copy.deepcopy(value), self.clock)
        saved, received = self.objects[key]
        ac.require(saved == value, "immutable_native_object_collision")
        return received


class NativeCompetition(unittest.TestCase):
    def setUp(self):
        self.authority_key = SigningKey.generate()
        self.authority = self.authority_key.verify_key.encode().hex()
        self.native_keys = [SigningKey.generate() for _ in range(3)]
        self.native = [key.verify_key.encode().hex() for key in self.native_keys]
        self.wallets = [types.SimpleNamespace(hotkey=Keypair.create_from_mnemonic(Keypair.generate_mnemonic()))
                        for _ in range(2)]
        start = periods.PERIOD_EPOCH + 100 * periods.PERIOD_SECONDS
        files = {"model.safetensors": ac.digest("weights"), "tokenizer.json": ac.digest("tokenizer")}
        source = dict(sha256=ac.digest("native source"), size=123)
        penalties = dict(invalid_batch_multiplier=0.5, penalize_structural=False, zero_epoch_after=0)
        sampling = dict(version="forced-inverse-cdf-replay-v1", generation="uncached-eager-inverse-cdf",
                        verification="exact-token-replay", max_attempts=128, randomness=ac.digest("randomness"))
        self.manifest = dict(epoch="nonpayable-live-reward-math-v1-fixture", start=start, deadline=start + 60,
            checkpoint=dict(id=ac.digest(files), files=files), source_bundle=source,
            environments=[dict(env_id="math", indices=[0, 1, 2, 3], spec={"version": "v1"}, harness={})],
            K=1, L=1, max_batches=3, transport_policy="direct-r2-v1", capabilities={n: "sealed-fixture" for n in self.native},
            payable=False, sampling_contract=sampling, sampling_source_hash=ac.digest("native sampler"),
            audit_policy=dict(mode="sampled", penalties=penalties), chain_execution_scope="operator-live-reward-bridge-only-v1")
        self.manifest["live_reward_contract"] = dict(version="live-verified-subset-reward-v1", payable=True,
            epoch=self.manifest["epoch"], reward_epoch="live-reward-fixture", starts_at=start,
            checkpoint=self.manifest["checkpoint"]["id"], source_sha256=source["sha256"], cutover_id="fixture-cutover",
            netuid=120, basis="fully-audited-unique-observed-subset",
            unchecked_duplicate_claims="unresolved-no-global-uniqueness-claim", penalties=penalties,
            units_per_point=1_000_000, rounding="floor-after-hour-aggregation", compute_chain_transactions=False)
        self.draft = ac.prepare(dict(manifest=self.manifest, manifest_envelope=self.envelope(self.manifest)),
                                authority=self.authority, task_id="affine-fixture", cap=0.3)
        self.enrollments = [ac.sign_enrollment(self.draft, key.encode(), wallet)
                            for key, wallet in zip(self.native_keys, self.wallets)]
        self.entry = ac.task_entry(self.draft, self.enrollments)
        self.now = self.manifest["deadline"] + 100
        receipts = {n: dict(sha256=ac.digest(n), size=1000, received_at=start + 5) for n in self.native}
        self.scores = dict(epoch_id=self.manifest["epoch"], checkpoint=self.manifest["checkpoint"]["id"],
            finalized_at=self.manifest["deadline"] + 1, payable=False, receipts=receipts,
            points=dict(zip(self.native, [2, 1, 1])), total=4,
            adjusted_points={n: 1.0 for n in self.native}, weights={n: 1 / 3 for n in self.native},
            penalty_policy=penalties, penalties={self.native[0]: dict(multiplier=0.5, confirmed_invalid_batches=1),
                **{n: dict(multiplier=1.0, confirmed_invalid_batches=0) for n in self.native[1:]}},
            provisional=True, score_basis="fully-audited-subset", duplicate_coverage="incomplete",
            unchecked_duplicate_claims_unresolved=True)
        challenge = dict(receipts=receipts, generated_after_freeze_at=self.manifest["deadline"] + 1)
        history = dict(version=1, authority=self.authority,
                       epochs=[dict(epoch_id=self.manifest["epoch"], deadline=self.manifest["deadline"], payable=False,
                           checkpoint={**self.manifest["checkpoint"], "read_urls": {"model": "fixture-rotatable-route"}},
                           source_bundle={**source, "binding": "epoch-signed"},
                           frozen={n: dict(sha256=r["sha256"], size=r["size"], url="fixture-rotatable-route") for n, r in receipts.items()},
                           objects={"scores": "signed-native-fixture-route"})])
        context = dict(contract=sampling, epoch=self.manifest["epoch"], checkpoint=self.manifest["checkpoint"]["id"])
        sampling_binding = ac.digest(context)
        self.bundles = []
        for number, native in enumerate(self.native[:2]):
            accepted, outcomes = [], []
            for index in ([0, 1] if number == 0 else [2]):
                rolls = [dict(env_id="math", index=index, classification=kind, seed=attempt,
                    sampling=dict(version=sampling["version"], binding_sha256=sampling_binding, attempt=attempt),
                    turns=[dict(prompt=[1], output=[attempt])]) for attempt, kind in enumerate(("positive", "negative"), 1)]
                accepted.append(dict(schema=2, epoch=self.manifest["epoch"], checkpoint=self.manifest["checkpoint"]["id"],
                                     env_id="math", index=index, sample_index=index, environment_version="v1", rollouts=rolls))
                outcomes.append(dict(batch=index, env_id="math", index=index, fully_audited=True, structural_valid=True, valid=True))
            report = dict(epoch=self.manifest["epoch"], submission_sha256=receipts[native]["sha256"],
                accepted=accepted, outcomes=outcomes, remote_job_id="fixture-native-job",
                sampling_assurance=dict(sampling_required=True, scope="fully-audited-rollouts-only",
                    version=sampling["version"], binding_sha256=sampling_binding, verification="exact-token-replay",
                    historical_execution_proven=False))
            self.bundles.append(dict(schema="affine-evidence-bundle/v1", authority=self.authority,
                epoch_id=self.manifest["epoch"], miner_id=native, submission_sha256=receipts[native]["sha256"],
                source_sha256=source["sha256"], env_id="math", requested_indices=None,
                envelopes={"manifest": self.draft["manifest_envelope"], "scores": self.envelope(self.scores),
                    "audit_challenge": self.envelope(challenge), "audit": self.envelope(report),
                    "history": self.envelope(history)}, byte_checks={}))

    def envelope(self, payload):
        return dict(payload=copy.deepcopy(payload), signer=self.authority,
                    signature=base64.b64encode(self.authority_key.sign(canonical(payload)).signature).decode())

    def archive(self, bundles=None, entry=None, draft=None):
        return ac.reward_archive(entry or self.entry, draft or self.draft, self.enrollments,
                                 self.bundles if bundles is None else bundles, now=self.now)

    def test_native_penalty_weights_burn_unenrolled_share_and_expose_scope(self):
        archive = self.archive()
        self.assertAlmostEqual(sum(archive["rewards_by_hotkey"].values()), 0.2)
        self.assertEqual(sum(archive["raw_unique_observed_points"].values()), 3)
        self.assertEqual(archive["scope"]["duplicate_coverage"], "incomplete")
        self.assertFalse(archive["scope"]["independent_gpu_replay"])
        text = canonical(archive).decode()
        self.assertTrue(all(native not in text for native in self.native))
        self.assertNotIn("sealed-fixture", text)

    def test_dual_signature_binding_duplicate_identity_and_hotkey(self):
        for field in ("hotkey_signature", "native_signature"):
            altered = copy.deepcopy(self.enrollments)
            altered[0][field] = ("00" * 64 if field == "hotkey_signature" else base64.b64encode(bytes(64)).decode())
            with self.subTest(field=field), self.assertRaises(Exception):
                ac.roster(self.draft, altered)
        with self.assertRaises(ValueError):
            ac.roster(self.draft, [self.enrollments[0]] * 2)
        another = ac.sign_enrollment(self.draft, self.native_keys[1].encode(), self.wallets[0])
        with self.assertRaises(ValueError):
            ac.roster(self.draft, [self.enrollments[0], another])

    def test_no_unapproved_provisional_scope_or_early_finalization(self):
        altered = copy.deepcopy(self.draft)
        manifest = copy.deepcopy(self.manifest)
        manifest.pop("live_reward_contract")
        altered = ac.prepare(dict(manifest=manifest, manifest_envelope=self.envelope(manifest)),
                             authority=self.authority, task_id=self.draft["task_id"], cap=0.3)
        enrollment = [ac.sign_enrollment(altered, key.encode(), wallet)
                      for key, wallet in zip(self.native_keys, self.wallets)]
        entry = ac.task_entry(altered, enrollment)
        bundles = copy.deepcopy(self.bundles)
        for bundle in bundles:
            bundle["envelopes"]["manifest"] = altered["manifest_envelope"]
        with self.assertRaises(ValueError):
            ac.reward_archive(entry, altered, enrollment, bundles, now=self.now)
        bundles = copy.deepcopy(self.bundles)
        scores = dict(self.scores, finalized_at=self.manifest["deadline"] - 1)
        for bundle in bundles:
            bundle["envelopes"]["scores"] = self.envelope(scores)
        with self.assertRaises(ValueError):
            self.archive(bundles)

    def test_missing_audit_stale_manifest_forged_score_and_bad_adjustments(self):
        with self.assertRaises(ValueError):
            self.archive(self.bundles[:1])
        bundles = copy.deepcopy(self.bundles)
        bundles[0]["envelopes"]["scores"]["payload"]["total"] = 999
        with self.assertRaises(Exception):
            self.archive(bundles)
        for field, value in (("adjusted_points", {n: 4 for n in self.native}),
                             ("weights", {n: 0.9 for n in self.native}), ("total", True)):
            bundles = copy.deepcopy(self.bundles)
            scores = dict(self.scores, **{field: value})
            for bundle in bundles:
                bundle["envelopes"]["scores"] = self.envelope(scores)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.archive(bundles)

    def test_registry_epoch_pool_guard_cap_follows_native_and_generation_refuses(self):
        with self.assertRaises(RegistryError):
            validate_registry({"one": self.entry, "two": replace(self.entry, task_id="other")})
        self.assertEqual(set_cap({self.entry.task_id: self.entry}, self.entry.task_id, 0.1)[self.entry.task_id].params["floor"], 0.1)
        from reliquary.validator.task_config import TaskConfigError, resolve_task_config

        with self.assertRaises(TaskConfigError):
            resolve_task_config({self.entry.task_id: self.entry}, self.entry.task_id,
                                profile_id=self.entry.profile_id, generation_contract=self.entry.contract)

    def test_single_create_before_deadline_then_idempotent_settlement_and_replay(self):
        async def run():
            store = Store(self.manifest["start"] + 20)
            default = replace(self.entry, task_id="default", mechanism="rl-discovered-price", contract=None,
                              params={**self.entry.params, "cap": 0.7, "floor": 0.0})
            registry = {"default": default}
            async def read_registry():
                return registry, "fixture-etag"
            async def create_task(entry):
                registry[entry.task_id] = entry
            with self.assertRaises(ValueError):
                await ac.declare(self.draft, self.enrollments, acknowledged=False, store=store,
                                 create_task=create_task, read_registry=read_registry, now=store.clock)
            self.assertFalse(store.objects)
            entry = await ac.declare(self.draft, self.enrollments, acknowledged=True, store=store,
                                     create_task=create_task, read_registry=read_registry, now=store.clock)
            await ac.declare(self.draft, self.enrollments, acknowledged=True, store=store,
                             create_task=create_task, read_registry=read_registry, now=store.clock)
            # Exercise the same pinned worker launcher with a harmless native
            # fixture. Its zero exit is deliberately not scoring evidence.
            from tests.unit.test_affine_runtime import AffineRuntimeTest
            from reliquary.integrations.affine import AffineRunner

            runtime = AffineRuntimeTest()
            runtime.setUp()
            self.addCleanup(runtime.doCleanups)
            config = runtime.config("import sys; assert sys.flags.isolated and sys.dont_write_bytecode\n")
            result = AffineRunner(config).run()
            self.assertEqual(result["stage"], "bootstrap_completed")
            self.assertIsNone(result["accepted"])
            archive = await ac.settle(entry, self.draft, self.enrollments, self.bundles, store=store, now=self.now)
            retry = await ac.settle(entry, self.draft, self.enrollments, self.bundles, store=store,
                                    now=self.now + periods.PERIOD_SECONDS * 2)
            self.assertEqual(archive, retry)
            self.assertEqual(len(store.objects), 2)
            readback = await ac.read_archive(entry, store=store)
            replay = periods.replay([readback], archive["entry_period"])
            self.assertAlmostEqual(sum(replay.values()), 0.2 * periods.PERIOD_ALPHA)
            self.assertEqual(periods.replay([readback], archive["entry_period"] + periods.REPLAY_DEPTH + 1), {})
            from reliquary.validator.weight_only import WeightOnlyValidator

            async def native_reader(_entry):
                return await ac.read_archive(_entry, store=store)
            paying_at = periods.PERIOD_EPOCH + archive["entry_period"] * periods.PERIOD_SECONDS
            weights = await WeightOnlyValidator._period_weights({entry.task_id: entry},
                         native_archive_reader=native_reader, now=paying_at)
            self.assertEqual(weights[entry.task_id], replay)
            combined = WeightOnlyValidator._replay_ema([], periods=weights, caps={entry.task_id: 0.01})
            self.assertAlmostEqual(sum(combined.values()), 0.01)
            self.assertEqual(await WeightOnlyValidator._period_weights({entry.task_id: entry},
                native_archive_reader=native_reader, now=paying_at + (periods.REPLAY_DEPTH + 1) * periods.PERIOD_SECONDS), {})
        asyncio.run(run())

    def test_precommit_deadline_and_same_native_epoch_cannot_be_reused(self):
        async def run():
            store = Store(self.manifest["deadline"] + 1)
            async def read_registry():
                return {"default": replace(self.entry, task_id="default", mechanism="rl-discovered-price", contract=None,
                    params={**self.entry.params, "cap": 0.7, "floor": 0})}, None
            with self.assertRaises(ValueError):
                await ac.declare(self.draft, self.enrollments, acknowledged=True, store=store,
                                 read_registry=read_registry, now=self.manifest["start"] + 1)
            draft = dict(self.draft, task_id="other")
            draft["base_digest"] = ac.digest(dict(task_id="other", contract=draft["contract"]))
            rows = [ac.sign_enrollment(draft, key.encode(), wallet) for key, wallet in zip(self.native_keys, self.wallets)]
            other = ac.task_entry(draft, rows)
            with self.assertRaises(ValueError):
                await store.create(ac.epoch_key(other, "commitment"), ac.commitment(other))
        asyncio.run(run())

    def test_real_r2_adapter_sends_create_only_and_conflicting_retries_refuse(self):
        from botocore.exceptions import ClientError

        clock = self.manifest["start"] + 1
        objects, puts = {}, []
        class Body:
            def __init__(self, data):
                self.data = data
            async def read(self, limit):
                return self.data[:limit]
        class Client:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def get_object(self, *, Bucket, Key):
                if Key not in objects:
                    raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
                raw = objects[Key]
                return dict(Body=Body(raw), ContentLength=len(raw), LastModified=datetime.fromtimestamp(clock, timezone.utc))
            async def put_object(self, **request):
                puts.append(request)
                if request["Key"] in objects:
                    raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
                objects[request["Key"]] = request["Body"]
        async def run():
            store = ac.NativeStore(bucket_name="fixture-native-competition")
            key, value = ac.epoch_key(self.entry, "commitment"), ac.commitment(self.entry)
            self.assertEqual(await store.create(key, value), clock)
            self.assertEqual(await store.create(key, value), clock)
            with self.assertRaises(ValueError):
                await store.create(key, dict(value, task_id="other"))
            self.assertEqual(ac.strict_json_loads(objects[key]), value)
        with patch("reliquary.infrastructure.storage.get_s3_client", return_value=Client()):
            asyncio.run(run())
        self.assertTrue(all(request["IfNoneMatch"] == "*" for request in puts))
        self.assertTrue(all(request["Bucket"] == "fixture-native-competition" for request in puts))

    def test_close_requires_settlement_and_decay_then_releases_cap(self):
        from reliquary.validator.corpus_close import TaskNotClosable, close_task

        calls = []
        async def registry():
            return {self.entry.task_id: self.entry}, None
        async def weights(_entries):
            return {}
        async def paying(_entries):
            return {self.entry.task_id: {"fixture-hotkey": 0.1}}
        async def set_task_cap(task, cap):
            calls.append(("cap", task, cap))
        async def retire(task, stamp):
            calls.append(("retire", task, stamp))
        kwargs = dict(read_registry=registry, period_weights=weights, set_cap=set_task_cap,
                      retire=retire, drand_round=lambda: 1000)
        async def missing(_entry):
            return None
        async def settled(_entry):
            return self.archive()
        with patch.object(ac, "read_archive", missing), self.assertRaises(TaskNotClosable):
            asyncio.run(close_task(self.entry.task_id, **kwargs))
        with patch.object(ac, "read_archive", settled), self.assertRaises(TaskNotClosable):
            asyncio.run(close_task(self.entry.task_id, **dict(kwargs, period_weights=paying)))
        self.assertFalse(calls)
        with patch.object(ac, "read_archive", settled):
            asyncio.run(close_task(self.entry.task_id, **kwargs))
        self.assertEqual(calls, [("cap", self.entry.task_id, 0.0), ("retire", self.entry.task_id, 1000)])

    def test_dry_settlement_is_offline_and_stdout_contains_no_private_identifiers(self):
        from reliquary.integrations.affine_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            root.chmod(0o700)
            draft_path = root / "draft.affine-private.json"
            output = root / "archive.affine-private.json"
            ac.write_private(draft_path, self.draft)
            arguments = ["competition", "settle", "--competition", str(draft_path), "--dry-run", "--out", str(output)]
            for number, document in enumerate(self.enrollments):
                path = root / f"enrollment-{number}.affine-private.json"
                ac.write_private(path, document)
                arguments += ["--enrollment", str(path)]
            for number, document in enumerate(self.bundles):
                path = root / f"evidence-{number}.affine-private.json"
                ac.write_private(path, document)
                arguments += ["--evidence", str(path)]
            stdout = io.StringIO()
            with patch("reliquary.infrastructure.task_registry_store.read_registry", side_effect=AssertionError("offline")), \
                    patch("reliquary.integrations.affine_competition_cli.time.time", return_value=self.now), redirect_stdout(stdout):
                self.assertEqual(main(arguments), 0)
            raw = stdout.getvalue()
            self.assertFalse(json.loads(raw)["archive_written"])
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertTrue(all(native not in raw for native in self.native))
            self.assertNotIn(str(root), raw)
            self.assertEqual(ac.read_private(output), self.archive())

    def test_zero_submission_finalized_epoch_burns_and_history_receipt_binding_is_required(self):
        bundle = copy.deepcopy(self.bundles[0])
        scores = dict(self.scores, receipts={}, points={}, adjusted_points={}, weights={}, total=0,
                      provisional=False, score_basis="full-audit", duplicate_coverage="complete",
                      unchecked_duplicate_claims_unresolved=False)
        bundle["envelopes"]["scores"] = self.envelope(scores)
        challenge = copy.deepcopy(bundle["envelopes"]["audit_challenge"]["payload"])
        challenge["receipts"] = {}
        bundle["envelopes"]["audit_challenge"] = self.envelope(challenge)
        bundle["envelopes"].pop("audit")
        history = copy.deepcopy(bundle["envelopes"]["history"]["payload"])
        history["epochs"][0]["frozen"] = {}
        bundle["envelopes"]["history"] = self.envelope(history)
        self.assertEqual(self.archive([bundle])["rewards_by_hotkey"], {})
        for changed in ("checkpoint", "deadline", "frozen"):
            bundles = copy.deepcopy(self.bundles)
            history = copy.deepcopy(bundles[0]["envelopes"]["history"]["payload"])
            row = history["epochs"][0]
            if changed == "checkpoint":
                row["checkpoint"]["id"] = ac.digest("other checkpoint")
            elif changed == "deadline":
                row["deadline"] += 1
            else:
                row["frozen"][self.native[0]]["sha256"] = ac.digest("other submission")
            for evidence in bundles:
                evidence["envelopes"]["history"] = self.envelope(history)
            with self.subTest(history=changed), self.assertRaises(ValueError):
                self.archive(bundles)

    def test_retired_task_cannot_start_settlement_and_archive_budget_is_checked(self):
        async def run():
            store = Store(self.manifest["start"] + 1)
            await store.create(ac.epoch_key(self.entry, "commitment"), ac.commitment(self.entry))
            retired = replace(self.entry, status="retired", retired_at=100)
            with self.assertRaises(ValueError):
                await ac.settle(retired, self.draft, self.enrollments, self.bundles, store=store, now=self.now)
            archive = await ac.settle(self.entry, self.draft, self.enrollments, self.bundles, store=store, now=self.now)
            self.assertEqual(await ac.read_archive(retired, store=store), archive)
            corrupted = copy.deepcopy(archive)
            corrupted["rewards_by_hotkey"][self.wallets[0].hotkey.ss58_address] = 1
            store.objects[ac.epoch_key(self.entry, "settlement")] = (corrupted, self.now)
            with self.assertRaises(ValueError):
                await ac.read_archive(self.entry, store=store)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
